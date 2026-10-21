"""CPU integration checks for shared-backbone early/late temporal fusion."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models import (
    DPTLayerFusionHead, FrozenCXRBackbone, build_change_head,
    checkpoint_feature_options, concatenate_layer_features,
)
from training import train_layer_ablation as trainer
from training.test_layer_ablation import miniature_data
from training.train_foundation_diff import parse_args


class FusionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.requested = []
        self.scale = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.backbone = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"model_type": "test"}))

    def _validate_layers(self, layers):
        if layers != [6, 9, 12]:
            raise ValueError("Expected unique requested depths")

    def forward_layers(self, image, layers):
        if torch.is_grad_enabled():
            raise AssertionError("Frozen extraction must run under no_grad")
        self._validate_layers(layers)
        self.calls += 1
        self.requested.append(list(layers))
        tokens = torch.nn.functional.adaptive_avg_pool2d(image, (2, 2)).flatten(2).transpose(1, 2)
        tokens = tokens.expand(-1, -1, 8) * self.scale
        return {layer: dict(patch_tokens=tokens * layer, cls_token=tokens.mean(1) * layer)
                for layer in layers}


class MiniViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=12)
        self.layernorm = torch.nn.LayerNorm(8)
        with torch.no_grad():
            self.layernorm.weight.copy_(torch.linspace(.5, 1.5, 8))
        self.calls = 0

    def forward(self, pixel_values, output_hidden_states):
        self.calls += 1
        spatial = torch.nn.functional.adaptive_avg_pool2d(pixel_values[:, :1], (2, 2))
        tokens = spatial.flatten(2).transpose(1, 2).expand(-1, -1, 8)
        tokens = torch.cat((tokens.mean(1, keepdim=True), tokens), dim=1)
        axis = torch.linspace(-1, 1, 8, device=tokens.device)
        self.states = tuple(tokens + torch.sin(axis * (layer + 1)) for layer in range(13))
        return SimpleNamespace(hidden_states=self.states, last_hidden_state=self.layernorm(self.states[-1]))


class LayerFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.constants = patch.multiple(
            trainer.C, BACKBONE_DIM=8, FEATURE_GRID=2, IMG_SIZE=4, EMBED_DIM=4,
            BACKBONE_IMG_SIZE=4, NUM_PATCH_TOKENS=4)
        self.constants.start()
        self.addCleanup(self.constants.stop)

    def args(self):
        args = parse_args(["--config", str(trainer.ROOT / "configs/layer_fusion_6_9_12_ce_30.yaml")])
        args.device = "cpu"
        args.dpt_channels = 8
        args.batch_size = 2
        return args

    def loaders(self, copies=1):
        data = miniature_data() * copies
        sampler = WeightedRandomSampler(torch.ones(len(data)), len(data), replacement=True)
        return (DataLoader(data, batch_size=2, sampler=sampler, drop_last=True),
                DataLoader(miniature_data(), batch_size=2))

    def test_architectures_ce_gradients_and_input_contract(self):
        for mode in ("early", "late"):
            with self.subTest(mode=mode):
                head = DPTLayerFusionHead(
                    backbone_dim=8, patch_dim=24, cls_dim=24, grid=2,
                    channels=8, embed_dim=4, out_size=4, num_heads=4, fusion_mode=mode)
                prior, current = [torch.randn(2, 4, 24, requires_grad=True) for _ in range(2)]
                cls_prior, cls_current = [torch.randn(2, 24, requires_grad=True) for _ in range(2)]
                logits, _ = head(prior, current, cls_prior, cls_current)
                self.assertEqual(tuple(logits.shape), (2, 3, 4, 4))
                target = torch.ones(2, 1, 4, 4) * .1
                target[1] *= -1
                args = self.args()
                loss, components = trainer.objective(logits, target, args, args.w_dice)
                expected = torch.nn.functional.cross_entropy(
                    logits, torch.cat((torch.ones(1, 4, 4), torch.full((1, 4, 4), 2))).long())
                torch.testing.assert_close(loss, expected)
                self.assertAlmostEqual(components["total"], components["ce"])
                loss.backward()
                for gradient in (prior.grad, current.grad):
                    for depth in gradient.split(8, dim=-1):
                        self.assertGreater(float(depth.abs().sum()), 0)
                self.assertGreater(float(cls_prior.grad.abs().sum()), 0)
                self.assertGreater(float(cls_current.grad.abs().sum()), 0)
                for name, parameter in head.named_parameters():
                    if name.startswith("z_proj."):
                        self.assertFalse(parameter.requires_grad)
                        self.assertIsNone(parameter.grad)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertEqual(len(head.temporal_blocks), 1 if mode == "early" else 3)
                self.assertEqual(head.projections[0].in_channels, 24 if mode == "early" else 8)
                with self.assertRaisesRegex(ValueError, "CLS"):
                    head(prior, current, cls_prior[:, :8], cls_current[:, :8])
                with self.assertRaisesRegex(ValueError, "fusion tokens"):
                    head(prior[:, :, :8], current[:, :, :8], cls_prior, cls_current)
        for invalid in (dict(patch_dim=768), dict(cls_dim=768),
                        dict(feature_layers=[6, 9]), dict(fusion_mode="mean")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DPTLayerFusionHead(**invalid)

    def test_matched_initialization_parameter_counts_and_independent_storage(self):
        args = self.args()
        trainer.validate_args(args)
        specs, branches = trainer.build_branches(args, "cpu", 2)
        self.assertEqual(set(branches), set(trainer.FUSION_ARMS))
        for left, right in (("S6", "S9"), ("S9", "S12"), ("A", "R9A"), ("B", "R9B")):
            for name, value in branches[left]["head"].state_dict().items():
                other = branches[right]["head"].state_dict()[name]
                torch.testing.assert_close(value, other, atol=0, rtol=0)
                self.assertNotEqual(value.data_ptr(), other.data_ptr())
            self.assertIsNot(branches[left]["optimizer"], branches[right]["optimizer"])
        for arm, branch in branches.items():
            for name, value in branch["head"].state_dict().items():
                reference = branches["A"]["head"].state_dict().get(name)
                if reference is not None and reference.shape == value.shape:
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
            trainable = {id(p) for p in branch["head"].parameters() if p.requires_grad}
            optimized = {id(p) for group in branch["optimizer"].param_groups for p in group["params"]}
            self.assertEqual(trainable, optimized)
            self.assertEqual(specs[arm]["kwargs"]["backbone_dim"], 8)
        for invalid in (dict(fusion_arms=["A"]), dict(fusion_arms=["A", "A", "B"]),
                        dict(w_dice=1.), dict(ablation_layers=[12, 9, 6])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                trainer.validate_args(SimpleNamespace(**(vars(args) | invalid)))

    def test_shared_extraction_augmentation_updates_and_clean_validation(self):
        args = self.args()
        train, val = self.loaders()
        _, branches = trainer.build_branches(args, "cpu", 2)
        before = {arm: trainer.cpu_weights(branch["head"]) for arm, branch in branches.items()}
        backbone = FusionBackbone()
        original_gt = [record["gt_diff"].clone() for record in train.dataset]
        with patch.object(trainer, "augment_cxr", wraps=trainer.augment_cxr) as augmentation:
            metrics, _, timing = trainer.run_epoch(
                branches, backbone, train, args, True, collect_diagnostics=False)
            self.assertEqual(augmentation.call_count, 2 * len(train))
        self.assertEqual(backbone.calls, 2 * len(train))
        self.assertEqual(timing["optimizer_steps_per_head"], 2)
        self.assertIsNone(backbone.scale.grad)
        for arm, branch in branches.items():
            self.assertTrue(any(not torch.equal(value, before[arm][name])
                                for name, value in branch["head"].state_dict().items()))
            self.assertTrue(all(parameter.grad is None for parameter in branch["head"].parameters()))
            self.assertAlmostEqual(metrics[arm]["ce"], metrics[arm]["total"])
        saved = {arm: trainer.cpu_weights(branch["head"]) for arm, branch in branches.items()}
        with patch.object(trainer, "augment_cxr") as augmentation:
            trainer.run_epoch(branches, backbone, val, args, False, collect_diagnostics=False)
            augmentation.assert_not_called()
        for arm, branch in branches.items():
            for name, value in branch["head"].state_dict().items():
                torch.testing.assert_close(value, saved[arm][name], atol=0, rtol=0)
        for record, expected in zip(train.dataset, original_gt):
            torch.testing.assert_close(record["gt_diff"], expected, atol=0, rtol=0)

    def test_resume_rng_and_reversed_branch_order_reproduce_updates(self):
        args = self.args()
        train, val = self.loaders()
        generators = trainer.configure_generators(train, val, args.seed)
        torch.manual_seed(17)
        _, branches = trainer.build_branches(args, "cpu", 2)
        trainer.run_epoch(branches, FusionBackbone(), train, args, True, collect_diagnostics=False)
        rng = trainer.capture_rng(generators)
        state = {arm: {key: copy.deepcopy(branch[key].state_dict())
                       for key in ("head", "optimizer", "scheduler")}
                 for arm, branch in branches.items()}
        trainer.run_epoch(branches, FusionBackbone(), train, args, True, collect_diagnostics=False)
        expected = {arm: trainer.cpu_weights(branch["head"]) for arm, branch in branches.items()}
        _, resumed = trainer.build_branches(args, "cpu", 2)
        for arm, branch in resumed.items():
            for key in ("head", "optimizer", "scheduler"):
                branch[key].load_state_dict(state[arm][key])
        trainer.restore_rng(rng, generators)
        trainer.run_epoch(dict(reversed(list(resumed.items()))), FusionBackbone(),
                          train, args, True, collect_diagnostics=False)
        for arm, branch in resumed.items():
            for name, value in branch["head"].state_dict().items():
                torch.testing.assert_close(value, expected[arm][name], atol=0, rtol=0)

    def test_inference_source_order_deduplication_and_normalization(self):
        for sources in ([6, 9, 12], [9, 9, 9]):
            with self.subTest(sources=sources):
                raw = MiniViT()
                with patch.object(FrozenCXRBackbone, "_build_backbone",
                                  return_value=(raw, [.5] * 3, [.5] * 3)):
                    backbone = FrozenCXRBackbone(
                        backbone_img_size=4, num_patch_tokens=4, selected_layers=sources)
                features = backbone(torch.rand(2, 1, 4, 4))
                self.assertEqual(raw.calls, 1)
                normalized = {layer: raw.layernorm(raw.states[layer]) for layer in set(sources)}
                for key, select in (("patch_tokens", slice(1, None)), ("cls_token", 0)):
                    expected = torch.cat([normalized[layer][:, select] for layer in sources], dim=-1)
                    torch.testing.assert_close(features[key], expected, atol=0, rtol=0)
                    self.assertFalse(features[key].requires_grad)
                self.assertTrue(all(not parameter.requires_grad for parameter in raw.parameters()))
        args = self.args()
        specs, branches = trainer.build_branches(args, "cpu", 2)
        for arm in branches:
            config = vars(args) | trainer.branch_metadata(arm)
            options = checkpoint_feature_options(specs[arm], config)
            if arm.startswith("S"):
                self.assertEqual(options["selected_layer"], trainer.branch_sources(arm)[0])
            else:
                self.assertEqual(options, dict(last_n_layers=1, selected_layers=trainer.branch_sources(arm)))
                config["backbone_layers"] = [12, 9, 6]
                with self.assertRaisesRegex(ValueError, "ordered sources"):
                    checkpoint_feature_options(specs[arm], config)
                config["backbone_layers"] = trainer.branch_sources(arm)
                config["fusion_mode"] = "late" if config["fusion_mode"] == "early" else "early"
                with self.assertRaisesRegex(ValueError, "fusion mode"):
                    checkpoint_feature_options(specs[arm], config)
        with self.assertRaisesRegex(ValueError, "Missing selected"):
            concatenate_layer_features({}, [9])

    def test_clinical_and_synthetic_inference_load_fusion_checkpoints(self):
        import nibabel as nib
        import numpy as np
        from evaluation import evaluate_clinical as clinical, evaluate_rq1 as synthetic

        args = self.args()
        specs, branches = trainer.build_branches(args, "cpu", 2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.json"
            annotation.write_text("[]")
            image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
            case = dict(pair_id="pair1", annotations={"Avi": str(annotation)})
            masks = (np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool))
            loaded = (image, image[::-1].copy(),
                      nib.Nifti1Image(image, np.eye(4)), {"Avi": masks})
            for arm in ("A", "B"):
                config = vars(args) | trainer.branch_metadata(arm)
                checkpoint = root / f"{arm}.pt"
                torch.save(dict(head_spec=specs[arm], head=trainer.cpu_weights(branches[arm]["head"]),
                                run_config=config, epoch=1), checkpoint)
                raw = MiniViT()
                clinical_args = SimpleNamespace(
                    num_pairs=1, clinical_root=root, preflight=False, ckpt=checkpoint,
                    out_dir=root / f"clinical_{arm}", expected_sha256=None, threshold=None, device="cpu")
                with patch.object(FrozenCXRBackbone, "_build_backbone",
                                  return_value=(raw, [.5] * 3, [.5] * 3)), \
                     patch.object(clinical, "discover_cases", return_value=[case]), \
                     patch.object(clinical, "clinical_tools"), \
                     patch.object(clinical, "load_case", return_value=loaded), \
                     patch.object(clinical, "plot_case"), patch("sys.stdout", io.StringIO()):
                    clinical.run(clinical_args)
                self.assertEqual(raw.calls, 2)
                exported = nib.load(clinical_args.out_dir / "pair1/output.nii.gz").get_fdata()
                positive = nib.load(clinical_args.out_dir / "pair1/probability_positive.nii.gz").get_fdata()
                negative = nib.load(clinical_args.out_dir / "pair1/probability_negative.nii.gz").get_fdata()
                np.testing.assert_allclose(exported, positive - negative, atol=1e-6)
                eval_args = SimpleNamespace(
                    device="cpu", review=False, ckpt=str(checkpoint), no_cache=True,
                    dataset_root=str(root), cache_dir=str(root / "absent_cache"),
                    split="val", out_dir=str(root / f"synthetic_{arm}"), threshold=None,
                    gt_threshold=None, rotation_k=None, flip_horizontal=None,
                    batch_size=2, num_workers=0)
                raw = MiniViT()
                with patch.object(FrozenCXRBackbone, "_build_backbone",
                                  return_value=(raw, [.5] * 3, [.5] * 3)), \
                     patch.object(synthetic, "parse_args", return_value=eval_args), \
                     patch.object(synthetic, "LongitudinalPairDataset", return_value=miniature_data()), \
                     patch.object(synthetic, "plot_per_type"), patch.object(synthetic, "plot_examples"), \
                     patch("sys.stdout", io.StringIO()):
                    synthetic.main()
                    self.assertEqual(raw.calls, 8)
                    eval_args.no_cache = False
                    with self.assertRaisesRegex(ValueError, "Fusion evaluation requires no_cache"):
                        synthetic.main()
                report = json.loads((Path(eval_args.out_dir) / "report_val.json").read_text())
                self.assertEqual(report["head_spec"], trainer.json_record(specs[arm]))

    def test_benchmark_training_checkpoint_reload_resume_and_scoped_publication(self):
        args = self.args()
        args.epochs, args.warmup_epochs = 1, 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest_split.jsonl").write_text("{}\n")
            (root / "results").mkdir()
            args.dataset_root = str(root)
            args.save_folder = str(root / "checkpoints")
            args.plots_folder = str(root / "plots")
            args.config = str(root / "config.yaml")
            Path(args.config).write_text("test: true\n")

            def loaders(*_):
                return FusionBackbone(), *self.loaders(copies=28)

            with patch.object(trainer, "make_loaders", side_effect=loaders), \
                 patch.object(trainer, "save_augmentation_preview"), \
                 patch.object(trainer, "CODE_FILES", ()), \
                 patch.object(trainer.subprocess, "check_output", return_value="test\n"), \
                 patch("sys.stdout", io.StringIO()):
                args.benchmark_batches = 100
                trainer.run(args)
                folder = Path(args.save_folder)
                self.assertFalse((folder / "shared_last.pt").exists())
                benchmark = json.loads((folder / "benchmark.json").read_text())
                self.assertEqual(len(benchmark["branches"]), 7)
                self.assertEqual(benchmark["train"]["measured_batches"], 100)
                self.assertEqual(benchmark["branches"]["A"]["trainable_parameters"],
                                 benchmark["branches"]["R9A"]["trainable_parameters"])
                self.assertEqual(benchmark["branches"]["B"]["trainable_parameters"],
                                 benchmark["branches"]["R9B"]["trainable_parameters"])
                args.benchmark_batches = 0
                with patch.object(trainer, "ROOT", root), \
                     patch.object(trainer, "publish") as publisher:
                    trainer.run(args, publish_results=True)
                    publisher.assert_called_once()
                    publication = publisher.call_args.args[0]
                    self.assertTrue(publication.name.startswith("layer_fusion_ablation_"))
                    self.assertFalse(list(publication.rglob("*.pt")))
                    self.assertEqual(len(list(publication.glob("arm_*/metrics.csv"))), 7)
                    self.assertEqual(len(list(publication.glob("arm_*/diagnostics/*val.json"))), 7)
                    args.resume = True
                    trainer.run(args)
                    args.w_dice = 1.
                    with self.assertRaisesRegex(ValueError, "CE-only"):
                        trainer.run(args)
                    args.w_dice = 0.
                    args.lr *= 2
                    with self.assertRaisesRegex(ValueError, "Benchmark configuration"):
                        trainer.run(args)
                saved = torch.load(folder / "shared_last.pt", weights_only=False)
                self.assertEqual(saved["epoch"], 1)
                for arm in trainer.FUSION_ARMS:
                    payload = torch.load(folder / f"arm_{arm}/best.pt", weights_only=False)
                    head = build_change_head(payload["head_spec"], "cpu").eval()
                    head.load_state_dict(payload["head"])
                    sources = trainer.branch_sources(arm)
                    backbone = FusionBackbone()
                    with torch.no_grad():
                        features = backbone.forward_layers(torch.rand(2, 1, 4, 4), [6, 9, 12])
                        packed = concatenate_layer_features(features, sources)
                        output, _ = head(packed["patch_tokens"], packed["patch_tokens"],
                                         packed["cls_token"], packed["cls_token"])
                    self.assertEqual(tuple(output.shape), (2, 3, 4, 4))
                    self.assertEqual(payload["run_config"]["feature_sources"], sources)
                    self.assertEqual(len(saved["branches"][arm]["history"]), 1)


if __name__ == "__main__":
    unittest.main()
