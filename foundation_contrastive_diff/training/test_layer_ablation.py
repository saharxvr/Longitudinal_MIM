"""CPU tests for the shared-backbone depth ablation."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.backbone import FrozenCXRBackbone
from training import train_layer_ablation as trainer
from training.train_foundation_diff import parse_args


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.backbone = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"model_type": "test"}))

    def _validate_layers(self, layers):
        if not layers:
            raise ValueError("No layers")

    @torch.no_grad()
    def forward_layers(self, image, layers):
        self.calls += 1
        tokens = image.flatten(2).transpose(1, 2)
        return {layer: dict(patch_tokens=tokens * layer, cls_token=tokens.mean(1) * layer)
                for layer in layers}


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(1, 3, 1)

    def forward(self, prior, current, cls_prior, cls_current):
        difference = (current - prior).transpose(1, 2).reshape(-1, 1, 4, 4)
        return self.projection(difference), cls_current - cls_prior


def miniature_data():
    records = []
    for number in range(8):
        records.append(dict(
            img_prior=torch.linspace(0, 1, 16).reshape(1, 4, 4),
            img_curr=torch.linspace(1, 0, 16).reshape(1, 4, 4),
            gt_diff=torch.ones(1, 4, 4) * (.1 if number % 2 else -.1),
            is_pathology=1, pair_id=f"pair{number}", change_group_id=f"group{number}",
            anomaly_type=1, direction=1))
    return records


class LayerAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self):
        args = parse_args(["--config", str(trainer.ROOT / "configs/layer_depth_6_9_12_aug_30.yaml")])
        args.device = "cpu"
        args.ablation_layers = [6, 9, 12]
        return args

    def loaders(self):
        data = miniature_data()
        sampler = WeightedRandomSampler(torch.ones(len(data)), len(data), replacement=True)
        return (DataLoader(data, batch_size=2, sampler=sampler, drop_last=True),
                DataLoader(data, batch_size=2))

    def branches(self, args, steps):
        with patch.object(trainer, "build_change_head", side_effect=lambda *a: TinyHead()):
            return trainer.build_branches(args, "cpu", steps)

    def test_config_and_selected_layer_parity_real_rad_dino(self):
        args = self.args()
        trainer.validate_args(args)
        self.assertEqual(args.epochs, 30)
        self.assertEqual((args.batch_size, args.accum), (16, 2))
        from transformers import AutoModel
        model = AutoModel.from_pretrained("microsoft/rad-dino", local_files_only=True)
        with patch.object(FrozenCXRBackbone, "_build_backbone",
                          return_value=(model, [.5] * 3, [.5] * 3)):
            backbone = FrozenCXRBackbone(backbone_img_size=28, num_patch_tokens=4)
        image = torch.linspace(0, 1, 28 * 28).reshape(1, 1, 28, 28)
        features = backbone.forward_layers(image, [6, 9, 12])
        with torch.no_grad():
            raw = model(pixel_values=backbone._prep(image), output_hidden_states=True)
        for layer in (6, 9, 12):
            expected = model.layernorm(raw.hidden_states[layer])
            torch.testing.assert_close(features[layer]["patch_tokens"], expected[:, -4:])
            torch.testing.assert_close(features[layer]["cls_token"], expected[:, 0])
            self.assertFalse(features[layer]["patch_tokens"].requires_grad)
        torch.testing.assert_close(features[12]["patch_tokens"], backbone(image)["patch_tokens"])
        backbone.selected_layer = 6
        torch.testing.assert_close(features[6]["cls_token"], backbone(image)["cls_token"])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        with self.assertRaises(ValueError):
            backbone.forward_layers(image, [0, 12])
        with self.assertRaises(ValueError):
            backbone.forward_layers(image, [6, 6])

    def test_shared_forward_updates_independent_heads_and_partial_accumulation(self):
        args = self.args()
        args.accum = 3
        train, val = self.loaders()
        trainer.configure_generators(train, val, args.seed)
        _, branches = self.branches(args, 1)
        first = trainer.cpu_weights(branches[6]["head"])
        for branch in branches.values():
            for name, value in branch["head"].state_dict().items():
                torch.testing.assert_close(value, first[name])
        backbone = TinyBackbone()
        before = trainer.cpu_weights(branches[9]["head"])
        loss, _ = trainer.objective(branches[6]["head"](
            torch.zeros(1, 16, 1), torch.ones(1, 16, 1),
            torch.zeros(1, 1), torch.ones(1, 1))[0], torch.ones(1, 1, 4, 4), args, 1.)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in branches[9]["head"].parameters()))
        for name, value in branches[9]["head"].state_dict().items():
            torch.testing.assert_close(value, before[name])
        _, _, timing = trainer.run_epoch(branches, backbone, train, args, True, collect_diagnostics=False)
        self.assertEqual(backbone.calls, 2 * len(train))
        self.assertEqual(timing["optimizer_steps_per_head"], 1)
        self.assertEqual(timing["discarded_accumulation_batches"], 1)
        for branch in branches.values():
            self.assertTrue(all(parameter.grad is None for parameter in branch["head"].parameters()))
        self.assertNotEqual(branches[6]["head"].projection.weight.data_ptr(),
                            branches[9]["head"].projection.weight.data_ptr())
        self.assertIsNot(branches[6]["optimizer"].state, branches[9]["optimizer"].state)
        weights = [trainer.cpu_weights(branch["head"]) for branch in branches.values()]
        trainer.run_epoch(branches, backbone, val, args, False, collect_diagnostics=False)
        for branch, saved in zip(branches.values(), weights):
            for name, value in branch["head"].state_dict().items():
                torch.testing.assert_close(value, saved[name], atol=0, rtol=0)

    def test_resume_and_branch_order_preserve_input_rng_and_updates(self):
        args = self.args()
        train, val = self.loaders()
        generators = trainer.configure_generators(train, val, args.seed)
        torch.manual_seed(21)
        _, branches = self.branches(args, 2)
        trainer.run_epoch(branches, TinyBackbone(), train, args, True, collect_diagnostics=False)
        saved_rng = trainer.capture_rng(generators)
        state = {layer: {key: copy.deepcopy(branch[key].state_dict())
                         for key in ("head", "optimizer", "scheduler")}
                 for layer, branch in branches.items()}
        trainer.run_epoch(branches, TinyBackbone(), train, args, True, collect_diagnostics=False)
        expected = {layer: trainer.cpu_weights(branch["head"]) for layer, branch in branches.items()}
        for branch in branches.values():
            self.assertEqual(branch["scheduler"].last_epoch, 4)
        _, resumed = self.branches(args, 2)
        for layer, branch in resumed.items():
            for key in ("head", "optimizer", "scheduler"):
                branch[key].load_state_dict(state[layer][key])
        trainer.restore_rng(saved_rng, generators)
        resumed = dict(reversed(list(resumed.items())))
        trainer.run_epoch(resumed, TinyBackbone(), train, args, True, collect_diagnostics=False)
        for layer, branch in resumed.items():
            for name, value in branch["head"].state_dict().items():
                torch.testing.assert_close(value, expected[layer][name], atol=0, rtol=0)

    def test_atomic_snapshot_export_and_clinical_layer_metadata(self):
        args = self.args()
        with tempfile.TemporaryDirectory() as temporary:
            args.save_folder = temporary
            spec, branches = self.branches(args, 2)
            for branch in branches.values():
                branch.update(best_head=trainer.cpu_weights(branch["head"]), best_epoch=1,
                              best_score=.5, best_val=dict(target_signed_dice=.5),
                              history=[dict(epoch=1, val_target_signed_dice=.5)])
            trainer.export_branches(branches, spec, args)
            for layer in branches:
                payload = torch.load(Path(temporary) / f"layer_{layer:02d}/best.pt", weights_only=False)
                self.assertEqual(payload["run_config"]["backbone_layer"], layer)
                self.assertEqual(payload["run_config"]["feature_normalization"], "final_layernorm")
            path = Path(temporary) / "shared_last.pt"
            trainer.atomic_save(dict(epoch=1), path)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)
            with patch.object(trainer.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    trainer.atomic_save(dict(epoch=2), path)
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_nonfinite_loss_fails_explicitly(self):
        args = self.args()
        train, _ = self.loaders()
        _, branches = self.branches(args, 2)
        with torch.no_grad():
            branches[6]["head"].projection.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            trainer.run_epoch(branches, TinyBackbone(), train, args, True, collect_diagnostics=False)

    def test_complete_benchmark_train_resume_and_scoped_publication(self):
        args = self.args()
        args.epochs, args.warmup_epochs = 2, 0
        args.batch_size = 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest_split.jsonl").write_text("{}\n")
            (root / "results").mkdir()
            args.dataset_root = str(root)
            args.save_folder = str(root / "checkpoints")
            args.plots_folder = str(root / "plots")
            args.config = str(root / "config.yaml")
            Path(args.config).write_text("test: true\n")
            data = miniature_data() * 28

            def loaders(*_):
                sampler = WeightedRandomSampler(torch.ones(len(data)), len(data), replacement=True)
                return (TinyBackbone(), DataLoader(data, batch_size=2, sampler=sampler, drop_last=True),
                        DataLoader(miniature_data(), batch_size=2))

            with patch.object(trainer, "make_loaders", side_effect=loaders), \
                 patch.object(trainer, "build_change_head", side_effect=lambda *a: TinyHead()), \
                 patch.object(trainer, "save_augmentation_preview"), \
                 patch.object(trainer, "CODE_FILES", ()), \
                 patch.object(trainer.subprocess, "check_output", return_value="test\n"):
                args.benchmark_batches = 100
                trainer.run(args)
                self.assertFalse((root / "checkpoints/shared_last.pt").exists())
                benchmark = json.loads((root / "checkpoints/benchmark.json").read_text())
                self.assertEqual(benchmark["train"]["measured_batches"], 100)
                args.benchmark_batches = 0
                with patch.object(trainer, "ROOT", root), patch.object(trainer, "publish") as publisher:
                    trainer.run(args, publish_results=True)
                    publisher.assert_called_once()
                    publication = publisher.call_args.args[0]
                    self.assertFalse(list(publication.rglob("*.pt")))
                    self.assertEqual(len(list(publication.glob("layer_*/metrics.csv"))), 3)
                    self.assertEqual(len(list(publication.glob("layer_*/diagnostics/*val.json"))), 6)
                saved = torch.load(root / "checkpoints/shared_last.pt", weights_only=False)
                self.assertEqual(saved["epoch"], 2)
                for layer in (6, 9, 12):
                    self.assertEqual(len(saved["branches"][layer]["history"]), 2)
                args.resume = True
                trainer.run(args)
                args.lr *= 2
                with self.assertRaisesRegex(ValueError, "Benchmark configuration"):
                    trainer.run(args)

    def test_clinical_inference_selects_checkpoint_layer(self):
        import nibabel as nib
        import numpy as np
        from evaluation import evaluate_clinical as clinical

        class ClinicalBackbone(TinyBackbone):
            def forward(self, image):
                return self.forward_layers(image, [6])[6]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotation = root / "annotation.json"
            annotation.write_text("[]")
            case = dict(pair_id="pair1", annotations={"Avi": str(annotation)})
            config = dict(loss_mode="ce_dice", backbone_layer=6, feature_normalization="final_layernorm",
                          rotation_k=3, flip_horizontal=True, eval_threshold=.5)
            checkpoint = root / "best.pt"
            state = dict(head_spec=dict(type="test", kwargs=dict(
                output_mode="three_class", backbone_dim=1, patch_dim=1, out_size=4)),
                head=TinyHead().state_dict(), run_config=config, epoch=1)
            torch.save(state, checkpoint)
            args = SimpleNamespace(num_pairs=1, clinical_root=root, preflight=False, ckpt=checkpoint,
                                   out_dir=root / "output", expected_sha256=None, threshold=None, device="cpu")
            masks = (np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool))
            loaded = (np.zeros((4, 4)), np.ones((4, 4)),
                      nib.Nifti1Image(np.zeros((4, 4)), np.eye(4)), {"Avi": masks})
            with patch.object(clinical, "discover_cases", return_value=[case]), \
                 patch.object(clinical, "clinical_tools"), \
                 patch.object(clinical, "load_case", return_value=loaded), \
                 patch.object(clinical, "plot_case"), \
                 patch("models.build_change_head", return_value=TinyHead()), \
                 patch("models.FrozenCXRBackbone", return_value=ClinicalBackbone()) as constructor:
                clinical.run(args)
                self.assertEqual(constructor.call_args.kwargs["selected_layer"], 6)
                state["run_config"]["feature_normalization"] = "invalid"
                torch.save(state, checkpoint)
                args.out_dir = root / "invalid"
                with self.assertRaisesRegex(ValueError, "Selected-layer checkpoint"):
                    clinical.run(args)

    def test_real_dpt_heads_share_features_and_backpropagate(self):
        args = self.args()
        spec = dict(type="dpt_cross_attention", kwargs=dict(
            backbone_dim=8, patch_dim=8, cls_dim=8, grid=2, channels=8, embed_dim=4,
            out_size=4, num_layers=1, num_heads=2, attention_depth=1, output_mode="three_class"))
        with patch.object(trainer, "head_spec", return_value=spec):
            _, branches = trainer.build_branches(args, "cpu", 2)

        class FeatureBackbone(TinyBackbone):
            def forward_layers(self, image, layers):
                self.calls += 1
                tokens = image.flatten(2)[:, :, :4].transpose(1, 2).expand(-1, -1, 8).contiguous()
                return {layer: dict(patch_tokens=tokens * layer, cls_token=tokens.mean(1) * layer)
                        for layer in layers}

        train, _ = self.loaders()
        before = trainer.cpu_weights(branches[6]["head"])
        trainer.run_epoch(branches, FeatureBackbone(), train, args, True)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in branches[6]["head"].state_dict().items()))


if __name__ == "__main__":
    unittest.main()
