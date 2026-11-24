"""Focused CPU tests for patient-size/global-attention controls and resumable cells."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from datasets.pair_dataset import audit_patient_splits, patient_subset_indices, record_patient
from models import build_change_head, checkpoint_feature_options
from models.dpt_head import JointSelfAttention
from training import train_architecture_data_ablation as trainer
from training.positional_debias import diagnose_features
from training.train_foundation_diff import parse_args


class TinyFrozen(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone_img_size = 16
        self.register_buffer("scales", torch.arange(1, 9).float())
        self.backbone = SimpleNamespace(config=SimpleNamespace(
            _commit_hash="tiny", to_dict=lambda: {"model_type": "tiny"}))

    @torch.no_grad()
    def forward(self, image):
        patches = F.adaptive_avg_pool2d(image.mean(1, keepdim=True), (2, 2))
        tokens = patches.flatten(2).transpose(1, 2) * self.scales
        return dict(patch_tokens=tokens, cls_token=tokens.mean(1))


def make_corpus(root):
    import nibabel as nib

    rows = []
    for split, patients in (("train", 16), ("val", 2), ("test", 2)):
        for patient in range(patients):
            case = f"volume-{split}-{patient}"
            for variant in range(4 if split == "train" else 2):
                pair_id = f"{case}/pair{variant}"
                directory = root / case / f"pair{variant}"
                directory.mkdir(parents=True)
                image = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
                gt = np.zeros((16, 16), np.float32)
                if variant % 3:
                    gt[2:8, 3:9] = .1 if variant % 2 else -.1
                for name, array in (("prior", image), ("current", image[::-1].copy()),
                                    ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(array, np.eye(4)), directory / f"{name}.nii.gz")
                rows.append(dict(pair_id=pair_id, case=case, split=split,
                                 change_group_id=f"{case}-change{variant // 2}",
                                 effective_anomaly_type="consolidation" if variant % 3 else "none",
                                 realized_change=bool(variant % 3), direction="positive"))
    (root / "manifest_split.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return rows


class GlobalAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def spec(self, rank=0):
        return dict(type="dpt_global_attention", kwargs=dict(
            backbone_dim=8, patch_dim=8, cls_dim=8, num_layers=1, grid=3,
            channels=8, num_heads=2, attention_depth=1, out_size=16,
            embed_dim=8, output_mode="three_class", debias_rank=rank))

    def test_joint_attention_uses_within_and_cross_image_tokens(self):
        block = JointSelfAttention(8, 2)
        prior = torch.randn(2, 9, 8, requires_grad=True)
        current = torch.randn_like(prior, requires_grad=True)
        output = block(prior, current)[0]
        output[:, 0].square().sum().backward()
        self.assertGreater(float(prior.grad[:, 1:].abs().sum()), 0)
        self.assertGreater(float(current.grad.abs().sum()), 0)

    def test_change_only_checkpoint_metric_excludes_empty_pathology_pairs(self):
        prediction = torch.ones(3, 1, 2, 2)
        ground_truth = torch.zeros_like(prediction)
        ground_truth[0] = .1
        counts = trainer.active_overlap_counts(
            prediction, ground_truth, torch.tensor([1, 1, 0]), .5, .001)
        self.assertEqual(counts["signed"], [4, 4, 4])
        self.assertEqual(counts["negative"], [0, 0, 0])

    def test_exclusive_run_lock_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.lock"
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with trainer.run_lock(path):
                    with self.assertRaises(FileExistsError):
                        with trainer.run_lock(path):
                            self.fail("Concurrent run acquired the same lock")
                    raise RuntimeError("failure")
            self.assertFalse(path.exists())
            with trainer.run_lock(path):
                self.assertTrue(path.is_file())

    def test_role_gradients_fixed_projection_and_checkpoint_roundtrip(self):
        spec = self.spec(2)
        head = build_change_head(spec)
        inputs = (torch.randn(2, 9, 8), torch.randn(2, 9, 8),
                  torch.randn(2, 8), torch.randn(2, 8))
        with self.assertRaisesRegex(ValueError, "basis"):
            head(*inputs)
        basis = torch.eye(8)[:, :2]
        head.set_debias_basis(basis)
        projected = head.clean_patch_tokens(inputs[0])
        torch.testing.assert_close(projected @ basis, torch.zeros(2, 9, 2))
        output, embedding = head(*inputs)
        self.assertEqual(output.shape, (2, 3, 16, 16))
        self.assertEqual(embedding.shape, (2, 8))
        output.square().mean().backward()
        for role in head.image_roles.grad:
            self.assertTrue(torch.isfinite(role).all())
            self.assertGreater(float(role.abs().sum()), 0)
        self.assertFalse(head.debias_basis.requires_grad)
        self.assertIsNone(head.debias_basis.grad)
        with torch.no_grad():
            self.assertFalse(torch.allclose(output, head(inputs[0] + 1, *inputs[1:])[0]))
            self.assertFalse(torch.allclose(output, head(inputs[0], inputs[1] + 1, *inputs[2:])[0]))
        restored = build_change_head(spec)
        restored.load_state_dict(head.state_dict())
        torch.testing.assert_close(restored(*inputs)[0], output, rtol=0, atol=0)
        options = checkpoint_feature_options(spec, dict(
            backbone_layer=9, feature_normalization="final_layernorm"))
        self.assertEqual(options, dict(last_n_layers=1, selected_layer=9))
        identity = build_change_head(self.spec())
        self.assertIs(identity.clean_patch_tokens(inputs[0]), inputs[0])
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            head.set_debias_basis(torch.ones(8, 2))


class ArchitectureDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = make_corpus(self.root)
        self.args = parse_args(["--config", str(trainer.ROOT / "configs/architecture_data_block9_aug.yaml")])
        self.args.device = "cpu"
        self.args.epochs, self.args.warmup_epochs = 2, 0
        self.args.dataset_root = str(self.root)
        self.args.run_root = str(self.root / "run")
        self.args.dpt_channels = 8
        self.args.data_fractions, self.args.seeds = [.5, 1.], [42]
        self.args.diagnostic_file = str(self.root / "diagnostic.pt")
        self.args.diagnostic_ranks = [0, 1, 2]
        self.args.diagnostic_train_pairs = 2
        self.constants = patch.multiple(
            trainer.C, BACKBONE_DIM=8, FEATURE_GRID=2, IMG_SIZE=16,
            EMBED_DIM=8, BACKBONE_IMG_SIZE=16, NUM_PATCH_TOKENS=4)
        self.constants.start()
        self.addCleanup(self.constants.stop)
        self.backbone_patch = patch.object(trainer, "FrozenCXRBackbone", TinyFrozen)
        self.backbone_patch.start()
        self.addCleanup(self.backbone_patch.stop)
        self.context = trainer.prepare(self.args)
        result = diagnose_features(self.context["backbone"], self.context["datasets"]["train"],
                                   [0, 1, 2], 2026, "cpu", 2)
        result["provenance"] = self.context["provenance"]
        trainer.atomic_save(result, Path(self.args.diagnostic_file))
        self.args.debias_rank = 1
        self.args.diagnostic_sha256 = trainer.sha256(self.args.diagnostic_file)

    def test_patient_nesting_complete_variants_and_leakage_rejection(self):
        audit = audit_patient_splits(self.rows)
        self.assertEqual(audit["train"]["pairs"], 64)
        train = [row for row in self.rows if row["split"] == "train"]
        previous = set()
        for fraction in (.1, .25, .5, 1.):
            indices = patient_subset_indices(train, fraction, 42)
            selected = {train[i]["pair_id"] for i in indices}
            patients = {record_patient(train[i]) for i in indices}
            self.assertLessEqual(previous, selected)
            self.assertEqual(selected, {r["pair_id"] for r in train if record_patient(r) in patients})
            self.assertEqual(indices, patient_subset_indices(train, fraction, 42))
            previous = selected
        for mutation in ("duplicate", "patient", "group", "missing"):
            rows = copy.deepcopy(self.rows)
            if mutation == "duplicate":
                rows.append(rows[0])
            elif mutation == "patient":
                rows[-1]["case"] = rows[0]["case"]
            elif mutation == "group":
                rows[-1]["change_group_id"] = rows[0]["change_group_id"]
            else:
                rows[0].pop("change_group_id")
            with self.assertRaises(ValueError):
                audit_patient_splits(rows)
        with self.assertRaisesRegex(ValueError, "training"):
            patient_subset_indices(self.rows, .5, 42)

    def test_sampler_membership_equal_draws_and_common_initialization(self):
        basis = trainer.locked_basis(self.args, self.context)
        heads, seen = {}, {}
        for arm in trainer.ARMS:
            identity, indices = trainer.cell_identity(self.args, self.context, arm, .5, 42)
            train, val, test, _ = trainer.make_cell_loaders(self.args, self.context, indices)
            self.assertEqual(len(train), self.context["updates_per_slot"] * self.args.accum)
            ids = [pair for batch in train for pair in batch["pair_id"]]
            self.assertLessEqual(set(ids), set(identity["selection"]["pair_ids"]))
            seen[arm] = ids
            self.assertEqual(val.dataset.records, self.context["datasets"]["val"].records)
            self.assertEqual(test.dataset.records, self.context["datasets"]["test"].records)
            heads[arm] = trainer.make_cell_head(self.args, arm, basis)[1]
        self.assertEqual(seen["B0"], seen["B1"])
        self.assertEqual(seen["B1"], seen["B2"])
        for name, value in heads["B0"].state_dict().items():
            torch.testing.assert_close(value, heads["B1"].state_dict()[name], rtol=0, atol=0)
        for (name, value), (_, other) in zip(heads["B1"].named_parameters(), heads["B2"].named_parameters()):
            torch.testing.assert_close(value, other, rtol=0, atol=0)
            self.assertNotEqual(value.data_ptr(), other.data_ptr(), name)
        self.assertTrue(all(not p.requires_grad for p in self.context["backbone"].parameters()))

    def test_diagnostic_lock_and_provenance_are_required(self):
        args = copy.copy(self.args)
        args.debias_rank = None
        with self.assertRaisesRegex(ValueError, "diagnose"):
            trainer.locked_basis(args, self.context)
        args.debias_rank = 3
        with self.assertRaisesRegex(ValueError, "evaluated"):
            trainer.locked_basis(args, self.context)
        changed = copy.deepcopy(self.context["provenance"])
        changed["features"]["backbone_layer"] = 12
        context = dict(self.context, provenance=changed)
        with self.assertRaisesRegex(ValueError, "provenance"):
            trainer.locked_basis(self.args, context)
        with open(self.args.diagnostic_file, "ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            trainer.locked_basis(self.args, self.context)

    def test_diagnostic_output_persists_and_cannot_be_overwritten(self):
        args = copy.copy(self.args)
        args.diagnostic_file = str(self.root / "new_diagnostic.pt")
        trainer.run_diagnostics(args, self.context)
        report = json.loads(Path(args.diagnostic_file).with_suffix(".json").read_text())
        self.assertEqual([row["rank"] for row in report["rows"]], [0, 1, 2])
        self.assertEqual(report["diagnostic_sha256"], trainer.sha256(args.diagnostic_file))
        self.assertTrue(Path(args.diagnostic_file).with_suffix(".png").is_file())
        self.assertLessEqual(set(report["training_pair_ids"]),
                             {record["pair_id"] for record in self.context["datasets"]["train"].records})
        with self.assertRaisesRegex(ValueError, "already exist"):
            trainer.run_diagnostics(args, self.context)

    def test_cell_exact_budget_and_interrupted_resume_equivalence(self):
        result = trainer.run_cell(self.args, self.context, "B1", .5, 42)
        folder = Path(self.args.run_root) / trainer.cell_name("B1", .5, 42)
        expected = torch.load(folder / "last.pt", weights_only=False)
        self.assertEqual(expected["optimizer_updates"], 4)
        self.assertEqual(expected["scheduler"]["last_epoch"], 4)
        self.assertEqual([row["train"]["sample_draws"] for row in expected["history"]], [64, 64])
        self.assertEqual(result["inventory"]["patients"], 8)
        resume_args = copy.copy(self.args)
        resume_args.run_root = str(self.root / "interrupted")
        real_save = trainer.atomic_save

        def interrupt(state, path):
            real_save(state, path)
            if path.name == "last.pt" and state["epoch"] == 1:
                raise RuntimeError("simulated interruption")

        with patch.object(trainer, "atomic_save", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                trainer.run_cell(resume_args, self.context, "B1", .5, 42)
        trainer.run_cell(resume_args, self.context, "B1", .5, 42, resume=True)
        resumed = torch.load(Path(resume_args.run_root) / folder.name / "last.pt", weights_only=False)
        for name, value in expected["head"].items():
            torch.testing.assert_close(value, resumed["head"][name], atol=0, rtol=0)
        for left, right in zip(expected["history"], resumed["history"]):
            self.assertEqual(left["lr"], right["lr"])
            self.assertEqual(left["val"]["target_signed_dice"], right["val"]["target_signed_dice"])
        changed = copy.copy(resume_args)
        changed.lr *= 2
        with self.assertRaisesRegex(ValueError, "identity"):
            trainer.run_cell(changed, self.context, "B1", .5, 42, resume=True)
        with open(folder / "best.pt", "ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            trainer.run_cell(self.args, self.context, "B1", .5, 42, resume=True)

    def test_nonfinite_batches_and_partial_accumulation_fail(self):
        basis = trainer.locked_basis(self.args, self.context)
        _, indices = trainer.cell_identity(self.args, self.context, "B0", .5, 42)
        train, _, _, _ = trainer.make_cell_loaders(self.args, self.context, indices)
        _, head = trainer.make_cell_head(self.args, "B0", basis)
        optimizer, scheduler = trainer.optimizer_state(self.args, head, 2)
        with self.assertRaisesRegex(ValueError, "accumulation"):
            trainer.run_pass(head, self.context["backbone"], [next(iter(train))],
                             self.args, optimizer, scheduler)
        with torch.no_grad():
            head.output[-1].weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            trainer.run_pass(head, self.context["backbone"], train, self.args, optimizer, scheduler)

    def test_benchmark_sweep_smoke_and_verified_completion_skips(self):
        with patch.object(trainer, "MIN_BENCHMARK_BATCHES", 2):
            trainer.run_benchmark(self.args, self.context, trainer.locked_basis(self.args, self.context))
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                self.assertEqual(kwargs["cwd"], trainer.ROOT)
                self.assertTrue(kwargs["check"])
                arm = command[command.index("--arm") + 1]
                fraction = float(command[command.index("--fraction") + 1])
                seed = int(command[command.index("--seed") + 1])
                trainer.run_cell(self.args, self.context, arm, fraction, seed, "--resume" in command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(trainer.subprocess, "run", side_effect=child):
                trainer.run_sweep(self.args, self.context)
                self.assertEqual(len(commands), 6)
                self.assertTrue(all("-u" in command and "--mode" in command for command in commands))
                trainer.run_sweep(self.args, self.context, resume=True)
                self.assertEqual(len(commands), 6)
            root = Path(self.args.run_root)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["cells"], 6)
            self.assertEqual(len(summary["groups"]), 6)
            self.assertTrue((root / "learning_curves.png").is_file())
            self.assertEqual(len((root / "paired_differences.csv").read_text().splitlines()), 5)
            for fraction in self.args.data_fractions:
                for arm in trainer.ARMS:
                    folder = root / trainer.cell_name(arm, fraction, 42)
                    state = torch.load(folder / "best.pt", weights_only=False)
                    head = build_change_head(state["head_spec"])
                    head.load_state_dict(state["head"])
                    self.assertEqual(checkpoint_feature_options(
                        state["head_spec"], state["run_config"])["selected_layer"], 9)
            with self.assertRaisesRegex(ValueError, "Existing sweep"):
                trainer.run_sweep(self.args, self.context)

    def test_sweep_executes_real_independent_child_processes(self):
        import yaml

        self.args.epochs = 1
        self.args.data_fractions = [1.]
        self.args.config = str(self.root / "smoke.yaml")
        Path(self.args.config).write_text(yaml.safe_dump(vars(self.args)), encoding="utf-8")
        bootstrap = (
            "import sys, torch\n"
            "from unittest.mock import patch\n"
            "from training.test_architecture_data_ablation import TinyFrozen\n"
            "from training import train_architecture_data_ablation as runner\n"
            "torch.set_num_threads(1)\n"
            "with patch.multiple(runner.C, BACKBONE_DIM=8, FEATURE_GRID=2, IMG_SIZE=16, "
            "EMBED_DIM=8, BACKBONE_IMG_SIZE=16, NUM_PATCH_TOKENS=4), "
            "patch.object(runner, 'FrozenCXRBackbone', TinyFrozen), "
            "patch.object(runner, 'MIN_BENCHMARK_BATCHES', 2):\n"
            "    runner.main(sys.argv[1:])\n"
        )
        execute = subprocess.run

        def child(command, **kwargs):
            return execute([sys.executable, "-u", "-c", bootstrap, *command[4:]], **kwargs)

        with patch.object(trainer, "MIN_BENCHMARK_BATCHES", 2):
            trainer.run_benchmark(self.args, self.context, trainer.locked_basis(self.args, self.context))
            with patch.object(trainer.subprocess, "run", side_effect=child):
                trainer.run_sweep(self.args, self.context)
        root = Path(self.args.run_root)
        self.assertEqual(json.loads((root / "summary.json").read_text())["cells"], 3)
        self.assertFalse(list(root.glob(".*.lock")))
        for arm in trainer.ARMS:
            checkpoint = torch.load(root / trainer.cell_name(arm, 1., 42) / "last.pt",
                                    weights_only=False)
            self.assertEqual(checkpoint["optimizer_updates"], 2)
            self.assertEqual(checkpoint["scheduler"]["last_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
