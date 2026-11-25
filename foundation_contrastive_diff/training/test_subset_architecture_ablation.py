"""Subset-sized sampling budgets, fraction-first scheduling and exact shared resume."""

import copy
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models import build_change_head, checkpoint_feature_options
from training import test_architecture_data_ablation as fixtures
from training import train_subset_architecture_ablation as runner


class SubsetEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ArchitectureDataTests.setUp(self)
        self.args = runner.configure(self.args, self.root / "subset_epochs")
        self.context, self.basis = runner.prepare(self.args)

    def test_subset_draws_complete_accumulation_and_lr_scale_with_actual_pairs(self):
        self.args.data_fractions = [.1, .25, .5, 1.]
        self.args.epochs, self.args.warmup_epochs = 4, 2
        full = len(self.context["datasets"]["train"])
        for fraction in self.args.data_fractions:
            args, context, budget = runner.group_setup(self.args, self.context, fraction, 42)
            identity, indices = runner.shared.group_identity(args, context, fraction, 42)
            train, val, test, _ = runner.base.make_cell_loaders(args, context, indices)
            self.assertEqual(budget["updates_per_epoch"], math.ceil(len(indices) / 32))
            self.assertLess(budget["samples_per_epoch"] - len(indices), 32)
            self.assertGreaterEqual(budget["samples_per_epoch"], len(indices))
            self.assertEqual(len(train), budget["updates_per_epoch"] * args.accum)
            ids = [pair for batch in train for pair in batch["pair_id"]]
            self.assertEqual(len(ids), budget["samples_per_epoch"])
            self.assertLessEqual(set(ids), set(identity["selection"]["pair_ids"]))
            self.assertEqual(len(val.dataset), len(self.context["datasets"]["val"]))
            self.assertEqual(len(test.dataset), len(self.context["datasets"]["test"]))
            branches = runner.shared.build_branches(args, self.basis, context["updates_per_slot"])
            for branch in branches.values():
                optimizer, scheduler = branch["optimizer"], branch["scheduler"]
                initial_lr = optimizer.param_groups[0]["lr"]
                self.assertAlmostEqual(initial_lr, args.lr / budget["warmup_updates"])
                for _ in range(budget["warmup_updates"] - 1):
                    optimizer.step()
                    scheduler.step()
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], args.lr)
                for _ in range(budget["total_updates"] - scheduler.last_epoch):
                    optimizer.step()
                    scheduler.step()
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.)
            if fraction < 1:
                self.assertLess(budget["samples_per_epoch"], full)

    def test_fraction_sweep_before_repeating_seeds(self):
        self.args.data_fractions, self.args.seeds = [.1, .25, .5, 1.], [42, 43, 44]
        order = runner.group_order(self.args)
        self.assertEqual(order[:4], [(.1, 42), (.25, 42), (.5, 42), (1., 42)])
        self.assertEqual(order[4:8], [(.1, 43), (.25, 43), (.5, 43), (1., 43)])
        self.assertEqual(len(order), 12)
        with self.assertRaisesRegex(ValueError, "matrix"):
            runner.group_setup(self.args, self.context, .2, 42)
        self.assertEqual(self.context["updates_per_slot"], 2)

    def test_fresh_heads_exact_resume_and_checkpoint_metadata(self):
        args, context, budget = runner.group_setup(self.args, self.context, .5, 42)
        before = runner.shared.build_branches(args, self.basis, budget["updates_per_epoch"])
        another_args, another_context, other_budget = runner.group_setup(self.args, self.context, 1., 42)
        after = runner.shared.build_branches(another_args, self.basis, other_budget["updates_per_epoch"])
        for arm in runner.base.ARMS:
            self.assertFalse(before[arm]["optimizer"].state)
            self.assertFalse(after[arm]["optimizer"].state)
            for name, value in before[arm]["head"].state_dict().items():
                torch.testing.assert_close(value, after[arm]["head"].state_dict()[name], rtol=0, atol=0)
        runner.run_group(self.args, self.context, self.basis, .5, 42)
        folder = Path(self.args.run_root) / runner.shared.group_name(.5, 42)
        expected = torch.load(folder / "shared_last.pt", weights_only=False)
        self.assertEqual(expected["identity"]["updates_per_slot"], 1)
        resume_args = copy.copy(self.args)
        resume_args.run_root = str((self.root / "resumed").resolve())
        save = runner.base.atomic_save

        def interrupt(state, path):
            save(state, path)
            if path.name == "shared_last.pt" and state["epoch"] == 1:
                raise RuntimeError("interrupted")

        with patch.object(runner.base, "atomic_save", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                runner.run_group(resume_args, self.context, self.basis, .5, 42)
        runner.run_group(resume_args, self.context, self.basis, .5, 42, resume=True)
        resumed_folder = Path(resume_args.run_root) / folder.name
        resumed = torch.load(resumed_folder / "shared_last.pt", weights_only=False)
        for arm in runner.base.ARMS:
            self.assertEqual(resumed["branches"][arm]["scheduler"]["last_epoch"], 2)
            for name, value in expected["branches"][arm]["head"].items():
                torch.testing.assert_close(value, resumed["branches"][arm]["head"][name], rtol=0, atol=0)
            state = torch.load(resumed_folder / arm / "best.pt", weights_only=False)
            state_args = state["run_config"]
            self.assertEqual(state_args["subset_budget"]["selected_pairs"], 32)
            self.assertEqual(state_args["subset_budget"]["total_updates"], 2)
            self.assertEqual(state_args["epoch_budget"], runner.BUDGET_POLICY)
            self.assertEqual(state["optimizer_updates"], state["epoch"])
            model = build_change_head(state["head_spec"])
            model.load_state_dict(state["head"])
            self.assertEqual(checkpoint_feature_options(state["head_spec"], state_args)["selected_layer"], 9)

    def benchmark(self):
        with patch.object(runner.shared, "MIN_BENCHMARK_BATCHES", 2):
            runner.shared.benchmark(self.args, self.context, self.basis)

    def test_complete_reduced_matrix_order_completion_and_safe_report(self):
        self.args.data_fractions, self.args.seeds = [.5, 1.], [42, 43]
        self.args.epochs = 1
        self.benchmark()
        commands = []

        def child(command, **kwargs):
            fraction = float(command[command.index("--fraction") + 1])
            seed = int(command[command.index("--seed") + 1])
            commands.append((fraction, seed))
            runner.run_group(self.args, self.context, self.basis, fraction, seed, "--resume" in command)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(runner.shared, "MIN_BENCHMARK_BATCHES", 2), \
             patch.object(runner.subprocess, "run", side_effect=child):
            runner.run_sweep(self.args, self.context, self.basis)
            runner.run_sweep(self.args, self.context, self.basis, resume=True)
        self.assertEqual(commands, [(.5, 42), (1., 42), (.5, 43), (1., 43)])
        with patch.object(runner.shared, "MIN_BENCHMARK_BATCHES", 2):
            with patch.object(runner.shared.base, "make_backbone", side_effect=AssertionError("GPU constructed")):
                validated = runner.report_context(self.args)
                with patch.object(runner, "report_context", return_value=validated), \
                     patch.object(runner, "ROOT", self.root):
                    destination = runner.export_status(self.args)
        report = json.loads((destination / "summary.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_models"], 12)
        self.assertEqual(report["progress"][0]["budget"]["updates_per_epoch"], 1)
        self.assertEqual(report["progress"][3]["budget"]["updates_per_epoch"], 2)
        self.assertNotIn("volume-train-", (destination / "summary.json").read_text())
        self.assertNotIn("pair_ids", (destination / "summary.json").read_text())
        self.assertNotIn(str(self.root), (destination / "summary.json").read_text())
        self.assertEqual(report["group_order"], runner.GROUP_ORDER)
    def test_old_budget_run_cannot_be_resumed_or_used_as_benchmark(self):
        args = copy.copy(self.args)
        args.epoch_budget = "old_full_data"
        self.benchmark()
        with patch.object(runner.shared, "MIN_BENCHMARK_BATCHES", 2):
            with self.assertRaisesRegex(ValueError, "differs"):
                runner.require_benchmark(args, self.context)
        original_args = copy.copy(self.args)
        original_args.run_root = str(self.root / "old")
        original_context = dict(self.context, updates_per_slot=2)
        runner.shared.run_group(original_args, original_context, self.basis, .5, 42)
        with self.assertRaisesRegex(ValueError, "differs"):
            runner.run_group(original_args, self.context, self.basis, .5, 42, resume=True)

    def test_real_child_cli_uses_subset_budget_without_modifying_old_root(self):
        import yaml

        self.args.data_fractions, self.args.epochs = [.5, 1.], 1
        source = self.root / "locked.yaml"
        source_args = dict(vars(self.args))
        source_args["run_root"] = str(self.root / "original")
        source.write_text(yaml.safe_dump(source_args), encoding="utf-8")
        self.args.config = str(source.resolve())
        original = Path(source_args["run_root"])
        original.mkdir()
        (original / "marker.txt").write_text("keep")
        source_bytes = source.read_bytes()
        self.benchmark()
        bootstrap = (
            "import sys, torch\n"
            "from unittest.mock import patch\n"
            "from training.test_architecture_data_ablation import TinyFrozen\n"
            "from training import train_architecture_data_ablation as base\n"
            "from training import train_shared_architecture_ablation as shared\n"
            "from training import train_subset_architecture_ablation as subset\n"
            "torch.set_num_threads(1)\n"
            "with patch.multiple(base.C, BACKBONE_DIM=8, FEATURE_GRID=2, IMG_SIZE=16, "
            "EMBED_DIM=8, BACKBONE_IMG_SIZE=16, NUM_PATCH_TOKENS=4), "
            "patch.object(base, 'FrozenCXRBackbone', TinyFrozen), "
            "patch.object(shared, 'MIN_BENCHMARK_BATCHES', 2):\n"
            "    subset.main(sys.argv[1:])\n"
        )
        execute = subprocess.run

        def child(command, **kwargs):
            try:
                return execute([sys.executable, "-u", "-c", bootstrap, *command[4:]], **kwargs)
            except subprocess.CalledProcessError:
                kwargs["stdout"].flush()
                self.fail(Path(kwargs["stdout"].name).read_text(encoding="utf-8"))

        with patch.object(runner.shared, "MIN_BENCHMARK_BATCHES", 2), \
             patch.object(runner.subprocess, "run", side_effect=child):
            runner.run_sweep(self.args, self.context, self.basis)
        for fraction, updates in ((.5, 1), (1., 2)):
            state = torch.load(Path(self.args.run_root) / runner.shared.group_name(
                fraction, 42) / "shared_last.pt", weights_only=False)
            self.assertEqual(state["identity"]["updates_per_slot"], updates)
            self.assertEqual(state["branches"]["B0"]["scheduler"]["last_epoch"], updates)
        self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual((original / "marker.txt").read_text(), "keep")
        self.assertFalse(list(Path(self.args.run_root).glob(".*.lock")))


if __name__ == "__main__":
    unittest.main()
