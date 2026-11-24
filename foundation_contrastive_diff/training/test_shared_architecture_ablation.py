"""Shared extraction, independent optimization, resume and compact-export checks."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models import build_change_head, checkpoint_feature_options
from training import train_architecture_data_ablation as base
from training import train_shared_architecture_ablation as shared
from training import test_architecture_data_ablation as fixtures


class SharedArchitectureTests(unittest.TestCase):
    setUp = fixtures.ArchitectureDataTests.setUp

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def shared_context(self):
        basis = base.locked_basis(self.args, self.context)
        context = dict(self.context, provenance=dict(
            self.context["provenance"],
            shared_code_sha256={shared.SOURCE: base.sha256(shared.ROOT / shared.SOURCE)}))
        return context, basis

    def test_one_extraction_per_image_and_independent_heads(self):
        context, basis = self.shared_context()
        _, indices = shared.group_identity(self.args, context, .5, 42)
        train, _, _, _ = base.make_cell_loaders(self.args, context, indices)
        shared.seed_training(42, "cpu")
        branches = shared.build_branches(self.args, basis, 2)
        first = base.cpu_weights(branches["B0"]["head"])
        with patch.object(context["backbone"], "forward", wraps=context["backbone"].forward) as extract:
            metrics, _, timing = shared.run_pass(
                branches, context["backbone"], train, self.args, training=True)
        self.assertEqual(extract.call_count, 2 * len(train))
        self.assertEqual(timing["optimizer_updates_per_head"], dict.fromkeys(base.ARMS, 2))
        for arm, branch in branches.items():
            self.assertEqual(metrics[arm]["sample_draws"], 64)
            self.assertTrue(any(not torch.equal(value, branch["head"].state_dict()[name])
                                for name, value in first.items()))
            self.assertEqual(branch["scheduler"].last_epoch, 2)
        self.assertIsNot(branches["B0"]["optimizer"].state, branches["B1"]["optimizer"].state)
        captured = []
        handles = [branch["head"].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
            for branch in branches.values()]
        try:
            shared.run_pass(branches, context["backbone"], [next(iter(train))], self.args)
        finally:
            for handle in handles:
                handle.remove()
        for value in captured[1:]:
            torch.testing.assert_close(value, captured[0], rtol=0, atol=0)

    def test_atomic_group_resume_matches_all_three_heads(self):
        context, basis = self.shared_context()
        shared.run_group(self.args, context, basis, .5, 42)
        folder = Path(self.args.run_root) / shared.group_name(.5, 42)
        expected = torch.load(folder / "shared_last.pt", weights_only=False)
        resume_args = copy.copy(self.args)
        resume_args.run_root = str(self.root / "resumed")
        save = base.atomic_save

        def interrupt(state, path):
            save(state, path)
            if path.name == "shared_last.pt" and state["epoch"] == 1:
                raise RuntimeError("interrupted")

        with patch.object(base, "atomic_save", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                shared.run_group(resume_args, context, basis, .5, 42)
        shared.run_group(resume_args, context, basis, .5, 42, resume=True)
        resumed_folder = Path(resume_args.run_root) / folder.name
        resumed = torch.load(resumed_folder / "shared_last.pt", weights_only=False)
        for arm in base.ARMS:
            for name, value in expected["branches"][arm]["head"].items():
                torch.testing.assert_close(value, resumed["branches"][arm]["head"][name], rtol=0, atol=0)
            state = torch.load(resumed_folder / arm / "best.pt", weights_only=False)
            head = build_change_head(state["head_spec"])
            head.load_state_dict(state["head"])
            self.assertEqual(checkpoint_feature_options(
                state["head_spec"], state["run_config"])["selected_layer"], 9)
        identity, _ = shared.group_identity(resume_args, context, .5, 42)
        self.assertEqual(len(shared.verify_group(resumed_folder, identity)), 3)
        with (resumed_folder / "B1" / "best.pt").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            shared.verify_group(resumed_folder, identity)

    def test_all_head_benchmark_sweep_and_safe_progress_export(self):
        context, basis = self.shared_context()
        self.args.data_fractions = [1.]
        self.args.epochs = 1
        with patch.object(shared, "MIN_BENCHMARK_BATCHES", 2):
            with self.assertRaisesRegex(ValueError, "all-head"):
                shared.require_benchmark(self.args, context)
            shared.benchmark(self.args, context, basis)
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                self.assertEqual(command[-5:-3], ["group", "--fraction"])
                shared.run_group(self.args, context, basis, 1., 42, "--resume" in command)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(shared.subprocess, "run", side_effect=child):
                shared.run_sweep(self.args, context, basis)
                shared.run_sweep(self.args, context, basis, resume=True)
            self.assertEqual(len(commands), 1)
            with patch.object(shared, "ROOT", self.root):
                source = self.root / shared.SOURCE
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes((base.ROOT / shared.SOURCE).read_bytes())
                destination = shared.export_status(self.args)
        text = (destination / "summary.json").read_text()
        report = json.loads(text)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_models"], 3)
        self.assertNotIn("volume-train-", text)
        self.assertNotIn("pair_ids", text)
        self.assertNotIn(str(self.root), text)

    def test_nonfinite_branch_stops_group_and_publication_is_scoped(self):
        context, basis = self.shared_context()
        _, indices = shared.group_identity(self.args, context, .5, 42)
        train, _, _, _ = base.make_cell_loaders(self.args, context, indices)
        branches = shared.build_branches(self.args, basis, 2)
        with torch.no_grad():
            branches["B1"]["head"].output[-1].weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "Nonfinite B1"):
            shared.run_pass(branches, context["backbone"], train, self.args, training=True)
        root = self.root / "study"
        destination = root / "results" / "test_shared_export"
        destination.mkdir(parents=True)
        (destination / "summary.json").write_text("{}")
        with patch.object(shared, "ROOT", root), \
             patch.object(shared.subprocess, "check_output", return_value="main\n"), \
             patch.object(shared.subprocess, "run") as git:
            shared.publish_report(destination)
        commands = [call.args[0] for call in git.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("pull" not in command for command in commands))
        self.assertEqual(commands[0][-1], destination.joinpath("summary.json").relative_to(
            root.parent).as_posix())
        self.assertTrue(all(call.kwargs["check"] for call in git.call_args_list))

    def test_real_child_cli_trains_three_heads_and_preserves_old_run(self):
        import yaml

        self.args.data_fractions = [1.]
        self.args.epochs = 1
        source = self.root / "tiny_locked.yaml"
        source.write_text(yaml.safe_dump(vars(self.args)), encoding="utf-8")
        original_root = Path(self.args.run_root)
        original_root.mkdir()
        marker = original_root / "old_run.txt"
        marker.write_text("preserve")
        args = copy.copy(self.args)
        args.config = str(source)
        args.run_root = str((self.root / "shared").resolve())
        context, basis = self.shared_context()
        bootstrap = (
            "import sys, torch\n"
            "from unittest.mock import patch\n"
            "from training.test_architecture_data_ablation import TinyFrozen\n"
            "from training import train_architecture_data_ablation as base\n"
            "from training import train_shared_architecture_ablation as shared\n"
            "torch.set_num_threads(1)\n"
            "with patch.multiple(base.C, BACKBONE_DIM=8, FEATURE_GRID=2, IMG_SIZE=16, "
            "EMBED_DIM=8, BACKBONE_IMG_SIZE=16, NUM_PATCH_TOKENS=4), "
            "patch.object(base, 'FrozenCXRBackbone', TinyFrozen), "
            "patch.object(shared, 'MIN_BENCHMARK_BATCHES', 2):\n"
            "    shared.main(sys.argv[1:])\n"
        )
        execute = subprocess.run

        def child(command, **kwargs):
            try:
                return execute([sys.executable, "-u", "-c", bootstrap, *command[4:]], **kwargs)
            except subprocess.CalledProcessError:
                kwargs["stdout"].flush()
                self.fail(Path(kwargs["stdout"].name).read_text(encoding="utf-8"))

        with patch.object(shared, "MIN_BENCHMARK_BATCHES", 2):
            shared.benchmark(args, context, basis)
            with patch.object(shared.subprocess, "run", side_effect=child):
                shared.run_sweep(args, context, basis)
        folder = Path(args.run_root) / shared.group_name(1., 42)
        state = torch.load(folder / "shared_last.pt", weights_only=False)
        self.assertEqual(set(state["branches"]), {"B0", "B1", "B2"})
        self.assertTrue((folder / "complete.json").is_file())
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse(list(Path(args.run_root).glob(".*.lock")))


if __name__ == "__main__":
    unittest.main()
