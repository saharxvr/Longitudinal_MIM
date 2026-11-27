"""Verify the campus Bash launcher's process-stop scope without touching real processes."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SCRIPT = Path(__file__).with_name("run_subset_clinical_ov.sh")


class CampusLauncherTests(unittest.TestCase):
    def setUp(self):
        source = SCRIPT.read_text(encoding="utf-8").split("<<'PY_STOP'\n", 1)[1].split("\nPY_STOP", 1)[0]
        self.module = {"__name__": "launcher_test"}
        exec(compile(source, str(SCRIPT), "exec"), self.module)
        self.root = "/exact/approved/run"
        self.command = ["python", "-u", "-m", "training.train_subset_architecture_ablation",
                        "--run-root", self.root, "--mode", "sweep"]
        self.processes = {
            10: (1, self.command),
            11: (10, ["python", "-m", "training.train_subset_architecture_ablation", "--mode", "group"]),
            12: (11, ["worker"]),
            99: (1, ["python", "-m", "training.train_subset_architecture_ablation",
                     "--run-root", "/unrelated/run", "--mode", "sweep"]),
        }
        self.stopped = set()
        self.kill = Mock(side_effect=lambda pid, sig: self.stopped.add(pid))
        self.module["os"] = SimpleNamespace(getuid=lambda: 123, kill=self.kill)
        self.module["time"] = SimpleNamespace(sleep=Mock())
        self.module["owned_processes"] = lambda: self.processes
        self.identity = lambda pid: None if pid in self.stopped else (
            123, str(pid * 100), tuple(self.processes[pid][1]), "S")
        self.module["process_identity"] = self.identity

    def test_explicit_stop_only_matches_approved_root_and_descendants(self):
        self.module["stop_training"](self.root, True)
        self.assertEqual(self.stopped, {10, 11, 12})
        self.assertEqual(self.kill.call_args_list[0].args[0], 10)
        self.assertNotIn(99, self.stopped)

    def test_no_stop_without_flag_and_no_stop_for_other_run(self):
        with self.assertRaisesRegex(SystemExit, "--stop-training"):
            self.module["stop_training"](self.root, False)
        self.kill.assert_not_called()
        self.module["stop_training"]("/absent/run", True)
        self.kill.assert_not_called()

    def test_changed_process_identity_or_owner_is_not_signalled(self):
        calls = {}

        def reused(pid):
            calls[pid] = calls.get(pid, 0) + 1
            identity = self.identity(pid)
            if pid == 10 and calls[pid] > 1:
                return (999, "new-start", identity[2], "S")
            return identity

        self.module["process_identity"] = reused
        with self.assertRaisesRegex(SystemExit, "identity changed"):
            self.module["stop_training"](self.root, True)
        self.kill.assert_not_called()

    def test_ambiguous_launchers_and_unresponsive_children_stop_launch(self):
        self.processes[20] = (1, self.command)
        with self.assertRaisesRegex(SystemExit, "Multiple"):
            self.module["stop_training"](self.root, True)
        self.kill.assert_not_called()
        self.processes.pop(20)
        self.kill.side_effect = None
        with self.assertRaisesRegex(SystemExit, "Still exiting"):
            self.module["stop_training"](self.root, True)
        self.assertEqual(self.module["time"].sleep.call_count, 30)


if __name__ == "__main__":
    unittest.main()
