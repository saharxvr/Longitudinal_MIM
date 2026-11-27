"""Checkpoint, geometry, real legacy OV and aggregate-only publication checks."""

import copy
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from evaluation import evaluate_subset_clinical_ov as evaluation
from evaluation.sweep_clinical_postprocessing import masks_for
from training import test_architecture_data_ablation as fixtures
from training import train_subset_architecture_ablation as subset


class SubsetClinicalOVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        fixtures.ArchitectureDataTests.setUp(self)
        self.args = subset.configure(self.args, self.root / "subset")
        self.args.data_fractions = [.5, 1.]
        self.args.epochs = 1
        self.context, self.basis = subset.prepare(self.args)
        self.clinical = self.root / "clinical"
        pair = self.clinical / "Pairs1" / "pair1"
        pair.mkdir(parents=True)
        image = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
        for name, array in (("0_prior.nii.gz", image), ("1_current.nii.gz", image[::-1].copy())):
            nib.save(nib.Nifti1Image(array, np.eye(4)), pair / name)
        ellipses = [
            {}, dict(cx=4, cy=4, rx=2, ry=2, angle=0, label="Appearance"),
            dict(cx=12, cy=12, rx=2, ry=2, angle=0, label="Disappearance")]
        for doctor in evaluation.PHYSICIANS:
            annotation = self.clinical / "Annotations" / doctor / "pair1.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps(ellipses), encoding="utf-8")
        self.tools = evaluation.clinical_tools()
        self.cases = evaluation.discover_cases(self.clinical, 1)

    def completed_models(self):
        subset.run_group(self.args, self.context, self.basis, .5, 42)
        return evaluation.load_checkpoints(self.args, self.context, [.5], 42)

    def test_best_validation_checkpoints_and_no_icu_selection(self):
        models = self.completed_models()
        self.assertEqual(set(models), {"B0_fraction_0.5_seed_42", "B1_fraction_0.5_seed_42",
                                      "B2_fraction_0.5_seed_42"})
        for model in models.values():
            self.assertEqual(model["metadata"]["epoch"], 1)
            self.assertEqual(model["metadata"]["selected_pairs"], 32)
            self.assertEqual(model["metadata"]["optimizer_updates"], 1)
        with self.assertRaises(FileNotFoundError):
            evaluation.load_checkpoints(self.args, self.context, [1.], 42)
        path = next(iter(models.values()))["path"]
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            evaluation.load_checkpoints(self.args, self.context, [.5], 42)

    def test_shared_encoder_restore_geometry_and_prediction_reuse(self):
        models = self.completed_models()
        identity = evaluation.inference_identity(models, self.cases, self.context, self.tools)
        inference = self.root / "inference"
        backbone = fixtures.TinyFrozen()
        with patch.object(evaluation, "FrozenCXRBackbone", return_value=backbone), \
             patch.object(backbone, "forward", wraps=backbone.forward) as extract:
            evaluation.infer(models, self.cases, self.context, self.tools, inference, identity, "cpu")
        self.assertEqual(extract.call_count, 2)
        evaluation.verify_predictions(inference, identity, self.cases)
        prior, current, current_image, _ = evaluation.load_case(self.cases[0], self.tools)
        features = self.context["provenance"]["features"]
        with torch.inference_mode():
            inputs = [evaluation.prepare_image(image, 16, 3, True) for image in (prior, current)]
            extracted = [backbone(image[None]) for image in inputs]
            for name, model in models.items():
                output, _ = model["head"](
                    extracted[0]["patch_tokens"], extracted[1]["patch_tokens"],
                    extracted[0]["cls_token"], extracted[1]["cls_token"])
                expected = evaluation.restore_prediction(
                    evaluation.signed_change_map(output)[0, 0], current.shape, 3, True)
                actual = nib.load(inference / name / "pair1" / "output.nii.gz")
                np.testing.assert_array_equal(np.asarray(actual.dataobj), expected)
                np.testing.assert_array_equal(actual.affine, current_image.affine)
        changed = copy.deepcopy(identity)
        changed["inputs"]["pair1"]["current"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluation.verify_predictions(inference, changed, self.cases)
        path = inference / next(iter(models)) / "pair1" / "output.nii.gz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "prediction changed"):
            evaluation.verify_predictions(inference, identity, self.cases)

    def test_real_ov_threshold_sweep_and_sign_specific_cc250(self):
        inference = self.root / "predictions"
        folder = inference / "model" / "pair1"
        folder.mkdir(parents=True)
        positive, negative = self.tools.load_labels_map(
            self.cases[0]["annotations"]["Avi"], (16, 16))
        scores = (positive > 0).astype(np.float32) * .8 - (negative < 0).astype(np.float32) * .6
        nib.save(nib.Nifti1Image(scores, np.eye(4)), folder / "output.nii.gz")
        original = evaluation.file_hash(folder / "output.nii.gz")
        with patch.object(self.tools, "plot_matrix"), patch.object(self.tools, "plot_curves"):
            rows, _ = evaluation.score_model(
                "model", self.cases, self.tools, inference, self.root / "ov",
                evaluation.THRESHOLDS, self.clinical)
        self.assertEqual(len(rows), 25)
        for row in rows:
            if row["threshold"] == .5:
                self.assertEqual(row["F1_all"], 1.)
            elif row["threshold"] == .75:
                self.assertEqual(row["F1_positive"], 1.)
                self.assertEqual(row["F1_negative"], 0.)
                self.assertAlmostEqual(row["F1_all"], 2 / 3)
            elif row["threshold"] == .9:
                self.assertEqual(row["F1_all"], 0.)
        self.assertEqual(original, evaluation.file_hash(folder / "output.nii.gz"))
        scores = np.zeros((512, 512), np.float32)
        scores[1, 1:251] = .9
        scores[3, 1:252] = -.9
        scores[5, 1:252] = .75
        masks = masks_for(scores, f"components_{evaluation.CC_CUTOFF}", .75)
        self.assertEqual(int(masks[0].sum()), 0)
        self.assertEqual(int(masks[1].sum()), 251)
        double = np.zeros((1024, 1024), np.float32)
        double[1, 1:1001] = .9
        double[3, 1:1002] = -.9
        masks = masks_for(double, f"components_{evaluation.CC_CUTOFF}", .75)
        self.assertEqual(int(masks[0].sum()), 0)
        self.assertEqual(int(masks[1].sum()), 1001)

    def test_end_to_end_ov_exports_aggregates_only_and_reuses_inference(self):
        models = self.completed_models()
        inference, destination = self.root / "inference", self.root / "aggregate"
        with patch.object(evaluation, "FrozenCXRBackbone", fixtures.TinyFrozen), \
             patch.object(self.tools, "plot_matrix"), patch.object(self.tools, "plot_curves"):
            evaluation.evaluate(
                models, self.cases, self.tools, self.context, inference, destination, self.clinical, "cpu")
            before = (inference / "complete.json").read_bytes()
            with patch.object(evaluation, "FrozenCXRBackbone", side_effect=AssertionError("Encoder reused")):
                evaluation.evaluate(
                    models, self.cases, self.tools, self.context, inference,
                    self.root / "aggregate_again", self.clinical, "cpu", True)
        self.assertEqual(before, (inference / "complete.json").read_bytes())
        self.assertEqual({path.name for path in destination.iterdir()}, set(evaluation.EXPORT_FILES))
        summary = json.loads((destination / "summary.json").read_text())
        self.assertEqual(len(summary["headline"]), 3)
        self.assertEqual(len(summary["sweep"]), 15)
        self.assertEqual(len((destination / "per_clinician.csv").read_text().splitlines()), 76)
        for path in destination.iterdir():
            if path.suffix != ".png":
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("pair1", text)
                self.assertNotIn("volume-train-", text)
                self.assertNotIn(str(self.root), text)
        self.assertFalse(list(destination.rglob("*.nii.gz")))
        self.assertFalse(list(destination.rglob("*.pt")))
        self.assertTrue((destination / "threshold_sweep.png").is_file())
        with patch.object(evaluation, "ROOT", self.root), \
             patch.object(evaluation.subprocess, "check_output", return_value="main\n"), \
             patch.object(evaluation.subprocess, "run") as git:
            with self.assertRaises(ValueError):
                evaluation.publish(destination)
            export = self.root / "results" / "report"
            export.parent.mkdir()
            destination.rename(export)
            evaluation.publish(export)
        self.assertEqual(len(git.call_args_list), 3)
        self.assertTrue(all("pull" not in call.args[0] for call in git.call_args_list))
        self.assertEqual(len(git.call_args_list[0].args[0][3:]), len(evaluation.EXPORT_FILES))
        (export / "unexpected.nii.gz").write_bytes(b"not an aggregate")
        with patch.object(evaluation, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "allowlist"):
                evaluation.publish(export)

    def test_changed_clinical_inputs_and_ov_mismatch_fail_explicitly(self):
        models = self.completed_models()
        identity = evaluation.inference_identity(models, self.cases, self.context, self.tools)
        inference = self.root / "inference"
        with patch.object(evaluation, "FrozenCXRBackbone", fixtures.TinyFrozen):
            evaluation.infer(models, self.cases, self.context, self.tools, inference, identity, "cpu")
        annotation = Path(self.cases[0]["annotations"]["Avi"])
        annotation.write_text("[]")
        changed = evaluation.inference_identity(models, self.cases, self.context, self.tools)
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluation.verify_predictions(inference, changed, self.cases)
        name = next(iter(models))
        real_read = evaluation.read_json

        def incorrect(path):
            result = real_read(path)
            if path.name == "agreement_matrices.json":
                result["per_label_all"][-1][0] = .123456789
            return result

        with patch.object(self.tools, "plot_matrix"), patch.object(self.tools, "plot_curves"), \
             patch.object(evaluation, "read_json", side_effect=incorrect):
            with self.assertRaisesRegex(ValueError, "disagrees"):
                evaluation.score_model(
                    name, self.cases, self.tools, inference, self.root / "ov",
                    evaluation.THRESHOLDS, self.clinical)

    def test_cli_checkpoint_scope_and_gpu_contention_guard(self):
        import yaml

        source = self.root / "locked.yaml"
        config = dict(vars(self.args), run_root=str(self.root / "original"))
        source.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.args.config = str(source.resolve())
        with patch.object(shared := subset.shared, "MIN_BENCHMARK_BATCHES", 2):
            shared.benchmark(self.args, self.context, self.basis)
        self.completed_models()
        command = [
            "--config", str(source), "--run-root", self.args.run_root,
            "--clinical-root", str(self.clinical), "--inference-dir", str(self.root / "new_inference"),
            "--out-dir", str(self.root / "numeric"), "--fractions", ".5", "--device", "cpu"]
        with patch.object(shared, "MIN_BENCHMARK_BATCHES", 2), \
             patch.object(evaluation, "discover_cases", return_value=self.cases), \
             patch.object(evaluation, "clinical_tools", return_value=self.tools), \
             patch.object(evaluation, "evaluate", return_value=self.root / "numeric") as run:
            evaluation.main(command)
        self.assertEqual(len(run.call_args.args[0]), 3)
        self.assertEqual(run.call_args.args[6], self.clinical)
        self.assertEqual(run.call_args.args[7], "cpu")
        with patch.object(evaluation.os, "name", "posix"), \
             patch.object(evaluation.os, "getuid", return_value=123, create=True), \
             patch.object(evaluation.subprocess, "run", return_value=subprocess.CompletedProcess(
                 ["pgrep"], 0, stdout="running training", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "Training job detected"):
                evaluation.check_gpu_job("cuda")
            evaluation.check_gpu_job("cpu")
        with self.assertRaisesRegex(ValueError, "Infinite"):
            evaluation.clean_ratios({"invalid": float("inf")})
        self.assertEqual(evaluation.clean_ratios({"undefined": float("nan")}), {"undefined": None})


if __name__ == "__main__":
    unittest.main()
