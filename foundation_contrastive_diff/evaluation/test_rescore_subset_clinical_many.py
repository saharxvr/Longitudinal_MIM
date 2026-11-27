"""Presentation formula parity and immutable saved-prediction CPU rescoring."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from scipy.ndimage import label

from evaluation import rescore_subset_clinical_many as rescore


class PresentationMatchingTests(unittest.TestCase):
    def test_split_merge_and_touch_match_presentation_example(self):
        prediction = np.zeros((20, 20), bool)
        reference = np.zeros_like(prediction)
        prediction[2:6, 2:6] = True
        prediction[2:6, 9:12] = True
        prediction[15:18, 2:5] = True
        reference[4:8, 4:10] = True
        reference[18:20, 5:8] = True
        terms = rescore.comparison_terms(rescore.components(prediction), rescore.components(reference))
        np.testing.assert_array_equal(terms, [3, 2, 3, 2, 1, 1])
        self.assertEqual(rescore.scores(terms), dict(
            one_to_many=1., one_to_one_presentation=.4, one_to_one_legacy=.4))
        reverse = rescore.comparison_terms(rescore.components(reference), rescore.components(prediction))
        self.assertEqual(rescore.scores(reverse)["one_to_many"], 1.)

    def test_diagonal_touch_empty_and_unmatched_regions(self):
        left, right = np.zeros((8, 8), bool), np.zeros((8, 8), bool)
        left[1, 1] = right[2, 2] = True
        terms = rescore.comparison_terms(rescore.components(left), rescore.components(right))
        self.assertEqual(rescore.scores(terms)["one_to_many"], 1.)
        self.assertEqual(rescore.scores(terms)["one_to_one_presentation"], 0.)
        right[:] = False
        self.assertEqual(rescore.scores(rescore.comparison_terms(
            rescore.components(left), rescore.components(right)))["one_to_many"], 0.)
        empty = rescore.scores(np.zeros(6, np.int64))
        self.assertEqual(empty["one_to_many"], 1.)
        self.assertEqual(empty["one_to_one_presentation"], 1.)
        self.assertIsNone(empty["one_to_one_legacy"])

    def test_model_first_matching_reproduces_tracked_presentation_reference(self):
        path = (rescore.ROOT.parent / "python_files" / "Sahar_work" /
                "observer_variability_nathalia_thaer_model.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "get_pairwise_detections")
        namespace = dict(np=np, label=label, STRUCT_3x3=rescore.original.STRUCTURE)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        generator = np.random.default_rng(42)
        for _ in range(25):
            prediction, reference = generator.random((2, 20, 20)) > .82
            expected, _ = namespace["get_pairwise_detections"](prediction, reference)
            actual = rescore.comparison_terms(rescore.components(prediction), rescore.components(reference))
            self.assertEqual(actual[4], expected)


class SavedPredictionRescoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rescore.original.torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clinical = self.root / "clinical"
        pair = self.clinical / "Pairs1" / "pair1"
        pair.mkdir(parents=True)
        for name in ("0_prior.nii.gz", "1_current.nii.gz"):
            nib.save(nib.Nifti1Image(np.zeros((16, 16), np.float32), np.eye(4)), pair / name)
        ellipses = [
            {}, dict(cx=4, cy=4, rx=2, ry=2, angle=0, label="Appearance"),
            dict(cx=12, cy=12, rx=2, ry=2, angle=0, label="Disappearance")]
        for doctor in rescore.original.PHYSICIANS:
            path = self.clinical / "Annotations" / doctor / "pair1.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(ellipses), encoding="utf-8")
        self.tools = rescore.original.clinical_tools()
        self.cases = rescore.original.discover_cases(self.clinical, 1)
        positive, negative = self.tools.load_labels_map(str(path), (16, 16))
        values = (positive > 0).astype(np.float32) * .8 - (negative < 0).astype(np.float32) * .6
        self.inference = self.root / "inference"
        models, hashes = {}, {}
        for fraction in rescore.original.FRACTIONS:
            for arm in rescore.original.shared.base.ARMS:
                name = f"{arm}_fraction_{fraction:g}_seed_42"
                models[name] = dict(
                    metadata=dict(model=name, arm=arm, patient_fraction=fraction, seed=42,
                                  epoch=1, selected_pairs=100, optimizer_updates=4,
                                  best_val_signed_dice=.7, checkpoint_sha256="b" * 64),
                    spec={"kwargs": {"out_size": 16}})
                output = self.inference / name / "pair1" / "output.nii.gz"
                output.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(values, np.eye(4)), output)
                hashes[output.relative_to(self.inference).as_posix()] = rescore.original.file_hash(output)
        identity = rescore.original.inference_identity(
            models, self.cases, dict(provenance={"features": {"manifest_sha256": "a" * 64}}), self.tools)
        self.record = dict(identity=identity, prediction_sha256=hashes)
        write = rescore.original.shared.base.write_json
        write(self.inference / "complete.json", self.record)
        self.previous = self.root / "previous"
        protocol = dict(
            pairs=1, seed=42, thresholds=list(rescore.THRESHOLDS), cc_cutoff=250, headline_threshold=.75,
            input_hash=rescore.original.shared.base.json_hash(identity["inputs"]),
            prediction_hash=rescore.original.shared.base.json_hash(hashes),
            evaluation_code_sha256=identity["evaluation_code_sha256"])
        write(self.previous / "summary.json", dict(
            protocol=protocol, headline=[dict(**model["metadata"], threshold=.75) for model in models.values()]))
        rows = []
        for name in models:
            for threshold in rescore.THRESHOLDS:
                values = ((1., 1., 1.) if threshold <= .5 else
                          (1., 0., 2 / 3) if threshold == .75 else (0., 0., 0.))
                for doctor in rescore.original.PHYSICIANS:
                    rows.append(dict(model=name, threshold=threshold, clinician=doctor,
                                     **dict(zip(("F1_positive", "F1_negative", "F1_all"), values))))
        rescore.original.shared.base.write_csv(self.previous / "per_clinician.csv", rows)
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        np.savez_compressed(self.baseline / "pair1.npz",
                            A=(positive > 0).astype(np.int8) - (negative < 0).astype(np.int8))
        write(self.baseline / "manifest.json", {"pairs": {"pair1": {"shape": [16, 16]}}})

    def run_rescore(self):
        with patch.object(rescore, "PRESENTATION_BASELINE", (1., 1.)), \
             patch.object(rescore.original, "FrozenCXRBackbone", side_effect=AssertionError("GPU/model constructed")), \
             patch.object(rescore.original, "build_change_head", side_effect=AssertionError("Head constructed")), \
             patch.object(rescore.original, "run_legacy", side_effect=AssertionError("OV rerun")):
            return rescore.rescore(self.inference, self.previous, self.clinical,
                                   self.baseline, self.root / "report", count=1)

    def test_cpu_end_to_end_reproduces_all_old_scores_without_mutating_sources(self):
        before = rescore.original.file_hash(self.inference / "complete.json")
        previous = rescore.original.file_hash(self.previous / "summary.json")
        result = self.run_rescore()
        self.assertEqual({path.name for path in result.iterdir()}, set(rescore.original.EXPORT_FILES))
        report = rescore.original.read_json(result / "summary.json")
        self.assertEqual(len(report["headline"]), 12)
        self.assertEqual(len(report["sweep"]), 60)
        for row in report["headline"]:
            self.assertAlmostEqual(row["mean_one_to_many_all"], 2 / 3)
            self.assertEqual(row["mean_one_to_many_positive"], 1.)
            self.assertEqual(row["mean_one_to_many_negative"], 0.)
            self.assertAlmostEqual(row["mean_one_to_one_legacy_all"], 2 / 3)
        self.assertEqual(report["baseline"]["one_to_many"], 1.)
        self.assertTrue(report["protocol"]["original_one_to_one_reproduced"])
        self.assertEqual(before, rescore.original.file_hash(self.inference / "complete.json"))
        self.assertEqual(previous, rescore.original.file_hash(self.previous / "summary.json"))
        for path in result.iterdir():
            if path.suffix != ".png":
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("pair1", text)
                self.assertNotIn(str(self.root), text)
        with self.assertRaisesRegex(ValueError, "new report"):
            self.run_rescore()

    def test_changed_prediction_or_annotations_fail_before_export(self):
        path = self.inference / next(iter(self.record["identity"]["checkpoints"])) / "pair1" / "output.nii.gz"
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "prediction changed"):
            self.run_rescore()
        path.write_bytes(original)
        annotation = Path(self.cases[0]["annotations"]["Avi"])
        annotation.write_text(annotation.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "images/annotations differ"):
            self.run_rescore()
        self.assertFalse((self.root / "report").exists())

    def test_previous_score_mismatch_or_baseline_failure_prevents_success_report(self):
        csv_path = self.previous / "per_clinician.csv"
        original = csv_path.read_text(encoding="utf-8")
        csv_path.write_text(original.replace(",1.0,1.0,1.0", ",0.0,1.0,1.0", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one-to-one OV result not reproduced"):
            self.run_rescore()
        csv_path.write_text(original, encoding="utf-8")
        with patch.object(rescore, "PRESENTATION_BASELINE", (.5, .5)):
            with self.assertRaisesRegex(ValueError, "baseline not reproduced"):
                rescore.rescore(self.inference, self.previous, self.clinical,
                                self.baseline, self.root / "report", count=1)
        self.assertFalse((self.root / "report").exists())

    def test_unexpected_model_or_changed_original_report_is_rejected(self):
        record = rescore.original.read_json(self.inference / "complete.json")
        record["identity"]["checkpoints"].pop(next(iter(record["identity"]["checkpoints"])))
        # A full model removal must not be accepted as partial successful evaluation.
        with self.assertRaises(ValueError):
            with patch.object(rescore.original, "verify_predictions", return_value=record):
                with patch.object(rescore.original, "read_json", side_effect=lambda path: record):
                    rescore.validate_saved_run(self.inference, self.previous, self.cases, self.tools)
        summary = rescore.original.read_json(self.previous / "summary.json")
        summary["protocol"]["cc_cutoff"] = 50
        rescore.original.shared.base.write_json(self.previous / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "operating points"):
            self.run_rescore()


if __name__ == "__main__":
    unittest.main()
