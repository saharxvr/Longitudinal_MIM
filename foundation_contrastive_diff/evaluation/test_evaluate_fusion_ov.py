"""Fixed operating-point boundary checks and real legacy-OV integration."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from evaluation import evaluate_fusion_ov as evaluation


class FusionOVTests(unittest.TestCase):
    def test_strict_threshold_component_boundary_and_signs(self):
        scores = np.zeros((512, 512), dtype=np.float32)
        scores[1, 1:251] = .9                 # Exactly 250: removed.
        scores[3, 1:252] = -.9               # 251: retained.
        scores[5, 1:252] = .75               # Equality: removed.
        scores[7, 1:252] = -.75              # Equality: removed.
        scores[10:20, 1:21] = .9             # Adjacent opposite signs stay separate.
        scores[20:30, 1:21] = -.9
        scores[np.arange(100, 351), np.arange(100, 351)] = .9  # Diagonal: 8-connected.
        result = evaluation.filtered_scores(scores)
        self.assertEqual(int((result > 0).sum()), 251)
        self.assertEqual(int((result < 0).sum()), 251)
        self.assertEqual(scores[1, 1], np.float32(.9))
        for invalid in (np.array([[np.nan]]), np.array([[2.]]), np.zeros((2, 2, 2))):
            with self.assertRaises(ValueError):
                evaluation.filtered_scores(invalid)

    def test_component_area_scales_with_native_resolution(self):
        scores = np.zeros((1024, 1024), dtype=np.float32)
        scores[1, 1:1001] = .9               # 1000 native = 250 equivalent.
        scores[3, 1:1002] = -.9
        result = evaluation.filtered_scores(scores)
        self.assertEqual(int((result > 0).sum()), 0)
        self.assertEqual(int((result < 0).sum()), 1001)

    def test_real_legacy_reports_match_filtered_masks_for_all_arms(self):
        tools = evaluation.clinical_tools()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clinical = root / "clinical"
            scans = clinical / "Pairs1" / "pair1"
            scans.mkdir(parents=True)
            shape = (64, 64)
            for name in ("0_prior.nii.gz", "1_current.nii.gz"):
                nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), scans / name)
            annotations = [{}, dict(cx=15, cy=15, rx=5, ry=5, angle=0, label="Appearance"),
                           dict(cx=45, cy=45, rx=5, ry=5, angle=0, label="Disappearance")]
            hashes = {}
            for doctor in evaluation.PHYSICIANS:
                path = clinical / "Annotations" / doctor / "pair1.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(annotations))
                hashes[f"pair1/{doctor}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            positive, negative = tools.load_labels_map(str(path), shape)
            scores = (positive > 0).astype(np.float32) * .9 - (negative < 0).astype(np.float32) * .9
            source = root / "predictions"
            original_hashes = {}
            for arm in evaluation.ARMS:
                folder = source / f"arm_{arm}"
                (folder / "pair1").mkdir(parents=True)
                output = folder / "pair1" / "output.nii.gz"
                nib.save(nib.Nifti1Image(scores, np.eye(4)), output)
                original_hashes[arm] = hashlib.sha256(output.read_bytes()).hexdigest()
                (folder / "metadata.json").write_text(json.dumps(dict(
                    num_pairs=1, annotation_sha256=hashes, checkpoint_epoch=17)))
                (folder / "report.json").write_text(json.dumps(dict(pairs_completed=1)))
            destination = root / "ov"
            with patch.object(evaluation, "clinical_tools", return_value=tools), \
                 patch.object(tools, "plot_matrix"), patch("sys.stdout", io.StringIO()):
                evaluation.run(source, clinical, destination, count=1)
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(set(summary), set(evaluation.ARMS))
            self.assertEqual(len((destination / "f1_per_clinician.csv").read_text().splitlines()), 36)
            for arm in evaluation.ARMS:
                self.assertEqual(hashlib.sha256(
                    (source / f"arm_{arm}" / "pair1" / "output.nii.gz").read_bytes()).hexdigest(),
                    original_hashes[arm])
                provenance = json.loads((destination / f"arm_{arm}" / "input_provenance.json").read_text())
                self.assertEqual(provenance["postprocessing"]["cc_cutoff"], 250)
                matrices = json.loads((destination / f"arm_{arm}" / "agreement_matrices.json").read_text())
                self.assertAlmostEqual(matrices["per_label_all"][-1][0], 1.)
            self.assertFalse(list(destination.rglob("*.nii.gz")))
            self.assertFalse(list(destination.rglob("comparison.png")))
            with self.assertRaisesRegex(ValueError, "new OV"):
                evaluation.run(source, clinical, destination, count=1)
            metadata_path = source / "arm_S6" / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["annotation_sha256"]["pair1/Avi"] = "changed"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "Annotation changed"):
                evaluation.run(source, clinical, root / "invalid", count=1)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
