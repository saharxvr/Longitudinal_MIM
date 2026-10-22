"""Prediction-only export round trip and input rejection."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from evaluation import export_fusion_predictions as exporter


class PredictionExportTests(unittest.TestCase):
    def test_round_trip_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (64, 64)
            scores = np.zeros(shape, np.float32)
            scores[10:20, 10:20] = .9
            scores[30:40, 30:40] = -.9
            current = root / "current.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), current)
            run, ov, baseline = root / "run", root / "ov", root / "baseline"
            for arm in ("A", "B"):
                path = run / "clinical100_test" / f"arm_{arm}" / "pair1" / "output.nii.gz"
                path.parent.mkdir(parents=True)
                nib.save(nib.Nifti1Image(scores, np.eye(4)), path)
                provenance = ov / f"arm_{arm}" / "input_provenance.json"
                provenance.parent.mkdir(parents=True)
                provenance.write_text(json.dumps(
                    dict(prediction_sha256={"pair1": exporter.sha(path)})))
            prediction = baseline / "pair1" / "output.nii.gz"
            prediction.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(scores, np.eye(4)), prediction)
            crop = prediction.parent / "crop_info.json"
            crop.write_text(json.dumps(dict(current=dict(
                sq_x_min=0, sq_y_min=0, square_size=64, orig_h=64, orig_w=64))))
            reference = root / "hashes.json"
            reference.write_text(json.dumps({"pair1": exporter.sha(prediction)}))
            destination = root / "export"
            with patch.object(exporter, "discover_cases", return_value=[
                    dict(pair_id="pair1", current=str(current))]), \
                    patch("sys.stdout", io.StringIO()):
                exporter.export(run, root, baseline, destination, ov, reference, count=1)
                with np.load(destination / "pair1.npz", allow_pickle=False) as bundle:
                    self.assertEqual(set(bundle.files), {"A", "B", "Itamar"})
                    for key in bundle.files:
                        self.assertEqual(bundle[key].dtype, np.int8)
                        np.testing.assert_array_equal(bundle[key], np.sign(scores))
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["num_pairs"], 1)
                self.assertEqual(manifest["pairs"]["pair1"]["baseline"], "Itamar")
                self.assertEqual({p.name for p in destination.iterdir()},
                                 {"manifest.json", "pair1.npz"})
                with self.assertRaisesRegex(ValueError, "new export"):
                    exporter.export(run, root, baseline, destination, ov, reference, count=1)
                reference.write_text(json.dumps({"pair1": "wrong"}))
                with self.assertRaisesRegex(FileNotFoundError, "historical baseline"):
                    exporter.export(run, root, baseline, root / "bad", ov, reference, count=1)
                self.assertFalse((root / "bad").exists())
                reference.write_text(json.dumps({"pair1": exporter.sha(prediction)}))
                crop.unlink()
                with self.assertRaisesRegex(FileNotFoundError, "crop_info"):
                    exporter.export(run, root, baseline, root / "missing", ov, reference, count=1)
                self.assertFalse((root / "missing").exists())
                source = run / "clinical100_test" / "arm_A" / "pair1" / "output.nii.gz"
                source.write_bytes(b"changed prediction")
                with self.assertRaisesRegex(FileNotFoundError, "A inference"):
                    exporter.export(run, root, baseline, root / "changed", ov, reference, count=1)


if __name__ == "__main__":
    unittest.main()
