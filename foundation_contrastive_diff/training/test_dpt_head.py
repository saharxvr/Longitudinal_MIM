"""CPU checks: python -m unittest training.test_dpt_head."""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.evaluate_rq1 import _tokens, run_eval, select_review_cases
from losses import change_map_loss
from models import DPTCrossAttentionHead, DPTDifferenceHead, DifferenceHead, build_change_head
from training.train_foundation_diff import head_spec, parse_args
from training import train_foundation_diff as trainer
from training.cache_features import slice_last_layer_cache


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_signed_overlap_empty_references_and_exports(self):
        import csv
        import gzip
        from evaluation.training_diagnostics import PredictionDiagnostics, gt_statistics

        gt = torch.tensor([[[[1., -1.], [0., 0.]]], [[[0., 0.], [0., 0.]]]])
        pred = torch.tensor([[[[-1., 1.], [0., 0.]]], [[[.8, 0.], [0., 0.]]]], requires_grad=True)
        diagnostics = PredictionDiagnostics([.5], [0., .05], target_mode="signed_mask", components=True)
        diagnostics.update(pred, gt, dict(anomaly_type=[1, 0], is_pathology=[1, 0], pair_id=["change", "empty"]))
        changed, empty = diagnostics.rows
        self.assertEqual(changed["support_dice"], 1)
        self.assertEqual(changed["signed_dice"], 0)
        self.assertEqual(changed["wrong_sign_pixels"], 2)
        self.assertIsNone(empty["signed_dice"])
        self.assertEqual(empty["predicted_area"], .25)
        self.assertEqual(empty["positive_precision"], 0)
        self.assertIsNone(pred.grad)
        self.assertAlmostEqual(changed["weighted_l1"], changed["foreground_l1_contribution"] +
                               changed["background_l1_contribution"])
        stats = gt_statistics([0., 1e-8, .04, -.2], [0., .05])
        self.assertEqual(stats["cutoff_0_pixels"], 3)
        self.assertEqual(stats["cutoff_0.05_pixels"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics.save(temporary, 1, "train")
            report = json.loads((Path(temporary) / "epoch_0001_train.json").read_text())
            self.assertEqual(report["groups"]["pathology"]["signed_dice"]["n"], 1)
            self.assertIsNone(report["groups"]["nuisance"]["signed_dice"]["mean"])
            with gzip.open(Path(temporary) / "epoch_0001_train.csv.gz", "rt") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_component_matching_agrees_with_legacy_reference_first(self):
        import numpy as np
        from evaluation.evaluate_clinical import clinical_tools
        from evaluation.training_diagnostics import component_counts

        tools = clinical_tools()
        tools.struct = tools.STRUCT_3x3
        generator = np.random.default_rng(42)
        for _ in range(12):
            pred, reference = generator.random((2, 16, 16)) > .7
            expected, _ = tools.get_pairwise_detections(reference, pred)
            self.assertEqual(component_counts(pred, reference)["matched"], expected)

    def test_gt_audit_end_to_end_and_split_exclusion(self):
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation.audit_gt import DEFAULT_THRESHOLDS, run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split, amplitude in (("train", .04), ("val", -.12), ("test", .2)):
                folder = root / "data" / split
                folder.mkdir(parents=True)
                image = np.arange(96, dtype=np.float32).reshape(8, 12)
                gt = np.zeros_like(image)
                gt[2:5, 4:8] = amplitude
                for name, values in (("prior", image), ("current", image + gt), ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(values, np.eye(4)), folder / f"{name}.nii.gz")
                records.append(dict(pair_id=split, split=split, effective_anomaly_type="consolidation"))
            (root / "data" / "manifest_split.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            args = SimpleNamespace(dataset_root=root / "data", out_dir=root / "audit", splits=["train", "val"],
                                   limit=0, seed=42, img_size=16, rotation_k=3, thresholds=DEFAULT_THRESHOLDS,
                                   component_thresholds=[0., .05], overlays_per_group=1)
            report = run(args)
            self.assertEqual(report["audited_pairs"], 2)
            self.assertFalse(report["issues"])
            self.assertEqual(len(report["previews"]), 2)
            self.assertTrue((args.out_dir / "cutoff_diagnostics.png").is_file())
            for preview in report["previews"]:
                with np.load(args.out_dir / preview["arrays"], allow_pickle=False) as arrays:
                    self.assertEqual(arrays["gt_native"].shape, (8, 12))
                    self.assertEqual(arrays["gt_training_float32"].shape, (16, 16))
                    self.assertEqual(str(arrays["pair_id"]), preview["pair_id"])
            self.assertNotIn("test", {row["split"] for row in report["summaries"]})
            summary = next(row for row in report["summaries"] if row["split"] == "train" and
                           row["stage"] == "training_float32" and row["anomaly_type"] == "overall")
            self.assertEqual(summary["cutoff_0.05_entirely_removed_pairs"], 1)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                run(args)

    def test_diagnostics_do_not_change_optimizer_update(self):
        class TinyHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.nn.Parameter(torch.tensor([[[[.2, -.1], [.05, 0.]]]]))

            def forward(self, *inputs):
                return self.output, None

        arguments = ["train", "--accum", "1", "--w_dice", "0", "--w_sign", "0", "--w_l2", "1"]
        with patch.object(sys, "argv", arguments):
            args = parse_args()
        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=torch.tensor([[[[.1, -.2], [0., 0.]]]]), is_pathology=[1], pair_id=["toy"])
        results = []
        for diagnostics in (None, trainer.make_diagnostics(args)):
            head = TinyHead()
            optimizer = torch.optim.SGD(head.parameters(), lr=.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
            trainer.train_one_epoch(head, None, [batch], optimizer, scheduler, "cpu", args,
                                    diagnostics=diagnostics)
            results.append(head.output.detach().clone())
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)


class ClinicalInferenceTests(unittest.TestCase):
    def test_legacy_original_and_square_crop_coordinates(self):
        import nibabel as nib
        import numpy as np
        from evaluation.evaluate_clinical import clinical_tools

        tools = clinical_tools()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.nii.gz"
            scores = np.zeros((4, 4), dtype=np.float32)
            scores[0, 1], scores[2, 3], scores[1, 2] = 0.8, -0.8, 0.5
            nib.save(nib.Nifti1Image(scores, np.eye(4)), output)
            positive, negative, restored = tools.load_model_labels_map(
                output, None, scores.shape, 0, 0, prediction_threshold=0.5)
            np.testing.assert_array_equal(restored, scores)
            np.testing.assert_array_equal(positive, scores > 0.5)
            np.testing.assert_array_equal(negative, scores < -0.5)
            with self.assertRaises(ValueError):
                tools.load_model_labels_map(output, None, (4, 5), 0, 0)
            crop = Path(temporary) / "crop_info.json"
            crop.write_text(json.dumps(dict(current=dict(sq_x_min=1, sq_y_min=2,
                                                         square_size=4, orig_h=7, orig_w=9))))
            expected = np.zeros((7, 9))
            expected[1:5, 2:6] = scores.T
            positive, negative, restored = tools.load_model_labels_map(
                output, crop, (9, 7), 0, 0)
            np.testing.assert_array_equal(restored, expected.T)
            np.testing.assert_array_equal(positive, expected.T > 0)
            np.testing.assert_array_equal(negative, expected.T < 0)

    def test_clinical_annotation_reader_does_not_require_seaborn(self):
        from evaluation.evaluate_clinical import clinical_tools

        clinical_tools()
        with patch.dict(sys.modules, {"seaborn": None}):
            tools = clinical_tools()
            with tempfile.TemporaryDirectory() as temporary:
                doctor = Path(temporary) / "Avi"
                doctor.mkdir()
                annotation = doctor / "pair1.json"
                annotation.write_text(json.dumps([{}, dict(cx=4, cy=5, rx=2, ry=2,
                                                           angle=0, label="Appearance")]))
                resolved = tools.resolve_annotation_path(temporary, "Avi", 1)
                positive, negative = tools.load_labels_map(resolved, (11, 13))
                self.assertTrue(positive.any())
                self.assertFalse(negative.any())

    def test_training_transform_and_asymmetric_inverse(self):
        import numpy as np
        from datasets.pair_dataset import _norm01, _to_chw
        from evaluation.evaluate_clinical import prepare_image, restore_prediction

        array = np.arange(17 * 29, dtype=np.float32).reshape(17, 29)
        expected = torch.rot90(_norm01(_to_chw(array, 32)), 3, (-2, -1))
        torch.testing.assert_close(prepare_image(array, 32, 3), expected)
        landmarks = np.zeros((17, 29), dtype=np.float32)
        landmarks[2, 5], landmarks[14, 24] = 1, -1
        for rotation in range(4):
            rotated = torch.rot90(torch.from_numpy(landmarks), rotation, (-2, -1))
            np.testing.assert_array_equal(restore_prediction(rotated, landmarks.shape, rotation),
                                          landmarks)

    def test_signed_metrics_and_empty_references(self):
        import numpy as np
        from evaluation.evaluate_clinical import score_masks

        positive = np.zeros((9, 13), dtype=bool)
        positive[2:5, 3:7] = True
        empty = np.zeros_like(positive)
        self.assertEqual(score_masks((positive, empty), (positive, empty))["signed_dice"], 1)
        self.assertEqual(score_masks((empty, positive), (positive, empty))["signed_dice"], 0)
        self.assertIsNone(score_masks((empty, empty), (empty, empty))["signed_dice"])
        self.assertGreater(score_masks((positive, empty), (empty, empty))["fp_area_if_no_change"], 0)

    def test_mocked_clinical_export_preserves_geometry_and_provenance(self):
        import contextlib
        import nibabel as nib
        import numpy as np
        from types import SimpleNamespace
        from evaluation import evaluate_clinical as clinical

        class TinyBackbone(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, image):
                return dict(patch_tokens=image, cls_token=torch.zeros(len(image), 1))

        class TinyHead(torch.nn.Module):
            def forward(self, prior, current, *args):
                return (current - prior).clamp(-1, 1), None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (17, 29)
            affine = np.diag([0.7, 0.9, 1, 1])
            prior, current = root / "prior.nii.gz", root / "current.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), prior)
            nib.save(nib.Nifti1Image(np.arange(17 * 29, dtype=np.float32).reshape(shape), affine), current)
            annotation = root / "annotation.json"
            annotation.write_text("[]")
            case = dict(pair_id="pair1", prior=str(prior), current=str(current),
                        annotations={doctor: str(annotation) for doctor in clinical.PHYSICIANS})
            tools = SimpleNamespace(load_labels_map=lambda path, shape:
                                    (np.ones(shape), np.zeros(shape)))
            checkpoint = root / "fake.pt"
            torch.save(dict(epoch=45, head={}, head_spec=dict(type="test_stub", kwargs=dict(
                backbone_dim=1, patch_dim=1, out_size=32)),
                run_config=dict(rotation_k=3, eval_threshold=0.5, target_mode="signed_mask")), checkpoint)
            args = SimpleNamespace(clinical_root=root, num_pairs=1, preflight=False,
                                   ckpt=checkpoint, out_dir=root / "out", device="cpu",
                                   threshold=None, expected_sha256=None)
            with patch.object(clinical, "discover_cases", return_value=[case]), \
                 patch.object(clinical, "clinical_tools", return_value=tools), \
                 patch("models.FrozenCXRBackbone", TinyBackbone), \
                 patch("models.build_change_head", return_value=TinyHead()), \
                 contextlib.redirect_stdout(io.StringIO()):
                clinical.run(args)
                with self.assertRaises(ValueError):
                    clinical.run(args)
            output = nib.load(args.out_dir / "pair1" / "output.nii.gz")
            self.assertEqual(output.shape, shape)
            np.testing.assert_allclose(output.affine, affine)
            self.assertEqual(json.loads((args.out_dir / "metadata.json").read_text())["checkpoint_epoch"], 45)
            self.assertEqual(json.loads((args.out_dir / "report.json").read_text())["pairs_completed"], 1)
            self.assertTrue((args.out_dir / "pair1" / "review.png").is_file())
            self.assertEqual(len((args.out_dir / "per_pair.csv").read_text().splitlines()), 7)


class RotatedMaskRunTests(unittest.TestCase):
    def test_small_cutoff_config_and_signed_target_boundary(self):
        config_root = Path(__file__).resolve().parents[1] / "configs"
        with patch.object(sys, "argv", ["train", "--config", str(config_root / "last1_rot90cw_mask_fp075.yaml")]):
            baseline = parse_args()
        with patch.object(sys, "argv", ["train", "--config", str(config_root / "last1_rot90cw_mask_thr0p0001_fp075.yaml")]):
            args = parse_args()
        allowed_changes = {"config", "save_folder", "plots_folder", "target_threshold",
                           "diagnostics_gt_thresholds", "diagnostics_components"}
        self.assertEqual({key: value for key, value in vars(args).items() if key not in allowed_changes},
                         {key: value for key, value in vars(baseline).items() if key not in allowed_changes})
        self.assertEqual(args.target_threshold, .0001)
        self.assertNotEqual(args.save_folder, baseline.save_folder)
        self.assertNotEqual(args.plots_folder, baseline.plots_folder)
        self.assertEqual((args.eval_threshold, args.eval_gt_threshold), (.5, .05))
        self.assertTrue(args.diagnostics_components)
        self.assertIn(.0001, args.diagnostics_gt_thresholds)
        gt = torch.tensor([[[[0., .000099, -.000099, .0001, -.0001, .000101, -.000101, .04]]]])
        target = torch.tensor([[[[0., 0., 0., 0., 0., 1., -1., 1.]]]])
        pred = torch.full_like(gt, .2, requires_grad=True)
        options = {key: getattr(args, key) for key in ("tau", "w_l1", "w_l2", "w_dice", "w_sign",
                   "pos_weight", "target_mode", "target_threshold", "background_weight")}
        loss, _ = change_map_loss(pred, gt, **options)
        weight = torch.where(target != 0, 11., .75)
        expected = (weight * ((pred - target).abs() + (pred - target).square())).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(float(change_map_loss(target, gt, **options)[0]), 0.)
        loss.backward()
        self.assertGreater(float(pred.grad[..., 0]), 0.)
        self.assertLess(float(pred.grad[..., 5]), 0.)
        self.assertGreater(float(pred.grad[..., 6]), 0.)
        diagnostics = trainer.make_diagnostics(args)
        diagnostics.update(pred.detach(), gt, dict(pair_id=["boundary"], is_pathology=[1]))
        self.assertAlmostEqual(diagnostics.rows[0]["target_area"], 3 / 8)

    def test_training_and_validation_use_mask_targets_and_raw_gt_support(self):
        class PerfectMaskHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output = torch.nn.Parameter(torch.tensor([[[[1.0, -1.0, 0.0]]]]))

            def forward(self, *inputs):
                return self.output, None

        batch = dict(p_prior=torch.zeros(1, 1, 1), p_curr=torch.zeros(1, 1, 1),
                     cls_prior=torch.zeros(1, 1), cls_curr=torch.zeros(1, 1),
                     gt_diff=torch.tensor([[[[0.08, -0.12, 0.0]]]]), is_pathology=torch.ones(1))
        arguments = ["train", "--target_mode", "signed_mask", "--background_weight", "0.75",
                     "--w_dice", "0", "--w_sign", "0", "--w_l2", "1", "--accum", "1"]
        with patch.object(sys, "argv", arguments):
            args = parse_args()
        head = PerfectMaskHead()
        optimizer = torch.optim.SGD(head.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        metrics = trainer.train_one_epoch(head, None, [batch], optimizer, scheduler, "cpu", args)
        self.assertEqual(metrics["total"], 0.0)
        validation = trainer.evaluate(head, None, [batch], "cpu", threshold=0.5,
                                      gt_threshold=0.05, target_mode="signed_mask", background_weight=0.75)
        self.assertAlmostEqual(validation["dice"], 1.0)
        self.assertAlmostEqual(validation["sens_pos"], 1.0, places=5)
        self.assertAlmostEqual(validation["sens_neg"], 1.0, places=5)
        self.assertEqual(validation["l1_ch"], 0.0)

    def test_native_rotated_cache_metadata_preview_and_reuse_guard(self):
        from training import cache_features
        from datasets.pair_dataset import CachedPairDataset

        sample = dict(img_prior=torch.arange(16).float().reshape(1, 4, 4) / 15,
                      img_curr=torch.ones(1, 4, 4), gt_diff=torch.full((1, 4, 4), -0.08),
                      anomaly_type=1, direction=0, is_pathology=1,
                      change_group_id="case/pair0", pair_id="case/pair0/variant0", rotation_k=3)

        class TinyBackbone(torch.nn.Module):
            def forward(self, images):
                values = images.mean((-2, -1))
                return dict(patch_tokens=values[:, None].expand(-1, 4, -1), cls_token=values)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest_split.jsonl").write_text(json.dumps(dict(
                pair_id=sample["pair_id"], split="train", effective_anomaly_type="consolidation")) + "\n")
            cache = root / "cache"
            preview = root / "input_preview.png"
            arguments = ["cache", "--dataset_root", str(root), "--cache_dir", str(cache),
                         "--last_n_layers", "1", "--rotation_k", "3", "--device", "cpu",
                         "--num_workers", "0", "--preview_path", str(preview)]
            with patch.object(sys, "argv", arguments), \
                 patch.object(cache_features, "LongitudinalPairDataset", return_value=[sample]) as dataset, \
                 patch.object(cache_features, "_build_backbone", return_value=TinyBackbone()):
                self.assertEqual(cache_features.main(), 0)
                dataset.assert_called_with(str(root), split=None, img_size=512, rotation_k=3)
                self.assertEqual(cache_features.main(), 0)
            self.assertTrue(preview.is_file())
            metadata = json.loads((cache / "cache_config.json").read_text())
            self.assertEqual(metadata["rotation_k"], 3)
            cached = CachedPairDataset(str(root), str(cache), rotation_k=3)
            self.assertEqual(cached[0]["rotation_k"], 3)
            torch.testing.assert_close(cached[0]["gt_diff"], sample["gt_diff"], atol=1e-4, rtol=0)
            with self.assertRaisesRegex(ValueError, "rotation"):
                CachedPairDataset(str(root), str(cache), rotation_k=0)
            arguments[arguments.index("--rotation_k") + 1] = "0"
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "configuration differs"):
                cache_features.main()

    def test_signed_mask_loss_and_modest_background_discount(self):
        gt = torch.tensor([[[[0.08, -0.12, 0.0, 0.04]]]])
        perfect = torch.tensor([[[[1.0, -1.0, 0.0, 0.0]]]])
        options = dict(w_l1=1, w_l2=1, w_dice=0, w_sign=0,
                       target_mode="signed_mask", target_threshold=0.05)
        self.assertEqual(float(change_map_loss(perfect, gt, **options)[0]), 0.0)
        wrong = perfect.clone()
        wrong[..., 0] = -1
        self.assertGreater(float(change_map_loss(wrong, gt, **options)[0]), 0.0)
        false_positive = perfect.clone()
        false_positive[..., 2] = 0.5
        original = change_map_loss(false_positive, gt, **options)[0]
        discounted = change_map_loss(false_positive, gt, background_weight=0.75, **options)[0]
        torch.testing.assert_close(discounted, 0.75 * original)
        missed = torch.zeros_like(gt)
        torch.testing.assert_close(change_map_loss(missed, gt, **options)[0],
                                   change_map_loss(missed, gt, background_weight=0.75, **options)[0])
        nuisance = change_map_loss(torch.full_like(gt, 0.5), torch.zeros_like(gt),
                                   background_weight=0.75, **options)[0]
        self.assertGreater(float(nuisance), 0.0)
        self.assertTrue(torch.isfinite(nuisance))

    def test_distinct_prediction_and_gt_thresholds(self):
        from evaluation.change_detection_metrics import directional_sensitivity

        pred = torch.tensor([[[[0.9, -0.8, 0.0]]]])
        gt = torch.tensor([[[[0.08, -0.12, 0.0]]]])
        metrics = directional_sensitivity(pred, gt, 0.5, 0.05)
        self.assertAlmostEqual(metrics["sensitivity_positive"], 1.0, places=5)
        self.assertAlmostEqual(metrics["sensitivity_negative"], 1.0, places=5)

    def test_dataset_rotates_images_and_signed_gt_together(self):
        import numpy as np
        import nibabel as nib
        from datasets.pair_dataset import LongitudinalPairDataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "case/pair0/variant0"
            folder.mkdir(parents=True)
            prior = np.arange(16, dtype=np.float32).reshape(4, 4)
            gt = np.zeros((4, 4), dtype=np.float32)
            gt[0, 1], gt[2, 3] = 0.2, -0.1
            for name, array in (("prior", prior), ("current", prior + gt), ("diff_map", gt)):
                nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f"{name}.nii.gz")
            record = dict(pair_id="case/pair0/variant0", split="train", realized_change=True,
                          effective_anomaly_type="consolidation")
            (root / "manifest_split.jsonl").write_text(json.dumps(record) + "\n")
            original = LongitudinalPairDataset(str(root), img_size=4)[0]
            for rotation in (1, 2, 3):
                rotated = LongitudinalPairDataset(str(root), img_size=4, rotation_k=rotation)[0]
                for key in ("img_prior", "img_curr", "gt_diff"):
                    torch.testing.assert_close(rotated[key], torch.rot90(original[key], rotation, (-2, -1)))
                self.assertEqual(rotated["rotation_k"], rotation)
            with self.assertRaises(ValueError):
                LongitudinalPairDataset(str(root), rotation_k=4)


class ReviewSelectionTests(unittest.TestCase):
    def test_review_export_end_to_end(self):
        import numpy as np
        import nibabel as nib
        from PIL import Image
        from evaluation import evaluate_rq1 as evaluator

        spec = {"type": "dpt_cross_attention", "kwargs": dict(
            backbone_dim=16, patch_dim=16, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12, num_layers=1)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index, value in enumerate((0.2, -0.2, 0.0)):
                pair_id = f"case/pair{index}/variant0"
                folder = root / pair_id
                folder.mkdir(parents=True)
                prior = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
                gt = np.zeros((32, 32), dtype=np.float32)
                gt[8:20, 8:20] = value
                for name, array in (("prior", prior), ("current", prior + gt), ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f"{name}.nii.gz")
                records.append(dict(pair_id=pair_id, split="val", realized_change=bool(value),
                                    effective_anomaly_type="consolidation" if value else "none",
                                    change_group_id=str(index)))
                cache = root / "cache" / pair_id
                cache.mkdir(parents=True)
                torch.save(dict(p_prior=torch.randn(25, 16), p_curr=torch.randn(25, 16),
                                cls_prior=torch.randn(16), cls_curr=torch.randn(16),
                                gt_diff=torch.from_numpy(gt)[None], anomaly_type=1 if value else 0,
                                direction=0, is_pathology=int(bool(value)),
                                change_group_id=str(index), pair_id=pair_id), cache / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(dict(head=build_change_head(spec).state_dict(), head_spec=spec,
                            epoch=55), checkpoint)
            output = root / "review"
            arguments = ["eval", "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--ckpt", str(checkpoint), "--split", "val", "--threshold", "0.05",
                         "--device", "cpu", "--batch_size", "2", "--num_workers", "0",
                         "--out_dir", str(output), "--review"]
            with patch.object(sys, "argv", arguments):
                evaluator.main()
            cases = json.loads((output / "review_cases.json").read_text())
            metadata = json.loads((output / "review_metadata.json").read_text())
            self.assertEqual(len(cases), 3)
            self.assertEqual(metadata["evaluated_pairs"], 3)
            self.assertEqual(metadata["checkpoint_epoch"], 55)
            self.assertEqual(metadata["checkpoint_sha256"], evaluator.file_sha256(checkpoint))
            self.assertTrue((output / "index.md").is_file())
            self.assertEqual(len((output / "per_pair_val.csv").read_text().splitlines()), 4)
            self.assertEqual(len(list(output.glob("contact_*.png"))), 3)
            for row in cases:
                with Image.open(output / row["panel"]) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertTrue(any(low != high for low, high in image.convert("RGB").getextrema()))
            for record in records:
                path = root / "cache" / record["pair_id"] / "feat.pt"
                cached = torch.load(path, weights_only=False)
                cached["rotation_k"] = 3
                cached["gt_diff"] = torch.rot90(cached["gt_diff"], 3, (-2, -1))
                torch.save(cached, path)
            run_config = dict(rotation_k=3, target_mode="signed_mask", eval_threshold=0.5,
                              eval_gt_threshold=0.05)
            torch.save(dict(head=build_change_head(spec).state_dict(), head_spec=spec,
                            epoch=1, run_config=run_config), checkpoint)
            rotated_arguments = list(arguments)
            threshold_index = rotated_arguments.index("--threshold")
            del rotated_arguments[threshold_index:threshold_index + 2]
            rotated_output = root / "rotated_review"
            rotated_arguments[rotated_arguments.index("--out_dir") + 1] = str(rotated_output)
            with patch.object(sys, "argv", rotated_arguments):
                evaluator.main()
            rotated_metadata = json.loads((rotated_output / "review_metadata.json").read_text())
            self.assertEqual(rotated_metadata["rotation_k"], 3)
            self.assertEqual(rotated_metadata["threshold"], 0.5)
            self.assertEqual(rotated_metadata["gt_threshold"], 0.05)
            self.assertEqual(rotated_metadata["target_mode"], "signed_mask")
            rotated_cases = json.loads((rotated_output / "review_cases.json").read_text())
            self.assertEqual({row["direction"] for row in rotated_cases if row["is_pathology"]},
                             {"positive", "negative"})
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "empty"):
                evaluator.main()
            rows = [dict(row) for row in cases]
            rows.pop()
            dataset = evaluator.CachedPairDataset(str(root), str(root / "cache"), split="val")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                evaluator.export_review(build_change_head(spec), None, dataset, rows, [],
                                        str(root), "val", output, "cpu", 0.05, img_size=32)
            (root / "cache" / records[0]["pair_id"] / "feat.pt").unlink()
            arguments[arguments.index("--out_dir") + 1] = str(root / "new_review")
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(FileNotFoundError, "complete cache"):
                evaluator.main()

    def test_stratified_selection_is_reproducible_and_diverse(self):
        rows = []
        for anomaly in ("consolidation", "pleural_effusion"):
            for direction in ("positive", "negative", "mixed"):
                for index in range(40):
                    rows.append(dict(pair_id=f"{anomaly}_{direction}_{index:02}",
                                     change_group_id=f"{anomaly}_{direction}_{index // 2}",
                                     anomaly_type=anomaly, direction=direction,
                                     is_pathology=1, dice=index / 40, nuisance_fp=None))
        for index in range(60):
            rows.append(dict(pair_id=f"nuisance_{index:02}", change_group_id="",
                             anomaly_type="none", direction="below_threshold",
                             is_pathology=0, dice=None, nuisance_fp=index / 60))
        chosen = select_review_cases(rows, per_group=12, nuisance_count=20)
        self.assertEqual(chosen, select_review_cases(list(reversed(rows)), 12, 20))
        self.assertEqual(len(chosen), 6 * 12 + 20)
        self.assertEqual(len({row["pair_id"] for row in chosen}), len(chosen))
        self.assertEqual({row["selection"] for row in chosen},
                         {"lowest_dice", "middle_dice", "highest_dice", "seeded_random", "highest_fp"})
        self.assertIn("nuisance_59", {row["pair_id"] for row in chosen})
        for group in {row["review_group"] for row in chosen} - {"nuisance"}:
            identities = [row["change_group_id"] for row in chosen if row["review_group"] == group]
            self.assertEqual(len(identities), len(set(identities)))

    def test_small_buckets_and_invalid_counts(self):
        row = dict(pair_id="one", anomaly_type="consolidation", direction="positive",
                   is_pathology=1, dice=0.5, nuisance_fp=None)
        self.assertEqual(len(select_review_cases([row])), 1)
        for count in (1, 2, 3):
            rows = [dict(row, pair_id=str(index)) for index in range(8)]
            self.assertEqual(len(select_review_cases(rows, per_group=count)), count)
        with self.assertRaises(ValueError):
            select_review_cases([row], per_group=0)

    def test_review_metrics_include_nuisance_and_direction(self):
        class FixedHead(torch.nn.Module):
            def forward(self, *inputs):
                return torch.tensor([[[[0.2, -0.2]]], [[[0.2, 0.0]]]]), None

        batch = dict(p_prior=torch.zeros(2, 1, 1), p_curr=torch.zeros(2, 1, 1),
                     cls_prior=torch.zeros(2, 1), cls_curr=torch.zeros(2, 1),
                     gt_diff=torch.tensor([[[[0.2, -0.2]]], [[[0.0, 0.0]]]]),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]),
                     pair_id=["pathology", "control"], change_group_id=["a", "b"])
        rows = []
        report, _ = run_eval(FixedHead(), None, [batch], "cpu", 0.05, review_rows=rows)
        self.assertEqual(rows[0]["direction"], "mixed")
        self.assertEqual(rows[0]["dice"], 1.0)
        self.assertEqual(rows[1]["nuisance_fp"], 0.5)
        self.assertIsNone(rows[1]["dice"])
        self.assertEqual(report["overall"]["n"], 1)


class DPTHeadTests(unittest.TestCase):
    head_type = "dpt"
    config_path = "configs/last4_dpt.yaml"
    num_layers = 4

    def setUp(self):
        torch.manual_seed(0)
        self.spec = {"type": self.head_type, "kwargs": dict(
            backbone_dim=16, patch_dim=16 * self.num_layers, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12)}
        if self.num_layers == 1:
            self.spec["kwargs"]["num_layers"] = 1
        self.prior = torch.randn(2, 25, 16 * self.num_layers)
        self.current = torch.randn_like(self.prior)
        self.cls_prior = torch.randn(2, 16)
        self.cls_current = torch.randn(2, 16)

    def test_forward_loss_backward_and_reload(self):
        head = build_change_head(self.spec)
        inputs = (self.prior, self.current, self.cls_prior, self.cls_current)
        prediction, embedding = head(*inputs)
        self.assertEqual(prediction.shape, (2, 1, 32, 32))
        self.assertEqual(embedding.shape, (2, 12))
        self.assertLessEqual(float(prediction.detach().abs().max()), 1.0)
        loss, _ = change_map_loss(prediction, torch.zeros_like(prediction),
                                  w_l1=1, w_l2=1, w_dice=0, w_sign=0)
        loss.backward()
        for project in head.projections:
            self.assertGreater(float(project.weight.grad.abs().sum()), 0)
        for parameter in head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer = torch.optim.AdamW(head.parameters())
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"head": head.state_dict(), "head_spec": self.spec,
                    "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_change_head(checkpoint["head_spec"])
        restored.load_state_dict(checkpoint["head"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(restored(*inputs)[0], head(*inputs)[0])

    def test_reject_wrong_cache(self):
        with self.assertRaises(ValueError):
            DPTDifferenceHead(patch_dim=768)
        head = build_change_head(self.spec)
        with self.assertRaises(ValueError):
            head(self.prior[..., :-1], self.current[..., :-1])
        with self.assertRaises(ValueError):
            head(self.prior[:, :-1], self.current[:, :-1])

    def test_default_head_and_config_override(self):
        self.assertIsInstance(build_change_head(head_spec(768)), DifferenceHead)
        with patch.object(sys, "argv", ["train", "--config", self.config_path,
                                        "--batch_size", "2"]):
            args = parse_args()
        self.assertEqual((args.head, args.feat_last_k, args.batch_size),
                 (self.head_type, self.num_layers, 2))
        spec = head_spec(768 * self.num_layers, args.head)
        self.assertEqual(spec["type"], self.head_type)
        self.assertEqual(spec["kwargs"].get("num_layers", 4), self.num_layers)
        self.assertEqual((args.w_l1, args.w_l2, args.w_dice, args.w_sign), (1, 1, 0, 0))

    def test_evaluator_and_feature_slicing(self):
        batch = dict(p_prior=self.prior, p_curr=self.current,
                     cls_prior=self.cls_prior, cls_curr=self.cls_current,
                     gt_diff=torch.zeros(2, 1, 32, 32),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]))
        batch["gt_diff"][0, 0, 8:20, 8:20] = 0.2
        sliced = _tokens(batch, None, "cpu", 16)
        torch.testing.assert_close(sliced[0], self.prior[..., -16:])
        report, _ = run_eval(build_change_head(self.spec), None, [batch], "cpu", 0.05,
                     self.spec["kwargs"]["patch_dim"])
        self.assertEqual(report["overall"]["n"], 1)
        self.assertEqual(report["nuisance_false_positive_area"]["n"], 1)

    def test_cached_training_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ("train", "val"):
                for index in range(2):
                    pair_id = f"{split}_{index}"
                    records.append(dict(pair_id=pair_id, split=split,
                                        effective_anomaly_type="none"))
                    folder = root / "cache" / pair_id
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=self.prior[index], p_curr=self.current[index],
                                    cls_prior=self.cls_prior[index], cls_curr=self.cls_current[index],
                                    gt_diff=torch.ones(1, 32, 32) * 0.2,
                                    anomaly_type=1, direction=1, is_pathology=1,
                                    change_group_id=pair_id, pair_id=pair_id), folder / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            arguments = ["train", "--config", self.config_path,
                         "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--save_folder", str(root / "checkpoints"),
                         "--plots_folder", str(root / "plots"), "--batch_size", "2",
                         "--accum", "1", "--num_workers", "0", "--device", "cpu",
                         "--dpt_channels", "8"]
            with patch.multiple(trainer.C, BACKBONE_DIM=16, FEATURE_GRID=5,
                                IMG_SIZE=32, EMBED_DIM=12), \
                    patch.object(trainer, "plot_curves"), \
                    patch.object(trainer, "save_sample_plots"):
                with patch.object(sys, "argv", arguments + ["--epochs", "1"]):
                    trainer.main()
                first = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
                self.assertEqual(first["best_dice"], first["val"]["dice"])
                with patch.object(sys, "argv", arguments + ["--epochs", "2", "--resume"]):
                    trainer.main()
            checkpoint = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["best_dice"], max(first["best_dice"], checkpoint["val"]["dice"]))
            self.assertEqual(checkpoint["head_spec"]["type"], self.head_type)
            self.assertEqual(len((root / "checkpoints" / "metrics.csv").read_text().splitlines()), 3)
            for epoch in (1, 2):
                for split in ("train", "val"):
                    report = root / "checkpoints" / "diagnostics" / f"epoch_{epoch:04d}_{split}.json"
                    diagnostics = json.loads(report.read_text())
                    self.assertEqual(diagnostics["groups"]["overall"]["pairs"], 2)
                    self.assertEqual(diagnostics["metadata"]["training_health"]["optimizer_steps"], 1)


class DPTCrossAttentionTests(DPTHeadTests):
    head_type = "dpt_cross_attention"
    config_path = "configs/last4_dpt_cross_attention.yaml"

    def test_temporal_attention_gradients_and_order(self):
        head = build_change_head(self.spec)
        prediction, _ = head(self.prior, self.current)
        prediction.square().mean().backward()
        for blocks in head.temporal_blocks:
            for block in blocks:
                gradient = block.attention.in_proj_weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
        with torch.no_grad():
            forward = head._project_streams(self.prior, self.current)
            reverse = head._project_streams(self.current, self.prior)
            identical = head._project_streams(self.prior, self.prior)
            offset = torch.randn_like(self.prior)
            shifted = head._project_streams(self.prior + offset, self.current + offset)
        for actual, swapped, unchanged in zip(forward, reverse, identical):
            torch.testing.assert_close(actual, -swapped)
            torch.testing.assert_close(unchanged, torch.zeros_like(unchanged))
        self.assertTrue(any(not torch.allclose(actual, shifted_actual)
                            for actual, shifted_actual in zip(forward, shifted)))

    def test_reject_invalid_attention_settings(self):
        for settings in ({"num_heads": 3}, {"num_heads": 0}, {"attention_depth": 0}):
            with self.assertRaises(ValueError):
                DPTCrossAttentionHead(**self.spec["kwargs"], **settings)


class SingleLayerCrossAttentionTests(DPTCrossAttentionTests):
    config_path = "configs/last1_dpt_cross_attention.yaml"
    num_layers = 1


class CacheSliceTests(unittest.TestCase):
    def test_layer_aware_launcher_preflight(self):
        script = Path(__file__).with_name("run_last4_dpt.sh").read_text()
        preflight = script.split("python -u - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(pair_id=split, split=split, patient_key=split)
                       for split in ("train", "val", "test")]
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            for layers in (1, 4):
                cache = root / str(layers)
                for record in records:
                    folder = cache / record["pair_id"]
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=torch.zeros(1369, 768 * layers).half(),
                                    p_curr=torch.zeros(1369, 768 * layers).half(),
                                    cls_prior=torch.zeros(768).half(), cls_curr=torch.zeros(768).half(),
                                    gt_diff=torch.zeros(1, 512, 512).half()), folder / "feat.pt")
                with patch.dict(os.environ, DATA=str(root), CACHE_DIR=str(cache), CACHE_LAYERS=str(layers)), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "mem_get_info", return_value=(12_000_000_000, 24_000_000_000)):
                    exec(compile(preflight, "launcher_preflight", "exec"), {})
                    if layers == 4:
                        with patch.dict(os.environ, CACHE_LAYERS="1"), self.assertRaises(SystemExit):
                            exec(compile(preflight, "launcher_preflight", "exec"), {})

    def test_compact_exact_resumable_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            (source / "pair").mkdir(parents=True)
            (root / "manifest_split.jsonl").write_text(
                json.dumps(dict(pair_id="pair", split="train")), encoding="utf-8")
            cached = dict(p_prior=torch.randn(25, 64).half(), p_curr=torch.randn(25, 64).half(),
                          cls_prior=torch.randn(16).half(), cls_curr=torch.randn(16).half(),
                          gt_diff=torch.randn(1, 32, 32).half(), pair_id="pair",
                          anomaly_type=1, direction=1, is_pathology=1, change_group_id="group")
            source_path = source / "pair" / "feat.pt"
            torch.save(cached, source_path)
            before = source_path.read_bytes()
            with patch.multiple(trainer.C, NUM_PATCH_TOKENS=25, BACKBONE_DIM=16):
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 1, "skipped": 0})
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 0, "skipped": 1})
                result = torch.load(target / "pair" / "feat.pt", weights_only=True)
                for key in ("p_prior", "p_curr"):
                    torch.testing.assert_close(result[key], cached[key][..., -16:])
                    self.assertEqual(result[key].untyped_storage().nbytes(), result[key].numel() * 2)
                torch.testing.assert_close(result["gt_diff"], cached["gt_diff"])
                self.assertEqual(source_path.read_bytes(), before)
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, source)
                result["p_prior"].zero_()
                torch.save(result, target / "pair" / "feat.pt")
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, target)
                source_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    slice_last_layer_cache(root, source, target)


if __name__ == "__main__":
    unittest.main()