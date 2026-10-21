"""Tests for case selection, export integrity, and checkpoint guards."""

import copy
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluation.export_layer_predictions import (
    LAYERS, ROOT, export_maps, file_sha256, load_diagnostics, save_case, select_cases,
    validate_checkpoint,
)


def diagnostic_rows():
    rows = {layer: {} for layer in LAYERS}
    for anomaly in ("consolidation", "fluid_overload", "pleural_effusion", "pneumothorax"):
        for winner in LAYERS:
            for index in range(3):
                pair = f"{anomaly}_{winner}_{index}"
                for layer in LAYERS:
                    rows[layer][pair] = dict(
                        pair_id=pair, change_group_id=pair, anomaly_type=anomaly, is_pathology="1",
                        positive_reference="1", negative_reference="0", signed_reference="1",
                        signed_dice="0.8" if layer == winner else "0.5", predicted_area="0.25")
    for index in range(5):
        pair = f"nuisance_{index}"
        for layer in LAYERS:
            rows[layer][pair] = dict(
                pair_id=pair, change_group_id=pair, anomaly_type="none", is_pathology="0",
                positive_reference="0", negative_reference="0", signed_reference="0",
                signed_dice="", predicted_area=str(index / 10))
    return rows


class ExportLayerPredictionTests(unittest.TestCase):
    def test_selection_is_bounded_diverse_and_deterministic(self):
        rows = diagnostic_rows()
        selected = select_cases(rows)
        self.assertEqual(selected, select_cases(rows))
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({row["change_group_id"] for row in selected}), 32)
        self.assertEqual(sum(row["selection"] == "seeded_pathology_control" for row in selected), 4)
        nuisance = [row for row in selected if row["selection"] == "high_nuisance_predicted_area"]
        self.assertEqual([row["pair_id"] for row in nuisance],
                         ["nuisance_4", "nuisance_3", "nuisance_2", "nuisance_1"])

    def test_shared_change_groups_are_not_selected_twice(self):
        rows = diagnostic_rows()
        for layer in LAYERS:
            rows[layer]["consolidation_6_1"]["change_group_id"] = "consolidation_6_0"
        selected = select_cases(rows)
        self.assertEqual(len({row["change_group_id"] for row in selected}), len(selected))

    def test_selection_rejects_missing_active_pathology(self):
        rows = {layer: {"nuisance": dict(is_pathology="0", signed_reference="0")}
                for layer in LAYERS}
        with self.assertRaisesRegex(ValueError, "No pathology"):
            select_cases(rows)

    def test_checkpoint_guards(self):
        state = dict(run_config=dict(
            backbone_layer=6, feature_normalization="final_layernorm", no_cache=True),
            epoch=18, best_score=0.8)
        self.assertNotIn("backbone_layer", validate_checkpoint(
            state, dict(best_epoch=18, best_score=0.8), 6))
        for field, value in (("epoch", 19), ("best_score", 0.7)):
            changed = copy.deepcopy(state)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, "Checkpoint metadata mismatch"):
                validate_checkpoint(changed, dict(best_epoch=18, best_score=0.8), 6)
        with self.assertRaisesRegex(ValueError, "Checkpoint metadata mismatch"):
            validate_checkpoint(state, dict(best_epoch=18, best_score=0.8), 9)

    def write_diagnostics(self, folder, rows):
        summary = {str(layer): dict(best_epoch=1, best_score=0.8) for layer in LAYERS}
        (folder / "summary.json").write_text(json.dumps(summary))
        for layer in LAYERS:
            directory = folder / f"layer_{layer:02d}" / "diagnostics"
            directory.mkdir(parents=True)
            stem = directory / "epoch_0001_val"
            stem.with_suffix(".json").write_text(json.dumps(dict(
                epoch=1, invalid_pairs=0, primary_prediction_threshold=0.5,
                primary_gt_threshold=0.05, metadata=dict(backbone_layer=layer),
                groups=dict(overall=dict(pairs=len(rows[layer]))))))
            with gzip.open(str(stem) + ".csv.gz", "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows[layer].values()))))
                writer.writeheader()
                writer.writerows(rows[layer].values())

    def test_diagnostic_loading_and_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows = diagnostic_rows()
            self.write_diagnostics(folder, rows)
            _, loaded = load_diagnostics(folder)
            self.assertEqual(loaded, rows)
            rows[9]["consolidation_6_0"]["positive_reference"] = "2"
            path = folder / "layer_09" / "diagnostics" / "epoch_0001_val.csv.gz"
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows[9].values()))))
                writer.writeheader()
                writer.writerows(rows[9].values())
            with self.assertRaisesRegex(ValueError, "reference mismatch"):
                load_diagnostics(folder)

    def test_float32_maps_reproduce_scores_and_have_no_images(self):
        arrays = dict(gt_diff=np.array([[0.1, 0], [0, -0.1]], dtype=np.float32))
        for layer in LAYERS:
            arrays[f"pred_layer_{layer:02d}"] = np.array(
                [[0.75, 0], [0, -0.9]], dtype=np.float32)
        rows = {layer: {"pair": dict(
            positive_reference="1", negative_reference="1",
            signed_dice="1", predicted_area="0.5")} for layer in LAYERS}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            case = dict(pair_id="pair")
            save_case(output, case, arrays, rows, 2)
            self.assertEqual(case["sha256"], file_sha256(output / case["file"]))
            with np.load(output / case["file"], allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), set(arrays))
                for name, expected in arrays.items():
                    self.assertEqual(stored[name].dtype, np.float32)
                    np.testing.assert_array_equal(stored[name], expected)
            wrong = copy.deepcopy(rows)
            wrong[9]["pair"]["signed_dice"] = "0.5"
            with self.assertRaisesRegex(ValueError, "does not reproduce diagnostics"):
                save_case(output, dict(pair_id="pair"), arrays, wrong, 2)
            wrong = copy.deepcopy(rows)
            wrong[6]["pair"]["positive_reference"] = "0"
            with self.assertRaisesRegex(ValueError, "GT differs"):
                save_case(output, dict(pair_id="pair"), arrays, wrong, 2)
            invalid = dict(arrays)
            invalid["pred_layer_06"] = arrays["pred_layer_06"].astype(np.float16)
            with self.assertRaisesRegex(ValueError, "Invalid prediction maps"):
                save_case(output, dict(pair_id="pair"), invalid, rows, 2)

    def test_nuisance_exports_keep_dice_undefined(self):
        arrays = {key: np.zeros((2, 2), dtype=np.float32) for key in
                  ("gt_diff", "pred_layer_06", "pred_layer_09", "pred_layer_12")}
        rows = {layer: {"pair": dict(
            positive_reference="0", negative_reference="0",
            signed_dice="", predicted_area="0")} for layer in LAYERS}
        with tempfile.TemporaryDirectory() as temporary:
            case = dict(pair_id="pair")
            save_case(Path(temporary), case, arrays, rows, 2)
            self.assertEqual(case["export_signed_dice"], {str(layer): None for layer in LAYERS})

    def test_end_to_end_export_with_tiny_cpu_models(self):
        class TinyDataset:
            def __init__(self, root, split, img_size, rotation_k, flip_horizontal):
                if (split, img_size, rotation_k, flip_horizontal) != ("val", 2, 3, True):
                    raise ValueError("Wrong dataset preprocessing")
                self.records = [dict(pair_id="pathology"), dict(pair_id="nuisance")]

            def __len__(self):
                return len(self.records)

            def __getitem__(self, index):
                image, gt = torch.zeros(1, 2, 2), torch.zeros(1, 2, 2)
                if index == 0:
                    image[0, 0, 0], gt[0, 0, 0] = 0.75, 0.1
                return dict(img_prior=torch.zeros_like(image), img_curr=image,
                            gt_diff=gt, pair_id=self.records[index]["pair_id"])

        class TinyBackbone:
            def __init__(self, **kwargs):
                self.backbone = SimpleNamespace(config=SimpleNamespace(_commit_hash="test_revision"))

            def to(self, device):
                return self

            def eval(self):
                return self

            def forward_layers(self, image, layers):
                return {layer: dict(patch_tokens=image, cls_token=image) for layer in layers}

        class TinyHead(torch.nn.Module):
            def forward(self, prior, current, cls_prior, cls_current):
                return current, cls_current

        original_to = torch.Tensor.to

        def cpu_to(tensor, *args, **kwargs):
            if args and args[0] == "cuda":
                args = ("cpu", *args[1:])
            return original_to(tensor, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, output = base / "checkpoints", base / "export"
            run.mkdir()
            manifest = base / "manifest_split.jsonl"
            manifest.write_text("tiny fixture")
            rows = {layer: {} for layer in LAYERS}
            for layer in LAYERS:
                for pair, pathology in (("pathology", True), ("nuisance", False)):
                    rows[layer][pair] = dict(
                        pair_id=pair, change_group_id=pair, anomaly_type="consolidation",
                        is_pathology=str(int(pathology)), positive_reference=str(int(pathology)),
                        negative_reference="0", signed_reference=str(int(pathology)),
                        signed_dice="1" if pathology else "", predicted_area="0.25" if pathology else "0")
            self.write_diagnostics(run, rows)
            (run / "summary.json").write_text(json.dumps({
                str(layer): dict(best_epoch=1, best_score=1.0) for layer in LAYERS}))
            (run / "provenance.json").write_text(json.dumps(dict(
                code_sha256={"evaluation/export_layer_predictions.py": file_sha256(
                    ROOT / "evaluation" / "export_layer_predictions.py")},
                manifest_sha256=file_sha256(manifest), model_id="test", backbone_revision="test_revision")))
            for layer in LAYERS:
                torch.save(dict(
                    run_config=dict(backbone_layer=layer, feature_normalization="final_layernorm",
                                    no_cache=True, dataset_root=str(base), rotation_k=3, flip_horizontal=True),
                    epoch=1, best_score=1.0, head_spec={"type": "tiny"}, head={}),
                    run / f"layer_{layer:02d}" / "best.pt")
            with patch("constants.IMG_SIZE", 2), \
                    patch("datasets.LongitudinalPairDataset", TinyDataset), \
                    patch("models.FrozenCXRBackbone", TinyBackbone), \
                    patch("models.build_change_head", side_effect=lambda spec, device: TinyHead()), \
                    patch.object(torch.Tensor, "to", cpu_to), \
                    patch("torch.cuda.is_available", return_value=True), \
                    patch("torch.cuda.get_device_name", return_value="CPU fixture"):
                export_maps(run, output)
                with self.assertRaisesRegex(ValueError, "new export directory"):
                    export_maps(run, output)
            result = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(result["cases"]), 2)
            self.assertTrue(result["raw_unfiltered_scores"])
            self.assertEqual(result["orientation"],
                             "upright dataset coordinates; identical for GT and all predictions")
            self.assertEqual(set(path.suffix for path in output.iterdir()), {".json", ".npz"})
            for case in result["cases"]:
                self.assertEqual(case["sha256"], file_sha256(output / case["file"]))


if __name__ == "__main__":
    unittest.main()
