"""Observer variability: Model vs single-doctor annotations on PNIMIT pairs.

This script compares model outputs in `predictions_pnimit/pair_A*_*/output.nii.gz`
against one doctor's JSON annotations in `annotation tool/pair_A4_1_2/pair_A*_*/`.

Outputs:
- per_pair_metrics.csv
- summary_metrics.json

Usage (from python_files):
    python observer_variability_pnimit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.draw import ellipse
from skimage.transform import resize as sk_resize


STRUCT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def _norm_pair_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _pick_annotation_json(pair_ann_dir: Path) -> Path | None:
    jsons = sorted(pair_ann_dir.glob("*.json"))
    if not jsons:
        return None
    return jsons[0]


def _find_json_for_pair(annotations_root: Path, pair_name: str) -> Path | None:
    pair_dir = annotations_root / pair_name
    if pair_dir.is_dir():
        p = _pick_annotation_json(pair_dir)
        if p is not None:
            return p

    target_norm = _norm_pair_token(pair_name)
    pair_parts = pair_name.lower().split("_")
    if len(pair_parts) >= 4:
        compact_variant = f"pair{pair_parts[1]}{pair_parts[2]}{pair_parts[3]}"
    else:
        compact_variant = pair_name.lower().replace("_", "")

    for json_path in sorted(annotations_root.glob("*.json")):
        stem_norm = _norm_pair_token(json_path.stem)
        if stem_norm == target_norm or compact_variant in stem_norm:
            return json_path

    for json_path in sorted(annotations_root.rglob("*.json")):
        stem_norm = _norm_pair_token(json_path.stem)
        if stem_norm == target_norm or target_norm in stem_norm:
            return json_path

    return None


def _choose_current_nii(pair_ann_dir: Path, expected_current_name: str | None) -> Path:
    nii_files = sorted([p for p in pair_ann_dir.iterdir() if p.name.endswith(".nii.gz")])
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected >=2 .nii.gz files in {pair_ann_dir}")

    if expected_current_name:
        for p in nii_files:
            if _safe_stem(p) == expected_current_name:
                return p

    return nii_files[1]


def load_xray(file_path: Path) -> np.ndarray:
    return nib.load(str(file_path)).get_fdata()


def load_labels_map(json_path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    label_mapping_dict = {
        ("Appearance", None, None): 3,
        ("Disappearance", None, None): -3,
        ("Persistence", "Increase", "Increase"): 2,
        ("Persistence", "Decrease", "Decrease"): -2,
        ("Persistence", "Increase", "None"): 1,
        ("Persistence", "Decrease", "None"): -1,
        ("Persistence", "None", "Increase"): 1,
        ("Persistence", "None", "Decrease"): -1,
        ("Persistence", "None", "None"): 0,
        ("Persistence", "Increase", "Decrease"): (1, -1),
        ("Persistence", "Decrease", "Increase"): (1, -1),
    }

    labels_map_pos = np.zeros(shape, dtype=np.float32)
    labels_map_neg = np.zeros(shape, dtype=np.float32)

    with json_path.open("r", encoding="utf-8") as f:
        json_labels = json.load(f)

    for item in json_labels[1:]:
        if not isinstance(item, dict):
            continue
        rr, cc = ellipse(
            item["cx"],
            item["cy"],
            item["rx"],
            item["ry"],
            shape=shape,
            rotation=np.deg2rad(item.get("angle", 0.0)),
        )

        label_type = item.get("label")
        persistence_size_change = item.get("size_change") if label_type == "Persistence" else None
        persistence_intensity_change = item.get("intensity_change") if label_type == "Persistence" else None
        mapped = label_mapping_dict.get((label_type, persistence_size_change, persistence_intensity_change), 0)

        if mapped == 0:
            continue
        if isinstance(mapped, int):
            if mapped > 0:
                labels_map_pos[rr, cc] = mapped
            else:
                labels_map_neg[rr, cc] = mapped
        else:
            labels_map_pos[rr, cc] = mapped[0]
            labels_map_neg[rr, cc] = mapped[1]

    return labels_map_pos, labels_map_neg


def load_model_labels_map(nif_path: Path, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    labels = nib.load(str(nif_path)).get_fdata()
    if labels.ndim > 2:
        labels = np.squeeze(labels)

    if labels.shape != target_shape:
        labels = sk_resize(labels, target_shape, order=1, preserve_range=True, anti_aliasing=False)

    return (labels > 0).astype(np.uint8), (labels < 0).astype(np.uint8)


def get_pairwise_detections(label_map1: np.ndarray, label_map2: np.ndarray) -> tuple[int, int]:
    ccs1, _ = label(label_map1 != 0, STRUCT)
    ccs2, _ = label(label_map2 != 0, STRUCT)

    agreements = 0
    disagreements = 0
    vals1 = np.unique(ccs1)

    for v in vals1:
        if v == 0:
            continue

        inter_in_2 = ccs2 * (ccs1 == v)
        vals2_inter, counts2_inter = np.unique(inter_in_2, return_counts=True)

        if 0 in vals2_inter:
            idx = vals2_inter.tolist().index(0)
            vals2_inter = np.delete(vals2_inter, idx)
            counts2_inter = np.delete(counts2_inter, idx)

        if len(vals2_inter) == 0:
            disagreements += 1
            continue

        inter_val_2 = vals2_inter[np.argsort(counts2_inter)[::-1][0]]
        ccs2[ccs2 == inter_val_2] = 0
        agreements += 1

    disagreements += max(0, len(np.unique(ccs2)) - 1)
    return agreements, disagreements


def get_hmdr_udpp_counts(model_map: np.ndarray, human_map: np.ndarray) -> tuple[int, int, int]:
    # HMDR: Human-Matched Detection Rate for model predictions.
    # UDPP: Unmatched model detections per pair.
    ccs_model, total_preds = label(model_map != 0, STRUCT)
    inter_ccs_model = ccs_model * (human_map != 0)
    overlapping = max(0, len(np.unique(inter_ccs_model)) - 1)
    not_overlapping = int(total_preds - overlapping)
    return int(overlapping), int(not_overlapping), int(total_preds)


def get_human_detection_rate(model_map: np.ndarray, human_map: np.ndarray) -> tuple[int, int]:
    # Sensitivity-like measure: fraction of human connected components detected by model.
    ccs_human, total_human = label(human_map != 0, STRUCT)
    inter_human = ccs_human * (model_map != 0)
    detected = max(0, len(np.unique(inter_human)) - 1)
    return int(detected), int(total_human)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate(annotations_root: Path, preds_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_dirs = [p for p in sorted(preds_root.iterdir()) if p.is_dir() and p.name.lower().startswith("pair_")]

    total_ag_pos = 0
    total_dis_pos = 0
    total_ag_neg = 0
    total_dis_neg = 0

    total_model_preds_pos = 0
    total_model_preds_neg = 0
    total_overlapping_model_preds_pos = 0
    total_overlapping_model_preds_neg = 0
    not_overlapping_model_preds_pos: list[int] = []
    not_overlapping_model_preds_neg: list[int] = []

    total_human_detected_pos = 0
    total_human_total_pos = 0
    total_human_detected_neg = 0
    total_human_total_neg = 0

    effective_pairs = 0
    per_pair_rows: list[dict] = []

    for pred_pair_dir in pair_dirs:
        pair_name = pred_pair_dir.name
        model_nif = pred_pair_dir / "output.nii.gz"
        if not model_nif.exists():
            continue

        ann_json = _find_json_for_pair(annotations_root, pair_name)
        ann_pair_dir = annotations_root / pair_name
        if ann_json is None or not ann_pair_dir.exists():
            continue

        try:
            with ann_json.open("r", encoding="utf-8") as f:
                ann_data = json.load(f)
            current_name = None
            if ann_data and isinstance(ann_data[0], str) and "|" in ann_data[0]:
                parts = [p.strip() for p in ann_data[0].split("|")]
                if len(parts) >= 2:
                    current_name = parts[1]

            current_nii = _choose_current_nii(ann_pair_dir, current_name)
            current = load_xray(current_nii)

            human_pos, human_neg = load_labels_map(ann_json, current.shape)
            model_pos, model_neg = load_model_labels_map(model_nif, current.shape)

            ag_pos, dis_pos = get_pairwise_detections(model_pos, human_pos)
            ag_neg, dis_neg = get_pairwise_detections(model_neg, human_neg)

            pai_pos = _safe_div(2 * ag_pos, 2 * ag_pos + dis_pos)
            pai_neg = _safe_div(2 * ag_neg, 2 * ag_neg + dis_neg)

            ag_all = ag_pos + ag_neg
            dis_all = dis_pos + dis_neg
            pai_all = _safe_div(ag_all, ag_all + dis_all)
            if (ag_pos + dis_pos == 0 and ag_neg + dis_neg > 0) or (ag_pos + dis_pos > 0 and ag_neg + dis_neg == 0):
                pai_all = pai_all * 0.5 + 0.5

            ov_pos, nov_pos, total_pred_pos = get_hmdr_udpp_counts(model_pos, human_pos)
            ov_neg, nov_neg, total_pred_neg = get_hmdr_udpp_counts(model_neg, human_neg)

            hum_det_pos, hum_total_pos = get_human_detection_rate(model_pos, human_pos)
            hum_det_neg, hum_total_neg = get_human_detection_rate(model_neg, human_neg)

            per_pair_rows.append(
                {
                    "pair": pair_name,
                    "agreements_pos": ag_pos,
                    "disagreements_pos": dis_pos,
                    "pai_pos": pai_pos,
                    "agreements_neg": ag_neg,
                    "disagreements_neg": dis_neg,
                    "pai_neg": pai_neg,
                    "agreements_all": ag_all,
                    "disagreements_all": dis_all,
                    "pai_all": pai_all,
                    "model_preds_pos": total_pred_pos,
                    "model_preds_neg": total_pred_neg,
                    "model_overlapping_pos": ov_pos,
                    "model_overlapping_neg": ov_neg,
                    "model_not_overlapping_pos": nov_pos,
                    "model_not_overlapping_neg": nov_neg,
                    "human_detected_pos": hum_det_pos,
                    "human_total_pos": hum_total_pos,
                    "human_detected_neg": hum_det_neg,
                    "human_total_neg": hum_total_neg,
                }
            )

            total_ag_pos += ag_pos
            total_dis_pos += dis_pos
            total_ag_neg += ag_neg
            total_dis_neg += dis_neg

            total_model_preds_pos += total_pred_pos
            total_model_preds_neg += total_pred_neg
            total_overlapping_model_preds_pos += ov_pos
            total_overlapping_model_preds_neg += ov_neg
            not_overlapping_model_preds_pos.append(nov_pos)
            not_overlapping_model_preds_neg.append(nov_neg)

            total_human_detected_pos += hum_det_pos
            total_human_total_pos += hum_total_pos
            total_human_detected_neg += hum_det_neg
            total_human_total_neg += hum_total_neg

            effective_pairs += 1
        except Exception as exc:
            print(f"[SKIP] {pair_name}: {exc}")
            continue

    if effective_pairs == 0:
        raise RuntimeError("No PNIMIT pairs with both model outputs and doctor annotations were evaluable.")

    summary = {
        "effective_pairs": effective_pairs,
        "pairwise_agreement_per_detection": {
            "positive": _safe_div(2 * total_ag_pos, 2 * total_ag_pos + total_dis_pos),
            "negative": _safe_div(2 * total_ag_neg, 2 * total_ag_neg + total_dis_neg),
            "all": _safe_div(2 * (total_ag_pos + total_ag_neg), 2 * (total_ag_pos + total_ag_neg) + (total_dis_pos + total_dis_neg)),
        },
        "model_hmdr": {
            "positive": _safe_div(total_overlapping_model_preds_pos, total_model_preds_pos),
            "negative": _safe_div(total_overlapping_model_preds_neg, total_model_preds_neg),
        },
        "model_udpp": {
            "positive_mean": float(np.mean(not_overlapping_model_preds_pos)) if not_overlapping_model_preds_pos else 0.0,
            "negative_mean": float(np.mean(not_overlapping_model_preds_neg)) if not_overlapping_model_preds_neg else 0.0,
            "positive_std": float(np.std(not_overlapping_model_preds_pos)) if not_overlapping_model_preds_pos else 0.0,
            "negative_std": float(np.std(not_overlapping_model_preds_neg)) if not_overlapping_model_preds_neg else 0.0,
        },
        "model_vs_human_detection_rate": {
            "positive": _safe_div(total_human_detected_pos, total_human_total_pos),
            "negative": _safe_div(total_human_detected_neg, total_human_total_neg),
        },
        "totals": {
            "agreements_pos": total_ag_pos,
            "disagreements_pos": total_dis_pos,
            "agreements_neg": total_ag_neg,
            "disagreements_neg": total_dis_neg,
            "model_preds_pos": total_model_preds_pos,
            "model_preds_neg": total_model_preds_neg,
            "human_total_pos": total_human_total_pos,
            "human_total_neg": total_human_total_neg,
        },
    }

    csv_path = out_dir / "per_pair_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_pair_rows)

    json_path = out_dir / "summary_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved per-pair metrics: {csv_path}")
    print(f"Saved summary metrics: {json_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observer variability for PNIMIT: model vs one annotator.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotation tool") / "pair_A4_1_2",
        help="Folder containing pair_A*_*/ annotation JSON+NIfTI data.",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        default=Path("predictions_pnimit"),
        help="Folder containing pair_A*_*/output.nii.gz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("annotation tool") / "observer_variability_pnimit_model_vs_doctor",
        help="Output directory for CSV/JSON metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.annotations_root, args.preds_root, args.out_dir)
