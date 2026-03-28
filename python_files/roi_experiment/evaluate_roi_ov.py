"""Masked Observer Variability evaluation across ROI approaches.

For each ROI mask type, this script:
  1. Loads clinician annotation label maps (pos / neg)
  2. Loads model prediction label maps (pos / neg)
  3. Multiplies each by the ROI mask → keeps only detections inside the ROI
  4. Computes PAI, HMDR, UDPP for all observer pairs (4 doctors + model)
  5. Writes per-ROI results to JSON + a combined comparison CSV

Usage:
    python roi_experiment/evaluate_roi_ov.py \
        --roi-masks-dir Sahar_work/files/roi_masks \
        --model-preds-dir Sahar_work/files/predictions_1_100 \
        --annotations-dir "annotation tool/Annotations" \
        --pairs-roots "annotation tool/Pairs1" ... "annotation tool/Pairs8" \
        --out-dir Sahar_work/files/roi_experiment_results \
        --start-pair 1 --end-pair 100
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label
from skimage.draw import ellipse
from skimage.transform import resize as sk_resize

STRUCT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

ROI_NAMES = [
    "full_image",
    "lungs",
    "lungs_heart",
    "lungs_mediastinum",
    "full_thorax",
    "lungs_margin5",
    "lungs_med_margin5",
    "lungs_convex_hull",
]

PHYSICIANS = ["Avi", "Benny", "Sigal", "Smadar"]
ALL_OBSERVERS = PHYSICIANS + ["Model"]


# ── Loaders (reused from original OV script) ────────────────────────────────

def _extract_pair_num(name: str) -> int | None:
    m = re.search(r"\d+", name)
    return int(m.group()) if m else None


def resolve_annotation_path(ann_base: str, physician: str, pair_num: int) -> str:
    physician_dir = f"{ann_base}/{physician}"
    for root, _, files in os.walk(physician_dir):
        for fname in files:
            if fname.lower().endswith(".json") and _extract_pair_num(fname) == pair_num:
                return f"{root}/{fname}"
    raise FileNotFoundError(f"No annotation for {physician} pair {pair_num}")


def resolve_model_output(preds_base: str, pair_num: int) -> str:
    for prefix in ("pair", "Pair"):
        p = f"{preds_base}/{prefix}{pair_num}/output.nii.gz"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Model output not found for pair {pair_num}")


def resolve_pair_path(pairs_roots: list[str], pair_num: int) -> str:
    for root in pairs_roots:
        for prefix in ("pair", "Pair"):
            p = f"{root}/{prefix}{pair_num}"
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"Pair folder not found for pair {pair_num}")


def load_xray(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def load_labels_map(json_path: str, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    label_mapping = {
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
    pos = np.zeros(shape)
    neg = np.zeros(shape)
    with open(json_path) as f:
        data = json.load(f)
    for item in data[1:]:
        if not isinstance(item, dict):
            continue
        rr, cc = ellipse(
            item["cx"], item["cy"], item["rx"], item["ry"],
            shape=shape, rotation=np.deg2rad(item.get("angle", 0.0)),
        )
        lt = item.get("label")
        sc = item.get("size_change") if lt == "Persistence" else None
        ic = item.get("intensity_change") if lt == "Persistence" else None
        mapped = label_mapping.get((lt, sc, ic), 0)
        if mapped == 0:
            continue
        if isinstance(mapped, int):
            (pos if mapped > 0 else neg)[rr, cc] = mapped
        else:
            pos[rr, cc] = mapped[0]
            neg[rr, cc] = mapped[1]
    return pos, neg


def load_model_labels_map(nif_path: str) -> tuple[np.ndarray, np.ndarray]:
    arr = nib.load(nif_path).get_fdata()
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    arr = sk_resize(arr, (768, 768), order=1, preserve_range=True, anti_aliasing=False)
    return (arr > 0).astype(int), (arr < 0).astype(int)


def load_roi_mask(roi_dir: Path, roi_name: str, pair_num: int, target_shape: tuple) -> np.ndarray:
    if roi_name == "full_image":
        return np.ones(target_shape[:2], dtype=np.uint8)
    mask_path = roi_dir / roi_name / f"pair{pair_num}" / "mask.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {mask_path}")
    mask = nib.load(str(mask_path)).get_fdata()
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.shape != target_shape[:2]:
        mask = sk_resize(mask, target_shape[:2], order=0, preserve_range=True, anti_aliasing=False)
    return (mask > 0).astype(np.uint8)


# ── Metrics (from original OV script) ───────────────────────────────────────

def get_pairwise_detections(map1: np.ndarray, map2: np.ndarray) -> tuple[int, int]:
    ccs1, _ = label(map1 != 0, STRUCT)
    ccs2, _ = label(map2 != 0, STRUCT)
    agreements = 0
    disagreements = 0
    for v in np.unique(ccs1):
        if v == 0:
            continue
        inter = ccs2 * (ccs1 == v)
        vals, counts = np.unique(inter, return_counts=True)
        if 0 in vals:
            idx = vals.tolist().index(0)
            vals = np.delete(vals, idx)
            counts = np.delete(counts, idx)
        if len(vals) == 0:
            disagreements += 1
            continue
        best = vals[np.argsort(counts)[::-1][0]]
        ccs2[ccs2 == best] = 0
        agreements += 1
    disagreements += max(0, len(np.unique(ccs2)) - 1)
    return agreements, disagreements


def get_hmdr_udpp(model_map: np.ndarray, human_maps: list[np.ndarray]) -> tuple[int, int, int]:
    ccs_model, total = label(model_map, STRUCT)
    union = np.zeros_like(human_maps[0])
    for hm in human_maps:
        union[hm != 0] = 1
    inter = ccs_model * union
    overlapping = max(0, len(np.unique(inter)) - 1)
    return overlapping, total - overlapping, total


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


# ── Per-ROI evaluation ──────────────────────────────────────────────────────

def evaluate_single_roi(
    roi_name: str,
    roi_dir: Path,
    model_preds_dir: str,
    annotations_dir: str,
    pairs_roots: list[str],
    start_pair: int,
    end_pair: int,
) -> dict:
    obs_idx = {n: i for i, n in enumerate(ALL_OBSERVERS)}
    N = len(ALL_OBSERVERS)

    pai_ag = np.zeros((N, N))
    pai_dis = np.zeros((N, N))
    total_labels = {obs: {"pos": 0, "neg": 0} for obs in ALL_OBSERVERS}

    # HMDR / UDPP for model
    total_model_preds_pos = 0
    total_model_preds_neg = 0
    total_overlap_pos = 0
    total_overlap_neg = 0
    udpp_pos_list: list[int] = []
    udpp_neg_list: list[int] = []

    evaluated = 0

    for pair_num in range(start_pair, end_pair + 1):
        try:
            pair_path = resolve_pair_path(pairs_roots, pair_num)
            nii_files = sorted(
                [f for f in os.listdir(pair_path)
                 if f.endswith(".nii.gz") and "_seg" not in f and "_lung" not in f],
            )
            if len(nii_files) < 2:
                continue
            current = load_xray(f"{pair_path}/{nii_files[1]}")
            shape = current.shape

            roi_mask = load_roi_mask(roi_dir, roi_name, pair_num, shape)

            # Load annotations
            phy_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for phy in PHYSICIANS:
                ann_path = resolve_annotation_path(annotations_dir, phy, pair_num)
                p, n = load_labels_map(ann_path, shape)
                phy_maps[phy] = (p * roi_mask, n * roi_mask)

            # Load model
            model_path = resolve_model_output(model_preds_dir, pair_num)
            mp, mn = load_model_labels_map(model_path)
            # Resize ROI mask to model map shape if needed
            roi_768 = roi_mask
            if mp.shape != roi_mask.shape:
                roi_768 = sk_resize(roi_mask, mp.shape, order=0, preserve_range=True, anti_aliasing=False)
                roi_768 = (roi_768 > 0).astype(int)
            phy_maps["Model"] = (mp * roi_768, mn * roi_768)

        except (FileNotFoundError, Exception) as exc:
            print(f"[{roi_name}] SKIP pair {pair_num}: {exc}")
            continue

        evaluated += 1

        # Count labels
        for obs_name, (pm, nm) in phy_maps.items():
            _, n_pos = label(pm != 0, STRUCT)
            _, n_neg = label(nm != 0, STRUCT)
            total_labels[obs_name]["pos"] += n_pos
            total_labels[obs_name]["neg"] += n_neg

        # PAI: pairwise for all observer combinations (pos + neg combined)
        obs_list = [(name, phy_maps[name]) for name in ALL_OBSERVERS]
        for (n1, (p1, neg1)), (n2, (p2, neg2)) in combinations(obs_list, 2):
            ag_p, dis_p = get_pairwise_detections(p1, p2)
            ag_n, dis_n = get_pairwise_detections(neg1, neg2)
            pai_ag[obs_idx[n1], obs_idx[n2]] += 2 * (ag_p + ag_n)
            pai_ag[obs_idx[n2], obs_idx[n1]] += 2 * (ag_p + ag_n)
            pai_dis[obs_idx[n1], obs_idx[n2]] += dis_p + dis_n
            pai_dis[obs_idx[n2], obs_idx[n1]] += dis_p + dis_n

        # HMDR / UDPP for model vs human union
        human_pos = [phy_maps[p][0] for p in PHYSICIANS]
        human_neg = [phy_maps[p][1] for p in PHYSICIANS]
        ov_p, nov_p, tp_p = get_hmdr_udpp(phy_maps["Model"][0], human_pos)
        ov_n, nov_n, tp_n = get_hmdr_udpp(phy_maps["Model"][1], human_neg)
        total_overlap_pos += ov_p
        total_overlap_neg += ov_n
        total_model_preds_pos += tp_p
        total_model_preds_neg += tp_n
        udpp_pos_list.append(nov_p)
        udpp_neg_list.append(nov_n)

    # Build PAI matrix
    pai_mat = np.eye(N)
    for i, j in combinations(range(N), 2):
        denom = pai_ag[i, j] + pai_dis[i, j]
        val = safe_div(pai_ag[i, j], denom)
        pai_mat[i, j] = val
        pai_mat[j, i] = val

    # Human-only PAI average (exclude model)
    hh_vals = []
    for i, j in combinations(range(len(PHYSICIANS)), 2):
        hh_vals.append(pai_mat[i, j])
    hh_pai_mean = float(np.mean(hh_vals)) if hh_vals else 0.0

    # Model-Human PAI average
    mh_vals = [pai_mat[obs_idx["Model"], obs_idx[p]] for p in PHYSICIANS]
    mh_pai_mean = float(np.mean(mh_vals)) if mh_vals else 0.0

    result = {
        "roi": roi_name,
        "evaluated_pairs": evaluated,
        "pai_matrix": {ALL_OBSERVERS[i]: {ALL_OBSERVERS[j]: round(pai_mat[i, j], 4) for j in range(N)} for i in range(N)},
        "pai_human_human_mean": round(hh_pai_mean, 4),
        "pai_model_human_mean": round(mh_pai_mean, 4),
        "hmdr_pos": round(safe_div(total_overlap_pos, total_model_preds_pos), 4),
        "hmdr_neg": round(safe_div(total_overlap_neg, total_model_preds_neg), 4),
        "udpp_pos_mean": round(float(np.mean(udpp_pos_list)), 4) if udpp_pos_list else 0.0,
        "udpp_neg_mean": round(float(np.mean(udpp_neg_list)), 4) if udpp_neg_list else 0.0,
        "total_labels": total_labels,
    }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate observer variability per ROI approach.")
    p.add_argument("--roi-masks-dir", type=Path, required=True)
    p.add_argument("--model-preds-dir", type=str, required=True)
    p.add_argument("--annotations-dir", type=str, required=True)
    p.add_argument("--pairs-roots", nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start-pair", type=int, default=1)
    p.add_argument("--end-pair", type=int, default=100)
    p.add_argument("--roi-names", nargs="+", default=ROI_NAMES,
                   help="Which ROIs to evaluate (default: all 10)")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    for roi_name in args.roi_names:
        print(f"\n{'='*60}")
        print(f"Evaluating ROI: {roi_name}")
        print(f"{'='*60}")
        result = evaluate_single_roi(
            roi_name=roi_name,
            roi_dir=args.roi_masks_dir,
            model_preds_dir=args.model_preds_dir,
            annotations_dir=args.annotations_dir,
            pairs_roots=args.pairs_roots,
            start_pair=args.start_pair,
            end_pair=args.end_pair,
        )
        all_results.append(result)

        # Save per-ROI JSON
        roi_json = args.out_dir / f"{roi_name}_metrics.json"
        with roi_json.open("w") as f:
            json.dump(result, f, indent=4)
        print(f"  H-H PAI mean: {result['pai_human_human_mean']:.4f}")
        print(f"  M-H PAI mean: {result['pai_model_human_mean']:.4f}")
        print(f"  HMDR pos/neg: {result['hmdr_pos']:.4f} / {result['hmdr_neg']:.4f}")
        print(f"  UDPP pos/neg: {result['udpp_pos_mean']:.2f} / {result['udpp_neg_mean']:.2f}")

    # Save comparison CSV
    csv_path = args.out_dir / "roi_comparison.csv"
    fieldnames = ["roi", "evaluated_pairs", "pai_human_human_mean", "pai_model_human_mean",
                  "hmdr_pos", "hmdr_neg", "udpp_pos_mean", "udpp_neg_mean"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nComparison saved: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
