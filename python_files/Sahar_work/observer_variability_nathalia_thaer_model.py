from __future__ import annotations

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.ndimage import label
from skimage.draw import ellipse


STRUCT_3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
DIFF_CMAP = colors.LinearSegmentedColormap.from_list(
    "my_gradient",
    (
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000)),
    ),
)


def generate_alpha_map(x: np.ndarray) -> np.ndarray:
    x_abs = np.abs(x)
    max_val = max(float(np.max(x_abs)), 0.07)
    return x_abs / max_val


def load_current_scan(pair_dir: Path) -> np.ndarray:
    scans = sorted([p for p in pair_dir.glob("*.nii.gz") if "_seg" not in p.name])
    if len(scans) < 2:
        raise FileNotFoundError(f"Expected >=2 scans in {pair_dir}")
    arr = nib.load(str(scans[1])).get_fdata().T
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    return arr.astype(np.float32)


def resize_to_shape(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if arr.shape == (h, w):
        return arr
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    out = pil.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def load_labels_map(json_path: Path, shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    # Compatible with annotation-tool JSONs. If persistence change fields are absent,
    # persistence is treated as neutral and ignored (same behavior as 0 class).
    labels_map_pos = np.zeros(shape_hw, dtype=np.int16)
    labels_map_neg = np.zeros(shape_hw, dtype=np.int16)

    if not json_path.exists():
        return labels_map_pos, labels_map_neg

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        return labels_map_pos, labels_map_neg

    for l in data[1:]:
        if not isinstance(l, dict):
            continue

        try:
            cx = float(l.get("cx", 0.0))
            cy = float(l.get("cy", 0.0))
            rx = float(l.get("rx", 0.0))
            ry = float(l.get("ry", 0.0))
            ang = float(l.get("angle", 0.0))
        except Exception:
            continue

        rr, cc = ellipse(cy, cx, ry, rx, shape=shape_hw, rotation=np.deg2rad(ang))

        label_type = str(l.get("label", ""))
        size_change = l.get("size_change", None)
        intensity_change = l.get("intensity_change", None)

        if label_type == "Appearance":
            labels_map_pos[rr, cc] = 3
        elif label_type == "Disappearance":
            labels_map_neg[rr, cc] = -3
        elif label_type == "Persistence":
            # Same mapping logic as ICU OV when fields exist.
            if size_change == "Increase" and intensity_change == "Increase":
                labels_map_pos[rr, cc] = 2
            elif size_change == "Decrease" and intensity_change == "Decrease":
                labels_map_neg[rr, cc] = -2
            elif (size_change == "Increase" and intensity_change == "None") or (
                size_change == "None" and intensity_change == "Increase"
            ):
                labels_map_pos[rr, cc] = 1
            elif (size_change == "Decrease" and intensity_change == "None") or (
                size_change == "None" and intensity_change == "Decrease"
            ):
                labels_map_neg[rr, cc] = -1
            elif (size_change == "Increase" and intensity_change == "Decrease") or (
                size_change == "Decrease" and intensity_change == "Increase"
            ):
                labels_map_pos[rr, cc] = 1
                labels_map_neg[rr, cc] = -1
            else:
                # No change or unknown schema => neutral
                pass

    return labels_map_pos, labels_map_neg


def load_model_labels_map(
    model_path: Path,
    shape_hw: tuple[int, int],
    min_cc_size: int = 50,
    min_cc_intensity: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_output = nib.load(str(model_path)).get_fdata().T.astype(np.float32)
    if model_output.ndim == 3:
        model_output = model_output[:, :, model_output.shape[2] // 2]

    model_output = resize_to_shape(model_output, shape_hw)

    pos_map = (model_output > 0).astype(np.uint8)
    neg_map = (model_output < 0).astype(np.uint8)

    for binary_map in (pos_map, neg_map):
        ccs, num_ccs = label(binary_map, STRUCT_3x3)
        for cc_val in range(1, num_ccs + 1):
            cc_mask = ccs == cc_val
            cc_size = int(np.sum(cc_mask))
            cc_mean_intensity = float(np.abs(model_output[cc_mask]).mean()) if cc_size > 0 else 0.0
            if cc_size < min_cc_size or cc_mean_intensity < min_cc_intensity:
                binary_map[cc_mask] = 0

    return pos_map, neg_map, model_output


def plot_diff_on_current(diff_map: np.ndarray, current: np.ndarray, out_path: Path) -> None:
    alphas = generate_alpha_map(diff_map)
    divnorm = colors.TwoSlopeNorm(
        vmin=min(float(np.min(diff_map)), -0.01),
        vcenter=0.0,
        vmax=max(float(np.max(diff_map)), 0.01),
    )
    fig, ax = plt.subplots()
    ax.imshow(current, cmap="gray")
    imm = ax.imshow(diff_map, alpha=alphas, cmap=DIFF_CMAP, norm=divnorm)
    plt.colorbar(imm, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pair_overlay(
    current: np.ndarray,
    full_model_output: np.ndarray,
    human_pos_union: np.ndarray,
    human_neg_union: np.ndarray,
    pair_name: str,
    plots_dir: Path,
) -> None:
    pair_plots_dir = plots_dir / pair_name
    pair_plots_dir.mkdir(parents=True, exist_ok=True)

    combined_annot = human_pos_union.copy()
    combined_annot[human_neg_union != 0] = human_neg_union[human_neg_union != 0]

    plot_diff_on_current(full_model_output, current, pair_plots_dir / "model_on_original.png")
    plot_diff_on_current(combined_annot, current, pair_plots_dir / "annotation_on_original.png")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes[0].imshow(current, cmap="gray")
    axes[0].set_title("Original Scan", fontsize=14, fontweight="bold")
    axes[0].set_axis_off()

    axes[1].imshow(current, cmap="gray")
    axes[1].imshow(
        full_model_output,
        alpha=generate_alpha_map(full_model_output),
        cmap=DIFF_CMAP,
        norm=colors.TwoSlopeNorm(
            vmin=min(float(np.min(full_model_output)), -0.01),
            vcenter=0.0,
            vmax=max(float(np.max(full_model_output)), 0.01),
        ),
    )
    axes[1].set_title("Model Output", fontsize=14, fontweight="bold")
    axes[1].set_axis_off()

    axes[2].imshow(current, cmap="gray")
    axes[2].imshow(
        combined_annot,
        alpha=generate_alpha_map(combined_annot),
        cmap=DIFF_CMAP,
        norm=colors.TwoSlopeNorm(
            vmin=min(float(np.min(combined_annot)), -0.01),
            vcenter=0.0,
            vmax=max(float(np.max(combined_annot)), 0.01),
        ),
    )
    axes[2].set_title("Annotation", fontsize=14, fontweight="bold")
    axes[2].set_axis_off()

    fig.suptitle(pair_name, fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.savefig(pair_plots_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_pairwise_detections(label_map1: np.ndarray, label_map2: np.ndarray) -> tuple[int, int]:
    ccs1, _ = label(label_map1 != 0, STRUCT_3x3)
    ccs2, _ = label(label_map2 != 0, STRUCT_3x3)

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
    return int(agreements), int(disagreements)


def get_hmdr_udpp_counts(model_map: np.ndarray, human_maps: list[np.ndarray]) -> tuple[int, int, int]:
    ccs_model, total_preds = label(model_map != 0, STRUCT_3x3)
    human_union_map = np.zeros_like(human_maps[0], dtype=np.uint8)
    for hm in human_maps:
        human_union_map[hm != 0] = 1
    inter_ccs_model = ccs_model * (human_union_map == 1)
    overlapping = max(0, len(np.unique(inter_ccs_model)) - 1)
    not_overlapping = int(total_preds - overlapping)
    return int(overlapping), int(not_overlapping), int(total_preds)


def get_sensitivity_at_consensus_levels(model_map: np.ndarray, human_maps: list[np.ndarray]) -> list[tuple[int, int]]:
    num_humans = len(human_maps)
    hm_bin = [(hm != 0).astype(np.uint8)[None, ...] for hm in human_maps]
    model_bin = model_map != 0

    sum_map = np.sum(np.concatenate(hm_bin, axis=0), axis=0).squeeze()
    hm_ccs = [label(hm.squeeze(), STRUCT_3x3)[0] for hm in hm_bin]

    consensus_map = np.zeros_like(sum_map)
    for human_map in hm_ccs:
        c_consensus_map = np.zeros_like(human_map)
        for v in np.unique(human_map):
            if v == 0:
                continue
            inter_val = sum_map * (human_map == v)
            val_level = int(np.max(inter_val))
            c_consensus_map[human_map == v] = val_level
        consensus_map = np.maximum(consensus_map, c_consensus_map)

    sens = []
    for i in range(num_humans):
        level = i + 1
        consensus_level_map = consensus_map >= level
        consensus_ccs, consensus_ccs_num = label(consensus_level_map, STRUCT_3x3)
        model_inter_map = consensus_ccs * model_bin
        model_detections_num = max(0, len(np.unique(model_inter_map)) - 1)
        sens.append((int(model_detections_num), int(consensus_ccs_num)))
    return sens


def evaluate_recall_fn_spec(obs_map: np.ndarray, ref_maps: list[np.ndarray]) -> dict:
    """Per-pair building blocks for recall / false-negative-pairs / specificity (one sign).

    Reference consensus is the union (level >= 1) of the reference observers' findings.
    Returns raw counts for a single pair; the caller aggregates across pairs and derives:
      - recall            = detected / ref_total
      - false-negative pair = (ref_total > 0 and detected == 0)
      - consensus-negative pair = (ref_total == 0); true-negative if also obs_count == 0
    """
    if not ref_maps:
        return {"ref_total": 0, "detected": 0, "obs_count": 0}
    detected, ref_total = get_sensitivity_at_consensus_levels(obs_map, ref_maps)[0]
    _, obs_count = label(obs_map != 0, STRUCT_3x3)
    return {"ref_total": int(ref_total), "detected": int(detected), "obs_count": int(obs_count)}


def plot_matrix(matrix_df: pd.DataFrame, output_path: Path, title: str) -> None:
    n = matrix_df.shape[0]
    figsize = max(6, int(n * 2.0))
    plt.figure(figsize=(figsize, figsize))
    ax = sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.25,
        linewidths=0.0,
        linecolor="black",
        cbar_kws={"shrink": 1.0, "aspect": 20, "label": "PAI"},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    separator_index = n - 1
    ax.axvline(x=separator_index, ymin=1 / n, color="black", linewidth=1.25, linestyle="--")
    ax.axhline(y=separator_index, xmax=1 - 1 / n, color="black", linewidth=1.25, linestyle="--")

    plt.xticks(rotation=45, ha="right", fontsize=11, fontweight="bold")
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_curves(pos_arr: list[float], neg_arr: list[float], name: str, out_path: Path) -> None:
    x_values = np.arange(1, len(pos_arr) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.plot(x_values, pos_arr, label="Positive changes", color="#003366", linestyle="-", linewidth=2.0, marker="o", markersize=6)
    ax.plot(x_values, neg_arr, label="Negative changes", color="#800000", linestyle="--", linewidth=2.0, marker="s", markersize=6)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_values)
    ax.set_xlim(0.5, len(pos_arr) + 0.5)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"Sensitivity at Consensus Levels ({name})", pad=12)
    ax.grid(True, which="major", linestyle=":", alpha=0.6, color="gray")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def run(args: argparse.Namespace) -> int:
    observers = ["Nathalia", "Thaer", "Model"]
    humans = ["Nathalia", "Thaer"]
    idx = {n: i for i, n in enumerate(observers)}

    out_dir: Path = args.out_dir
    plots_dir: Path | None = None if args.disable_plots else args.plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    n_dir = args.annotations_json_root / "Nathalia"
    t_dir = args.annotations_json_root / "Thaer"

    n_pairs = {p.stem.replace("Nathalia_", "", 1): p for p in sorted(n_dir.glob("Nathalia_*.json"))}
    t_pairs = {p.stem.replace("Thaer_", "", 1): p for p in sorted(t_dir.glob("Thaer_*.json"))}
    all_pairs = sorted(set(n_pairs) | set(t_pairs))

    if args.max_pairs > 0:
        all_pairs = all_pairs[: args.max_pairs]

    num_obs = len(observers)
    num_humans = len(humans)

    sens_model_pos = [[0, 0] for _ in range(num_humans)]
    sens_model_neg = [[0, 0] for _ in range(num_humans)]

    # Each human against the other human (1 consensus level only when only one other human).
    sens_human_pos = {h: [[0, 0] for _ in range(max(1, num_humans - 1))] for h in humans}
    sens_human_neg = {h: [[0, 0] for _ in range(max(1, num_humans - 1))] for h in humans}

    pairwise_agreement_mat_pos = np.eye(num_obs, dtype=np.float64)
    pairwise_agreement_mat_neg = np.eye(num_obs, dtype=np.float64)
    pairwise_disagreement_mat_pos = np.zeros((num_obs, num_obs), dtype=np.float64)
    pairwise_disagreement_mat_neg = np.zeros((num_obs, num_obs), dtype=np.float64)

    pairwise_agreement_per_pair_mat_pos = np.eye(num_obs, dtype=np.float64)
    pairwise_agreement_per_pair_mat_neg = np.eye(num_obs, dtype=np.float64)

    agreements_num_list_pos = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    agreements_num_list_neg = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    disagreements_num_list_pos = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    disagreements_num_list_neg = [[[] for _ in range(num_obs)] for __ in range(num_obs)]

    total_labels_pos = {obs: [] for obs in observers}
    total_labels_neg = {obs: [] for obs in observers}

    total_preds_pos = {obs: 0 for obs in observers}
    total_preds_neg = {obs: 0 for obs in observers}
    total_overlapping_pos = {obs: 0 for obs in observers}
    total_overlapping_neg = {obs: 0 for obs in observers}
    not_overlapping_pos = {obs: [] for obs in observers}
    not_overlapping_neg = {obs: [] for obs in observers}

    pairs_processed = 0

    for pair in all_pairs:
        pair_dir = args.pairs_root / pair
        model_path = args.model_preds_root / pair / "output.nii.gz"

        if not pair_dir.exists() or not model_path.exists():
            continue

        try:
            current = load_current_scan(pair_dir)
        except Exception:
            continue

        shape_hw = tuple(current.shape)

        n_pos, n_neg = load_labels_map(n_pairs.get(pair, Path("")), shape_hw)
        t_pos, t_neg = load_labels_map(t_pairs.get(pair, Path("")), shape_hw)
        try:
            m_pos, m_neg, full_model_output = load_model_labels_map(
                model_path,
                shape_hw,
                min_cc_size=args.min_cc_size,
                min_cc_intensity=args.min_cc_intensity,
            )
        except Exception:
            continue

        pos_maps = {"Nathalia": n_pos, "Thaer": t_pos, "Model": m_pos}
        neg_maps = {"Nathalia": n_neg, "Thaer": t_neg, "Model": m_neg}

        if plots_dir is not None:
            human_pos_union = np.where((n_pos != 0) | (t_pos != 0), 1, 0).astype(np.float32)
            human_neg_union = np.where((n_neg != 0) | (t_neg != 0), -1, 0).astype(np.float32)
            plot_pair_overlay(current, full_model_output, human_pos_union, human_neg_union, pair, plots_dir)

        # Sensitivity: model against both humans.
        cur_sens_pos = get_sensitivity_at_consensus_levels(m_pos, [n_pos, t_pos])
        cur_sens_neg = get_sensitivity_at_consensus_levels(m_neg, [n_neg, t_neg])
        for k, (d, c) in enumerate(cur_sens_pos):
            sens_model_pos[k][0] += d
            sens_model_pos[k][1] += c
        for k, (d, c) in enumerate(cur_sens_neg):
            sens_model_neg[k][0] += d
            sens_model_neg[k][1] += c

        # Sensitivity: each human against the other one.
        sens_n_pos = get_sensitivity_at_consensus_levels(n_pos, [t_pos])
        sens_n_neg = get_sensitivity_at_consensus_levels(n_neg, [t_neg])
        sens_t_pos = get_sensitivity_at_consensus_levels(t_pos, [n_pos])
        sens_t_neg = get_sensitivity_at_consensus_levels(t_neg, [n_neg])
        for k, (d, c) in enumerate(sens_n_pos):
            sens_human_pos["Nathalia"][k][0] += d
            sens_human_pos["Nathalia"][k][1] += c
        for k, (d, c) in enumerate(sens_n_neg):
            sens_human_neg["Nathalia"][k][0] += d
            sens_human_neg["Nathalia"][k][1] += c
        for k, (d, c) in enumerate(sens_t_pos):
            sens_human_pos["Thaer"][k][0] += d
            sens_human_pos["Thaer"][k][1] += c
        for k, (d, c) in enumerate(sens_t_neg):
            sens_human_neg["Thaer"][k][0] += d
            sens_human_neg["Thaer"][k][1] += c

        # HMDR / UDPP for all observers against human union of the other humans.
        ov, nov, tp = get_hmdr_udpp_counts(m_pos, [n_pos, t_pos])
        total_preds_pos["Model"] += tp
        total_overlapping_pos["Model"] += ov
        not_overlapping_pos["Model"].append(nov)
        ov, nov, tp = get_hmdr_udpp_counts(m_neg, [n_neg, t_neg])
        total_preds_neg["Model"] += tp
        total_overlapping_neg["Model"] += ov
        not_overlapping_neg["Model"].append(nov)

        ov, nov, tp = get_hmdr_udpp_counts(n_pos, [t_pos])
        total_preds_pos["Nathalia"] += tp
        total_overlapping_pos["Nathalia"] += ov
        not_overlapping_pos["Nathalia"].append(nov)
        ov, nov, tp = get_hmdr_udpp_counts(n_neg, [t_neg])
        total_preds_neg["Nathalia"] += tp
        total_overlapping_neg["Nathalia"] += ov
        not_overlapping_neg["Nathalia"].append(nov)

        ov, nov, tp = get_hmdr_udpp_counts(t_pos, [n_pos])
        total_preds_pos["Thaer"] += tp
        total_overlapping_pos["Thaer"] += ov
        not_overlapping_pos["Thaer"].append(nov)
        ov, nov, tp = get_hmdr_udpp_counts(t_neg, [n_neg])
        total_preds_neg["Thaer"] += tp
        total_overlapping_neg["Thaer"] += ov
        not_overlapping_neg["Thaer"].append(nov)

        # PAI counts.
        for maps, ag_mat, dis_mat, ag_pp_mat, ag_lists, dis_lists, totals in [
            (pos_maps, pairwise_agreement_mat_pos, pairwise_disagreement_mat_pos, pairwise_agreement_per_pair_mat_pos, agreements_num_list_pos, disagreements_num_list_pos, total_labels_pos),
            (neg_maps, pairwise_agreement_mat_neg, pairwise_disagreement_mat_neg, pairwise_agreement_per_pair_mat_neg, agreements_num_list_neg, disagreements_num_list_neg, total_labels_neg),
        ]:
            for n1, n2 in combinations(observers, 2):
                a, d = get_pairwise_detections(maps[n1], maps[n2])
                i1, i2 = idx[n1], idx[n2]
                ag_mat[i1, i2] += 2 * a
                ag_mat[i2, i1] += 2 * a
                dis_mat[i1, i2] += d
                dis_mat[i2, i1] += d
                c_pai = np.nan_to_num(2 * a / np.array(2 * a + d), nan=1.0, posinf=1.0, neginf=1.0)
                ag_pp_mat[i1, i2] += c_pai
                ag_pp_mat[i2, i1] += c_pai
                ag_lists[i1][i2].append(2 * a)
                ag_lists[i2][i1].append(2 * a)
                dis_lists[i1][i2].append(d)
                dis_lists[i2][i1].append(d)

            for obs in observers:
                _, ccs_num = label(maps[obs] != 0, STRUCT_3x3)
                totals[obs].append(int(ccs_num))

        pairs_processed += 1

    if pairs_processed == 0:
        raise RuntimeError("No pairs were processed. Check paths and file availability.")

    # Normalize per-pair PAI accumulators by actual pairs processed.
    pairwise_agreement_per_pair_mat_pos /= pairs_processed
    pairwise_agreement_per_pair_mat_neg /= pairs_processed

    # Final sensitivity outputs.
    sens_dict = {
        f"Sensitivity Consensus Level {i + 1} (Positive)": safe_div(s[0], s[1])
        for i, s in enumerate(sens_model_pos)
    }
    sens_dict.update(
        {
            f"Sensitivity Consensus Level {i + 1} (Negative)": safe_div(s[0], s[1])
            for i, s in enumerate(sens_model_neg)
        }
    )
    sens_dict.update(
        {
            f"Total detections & changes at Consensus Level {i + 1} (Positive)": (int(s[0]), int(s[1]))
            for i, s in enumerate(sens_model_pos)
        }
    )
    sens_dict.update(
        {
            f"Total detections & changes at Consensus Level {i + 1} (Negative)": (int(s[0]), int(s[1]))
            for i, s in enumerate(sens_model_neg)
        }
    )

    (out_dir / "sensitivity_measures.json").write_text(json.dumps(sens_dict, indent=4), encoding="utf-8")

    plot_curves(
        [safe_div(s[0], s[1]) for s in sens_model_pos],
        [safe_div(s[0], s[1]) for s in sens_model_neg],
        "Model",
        out_dir / "sensitivity_consensus_levels_model.png",
    )
    for h in humans:
        plot_curves(
            [safe_div(s[0], s[1]) for s in sens_human_pos[h]],
            [safe_div(s[0], s[1]) for s in sens_human_neg[h]],
            h,
            out_dir / f"sensitivity_consensus_levels_{h.lower()}.png",
        )

    # HMDR / UDPP outputs.
    precision = {}
    for obs in observers:
        precision[f"{obs} HMDR (Positive)"] = safe_div(total_overlapping_pos[obs], total_preds_pos[obs])
        precision[f"{obs} HMDR (Negative)"] = safe_div(total_overlapping_neg[obs], total_preds_neg[obs])
        precision[f"UDPP {obs} (Positive)"] = safe_div(sum(not_overlapping_pos[obs]), pairs_processed)
        precision[f"UDPP {obs} (Negative)"] = safe_div(sum(not_overlapping_neg[obs]), pairs_processed)
        precision[f"UDPP STD {obs} (Positive)"] = float(np.std(not_overlapping_pos[obs])) if not_overlapping_pos[obs] else 0.0
        precision[f"UDPP STD {obs} (Negative)"] = float(np.std(not_overlapping_neg[obs])) if not_overlapping_neg[obs] else 0.0

    (out_dir / "precision_measures.json").write_text(json.dumps(precision, indent=4), encoding="utf-8")

    # PAI matrices.
    mat_per_label_pos = pairwise_agreement_mat_pos / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos)
    mat_per_label_neg = pairwise_agreement_mat_neg / (pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_label_all = (pairwise_agreement_mat_pos + pairwise_agreement_mat_neg) / (
        pairwise_agreement_mat_pos
        + pairwise_disagreement_mat_pos
        + pairwise_agreement_mat_neg
        + pairwise_disagreement_mat_neg
    )

    mat_per_pair_pos = pairwise_agreement_per_pair_mat_pos
    mat_per_pair_neg = pairwise_agreement_per_pair_mat_neg

    mat_per_pair_all = np.eye(num_obs, dtype=np.float64)
    for n1, n2 in combinations(observers, 2):
        i1, i2 = idx[n1], idx[n2]
        pai_per_pair = 0.0
        for k in range(len(agreements_num_list_pos[i1][i2])):
            ag_pos = agreements_num_list_pos[i1][i2][k]
            ag_neg = agreements_num_list_neg[i1][i2][k]
            ag_all = ag_pos + ag_neg
            dis_pos = disagreements_num_list_pos[i1][i2][k]
            dis_neg = disagreements_num_list_neg[i1][i2][k]
            dis_all = dis_pos + dis_neg
            c_pai = np.nan_to_num(ag_all / np.array(ag_all + dis_all), nan=1.0, posinf=1.0, neginf=1.0)
            if (ag_pos + dis_pos == 0 and ag_neg + dis_neg > 0) or (ag_pos + dis_pos > 0 and ag_neg + dis_neg == 0):
                c_pai = c_pai * 0.5 + 0.5
            pai_per_pair += float(c_pai)
        mat_per_pair_all[i1, i2] = pai_per_pair / pairs_processed
        mat_per_pair_all[i2, i1] = pai_per_pair / pairs_processed

    per_label_df_pos = pd.DataFrame(mat_per_label_pos, index=observers, columns=observers)
    per_label_df_neg = pd.DataFrame(mat_per_label_neg, index=observers, columns=observers)
    per_label_df_all = pd.DataFrame(mat_per_label_all, index=observers, columns=observers)
    per_pair_df_pos = pd.DataFrame(mat_per_pair_pos, index=observers, columns=observers)
    per_pair_df_neg = pd.DataFrame(mat_per_pair_neg, index=observers, columns=observers)
    per_pair_df_all = pd.DataFrame(mat_per_pair_all, index=observers, columns=observers)

    plot_matrix(per_label_df_pos, out_dir / "per_label_agreement_pos.png", "Pairwise Agreement Index Per Detection (positive)")
    plot_matrix(per_label_df_neg, out_dir / "per_label_agreement_neg.png", "Pairwise Agreement Index Per Detection (negative)")
    plot_matrix(per_label_df_all, out_dir / "per_label_agreement_all.png", "Pairwise Agreement Index Per Detection (all)")
    plot_matrix(per_pair_df_pos, out_dir / "per_pair_agreement_pos.png", "Pairwise Agreement Index Per Pair (positive)")
    plot_matrix(per_pair_df_neg, out_dir / "per_pair_agreement_neg.png", "Pairwise Agreement Index Per Pair (negative)")
    plot_matrix(per_pair_df_all, out_dir / "per_pair_agreement_all.png", "Pairwise Agreement Index Per Pair (all)")

    # Total label statistics.
    def stats_dict(arr_list: list[int]) -> tuple[int, float, float, int, int]:
        arr = np.array(arr_list, dtype=np.int32)
        return (
            int(np.sum(arr)),
            float(np.mean(arr)) if arr.size else 0.0,
            float(np.std(arr)) if arr.size else 0.0,
            int(np.max(arr)) if arr.size else 0,
            int(np.min(arr)) if arr.size else 0,
        )

    total_labels_data_pos = {obs: stats_dict(total_labels_pos[obs]) for obs in observers}
    total_labels_data_neg = {obs: stats_dict(total_labels_neg[obs]) for obs in observers}
    total_labels_data_all = {
        obs: stats_dict([total_labels_pos[obs][k] + total_labels_neg[obs][k] for k in range(len(total_labels_pos[obs]))])
        for obs in observers
    }

    (out_dir / "total_labels_marked_pos.json").write_text(json.dumps(total_labels_data_pos, indent=4), encoding="utf-8")
    (out_dir / "total_labels_marked_neg.json").write_text(json.dumps(total_labels_data_neg, indent=4), encoding="utf-8")
    (out_dir / "total_labels_marked_all.json").write_text(json.dumps(total_labels_data_all, indent=4), encoding="utf-8")

    summary = {
        "pairs_processed": pairs_processed,
        "observers": observers,
        "annotations_root": str(args.annotations_json_root),
        "model_preds_root": str(args.model_preds_root),
        "pairs_root": str(args.pairs_root),
        "min_cc_size": int(args.min_cc_size),
        "min_cc_intensity": float(args.min_cc_intensity),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {out_dir}")
    if plots_dir is not None:
        print(f"Overlay plots saved to: {plots_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observer Variability for Nathalia/Thaer/Model (ICU OV-style outputs)."
    )
    parser.add_argument(
        "--annotations-json-root",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named",
        help="Root containing Nathalia/ and Thaer/ JSON folders.",
    )
    parser.add_argument(
        "--model-preds-root",
        type=Path,
        default=Path("python_files") / "annotation tool" / "predictions_pnimit_lung",
        help="Root containing per-pair model outputs (output.nii.gz).",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Pairs_PNIMIT_1_pairs",
        help="Root containing pair folders with scans.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named" / "ov_sq_style_nathalia_thaer_model",
        help="Output directory for all OV results.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named" / "ov_sq_style_nathalia_thaer_model" / "pair_overlays",
        help="Directory for per-pair overlay plots.",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Skip per-pair overlay image generation (faster).",
    )
    parser.add_argument("--min-cc-size", type=int, default=50)
    parser.add_argument("--min-cc-intensity", type=float, default=0.03)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all available pairs.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
