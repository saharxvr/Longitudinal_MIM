from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_disagreement_levels as gdl
import observer_variability_main_loo_metrics as loo
import observer_variability_nathalia_thaer_model as base


HUMANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]
GENERATED_OBSERVERS = [
    "ICU Model",
    "Claude Ellipses",
    "Claude Heatmap",
    "Claude Precision",
]
OBSERVERS = [*HUMANS, *GENERATED_OBSERVERS]


def load_full_prediction(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    output = nib.load(str(path)).get_fdata().astype(np.float32)
    if output.ndim == 3:
        output = output[..., output.shape[2] // 2]
    if output.shape != shape:
        raise ValueError(f"Prediction shape {output.shape} does not match current scan {shape}: {path}")
    return (output > 0).astype(np.uint8), (output < 0).astype(np.uint8)


def pairwise_matrices(
    maps_by_pair: list[dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, np.ndarray]:
    observer_index = {observer: index for index, observer in enumerate(OBSERVERS)}
    size = len(OBSERVERS)
    agreements = {"pos": np.zeros((size, size)), "neg": np.zeros((size, size))}
    disagreements = {"pos": np.zeros((size, size)), "neg": np.zeros((size, size))}
    per_pair = {
        "pos": np.zeros((size, size)),
        "neg": np.zeros((size, size)),
        "all": np.zeros((size, size)),
    }

    for maps in maps_by_pair:
        for left, right in combinations(OBSERVERS, 2):
            left_index = observer_index[left]
            right_index = observer_index[right]
            sign_results = {}
            for sign, map_index in (("pos", 0), ("neg", 1)):
                agreement, disagreement = base.get_pairwise_detections(
                    maps[left][map_index],
                    maps[right][map_index],
                )
                sign_results[sign] = (agreement, disagreement)
                for row, column in ((left_index, right_index), (right_index, left_index)):
                    agreements[sign][row, column] += 2 * agreement
                    disagreements[sign][row, column] += disagreement
                    denominator = 2 * agreement + disagreement
                    per_pair[sign][row, column] += 2 * agreement / denominator if denominator else 1.0

            positive_agreement, positive_disagreement = sign_results["pos"]
            negative_agreement, negative_disagreement = sign_results["neg"]
            all_agreement = 2 * (positive_agreement + negative_agreement)
            all_disagreement = positive_disagreement + negative_disagreement
            denominator = all_agreement + all_disagreement
            score = all_agreement / denominator if denominator else 1.0
            positive_empty = 2 * positive_agreement + positive_disagreement == 0
            negative_empty = 2 * negative_agreement + negative_disagreement == 0
            if positive_empty != negative_empty:
                score = score * 0.5 + 0.5
            per_pair["all"][left_index, right_index] += score
            per_pair["all"][right_index, left_index] += score

    matrices = {}
    pair_count = len(maps_by_pair)
    for sign in ("pos", "neg"):
        denominator = agreements[sign] + disagreements[sign]
        matrices[f"per_detection_{sign}"] = agreements[sign] / np.where(denominator == 0, 1, denominator)
        matrices[f"per_pair_{sign}"] = per_pair[sign] / pair_count
    all_agreement = agreements["pos"] + agreements["neg"]
    all_denominator = (
        agreements["pos"]
        + disagreements["pos"]
        + agreements["neg"]
        + disagreements["neg"]
    )
    matrices["per_detection_all"] = all_agreement / np.where(all_denominator == 0, 1, all_denominator)
    matrices["per_pair_all"] = per_pair["all"] / pair_count
    for matrix in matrices.values():
        np.fill_diagonal(matrix, 1.0)
    return matrices


def plot_matrix(matrix: np.ndarray, output_path: Path, title: str) -> None:
    frame = pd.DataFrame(matrix, index=OBSERVERS, columns=OBSERVERS)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    sns.heatmap(
        frame,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Pairwise Agreement Index"},
        ax=ax,
    )
    human_boundary = len(HUMANS)
    ax.axvline(human_boundary, color="black", linewidth=2)
    ax.axhline(human_boundary, color="black", linewidth=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title, fontweight="bold", pad=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def consensus_sensitivity(
    maps_by_pair: list[dict[str, tuple[np.ndarray, np.ndarray]]],
) -> dict[str, dict[str, list[float] | list[int]]]:
    totals = {
        observer: {
            "detected_pos": [0] * len(HUMANS),
            "total_pos": [0] * len(HUMANS),
            "detected_neg": [0] * len(HUMANS),
            "total_neg": [0] * len(HUMANS),
        }
        for observer in GENERATED_OBSERVERS
    }
    for maps in maps_by_pair:
        human_pos = [maps[human][0] for human in HUMANS]
        human_neg = [maps[human][1] for human in HUMANS]
        for observer in GENERATED_OBSERVERS:
            for sign, observed, references in (
                ("pos", maps[observer][0], human_pos),
                ("neg", maps[observer][1], human_neg),
            ):
                for level, (detected, total) in enumerate(
                    base.get_sensitivity_at_consensus_levels(observed, references)
                ):
                    totals[observer][f"detected_{sign}"][level] += detected
                    totals[observer][f"total_{sign}"][level] += total

    output = {}
    for observer, counts in totals.items():
        output[observer] = dict(counts)
        for sign in ("pos", "neg"):
            output[observer][f"recall_{sign}"] = [
                detected / total if total else 0.0
                for detected, total in zip(
                    counts[f"detected_{sign}"],
                    counts[f"total_{sign}"],
                )
            ]
    return output


def plot_consensus_sensitivity(metrics: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    colors = ["#222222", "#0072B2", "#D55E00", "#009E73"]
    for axis, sign, title in (
        (axes[0], "pos", "Positive / worsening change"),
        (axes[1], "neg", "Negative / improving change"),
    ):
        for observer, color in zip(GENERATED_OBSERVERS, colors):
            axis.plot(
                range(1, len(HUMANS) + 1),
                metrics[observer][f"recall_{sign}"],
                marker="o",
                linewidth=2,
                label=observer,
                color=color,
            )
        axis.set_ylim(0, 1.02)
        axis.set_xticks(range(1, len(HUMANS) + 1))
        axis.set_xlabel("Pathologist consensus level")
        axis.set_ylabel("Sensitivity")
        axis.set_title(title, fontweight="bold")
        axis.grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True)
    fig.suptitle("Generated Observers vs Five-Pathologist Consensus", fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def arm_agreement_rows(
    pair_numbers: list[int],
    maps_by_pair: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    ellipse_observer: str,
    heatmap_observer: str,
) -> list[dict]:
    rows = []
    for pair_number, maps in zip(pair_numbers, maps_by_pair):
        ellipse_pos, ellipse_neg = maps[ellipse_observer]
        heatmap_pos, heatmap_neg = maps[heatmap_observer]
        ellipse_has_pos = bool(np.any(ellipse_pos))
        ellipse_has_neg = bool(np.any(ellipse_neg))
        heatmap_has_pos = bool(np.any(heatmap_pos))
        heatmap_has_neg = bool(np.any(heatmap_neg))
        positive_agreement, positive_disagreement = base.get_pairwise_detections(
            ellipse_pos, heatmap_pos
        )
        negative_agreement, negative_disagreement = base.get_pairwise_detections(
            ellipse_neg, heatmap_neg
        )
        rows.append(
            {
                "pair_id": f"pair{pair_number}",
                "ellipse_has_change": ellipse_has_pos or ellipse_has_neg,
                "heatmap_has_change": heatmap_has_pos or heatmap_has_neg,
                "change_presence_agrees": (ellipse_has_pos or ellipse_has_neg)
                == (heatmap_has_pos or heatmap_has_neg),
                "positive_presence_agrees": ellipse_has_pos == heatmap_has_pos,
                "negative_presence_agrees": ellipse_has_neg == heatmap_has_neg,
                "positive_overlap_detections": positive_agreement,
                "positive_nonoverlap_detections": positive_disagreement,
                "negative_overlap_detections": negative_agreement,
                "negative_nonoverlap_detections": negative_disagreement,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    experiment_root = (
        annotation_root
        / "LLM_Pathology_Heatmaps"
        / "claude_sonnet_5_full_experiment"
    )
    parser = argparse.ArgumentParser(description="Compare Claude experiment arms as additional observers.")
    parser.add_argument("--annotation-root", type=Path, default=annotation_root)
    parser.add_argument("--experiment-root", type=Path, default=experiment_root)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=repo_root
        / "python_files"
        / "Sahar_work"
        / "files"
        / "predictions"
        / "itamar_segs_sq_plus_lmm5_99_100",
    )
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--annotator-label", default="Claude")
    parser.add_argument("--out-dir", type=Path, default=experiment_root / "comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global GENERATED_OBSERVERS, OBSERVERS
    GENERATED_OBSERVERS = [
        "ICU Model",
        f"{args.annotator_label} Ellipses",
        f"{args.annotator_label} Heatmap",
        f"{args.annotator_label} Precision",
    ]
    OBSERVERS = [*HUMANS, *GENERATED_OBSERVERS]
    ellipse_observer, heatmap_observer, precision_observer = GENERATED_OBSERVERS[1:]
    human_indices = {
        human: gdl._build_pair_index(args.annotation_root / "Annotations" / human)
        for human in HUMANS
    }
    pair_roots = [args.annotation_root / f"Pairs{batch}" for batch in range(1, 9)]
    ellipse_root = args.experiment_root / "rendered_ellipses" / "predictions"
    heatmap_root = args.experiment_root / "rendered_heatmap" / "predictions"

    maps_by_pair = []
    processed_pairs = []
    skipped = []
    for pair_number in range(1, args.num_pairs + 1):
        try:
            pair_dir = gdl.find_pair_path(pair_roots, pair_number)
            if pair_dir is None:
                raise FileNotFoundError("pair directory missing")
            scans = sorted(
                path
                for path in pair_dir.glob("*.nii.gz")
                if not path.name.endswith("_lung_seg.nii.gz")
            )
            if len(scans) != 2:
                raise ValueError(f"expected two scans; found {len(scans)}")
            shape = tuple(nib.load(str(scans[1])).shape[:2])
            maps = {}
            for human in HUMANS:
                annotation_path = human_indices[human].get(pair_number)
                if annotation_path is None:
                    raise FileNotFoundError(f"missing {human} annotation")
                maps[human] = gdl.load_labels_map(annotation_path, shape)

            model_path, crop_info_path = loo._resolve_pred(args.model_root, pair_number)
            if model_path is None or crop_info_path is None:
                raise FileNotFoundError("missing ICU model prediction or crop_info.json")
            maps["ICU Model"] = loo.load_model_maps_crop_info(
                model_path,
                crop_info_path,
                shape,
                min_cc_size=0,
                min_cc_intensity=0.0,
            )
            maps[ellipse_observer] = load_full_prediction(
                ellipse_root / f"pair{pair_number}" / "output.nii.gz",
                shape,
            )
            maps[heatmap_observer] = load_full_prediction(
                heatmap_root / f"pair{pair_number}" / "output.nii.gz",
                shape,
            )
            maps[precision_observer] = load_full_prediction(
                args.experiment_root
                / "rendered_heatmap_precise"
                / "predictions"
                / f"pair{pair_number}"
                / "output.nii.gz",
                shape,
            )
            maps_by_pair.append(maps)
            processed_pairs.append(pair_number)
        except Exception as exc:
            skipped.append({"pair_id": f"pair{pair_number}", "error": str(exc)})

    if not maps_by_pair:
        raise RuntimeError("No complete pairs were available for comparison")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrices = pairwise_matrices(maps_by_pair)
    titles = {
        "per_detection_all": "Pairwise Agreement per Detection — All Changes",
        "per_detection_pos": "Pairwise Agreement per Detection — Positive Changes",
        "per_detection_neg": "Pairwise Agreement per Detection — Negative Changes",
        "per_pair_all": "Pairwise Agreement per Pair — All Changes",
        "per_pair_pos": "Pairwise Agreement per Pair — Positive Changes",
        "per_pair_neg": "Pairwise Agreement per Pair — Negative Changes",
    }
    for name, matrix in matrices.items():
        pd.DataFrame(matrix, index=OBSERVERS, columns=OBSERVERS).to_csv(
            args.out_dir / f"{name}.csv"
        )
        plot_matrix(matrix, args.out_dir / f"{name}.png", titles[name])

    sensitivity = consensus_sensitivity(maps_by_pair)
    plot_consensus_sensitivity(sensitivity, args.out_dir / "consensus_sensitivity.png")

    arm_rows = arm_agreement_rows(
        processed_pairs,
        maps_by_pair,
        ellipse_observer,
        heatmap_observer,
    )
    agreement_slug = args.annotator_label.lower().replace(" ", "_")
    with (args.out_dir / f"{agreement_slug}_arm_agreement_by_pair.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(arm_rows[0]))
        writer.writeheader()
        writer.writerows(arm_rows)

    presence_agreement = float(np.mean([row["change_presence_agrees"] for row in arm_rows]))
    positive_agreement = float(np.mean([row["positive_presence_agrees"] for row in arm_rows]))
    negative_agreement = float(np.mean([row["negative_presence_agrees"] for row in arm_rows]))
    summary = {
        "pairs_processed": len(processed_pairs),
        "processed_pair_numbers": processed_pairs,
        "skipped": skipped,
        "observers": OBSERVERS,
        f"{agreement_slug}_arm_pair_level_agreement": {
            "any_change_presence": presence_agreement,
            "positive_change_presence": positive_agreement,
            "negative_change_presence": negative_agreement,
        },
        "consensus_sensitivity": sensitivity,
        "pairwise_matrices": {
            name: matrix.round(6).tolist()
            for name, matrix in matrices.items()
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "pairs_processed": len(processed_pairs),
        "skipped": len(skipped),
        f"{agreement_slug}_arm_pair_level_agreement": summary[
            f"{agreement_slug}_arm_pair_level_agreement"
        ],
        "output": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
