from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PHYSICIANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]

# Matrices provided from the 98-pair no-CC run (rows/cols: Avi, Benny, Sigal, Smadar, Nitzan, Model)
PER_LABEL_POS = np.array([
    [1.0, 0.50777202, 0.64197531, 0.56043956, 0.60824742, 0.52261307],
    [0.50777202, 1.0, 0.57458564, 0.47761194, 0.5258216, 0.4587156],
    [0.64197531, 0.57458564, 1.0, 0.57647059, 0.64835165, 0.51336898],
    [0.56043956, 0.47761194, 0.57647059, 1.0, 0.63366337, 0.46376812],
    [0.60824742, 0.5258216, 0.64835165, 0.63366337, 1.0, 0.50228311],
    [0.52261307, 0.4587156, 0.51336898, 0.46376812, 0.50228311, 1.0],
])

PER_LABEL_NEG = np.array([
    [1.0, 0.42647059, 0.52380952, 0.44094488, 0.55555556, 0.4109589],
    [0.42647059, 1.0, 0.43055556, 0.45517241, 0.48148148, 0.34146341],
    [0.52380952, 0.43055556, 1.0, 0.54814815, 0.60526316, 0.48051948],
    [0.44094488, 0.45517241, 0.54814815, 1.0, 0.64052288, 0.42580645],
    [0.55555556, 0.48148148, 0.60526316, 0.64052288, 1.0, 0.52325581],
    [0.4109589, 0.34146341, 0.48051948, 0.42580645, 0.52325581, 1.0],
])

PER_LABEL_ALL = np.array([
    [1.0, 0.47416413, 0.59027778, 0.51132686, 0.58579882, 0.47536232],
    [0.47416413, 1.0, 0.51076923, 0.46820809, 0.50666667, 0.40837696],
    [0.59027778, 0.51076923, 1.0, 0.56393443, 0.62874251, 0.49853372],
    [0.51132686, 0.46820809, 0.56393443, 1.0, 0.63661972, 0.44751381],
    [0.58579882, 0.50666667, 0.62874251, 0.63661972, 1.0, 0.51150895],
    [0.47536232, 0.40837696, 0.49853372, 0.44751381, 0.51150895, 1.0],
])

PER_PAIR_POS = np.array([
    [1.0, 0.4452381, 0.6122449, 0.56156463, 0.58809524, 0.51972789],
    [0.4452381, 1.0, 0.57040816, 0.43741497, 0.47993197, 0.43741497],
    [0.6122449, 0.57040816, 1.0, 0.6037415, 0.63061224, 0.5],
    [0.56156463, 0.43741497, 0.6037415, 1.0, 0.63673469, 0.51904762],
    [0.58809524, 0.47993197, 0.63061224, 0.63673469, 1.0, 0.51258503],
    [0.51972789, 0.43741497, 0.5, 0.51904762, 0.51258503, 1.0],
])

PER_PAIR_NEG = np.array([
    [1.0, 0.49829932, 0.60884354, 0.59183673, 0.64081633, 0.51258503],
    [0.49829932, 1.0, 0.51496599, 0.54557823, 0.51632653, 0.44591837],
    [0.60884354, 0.51496599, 1.0, 0.6452381, 0.70034014, 0.59115646],
    [0.59183673, 0.54557823, 0.6452381, 1.0, 0.70646259, 0.55],
    [0.64081633, 0.51632653, 0.70034014, 0.70646259, 1.0, 0.58027211],
    [0.51258503, 0.44591837, 0.59115646, 0.55, 0.58027211, 1.0],
])

PER_PAIR_ALL = np.array([
    [1.0, 0.525, 0.64105928, 0.59453353, 0.64958698, 0.54281665],
    [0.525, 1.0, 0.58948008, 0.54016035, 0.55726433, 0.50183026],
    [0.64105928, 0.58948008, 1.0, 0.65347425, 0.69975705, 0.58254778],
    [0.59453353, 0.54016035, 0.65347425, 1.0, 0.69849368, 0.55741011],
    [0.64958698, 0.55726433, 0.69975705, 0.69849368, 1.0, 0.5781827],
    [0.54281665, 0.50183026, 0.58254778, 0.55741011, 0.5781827, 1.0],
])


def upper_tri_values(mat: np.ndarray, n: int = 5) -> np.ndarray:
    sub = mat[:n, :n]
    iu = np.triu_indices(n, k=1)
    return sub[iu]


def describe_distribution(values: np.ndarray) -> dict[str, float | str]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": vmin,
        "max": vmax,
        "range": vmax - vmin,
        "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
    }


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def exact_sign_flip_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Two-sided exact sign-flip test on paired differences of means.
    d = x - y
    obs = float(np.mean(d))
    n = d.size
    means = []
    for mask in range(1 << n):
        signs = np.ones(n, dtype=float)
        for b in range(n):
            if (mask >> b) & 1:
                signs[b] = -1.0
        means.append(float(np.mean(signs * d)))
    means = np.array(means)
    p = float(np.mean(np.abs(means) >= abs(obs)))
    return obs, p


def plot_heatmap(mat: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    })
    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    df = pd.DataFrame(mat, index=labels, columns=labels)
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.25,
        linewidths=0.0,
        linecolor="black",
        cbar_kws={"shrink": 1.0, "aspect": 20, "label": "PAI"},
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.tick_params(axis="x", rotation=45, labelsize=12)
    ax.tick_params(axis="y", rotation=0, labelsize=12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_boxplot(data: dict[str, np.ndarray], title: str, ylabel: str, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    })
    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    keys = list(data.keys())
    vals = [data[k] for k in keys]
    ax.boxplot(vals, tick_labels=keys, showmeans=True)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="gray")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_consensus_sensitivity(sens: dict, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    lvls = [1, 2, 3, 4, 5]
    pos = [sens[f"Sensitivity Consensus Level {k} (Positive)"] for k in lvls]
    neg = [sens[f"Sensitivity Consensus Level {k} (Negative)"] for k in lvls]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    ax.plot(lvls, pos, label="Positive changes", color="#003366", linestyle="-", linewidth=2.0, marker="o", markersize=6)
    ax.plot(lvls, neg, label="Negative changes", color="#800000", linestyle="--", linewidth=2.0, marker="s", markersize=6)
    ax.set_title("Sensitivity at Consensus Levels (Physicians)", pad=15)
    ax.set_xlabel("Consensus Level")
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(lvls)
    ax.set_xlim(0.5, len(lvls) + 0.5)
    ax.grid(True, which="major", linestyle=":", alpha=0.6, color="gray")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")

    for x, y in zip(lvls, pos):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="#003366")
    for x, y in zip(lvls, neg):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color="#800000")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_physician_counts(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    x = np.arange(len(df))
    w = 0.28
    ax.bar(x - w, df["mean_pos"], width=w, label="Positive")
    ax.bar(x, df["mean_neg"], width=w, label="Negative")
    ax.bar(x + w, df["mean_all"], width=w, label="All")
    ax.set_xticks(x)
    ax.set_xticklabels(df["physician"])
    ax.set_ylabel("Mean labels per pair")
    ax.set_title("Label Burden by Physician")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build physician-only OV tables and plots from 98-pair outputs.")
    parser.add_argument(
        "--ov-dir",
        type=Path,
        default=Path("Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98"),
        help="Directory containing sensitivity and label-count json files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/paper_stats"),
        help="Output directory for generated paper tables and plots",
    )
    args = parser.parse_args()

    ov_dir = args.ov_dir
    out_dir = args.out_dir
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with (ov_dir / "sensitivity_measures.json").open("r", encoding="utf-8") as f:
        sens = json.load(f)
    with (ov_dir / "total_labels_marked_pos.json").open("r", encoding="utf-8") as f:
        labels_pos = json.load(f)
    with (ov_dir / "total_labels_marked_neg.json").open("r", encoding="utf-8") as f:
        labels_neg = json.load(f)
    with (ov_dir / "total_labels_marked_all.json").open("r", encoding="utf-8") as f:
        labels_all = json.load(f)

    # Human-human distributions (10 physician pairs)
    pp_pos = upper_tri_values(PER_PAIR_POS, 5)
    pp_neg = upper_tri_values(PER_PAIR_NEG, 5)
    pp_all = upper_tri_values(PER_PAIR_ALL, 5)

    pl_pos = upper_tri_values(PER_LABEL_POS, 5)
    pl_neg = upper_tri_values(PER_LABEL_NEG, 5)
    pl_all = upper_tri_values(PER_LABEL_ALL, 5)

    # Table: primary agreement summary
    rows = []
    for metric_name, arr in [
        ("Pairwise agreement - Positive", pp_pos),
        ("Pairwise agreement - Negative", pp_neg),
        ("Pairwise agreement - All", pp_all),
        ("Per-detection agreement - Positive", pl_pos),
        ("Per-detection agreement - Negative", pl_neg),
        ("Per-detection agreement - All", pl_all),
    ]:
        d = describe_distribution(arr)
        ci_lo, ci_hi = bootstrap_ci_mean(arr)
        d["metric"] = metric_name
        d["mean_ci95"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        rows.append(d)

    summary_df = pd.DataFrame(rows)[[
        "metric", "mean", "std", "median", "q1", "q3", "iqr", "min", "max", "range", "mean_pm_std", "mean_ci95"
    ]]
    summary_df.to_csv(tables_dir / "agreement_summary_human_human.csv", index=False)

    # Table: per-physician annotation burden
    burden_rows = []
    for phy in PHYSICIANS:
        p = labels_pos[phy]
        n = labels_neg[phy]
        a = labels_all[phy]
        burden_rows.append(
            {
                "physician": phy,
                "total_pos": int(p[0]),
                "mean_pos": float(p[1]),
                "std_pos": float(p[2]),
                "max_pos": int(p[3]),
                "min_pos": int(p[4]),
                "total_neg": int(n[0]),
                "mean_neg": float(n[1]),
                "std_neg": float(n[2]),
                "max_neg": int(n[3]),
                "min_neg": int(n[4]),
                "total_all": int(a[0]),
                "mean_all": float(a[1]),
                "std_all": float(a[2]),
                "max_all": int(a[3]),
                "min_all": int(a[4]),
            }
        )
    burden_df = pd.DataFrame(burden_rows)
    burden_df.to_csv(tables_dir / "annotation_burden_by_physician.csv", index=False)

    # Table: consensus sensitivity
    sens_rows = []
    for k in [1, 2, 3, 4, 5]:
        p_num, p_den = sens[f"Total detections & changes at Consensus Level {k} (Positive)"]
        n_num, n_den = sens[f"Total detections & changes at Consensus Level {k} (Negative)"]
        sens_rows.append(
            {
                "consensus_level": k,
                "sensitivity_positive": float(sens[f"Sensitivity Consensus Level {k} (Positive)"]),
                "detected_positive": int(p_num),
                "total_positive": int(p_den),
                "sensitivity_negative": float(sens[f"Sensitivity Consensus Level {k} (Negative)"]),
                "detected_negative": int(n_num),
                "total_negative": int(n_den),
            }
        )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(tables_dir / "consensus_sensitivity_summary.csv", index=False)

    # Additional inferential test available from current matrices
    delta_pair, p_pair = exact_sign_flip_pvalue(pp_pos, pp_neg)
    delta_det, p_det = exact_sign_flip_pvalue(pl_pos, pl_neg)
    infer_df = pd.DataFrame(
        [
            {
                "comparison": "Pairwise Positive vs Negative",
                "mean_difference_pos_minus_neg": delta_pair,
                "exact_sign_flip_pvalue": p_pair,
            },
            {
                "comparison": "Per-detection Positive vs Negative",
                "mean_difference_pos_minus_neg": delta_det,
                "exact_sign_flip_pvalue": p_det,
            },
        ]
    )
    infer_df.to_csv(tables_dir / "positive_vs_negative_tests.csv", index=False)

    # Plots
    plot_heatmap(PER_PAIR_POS[:5, :5], PHYSICIANS, "Per-pair Agreement (Positive, H-H)", plots_dir / "heatmap_per_pair_positive_hh.png")
    plot_heatmap(PER_PAIR_NEG[:5, :5], PHYSICIANS, "Per-pair Agreement (Negative, H-H)", plots_dir / "heatmap_per_pair_negative_hh.png")
    plot_heatmap(PER_PAIR_ALL[:5, :5], PHYSICIANS, "Per-pair Agreement (All, H-H)", plots_dir / "heatmap_per_pair_all_hh.png")

    plot_heatmap(PER_LABEL_POS[:5, :5], PHYSICIANS, "Per-detection Agreement (Positive, H-H)", plots_dir / "heatmap_per_detection_positive_hh.png")
    plot_heatmap(PER_LABEL_NEG[:5, :5], PHYSICIANS, "Per-detection Agreement (Negative, H-H)", plots_dir / "heatmap_per_detection_negative_hh.png")
    plot_heatmap(PER_LABEL_ALL[:5, :5], PHYSICIANS, "Per-detection Agreement (All, H-H)", plots_dir / "heatmap_per_detection_all_hh.png")

    plot_boxplot(
        {
            "Positive": pp_pos,
            "Negative": pp_neg,
            "All": pp_all,
        },
        "Human-Human Pairwise Agreement Distribution",
        "Agreement",
        plots_dir / "boxplot_pairwise_hh.png",
    )

    plot_boxplot(
        {
            "Positive": pl_pos,
            "Negative": pl_neg,
            "All": pl_all,
        },
        "Human-Human Per-detection Agreement Distribution",
        "Agreement",
        plots_dir / "boxplot_per_detection_hh.png",
    )

    plot_consensus_sensitivity(sens, plots_dir / "consensus_sensitivity_positive_negative.png")
    plot_physician_counts(burden_df, plots_dir / "annotation_burden_by_physician.png")

    # Save raw H-H pair vectors for transparency.
    raw_df = pd.DataFrame(
        {
            "pair_index": list(range(1, 11)),
            "pairwise_positive": pp_pos,
            "pairwise_negative": pp_neg,
            "pairwise_all": pp_all,
            "per_detection_positive": pl_pos,
            "per_detection_negative": pl_neg,
            "per_detection_all": pl_all,
        }
    )
    raw_df.to_csv(tables_dir / "raw_hh_pair_values.csv", index=False)

    print(f"Saved tables to: {tables_dir.resolve()}")
    print(f"Saved plots to: {plots_dir.resolve()}")
    print("Note: Multi-rater kappa/alpha cannot be recovered exactly from only aggregate matrices and totals.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
