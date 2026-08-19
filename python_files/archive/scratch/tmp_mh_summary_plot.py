"""Plot M-H agreement summary across all ROI types as a heatmap."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def make_heatmap(data, roi_names, title, out_path, vmin, vmax, center):
    df = pd.DataFrame(data, index=roi_names)
    df["Avg"] = df.mean(axis=1)

    N_rows, N_cols = df.shape
    figsize_w = max(8, int(N_cols * 1.2))
    figsize_h = max(6, int(N_rows * 0.7))

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))

    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "aspect": 20, "label": "PAI"},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        ax=ax,
    )

    # Separator before Avg column
    ax.axvline(x=N_cols - 1, color="black", linewidth=1.5, linestyle="--")

    # Group labels
    human_center = (N_cols - 1) / 2
    avg_center = N_cols - 0.5
    text_y = -0.03

    ax.text(
        human_center, text_y,
        "M - H (per human)",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", fontstyle="italic", color="#333333",
        transform=ax.get_xaxis_transform(),
    )
    ax.text(
        avg_center, text_y,
        "Avg",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", fontstyle="italic", color="#333333",
        transform=ax.get_xaxis_transform(),
    )

    plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
    plt.yticks(rotation=0, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=30)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to: {out_path}")


roi_names = [
    "full_image",
    "full_thorax",
    "icu_segs",
    "lungs",
    "lungs_convex_hull",
    "lungs_heart",
    "lungs_margin5",
    "lungs_med_margin5",
    "lungs_mediastinum",
]

# --- Per Detection (all) ---
per_det_data = {
    "Avi":    [0.45, 0.48, 0.48, 0.43, 0.43, 0.42, 0.45, 0.44, 0.41],
    "Benny":  [0.36, 0.41, 0.40, 0.35, 0.35, 0.34, 0.38, 0.37, 0.37],
    "Sigal":  [0.45, 0.44, 0.50, 0.45, 0.45, 0.44, 0.49, 0.48, 0.44],
    "Smadar": [0.42, 0.46, 0.47, 0.43, 0.43, 0.42, 0.50, 0.48, 0.44],
    "Nitzan": [0.45, 0.55, 0.55, 0.49, 0.49, 0.48, 0.53, 0.54, 0.50],
}

make_heatmap(
    per_det_data, roi_names,
    "M-H Pairwise Agreement Index Per Detection (all)\nAcross ROI Types",
    "Sahar_work/files/ov_results/mh_agreement_summary_all_rois.png",
    vmin=0.30, vmax=0.60, center=0.45,
)

# --- Per Pair (all) ---
per_pair_data = {
    "Avi":    [0.53, 0.56, 0.58, 0.54, 0.54, 0.51, 0.55, 0.54, 0.51],
    "Benny":  [0.42, 0.52, 0.48, 0.46, 0.46, 0.45, 0.45, 0.45, 0.45],
    "Sigal":  [0.52, 0.53, 0.59, 0.53, 0.53, 0.53, 0.57, 0.56, 0.52],
    "Smadar": [0.53, 0.56, 0.56, 0.56, 0.56, 0.54, 0.61, 0.58, 0.56],
    "Nitzan": [0.51, 0.59, 0.61, 0.55, 0.55, 0.54, 0.60, 0.59, 0.54],
}

make_heatmap(
    per_pair_data, roi_names,
    "M-H Pairwise Agreement Index Per Pair (all)\nAcross ROI Types",
    "Sahar_work/files/ov_results/mh_agreement_per_pair_summary_all_rois.png",
    vmin=0.40, vmax=0.65, center=0.52,
)
