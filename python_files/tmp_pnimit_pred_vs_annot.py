"""Side-by-side collage: Doctor annotations vs model predictions (lungs_med5 & full_thorax) for PNIMIT pairs."""
import matplotlib
matplotlib.use("Agg")

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
import numpy as np
from matplotlib import colors

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
ANNOT_TOOL = BASE / "annotation tool"
PAIRS_ROOT = ANNOT_TOOL / "Pairs_PNIMIT_1_pairs"
PREDS_MED5 = ANNOT_TOOL / "predictions_pnimit_lungs_med5"
PREDS_THORAX = ANNOT_TOOL / "predictions_pnimit_full_thorax"
ANNOTATIONS_DIR = BASE / "Sahar_work" / "files" / "pair_A4_1_2"
OUT_DIR = BASE / "Sahar_work" / "files" / "pnimit_pred_vs_annot_collages"

# ── Colormap (same as Prediction.py) ────────────────────────────────────────
differential_grad = colors.LinearSegmentedColormap.from_list("my_gradient", (
    (0.000, (0.235, 1.000, 0.239)),
    (0.400, (0.000, 1.000, 0.702)),
    (0.500, (1.000, 0.988, 0.988)),
    (0.600, (1.000, 0.604, 0.000)),
    (1.000, (0.682, 0.000, 0.000)),
))

# Annotation label colors
LABEL_COLORS = {
    "Appearance": "lime",
    "Disappearance": "cyan",
    "Increase": "red",
    "Decrease": "blue",
    "Change in position": "yellow",
}
DEFAULT_COLOR = "magenta"


def load_nii_2d(path):
    data = nib.load(str(path)).get_fdata()
    if data.ndim == 3:
        data = np.squeeze(data)
    return data


def generate_alpha_map(x):
    x_abs = np.abs(x)
    max_val = max(np.max(x_abs), 0.07)
    return x_abs / max_val


def find_annotation_json(pair_name):
    """Find the annotation JSON for a PNIMIT pair."""
    pair_dir = ANNOTATIONS_DIR / pair_name
    if not pair_dir.exists():
        return None
    for f in pair_dir.iterdir():
        if f.suffix == ".json":
            return f
    return None


def draw_annotations_on_ax(ax, annotations, img_shape):
    """Draw annotation ellipses on an axis."""
    for annot in annotations:
        if not isinstance(annot, dict):
            continue
        cx = annot.get("cx", 0)
        cy = annot.get("cy", 0)
        rx = annot.get("rx", 0)
        ry = annot.get("ry", 0)
        angle = annot.get("angle", 0)
        label = annot.get("label", "")
        tag = annot.get("tag", "")
        comment = annot.get("comment", "")

        color = LABEL_COLORS.get(label, DEFAULT_COLOR)

        if rx > 0 and ry > 0:
            ellipse = patches.Ellipse(
                (cx, cy), width=2 * rx, height=2 * ry, angle=angle,
                linewidth=2, edgecolor=color, facecolor="none", linestyle="-"
            )
            ax.add_patch(ellipse)
        else:
            ax.plot(cx, cy, "x", color=color, markersize=10, markeredgewidth=2)

        # Label text
        display_text = tag or comment or label
        if display_text:
            ax.text(cx, cy - max(ry, 10) - 5, display_text,
                    fontsize=6, color=color, ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))


def draw_prediction_on_ax(ax, current_img, output, title):
    """Draw model prediction overlay on current image."""
    ax.imshow(current_img, cmap="gray")
    alphas = generate_alpha_map(output)
    vmin = min(np.min(output), -0.01)
    vmax = max(np.max(output), 0.01)
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(output, alpha=alphas, cmap=differential_grad, norm=divnorm)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_axis_off()
    return im


def process_pair(pair_name):
    pair_dir = PAIRS_ROOT / pair_name
    if not pair_dir.exists():
        return False

    # Load current image (second nii.gz) — only raw SynapseExport files
    nii_files = sorted([
        f for f in pair_dir.iterdir()
        if f.name.endswith(".nii.gz")
        and "_seg" not in f.name
        and "_lung_seg" not in f.name
        and "_notranspose" not in f.name
        and "_resized" not in f.name
        and f.name.startswith("SynapseExport")
    ])
    if len(nii_files) < 2:
        print(f"  [SKIP] {pair_name}: only {len(nii_files)} raw nii files")
        return False
    current_img = load_nii_2d(nii_files[1])

    # Load predictions
    pred_med5_path = PREDS_MED5 / pair_name / "output.nii.gz"
    pred_thorax_path = PREDS_THORAX / pair_name / "output.nii.gz"

    if not pred_med5_path.exists() or not pred_thorax_path.exists():
        print(f"  [SKIP] Missing predictions for {pair_name}")
        return False

    pred_med5 = load_nii_2d(pred_med5_path)
    pred_thorax = load_nii_2d(pred_thorax_path)

    # Load annotations
    annot_path = find_annotation_json(pair_name)
    annotations = []
    if annot_path:
        with open(annot_path) as f:
            annot_data = json.load(f)
        # First element is the image reference string, rest are annotations
        annotations = [a for a in annot_data if isinstance(a, dict)]

    # Create collage: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Current image with doctor annotations
    axes[0].imshow(current_img, cmap="gray")
    draw_annotations_on_ax(axes[0], annotations, current_img.shape)
    n_annots = len(annotations)
    axes[0].set_title(f"Doctor Annotations ({n_annots})", fontsize=10, fontweight="bold")
    axes[0].set_axis_off()

    # Panel 2: lungs_med5 prediction
    draw_prediction_on_ax(axes[1], current_img, pred_med5, "Pred: lungs_med_margin5")

    # Panel 3: full_thorax prediction
    im = draw_prediction_on_ax(axes[2], current_img, pred_thorax, "Pred: full_thorax")

    fig.suptitle(pair_name, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / f"{pair_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    pair_dirs = sorted([
        d.name for d in PAIRS_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("pair_")
    ])
    print(f"Found {len(pair_dirs)} PNIMIT pairs")

    success = 0
    for i, pair_name in enumerate(pair_dirs, 1):
        print(f"[{i}/{len(pair_dirs)}] {pair_name}")
        if process_pair(pair_name):
            success += 1

    print(f"\nDone. {success}/{len(pair_dirs)} collages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
