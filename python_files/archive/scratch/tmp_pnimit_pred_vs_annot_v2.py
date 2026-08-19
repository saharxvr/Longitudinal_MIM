"""Side-by-side collage: Doctor annotations vs model predictions for PNIMIT pairs.

Uses the pre-rendered model_output.png from each prediction dir (correct 512x512 space),
and shows the full CXR with doctor annotation ellipses.
"""
import matplotlib
matplotlib.use("Agg")

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
ANNOT_TOOL = BASE / "annotation tool"
PAIRS_ROOT = ANNOT_TOOL / "Pairs_PNIMIT_1_pairs"
PREDS_MED5 = ANNOT_TOOL / "predictions_pnimit_lungs_med5"
PREDS_THORAX = ANNOT_TOOL / "predictions_pnimit_full_thorax"
ANNOTATIONS_DIR = BASE / "Sahar_work" / "files" / "pair_A4_1_2"
OUT_DIR = BASE / "Sahar_work" / "files" / "pnimit_pred_vs_annot_collages"

# Annotation label colors
LABEL_COLORS = {
    "Appearance": "lime",
    "Disappearance": "cyan",
    "Increase": "red",
    "Decrease": "blue",
    "Change in position": "yellow",
}
DEFAULT_COLOR = "magenta"


def load_cxr_for_display(path):
    """Load NIfTI CXR and transpose for correct display orientation."""
    data = nib.load(str(path)).get_fdata()
    if data.ndim == 3:
        data = np.squeeze(data)
    return data.T  # transpose to match annotation tool display


def find_annotation_json(pair_name):
    pair_dir = ANNOTATIONS_DIR / pair_name
    if not pair_dir.exists():
        return None
    for f in pair_dir.iterdir():
        if f.suffix == ".json":
            return f
    return None


def draw_annotations_on_ax(ax, annotations, img_shape):
    """Draw annotations, scaling from 792x792 tool coords to actual image size."""
    TOOL_SIZE = 792
    # img_shape is (H, W) after transpose
    scale_x = img_shape[1] / TOOL_SIZE  # width
    scale_y = img_shape[0] / TOOL_SIZE  # height

    for annot in annotations:
        if not isinstance(annot, dict):
            continue
        cx = annot.get("cx", 0) * scale_x
        cy = annot.get("cy", 0) * scale_y
        rx = annot.get("rx", 0) * scale_x
        ry = annot.get("ry", 0) * scale_y
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

        display_text = tag or comment or label
        if display_text:
            ax.text(cx, cy - max(ry, 10) - 5, display_text,
                    fontsize=6, color=color, ha="center", va="bottom",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6, edgecolor="none"))


def process_pair(pair_name):
    pair_dir = PAIRS_ROOT / pair_name
    if not pair_dir.exists():
        return False

    # Find the current image (second nii.gz)
    nii_files = sorted([
        f for f in pair_dir.iterdir()
        if f.name.endswith(".nii.gz")
        and "_seg" not in f.name
        and "_notranspose" not in f.name
        and "_resized" not in f.name
        and f.name.startswith("SynapseExport")
    ])
    if len(nii_files) < 2:
        print(f"  [SKIP] {pair_name}: only {len(nii_files)} raw nii files")
        return False

    # Load full CXR with transpose for display
    current_img = load_cxr_for_display(nii_files[1])

    # Load pre-rendered model_output PNGs
    med5_png = PREDS_MED5 / pair_name / "model_output.png"
    thorax_png = PREDS_THORAX / pair_name / "model_output.png"

    if not med5_png.exists() or not thorax_png.exists():
        print(f"  [SKIP] Missing model_output.png for {pair_name}")
        return False

    med5_img = mpimg.imread(str(med5_png))
    thorax_img = mpimg.imread(str(thorax_png))

    # Load annotations
    annot_path = find_annotation_json(pair_name)
    annotations = []
    if annot_path:
        with open(annot_path) as f:
            annot_data = json.load(f)
        annotations = [a for a in annot_data if isinstance(a, dict)]

    # Create collage: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: Full CXR with doctor annotations
    axes[0].imshow(current_img, cmap="gray")
    draw_annotations_on_ax(axes[0], annotations, current_img.shape)
    n_annots = len(annotations)
    axes[0].set_title(f"Doctor Annotations ({n_annots})", fontsize=12, fontweight="bold")
    axes[0].set_axis_off()

    # Panel 2: lungs_med5 model output (pre-rendered PNG)
    axes[1].imshow(med5_img)
    axes[1].set_title("Pred: lungs_med_margin5", fontsize=12, fontweight="bold")
    axes[1].set_axis_off()

    # Panel 3: full_thorax model output (pre-rendered PNG)
    axes[2].imshow(thorax_img)
    axes[2].set_title("Pred: full_thorax", fontsize=12, fontweight="bold")
    axes[2].set_axis_off()

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
