"""Preview ROI masks: 10 ROI overlays per pair, shown as a grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

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


def _load_2d(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    return arr


def _norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def resolve_pair_path(pairs_roots: list[str], pair_num: int) -> Path:
    for root in pairs_roots:
        for name in [f"pair{pair_num}", f"Pair{pair_num}", f"pair_{pair_num}"]:
            p = Path(root) / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Pair {pair_num} not found")


def pick_current_nii(pair_dir: Path) -> tuple[Path, str]:
    """Return (current_nii_path, image_stem) e.g. ('.../9B.nii.gz', '9B')."""
    nii_files = sorted(
        [p for p in pair_dir.iterdir()
         if p.name.endswith(".nii.gz") and "_seg" not in p.name and "_lung_seg" not in p.name],
    )
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected >=2 raw .nii.gz in {pair_dir}")
    current = nii_files[1]
    stem = current.name[:-7]  # strip .nii.gz
    return current, stem


def main():
    p = argparse.ArgumentParser(description="Preview ROI masks as overlay grid.")
    p.add_argument("--pairs-roots", nargs="+", required=True)
    p.add_argument("--roi-masks-dir", type=Path, required=True)
    p.add_argument("--pairs", nargs="+", type=int, required=True,
                   help="Pair numbers to preview")
    p.add_argument("--out", type=Path, default=Path("Sahar_work/files/roi_masks/roi_preview.png"))
    args = p.parse_args()

    pairs = args.pairs
    n_pairs = len(pairs)
    n_rois = len(ROI_NAMES)

    fig, axes = plt.subplots(n_pairs, n_rois, figsize=(2.8 * n_rois, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for row, pair_num in enumerate(pairs):
        try:
            pair_dir = resolve_pair_path(args.pairs_roots, pair_num)
            nii_path, img_stem = pick_current_nii(pair_dir)
            img = _norm(_load_2d(nii_path))
        except FileNotFoundError as e:
            print(f"[SKIP] pair {pair_num}: {e}")
            for col in range(n_rois):
                axes[row, col].axis("off")
            continue

        for col, roi_name in enumerate(ROI_NAMES):
            ax = axes[row, col]
            mask_path = args.roi_masks_dir / roi_name / f"{img_stem}_seg.nii.gz"
            ax.imshow(img.T, cmap="gray")
            if mask_path.exists():
                mask = (_load_2d(mask_path) > 0).astype(np.float32)
                ax.imshow(mask.T, cmap="Reds", alpha=0.35)
                fg_pct = 100 * mask.sum() / mask.size
                ax.set_xlabel(f"{fg_pct:.0f}%", fontsize=7)
            else:
                ax.set_xlabel("missing", fontsize=7, color="red")
            if row == 0:
                ax.set_title(roi_name.replace("_", "\n"), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"pair {pair_num}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("ROI Mask Preview", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
