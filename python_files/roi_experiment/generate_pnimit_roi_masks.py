"""Generate full_thorax and lungs_med_margin5 ROI masks for PNIMIT pairs.

Reuses core functions from generate_roi_masks.py.

Usage:
    python roi_experiment/generate_pnimit_roi_masks.py \
        --pairs-root "annotation tool/Pairs_PNIMIT_1_pairs" \
        --out-dir Sahar_work/files/roi_masks_pnimit \
        --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from generate_roi_masks import (
    _load_nifti_2d,
    _load_pspnet,
    _save_mask,
    build_all_roi_masks,
    compute_channel_masks,
)

# Only generate these two ROI types
TARGET_ROIS = ["full_thorax", "lungs_med_margin5"]


def pick_both_nii(pair_dir: Path) -> tuple[Path, Path]:
    """Return (prior_path, current_path) — the two raw .nii.gz files."""
    nii_files = sorted(
        [p for p in pair_dir.iterdir()
         if p.name.endswith(".nii.gz")
         and "_lung_seg" not in p.name
         and "_seg" not in p.name],
    )
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected >=2 raw .nii.gz in {pair_dir}, found {len(nii_files)}")
    return nii_files[0], nii_files[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ROI masks for PNIMIT pairs.")
    p.add_argument("--pairs-root", type=Path, required=True,
                   help="Directory containing PNIMIT pair folders (pair_A1_1_2, ...)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output root for ROI masks.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, targets = _load_pspnet(device)

    pairs_root = args.pairs_root
    pair_dirs = sorted([d for d in pairs_root.iterdir() if d.is_dir() and d.name.startswith("pair_")])

    total = len(pair_dirs)
    print(f"Found {total} PNIMIT pair directories")

    success = 0
    failures = []

    for i, pair_dir in enumerate(pair_dirs, 1):
        pair_name = pair_dir.name
        try:
            prior_path, current_path = pick_both_nii(pair_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {pair_name}: {e}")
            failures.append(pair_name)
            continue

        for nii_path in [prior_path, current_path]:
            img_stem = nii_path.name
            if img_stem.endswith(".nii.gz"):
                img_stem = img_stem[:-7]

            img_np, ref_nii = _load_nifti_2d(nii_path)
            channel_masks = compute_channel_masks(img_np, model, targets, device)
            roi_masks = build_all_roi_masks(channel_masks, img_np.shape)

            for roi_name in TARGET_ROIS:
                mask = roi_masks[roi_name]
                out_path = args.out_dir / roi_name / pair_name / f"{img_stem}_seg.nii.gz"
                _save_mask(mask, ref_nii, out_path)

            fg_summary = {n: int(roi_masks[n].sum()) for n in TARGET_ROIS}
            print(f"[{i}/{total}] {pair_name} / {img_stem}: {fg_summary}")

        success += 1

    print(f"\nDone. {success}/{total} pairs processed, {len(failures)} failures.")
    if failures:
        print(f"Failures: {failures}")


if __name__ == "__main__":
    main()
