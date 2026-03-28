"""Apply precomputed ROI masks to existing model predictions.

Takes model output.nii.gz files (from the original Prediction.py) and
ROI masks (from generate_roi_masks.py), multiplies them (zeroing
predictions outside each ROI), and saves the masked outputs.

Size pipeline:
  - Model output:   loaded from NIfTI, squeezed to 2D, resized to mask shape
  - ROI mask:       loaded from NIfTI (768×768 native resolution)
  - Masked output:  same shape as mask → saved as NIfTI
  - OV script:      loads NIfTI, resizes to (768,768) → spatial alignment OK
"""

import argparse
import os
import re
import sys

import nibabel as nib
import numpy as np
from skimage.transform import resize as sk_resize


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


def _pair_sort_key(name: str) -> int:
    nums = re.findall(r"\d+", os.path.basename(name))
    return int(nums[-1]) if nums else 0


def load_output(pred_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load output.nii.gz, squeeze to 2D.  Returns (data_2d, affine)."""
    path = os.path.join(pred_dir, "output.nii.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    nii = nib.load(path)
    data = nii.get_fdata()
    if data.ndim > 2:
        data = np.squeeze(data)
    return data.astype(np.float32), nii.affine


def load_roi_mask(roi_masks_dir: str, roi_name: str, pair_name: str) -> np.ndarray | None:
    """Load a precomputed binary ROI mask.  Returns None for full_image."""
    if roi_name == "full_image":
        return None
    for variant in [pair_name, pair_name.lower()]:
        mask_path = os.path.join(roi_masks_dir, roi_name, variant, "mask.nii.gz")
        if os.path.isfile(mask_path):
            arr = nib.load(mask_path).get_fdata()
            arr = np.squeeze(arr)
            return (arr > 0).astype(np.float32)
    return None


def apply_mask(output: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Resize output to mask shape if needed, then multiply."""
    if output.shape != mask.shape:
        output = sk_resize(
            output, mask.shape,
            order=1, preserve_range=True, anti_aliasing=False,
        ).astype(np.float32)
    return output * mask


def main():
    parser = argparse.ArgumentParser(
        description="Apply ROI masks to model predictions.")
    parser.add_argument("--preds-dir", required=True,
                        help="Dir with pair*/output.nii.gz (original predictions).")
    parser.add_argument("--roi-masks-dir", required=True,
                        help="Root dir with {roi_name}/{pair}/mask.nii.gz.")
    parser.add_argument("--out-dir", required=True,
                        help="Output root.  Writes {roi_name}/{pair}/output.nii.gz.")
    parser.add_argument("--roi-names", nargs="+", default=None,
                        help=f"ROIs to process.  Default: all {len(ROI_NAMES)}.")
    parser.add_argument("--start-pair", type=int, default=1)
    parser.add_argument("--end-pair", type=int, default=100)

    args = parser.parse_args()
    roi_names = args.roi_names or ROI_NAMES

    # Discover pair directories
    pair_dirs = sorted(
        [d for d in os.listdir(args.preds_dir)
         if os.path.isdir(os.path.join(args.preds_dir, d))],
        key=_pair_sort_key,
    )

    done = 0
    skipped = 0

    for pair_name in pair_dirs:
        pair_num = _pair_sort_key(pair_name)
        if pair_num < args.start_pair or pair_num > args.end_pair:
            continue

        pred_path = os.path.join(args.preds_dir, pair_name)
        out_nii = os.path.join(pred_path, "output.nii.gz")
        if not os.path.isfile(out_nii):
            print(f"[SKIP] {pair_name}: no output.nii.gz")
            skipped += 1
            continue

        output, affine = load_output(pred_path)

        for roi_name in roi_names:
            out_dir = os.path.join(args.out_dir, roi_name, pair_name)
            out_file = os.path.join(out_dir, "output.nii.gz")
            if os.path.isfile(out_file):
                continue  # already done

            if roi_name == "full_image":
                # No masking needed — just copy the output at the right size.
                # Ensure 768×768 so it matches annotation resolution.
                if output.shape != (768, 768):
                    masked = sk_resize(
                        output, (768, 768),
                        order=1, preserve_range=True, anti_aliasing=False,
                    ).astype(np.float32)
                else:
                    masked = output.copy()
            else:
                mask = load_roi_mask(args.roi_masks_dir, roi_name, pair_name)
                if mask is None:
                    print(f"  [SKIP] {roi_name}/{pair_name}: mask not found")
                    skipped += 1
                    continue
                masked = apply_mask(output, mask)

            os.makedirs(out_dir, exist_ok=True)
            nii = nib.Nifti1Image(masked, affine)
            nib.save(nii, out_file)
            done += 1

        print(f"{pair_name} done")

    print(f"\nFinished.  Written: {done}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
