#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image


def convert_png_to_nifti(png_path, delete_png=False):
    """
    Convert a single PNG file to NIfTI (.nii.gz) in the same directory.
    """
    # Load image
    img = Image.open(png_path)
    data = np.array(img)

    # Convert RGB → grayscale if needed
    if data.ndim == 3:
        data = data.mean(axis=-1)

    data = data.astype(np.float32)

    # Identity affine
    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(data, affine)

    out_path = os.path.splitext(png_path)[0] + ".nii.gz"
    nib.save(nifti_img, out_path)

    if delete_png:
        os.remove(png_path)

    return out_path


def process_directory(root_dir, delete_png=False):
    """
    Recursively process all PNG files under root_dir.
    """
    count = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(".png"):
                png_path = os.path.join(root, fname)
                out_path = convert_png_to_nifti(png_path, delete_png)
                print(f"[OK] {png_path} → {out_path}")
                count += 1

    print(f"\nDone. Converted {count} PNG files.")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert PNG files to NIfTI, preserving directory structure."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing PNG files"
    )
    parser.add_argument(
        "--delete-png",
        action="store_true",
        help="Delete PNG files after conversion"
    )

    args = parser.parse_args()
    process_directory(args.root_dir, args.delete_png)


if __name__ == "__main__":
    main()
