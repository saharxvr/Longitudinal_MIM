#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nibabel as nib
from PIL import Image


def convert_image_to_nifti(image_path, delete_source=False):
    """
    Convert a single image file to NIfTI (.nii.gz) in the same directory.
    """
    # Load image
    img = Image.open(image_path)
    data = np.array(img)

    # Convert RGB → grayscale if needed
    if data.ndim == 3:
        data = data.mean(axis=-1)

    data = data.astype(np.float32)

    # Identity affine
    affine = np.eye(4)

    nifti_img = nib.Nifti1Image(data, affine)

    out_path = os.path.splitext(image_path)[0] + ".nii.gz"
    nib.save(nifti_img, out_path)

    if delete_source:
        os.remove(image_path)

    return out_path


def process_directory(root_dir, delete_source=False):
    """
    Recursively process all image files under root_dir.
    """
    count = 0
    supported_exts = {".png", ".jpg", ".jpeg"}

    for root, _, files in os.walk(root_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_exts:
                image_path = os.path.join(root, fname)
                out_path = convert_image_to_nifti(image_path, delete_source)
                print(f"[OK] {image_path} → {out_path}")
                count += 1

    print(f"\nDone. Converted {count} image files.")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert PNG/JPG/JPEG files to NIfTI, preserving directory structure."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory containing PNG files"
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source image files after conversion"
    )

    args = parser.parse_args()
    process_directory(args.root_dir, args.delete_source)


if __name__ == "__main__":
    main()
