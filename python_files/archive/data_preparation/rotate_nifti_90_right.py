#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def iter_nifti_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            yield p


def rotate_right(data: np.ndarray) -> np.ndarray:
    if data.ndim < 2:
        return data
    return np.rot90(data, k=-1, axes=(0, 1))


def default_out_path(src: Path, in_place: bool) -> Path:
    if in_place:
        return src
    name = src.name
    if name.lower().endswith(".nii.gz"):
        stem = name[:-7]
        return src.with_name(f"{stem}_rotR90.nii.gz")
    if name.lower().endswith(".nii"):
        stem = name[:-4]
        return src.with_name(f"{stem}_rotR90.nii")
    return src.with_name(f"{name}_rotR90")


def process_root(root: Path, in_place: bool) -> int:
    if not root.exists():
        print(f"[SKIP] Missing root: {root}")
        return 0

    count = 0
    for nii_path in iter_nifti_files(root):
        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)
        rotated = rotate_right(data)

        out_path = default_out_path(nii_path, in_place)
        out_img = nib.Nifti1Image(rotated, img.affine, img.header)
        nib.save(out_img, str(out_path))

        count += 1
        print(f"[OK] {nii_path} -> {out_path}")

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Rotate NIfTI files 90 degrees clockwise (to the right).")
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more root folders to scan recursively for .nii/.nii.gz files.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files. If omitted, writes *_rotR90 copies.",
    )
    args = parser.parse_args()

    total = 0
    for root in args.roots:
        c = process_root(root, in_place=args.in_place)
        total += c
        print(f"[ROOT DONE] {root}: {c} file(s)")

    print(f"Done. Rotated {total} NIfTI file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
