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
        n = p.name.lower()
        if n.endswith(".nii") or n.endswith(".nii.gz"):
            yield p


def flip_data(data: np.ndarray, axis: int) -> np.ndarray:
    if data.ndim <= axis:
        return data
    return np.flip(data, axis=axis)


def out_path_for(src: Path, in_place: bool, suffix: str) -> Path:
    if in_place:
        return src
    n = src.name
    if n.lower().endswith(".nii.gz"):
        return src.with_name(n[:-7] + suffix + ".nii.gz")
    if n.lower().endswith(".nii"):
        return src.with_name(n[:-4] + suffix + ".nii")
    return src.with_name(n + suffix)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flip NIfTI files along selected axis.")
    parser.add_argument("roots", nargs="+", type=Path, help="Root folders to process recursively.")
    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="y",
        help="Flip axis in image plane: y=left-right mirror fix (default), x=up-down.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite original files.")
    args = parser.parse_args()

    axis_num = 1 if args.axis == "y" else 0
    suffix = "_flipY" if axis_num == 1 else "_flipX"

    total = 0
    for root in args.roots:
        count = 0
        if not root.exists():
            print(f"[SKIP] Missing root: {root}")
            continue
        for p in iter_nifti_files(root):
            img = nib.load(str(p))
            data = img.get_fdata(dtype=np.float32)
            out = flip_data(data, axis_num)
            dst = out_path_for(p, args.in_place, suffix)
            nib.save(nib.Nifti1Image(out, img.affine, img.header), str(dst))
            count += 1
            print(f"[OK] {p} -> {dst}")
        total += count
        print(f"[ROOT DONE] {root}: {count} file(s)")

    print(f"Done. Flipped {total} NIfTI file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
