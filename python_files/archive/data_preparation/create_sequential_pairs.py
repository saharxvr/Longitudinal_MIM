#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def find_images(case_dir: Path, ext: str) -> list[Path]:
    if ext == ".nii.gz":
        files = [p for p in case_dir.iterdir() if p.is_file() and p.name.lower().endswith(".nii.gz")]
    else:
        files = [p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()]

    return sorted(files, key=natural_key)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create sequential pair folders from each case directory (A1, A2, ...)."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder containing case subdirectories (e.g., A1, A2, ...).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to create pair folders (default: <input-root>_pairs).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".nii.gz",
        help="File extension to pair (default: .nii.gz).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How files are placed into pair folders.",
    )

    args = parser.parse_args()

    input_root = args.input_root
    if args.output_root is None:
        output_root = input_root.parent / f"{input_root.name}_pairs"
    else:
        output_root = args.output_root

    output_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()], key=natural_key)
    total_pairs = 0

    for case_dir in case_dirs:
        images = find_images(case_dir, args.ext)
        if len(images) < 2:
            print(f"[SKIP] {case_dir.name}: found {len(images)} file(s) with ext {args.ext}")
            continue

        for idx in range(len(images) - 1):
            img1 = images[idx]
            img2 = images[idx + 1]

            pair_dir = output_root / f"pair_{case_dir.name}_{idx + 1}_{idx + 2}"
            pair_dir.mkdir(parents=True, exist_ok=True)

            dst1 = pair_dir / img1.name
            dst2 = pair_dir / img2.name

            if args.copy_mode == "hardlink":
                if not dst1.exists():
                    dst1.hardlink_to(img1)
                if not dst2.exists():
                    dst2.hardlink_to(img2)
            else:
                shutil.copy2(img1, dst1)
                shutil.copy2(img2, dst2)

            total_pairs += 1
            print(f"[OK] {pair_dir.name}: {img1.name}, {img2.name}")

    print(f"\nDone. Created {total_pairs} pair directories in: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
