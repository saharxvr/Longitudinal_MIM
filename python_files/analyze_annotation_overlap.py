"""Analyze pixel-level overlap between annotators' ellipse annotations.

This script rasterizes the rotated ellipses saved by the annotation tool JSONs (792x792 canvas
coordinates) and computes:

- For each pair (scan): how many annotators overlap on each pixel (consensus count map)
- Which *individual ellipses* have zero pixel overlap with any other annotator (isolated)

Outputs CSVs with per-ellipse overlap stats and per-pair summary.

Typical usage:
    python analyze_annotation_overlap.py --pair-start 30 --pair-end 60

By default it reads:
    annotation tool/Annotations/<Person>/**.json

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PAIR_RE = re.compile(r"pair\D*(\d+)", re.IGNORECASE)
P_PREFIX_RE = re.compile(r"\bp\s*_?(\d+)\b", re.IGNORECASE)
LEADING_NUM_RE = re.compile(r"^(\d+)")
ANY_NUM_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class EllipseAnn:
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float
    label: str
    comment: str = ""
    size_change: str | None = None
    intensity_change: str | None = None
    tag: str | None = None
    tag_other: str | None = None


def parse_pair_number(path: Path) -> int | None:
    text = path.stem

    match = PAIR_RE.search(text)
    if match:
        return int(match.group(1))

    match = P_PREFIX_RE.search(text)
    if match:
        return int(match.group(1))

    match = LEADING_NUM_RE.search(text)
    if match:
        return int(match.group(1))

    match = ANY_NUM_RE.search(text)
    if match:
        return int(match.group(1))

    return None


def load_annotation_file(json_path: Path) -> tuple[str | None, str | None, list[EllipseAnn]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Unexpected JSON format")

    prior_name: str | None = None
    current_name: str | None = None
    header = data[0]
    if isinstance(header, str) and "|" in header:
        parts = [p.strip() for p in header.split("|")]
        if len(parts) >= 2:
            prior_name, current_name = parts[0], parts[1]

    ellipses: list[EllipseAnn] = []
    for item in data[1:]:
        if not isinstance(item, dict):
            continue
        ellipses.append(
            EllipseAnn(
                cx=float(item["cx"]),
                cy=float(item["cy"]),
                rx=float(item["rx"]),
                ry=float(item["ry"]),
                angle_deg=float(item.get("angle", 0.0)),
                label=str(item.get("label", "")),
                comment=str(item.get("comment", "")),
                size_change=(str(item.get("size_change")) if "size_change" in item else None),
                intensity_change=(str(item.get("intensity_change")) if "intensity_change" in item else None),
                tag=(str(item.get("tag")) if "tag" in item else None),
                tag_other=(str(item.get("tag_other")) if "tag_other" in item else None),
            )
        )

    return prior_name, current_name, ellipses


def iter_person_annotation_files(annotations_root: Path, persons: list[str] | None) -> list[tuple[str, Path]]:
    if persons:
        person_dirs = [annotations_root / p for p in persons]
    else:
        person_dirs = [p for p in annotations_root.iterdir() if p.is_dir()]

    out: list[tuple[str, Path]] = []
    for person_dir in person_dirs:
        if not person_dir.exists():
            continue
        for p in sorted(person_dir.rglob("*.json")):
            out.append((person_dir.name, p))
    return out


def choose_one_file_per_person_pair(files: Iterable[tuple[str, Path]]) -> dict[tuple[int, str], Path]:
    """If duplicates exist for the same (pair, person), keep the newest mtime."""
    best: dict[tuple[int, str], Path] = {}
    for person, path in files:
        pair = parse_pair_number(path)
        if pair is None:
            continue
        key = (pair, person)
        if key not in best:
            best[key] = path
            continue
        if path.stat().st_mtime > best[key].stat().st_mtime:
            best[key] = path
    return best


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, math.floor(v))))


def rasterize_rotated_ellipse(ann: EllipseAnn, height: int, width: int) -> tuple[tuple[slice, slice], np.ndarray]:
    """Return ((yslice, xslice), mask_roi) for the ellipse."""
    # Bounding box (conservative) for a rotated ellipse.
    # Use max radius as quick bound; keeps code simple and fast enough.
    r = max(ann.rx, ann.ry)

    x0 = _clamp_int(ann.cx - r - 2, 0, width - 1)
    x1 = _clamp_int(ann.cx + r + 2, 0, width - 1)
    y0 = _clamp_int(ann.cy - r - 2, 0, height - 1)
    y1 = _clamp_int(ann.cy + r + 2, 0, height - 1)

    xs = slice(x0, x1 + 1)
    ys = slice(y0, y1 + 1)

    # Coordinate grid in ROI
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    dx = xx.astype(np.float32) - np.float32(ann.cx)
    dy = yy.astype(np.float32) - np.float32(ann.cy)

    a = np.float32(math.radians(ann.angle_deg))
    cos_a = np.float32(math.cos(float(a)))
    sin_a = np.float32(math.sin(float(a)))

    # Rotate into ellipse-aligned coordinates
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a

    rx = np.float32(max(ann.rx, 1e-6))
    ry = np.float32(max(ann.ry, 1e-6))

    inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
    return (ys, xs), inside


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze annotation overlap across annotators.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("annotation tool") / "Annotations",
        help="Root folder containing per-person annotation subfolders.",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=None,
        help="Optional list of person folder names. If omitted, uses all.",
    )
    parser.add_argument("--pair-start", type=int, default=None, help="Inclusive start pair number.")
    parser.add_argument("--pair-end", type=int, default=None, help="Inclusive end pair number.")
    parser.add_argument("--image-size", type=int, default=792, help="Canvas size used by the tool (default 792).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("annotation tool") / "OverlapReports",
        help="Output directory for CSV reports.",
    )

    args = parser.parse_args(argv)

    annotations_root: Path = args.annotations
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = iter_person_annotation_files(annotations_root, args.persons)
    best_files = choose_one_file_per_person_pair(all_files)

    # Group by pair
    pairs: dict[int, dict[str, Path]] = {}
    for (pair, person), path in best_files.items():
        if args.pair_start is not None and pair < args.pair_start:
            continue
        if args.pair_end is not None and pair > args.pair_end:
            continue
        pairs.setdefault(pair, {})[person] = path

    if not pairs:
        print("No pairs found in the requested range.")
        return 2

    h = w = int(args.image_size)

    ellipses_csv = out_dir / "overlap_ellipses.csv"
    pairs_csv = out_dir / "overlap_pairs_summary.csv"

    with ellipses_csv.open("w", newline="", encoding="utf-8") as f_ell, pairs_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_pair:
        ell_writer = csv.DictWriter(
            f_ell,
            fieldnames=[
                "pair",
                "person",
                "annotation_file",
                "ellipse_index",
                "label",
                "tag",
                "tag_other",
                "total_pixels",
                "overlap_pixels_with_others",
                "overlap_fraction",
                "isolated",
            ],
        )
        ell_writer.writeheader()

        pair_writer = csv.DictWriter(
            f_pair,
            fieldnames=[
                "pair",
                "num_persons",
                "total_annotated_pixels_union",
                "shared_pixels_count_ge_2",
                "max_overlap_count",
                "overlap_histogram",
                "isolated_ellipses_count",
            ],
        )
        pair_writer.writeheader()

        for pair_num in sorted(pairs.keys()):
            person_to_path = pairs[pair_num]

            person_masks: dict[str, np.ndarray] = {}
            person_ellipses: dict[str, list[EllipseAnn]] = {}

            # Build per-person masks
            for person, path in person_to_path.items():
                try:
                    _, _, ellipses = load_annotation_file(path)
                except Exception as e:
                    print(f"[SKIP] pair{pair_num} {person}: failed reading {path}: {e}")
                    continue

                mask = np.zeros((h, w), dtype=bool)
                for ann in ellipses:
                    (ys, xs), roi = rasterize_rotated_ellipse(ann, h, w)
                    mask[ys, xs] |= roi

                person_masks[person] = mask
                person_ellipses[person] = ellipses

            if not person_masks:
                continue

            # Count how many annotators cover each pixel
            count_map = np.zeros((h, w), dtype=np.uint8)
            for mask in person_masks.values():
                count_map += mask.astype(np.uint8)

            union_mask = count_map > 0
            shared_mask = count_map >= 2

            # histogram of overlap levels (e.g., 1:1234;2:56;3:0)
            max_level = int(count_map.max())
            hist_parts: list[str] = []
            for level in range(1, max_level + 1):
                hist_parts.append(f"{level}:{int((count_map == level).sum())}")
            overlap_histogram = ";".join(hist_parts)

            isolated_ellipses_count = 0

            # Per-ellipse overlap stats
            for person, ellipses in person_ellipses.items():
                mask_self = person_masks[person]
                others_present = (count_map.astype(np.int16) - mask_self.astype(np.int16)) > 0

                for i, ann in enumerate(ellipses):
                    (ys, xs), roi = rasterize_rotated_ellipse(ann, h, w)
                    total_pixels = int(roi.sum())
                    overlap_pixels = int((roi & others_present[ys, xs]).sum())
                    overlap_fraction = float(overlap_pixels / total_pixels) if total_pixels > 0 else 0.0
                    isolated = overlap_pixels == 0 and total_pixels > 0
                    if isolated:
                        isolated_ellipses_count += 1

                    tag = ann.tag
                    if tag == "Other" and ann.tag_other:
                        tag_other = ann.tag_other
                    else:
                        tag_other = "" if ann.tag_other is None else ann.tag_other

                    ell_writer.writerow(
                        {
                            "pair": pair_num,
                            "person": person,
                            "annotation_file": str(person_to_path[person].relative_to(annotations_root)),
                            "ellipse_index": i,
                            "label": ann.label,
                            "tag": "" if tag is None else tag,
                            "tag_other": tag_other,
                            "total_pixels": total_pixels,
                            "overlap_pixels_with_others": overlap_pixels,
                            "overlap_fraction": f"{overlap_fraction:.6f}",
                            "isolated": int(isolated),
                        }
                    )

            pair_writer.writerow(
                {
                    "pair": pair_num,
                    "num_persons": len(person_masks),
                    "total_annotated_pixels_union": int(union_mask.sum()),
                    "shared_pixels_count_ge_2": int(shared_mask.sum()),
                    "max_overlap_count": int(count_map.max()),
                    "overlap_histogram": overlap_histogram,
                    "isolated_ellipses_count": isolated_ellipses_count,
                }
            )

    print("Done.")
    print(f"Per-ellipse CSV: {ellipses_csv.resolve()}")
    print(f"Per-pair summary CSV: {pairs_csv.resolve()}")
    print("Tip: filter overlap_ellipses.csv where isolated==1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
