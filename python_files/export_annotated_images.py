"""Export per-person annotated images.

Reads JSON ellipse annotations saved by `annotation tool/main.py` and renders them on the *current*
image (the second `.nii.gz` in the pair folder), saving PNGs into an output folder.

Default inputs:
- Annotations: `annotation tool/Annotations/<Person>/*.json`
- Pairs:       `annotation tool/Pairs*/pair<k>/*.nii.gz`

Usage (from repo root):
    python export_annotated_images.py
    python export_annotated_images.py --persons Avi Sigal
    python export_annotated_images.py --output annotated_exports

"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def _safe_stem(path: Path) -> str:
    # Path.stem of "C2.nii.gz" is "C2.nii"; we want "C2".
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def parse_pair_number(annotation_path: Path) -> int | None:
    text = annotation_path.stem

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


def load_annotation(annotation_path: Path) -> tuple[str | None, str | None, list[EllipseAnn]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected JSON format in {annotation_path}")

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


def find_pair_dir(pairs_root: Path, pair_number: int) -> Path | None:
    # Pair folders are under `annotation tool/Pairs*/pair<k>`.
    for sub in sorted(pairs_root.glob("Pairs*")):
        if not sub.is_dir():
            continue
        candidate = sub / f"pair{pair_number}"
        if candidate.is_dir():
            return candidate
    return None


def choose_current_nii(pair_dir: Path, expected_current_name: str | None) -> Path:
    nii_files = sorted([p for p in pair_dir.iterdir() if p.name.endswith(".nii.gz")])
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected 2 .nii.gz files in {pair_dir}, found {len(nii_files)}")

    if expected_current_name:
        for p in nii_files:
            if _safe_stem(p) == expected_current_name:
                return p

    # Matches the annotation tool: second file after sorting.
    return nii_files[1]


def load_nii_as_pil_gray(nii_path: Path, out_size: int = 792) -> Image.Image:
    nii = nib.load(str(nii_path))
    data = nii.get_fdata().T
    if data.ndim == 3:
        data = data[:, :, data.shape[2] // 2]

    data = np.asarray(data, dtype=np.float32)
    mn = float(np.min(data))
    mx = float(np.max(data))
    if mx <= mn:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        norm = (data - mn) / (mx - mn)

    img_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(img_u8, mode="L")
    im = im.resize((out_size, out_size))
    return im


def ellipse_poly_points(ann: EllipseAnn, steps: int = 96) -> list[tuple[float, float]]:
    a = math.radians(ann.angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    points: list[tuple[float, float]] = []

    for i in range(steps + 1):
        t = 2.0 * math.pi * (i / steps)
        x = ann.rx * math.cos(t)
        y = ann.ry * math.sin(t)
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        points.append((ann.cx + xr, ann.cy + yr))

    return points


def draw_annotation_overlay(
    base_gray: Image.Image,
    ellipses: Iterable[EllipseAnn],
    *,
    line_width: int = 3,
) -> Image.Image:
    im = base_gray.convert("RGB")
    draw = ImageDraw.Draw(im)

    label_colors = {
        "Appearance": (255, 0, 0),
        "Disappearance": (0, 255, 0),
        "Persistence": (255, 255, 0),
    }
    persistence_colors = {
        "Increase": (255, 0, 0),
        "Decrease": (0, 255, 0),
        "None": (255, 255, 0),
    }

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for ann in ellipses:
        color = label_colors.get(ann.label, (255, 255, 255))
        pts = ellipse_poly_points(ann)
        # Outline
        draw.line(pts, fill=color, width=line_width, joint="curve")

        # Optional persistence indicator circles (approximate the tool)
        if ann.label == "Persistence" and ann.size_change and ann.intensity_change:
            space = 8
            r = 6
            c1 = persistence_colors.get(ann.size_change, (255, 255, 0))
            c2 = persistence_colors.get(ann.intensity_change, (255, 255, 0))
            draw.ellipse((ann.cx - space - r, ann.cy - r, ann.cx - space + r, ann.cy + r), fill=c1, outline=None)
            draw.ellipse((ann.cx + space - r, ann.cy - r, ann.cx + space + r, ann.cy + r), fill=c2, outline=None)

        # Text label (kept minimal)
        text_parts: list[str] = [ann.label]
        if ann.tag:
            text_parts.append(ann.tag_other if ann.tag == "Other" and ann.tag_other else ann.tag)
        text = " | ".join([p for p in text_parts if p])
        if text:
            tx = float(ann.cx - ann.rx)
            ty = float(ann.cy - ann.ry) - 12.0
            # background for readability
            if font is not None:
                bbox = draw.textbbox((tx, ty), text, font=font)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
            else:
                draw.text((tx, ty), text, fill=(255, 255, 255))

    return im


def iter_annotation_files(annotations_root: Path, persons: list[str] | None) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    if persons:
        person_dirs = [annotations_root / p for p in persons]
    else:
        person_dirs = [p for p in annotations_root.iterdir() if p.is_dir()]

    for person_dir in person_dirs:
        if not person_dir.exists():
            continue
        for ann_path in sorted(person_dir.rglob("*.json")):
            results.append((person_dir.name, ann_path))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export annotated images per annotator.")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("annotation tool") / "Annotations",
        help="Root folder containing per-person annotation subfolders.",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("annotation tool"),
        help="Root folder containing Pairs1, Pairs2, ...",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotation tool") / "AnnotatedExports",
        help="Output root folder.",
    )
    parser.add_argument(
        "--layout",
        choices=["by_person", "by_pair"],
        default="by_person",
        help="Output layout: by_person => <out>/<Person>/..., by_pair => <out>/pair<k>/<Person>.png",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=None,
        help="Optional list of person folder names (e.g., Avi Sigal). If omitted, exports all.",
    )
    parser.add_argument(
        "--pair-start",
        type=int,
        default=None,
        help="Optional inclusive start pair number to export (e.g., 30).",
    )
    parser.add_argument(
        "--pair-end",
        type=int,
        default=None,
        help="Optional inclusive end pair number to export (e.g., 60).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of annotation files to export (0 = no limit).",
    )

    args = parser.parse_args(argv)

    annotations_root: Path = args.annotations
    pairs_root: Path = args.pairs_root
    out_root: Path = args.output

    ann_files = iter_annotation_files(annotations_root, args.persons)

    if not ann_files:
        print(f"No annotation JSON files found under: {annotations_root}")
        return 2

    exported = 0
    skipped = 0
    attempted = 0

    for person, ann_path in ann_files:
        pair_num = parse_pair_number(ann_path)
        if pair_num is None:
            print(f"[SKIP] Could not parse pair number: {ann_path}")
            skipped += 1
            continue

        if args.pair_start is not None and pair_num < args.pair_start:
            continue
        if args.pair_end is not None and pair_num > args.pair_end:
            continue

        if args.limit and args.limit > 0 and attempted >= args.limit:
            break
        attempted += 1

        pair_dir = find_pair_dir(pairs_root, pair_num)
        if pair_dir is None:
            print(f"[SKIP] Could not find pair folder for pair{pair_num} (from {ann_path})")
            skipped += 1
            continue

        try:
            prior_name, current_name, ellipses = load_annotation(ann_path)
            cur_nii = choose_current_nii(pair_dir, current_name)
            base = load_nii_as_pil_gray(cur_nii)
            out_im = draw_annotation_overlay(base, ellipses)

            current_stem = _safe_stem(cur_nii)
            if args.layout == "by_person":
                out_dir = out_root / person
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"pair{pair_num}_{current_stem}.png"
            else:
                out_dir = out_root / f"pair{pair_num}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{person}.png"
            out_im.save(out_path)

            exported += 1
        except Exception as e:
            print(f"[SKIP] {ann_path}: {e}")
            skipped += 1

    print(f"Done. Exported {exported} images. Skipped {skipped}.")
    print(f"Output folder: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
