from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAIR_RE = re.compile(r"pair(\d+)", re.IGNORECASE)


@dataclass
class IsolatedLabel:
    person: str
    label: str
    tag: str
    tag_other: str
    cx: float
    cy: float


def pair_from_name(path: Path) -> int | None:
    match = PAIR_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _split_ann_file_path(raw_path: str) -> Path:
    text = (raw_path or "").strip().replace("/", "\\")
    parts = [p for p in text.split("\\") if p]
    return Path(*parts)


def _load_json_items(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [x for x in data[1:] if isinstance(x, dict)]


def read_isolated_entries(csv_path: Path, annotations_root: Path) -> dict[int, dict[str, list[IsolatedLabel]]]:
    grouped: dict[int, dict[str, list[IsolatedLabel]]] = defaultdict(lambda: defaultdict(list))
    loaded_json_cache: dict[Path, list[dict]] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("isolated") != "1":
                continue

            pair = int(row["pair"])
            person = (row.get("person") or "Unknown").strip()
            ellipse_index = int(row.get("ellipse_index") or 0)

            ann_rel = _split_ann_file_path(row.get("annotation_file") or "")
            ann_path = annotations_root / ann_rel

            if ann_path not in loaded_json_cache:
                if not ann_path.exists():
                    continue
                loaded_json_cache[ann_path] = _load_json_items(ann_path)

            items = loaded_json_cache[ann_path]
            if ellipse_index < 0 or ellipse_index >= len(items):
                continue

            item = items[ellipse_index]
            grouped[pair][person].append(
                IsolatedLabel(
                    person=person,
                    label=str(item.get("label", row.get("label") or "Unknown")),
                    tag=str(item.get("tag", row.get("tag") or "")).strip(),
                    tag_other=str(item.get("tag_other", row.get("tag_other") or "")).strip(),
                    cx=float(item.get("cx", 0.0)),
                    cy=float(item.get("cy", 0.0)),
                )
            )

    return grouped


def _label_text(entry: IsolatedLabel) -> str:
    base = entry.label
    if entry.tag:
        detail = entry.tag_other if entry.tag.lower() == "other" and entry.tag_other else entry.tag
        base = f"{base}/{detail}"
    return f"{base}: disagreement (no overlap)"


def _draw_pair_highlights(
    image: Image.Image,
    persons: list[str],
    entries_by_person: dict[str, list[IsolatedLabel]],
    *,
    person_rows: int,
    width: int,
    height: int,
    pad: int,
    header_h: int,
    image_canvas_size: int,
) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    person_rows = max(1, int(person_rows))
    person_cols = (len(persons) + person_rows - 1) // person_rows

    grid_w = width - pad * (person_cols + 1)
    grid_h = height - header_h - pad * (person_rows + 1)
    cell_w = max(1, grid_w // person_cols)
    cell_h = max(1, grid_h // person_rows)

    fitted_side = min(cell_w, cell_h)
    scale = fitted_side / float(image_canvas_size)
    in_cell_offset_x = (cell_w - fitted_side) // 2
    in_cell_offset_y = (cell_h - fitted_side) // 2

    for person_idx, person in enumerate(persons):
        entries = entries_by_person.get(person, [])
        if not entries:
            continue

        pr = person_idx // person_cols
        pc = person_idx % person_cols

        cell_x0 = pad + pc * (cell_w + pad)
        cell_y0 = header_h + pad + pr * (cell_h + pad)

        for idx, entry in enumerate(entries):
            x = int(cell_x0 + in_cell_offset_x + entry.cx * scale)
            y = int(cell_y0 + in_cell_offset_y + entry.cy * scale)

            ring_r = 10
            draw.ellipse((x - ring_r, y - ring_r, x + ring_r, y + ring_r), outline=(255, 0, 255), width=4)

            text = _label_text(entry)
            tx = x + 12
            ty = y - 16 + (idx % 2) * 16

            if tx > width - 420:
                tx = x - 410
            if ty < 0:
                ty = 0
            if ty > height - 20:
                ty = height - 20

            bbox = draw.textbbox((tx, ty), text, font=font)
            draw.rectangle(bbox, fill=(255, 255, 0))
            draw.text((tx, ty), text, fill=(220, 0, 0), font=font)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Highlight disagreement labels directly on collages.")
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=Path("annotation tool") / "OverlapReports" / "61_100" / "overlap_ellipses.csv",
        help="Path to overlap_ellipses.csv",
    )
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotation tool") / "Annotations",
        help="Root folder for annotation JSON files",
    )
    parser.add_argument(
        "--src-collages",
        type=Path,
        default=Path("annotation tool") / "Collages" / "disagreement_61_100",
        help="Source folder containing collage_pair*.png",
    )
    parser.add_argument(
        "--out-collages",
        type=Path,
        default=Path("annotation tool") / "Collages" / "disagreement_61_100_highlighted",
        help="Output folder for highlighted collages",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=["Avi", "Benny", "Sigal", "Smadar"],
        help="Person column order used in collage generation",
    )
    parser.add_argument("--person-rows", type=int, default=1, help="Number of person rows in collage grid.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--pad", type=int, default=4)
    parser.add_argument("--header", type=int, default=44)
    parser.add_argument("--image-canvas-size", type=int, default=792)
    args = parser.parse_args()

    isolated_entries = read_isolated_entries(args.overlap_csv, args.annotations_root)
    args.out_collages.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path in sorted(args.src_collages.glob("collage_pair*.png")):
        pair_num = pair_from_name(img_path)
        if pair_num is None:
            continue

        entries_by_person = isolated_entries.get(pair_num, {})
        with Image.open(img_path) as img:
            highlighted = _draw_pair_highlights(
                img,
                args.persons,
                entries_by_person,
                person_rows=args.person_rows,
                width=args.width,
                height=args.height,
                pad=args.pad,
                header_h=args.header,
                image_canvas_size=args.image_canvas_size,
            )
            out_path = args.out_collages / img_path.name
            highlighted.save(out_path)
            saved += 1

    print(f"Saved {saved} highlighted collages to: {args.out_collages.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
