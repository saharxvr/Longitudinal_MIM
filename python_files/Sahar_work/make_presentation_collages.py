from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parent.parent))

from export_annotated_images import (
    draw_annotation_overlay,
    find_pair_dir,
    load_annotation,
    load_nii_as_pil_gray,
    parse_pair_number,
)


PERSONS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]


def _pair_num_from_name(name: str) -> int | None:
    m = re.match(r"^pair(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))


def _iter_pair_nums(ov_plots_root: Path) -> list[int]:
    nums: list[int] = []
    if not ov_plots_root.exists():
        return nums
    for p in ov_plots_root.iterdir():
        if p.is_dir():
            n = _pair_num_from_name(p.name)
            if n is not None:
                nums.append(n)
    return sorted(set(nums))


def _fit_image(img: Image.Image, target_w: int, target_h: int, bg=(255, 255, 255)) -> Image.Image:
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        return Image.new("RGB", (target_w, target_h), bg)

    scale = min(target_w / float(src_w), target_h / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    fitted = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    x = (target_w - fitted.width) // 2
    y = (target_h - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _load_font(size: int = 56) -> ImageFont.ImageFont:
    # Prefer TrueType fonts for presentation readability.
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _choose_current_non_seg_nii(pair_dir: Path, expected_current_name: str | None) -> Path:
    non_seg = sorted(
        [p for p in pair_dir.iterdir() if p.name.endswith(".nii.gz") and not p.name.endswith("_lung_seg.nii.gz")]
    )
    if len(non_seg) < 2:
        raise FileNotFoundError(f"Expected at least 2 non-seg .nii.gz files in {pair_dir}, found {len(non_seg)}")

    if expected_current_name:
        for p in non_seg:
            if _safe_stem(p) == expected_current_name:
                return p

    # Matches annotation workflow: second sorted non-seg image is current (B)
    return non_seg[1]


def _index_annotations(annotations_root: Path) -> dict[str, dict[int, Path]]:
    ann_idx: dict[str, dict[int, Path]] = {p: {} for p in PERSONS}
    for person in PERSONS:
        person_dir = annotations_root / person
        if not person_dir.exists():
            continue
        for ann_path in sorted(person_dir.rglob("*.json")):
            n = parse_pair_number(ann_path)
            if n is not None:
                ann_idx[person][n] = ann_path
    return ann_idx


def _build_prior_current_collage(prior: Image.Image, current: Image.Image, canvas_w: int, canvas_h: int, pad: int) -> Image.Image:
    font = _load_font(60)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_h = 0
    panel_w = (canvas_w - 3 * pad) // 2
    panel_h = canvas_h - title_h - 2 * pad

    prior_fit = _fit_image(prior, panel_w, panel_h)
    current_fit = _fit_image(current, panel_w, panel_h)

    x0 = pad
    y0 = title_h + pad
    canvas.paste(prior_fit, (x0, y0))
    canvas.paste(current_fit, (x0 + panel_w + pad, y0))

    # Draw labels last (on top) with red fill and red border stroke.
    draw.text((x0 + 6, y0 + 6), "Prior", fill=(255, 0, 0), font=font, stroke_width=3, stroke_fill=(180, 0, 0))
    draw.text(
        (x0 + panel_w + pad + 6, y0 + 6),
        "Current",
        fill=(255, 0, 0),
        font=font,
        stroke_width=3,
        stroke_fill=(180, 0, 0),
    )
    return canvas


def _build_annotators_model_collage(
    tiles: list[tuple[str, Image.Image | None]],
    canvas_w: int,
    canvas_h: int,
    pad: int,
) -> Image.Image:
    font = _load_font(52)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    rows, cols = 2, 3
    title_h = 0
    label_h = 0
    cell_w = (canvas_w - (cols + 1) * pad) // cols
    cell_h = (canvas_h - title_h - (rows + 1) * pad) // rows
    image_h = max(1, cell_h - label_h)

    for idx, (label, tile) in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (cell_w + pad)
        y0 = title_h + pad + r * (cell_h + pad)

        if tile is None:
            block = Image.new("RGB", (cell_w, image_h), (245, 245, 245))
            d2 = ImageDraw.Draw(block)
            d2.text((10, 10), "missing", fill=(100, 100, 100), font=font)
        else:
            block = _fit_image(tile, cell_w, image_h)
        canvas.paste(block, (x0, y0))
        # Draw labels last (on top) with red fill and red border stroke.
        draw.text((x0 + 6, y0 + 6), label, fill=(255, 0, 0), font=font, stroke_width=3, stroke_fill=(180, 0, 0))

    return canvas


def build_pair_collages(args: argparse.Namespace) -> None:
    pairs_root = Path(args.pairs_root)
    annotations_root = Path(args.annotations_root)
    ov_plots_root = Path(args.ov_plots_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_idx = _index_annotations(annotations_root)

    if args.pairs:
        pair_nums = sorted(set(args.pairs))
    else:
        pair_nums = _iter_pair_nums(ov_plots_root)

    if not pair_nums:
        raise RuntimeError("No pairs found. Pass --pairs or provide an ov plots folder with pair directories.")

    generated = 0
    for pair_num in pair_nums:
        pair_dir = find_pair_dir(pairs_root, pair_num)
        if pair_dir is None:
            continue

        # Collage 1: prior/current at max size.
        try:
            nii_files = sorted([p for p in pair_dir.iterdir() if p.name.endswith(".nii.gz") and not p.name.endswith("_lung_seg.nii.gz")])
            if len(nii_files) < 2:
                continue
            prior_nii = nii_files[0]
            current_nii = _choose_current_non_seg_nii(pair_dir, None)

            prior_im = load_nii_as_pil_gray(prior_nii).convert("RGB")
            current_im = load_nii_as_pil_gray(current_nii).convert("RGB")
            prior_current = _build_prior_current_collage(prior_im, current_im, args.canvas_width, args.canvas_height, args.pad)
            prior_current.save(out_dir / f"pair{pair_num}_prior_current.png")
        except Exception:
            continue

        # Collage 2: 5 annotators + model output (full image overlays), 2x3.
        tiles: list[tuple[str, Image.Image | None]] = []
        for person in PERSONS:
            ann_path = ann_idx.get(person, {}).get(pair_num)
            if ann_path is None:
                tiles.append((person, None))
                continue
            try:
                _, current_name, ellipses = load_annotation(ann_path)
                cur_nii_for_person = _choose_current_non_seg_nii(pair_dir, current_name)
                base = load_nii_as_pil_gray(cur_nii_for_person).convert("RGB")
                overlay = draw_annotation_overlay(base, ellipses)
                tiles.append((person, overlay))
            except Exception:
                tiles.append((person, None))

        model_path = ov_plots_root / f"pair{pair_num}" / "model_on_original.png"
        model_img: Image.Image | None = None
        if model_path.exists():
            try:
                model_img = Image.open(model_path).convert("RGB")
            except Exception:
                model_img = None
        tiles.append(("Model", model_img))

        ann_model = _build_annotators_model_collage(tiles, args.canvas_width, args.canvas_height, args.pad)
        ann_model.save(out_dir / f"pair{pair_num}_annotators_model.png")
        generated += 1

    print(f"Generated 2 collages per pair for {generated} pairs in: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-pair presentation collages.")
    p.add_argument("--pairs-root", default="annotation tool", help="Root containing Pairs1..PairsN")
    p.add_argument("--annotations-root", default="annotation tool/Annotations", help="Root containing annotation JSONs")
    p.add_argument("--ov-plots-root", default="Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/plots", help="Root containing pair plot folders with model_on_original.png")
    p.add_argument("--out-dir", default="Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/per_pair_collages", help="Output directory")
    p.add_argument("--pairs", nargs="*", type=int, default=None, help="Optional explicit pair numbers")
    p.add_argument("--canvas-width", type=int, default=3840)
    p.add_argument("--canvas-height", type=int, default=2160)
    p.add_argument("--pad", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    build_pair_collages(parse_args())
