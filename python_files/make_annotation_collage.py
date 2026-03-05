"""Create a PowerPoint-slide-friendly collage from exported annotated images.

Assumes you already exported per-pair images using:
    python export_annotated_images.py --layout by_pair ...

Input layout (default):
    annotation tool/AnnotatedExports/pair<PAIR>/<PERSON>.png

Output:
    A single PNG sized for a 16:9 slide (default 1920x1080).

Example:
    python make_annotation_collage.py --pairs 32 33 35 36 37

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import nibabel as nib
import numpy as np


DEFAULT_PERSONS = ["Avi", "Benny", "Nitzan", "Sigal", "Smadar"]


def find_pair_dir(pairs_root: Path, pair_number: int) -> Path | None:
    for sub in sorted(pairs_root.glob("Pairs*")):
        if not sub.is_dir():
            continue
        candidate = sub / f"pair{pair_number}"
        if candidate.is_dir():
            return candidate
    return None


def choose_prior_current_nii(pair_dir: Path) -> tuple[Path, Path]:
    nii_files = sorted([p for p in pair_dir.iterdir() if p.name.endswith(".nii.gz")])
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected 2 .nii.gz files in {pair_dir}, found {len(nii_files)}")
    return nii_files[0], nii_files[1]


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


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        # Default bitmap font (always available)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _fit_into(img: Image.Image, w: int, h: int) -> Image.Image:
    if w <= 0 or h <= 0:
        return img
    im = img.copy()
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = (w - im.width) // 2
    y = (h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Make a PPT-friendly collage of annotated images.")
    parser.add_argument(
        "--in-root",
        type=Path,
        default=Path("annotation tool") / "AnnotatedExports",
        help="Root folder with per-pair exported images.",
    )
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("annotation tool"),
        help="Root folder containing Pairs1, Pairs2, ... (used to load prior/current scans).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=int,
        required=True,
        help="Pair numbers (e.g., 32 33 35 36 37)",
    )
    parser.add_argument(
        "--one-per-slide",
        action="store_true",
        help="If set, generates one PNG per pair (recommended for slides).",
    )
    parser.add_argument(
        "--raw-ab-only",
        action="store_true",
        help="If set, ignores annotations and renders only A(prior) and B(current) raw scans for each pair.",
    )
    parser.add_argument(
        "--include-ab",
        action="store_true",
        help="Include A (prior) and B (current) per person. A is the prior scan, B is the current scan (annotated PNG if exists).",
    )
    parser.add_argument(
        "--persons",
        nargs="*",
        default=None,
        help="Persons/columns to include (default: Avi Benny Nitzan Sigal Smadar)",
    )
    parser.add_argument(
        "--person-rows",
        type=int,
        default=1,
        help="Arrange persons into this many rows (default 1). For 5 persons, 2 gives a 3x2 grid.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: annotation tool/Collages/collage_pairs_<...>.png)",
    )
    parser.add_argument("--width", type=int, default=1920, help="Output width in pixels (16:9 default 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output height in pixels (16:9 default 1080)")
    parser.add_argument("--pad", type=int, default=12, help="Padding between cells")
    parser.add_argument("--header", type=int, default=60, help="Header row height for column labels")
    parser.add_argument("--rowlabel", type=int, default=120, help="Left column width for row labels")
    parser.add_argument(
        "--minimal-gap",
        action="store_true",
        help="Use tighter padding and labels for slide usage.",
    )

    args = parser.parse_args(argv)

    in_root: Path = args.in_root
    pairs_root: Path = args.pairs_root
    pairs: list[int] = list(args.pairs)
    persons: list[str] = list(args.persons) if args.persons else list(DEFAULT_PERSONS)

    W, H = int(args.width), int(args.height)
    pad = int(args.pad)
    header_h = int(args.header)
    rowlabel_w = int(args.rowlabel)

    if args.minimal_gap:
        pad = min(pad, 4)
        header_h = min(header_h, 44)
        rowlabel_w = min(rowlabel_w, 80)

    out_dir = Path("annotation tool") / "Collages"
    out_dir.mkdir(parents=True, exist_ok=True)

    def render_raw_ab(pair: int, out_path: Path) -> None:
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = _load_font(14)

        pair_dir = find_pair_dir(pairs_root, int(pair))
        prior_im: Image.Image | None = None
        current_im: Image.Image | None = None
        if pair_dir is not None:
            prior_nii, current_nii = choose_prior_current_nii(pair_dir)
            prior_im = load_nii_as_pil_gray(prior_nii).convert("RGB")
            current_im = load_nii_as_pil_gray(current_nii).convert("RGB")

        # Layout: two panels side-by-side
        effective_pad = pad
        if args.minimal_gap:
            effective_pad = min(effective_pad, 4)

        header = 0
        left = effective_pad
        top = effective_pad + header
        panel_w = (W - effective_pad * 3) // 2
        panel_h = H - effective_pad * 2 - header

        def paste_panel(x0: int, img: Image.Image | None, label: str) -> None:
            x1 = x0 + panel_w
            y0 = top
            y1 = y0 + panel_h
            draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))
            if img is not None:
                fitted = _fit_into(img, panel_w, panel_h)
                canvas.paste(fitted, (x0, y0))
            else:
                missing = "missing"
                bbox = draw.textbbox((0, 0), missing, font=font)
                mx = x0 + (panel_w - (bbox[2] - bbox[0])) // 2
                my = y0 + (panel_h - (bbox[3] - bbox[1])) // 2
                draw.text((mx, my), missing, fill=(90, 90, 90), font=font)

            # Label strip
            draw.rectangle((x0, y0, x0 + 28, y0 + 18), fill=(255, 255, 255))
            draw.text((x0 + 6, y0 + 2), label, fill=(0, 0, 0), font=font)

        paste_panel(left, prior_im, "A")
        paste_panel(left + panel_w + effective_pad, current_im, "B")

        # Pair label
        draw.text((8, 8), f"pair{pair}", fill=(0, 0, 0), font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        print(f"Saved collage: {out_path.resolve()}")

    def render(pairs_for_canvas: list[int], out_path: Path) -> None:
        rows = len(pairs_for_canvas)
        person_rows = max(1, int(args.person_rows))
        person_cols = int(math.ceil(len(persons) / person_rows))
        ab_cols = 2 if args.include_ab else 1
        grid_cols = person_cols * ab_cols

        # If single-row slide, drop the left row label column to maximize space.
        effective_rowlabel_w = 0 if rows == 1 else rowlabel_w

        # When we wrap persons into multiple rows, we keep the header only as a small title strip.
        effective_header_h = header_h
        show_column_headers = person_rows == 1
        if not show_column_headers:
            effective_header_h = min(effective_header_h, 44)

        grid_w = W - effective_rowlabel_w - pad * (grid_cols + 1)
        grid_h = H - effective_header_h - pad * (rows * person_rows + 1)
        cell_w = max(1, grid_w // grid_cols)
        cell_h = max(1, grid_h // (rows * person_rows))

        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        font = _load_font(14)

        # Column headers (only when persons are in a single row)
        if show_column_headers:
            for j, person in enumerate(persons):
                x0 = effective_rowlabel_w + pad + (j * ab_cols) * (cell_w + pad)
                x1 = x0 + (ab_cols * cell_w) + ((ab_cols - 1) * pad)
                y0 = 0
                y1 = effective_header_h
                draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))
                text = person
                bbox = draw.textbbox((0, 0), text, font=font)
                tx = x0 + ((x1 - x0) - (bbox[2] - bbox[0])) // 2
                ty = y0 + (effective_header_h - (bbox[3] - bbox[1])) // 2
                draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

        # Rows (pairs) + images
        for i, pair in enumerate(pairs_for_canvas):
            base_y0 = effective_header_h + pad + i * (person_rows * cell_h + pad)
            base_y1 = base_y0 + person_rows * cell_h

            prior_im: Image.Image | None = None
            current_base_im: Image.Image | None = None
            if args.include_ab:
                pair_dir = find_pair_dir(pairs_root, int(pair))
                if pair_dir is not None:
                    try:
                        prior_nii, current_nii = choose_prior_current_nii(pair_dir)
                        prior_im = load_nii_as_pil_gray(prior_nii).convert("RGB")
                        current_base_im = load_nii_as_pil_gray(current_nii).convert("RGB")
                    except Exception:
                        prior_im = None
                        current_base_im = None

            if rows > 1:
                draw.rectangle((0, base_y0, effective_rowlabel_w, base_y1), fill=(255, 255, 255))
                row_text = f"pair{pair}"
                bbox = draw.textbbox((0, 0), row_text, font=font)
                tx = (effective_rowlabel_w - (bbox[2] - bbox[0])) // 2
                ty = base_y0 + ((person_rows * cell_h) - (bbox[3] - bbox[1])) // 2
                draw.text((tx, ty), row_text, fill=(0, 0, 0), font=font)
            else:
                # Single-row: put the pair label in the top-left corner.
                title = f"pair{pair}"
                draw.text((8, 8), title, fill=(0, 0, 0), font=font)

            for idx, person in enumerate(persons):
                pr = idx // person_cols
                pc = idx % person_cols
                y0 = base_y0 + pr * cell_h
                y1 = y0 + cell_h

                base_x = effective_rowlabel_w + pad + (pc * ab_cols) * (cell_w + pad)

                def paste_cell(col_offset: int, img: Image.Image | None, label: str) -> None:
                    x0 = base_x + col_offset * (cell_w + pad)
                    x1 = x0 + cell_w
                    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255))

                    if img is None:
                        missing = "missing"
                        bbox = draw.textbbox((0, 0), missing, font=font)
                        mx = x0 + (cell_w - (bbox[2] - bbox[0])) // 2
                        my = y0 + (cell_h - (bbox[3] - bbox[1])) // 2
                        draw.text((mx, my), missing, fill=(90, 90, 90), font=font)
                    else:
                        fitted = _fit_into(img, cell_w, cell_h)
                        canvas.paste(fitted, (x0, y0))

                    # In-cell minimal label
                    if not show_column_headers or args.include_ab:
                        draw.rectangle((x0, y0, x0 + 88, y0 + 18), fill=(255, 255, 255))
                        draw.text((x0 + 4, y0 + 2), label, fill=(0, 0, 0), font=font)

                if args.include_ab:
                    # A: prior scan
                    paste_cell(0, prior_im, f"{person} A")

                    # B: current scan (annotated per person if exists)
                    img_path = in_root / f"pair{pair}" / f"{person}.png"
                    if img_path.exists():
                        b_img = Image.open(img_path).convert("RGB")
                    else:
                        b_img = current_base_im
                    paste_cell(1, b_img, f"{person} B")
                else:
                    img_path = in_root / f"pair{pair}" / f"{person}.png"
                    im = Image.open(img_path).convert("RGB") if img_path.exists() else None
                    paste_cell(0, im, person)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        print(f"Saved collage: {out_path.resolve()}")

    if args.one_per_slide:
        for pair in pairs:
            if args.raw_ab_only:
                out_path = out_dir / f"collage_pair{pair}_rawAB.png"
                render_raw_ab(pair, out_path)
            else:
                out_path = out_dir / f"collage_pair{pair}.png"
                render([pair], out_path)
        return 0

    # Single collage containing all pairs
    if args.out is not None:
        out_path = args.out
    else:
        out_path = out_dir / ("collage_pairs_" + "_".join(str(p) for p in pairs) + ".png")
    render(pairs, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
