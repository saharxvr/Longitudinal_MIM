"""Per-pair collages with the 5 physicians only, organized by disagreement level.

For each pair, builds a single collage containing the 5 physician annotation
overlays at the maximum size (no model column). Output is grouped into one
subdirectory per disagreement level using the values produced by
``get_disagreement_levels.py``.

Reuses the rendering helpers from ``make_presentation_collages.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from make_presentation_collages import (  # type: ignore[import-not-found]
    PERSONS,
    _build_annotators_model_collage,
    _choose_current_non_seg_nii,
    _index_annotations,
)
from export_annotated_images import (  # type: ignore[import-not-found]
    EllipseAnn,
    draw_annotation_overlay,
    find_pair_dir,
    load_annotation,
    load_nii_as_pil_gray,
)

# OV mapping rule (matches Observer_Variability_*.py exactly).
# Ellipses that map to 0 (Persistence with no size and no intensity change)
# are not counted in pos or neg consensus, so we should not draw them either.
_LABEL_MAP_RULES = {
    ('Appearance', None, None): 3,
    ('Disappearance', None, None): -3,
    ('Persistence', 'Increase', 'Increase'): 2,
    ('Persistence', 'Decrease', 'Decrease'): -2,
    ('Persistence', 'Increase', 'None'): 1,
    ('Persistence', 'Decrease', 'None'): -1,
    ('Persistence', 'None', 'Increase'): 1,
    ('Persistence', 'None', 'Decrease'): -1,
    ('Persistence', 'None', 'None'): 0,
    ('Persistence', 'Increase', 'Decrease'): (1, -1),
    ('Persistence', 'Decrease', 'Increase'): (1, -1),
}


def _filter_ov_counted_ellipses(ellipses: list[EllipseAnn]) -> list[EllipseAnn]:
    """Keep only ellipses that the OV mapping actually counts (mapped != 0).

    This drops Persistence with size_change=None and intensity_change=None,
    matching the skip behavior in load_labels_map of the OV scripts.
    """
    kept: list[EllipseAnn] = []
    for ann in ellipses:
        if ann.label == 'Persistence':
            s = ann.size_change
            i = ann.intensity_change
        else:
            s = None
            i = None
        c = _LABEL_MAP_RULES.get((ann.label, s, i))
        if c is None or c == 0:
            continue
        kept.append(ann)
    return kept


def _build_doctors_only_collage(
    tiles: list[tuple[str, Image.Image | None]],
    canvas_w: int,
    canvas_h: int,
    pad: int,
) -> Image.Image:
    """Render 5 physician overlays in a 2x3 grid (last cell intentionally blank).

    Reuses the existing 2x3 layout helper to keep visual style identical to the
    per_pair_collages output (max size, padding, label style).
    """
    layout_tiles: list[tuple[str, Image.Image | None]] = list(tiles)
    while len(layout_tiles) < 6:
        layout_tiles.append(("", None))
    return _build_annotators_model_collage(layout_tiles, canvas_w, canvas_h, pad)


def _load_disagreement_levels(csv_path: Path, level_col: str) -> dict[int, int]:
    df = pd.read_csv(csv_path)
    if level_col not in df.columns:
        raise KeyError(
            f"Column '{level_col}' not found in {csv_path}. Available: {list(df.columns)}"
        )
    return {int(row['pair']): int(row[level_col]) for _, row in df.iterrows()}


def build_pair_collages_by_level(args: argparse.Namespace) -> None:
    pairs_root = Path(args.pairs_root)
    annotations_root = Path(args.annotations_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_idx = _index_annotations(annotations_root)
    levels_by_pair = _load_disagreement_levels(Path(args.levels_csv), args.level_column)

    if args.pairs:
        pair_nums = sorted(set(args.pairs))
    else:
        pair_nums = sorted(levels_by_pair.keys())

    generated = 0
    skipped: list[tuple[int, str]] = []

    for pair_num in pair_nums:
        if pair_num not in levels_by_pair:
            skipped.append((pair_num, 'no level'))
            continue

        pair_dir = find_pair_dir(pairs_root, pair_num)
        if pair_dir is None:
            skipped.append((pair_num, 'pair dir missing'))
            continue

        tiles: list[tuple[str, Image.Image | None]] = []
        for person in PERSONS:
            ann_path = ann_idx.get(person, {}).get(pair_num)
            if ann_path is None:
                tiles.append((person, None))
                continue
            try:
                _, current_name, ellipses = load_annotation(ann_path)
                ellipses = _filter_ov_counted_ellipses(ellipses)
                cur_nii_for_person = _choose_current_non_seg_nii(pair_dir, current_name)
                base = load_nii_as_pil_gray(cur_nii_for_person).convert("RGB")
                overlay = draw_annotation_overlay(base, ellipses)
                tiles.append((person, overlay))
            except Exception as e:
                tiles.append((person, None))
                print(f"[warn] pair{pair_num} {person}: {e}")

        if all(tile is None for _, tile in tiles):
            skipped.append((pair_num, 'no doctor overlays'))
            continue

        collage = _build_doctors_only_collage(
            tiles, args.canvas_width, args.canvas_height, args.pad
        )

        level = levels_by_pair[pair_num]
        level_dir = out_dir / f"disagreement_level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        collage.save(level_dir / f"pair{pair_num}_doctors.png")
        generated += 1

    print(f"Generated {generated} collages in: {out_dir.resolve()}")
    if skipped:
        print(f"Skipped {len(skipped)} pairs:")
        for pn, reason in skipped:
            print(f"  pair{pn}: {reason}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pairs-root",
        default="annotation tool",
        help="Root containing Pairs1..PairsN",
    )
    p.add_argument(
        "--annotations-root",
        default="annotation tool/Annotations",
        help="Root containing annotation JSONs",
    )
    p.add_argument(
        "--levels-csv",
        default="Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/disagreement_levels.csv",
        help="CSV produced by get_disagreement_levels.py",
    )
    p.add_argument(
        "--level-column",
        default="all_disagreement_level",
        help="Column from the CSV to use for grouping (e.g. all_/pos_/neg_disagreement_level)",
    )
    p.add_argument(
        "--out-dir",
        default="Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/per_pair_collages_doctors_by_disagreement",
        help="Output directory; one subdir per disagreement level will be created",
    )
    p.add_argument(
        "--pairs",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit pair numbers (default: all from levels CSV)",
    )
    p.add_argument("--canvas-width", type=int, default=3840)
    p.add_argument("--canvas-height", type=int, default=2160)
    p.add_argument("--pad", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    build_pair_collages_by_level(parse_args())
