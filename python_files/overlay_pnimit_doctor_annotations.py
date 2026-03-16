"""Overlay one doctor's ellipse annotations on PNIMIT model prediction images.

This script matches annotation pair folders (e.g., pair_A4_1_2) to prediction folders
under predictions_pnimit and renders a combined image with:
- current CXR (grayscale)
- model output heatmap (from output.nii.gz)
- doctor ellipse outlines + compact labels

Default usage (from python_files):
    python overlay_pnimit_doctor_annotations.py

Example custom usage:
    python overlay_pnimit_doctor_annotations.py \
        --annotations-root "annotation tool/pair_A4_1_2" \
        --predictions-root "predictions_pnimit" \
        --out-name "model_output_with_doctor.png"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EllipseAnn:
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float
    label: str
    tag: str
    tag_other: str


def _safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _load_annotation(json_path: Path) -> tuple[str | None, str | None, list[EllipseAnn]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Unexpected annotation JSON format: {json_path}")

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
                cx=float(item.get("cx", 0.0)),
                cy=float(item.get("cy", 0.0)),
                rx=float(item.get("rx", 0.0)),
                ry=float(item.get("ry", 0.0)),
                angle_deg=float(item.get("angle", 0.0)),
                label=str(item.get("label", "")),
                tag=str(item.get("tag", "")),
                tag_other=str(item.get("tag_other", "")),
            )
        )

    return prior_name, current_name, ellipses


def _pick_annotation_json(pair_ann_dir: Path) -> Path | None:
    jsons = sorted(pair_ann_dir.glob("*.json"))
    if not jsons:
        return None
    return jsons[0]


def _norm_pair_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _find_json_for_pair(annotations_root: Path, pair_name: str) -> Path | None:
    # 1) Preferred: JSON inside pair directory (annotation tool export convention).
    pair_dir = annotations_root / pair_name
    if pair_dir.is_dir():
        p = _pick_annotation_json(pair_dir)
        if p is not None:
            return p

    # 2) Try root-level JSON files named after the pair (possibly with different separators/case).
    target_norm = _norm_pair_token(pair_name)
    pair_parts = pair_name.lower().split("_")
    if len(pair_parts) >= 4:
        compact_variant = f"pair{pair_parts[1]}{pair_parts[2]}{pair_parts[3]}"
    else:
        compact_variant = pair_name.lower().replace("_", "")

    for json_path in sorted(annotations_root.glob("*.json")):
        stem_norm = _norm_pair_token(json_path.stem)
        if stem_norm == target_norm or stem_norm.endswith(compact_variant) or compact_variant in stem_norm:
            return json_path

    # 3) Fallback: recursive scan and compare normalized file stem with target pair token.
    for json_path in sorted(annotations_root.rglob("*.json")):
        stem_norm = _norm_pair_token(json_path.stem)
        if stem_norm == target_norm or target_norm in stem_norm or stem_norm in target_norm:
            return json_path

    return None


def _load_nii_2d_for_display(nii_path: Path, out_size: int = 792) -> np.ndarray:
    arr = nib.load(str(nii_path)).get_fdata().T
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    arr = arr.astype(np.float32)

    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = (arr - mn) / (mx - mn)

    img_u8 = (np.clip(norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_im = Image.fromarray(img_u8, mode="L").resize((out_size, out_size), Image.Resampling.BILINEAR)
    return np.asarray(pil_im, dtype=np.float32) / 255.0


def _choose_current_nii(pair_ann_dir: Path, expected_current_name: str | None) -> Path:
    nii_files = sorted([p for p in pair_ann_dir.iterdir() if p.name.endswith(".nii.gz")])
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected at least 2 .nii.gz files in {pair_ann_dir}")

    if expected_current_name:
        for p in nii_files:
            if _safe_stem(p) == expected_current_name:
                return p

    # Matches annotation tool convention.
    return nii_files[1]


def _generate_alpha_map(x: np.ndarray) -> np.ndarray:
    x_abs = np.abs(x)
    max_val = max(float(np.max(x_abs)), 0.07)
    return x_abs / max_val


def _prediction_cmap() -> colors.LinearSegmentedColormap:
    return colors.LinearSegmentedColormap.from_list(
        "prediction_diff",
        (
            (0.000, (0.235, 1.000, 0.239)),
            (0.400, (0.000, 1.000, 0.702)),
            (0.500, (1.000, 0.988, 0.988)),
            (0.600, (1.000, 0.604, 0.000)),
            (1.000, (0.682, 0.000, 0.000)),
        ),
    )


def _draw_rotated_ellipse(ax: plt.Axes, ann: EllipseAnn, color: tuple[float, float, float]) -> None:
    a = math.radians(ann.angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)

    pts_x: list[float] = []
    pts_y: list[float] = []
    steps = 120
    for i in range(steps + 1):
        t = 2.0 * math.pi * (i / steps)
        x = ann.rx * math.cos(t)
        y = ann.ry * math.sin(t)
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        pts_x.append(ann.cx + xr)
        pts_y.append(ann.cy + yr)

    ax.plot(pts_x, pts_y, color=color, linewidth=2.5)


def _label_text(ann: EllipseAnn) -> str:
    if ann.tag:
        if ann.tag.lower() == "other" and ann.tag_other:
            return f"{ann.label} | {ann.tag_other}"
        return f"{ann.label} | {ann.tag}"
    return ann.label


def _annotation_color(label: str) -> tuple[float, float, float]:
    mapping = {
        "Appearance": (1.0, 0.0, 1.0),
        "Disappearance": (0.2, 0.6, 1.0),
        "Persistence": (1.0, 1.0, 0.0),
    }
    return mapping.get(label, (1.0, 1.0, 1.0))


def _render_overlay(
    current_img: np.ndarray,
    pred_map: np.ndarray,
    ellipses: list[EllipseAnn],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.imshow(current_img, cmap="gray", origin="upper")

    alphas = _generate_alpha_map(pred_map)
    divnorm = colors.TwoSlopeNorm(vmin=min(float(np.min(pred_map)), -0.01), vcenter=0.0, vmax=max(float(np.max(pred_map)), 0.01))
    imm = ax.imshow(pred_map, alpha=alphas, cmap=_prediction_cmap(), norm=divnorm, origin="upper")
    fig.colorbar(imm, fraction=0.05, pad=0.04)

    for ann in ellipses:
        color = _annotation_color(ann.label)
        _draw_rotated_ellipse(ax, ann, color)
        tx = ann.cx - ann.rx
        ty = ann.cy - ann.ry - 10
        ax.text(
            tx,
            ty,
            _label_text(ann),
            fontsize=7,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _iter_prediction_pair_dirs(predictions_root: Path) -> list[Path]:
    return [p for p in sorted(predictions_root.iterdir()) if p.is_dir() and p.name.lower().startswith("pair_")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay doctor annotations on PNIMIT model outputs.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=Path("annotation tool") / "pair_A4_1_2",
        help="Root containing pair annotation folders (pair_A*_*/).",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("predictions_pnimit"),
        help="Root containing prediction folders (pair_A*_*/output.nii.gz).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="model_output_with_doctor.png",
        help="Output file name for each prediction pair folder.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Optional separate output root. If omitted, saves in each prediction pair folder.",
    )
    args = parser.parse_args(argv)

    ann_root: Path = args.annotations_root
    pred_root: Path = args.predictions_root

    if not ann_root.exists():
        raise FileNotFoundError(f"Annotations root not found: {ann_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Predictions root not found: {pred_root}")

    done = 0
    skipped = 0

    for pred_pair_dir in _iter_prediction_pair_dirs(pred_root):
        pair_name = pred_pair_dir.name
        pred_map_path = pred_pair_dir / "output.nii.gz"

        if not pred_map_path.exists():
            print(f"[SKIP] Missing prediction output for {pair_name}")
            skipped += 1
            continue

        pair_ann_dir = ann_root / pair_name
        json_path = _find_json_for_pair(ann_root, pair_name)
        if json_path is None:
            print(f"[SKIP] No JSON annotation for {pair_name}")
            skipped += 1
            continue

        try:
            _, current_name, ellipses = _load_annotation(json_path)
            if not pair_ann_dir.exists():
                raise FileNotFoundError(f"Missing annotation pair folder with NIfTI files: {pair_ann_dir}")
            current_nii = _choose_current_nii(pair_ann_dir, current_name)
            current_img = _load_nii_2d_for_display(current_nii, out_size=792)

            pred_map = nib.load(str(pred_map_path)).get_fdata().T
            if pred_map.ndim == 3:
                pred_map = pred_map[:, :, pred_map.shape[2] // 2]
            pred_map = pred_map.astype(np.float32)

            # Keep prediction map aligned with the annotation tool canvas.
            pred_map_resized = np.asarray(
                Image.fromarray(pred_map).resize((792, 792), Image.Resampling.BILINEAR), dtype=np.float32
            )

            if args.out_root is None:
                out_path = pred_pair_dir / args.out_name
            else:
                out_path = args.out_root / pair_ann_dir.name / args.out_name

            _render_overlay(
                current_img=current_img,
                pred_map=pred_map_resized,
                ellipses=ellipses,
                out_path=out_path,
                title="Model output + Doctor annotation",
            )
            done += 1
            print(f"[OK] {pair_name} -> {out_path}")
        except Exception as e:
            print(f"[SKIP] {pair_name}: {e}")
            skipped += 1

    print(f"Done. Created {done} overlays. Skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
