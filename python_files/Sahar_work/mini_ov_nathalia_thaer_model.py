from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class EllipseAnn:
    cx: float
    cy: float
    rx: float
    ry: float
    angle_deg: float
    label: str
    tag: str


@dataclass(frozen=True)
class ModelComponent:
    ys: slice
    xs: slice
    mask: np.ndarray


def load_ellipses(json_path: Path) -> list[EllipseAnn]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        return []
    out: list[EllipseAnn] = []
    for item in data[1:]:
        if not isinstance(item, dict):
            continue
        out.append(
            EllipseAnn(
                cx=float(item.get("cx", 0.0)),
                cy=float(item.get("cy", 0.0)),
                rx=float(item.get("rx", 0.0)),
                ry=float(item.get("ry", 0.0)),
                angle_deg=float(item.get("angle", 0.0)),
                label=str(item.get("label", "")),
                tag=str(item.get("tag", "")),
            )
        )
    return out


def rasterize_ellipse(ann: EllipseAnn, h: int, w: int) -> tuple[tuple[slice, slice], np.ndarray]:
    if ann.rx <= 0 or ann.ry <= 0:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), dtype=bool)

    rmax = max(ann.rx, ann.ry)
    x0 = max(0, int(np.floor(ann.cx - rmax - 2)))
    x1 = min(w, int(np.ceil(ann.cx + rmax + 3)))
    y0 = max(0, int(np.floor(ann.cy - rmax - 2)))
    y1 = min(h, int(np.ceil(ann.cy + rmax + 3)))
    if x0 >= x1 or y0 >= y1:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), dtype=bool)

    ys, xs = np.ogrid[y0:y1, x0:x1]
    x = xs.astype(np.float64) - ann.cx
    y = ys.astype(np.float64) - ann.cy

    t = np.deg2rad(ann.angle_deg)
    ct = np.cos(t)
    st = np.sin(t)
    xr = ct * x + st * y
    yr = -st * x + ct * y
    roi = (xr * xr) / (ann.rx * ann.rx) + (yr * yr) / (ann.ry * ann.ry) <= 1.0
    return (slice(y0, y1), slice(x0, x1)), roi


def build_doc_mask(ellipses: list[EllipseAnn], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for ann in ellipses:
        (ys, xs), roi = rasterize_ellipse(ann, h, w)
        if roi.size:
            m[ys, xs] |= roi
    return m


def load_model_map(pred_pair_dir: Path, h: int, w: int) -> np.ndarray:
    out_path = pred_pair_dir / "output.nii.gz"
    if not out_path.exists():
        return np.zeros((h, w), dtype=np.float32)

    arr = nib.load(str(out_path)).get_fdata().T.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]

    if arr.shape != (h, w):
        arr_img = Image.fromarray(arr, mode="F")
        arr = np.asarray(arr_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)

    return arr


def model_components(binary_mask: np.ndarray) -> list[ModelComponent]:
    lbl, n = ndimage.label(binary_mask)
    if n <= 0:
        return []

    out: list[ModelComponent] = []
    for idx in range(1, n + 1):
        ys, xs = np.where(lbl == idx)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        roi = lbl[y0:y1, x0:x1] == idx
        out.append(ModelComponent(slice(y0, y1), slice(x0, x1), roi))
    return out


def directional_doc_vs_mask(ellipses: list[EllipseAnn], dst_mask: np.ndarray, h: int, w: int) -> tuple[int, int]:
    agree = 0
    disagree = 0
    for ann in ellipses:
        (ys, xs), roi = rasterize_ellipse(ann, h, w)
        if roi.size and int((roi & dst_mask[ys, xs]).sum()) >= 1:
            agree += 1
        else:
            disagree += 1
    return agree, disagree


def directional_components_vs_mask(components: list[ModelComponent], dst_mask: np.ndarray) -> tuple[int, int]:
    agree = 0
    disagree = 0
    for comp in components:
        if int((comp.mask & dst_mask[comp.ys, comp.xs]).sum()) >= 1:
            agree += 1
        else:
            disagree += 1
    return agree, disagree


def add_counts(totals: dict[str, int], key_prefix: str, agree: int, disagree: int) -> None:
    totals[f"{key_prefix}_agree"] += int(agree)
    totals[f"{key_prefix}_disagree"] += int(disagree)
    totals[f"{key_prefix}_total"] += int(agree + disagree)


def rate(agree: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(agree / total)


def run(args: argparse.Namespace) -> int:
    h = int(args.canvas)
    w = int(args.canvas)

    json_root = args.json_root
    n_dir = json_root / "Nathalia"
    t_dir = json_root / "Thaer"

    n_files = {p.stem.replace("Nathalia_", "", 1): p for p in sorted(n_dir.glob("Nathalia_*.json"))}
    t_files = {p.stem.replace("Thaer_", "", 1): p for p in sorted(t_dir.glob("Thaer_*.json"))}
    all_pairs = sorted(set(n_files) | set(t_files))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "n_vs_t_agree": 0,
        "n_vs_t_disagree": 0,
        "n_vs_t_total": 0,
        "t_vs_n_agree": 0,
        "t_vs_n_disagree": 0,
        "t_vs_n_total": 0,
        "n_vs_m_agree": 0,
        "n_vs_m_disagree": 0,
        "n_vs_m_total": 0,
        "t_vs_m_agree": 0,
        "t_vs_m_disagree": 0,
        "t_vs_m_total": 0,
        "m_vs_n_agree": 0,
        "m_vs_n_disagree": 0,
        "m_vs_n_total": 0,
        "m_vs_t_agree": 0,
        "m_vs_t_disagree": 0,
        "m_vs_t_total": 0,
    }

    per_pair: list[dict[str, object]] = []

    for pair in all_pairs:
        n_ells = load_ellipses(n_files[pair]) if pair in n_files else []
        t_ells = load_ellipses(t_files[pair]) if pair in t_files else []

        n_mask = build_doc_mask(n_ells, h, w)
        t_mask = build_doc_mask(t_ells, h, w)

        model_map = load_model_map(args.predictions_root / pair, h=h, w=w)
        model_mask = np.abs(model_map) >= float(args.model_abs_threshold)
        comps = model_components(model_mask)

        n_t_ag, n_t_dis = directional_doc_vs_mask(n_ells, t_mask, h=h, w=w)
        t_n_ag, t_n_dis = directional_doc_vs_mask(t_ells, n_mask, h=h, w=w)

        n_m_ag, n_m_dis = directional_doc_vs_mask(n_ells, model_mask, h=h, w=w)
        t_m_ag, t_m_dis = directional_doc_vs_mask(t_ells, model_mask, h=h, w=w)

        m_n_ag, m_n_dis = directional_components_vs_mask(comps, n_mask)
        m_t_ag, m_t_dis = directional_components_vs_mask(comps, t_mask)

        add_counts(totals, "n_vs_t", n_t_ag, n_t_dis)
        add_counts(totals, "t_vs_n", t_n_ag, t_n_dis)
        add_counts(totals, "n_vs_m", n_m_ag, n_m_dis)
        add_counts(totals, "t_vs_m", t_m_ag, t_m_dis)
        add_counts(totals, "m_vs_n", m_n_ag, m_n_dis)
        add_counts(totals, "m_vs_t", m_t_ag, m_t_dis)

        per_pair.append(
            {
                "pair": pair,
                "n_total": len(n_ells),
                "t_total": len(t_ells),
                "m_components_total": len(comps),
                "n_vs_t_agree": n_t_ag,
                "n_vs_t_disagree": n_t_dis,
                "t_vs_n_agree": t_n_ag,
                "t_vs_n_disagree": t_n_dis,
                "n_vs_m_agree": n_m_ag,
                "n_vs_m_disagree": n_m_dis,
                "t_vs_m_agree": t_m_ag,
                "t_vs_m_disagree": t_m_dis,
                "m_vs_n_agree": m_n_ag,
                "m_vs_n_disagree": m_n_dis,
                "m_vs_t_agree": m_t_ag,
                "m_vs_t_disagree": m_t_dis,
                "model_pixels": int(model_mask.sum()),
            }
        )

    summary = {
        "criterion": "agreement if overlap_pixels >= 1",
        "model_abs_threshold": float(args.model_abs_threshold),
        "pairs_count": len(all_pairs),
        "counts": totals,
        "rates": {
            "n_vs_t": rate(totals["n_vs_t_agree"], totals["n_vs_t_total"]),
            "t_vs_n": rate(totals["t_vs_n_agree"], totals["t_vs_n_total"]),
            "n_vs_m": rate(totals["n_vs_m_agree"], totals["n_vs_m_total"]),
            "t_vs_m": rate(totals["t_vs_m_agree"], totals["t_vs_m_total"]),
            "m_vs_n": rate(totals["m_vs_n_agree"], totals["m_vs_n_total"]),
            "m_vs_t": rate(totals["m_vs_t_agree"], totals["m_vs_t_total"]),
            "doc_doc_mean": 0.5
            * (
                rate(totals["n_vs_t_agree"], totals["n_vs_t_total"])
                + rate(totals["t_vs_n_agree"], totals["t_vs_n_total"])
            ),
            "doc_model_mean": 0.25
            * (
                rate(totals["n_vs_m_agree"], totals["n_vs_m_total"])
                + rate(totals["t_vs_m_agree"], totals["t_vs_m_total"])
                + rate(totals["m_vs_n_agree"], totals["m_vs_n_total"])
                + rate(totals["m_vs_t_agree"], totals["m_vs_t_total"])
            ),
        },
    }

    pair_csv = out_dir / "mini_ov_per_pair.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_pair[0].keys()) if per_pair else ["pair"])
        writer.writeheader()
        for row in per_pair:
            writer.writerow(row)

    out_json = out_dir / "mini_ov_summary.json"
    out_json.write_text(
        json.dumps({"summary": summary, "per_pair": per_pair}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {pair_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini OV for Nathalia, Thaer, and model outputs using 1-pixel overlap criterion."
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named",
        help="Root with Nathalia/ and Thaer/ JSON-only folders.",
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("python_files") / "annotation tool" / "predictions_pnimit_lung",
        help="Root of model prediction pair directories containing output.nii.gz.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named" / "mini_ov_nathalia_thaer_model",
        help="Output directory for mini OV summary files.",
    )
    parser.add_argument(
        "--canvas",
        type=int,
        default=792,
        help="Canvas size for annotation rasterization.",
    )
    parser.add_argument(
        "--model-abs-threshold",
        type=float,
        default=0.05,
        help="Absolute threshold on model output map for binary model mask.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
