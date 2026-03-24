"""Compare inference-relevant image statistics for 1-98 pairs vs PNIMIT pairs.

This script scans pair folders, reads image NIfTI files, optionally reads matching
lung masks, computes per-image metrics, and writes:
1) a detailed per-image CSV
2) a dataset-level comparison CSV
3) a human-readable text summary
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PAIR_NUM_RE = re.compile(r"^(?:pair|Pair)(\d+)$")


@dataclass
class ImageRecord:
    dataset: str
    pair_dir: str
    image_file: str
    mask_file: str
    shape_h: float
    shape_w: float
    spacing_x: float
    spacing_y: float
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    intensity_median: float
    intensity_p01: float
    intensity_p05: float
    intensity_p25: float
    intensity_p75: float
    intensity_p95: float
    intensity_p99: float
    intensity_iqr: float
    dynamic_range_p99_p01: float
    zero_fraction: float
    grad_mag_mean: float
    grad_mag_std: float
    entropy_256: float
    lung_mask_available: float
    lung_fraction: float
    lung_intensity_mean: float
    lung_intensity_std: float

    def to_row(self) -> Dict[str, float | str]:
        return self.__dict__.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare image statistics between 1-98 pairs and PNIMIT pairs."
    )
    parser.add_argument(
        "--pairs-roots",
        nargs="+",
        default=[
            "python_files/annotation tool/Pairs1",
            "python_files/annotation tool/Pairs2",
            "python_files/annotation tool/Pairs3",
            "python_files/annotation tool/Pairs4",
            "python_files/annotation tool/Pairs5",
            "python_files/annotation tool/Pairs6",
            "python_files/annotation tool/Pairs7",
            "python_files/annotation tool/Pairs8",
        ],
        help="Roots that contain pair1..pair98 folders.",
    )
    parser.add_argument(
        "--pnimit-root",
        default="python_files/annotation tool/Pairs_PNIMIT_1_pairs",
        help="Root that contains PNIMIT pair_A* folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="python_files/Sahar_work/files/pairs_vs_pnimit_image_stats",
        help="Directory to write CSV and summary outputs.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=256,
        help="Number of bins for normalized intensity histogram.",
    )
    return parser.parse_args()


def _pair_number(folder_name: str) -> Optional[int]:
    match = PAIR_NUM_RE.match(folder_name)
    if not match:
        return None
    return int(match.group(1))


def _iter_pair_dirs(roots: Sequence[Path], keep_1_to_98_only: bool) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if keep_1_to_98_only:
                num = _pair_number(child.name)
                if num is None or num < 1 or num > 98:
                    continue
            yield child


def _nifti_spacing_xy(nifti_obj: nib.Nifti1Image) -> Tuple[float, float]:
    zooms = nifti_obj.header.get_zooms()
    sx = float(zooms[0]) if len(zooms) >= 1 else math.nan
    sy = float(zooms[1]) if len(zooms) >= 2 else math.nan
    return sx, sy


def _to_2d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    out = np.squeeze(out)
    if out.ndim == 2:
        return out
    if out.ndim < 2:
        raise ValueError(f"Expected at least 2D image, got shape {out.shape}")
    # Keep first two dimensions and use the middle index for remaining axes.
    slicing = [slice(None), slice(None)]
    for axis in range(2, out.ndim):
        slicing.append(out.shape[axis] // 2)
    return out[tuple(slicing)]


def _safe_entropy(values: np.ndarray, bins: int = 256) -> float:
    if values.size == 0:
        return math.nan
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return math.nan
    if vmin == vmax:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax), density=False)
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return math.nan
    probs /= total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _find_mask_for_image(image_path: Path) -> Optional[Path]:
    name = image_path.name
    if not name.endswith(".nii.gz"):
        return None
    stem = name[: -len(".nii.gz")]
    candidate = image_path.parent / f"{stem}_lung_seg.nii.gz"
    return candidate if candidate.exists() else None


def _iter_image_files(pair_dir: Path) -> Iterable[Path]:
    for nii_path in sorted(pair_dir.glob("*.nii.gz")):
        n = nii_path.name
        if "_lung_seg" in n:
            continue
        if n.lower() == "output.nii.gz":
            continue
        yield nii_path


def compute_record(dataset: str, pair_dir: Path, image_path: Path) -> ImageRecord:
    img_nii = nib.load(str(image_path))
    img = img_nii.get_fdata(dtype=np.float32)
    img2d = _to_2d(img)
    flat = img2d.reshape(-1)
    sx, sy = _nifti_spacing_xy(img_nii)

    p01, p05, p25, p75, p95, p99 = [
        float(v)
        for v in np.percentile(flat, [1.0, 5.0, 25.0, 75.0, 95.0, 99.0])
    ]
    grad_y, grad_x = np.gradient(img2d.astype(np.float32, copy=False))
    grad_mag = np.sqrt(grad_x**2 + grad_y**2, dtype=np.float32)

    mask_path = _find_mask_for_image(image_path)
    lung_available = 0.0
    lung_fraction = math.nan
    lung_mean = math.nan
    lung_std = math.nan

    if mask_path is not None:
        mask_nii = nib.load(str(mask_path))
        mask = _to_2d(mask_nii.get_fdata(dtype=np.float32))
        mask_bin = mask > 0.5
        lung_available = 1.0
        lung_fraction = float(mask_bin.mean())
        if mask_bin.any():
            in_lung = img2d[mask_bin]
            lung_mean = float(np.mean(in_lung))
            lung_std = float(np.std(in_lung))

    return ImageRecord(
        dataset=dataset,
        pair_dir=pair_dir.name,
        image_file=image_path.name,
        mask_file=mask_path.name if mask_path else "",
        shape_h=float(img2d.shape[0]),
        shape_w=float(img2d.shape[1]),
        spacing_x=sx,
        spacing_y=sy,
        intensity_min=float(np.min(flat)),
        intensity_max=float(np.max(flat)),
        intensity_mean=float(np.mean(flat)),
        intensity_std=float(np.std(flat)),
        intensity_median=float(np.median(flat)),
        intensity_p01=p01,
        intensity_p05=p05,
        intensity_p25=p25,
        intensity_p75=p75,
        intensity_p95=p95,
        intensity_p99=p99,
        intensity_iqr=float(p75 - p25),
        dynamic_range_p99_p01=float(p99 - p01),
        zero_fraction=float((img2d == 0).mean()),
        grad_mag_mean=float(np.mean(grad_mag)),
        grad_mag_std=float(np.std(grad_mag)),
        entropy_256=_safe_entropy(flat),
        lung_mask_available=lung_available,
        lung_fraction=lung_fraction,
        lung_intensity_mean=lung_mean,
        lung_intensity_std=lung_std,
    )


def summarize_records(records: Sequence[ImageRecord]) -> Dict[str, float]:
    metric_names = [
        "shape_h",
        "shape_w",
        "spacing_x",
        "spacing_y",
        "intensity_mean",
        "intensity_std",
        "intensity_median",
        "intensity_p01",
        "intensity_p05",
        "intensity_p95",
        "intensity_p99",
        "intensity_iqr",
        "dynamic_range_p99_p01",
        "zero_fraction",
        "grad_mag_mean",
        "grad_mag_std",
        "entropy_256",
        "lung_mask_available",
        "lung_fraction",
        "lung_intensity_mean",
        "lung_intensity_std",
    ]
    out: Dict[str, float] = {"n_images": float(len(records))}
    for metric in metric_names:
        values = np.array([getattr(r, metric) for r in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            for suffix in ["mean", "std", "median", "q25", "q75", "min", "max"]:
                out[f"{metric}_{suffix}"] = math.nan
            continue
        out[f"{metric}_mean"] = float(np.mean(values))
        out[f"{metric}_std"] = float(np.std(values))
        out[f"{metric}_median"] = float(np.median(values))
        out[f"{metric}_q25"] = float(np.percentile(values, 25))
        out[f"{metric}_q75"] = float(np.percentile(values, 75))
        out[f"{metric}_min"] = float(np.min(values))
        out[f"{metric}_max"] = float(np.max(values))
    return out


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_normalized_histogram(
    image_paths: Sequence[Path],
    bins: int,
    vmin: float,
    vmax: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    counts = np.zeros(bins, dtype=np.float64)
    total_pixels = 0
    for image_path in image_paths:
        img_nii = nib.load(str(image_path))
        flat = _to_2d(img_nii.get_fdata(dtype=np.float32)).reshape(-1)
        hist, bin_edges = np.histogram(flat, bins=bins, range=(vmin, vmax), density=False)
        counts += hist.astype(np.float64)
        total_pixels += int(flat.size)
    if total_pixels <= 0:
        return np.zeros(bins, dtype=np.float64), np.linspace(vmin, vmax, bins + 1), 0
    return counts / float(total_pixels), bin_edges.astype(np.float64), total_pixels


def save_histogram_outputs(
    output_dir: Path,
    bins: int,
    pairs_hist: np.ndarray,
    pnimit_hist: np.ndarray,
    bin_edges: np.ndarray,
    pairs_total_pixels: int,
    pnimit_total_pixels: int,
) -> Tuple[Path, Path]:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pairs_counts = pairs_hist * float(pairs_total_pixels)
    pnimit_counts = pnimit_hist * float(pnimit_total_pixels)

    hist_csv = output_dir / "intensity_histogram_normalized.csv"
    rows: List[Dict[str, float | str]] = []
    for i in range(bins):
        rows.append(
            {
                "bin_index": i,
                "bin_left": float(bin_edges[i]),
                "bin_right": float(bin_edges[i + 1]),
                "bin_center": float(centers[i]),
                "pairs_1_98_normalized": float(pairs_hist[i]),
                "pnimit_normalized": float(pnimit_hist[i]),
                "pairs_1_98_count": float(pairs_counts[i]),
                "pnimit_count": float(pnimit_counts[i]),
            }
        )
    write_csv(hist_csv, rows)

    hist_png = output_dir / "intensity_histogram_normalized.png"
    plt.figure(figsize=(10, 5))
    plt.plot(centers, pairs_hist, label="pairs_1_98", linewidth=2)
    plt.plot(centers, pnimit_hist, label="pnimit", linewidth=2)
    plt.xlabel("Intensity")
    plt.ylabel("Normalized pixel count")
    plt.title("Intensity Histogram (Normalized by Pixel Count)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_png, dpi=180)
    plt.close()

    return hist_csv, hist_png


def format_summary(
    pairs_summary: Dict[str, float],
    pnimit_summary: Dict[str, float],
) -> str:
    interesting = [
        "shape_h_mean",
        "shape_w_mean",
        "spacing_x_mean",
        "spacing_y_mean",
        "intensity_mean_mean",
        "intensity_std_mean",
        "dynamic_range_p99_p01_mean",
        "zero_fraction_mean",
        "grad_mag_mean_mean",
        "entropy_256_mean",
        "lung_mask_available_mean",
        "lung_fraction_mean",
        "lung_intensity_mean_mean",
        "lung_intensity_std_mean",
    ]
    lines: List[str] = []
    lines.append("Image Statistics Comparison: 1-98 pairs vs PNIMIT")
    lines.append("")
    lines.append(f"1-98 image count: {int(pairs_summary.get('n_images', 0))}")
    lines.append(f"PNIMIT image count: {int(pnimit_summary.get('n_images', 0))}")
    lines.append("")
    lines.append("Selected metrics (dataset means):")
    lines.append("metric,pairs_1_98,pnimit,delta(pnimit-pairs)")
    for key in interesting:
        pv = pairs_summary.get(key, math.nan)
        qv = pnimit_summary.get(key, math.nan)
        delta = qv - pv if np.isfinite(pv) and np.isfinite(qv) else math.nan
        lines.append(f"{key},{pv:.6g},{qv:.6g},{delta:.6g}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    pairs_roots = [repo_root / Path(p) for p in args.pairs_roots]
    pnimit_root = repo_root / Path(args.pnimit_root)
    output_dir = repo_root / Path(args.output_dir)

    pairs_records: List[ImageRecord] = []
    pnimit_records: List[ImageRecord] = []
    pairs_image_paths: List[Path] = []
    pnimit_image_paths: List[Path] = []

    for pair_dir in _iter_pair_dirs(pairs_roots, keep_1_to_98_only=True):
        for image_path in _iter_image_files(pair_dir):
            try:
                pairs_records.append(compute_record("pairs_1_98", pair_dir, image_path))
                pairs_image_paths.append(image_path)
            except Exception as exc:
                print(f"[WARN] Failed reading {image_path}: {exc}")

    for pair_dir in _iter_pair_dirs([pnimit_root], keep_1_to_98_only=False):
        for image_path in _iter_image_files(pair_dir):
            try:
                pnimit_records.append(compute_record("pnimit", pair_dir, image_path))
                pnimit_image_paths.append(image_path)
            except Exception as exc:
                print(f"[WARN] Failed reading {image_path}: {exc}")

    all_records = pairs_records + pnimit_records
    per_image_rows = [r.to_row() for r in all_records]

    pairs_summary = summarize_records(pairs_records)
    pnimit_summary = summarize_records(pnimit_records)

    summary_rows = [
        {"dataset": "pairs_1_98", **pairs_summary},
        {"dataset": "pnimit", **pnimit_summary},
    ]

    per_image_csv = output_dir / "per_image_stats.csv"
    summary_csv = output_dir / "dataset_summary.csv"
    summary_txt = output_dir / "comparison_summary.txt"

    write_csv(per_image_csv, per_image_rows)
    write_csv(summary_csv, summary_rows)
    summary_txt.write_text(format_summary(pairs_summary, pnimit_summary), encoding="utf-8")

    all_records_nonempty = [r for r in all_records if np.isfinite(r.intensity_min) and np.isfinite(r.intensity_max)]
    if all_records_nonempty:
        global_min = float(min(r.intensity_min for r in all_records_nonempty))
        global_max = float(max(r.intensity_max for r in all_records_nonempty))
        if global_max == global_min:
            global_max = global_min + 1.0

        pairs_hist, bin_edges, pairs_total_pixels = compute_normalized_histogram(
            pairs_image_paths,
            bins=args.hist_bins,
            vmin=global_min,
            vmax=global_max,
        )
        pnimit_hist, _, pnimit_total_pixels = compute_normalized_histogram(
            pnimit_image_paths,
            bins=args.hist_bins,
            vmin=global_min,
            vmax=global_max,
        )
        hist_csv, hist_png = save_histogram_outputs(
            output_dir=output_dir,
            bins=args.hist_bins,
            pairs_hist=pairs_hist,
            pnimit_hist=pnimit_hist,
            bin_edges=bin_edges,
            pairs_total_pixels=pairs_total_pixels,
            pnimit_total_pixels=pnimit_total_pixels,
        )
    else:
        hist_csv = output_dir / "intensity_histogram_normalized.csv"
        hist_png = output_dir / "intensity_histogram_normalized.png"
        write_csv(hist_csv, [])

    print(f"Wrote: {per_image_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote: {hist_csv}")
    print(f"Wrote: {hist_png}")
    print(f"Images analyzed -> pairs_1_98: {len(pairs_records)}, pnimit: {len(pnimit_records)}")


if __name__ == "__main__":
    main()
