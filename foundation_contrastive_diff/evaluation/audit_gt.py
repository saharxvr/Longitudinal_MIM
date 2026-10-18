"""Audit continuous signed GT before choosing a loss; never run a model or alter data."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from nibabel.filebasedimages import ImageFileError

from datasets.pair_dataset import LongitudinalPairDataset, _load_nii_2d, _to_chw, _norm01, orient_tensor
from evaluation.training_diagnostics import gt_statistics, ratio


DEFAULT_THRESHOLDS = [0, 1e-8, 1e-6, 1e-4, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def component_statistics(values, thresholds):
    from scipy.ndimage import label

    result = {}
    for threshold in thresholds:
        for sign, mask in (("positive", values > threshold), ("negative", values < -threshold)):
            labels, count = label(mask, structure=np.ones((3, 3)))
            sizes = np.bincount(labels.ravel())[1:]
            prefix = f"components_{threshold:g}_{sign}"
            pixels = int(sizes.sum())
            result[f"{prefix}_pixels"] = pixels
            result[f"{prefix}_area"] = float(mask.mean())
            result[f"{prefix}_count"] = count
            result[f"{prefix}_under_5px"] = int((sizes < 5).sum())
            result[f"{prefix}_under_50px"] = int((sizes < 50).sum())
            result[f"{prefix}_under_5px_pixels"] = int(sizes[sizes < 5].sum())
            result[f"{prefix}_under_50px_pixels"] = int(sizes[sizes < 50].sum())
            result[f"{prefix}_under_50px_pixel_fraction"] = ratio(sizes[sizes < 50].sum(), pixels)
            result[f"{prefix}_median_size"] = float(np.median(sizes)) if count else None
            result[f"{prefix}_max_size"] = int(sizes.max()) if count else None
            result[f"{prefix}_largest_component_pixel_fraction"] = ratio(sizes.max(), pixels) if count else None
    return result


def verify_alignment(dataset, index):
    import nibabel as nib

    record = dataset.records[index]
    sources = [nib.load(dataset._abspath(record, key, filename)) for key, filename in
               (("prior_path", "prior.nii.gz"), ("current_path", "current.nii.gz"),
                ("diff_map_path", "diff_map.nii.gz"))]
    arrays = [np.asarray(source.dataobj, dtype=np.float32).squeeze() for source in sources]
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise ValueError("DRRs and GT must be finite 2D arrays")
    if any(array.shape != arrays[1].shape for array in arrays):
        raise ValueError("DRR/GT source shape mismatch")
    if any(not np.allclose(source.affine, sources[1].affine, rtol=0, atol=1e-5)
           for source in sources):
        raise ValueError("DRR/GT source affine mismatch")
    if np.abs(arrays[2]).max() > 1:
        raise ValueError("Source GT exceeds [-1,1]; clipping would change supervision")
    sample = dataset[index]
    for key, array in zip(("img_prior", "img_curr", "gt_diff"), arrays):
        resized = _to_chw(array, dataset.img_size)
        expected = resized if key == "gt_diff" else _norm01(resized)
        corrected = sample[key]
        if dataset.rotation_k == 3 and dataset.flip_horizontal:
            torch.testing.assert_close(corrected, expected.transpose(-2, -1), rtol=0, atol=0)
        restored = orient_tensor(corrected, dataset.rotation_k, dataset.flip_horizontal, inverse=True)
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    return sample


def render_overlay(sample, destination, pair_id, target_threshold=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = sample["gt_diff"][0].numpy()
    current = sample["img_curr"][0].numpy()
    peak = float(np.abs(gt).max())
    scale = peak if peak else 1.0
    figure, axes = plt.subplots(2, 4, figsize=(17, 9))
    axes = axes.ravel()
    axes[0].imshow(sample["img_prior"][0], cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(current, cmap="gray", vmin=0, vmax=1)
    fixed = axes[2].imshow(gt, cmap="bwr", vmin=-1, vmax=1)
    scaled = axes[3].imshow(gt, cmap="bwr", vmin=-scale, vmax=scale)
    figure.colorbar(fixed, ax=axes[2], fraction=0.046)
    figure.colorbar(scaled, ax=axes[3], fraction=0.046)
    axes[4].imshow(current, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(np.ma.masked_where(gt == 0, gt), cmap="bwr", vmin=-scale, vmax=scale, alpha=0.55)
    masks = (np.sign(gt), np.where(np.abs(gt) > target_threshold, np.sign(gt), 0),
             np.where((gt != 0) & (np.abs(gt) <= target_threshold), np.sign(gt), 0))
    for axis, mask in zip(axes[5:], masks):
        axis.imshow(current, cmap="gray", vmin=0, vmax=1)
        axis.imshow(np.ma.masked_where(mask == 0, mask), cmap="bwr", vmin=-1, vmax=1, alpha=0.55)
    titles = ("Prior", "Current", "Raw GT [-1,1]", f"Raw GT only: +/-{scale:.5g}",
              "Continuous GT overlay (GT-only scale)", "Sign GT: no amplitude cutoff",
              f"Signed training GT: abs(GT)>{target_threshold:g}",
              f"Nonzero support removed at {target_threshold:g}")
    for axis, title in zip(axes, titles):
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(f"{pair_id}\nRed=positive; blue=negative; peak={peak:.6g}; training input coordinates; "
                   f"rotation={sample['rotation_k']}; horizontal flip={sample.get('flip_horizontal', False)}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(destination, summaries, histograms, edges, thresholds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for row in summaries:
        if row["stage"] != "training_float32" or row["anomaly_type"] != "overall":
            continue
        retained = [row[f"cutoff_{value:g}_micro_support_retained"] for value in thresholds]
        lost = [row[f"cutoff_{value:g}_entirely_removed_pairs"] for value in thresholds]
        axes[0].plot(thresholds, retained, marker=".", label=row["split"])
        axes[1].plot(thresholds, lost, marker=".", label=row["split"])
    centers = np.sqrt(np.maximum(edges[:-1], 1e-14) * edges[1:])
    for (split, anomaly, stage), counts in sorted(histograms.items()):
        if stage == "training_float32":
            axes[2].plot(centers[:-1], counts[:-1], label=f"{split}/{anomaly}")
    for axis in axes[:2]:
        axis.set_xscale("symlog", linthresh=1e-6)
        axis.axvline(.05, color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("GT amplitude cutoff")
        axis.legend()
    axes[0].set_ylabel("Fraction of exact-nonzero pixels retained")
    axes[1].set_ylabel("Nonempty GT maps entirely removed")
    axes[2].set_xscale("log")
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set_xlabel("Nonzero absolute GT magnitude")
    axes[2].set_ylabel("Pixels per log-spaced bin")
    axes[2].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(destination, dpi=120)
    plt.close(figure)


def run(args):
    if args.out_dir.exists():
        raise ValueError("Use a new output directory to preserve previous audits")
    if args.limit < 0 or args.overlays_per_group < 0:
        raise ValueError("limit and overlays_per_group must be nonnegative")
    if any(not np.isfinite(value) or not 0 <= value < 1
           for value in [*args.thresholds, *args.component_thresholds]):
        raise ValueError("Thresholds must be finite and in [0,1)")
    dataset = LongitudinalPairDataset(str(args.dataset_root), img_size=args.img_size,
                                      rotation_k=args.rotation_k,
                                      flip_horizontal=getattr(args, "flip_horizontal", False))
    indices = [index for index, record in enumerate(dataset.records) if record.get("split") in args.splits]
    if not indices:
        raise ValueError("No matching split records; a split manifest is required")
    pair_ids = [dataset.records[index]["pair_id"] for index in indices]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Duplicate pair IDs in selected split records")
    generator = np.random.default_rng(args.seed)
    generator.shuffle(indices)
    if args.limit:
        indices = indices[:args.limit]
    thresholds = sorted(set([0, 0.05, *args.thresholds]))
    args.out_dir.mkdir(parents=True)
    gallery = args.out_dir / "cases"
    gallery.mkdir()
    rows, invalid, previews = [], [], []
    aligned_pairs = 0
    preview_counts = Counter()
    histogram_edges = np.concatenate(([0.], np.logspace(-12, 0, 49), [np.finfo(np.float32).max]))
    histograms = defaultdict(lambda: np.zeros(len(histogram_edges) - 1, dtype=np.int64))
    for position, index in enumerate(indices):
        record = dataset.records[index]
        pair_id = record["pair_id"]
        anomaly = record.get("effective_anomaly_type", record.get("anomaly_type", "none"))
        split = record["split"]
        path = Path(dataset._abspath(record, "diff_map_path", "diff_map.nii.gz"))
        try:
            aligned_sample = None
            if getattr(args, "verify_alignment", False):
                aligned_sample = verify_alignment(dataset, index)
                aligned_pairs += 1
            native = _load_nii_2d(str(path))
            if native.ndim != 2 or not np.isfinite(native).all():
                raise ValueError("GT must be a finite 2D array before clipping/resizing")
            resized = _to_chw(native, args.img_size).clamp(-1, 1)
            resized = orient_tensor(resized, args.rotation_k, dataset.flip_horizontal)
            processed = resized[0].numpy()
            quantized = resized.half().float()[0].numpy()
            for stage, values in (("native", native), ("training_float32", processed),
                                  ("training_float16_simulated", quantized)):
                magnitude = np.abs(values)
                row = dict(pair_id=pair_id, split=split, anomaly_type=anomaly, stage=stage,
                           source_gt=str(path.resolve()), height=values.shape[0], width=values.shape[1],
                           source_array_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                           **gt_statistics(values, thresholds),
                           **component_statistics(values, args.component_thresholds))
                row["out_of_range_pixels"] = int((magnitude > 1).sum())
                row["total_variation_per_edge"] = ratio(
                    np.abs(np.diff(values, axis=0)).sum() + np.abs(np.diff(values, axis=1)).sum(),
                    (values.shape[0] - 1) * values.shape[1] + values.shape[0] * (values.shape[1] - 1))
                if stage == "training_float16_simulated":
                    row["nonzero_pixels_lost_to_float16"] = int(((processed != 0) & (values == 0)).sum())
                    row["cutoff_005_flips_from_float32"] = int(((np.abs(processed) > .05) != (magnitude > .05)).sum())
                    for threshold in thresholds:
                        row[f"cutoff_{threshold:g}_flips_from_float32"] = int(
                            ((np.abs(processed) > threshold) != (magnitude > threshold)).sum())
                rows.append(row)
                histograms[(split, anomaly, stage)] += np.histogram(magnitude[magnitude > 0], bins=histogram_edges)[0]
            category = ("empty" if not np.any(processed) else "below_005" if np.abs(processed).max() <= .05
                        else "above_005")
            direction = "mixed" if np.any(processed > 0) and np.any(processed < 0) else (
                "positive" if np.any(processed > 0) else "negative" if np.any(processed < 0) else "empty")
            key = (split, anomaly, category, direction)
            if preview_counts[key] < args.overlays_per_group:
                try:
                    sample = aligned_sample if aligned_sample is not None else dataset[index]
                    np.testing.assert_array_equal(sample["gt_diff"].numpy(), resized.numpy())
                    filename = hashlib.sha256(pair_id.encode()).hexdigest()[:16] + ".png"
                    render_overlay(sample, gallery / filename, pair_id, getattr(args, "target_threshold", 0.05))
                    preview = dict(pair_id=pair_id, split=split, anomaly_type=anomaly,
                                   category=category, direction=direction, panel=f"cases/{filename}")
                    if getattr(args, "save_arrays", True):
                        array_file = Path(filename).with_suffix(".npz").name
                        np.savez_compressed(gallery / array_file, gt_native=native,
                                            gt_training_float32=processed, gt_training_float16_simulated=quantized,
                                            img_prior=sample["img_prior"].numpy(), img_current=sample["img_curr"].numpy(),
                                            pair_id=np.asarray(pair_id), source_gt=np.asarray(str(path.resolve())))
                        preview["arrays"] = f"cases/{array_file}"
                    previews.append(preview)
                    preview_counts[key] += 1
                except (OSError, ValueError, AssertionError, ImageFileError) as error:
                    invalid.append(dict(pair_id=pair_id, stage="overlay", error=str(error)))
        except (OSError, ValueError, RuntimeError, AssertionError, ImageFileError) as error:
            invalid.append(dict(pair_id=pair_id, stage="gt", error=str(error)))
        if (position + 1) % 100 == 0 or position + 1 == len(indices):
            print(f"[GT audit] {position + 1}/{len(indices)}; overlays={len(previews)}; issues={len(invalid)}", flush=True)
    write_csv(args.out_dir / "per_pair.csv", rows)
    summaries = []
    for split in args.splits:
        for stage in ("native", "training_float32", "training_float16_simulated"):
            for anomaly in ("overall", *sorted({row["anomaly_type"] for row in rows})):
                selected = [row for row in rows if row["split"] == split and row["stage"] == stage
                            and (anomaly == "overall" or row["anomaly_type"] == anomaly)]
                if not selected:
                    continue
                summary = dict(split=split, stage=stage, anomaly_type=anomaly, pairs=len(selected),
                               nonempty_pairs=sum(row["nonzero_pixels"] > 0 for row in selected))
                for threshold in thresholds:
                    prefix = f"cutoff_{threshold:g}"
                    summary[f"{prefix}_entirely_removed_pairs"] = sum(row[f"{prefix}_entirely_removed"] for row in selected)
                    summary[f"{prefix}_micro_support_retained"] = ratio(
                        sum(row[f"{prefix}_pixels"] for row in selected), sum(row["nonzero_pixels"] for row in selected))
                for key in selected[0]:
                    if key in ("pair_id", "split", "anomaly_type", "stage", "source_gt", "source_array_sha256"):
                        continue
                    values = [row[key] for row in selected if row.get(key) is not None]
                    summary[f"{key}_pair_mean"] = float(np.mean(values)) if values else None
                summaries.append(summary)
    write_csv(args.out_dir / "summary.csv", summaries)
    plot_summary(args.out_dir / "cutoff_diagnostics.png", summaries, histograms, histogram_edges, thresholds)
    manifest = next((args.dataset_root / name for name in ("manifest_split.jsonl", "manifest.jsonl")
                     if (args.dataset_root / name).is_file()))
    report = dict(parameters={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  selected_pairs=len(indices), audited_pairs=len({row["pair_id"] for row in rows}),
                  alignment_verified_pairs=aligned_pairs,
                  effective_thresholds=thresholds,
                  summaries=summaries, issues=invalid, previews=previews,
                  histogram_edges=histogram_edges.tolist(),
                  nonzero_abs_histograms=[dict(split=key[0], anomaly_type=key[1], stage=key[2], counts=value.tolist())
                                          for key, value in sorted(histograms.items())],
                  notes=["No model or loss selection is performed; inspect masks before choosing a cutoff.",
                         "Exact nonzero support may include interpolation/numerical tails, not only lesions.",
                         "Float16 values are simulated, not read from feature caches.",
                         "Component sizes are in pixels of the explicitly named stage.",
                         "Per-pair quantiles and pooled histogram counts are different statistics.",
                         "A limited run is a seeded global sample, not a stratified full-dataset estimate."])
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Continuous GT Audit", "", f"Selected pairs: {len(indices)}; issues: {len(invalid)}.", "",
             "[Summary CSV](summary.csv) | [Per-pair CSV](per_pair.csv) | [Report and histograms](report.json)", "",
             "Raw means continuous transformed GT, not a sign mask. Scales on the raw-GT zoom/overlay",
             "depend on GT alone, never prediction. Red=positive, blue=negative. Exact sign support",
             "can include tiny numerical/interpolation tails; this report does not select a loss.", "",
             "The CSV separates native arrays, training float32 preprocessing, and simulated float16.", "",
             "Selected example NPZ files contain real arrays (load with allow_pickle=False), not PNG-derived values.", "",
             "![Cutoff retention and magnitude distributions](cutoff_diagnostics.png)", "",
             "## Training-Space Cutoff Impact", "",
             "| Split | Type | Pairs | Nonempty | Entirely removed at 0.05 | Nonzero support retained |",
             "|---|---|---:|---:|---:|---:|"]
    for summary in summaries:
        if summary["stage"] == "training_float32":
            retained = summary["cutoff_0.05_micro_support_retained"]
            lines.append(f"| {summary['split']} | {summary['anomaly_type']} | {summary['pairs']} | "
                         f"{summary['nonempty_pairs']} | {summary['cutoff_0.05_entirely_removed_pairs']} | "
                         + (f"{retained:.3%} |" if retained is not None else "NA |"))
    lines += ["", "## Raw GT and Signed-Mask Overlays", ""]
    for preview in previews:
        lines += [f"### {preview['split']} / {preview['anomaly_type']} / {preview['category']}", "",
                  f"{preview['pair_id']}", "", f"![GT comparison]({preview['panel']})", ""]
        if "arrays" in preview:
            lines += [f"[Download GT and input arrays]({preview['arrays']})", ""]
    (args.out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[GT audit] report: {args.out_dir / 'index.md'}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"])
    parser.add_argument("--limit", type=int, default=0, help="0 audits all records in selected splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--rotation-k", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--verify-alignment", action="store_true")
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--component-thresholds", type=float, nargs="+", default=[0., .05])
    parser.add_argument("--overlays-per-group", type=int, default=2)
    parser.add_argument("--save-arrays", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(1 if result["issues"] else 0)