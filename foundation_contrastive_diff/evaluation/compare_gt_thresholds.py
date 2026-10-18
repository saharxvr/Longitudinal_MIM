"""Compare saved train/val GT audits and optional exact cached examples; no model inference."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.audit_gt import component_statistics, write_csv
from evaluation.training_diagnostics import gt_statistics, ratio


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize_audit(path, thresholds):
    groups = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] != "training_float16_simulated" or row["split"] not in ("train", "val"):
                continue
            if row["pair_id"] in seen:
                raise ValueError("Duplicate pair ID in selected audit stage")
            seen.add(row["pair_id"])
            if int(row["nonfinite"]) or int(row["out_of_range_pixels"]):
                raise ValueError("Invalid GT in source audit")
            for kind in ("overall", row["anomaly_type"]):
                groups[(row["split"], kind)].append(row)
    if not seen:
        raise ValueError("No train/val simulated-float16 audit rows")
    summaries = []
    for (split, kind), rows in sorted(groups.items()):
        pixels = sum(int(row["pixels"]) for row in rows)
        nonzero = sum(int(row["nonzero_pixels"]) for row in rows)
        mass = sum(float(row["mean_abs"]) * int(row["pixels"]) for row in rows)
        for threshold in thresholds:
            prefix = f"cutoff_{threshold:g}"
            kept = sum(int(row[f"{prefix}_pixels"]) for row in rows)
            areas = [float(row[f"{prefix}_area"]) for row in rows]
            summary = dict(split=split, anomaly_type=kind, threshold=threshold, pairs=len(rows),
                           originally_nonempty_pairs=sum(int(row["nonzero_pixels"]) > 0 for row in rows),
                           active_pairs=sum(int(row[f"{prefix}_pixels"]) > 0 for row in rows),
                           entirely_removed_pairs=sum(int(row[f"{prefix}_entirely_removed"]) for row in rows),
                           support_retained=ratio(kept, nonzero),
                           mass_retained=ratio(sum(float(row[f"{prefix}_mass_retained"] or 0) *
                                                   float(row["mean_abs"]) * int(row["pixels"]) for row in rows), mass),
                           foreground_area=ratio(kept, pixels), area_p95=float(np.percentile(areas, 95)),
                           area_max=max(areas),
                           component_metrics_available=all(f"components_{threshold:g}_positive_pixels" in row
                                                           for row in rows))
            if summary["component_metrics_available"]:
                for sign in ("positive", "negative"):
                    component = f"components_{threshold:g}_{sign}"
                    sign_pixels = sum(int(row[f"{component}_pixels"]) for row in rows)
                    counts = [int(row[f"{component}_count"]) for row in rows]
                    summary[f"{sign}_pixels"] = sign_pixels
                    summary[f"{sign}_area"] = ratio(sign_pixels, pixels)
                    summary[f"{sign}_pairs"] = sum(int(row[f"{component}_pixels"]) > 0 for row in rows)
                    summary[f"{sign}_entirely_removed_pairs"] = sum(
                        int(row[f"{sign}_pixels"]) > 0 and not int(row[f"{component}_pixels"]) for row in rows)
                    summary[f"{sign}_components"] = sum(counts)
                    summary[f"{sign}_components_per_active_pair"] = ratio(sum(counts), summary[f"{sign}_pairs"])
                    summary[f"{sign}_components_p95"] = float(np.percentile(counts, 95))
                    summary[f"{sign}_components_max"] = max(counts)
                    for cutoff in (5, 50):
                        tiny_count = sum(int(row[f"{component}_under_{cutoff}px"]) for row in rows)
                        tiny_pixels = sum(int(row[f"{component}_under_{cutoff}px_pixels"]) for row in rows)
                        summary[f"{sign}_under_{cutoff}px_components"] = tiny_count
                        summary[f"{sign}_under_{cutoff}px_component_fraction"] = ratio(tiny_count, sum(counts))
                        summary[f"{sign}_under_{cutoff}px_pixel_fraction"] = ratio(tiny_pixels, sign_pixels)
                summary["background_area"] = 1 - summary["positive_area"] - summary["negative_area"]
                summary["mixed_sign_pairs"] = sum(
                    int(row[f"components_{threshold:g}_positive_pixels"]) > 0 and
                    int(row[f"components_{threshold:g}_negative_pixels"]) > 0 for row in rows)
                summary["quantization_threshold_flips"] = sum(
                    int(row.get(f"cutoff_{threshold:g}_flips_from_float32", 0)) for row in rows)
            summaries.append(summary)
    return summaries


def render_example(arrays, gt, thresholds, destination, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    current = arrays["current_cache_input"].squeeze()
    axes[0, 0].imshow(arrays["prior_cache_input"].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Prior: actual cache input")
    axes[0, 1].imshow(current, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Current: actual cache input")
    peak = max(float(np.abs(gt).max()), 1e-8)
    axes[0, 2].imshow(gt, cmap="bwr", vmin=-peak, vmax=peak)
    axes[0, 2].set_title(f"Continuous cached GT: +/-{peak:.4g}")
    for axis, threshold in zip(axes[1], thresholds):
        target = np.where(np.abs(gt) > threshold, np.sign(gt), 0)
        stats = component_statistics(gt, [threshold])
        axis.imshow(current, cmap="gray", vmin=0, vmax=1)
        axis.imshow(np.ma.masked_where(target == 0, target), cmap="bwr", vmin=-1, vmax=1, alpha=.55)
        positive = stats[f"components_{threshold:g}_positive_count"]
        negative = stats[f"components_{threshold:g}_negative_count"]
        axis.set_title(f"GT cutoff {threshold:g}: area {100 * (target != 0).mean():.2f}%\n"
                       f"CC positive={positive}, negative={negative}")
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(f"{name}: same cached GT, no model predictions; red=positive, blue=negative")
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def run(args):
    thresholds = sorted(set(args.thresholds))
    if len(thresholds) != 3 or any(not np.isfinite(value) or not 0 < value < 1 for value in thresholds):
        raise ValueError("Provide three distinct finite thresholds in (0,1)")
    if args.out_dir.exists():
        raise ValueError("Use a new output directory")
    if getattr(args, "publish", False):
        parent = args.out_dir.resolve().parent
        if (args.audit_dir.resolve().parent != parent or
                {path.resolve() for path in parent.iterdir()} != {args.audit_dir.resolve()}):
            raise ValueError("Publication requires a dedicated parent containing only the source audit")
    audit_report = json.loads((args.audit_dir / "report.json").read_text(encoding="utf-8"))
    if audit_report["issues"]:
        raise ValueError("Resolve source audit issues before comparing targets")
    summaries = summarize_audit(args.audit_dir / "per_pair.csv", thresholds)
    examples = []
    sources = []
    paths = sorted(args.evidence_dir.glob("drr_*/exact_tensors.npz")) if args.evidence_dir else []
    if args.evidence_dir and not paths:
        raise ValueError("No exact DRR arrays in evidence directory")
    args.out_dir.mkdir(parents=True)
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            gt = arrays["cached_continuous_gt"].squeeze()
            fresh = arrays["fresh_continuous_gt"].squeeze()
            if gt.ndim != 2 or not np.isfinite(gt).all() or np.abs(gt).max() > 1:
                raise ValueError("Cached GT must be finite 2D and in [-1,1]")
            np.testing.assert_array_equal(gt, fresh.astype(np.float16).astype(np.float32))
            stats = component_statistics(gt, thresholds)
            for threshold in thresholds:
                row = dict(case=path.parent.name, threshold=threshold,
                           **gt_statistics(gt, [threshold]), **component_statistics(gt, [threshold]))
                positive = gt > threshold
                negative = gt < -threshold
                np.testing.assert_array_equal(positive | negative, np.abs(gt) > threshold)
                if (positive & negative).any():
                    raise AssertionError("Positive and negative classes overlap")
                row["quantization_threshold_flips"] = int(((np.abs(fresh) > threshold) !=
                                                           (np.abs(gt) > threshold)).sum())
                row["background_pixels"] = int(gt.size - positive.sum() - negative.sum())
                examples.append(row)
            del stats
            render_example(arrays, gt, thresholds, args.out_dir / f"{path.parent.name}.png", path.parent.name)
        sources.append(dict(path=str(path.resolve()), sha256=file_hash(path)))
    notes = ["Full-cohort measurements use saved simulated-float16 GT, not a new cache read.",
             "Rotation/reflection preserve pixel counts and 8-connected component topology.",
             "Exact examples are selected TRAIN cases, not representative cohort estimates.",
             "CCs are sign-separated, 8-connected; tiny means strictly fewer than 5 or 50 pixels.",
             "More components need not mean more changed area: a cutoff can split a component.",
             "All-background controls have undefined support/mass retention, not zero retention.",
             "Fewer pixels/components do not prove more biologically correct targets.",
             "No CC removal, GT normalization, model inference, training or test-split tuning."]
    report = dict(thresholds=thresholds, stage="training_float16_simulated", notes=notes,
                  manifest_sha256=audit_report["manifest_sha256"],
                  source_audit_parameters=audit_report["parameters"],
                  source_report_sha256=file_hash(args.audit_dir / "report.json"),
                  source_csv_sha256=file_hash(args.audit_dir / "per_pair.csv"),
                  code_sha256=file_hash(Path(__file__)), exact_sources=sources,
                  summaries=summaries, examples=examples)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_csv(args.out_dir / "examples.csv", examples)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# GT Threshold Comparison", "", *[f"- {note}" for note in notes], "",
             "[Full summary](summary.csv) | [Provenance and measurements](report.json)", "",
             "## Full Train/Validation Retention", "",
             "| Split | Cutoff | Active / originally active | Maps erased | Support retained | Absolute mass retained | Mean image area | Full-cohort CC available |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        if row["anomaly_type"] != "overall":
            continue
        lines.append(f"| {row['split']} | {row['threshold']:g} | {row['active_pairs']} / {row['originally_nonempty_pairs']} | "
                     f"{row['entirely_removed_pairs']} | {row['support_retained']:.3%} | {row['mass_retained']:.3%} | "
                     f"{row['foreground_area']:.3%} | {row['component_metrics_available']} |")
    if examples:
        lines += ["", "## Exact Cached Examples", "", "[Per-example measurements](examples.csv)", "",
                  "| Case | Cutoff | Positive CC | Negative CC | Positive area | Negative area | CC <50px (+/-) | Pixels in CC <50px (+/-) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in examples:
            prefix = f"components_{row['threshold']:g}"
            lines.append(f"| {row['case']} | {row['threshold']:g} | {row[prefix + '_positive_count']} | "
                         f"{row[prefix + '_negative_count']} | {row[prefix + '_positive_area']:.3%} | "
                         f"{row[prefix + '_negative_area']:.3%} | {row[prefix + '_positive_under_50px']} / "
                         f"{row[prefix + '_negative_under_50px']} | {row[prefix + '_positive_under_50px_pixels']} / "
                         f"{row[prefix + '_negative_under_50px_pixels']} |")
        for path in paths:
            lines += ["", f"## {path.parent.name}", "", f"![Same-case cutoff overlays]({path.parent.name}.png)"]
    (args.out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Threshold comparison: {args.out_dir / 'index.md'}", flush=True)
    if getattr(args, "publish", False):
        from training.run_upright_clinical import publish
        publish(args.out_dir.parent)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs=3, default=[.001, .01, .02])
    parser.add_argument("--publish", action="store_true",
                        help="Publish a dedicated results parent containing only this comparison and its fresh audit")
    run(parser.parse_args())