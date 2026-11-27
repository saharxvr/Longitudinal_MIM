"""CPU-only presentation-metric rescoring of immutable, saved ICU100 predictions."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import warnings

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label

from evaluation import evaluate_subset_clinical_ov as original


ROOT = original.ROOT
THRESHOLDS = original.THRESHOLDS
SIGNS = ("positive", "negative", "all")
METRICS = ("one_to_many", "one_to_one_presentation", "one_to_one_legacy")
BASELINE = ROOT / "results" / "clinical100_AB_Itamar_masks_20261022_194033_3343292"
PRESENTATION_BASELINE = (.5095313748322834, .4751602245426604)


def components(mask):
    labels, count = label(mask, original.STRUCTURE)
    return dict(labels=labels, count=int(count), slices=find_objects(labels),
                near=binary_dilation(mask, structure=original.STRUCTURE))


def with_partner(labels, other):
    return int(np.unique(labels[other & (labels > 0)]).size)


def comparison_terms(prediction, reference):
    """Same component/partner rules as the presentation's comparison_terms."""
    presentation, _ = original.match_counts(
        prediction["labels"], prediction["slices"], reference["labels"], reference["count"])
    legacy, _ = original.match_counts(
        reference["labels"], reference["slices"], prediction["labels"], prediction["count"])
    return np.array([
        prediction["count"], reference["count"],
        with_partner(prediction["labels"], reference["near"]),
        with_partner(reference["labels"], prediction["near"]),
        presentation, legacy], dtype=np.int64)


def scores(terms):
    predicted, reference, matched_pred, matched_ref, presentation, legacy = terms
    denominator = int(predicted + reference)
    return dict(
        one_to_many=float((matched_pred + matched_ref) / denominator) if denominator else 1.,
        one_to_one_presentation=float(2 * presentation / denominator) if denominator else 1.,
        one_to_one_legacy=float(2 * legacy / denominator) if denominator else None)


def validate_saved_run(inference, previous, cases, tools):
    record = original.read_json(inference / "complete.json")
    identity = record["identity"]
    original.verify_predictions(inference, identity, cases)
    if original.input_hashes(cases) != identity["inputs"]:
        raise ValueError("Current clinical images/annotations differ from saved inference")
    expected_names = {f"{arm}_fraction_{fraction:g}_seed_42"
                      for arm in original.shared.base.ARMS for fraction in original.FRACTIONS}
    if set(identity["checkpoints"]) != expected_names:
        raise ValueError("Rescoring requires exactly the twelve seed42/four-fraction models")
    sources = {
        "evaluate_subset_clinical_ov": original.__file__,
        "clinical_preprocessing": original.prepare_image.__code__.co_filename,
        "postprocessing": original.masks_for.__code__.co_filename,
        "legacy_ov": tools.__file__,
    }
    if {key: original.file_hash(path) for key, path in sources.items()} != identity["evaluation_code_sha256"]:
        raise ValueError("Original inference/evaluation sources changed; do not bypass their provenance")
    summary = original.read_json(previous / "summary.json")
    protocol = summary["protocol"]
    if (protocol["pairs"] != len(cases) or protocol["seed"] != 42
            or protocol["thresholds"] != list(THRESHOLDS) or protocol["cc_cutoff"] != 250
            or protocol["headline_threshold"] != .75
            or protocol["input_hash"] != original.shared.base.json_hash(identity["inputs"])
            or protocol["prediction_hash"] != original.shared.base.json_hash(record["prediction_sha256"])
            or protocol["evaluation_code_sha256"] != identity["evaluation_code_sha256"]):
        raise ValueError("Previous OV report does not match the saved predictions and operating points")
    headline = {row["model"]: row for row in summary["headline"]}
    if set(headline) != expected_names or any(
            any(headline[name][key] != value for key, value in metadata.items())
            for name, metadata in identity["checkpoints"].items()):
        raise ValueError("Previous OV checkpoint metadata differs from saved inference")
    with (previous / "per_clinician.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_keys = {(name, threshold, doctor) for name in expected_names
                     for threshold in THRESHOLDS for doctor in original.PHYSICIANS}
    indexed = {(row["model"], float(row["threshold"]), row["clinician"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected_keys:
        raise ValueError("Previous one-to-one OV table is incomplete or duplicated")
    return record, indexed


def reference_components(case, shape, tools):
    references, warning_count, invalid_radii = [], 0, 0
    for doctor in original.PHYSICIANS:
        path = Path(case["annotations"][doctor])
        annotations = json.loads(path.read_text(encoding="utf-8"))
        invalid_radii += sum(item["rx"] <= 0 or item["ry"] <= 0 for item in annotations[1:])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            positive, negative = tools.load_labels_map(str(path), shape)
        warning_count += sum(issubclass(w.category, RuntimeWarning) for w in caught)
        references.append((components(positive > 0), components(negative < 0)))
    return references, warning_count, invalid_radii


def baseline_terms(cases, tools, directory):
    manifest = original.read_json(directory / "manifest.json")
    totals = np.zeros((len(original.PHYSICIANS), 2, 6), dtype=np.int64)
    hashes = {}
    for case in cases:
        current = nib.load(case["current"])
        shape = tuple(size for size in current.shape if size != 1)
        if manifest["pairs"][case["pair_id"]]["shape"] != list(shape):
            raise ValueError("Presentation Arm A mask/current geometry differs")
        path = directory / f"{case['pair_id']}.npz"
        hashes[case["pair_id"]] = original.file_hash(path)
        with np.load(path, allow_pickle=False) as bundle:
            mask = bundle["A"]
            if mask.shape != shape or not np.isin(mask, (-1, 0, 1)).all():
                raise ValueError("Invalid presentation Arm A mask")
        references, _, _ = reference_components(case, shape, tools)
        for sign, boolean in enumerate((mask > 0, mask < 0)):
            predicted = components(boolean)
            for reader, reference in enumerate(references):
                totals[reader, sign] += comparison_terms(predicted, reference[sign])
    return totals, original.shared.base.json_hash(hashes)


def verify_baseline(totals):
    per_reader = [scores(values.sum(0)) for values in totals]
    means = [statistics.mean(row[metric] for row in per_reader)
             for metric in ("one_to_many", "one_to_one_presentation")]
    if not np.allclose(means, PRESENTATION_BASELINE, atol=1e-12, rtol=0):
        raise ValueError(f"Presentation Arm A baseline not reproduced: got {means}, expected {PRESENTATION_BASELINE}")
    return dict(
        model="Previous fusion Arm A (blocks 6/9/12, CE-only)",
        threshold=.75, cc_cutoff=250, one_to_many=means[0], one_to_one_presentation=means[1],
        one_to_one_legacy=statistics.mean(row["one_to_one_legacy"] for row in per_reader),
        reproduced_from_saved_masks=True)


def rescore(inference, previous, clinical_root, baseline_dir, destination, count=100):
    if destination.exists():
        raise ValueError("Use a new report directory; previous results are never overwritten")
    tools = original.clinical_tools()
    cases = original.discover_cases(clinical_root, count)
    record, previous_rows = validate_saved_run(inference, previous, cases, tools)
    models = record["identity"]["checkpoints"]
    totals = {name: np.zeros((len(THRESHOLDS), len(original.PHYSICIANS), 2, 6), dtype=np.int64)
              for name in models}
    warning_count, invalid_radii = 0, 0
    for number, case in enumerate(cases, 1):
        current = nib.load(case["current"])
        shape = tuple(size for size in current.shape if size != 1)
        references, caught, invalid = reference_components(case, shape, tools)
        warning_count += caught
        invalid_radii += invalid
        for name in models:
            image = nib.load(inference / name / case["pair_id"] / "output.nii.gz")
            values = np.asarray(image.dataobj, dtype=np.float32).squeeze()
            if (values.shape != shape or values.ndim != 2 or not np.isfinite(values).all()
                    or (np.abs(values) > 1.000001).any()
                    or not np.allclose(image.affine, current.affine, rtol=0, atol=1e-5)):
                raise ValueError("Saved prediction has invalid scores or current-image geometry")
            for index, threshold in enumerate(THRESHOLDS):
                masks = original.masks_for(values, "components_250", threshold)
                for sign, mask in enumerate(masks):
                    predicted = components(mask)
                    for reader, reference in enumerate(references):
                        totals[name][index, reader, sign] += comparison_terms(predicted, reference[sign])
        if number == 1 or number % 10 == 0:
            print(f"[CPU rescoring] {number}/{len(cases)} pairs; all twelve models/five thresholds", flush=True)
    rows = []
    for name in models:
        for index, threshold in enumerate(THRESHOLDS):
            for reader, doctor in enumerate(original.PHYSICIANS):
                row = dict(model=name, threshold=threshold, clinician=doctor)
                for sign, values in (
                        ("positive", totals[name][index, reader, 0]),
                        ("negative", totals[name][index, reader, 1]),
                        ("all", totals[name][index, reader].sum(0))):
                    result = scores(values)
                    previous_value = previous_rows[(name, threshold, doctor)][f"F1_{sign}"]
                    previous_value = float(previous_value) if previous_value else None
                    actual = result["one_to_one_legacy"]
                    if ((actual is None) != (previous_value is None) or (
                            actual is not None and not np.isclose(actual, previous_value, atol=1e-12, rtol=0))):
                        raise ValueError(f"Previous one-to-one OV result not reproduced: {name}/{threshold}/{doctor}/{sign}")
                    row.update({f"{metric}_{sign}": value for metric, value in result.items()})
                    for key, value in zip(
                            ("model_regions", "reference_regions", "matched_model", "matched_reference",
                             "presentation_matches", "legacy_matches"), values):
                        row[f"{key}_{sign}"] = int(value)
                rows.append(row)
    baseline_values, baseline_hash = baseline_terms(cases, tools, baseline_dir)
    baseline = verify_baseline(baseline_values)
    if original.input_hashes(cases) != record["identity"]["inputs"]:
        raise ValueError("Clinical inputs/annotations changed during rescoring")
    original.verify_predictions(inference, record["identity"], cases)
    summaries = []
    for name, metadata in models.items():
        for threshold in THRESHOLDS:
            selected = [row for row in rows if row["model"] == name and row["threshold"] == threshold]
            row = dict(**metadata, threshold=threshold, cc_cutoff=250)
            for metric in METRICS:
                for sign in SIGNS:
                    values = [item[f"{metric}_{sign}"] for item in selected
                              if item[f"{metric}_{sign}"] is not None]
                    row[f"mean_{metric}_{sign}"] = statistics.mean(values) if values else None
                    row[f"reader_SD_{metric}_{sign}"] = statistics.stdev(values) if len(values) > 1 else None
            summaries.append(row)
    protocol = dict(
        cohort="Repeatedly evaluated ICU100; exploratory, one training seed; no metric/threshold-based weight selection",
        pairs=len(cases), models=len(models), readers=list(original.PHYSICIANS),
        thresholds=list(THRESHOLDS), headline_threshold=.75, cc_cutoff=250,
        filtering="Unchanged strict signed thresholds; separate-sign 8-connected CC >250*image_area/512^2",
        one_to_many="(matched model regions + matched reference regions)/(model regions + reference regions); "
                    "each region counts once if any same-sign partner overlaps or touches (8-neighbours)",
        one_to_one_presentation="Overlap-only greedy matching, model-first; same order as presentation reference",
        one_to_one_legacy="Overlap-only greedy matching, reference-first; reproduces the previous OV table",
        aggregation="Pool counts over pairs and both signs for each reader; then mean/sample SD across five readers",
        empty_handling="Presentation ratios: empty/empty=1. Legacy OV undefined empty/empty remains null.",
        original_one_to_one_reproduced=True, presentation_baseline=baseline,
        input_hash=original.shared.base.json_hash(record["identity"]["inputs"]),
        prediction_hash=original.shared.base.json_hash(record["prediction_sha256"]),
        baseline_masks_hash=baseline_hash,
        source_sha256={key: original.file_hash(path) for key, path in (
            ("rescoring", __file__), ("postprocessing", original.masks_for.__code__.co_filename),
            ("legacy_ov", tools.__file__))},
        original_report_sha256=original.file_hash(previous / "summary.json"),
        annotation_runtime_warnings=warning_count, annotation_nonpositive_radius_records=invalid_radii,
        annotation_policy="Original rasterization unchanged; warnings counted explicitly, not proof of annotation validity",
        execution="CPU-only saved-prediction rescoring; no model construction, inference, training or input modification",
        publication="Aggregates only: no patient/case IDs, images, annotations, paths, predictions or weights")
    destination.mkdir(parents=True, exist_ok=False)
    original.shared.base.write_json(destination / "protocol.json", protocol)
    original.shared.base.write_json(destination / "summary.json", dict(
        protocol=protocol, baseline=baseline, headline=[row for row in summaries if row["threshold"] == .75],
        sweep=summaries))
    original.shared.base.write_csv(destination / "checkpoints.csv", list(models.values()))
    original.shared.base.write_csv(destination / "per_clinician.csv", rows)
    original.shared.base.write_csv(destination / "threshold_summary.csv", summaries)
    plot_results(summaries, baseline, destination)
    lines = [
        "# ICU100 presentation-metric rescoring", "",
        "Same saved predictions, thresholds and CC250 filtering; no retraining or inference.",
        "Previous fusion Arm A reproduced: one-to-many 0.509531; presentation one-to-one 0.475160.",
        "The old fusion model is not the same architecture/loss as the new block9 models.",
        "Headline threshold0.75; other thresholds remain exploratory.", "",
        "| Model | Training patients | One-to-many overlap/touch | One-to-one presentation | Previous one-to-one OV |",
        "|---|---:|---:|---:|---:|"]
    for row in summaries:
        if row["threshold"] == .75:
            lines.append(f"| {row['model']} | {row['patient_fraction']:.0%} | "
                         + " | ".join(f"{row[f'mean_{metric}_all']:.4f}"
                                      if row[f"mean_{metric}_all"] is not None else "NA" for metric in METRICS)
                         + " |")
    lines.extend(["", "[Exact per-reader scores/counts](per_clinician.csv) | [Sweep](threshold_summary.csv)",
                  "", "![Presentation-metric threshold sweep](threshold_sweep.png)",
                  "", "![Same-metric data comparison](data_fraction_f1.png)", ""])
    (destination / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


def plot_results(rows, baseline, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = sorted({row["patient_fraction"] for row in rows})
    figure, axes = plt.subplots(1, len(fractions), figsize=(5 * len(fractions), 4), squeeze=False)
    for axis, fraction in zip(axes[0], fractions):
        for arm in original.shared.base.ARMS:
            points = [row for row in rows if row["patient_fraction"] == fraction and row["arm"] == arm]
            axis.plot([row["threshold"] for row in points],
                      [row["mean_one_to_many_all"] for row in points], marker="o", label=arm)
        axis.set(title=f"{fraction:.0%} training patients", xlabel="Signed-score threshold",
                 ylabel="One-to-many overlap/touch F1", ylim=(0, 1))
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "threshold_sweep.png", dpi=120)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("one_to_many", "one_to_one_presentation")):
        for arm in original.shared.base.ARMS:
            points = [row for row in rows if row["arm"] == arm and row["threshold"] == .75]
            axis.plot([row["selected_pairs"] for row in points],
                      [row[f"mean_{metric}_all"] for row in points], marker="o", label=arm)
        axis.axhline(baseline[metric], color="black", linestyle="--", label="Previous fusion A")
        axis.set(title=metric.replace("_", " "), xlabel="Actual training pairs",
                 ylabel="Mean reader F1; threshold0.75/CC250", ylim=(0, 1))
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "data_fraction_f1.png", dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--clinical-root", type=Path, required=True)
    parser.add_argument("--baseline-masks", type=Path, default=BASELINE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    original.torch.set_num_threads(1)
    if args.publish:
        args.out_dir.resolve().relative_to((ROOT / "results").resolve())
    with original.shared.base.run_lock(args.inference_dir.with_suffix(".rescoring.lock")):
        destination = rescore(
            args.inference_dir, args.original_report, args.clinical_root,
            args.baseline_masks, args.out_dir)
    if args.publish:
        original.publish(destination)


if __name__ == "__main__":
    main()
