"""Exploratory clinical100 postprocessing sweep; never modifies saved predictions."""

import argparse
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import find_objects, gaussian_filter, label

from evaluation.compare_clinical_models import write_csv, write_json
from evaluation.evaluate_clinical import PHYSICIANS, clinical_tools, discover_cases


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = np.ones((3, 3), dtype=int)


def legacy_postprocess():
    """Load only the actual legacy functions, avoiding unrelated model imports."""
    path = ROOT.parent / "python_files/archive/evaluation/Evaluation/Prediction.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("postprocess", "remove_small_ccs")]
    if len(functions) != 2:
        raise ValueError("Expected both legacy cleanup functions")
    namespace = dict(torch=torch, np=np, label=label, struct=STRUCTURE)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["postprocess"], hashlib.sha256(path.read_bytes()).hexdigest()


def match_counts(reference_labels, reference_slices, prediction_labels, prediction_count):
    """Same reference-first greedy matching and overlap tie order as legacy OV."""
    consumed = set()
    agreements = 0
    for component, region in enumerate(reference_slices, 1):
        values = prediction_labels[region][reference_labels[region] == component]
        ids, counts = np.unique(values, return_counts=True)
        available = np.array([value != 0 and value not in consumed for value in ids], dtype=bool)
        ids, counts = ids[available], counts[available]
        if ids.size:
            consumed.add(int(ids[np.argsort(counts)[::-1][0]]))
            agreements += 1
    disagreements = len(reference_slices) + prediction_count - 2 * agreements
    return agreements, disagreements


def configurations():
    result = []
    for method in ("raw", "components_50", "components_250", "components_500", "components_1000",
                   "gaussian_1", "gaussian_2", "gaussian_1_components_250"):
        for threshold in (.1, .25, .5, .75, .9):
            result.append(dict(method=method, threshold=threshold))
    for threshold in (0., .1, .25, .5):
        result.append(dict(method="itamar_exact_native", threshold=threshold))
    return result


def masks_for(scores, method, threshold):
    sigma = 2 if method == "gaussian_2" else 1
    if method.startswith("gaussian"):
        scores = gaussian_filter(scores, sigma=tuple(sigma * size / 512 for size in scores.shape))
    masks = [scores > threshold, scores < -threshold]
    if "components_" in method:
        cutoff = int(method.rsplit("_", 1)[1]) * scores.size / (512 * 512)
        for index, mask in enumerate(masks):
            components, _ = label(mask, STRUCTURE)
            sizes = np.bincount(components.ravel())
            keep = sizes > cutoff
            keep[0] = False
            masks[index] = keep[components]
    return masks


def f1(agreement, disagreement):
    denominator = 2 * agreement + disagreement
    if not denominator:
        raise ValueError("No reference or prediction components across the cohort")
    return 2 * agreement / denominator


def run(model_paths, destination, clinical_root):
    destination.mkdir(parents=True, exist_ok=False)
    tools = clinical_tools()
    tools.struct = STRUCTURE
    cases = discover_cases(clinical_root, 100)
    cleanup, cleanup_hash = legacy_postprocess()
    configs = configurations()
    rows, per_pair = [], []
    newline_normalized_annotations = set()
    for model_path in model_paths:
        verification = json.loads((model_path / "verification.json").read_text())
        totals = np.zeros((len(configs), len(PHYSICIANS), 2, 2), dtype=np.int64)
        areas = np.zeros(len(configs))
        empty_areas = [[] for _ in configs]
        for case_number, case in enumerate(cases, 1):
            pair = case["pair_id"]
            output = model_path / pair / "output.nii.gz"
            if hashlib.sha256(output.read_bytes()).hexdigest() != verification["prediction_sha256"][pair]:
                raise ValueError(f"Prediction hash mismatch: {output}")
            scores = np.asarray(nib.load(output).dataobj, dtype=np.float32).squeeze()
            if scores.ndim != 2 or not np.isfinite(scores).all():
                raise ValueError(f"Invalid prediction: {output}")
            references = []
            human_masks = []
            for doctor in PHYSICIANS:
                annotation = Path(case["annotations"][doctor])
                payload = annotation.read_bytes()
                expected = verification["annotation_sha256"][f"{pair}/{doctor}"]
                if hashlib.sha256(payload).hexdigest() != expected:
                    if hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() != expected:
                        raise ValueError(f"Annotation hash mismatch: {annotation}")
                    newline_normalized_annotations.add(f"{pair}/{doctor}")
                positive, negative = tools.load_labels_map(str(annotation), scores.shape)
                masks = (positive > 0, negative < 0)
                human_masks.append(masks)
                reference = []
                for mask in masks:
                    components, _ = label(mask, STRUCTURE)
                    reference.append((components, find_objects(components)))
                references.append(reference)
            empty_consensus = not any((np.stack([masks[sign] for masks in human_masks]).sum(0) >= 3).any()
                                      for sign in (0, 1))
            cleaned = cleanup(torch.from_numpy(scores.copy())).numpy()
            for index, config in enumerate(configs):
                source = cleaned if config["method"] == "itamar_exact_native" else scores
                masks = masks_for(source, config["method"], config["threshold"])
                area = float(np.logical_or(*masks).mean())
                areas[index] += area
                if empty_consensus:
                    empty_areas[index].append(area)
                for sign, mask in enumerate(masks):
                    components, count = label(mask, STRUCTURE)
                    for reader, reference in enumerate(references):
                        agreement, disagreement = match_counts(*reference[sign], components, count)
                        totals[index, reader, sign] += (agreement, disagreement)
                        per_pair.append(dict(model=model_path.name, pair=pair, **config,
                                             clinician=PHYSICIANS[reader], sign=sign,
                                             matches=agreement, unmatched=disagreement))
            if case_number % 20 == 0:
                print(f"{model_path.name}: {case_number}/100", flush=True)
        baseline = {}
        with (model_path / "detection_f1/f1_per_clinician.csv").open(newline="") as stream:
            baseline = {row["clinician"]: row for row in csv.DictReader(stream)}
        for index, config in enumerate(configs):
            for reader, doctor in enumerate(PHYSICIANS):
                positive, negative = totals[index, reader]
                combined = positive + negative
                values = dict(f1_positive=f1(*positive), f1_negative=f1(*negative), f1_all=f1(*combined))
                if config == dict(method="raw", threshold=.5):
                    for key, column in (("f1_positive", "F1 pos"), ("f1_negative", "F1 neg"), ("f1_all", "F1 all")):
                        if not np.isclose(values[key], float(baseline[doctor][column]), atol=1e-12, rtol=0):
                            raise ValueError(f"Baseline reproduction failed: {doctor} {key}")
                rows.append(dict(model=model_path.name, **config, clinician=doctor, **values,
                                 predicted_area_mean=areas[index] / 100,
                                 empty_consensus_area_mean=float(np.mean(empty_areas[index])),
                                 empty_consensus_area_max=float(np.max(empty_areas[index]))))
    write_csv(destination / "per_clinician.csv", rows)
    with gzip.open(destination / "per_pair_counts.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_pair[0]))
        writer.writeheader()
        writer.writerows(per_pair)
    summary = []
    for path in model_paths:
        for config in configs:
            selected = [row for row in rows if row["model"] == path.name
                        and all(row[key] == value for key, value in config.items())]
            summary.append(dict(model=path.name, **config,
                                **{key: float(np.mean([row[key] for row in selected]))
                                   for key in ("f1_positive", "f1_negative", "f1_all",
                                               "predicted_area_mean", "empty_consensus_area_mean",
                                               "empty_consensus_area_max")}))
    write_csv(destination / "summary.csv", summary)
    write_json(destination / "protocol.json", dict(
        exploratory=True, selection_cohort="same repeatedly used clinical100; not independent test",
        configurations=configs, models=[str(path.resolve()) for path in model_paths],
        baseline_reproduced=True, connectivity=8, matcher="legacy reference-first greedy one-to-one",
        windows_crlf_to_lf_annotations=sorted(newline_normalized_annotations),
        legacy_source_sha256=cleanup_hash,
        legacy_application="Exact legacy function on restored native scores; literal 250 pixels and 512^2 band rule. Not historical model-space preprocessing.",
        modern_components="Separate signs, retain area > cutoff scaled by native_area/512^2",
        smoothing="Gaussian sigma in equivalent 512-pixel coordinates, reflect boundary; threshold afterwards"))
    for path in model_paths:
        ranked = sorted((row for row in summary if row["model"] == path.name), key=lambda row: row["f1_all"], reverse=True)
        print(json.dumps(ranked[:5], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--clinical-root", type=Path, default=ROOT.parent / "python_files/annotation tool")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.models, args.out_dir, args.clinical_root)
