"""Legacy OV reports from saved heads at fixed signed threshold and CC cutoff."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects, label

from evaluation.compare_clinical_models import (
    legacy_summary, read_json, run_legacy, write_csv, write_json,
)
from evaluation.evaluate_clinical import PHYSICIANS, clinical_tools, discover_cases
from evaluation.sweep_clinical_postprocessing import STRUCTURE, masks_for, match_counts


ARMS = ("S6", "S9", "S12", "A", "R9A", "B", "R9B")


def filtered_scores(scores):
    if (scores.ndim != 2 or not np.isfinite(scores).all()
            or (np.abs(scores) > 1.000001).any()):
        raise ValueError("Expected finite two-dimensional signed scores in [-1,1]")
    positive, negative = masks_for(scores, "components_250", .75)
    return positive.astype(np.float32) - negative.astype(np.float32)


def run(source, clinical_root, destination, arms=ARMS, count=100):
    if not 1 <= count <= 100 or not arms or len(set(arms)) != len(arms):
        raise ValueError("Require 1..100 pairs and unique arms")
    if any(arm not in ARMS for arm in arms):
        raise ValueError("Unknown fusion arm")
    if destination.exists():
        raise ValueError("Use a new OV output directory")
    cases = discover_cases(clinical_root, count)
    tools = clinical_tools()
    tools.struct = STRUCTURE
    configs = {}
    annotation_hashes = {}
    for case in cases:
        for doctor in PHYSICIANS:
            path = Path(case["annotations"][doctor])
            payload = path.read_bytes()
            annotation_hashes[f'{case["pair_id"]}/{doctor}'] = (
                hashlib.sha256(payload).hexdigest(),
                hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest())
    for arm in arms:
        folder = source / f"arm_{arm}"
        metadata = read_json(folder / "metadata.json")
        report = read_json(folder / "report.json")
        if metadata["num_pairs"] < count or report["pairs_completed"] < count:
            raise ValueError(f"Incomplete saved inference: {arm}")
        for key, hashes in annotation_hashes.items():
            if metadata["annotation_sha256"][key] not in hashes:
                raise ValueError(f"Annotation changed: {arm}/{key}")
        for case in cases:
            path = folder / case["pair_id"] / "output.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
        configs[arm] = metadata
    destination.mkdir(parents=True)
    rows, summaries = [], {}
    for arm in arms:
        prediction_hashes = {}
        totals = np.zeros((len(PHYSICIANS), 2, 2), dtype=np.int64)
        with TemporaryDirectory(prefix="fusion_ov_") as temporary:
            filtered = Path(temporary)
            for case in cases:
                pair = case["pair_id"]
                path = source / f"arm_{arm}" / pair / "output.nii.gz"
                image = nib.load(path)
                scores = np.asarray(image.dataobj, dtype=np.float32).squeeze()
                current = nib.load(case["current"])
                if (scores.shape != tuple(size for size in current.shape if size != 1)
                        or not np.allclose(image.affine, current.affine)):
                    raise ValueError(f"Prediction/current geometry mismatch: {arm}/{pair}")
                masks = filtered_scores(scores)
                prediction_hashes[pair] = hashlib.sha256(path.read_bytes()).hexdigest()
                (filtered / pair).mkdir()
                header = image.header.copy()
                header.set_data_dtype(np.float32)
                nib.save(nib.Nifti1Image(masks, image.affine, header),
                         filtered / pair / "output.nii.gz")
                for reader, doctor in enumerate(PHYSICIANS):
                    positive, negative = tools.load_labels_map(
                        str(case["annotations"][doctor]), scores.shape)
                    for sign, reference_mask in enumerate((positive > 0, negative < 0)):
                        reference, _ = label(reference_mask, STRUCTURE)
                        prediction, components = label(masks > 0 if sign == 0 else masks < 0,
                                                       STRUCTURE)
                        totals[reader, sign] += match_counts(
                            reference, find_objects(reference), prediction, components)
            output = destination / f"arm_{arm}"
            print(f"[OV] {arm}: scoring {count} filtered pairs", flush=True)
            run_legacy(tools, filtered, clinical_root, output, count, "original", 0.)
        matrices = read_json(output / "agreement_matrices.json")
        model_index = matrices["observers"].index("Model")
        for reader, doctor in enumerate(PHYSICIANS):
            row = dict(arm=arm, clinician=doctor)
            index = matrices["observers"].index(doctor)
            for sign, counts in (
                    ("pos", totals[reader, 0]), ("neg", totals[reader, 1]),
                    ("all", totals[reader].sum(0))):
                matches, unmatched = counts
                expected = 2 * matches / (2 * matches + unmatched) if (2 * matches + unmatched) else np.nan
                actual = matrices[f"per_label_{sign}"][model_index][index]
                if not np.isclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True):
                    raise ValueError(f"Filtered-mask/OV F1 mismatch: {arm}/{doctor}/{sign}")
                row[f"F1_{sign}"] = actual if np.isfinite(actual) else None
            rows.append(row)
        summaries[arm] = legacy_summary(output)
        write_json(output / "input_provenance.json", dict(
            original_inference=configs[arm], prediction_sha256=prediction_hashes,
            code_sha256={name: hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
                         for name, module_path in (
                             ("evaluate_fusion_ov", __file__),
                             ("legacy_ov", tools.__file__),
                             ("postprocessing", masks_for.__code__.co_filename))},
            postprocessing=dict(
                threshold=.75, comparison="strict |score| > threshold",
                cc_cutoff=250, area_space="equivalent 512x512 pixels",
                cc_comparison="strict area > cutoff", connectivity=8,
                separate_signs=True, smoothing=False),
            warning="Requested fixed operating point, NOT exact Itamar legacy cleanup.",
            ov_input="temporary filtered signed masks; original predictions unchanged"))
        print(f"[OV] {arm}: complete; all per-reader F1 values verified", flush=True)
    write_csv(destination / "f1_per_clinician.csv", rows)
    write_json(destination / "summary.json", summaries)
    lines = [
        "# Clinical100 fusion OV: threshold 0.75, CC >250", "",
        "Fixed exploratory operating point; not exact Itamar cleanup.",
        "Sign-specific 8-connected components; area scaled from 512x512.",
        "Predictions stay in original image coordinates; annotations are unchanged.",
        "Detection F1 is not pixel Dice. Historical per-pair PAI retains legacy boosts.",
        "F1 with no reference or prediction components is undefined (blank/NA).",
        "", "| Arm | Mean reader F1 + | Mean reader F1 - | Mean reader F1 all |",
        "|---|---:|---:|---:|",
    ]
    for arm in arms:
        values = []
        for sign in ("pos", "neg", "all"):
            available = [row[f"F1_{sign}"] for row in rows
                         if row["arm"] == arm and row[f"F1_{sign}"] is not None]
            values.append(f"{np.mean(available):.4f}" if available else "NA")
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    lines.append("")
    lines += [f"- [{arm} full OV outputs](arm_{arm}/protocol.json)" for arm in arms]
    (destination / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[complete] {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-run", type=Path, required=True)
    parser.add_argument("--clinical-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--num-pairs", type=int, default=100)
    args = parser.parse_args()
    run(args.clinical_run, args.clinical_root, args.out_dir, args.arms, args.num_pairs)


if __name__ == "__main__":
    main()
