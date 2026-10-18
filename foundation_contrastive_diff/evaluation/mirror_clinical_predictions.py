"""Evaluate a display-horizontal mirror ablation of saved clinical predictions."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from evaluation import compare_clinical_models as comparison
from evaluation.evaluate_clinical import (
    PHYSICIANS, discover_cases, export_case, load_case, prepare_image, score_masks,
    summarize,
)


def mirror_prediction(scores):
    scores = np.asarray(scores)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected finite 2D source-space prediction")
    return np.flip(scores, axis=0).copy()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.out_dir.exists():
        raise ValueError("Use a new output directory; originals must remain unchanged")
    metadata_path = args.clinical_run / "metadata.json"
    metadata = comparison.read_json(metadata_path)
    if metadata.get("analysis_transform"):
        raise ValueError("Expected original predictions, not an already transformed run")
    if metadata["prediction_threshold"] != .5:
        raise ValueError("Expected the fixed clinical prediction threshold 0.5")
    if not 1 <= args.num_pairs <= min(100, metadata["num_pairs"]):
        raise ValueError("Invalid clinical pair count")
    cases = discover_cases(args.clinical_root, args.num_pairs)
    tools = comparison.clinical_tools()
    source_hashes = {}
    for case in cases:
        source = args.clinical_run / case["pair_id"] / "output.nii.gz"
        source_hashes[case["pair_id"]] = file_hash(source)
        for doctor, annotation in case["annotations"].items():
            payload = Path(annotation).read_bytes()
            expected = metadata["annotation_sha256"][f"{case['pair_id']}/{doctor}"]
            if expected not in (hashlib.sha256(payload).hexdigest(),
                                hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()):
                raise ValueError(f"Changed annotation: {case['pair_id']}/{doctor}")
    args.out_dir.mkdir(parents=True)
    predictions = args.out_dir / "predictions"
    predictions.mkdir()
    original_rows, mirrored_rows = [], []
    config = metadata.get("run_config", {})
    for case in cases:
        prior, current, current_nifti, annotations = load_case(case, tools)
        source = nib.load(args.clinical_run / case["pair_id"] / "output.nii.gz")
        scores = np.asarray(source.dataobj, dtype=np.float32)
        if scores.shape != current.shape or not np.array_equal(source.affine, current_nifti.affine):
            raise ValueError(f"{case['pair_id']}: source prediction geometry differs from current scan")
        if not np.isfinite(scores).all() or np.any(np.abs(scores) > 1):
            raise ValueError(f"{case['pair_id']}: invalid source scores")
        mirrored = mirror_prediction(scores)
        if not np.array_equal(mirrored.T, np.fliplr(scores.T)):
            raise AssertionError("Mirror must be horizontal in the displayed image")
        predicted = (scores > .5, scores < -.5)
        for doctor, masks in annotations.items():
            original_rows.append(dict(pair_id=case["pair_id"], observer=doctor,
                                      **score_masks(predicted, masks)))
        votes = np.asarray(list(annotations.values())).sum(axis=0)
        original_rows.append(dict(pair_id=case["pair_id"], observer="consensus_3_of_5",
                                  **score_masks(predicted, (votes[0] >= 3, votes[1] >= 3))))
        model_input = prepare_image(current, metadata.get("image_size", 512),
                                    config.get("rotation_k", metadata.get("rotation_k", 0)),
                                    config.get("flip_horizontal", False))[0].numpy()
        mirrored_rows.extend(export_case(
            case, mirrored, prior, current, model_input, current_nifti, annotations,
            predictions, .5,
            run_label=f"MIRROR ABLATION - epoch {metadata.get('checkpoint_epoch')} - threshold 0.5"))
        restored = nib.load(predictions / case["pair_id"] / "output.nii.gz")
        if not np.array_equal(np.asarray(restored.dataobj), mirrored):
            raise AssertionError(f"{case['pair_id']}: saved mirror differs")
        if not np.array_equal(restored.affine, source.affine):
            raise AssertionError(f"{case['pair_id']}: affine changed")
        mask = np.asarray(nib.load(predictions / case["pair_id"] / "mask.nii.gz").dataobj)
        expected_mask = (mirrored > .5).astype(np.int8) - (mirrored < -.5).astype(np.int8)
        if not np.array_equal(mask, expected_mask):
            raise AssertionError(f"{case['pair_id']}: saved signed mask differs")
        print(f"[mirror] {case['pair_id']}: values, horizontal display flip, affine and mask verified", flush=True)
    derived = dict(metadata)
    derived.update(num_pairs=args.num_pairs, analysis_transform=dict(
        kind="exploratory_display_horizontal_mirror", nifti_array_axis=0,
        source_run=str(args.clinical_run.resolve()),
        source_metadata_sha256=file_hash(metadata_path), source_prediction_sha256=source_hashes,
        annotations_transformed=False, threshold_changed=False,
        note="Post-hoc orientation ablation, not a verified coordinate correction or new inference"))
    comparison.write_json(predictions / "metadata.json", derived)
    comparison.write_csv(predictions / "per_pair.csv", mirrored_rows)
    comparison.write_json(predictions / "report.json", summarize(mirrored_rows))
    comparison.write_csv(args.out_dir / "per_pair_original.csv", original_rows)
    for name, clinical_run in (("original", args.clinical_run), ("mirrored", predictions)):
        comparison.run(SimpleNamespace(
            clinical_run=clinical_run, clinical_root=args.clinical_root,
            out_dir=args.out_dir / f"ov_{name}", num_pairs=args.num_pairs,
            foundation_only=True, itamar_preds=None))
    original_ov = args.out_dir / "ov_original" / f"foundation_{args.num_pairs}"
    mirrored_ov = args.out_dir / "ov_mirrored" / f"foundation_{args.num_pairs}"
    original_matrices = comparison.read_json(original_ov / "agreement_matrices.json")
    mirrored_matrices = comparison.read_json(mirrored_ov / "agreement_matrices.json")
    if original_matrices["observers"] != mirrored_matrices["observers"]:
        raise AssertionError("Observer order changed")
    human_indices = [original_matrices["observers"].index(doctor) for doctor in PHYSICIANS]
    model_index = original_matrices["observers"].index("Model")
    for kind in ("per_label", "per_pair"):
        for sign in ("pos", "neg", "all"):
            before = np.asarray(original_matrices[f"{kind}_{sign}"])[np.ix_(human_indices, human_indices)]
            after = np.asarray(mirrored_matrices[f"{kind}_{sign}"])[np.ix_(human_indices, human_indices)]
            if not np.allclose(before, after, atol=0, rtol=0, equal_nan=True):
                raise AssertionError("Human-human agreement changed")
    rows = []
    for doctor, human_index in zip(PHYSICIANS, human_indices):
        row = dict(clinician=doctor)
        for sign in ("pos", "neg", "all"):
            before = original_matrices[f"per_label_{sign}"][model_index][human_index]
            after = mirrored_matrices[f"per_label_{sign}"][model_index][human_index]
            row.update({f"original_F1_{sign}": before, f"mirrored_F1_{sign}": after,
                        f"delta_F1_{sign}": after - before})
        rows.append(row)
    comparison.write_csv(args.out_dir / "f1_comparison.csv", rows)
    for case in cases:
        if file_hash(args.clinical_run / case["pair_id"] / "output.nii.gz") != source_hashes[case["pair_id"]]:
            raise AssertionError("Original prediction changed")
    comparison.write_json(args.out_dir / "verification.json", dict(
        num_pairs=args.num_pairs, original_predictions_unchanged=True,
        human_human_matrices_unchanged=True, transform=derived["analysis_transform"],
        original_spatial=summarize(original_rows), mirrored_spatial=summarize(mirrored_rows),
        original_ov=comparison.legacy_summary(original_ov),
        mirrored_ov=comparison.legacy_summary(mirrored_ov)))
    lines = ["# Horizontal Mirror Ablation", "",
             "Exploratory post-hoc analysis, not a verified coordinate correction.", "",
             "[Per-clinician detection F1 comparison](f1_comparison.csv)", "",
             f"[Original OV matrix](ov_original/foundation_{args.num_pairs}/per_label_agreement_all.png)",
             f"[Mirrored OV matrix](ov_mirrored/foundation_{args.num_pairs}/per_label_agreement_all.png)", "",
             "## Mirrored Collages", ""]
    lines.extend(f"- [{case['pair_id']}](predictions/{case['pair_id']}/review.png)" for case in cases)
    (args.out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[mirror] complete: {args.out_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-run", type=Path, required=True)
    parser.add_argument("--clinical-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())