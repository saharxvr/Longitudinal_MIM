from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


SUPPORT_THRESHOLD = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment_module(annotation_root: Path):
    path = annotation_root / "llm_semantic_diff_experiment.py"
    spec = importlib.util.spec_from_file_location("llm_semantic_diff_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_json(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def semantic_signature(data: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (finding["pathology"], finding["change_type"])
            for finding in data["findings"]
        )
    )


def sign_presence(data: dict, positive: bool) -> bool:
    positive_types = {"new", "increased"}
    return any(
        (finding["change_type"] in positive_types) == positive
        for finding in data["findings"]
    )


def pathology_jaccard(left: dict, right: dict) -> float:
    left_set = {finding["pathology"] for finding in left["findings"]}
    right_set = {finding["pathology"] for finding in right["findings"]}
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def binary_dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = int(left.sum() + right.sum())
    return float(2 * np.logical_and(left, right).sum() / denominator) if denominator else 1.0


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def soft_dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(left.sum() + right.sum())
    return float(2 * np.minimum(left, right).sum() / denominator) if denominator else 1.0


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def map_metrics(left: np.ndarray, right: np.ndarray, sign: int) -> dict[str, float]:
    left_sign = np.clip(left * sign, 0.0, 1.0)
    right_sign = np.clip(right * sign, 0.0, 1.0)
    left_support = left_sign >= SUPPORT_THRESHOLD
    right_support = right_sign >= SUPPORT_THRESHOLD
    return {
        "dice": binary_dice(left_support, right_support),
        "iou": binary_iou(left_support, right_support),
        "soft_dice": soft_dice(left_sign, right_sign),
    }


def overlay(image: np.ndarray, signed_map: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(signed_map)))
    normalized = signed_map / max_abs if max_abs else signed_map
    base = np.repeat(image[..., None], 3, axis=2)
    color = np.zeros_like(base)
    color[..., 0] = normalized > 0
    color[..., 1] = normalized < 0
    alpha = np.abs(normalized)[..., None] * 0.72
    return base * (1 - alpha) + color * alpha


def response_title(label: str, response: dict) -> str:
    if response["no_clear_semantic_change"]:
        result = "No clear change"
    else:
        result = "; ".join(
            f"{finding['pathology']}: {finding['change_type']}"
            for finding in response["findings"]
        )
    return f"{label}\n{result}"


def make_collage(
    pair_id: str,
    current: np.ndarray,
    responses: dict[str, dict],
    maps: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    labels = ["baseline_original", "run_01", "run_02", "run_03", "run_04", "run_05"]
    panels = [("Current", np.repeat(current[..., None], 3, axis=2))]
    panels.extend(
        (response_title(label, responses[label]), overlay(current, maps[label]))
        for label in labels
    )
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), dpi=150)
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10, fontweight="bold", linespacing=1.2)
        axis.set_axis_off()
    for axis in axes.flat[len(panels):]:
        axis.set_axis_off()
    fig.suptitle(
        f"{pair_id} - GPT-5.4 precision repeatability; each overlay normalized separately",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95), h_pad=3.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    parser = argparse.ArgumentParser(description="Analyze GPT-5.4 precision repeatability.")
    parser.add_argument("--annotation-root", type=Path, default=annotation_root)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=annotation_root
        / "LLM_Pathology_Heatmaps"
        / "gpt_5_4_precision_repeatability",
    )
    return parser.parse_args()


def verify_snapshot(study_root: Path, protocol: dict) -> None:
    prompt = protocol["prompt_snapshot"]
    prompt_path = study_root / prompt["path"]
    if sha256(prompt_path) != prompt["sha256"]:
        raise ValueError(f"Prompt snapshot hash mismatch: {prompt_path}")
    for item in protocol["files"]:
        path = study_root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot hash mismatch: {path}")
    for run in protocol["runs"]:
        path = study_root / run["prompt_path"]
        if sha256(path) != run["prompt_sha256"]:
            raise ValueError(f"Run prompt hash mismatch: {path}")


def main() -> int:
    args = parse_args()
    protocol = json.loads((args.study_root / "protocol.json").read_text(encoding="utf-8"))
    verify_snapshot(args.study_root, protocol)
    experiment = load_experiment_module(args.annotation_root)
    pair_numbers = protocol["sampling"]["pair_numbers"]
    fresh_labels = [run["run_id"] for run in protocol["runs"]]
    all_labels = ["baseline_original", *fresh_labels]
    analysis_root = args.study_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    responses_by_pair: dict[str, dict[str, dict]] = {}
    maps_by_pair: dict[str, dict[str, np.ndarray]] = {}
    response_manifest = []
    validation_errors = []

    for pair_number in pair_numbers:
        pair_id = f"pair{pair_number}"
        pair_dir = experiment.find_pair_dir(args.annotation_root, pair_number)
        _, current_path = experiment.scan_files(pair_dir)
        responses_by_pair[pair_id] = {}
        maps_by_pair[pair_id] = {}
        for label in all_labels:
            response_path = (
                args.study_root / "baseline_original" / "responses" / f"{pair_id}.json"
                if label == "baseline_original"
                else args.study_root / "runs" / label / "responses" / f"{pair_id}.json"
            )
            try:
                data = experiment.validate_common(
                    json.loads(response_path.read_text(encoding="utf-8")),
                    pair_id,
                )
                signed_map = experiment.render_heatmap_response(data, current_path)
                responses_by_pair[pair_id][label] = data
                maps_by_pair[pair_id][label] = signed_map
                rendered_root = (
                    args.study_root / "baseline_original" / "rendered"
                    if label == "baseline_original"
                    else args.study_root / "runs" / label / "rendered"
                )
                png_path = rendered_root / "heatmaps" / f"{pair_id}_signed_heatmap.png"
                nii_path = rendered_root / "predictions" / pair_id / "output.nii.gz"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                experiment.signed_map_to_rgb(signed_map).save(png_path)
                temporary_nii_path = nii_path.with_name("output.tmp.nii.gz")
                if temporary_nii_path.exists():
                    temporary_nii_path.unlink()
                experiment.save_nifti(
                    signed_map,
                    current_path,
                    temporary_nii_path,
                )
                temporary_nii_path.replace(nii_path)
                roundtrip = nib.load(str(nii_path)).get_fdata()
                if roundtrip.ndim == 3:
                    roundtrip = roundtrip[..., roundtrip.shape[2] // 2]
                max_error = float(np.max(np.abs(roundtrip.T - signed_map)))
                if max_error != 0.0:
                    raise ValueError(f"NIfTI round-trip error {max_error}")
                response_manifest.append(
                    {
                        "run": label,
                        "pair_id": pair_id,
                        "response_path": str(response_path.relative_to(args.study_root)),
                        "response_sha256": sha256(response_path),
                        "canonical_json_sha256": hashlib.sha256(
                            canonical_json(data).encode()
                        ).hexdigest(),
                        "finding_count": len(data["findings"]),
                        "rendered_nifti": str(nii_path.relative_to(args.study_root)),
                        "nifti_roundtrip_max_abs_error": max_error,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                validation_errors.append({"run": label, "pair_id": pair_id, "error": str(exc)})

    if validation_errors:
        (analysis_root / "validation_errors.json").write_text(
            json.dumps(validation_errors, indent=2),
            encoding="utf-8",
        )
        raise ValueError(f"{len(validation_errors)} response(s) failed; see validation_errors.json")

    with (analysis_root / "response_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_manifest[0]))
        writer.writeheader()
        writer.writerows(response_manifest)

    pairwise_rows = []
    baseline_rows = []
    per_pair_rows = []
    for pair_number in pair_numbers:
        pair_id = f"pair{pair_number}"
        responses = responses_by_pair[pair_id]
        maps = maps_by_pair[pair_id]
        for left_label, right_label in itertools.combinations(fresh_labels, 2):
            left = responses[left_label]
            right = responses[right_label]
            left_map = maps[left_label]
            right_map = maps[right_label]
            pos = map_metrics(left_map, right_map, 1)
            neg = map_metrics(left_map, right_map, -1)
            pairwise_rows.append(
                {
                    "pair_id": pair_id,
                    "left_run": left_label,
                    "right_run": right_label,
                    "exact_json_agreement": canonical_json(left) == canonical_json(right),
                    "semantic_signature_agreement": semantic_signature(left)
                    == semantic_signature(right),
                    "any_change_agreement": left["no_clear_semantic_change"]
                    == right["no_clear_semantic_change"],
                    "positive_presence_agreement": sign_presence(left, True)
                    == sign_presence(right, True),
                    "negative_presence_agreement": sign_presence(left, False)
                    == sign_presence(right, False),
                    "pathology_jaccard": pathology_jaccard(left, right),
                    "signed_map_correlation": finite_correlation(left_map, right_map),
                    **{f"positive_{key}": value for key, value in pos.items()},
                    **{f"negative_{key}": value for key, value in neg.items()},
                }
            )

        baseline_response = responses["baseline_original"]
        baseline_map = maps["baseline_original"]
        for run_label in fresh_labels:
            fresh_response = responses[run_label]
            fresh_map = maps[run_label]
            pos = map_metrics(baseline_map, fresh_map, 1)
            neg = map_metrics(baseline_map, fresh_map, -1)
            baseline_rows.append(
                {
                    "pair_id": pair_id,
                    "fresh_run": run_label,
                    "exact_json_agreement": canonical_json(baseline_response)
                    == canonical_json(fresh_response),
                    "semantic_signature_agreement": semantic_signature(baseline_response)
                    == semantic_signature(fresh_response),
                    "any_change_agreement": baseline_response["no_clear_semantic_change"]
                    == fresh_response["no_clear_semantic_change"],
                    "positive_presence_agreement": sign_presence(baseline_response, True)
                    == sign_presence(fresh_response, True),
                    "negative_presence_agreement": sign_presence(baseline_response, False)
                    == sign_presence(fresh_response, False),
                    "pathology_jaccard": pathology_jaccard(
                        baseline_response, fresh_response
                    ),
                    "signed_map_correlation": finite_correlation(
                        baseline_map, fresh_map
                    ),
                    **{f"positive_{key}": value for key, value in pos.items()},
                    **{f"negative_{key}": value for key, value in neg.items()},
                }
            )

        pair_rows = [row for row in pairwise_rows if row["pair_id"] == pair_id]
        signatures = [semantic_signature(responses[label]) for label in fresh_labels]
        modal_count = Counter(signatures).most_common(1)[0][1]
        per_pair_rows.append(
            {
                "pair_id": pair_id,
                "unique_semantic_signatures": len(set(signatures)),
                "modal_semantic_signature_fraction": modal_count / len(fresh_labels),
                "all_fresh_runs_semantically_identical": len(set(signatures)) == 1,
                "all_fresh_runs_exact_json_identical": len(
                    {canonical_json(responses[label]) for label in fresh_labels}
                )
                == 1,
                "pairwise_any_change_agreement": float(
                    np.mean([row["any_change_agreement"] for row in pair_rows])
                ),
                "pairwise_positive_presence_agreement": float(
                    np.mean([row["positive_presence_agreement"] for row in pair_rows])
                ),
                "pairwise_negative_presence_agreement": float(
                    np.mean([row["negative_presence_agreement"] for row in pair_rows])
                ),
                "mean_pathology_jaccard": float(
                    np.mean([row["pathology_jaccard"] for row in pair_rows])
                ),
                "mean_positive_dice": float(
                    np.mean([row["positive_dice"] for row in pair_rows])
                ),
                "mean_negative_dice": float(
                    np.mean([row["negative_dice"] for row in pair_rows])
                ),
                "mean_signed_map_correlation": float(
                    np.mean([row["signed_map_correlation"] for row in pair_rows])
                ),
            }
        )

        _, current_path = experiment.scan_files(
            experiment.find_pair_dir(args.annotation_root, pair_number)
        )
        make_collage(
            pair_id,
            experiment.load_display_array(current_path),
            responses,
            maps,
            analysis_root / "collages" / f"{pair_id}_repeatability.png",
        )

    for name, rows in (
        ("pairwise_fresh_runs.csv", pairwise_rows),
        ("original_baseline_vs_fresh.csv", baseline_rows),
        ("per_pair_summary.csv", per_pair_rows),
    ):
        with (analysis_root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    metric_names = [
        "exact_json_agreement",
        "semantic_signature_agreement",
        "any_change_agreement",
        "positive_presence_agreement",
        "negative_presence_agreement",
        "pathology_jaccard",
        "positive_dice",
        "negative_dice",
        "positive_soft_dice",
        "negative_soft_dice",
        "signed_map_correlation",
    ]
    summary = {
        "study": protocol["study"],
        "analysis_created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": pair_numbers,
        "fresh_runs": len(fresh_labels),
        "fresh_run_pairwise_comparisons": len(pairwise_rows),
        "support_threshold": SUPPORT_THRESHOLD,
        "primary_fresh_run_metrics": {
            name: float(np.mean([float(row[name]) for row in pairwise_rows]))
            for name in metric_names
        },
        "secondary_original_baseline_metrics": {
            name: float(np.mean([float(row[name]) for row in baseline_rows]))
            for name in metric_names
        },
        "pair_level": {
            "pairs_with_one_semantic_signature": sum(
                row["all_fresh_runs_semantically_identical"] for row in per_pair_rows
            ),
            "pairs_with_exact_json_identity_across_all_runs": sum(
                row["all_fresh_runs_exact_json_identical"] for row in per_pair_rows
            ),
            "mean_unique_semantic_signatures": float(
                np.mean([row["unique_semantic_signatures"] for row in per_pair_rows])
            ),
        },
        "validation": {
            "responses_expected": len(all_labels) * len(pair_numbers),
            "responses_validated": len(response_manifest),
            "nifti_roundtrips_exact": all(
                row["nifti_roundtrip_max_abs_error"] == 0.0
                for row in response_manifest
            ),
            "snapshot_hashes_verified": True,
        },
        "interpretation_constraints": {
            "provider_cache_disabled_verified": False,
            "cache_mitigation": protocol["design"]["cache_mitigation"],
            "temperature_and_seed": "not exposed by the agent runtime",
        },
    }
    (analysis_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
