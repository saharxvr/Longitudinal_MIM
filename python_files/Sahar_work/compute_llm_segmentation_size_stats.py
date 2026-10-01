from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_disagreement_levels as gdl
import observer_variability_main_loo_metrics as loo


HUMANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]
VISIBLE_THRESHOLD = 0.10


def load_signed(path: Path, shape: tuple[int, int]) -> np.ndarray:
    output = nib.load(str(path)).get_fdata().astype(np.float32)
    if output.ndim == 3:
        output = output[..., output.shape[2] // 2]
    if output.shape != shape:
        raise ValueError(f"Expected {shape}, found {output.shape}: {path}")
    return output


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    experiments_root = annotation_root / "LLM_Pathology_Heatmaps"
    claude_root = experiments_root / "claude_sonnet_5_full_experiment"
    gpt_root = experiments_root / "gpt_5_4_full_experiment"
    model_root = (
        repo_root
        / "python_files"
        / "Sahar_work"
        / "files"
        / "predictions"
        / "itamar_segs_sq_plus_lmm5_99_100"
    )
    output_root = experiments_root / "claude_gpt_comparison" / "segmentation_size"
    output_root.mkdir(parents=True, exist_ok=True)

    generated_roots = {
        "Claude Ellipses": claude_root / "rendered_ellipses" / "predictions",
        "Claude Direct": claude_root / "rendered_heatmap" / "predictions",
        "Claude Precision": claude_root
        / "rendered_heatmap_precise"
        / "predictions",
        "GPT-5.4 Ellipses": gpt_root / "rendered_ellipses" / "predictions",
        "GPT-5.4 Direct": gpt_root / "rendered_heatmap" / "predictions",
        "GPT-5.4 Precision": gpt_root
        / "rendered_heatmap_precise"
        / "predictions",
    }
    human_indices = {
        human: gdl._build_pair_index(annotation_root / "Annotations" / human)
        for human in HUMANS
    }
    pair_roots = [annotation_root / f"Pairs{batch}" for batch in range(1, 9)]
    coverage: dict[str, dict[str, list[float]]] = {
        observer: {"evaluation": [], "visible": [], "outside_any_human": []}
        for observer in ["Pathologists", "ICU Model", *generated_roots]
    }
    rows = []

    for pair_number in range(1, 101):
        pair_id = f"pair{pair_number}"
        pair_dir = gdl.find_pair_path(pair_roots, pair_number)
        if pair_dir is None:
            raise FileNotFoundError(f"Missing {pair_id}")
        scans = sorted(
            path
            for path in pair_dir.glob("*.nii.gz")
            if not path.name.endswith("_lung_seg.nii.gz")
        )
        shape = tuple(nib.load(str(scans[1])).shape[:2])
        pixels = int(np.prod(shape))

        human_unions = []
        any_human = np.zeros(shape, dtype=bool)
        for human in HUMANS:
            positive, negative = gdl.load_labels_map(
                human_indices[human][pair_number],
                shape,
            )
            union = np.logical_or(positive != 0, negative != 0)
            human_unions.append(union)
            any_human |= union
        human_fraction = float(
            np.mean([union.sum() / pixels for union in human_unions])
        )
        coverage["Pathologists"]["evaluation"].append(human_fraction)
        coverage["Pathologists"]["visible"].append(human_fraction)
        coverage["Pathologists"]["outside_any_human"].append(0.0)
        rows.append(
            {
                "pair_id": pair_id,
                "observer": "Pathologists",
                "evaluation_support_fraction": human_fraction,
                "visible_support_fraction": human_fraction,
                "fraction_of_support_outside_any_human_annotation": 0.0,
            }
        )

        model_path, crop_info_path = loo._resolve_pred(model_root, pair_number)
        if model_path is None or crop_info_path is None:
            raise FileNotFoundError(f"Missing ICU model prediction for {pair_id}")
        model_positive, model_negative = loo.load_model_maps_crop_info(
            model_path,
            crop_info_path,
            shape,
            min_cc_size=0,
            min_cc_intensity=0.0,
        )
        model_support = np.logical_or(model_positive != 0, model_negative != 0)
        model_fraction = float(model_support.sum() / pixels)
        model_outside = (
            float(np.logical_and(model_support, ~any_human).sum() / model_support.sum())
            if model_support.any()
            else 0.0
        )
        coverage["ICU Model"]["evaluation"].append(model_fraction)
        coverage["ICU Model"]["visible"].append(model_fraction)
        coverage["ICU Model"]["outside_any_human"].append(model_outside)
        rows.append(
            {
                "pair_id": pair_id,
                "observer": "ICU Model",
                "evaluation_support_fraction": model_fraction,
                "visible_support_fraction": model_fraction,
                "fraction_of_support_outside_any_human_annotation": model_outside,
            }
        )

        for observer, root in generated_roots.items():
            signed = load_signed(root / pair_id / "output.nii.gz", shape)
            evaluation_support = signed != 0
            visible_support = np.abs(signed) >= VISIBLE_THRESHOLD
            evaluation_fraction = float(evaluation_support.sum() / pixels)
            visible_fraction = float(visible_support.sum() / pixels)
            outside = (
                float(
                    np.logical_and(evaluation_support, ~any_human).sum()
                    / evaluation_support.sum()
                )
                if evaluation_support.any()
                else 0.0
            )
            coverage[observer]["evaluation"].append(evaluation_fraction)
            coverage[observer]["visible"].append(visible_fraction)
            coverage[observer]["outside_any_human"].append(outside)
            rows.append(
                {
                    "pair_id": pair_id,
                    "observer": observer,
                    "evaluation_support_fraction": evaluation_fraction,
                    "visible_support_fraction": visible_fraction,
                    "fraction_of_support_outside_any_human_annotation": outside,
                }
            )

    with (output_root / "per_pair_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pairs": 100,
        "evaluation_support_definition": "all pixels with signed prediction != 0, matching existing observer comparison",
        "visible_support_definition": f"all pixels with abs(signed prediction) >= {VISIBLE_THRESHOLD}",
        "pathologist_definition": "mean individual annotated union area per pair",
        "observers": {
            observer: {
                "evaluation_support_fraction": summarize(values["evaluation"]),
                "visible_support_fraction": summarize(values["visible"]),
                "fraction_of_support_outside_any_human_annotation": summarize(
                    values["outside_any_human"]
                ),
            }
            for observer, values in coverage.items()
        },
    }
    human_median = summary["observers"]["Pathologists"][
        "evaluation_support_fraction"
    ]["median"]
    model_median = summary["observers"]["ICU Model"][
        "evaluation_support_fraction"
    ]["median"]
    summary["median_coverage_ratios"] = {
        observer: {
            "vs_pathologist": (
                metrics["evaluation_support_fraction"]["median"] / human_median
                if human_median
                else None
            ),
            "vs_icu_model": (
                metrics["evaluation_support_fraction"]["median"] / model_median
                if model_median
                else None
            ),
        }
        for observer, metrics in summary["observers"].items()
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
