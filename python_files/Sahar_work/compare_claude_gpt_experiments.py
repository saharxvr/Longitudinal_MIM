from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_llm_annotator_experiment as comparison
import get_disagreement_levels as gdl
import make_llm_experiment_collages as collages
import observer_variability_main_loo_metrics as loo
import observer_variability_nathalia_thaer_model as base


HUMANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]
ARMS = {
    "ellipses": ("rendered_ellipses", "responses_ellipses"),
    "heatmap": ("rendered_heatmap", "responses_heatmap"),
    "precision": ("rendered_heatmap_precise", "responses_heatmap_precise"),
}


def detection_pai(left: np.ndarray, right: np.ndarray) -> float:
    agreement, disagreement = base.get_pairwise_detections(left, right)
    denominator = 2 * agreement + disagreement
    return float(2 * agreement / denominator) if denominator else 1.0


def response_label(response: dict) -> str:
    if response["no_clear_semantic_change"]:
        return "No clear change"
    return "; ".join(
        f"{item['pathology']}: {item['change_type']}"
        for item in response["findings"]
    )


def load_response(root: Path, response_dir: str, pair_id: str) -> dict:
    return json.loads(
        (root / response_dir / f"{pair_id}.json").read_text(encoding="utf-8")
    )


def consensus_counts(
    observed: tuple[np.ndarray, np.ndarray],
    human_maps: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, int]:
    output = {}
    for sign, map_index in (("pos", 0), ("neg", 1)):
        values = base.get_sensitivity_at_consensus_levels(
            observed[map_index],
            [maps[map_index] for maps in human_maps],
        )[-1]
        output[f"level5_{sign}_detected"] = int(values[0])
        output[f"level5_{sign}_total"] = int(values[1])
    return output


def make_combined_collage(
    pair_number: int,
    prior: np.ndarray,
    current: np.ndarray,
    consensus: np.ndarray,
    model: np.ndarray,
    claude_maps: dict[str, np.ndarray],
    gpt_maps: dict[str, np.ndarray],
    claude_responses: dict[str, dict],
    gpt_responses: dict[str, dict],
    output_path: Path,
) -> None:
    panels = [
        ("Prior", np.repeat(prior[..., None], 3, axis=2)),
        ("Current", np.repeat(current[..., None], 3, axis=2)),
        ("5-pathologist vote", collages.overlay_rgb(current, consensus)),
        ("ICU model", collages.overlay_rgb(current, model)),
    ]
    for label, maps, responses in (
        ("Claude", claude_maps, claude_responses),
        ("GPT-5.4", gpt_maps, gpt_responses),
    ):
        for arm, arm_label in (
            ("ellipses", "ellipses"),
            ("heatmap", "direct"),
            ("precision", "precision"),
        ):
            panels.append(
                (
                    f"{label} {arm_label}\n{response_label(responses[arm])}",
                    collages.overlay_rgb(current, maps[arm]),
                )
            )

    fig, axes = plt.subplots(3, 4, figsize=(24, 16), dpi=130)
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10, fontweight="bold", linespacing=1.2)
        axis.set_axis_off()
    for axis in axes.flat[len(panels):]:
        axis.set_axis_off()
    fig.suptitle(
        f"Pair {pair_number} - independently normalized overlays; "
        "red: worsening, green: improving",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96), h_pad=3.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    experiments_root = annotation_root / "LLM_Pathology_Heatmaps"
    parser = argparse.ArgumentParser(
        description="Compare blinded Claude and GPT semantic-difference experiments."
    )
    parser.add_argument("--annotation-root", type=Path, default=annotation_root)
    parser.add_argument(
        "--claude-root",
        type=Path,
        default=experiments_root / "claude_sonnet_5_full_experiment",
    )
    parser.add_argument(
        "--gpt-root",
        type=Path,
        default=experiments_root / "gpt_5_4_full_experiment",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=repo_root
        / "python_files"
        / "Sahar_work"
        / "files"
        / "predictions"
        / "itamar_segs_sq_plus_lmm5_99_100",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiments_root / "claude_gpt_comparison",
    )
    parser.add_argument("--num-pairs", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = collages.load_experiment_module(args.annotation_root)
    human_indices = {
        human: gdl._build_pair_index(args.annotation_root / "Annotations" / human)
        for human in HUMANS
    }
    rows = []

    for pair_number in range(1, args.num_pairs + 1):
        pair_id = f"pair{pair_number}"
        pair_dir = experiment.find_pair_dir(args.annotation_root, pair_number)
        prior_path, current_path = experiment.scan_files(pair_dir)
        native_shape = tuple(nib.load(str(current_path)).shape[:2])
        prior = experiment.load_display_array(prior_path)
        current = experiment.load_display_array(current_path)

        human_maps = [
            gdl.load_labels_map(human_indices[human][pair_number], native_shape)
            for human in HUMANS
        ]
        model_path, crop_info_path = loo._resolve_pred(args.model_root, pair_number)
        if model_path is None or crop_info_path is None:
            raise FileNotFoundError(f"Missing ICU model output for {pair_id}")
        model_maps = loo.load_model_maps_crop_info(
            model_path,
            crop_info_path,
            native_shape,
            min_cc_size=0,
            min_cc_intensity=0.0,
        )
        model_display = collages.load_model_signed_map(
            model_path,
            crop_info_path,
            native_shape,
        ).T
        consensus_display = collages.pathologist_consensus(
            human_indices,
            pair_number,
            native_shape,
        ).T

        native_maps: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            "claude": {},
            "gpt": {},
        }
        display_maps: dict[str, dict[str, np.ndarray]] = {"claude": {}, "gpt": {}}
        responses: dict[str, dict[str, dict]] = {"claude": {}, "gpt": {}}
        for experiment_name, root in (
            ("claude", args.claude_root),
            ("gpt", args.gpt_root),
        ):
            for arm, (rendered_dir, response_dir) in ARMS.items():
                prediction = root / rendered_dir / "predictions" / pair_id / "output.nii.gz"
                native_maps[experiment_name][arm] = comparison.load_full_prediction(
                    prediction,
                    native_shape,
                )
                display_maps[experiment_name][arm] = collages.load_native_map(prediction).T
                responses[experiment_name][arm] = load_response(
                    root,
                    response_dir,
                    pair_id,
                )

        row: dict[str, object] = {"pair_id": pair_id}
        row.update(
            {
                f"model_{key}": value
                for key, value in consensus_counts(model_maps, human_maps).items()
            }
        )
        for arm in ARMS:
            claude_pos, claude_neg = native_maps["claude"][arm]
            gpt_pos, gpt_neg = native_maps["gpt"][arm]
            row[f"{arm}_positive_pai"] = detection_pai(claude_pos, gpt_pos)
            row[f"{arm}_negative_pai"] = detection_pai(claude_neg, gpt_neg)
            row[f"{arm}_positive_presence_agrees"] = bool(np.any(claude_pos)) == bool(
                np.any(gpt_pos)
            )
            row[f"{arm}_negative_presence_agrees"] = bool(np.any(claude_neg)) == bool(
                np.any(gpt_neg)
            )
            for name, maps in (
                ("claude", native_maps["claude"][arm]),
                ("gpt", native_maps["gpt"][arm]),
            ):
                row.update(
                    {
                        f"{name}_{arm}_{key}": value
                        for key, value in consensus_counts(maps, human_maps).items()
                    }
                )
                row[f"{name}_{arm}_label"] = response_label(responses[name][arm])

        rows.append(row)
        make_combined_collage(
            pair_number,
            prior,
            current,
            consensus_display,
            model_display,
            display_maps["claude"],
            display_maps["gpt"],
            responses["claude"],
            responses["gpt"],
            args.output_root / "collages" / f"{pair_id}_collage.png",
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "per_pair_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pairs_processed": len(rows),
        "corresponding_arm_agreement": {
            arm: {
                "mean_positive_detection_pai": float(
                    np.mean([row[f"{arm}_positive_pai"] for row in rows])
                ),
                "mean_negative_detection_pai": float(
                    np.mean([row[f"{arm}_negative_pai"] for row in rows])
                ),
                "positive_presence_agreement": float(
                    np.mean([row[f"{arm}_positive_presence_agrees"] for row in rows])
                ),
                "negative_presence_agreement": float(
                    np.mean([row[f"{arm}_negative_presence_agrees"] for row in rows])
                ),
            }
            for arm in ARMS
        },
        "coordinate_audits": {
            "claude": json.loads(
                (
                    args.claude_root
                    / "collages_precision"
                    / "coordinate_audit_summary.json"
                ).read_text(encoding="utf-8")
            ),
            "gpt": json.loads(
                (
                    args.gpt_root
                    / "collages_precision"
                    / "coordinate_audit_summary.json"
                ).read_text(encoding="utf-8")
            ),
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
