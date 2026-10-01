from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_disagreement_levels as gdl
import observer_variability_main_loo_metrics as loo


HUMANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]


def load_experiment_module(annotation_root: Path):
    path = annotation_root / "llm_semantic_diff_experiment.py"
    spec = importlib.util.spec_from_file_location("llm_semantic_diff_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_map(path: Path) -> np.ndarray:
    output = nib.load(str(path)).get_fdata().astype(np.float32)
    if output.ndim == 3:
        output = output[..., output.shape[2] // 2]
    return output


def load_model_signed_map(
    model_path: Path,
    crop_info_path: Path,
    native_shape: tuple[int, int],
) -> np.ndarray:
    from skimage.transform import resize

    crop = json.loads(crop_info_path.read_text(encoding="utf-8"))["current"]
    output = nib.load(str(model_path)).get_fdata()
    if output.ndim == 3:
        output = output[:, :, 0]
    output = output.T
    square_size = int(crop["square_size"])
    resized = resize(
        output,
        (square_size, square_size),
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    )
    canvas = np.zeros((int(crop["orig_h"]), int(crop["orig_w"])), dtype=np.float32)
    row = int(crop["sq_x_min"])
    column = int(crop["sq_y_min"])
    canvas[row:row + square_size, column:column + square_size] = resized
    native = canvas.T
    if native.shape != native_shape:
        height, width = native_shape
        native = np.asarray(
            Image.fromarray(native, mode="F").resize(
                (width, height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    return native


def pathologist_consensus(
    annotation_indices: dict[str, dict[int, Path]],
    pair_number: int,
    native_shape: tuple[int, int],
) -> np.ndarray:
    positive = np.zeros(native_shape, dtype=np.float32)
    negative = np.zeros(native_shape, dtype=np.float32)
    for human in HUMANS:
        path = annotation_indices[human].get(pair_number)
        if path is None:
            raise FileNotFoundError(f"Missing {human} annotation for pair{pair_number}")
        pos_map, neg_map = gdl.load_labels_map(path, native_shape)
        positive += pos_map != 0
        negative += neg_map != 0
    return np.clip((positive - negative) / len(HUMANS), -1.0, 1.0)


def overlay_rgb(image: np.ndarray, signed_map: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(signed_map)))
    normalized_map = signed_map / max_abs if max_abs > 0.0 else signed_map
    base = np.repeat(np.clip(image[..., None], 0.0, 1.0), 3, axis=2)
    color = np.zeros_like(base)
    color[..., 0] = normalized_map > 0
    color[..., 1] = normalized_map < 0
    alpha = np.abs(normalized_map)[..., None] * 0.72
    return base * (1.0 - alpha) + color * alpha


def response_label(response: dict) -> str:
    if response["no_clear_semantic_change"]:
        return "No clear change"
    return "\n".join(
        f"{finding['pathology']}: {finding['change_type']} ({finding['confidence']})"
        for finding in response["findings"]
    )


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def make_pair_collage(
    pair_number: int,
    prior: np.ndarray,
    current: np.ndarray,
    consensus: np.ndarray,
    model: np.ndarray,
    ellipse_map: np.ndarray,
    heatmap: np.ndarray,
    precise_heatmap: np.ndarray,
    ellipse_response: dict,
    heatmap_response: dict,
    precise_response: dict,
    annotator_label: str,
    output_path: Path,
) -> None:
    panels = [
        ("Prior", np.repeat(prior[..., None], 3, axis=2)),
        ("Current", np.repeat(current[..., None], 3, axis=2)),
        ("5-pathologist vote", overlay_rgb(current, consensus)),
        ("ICU model", overlay_rgb(current, model)),
        (
            f"{annotator_label} ellipses\n{response_label(ellipse_response)}",
            overlay_rgb(current, ellipse_map),
        ),
        (
            f"{annotator_label} direct heatmap\n{response_label(heatmap_response)}",
            overlay_rgb(current, heatmap),
        ),
        (
            f"{annotator_label} precision heatmap\n{response_label(precise_response)}",
            overlay_rgb(current, precise_heatmap),
        ),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 11), dpi=150)
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=11, fontweight="bold", linespacing=1.25)
        axis.set_axis_off()
    for axis in axes.flat[len(panels):]:
        axis.set_axis_off()
    fig.suptitle(
        f"Pair {pair_number} — each overlay normalized separately; "
        "red: appearance/increase, green: disappearance/decrease",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=3.8, w_pad=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    experiment_root = (
        annotation_root
        / "LLM_Pathology_Heatmaps"
        / "claude_sonnet_5_full_experiment"
    )
    parser = argparse.ArgumentParser(description="Audit coordinates and create experiment collages.")
    parser.add_argument("--annotation-root", type=Path, default=annotation_root)
    parser.add_argument("--experiment-root", type=Path, default=experiment_root)
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
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--annotator-label", default="Claude")
    parser.add_argument("--annotator-file-name", default="Claude_Sonnet_5")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=experiment_root / "collages_precision",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment = load_experiment_module(args.annotation_root)
    annotation_indices = {
        human: gdl._build_pair_index(args.annotation_root / "Annotations" / human)
        for human in HUMANS
    }
    audit_rows = []

    for pair_number in range(1, args.num_pairs + 1):
        pair_id = f"pair{pair_number}"
        pair_dir = experiment.find_pair_dir(args.annotation_root, pair_number)
        prior_path, current_path = experiment.scan_files(pair_dir)
        prior = experiment.load_display_array(prior_path)
        current = experiment.load_display_array(current_path)
        native_shape = tuple(nib.load(str(current_path)).shape[:2])

        ellipse_response = json.loads(
            (args.experiment_root / "responses_ellipses" / f"{pair_id}.json").read_text(
                encoding="utf-8"
            )
        )
        heatmap_response = json.loads(
            (args.experiment_root / "responses_heatmap" / f"{pair_id}.json").read_text(
                encoding="utf-8"
            )
        )
        precise_response = json.loads(
            (
                args.experiment_root
                / "responses_heatmap_precise"
                / f"{pair_id}.json"
            ).read_text(encoding="utf-8")
        )
        ellipse_annotations, expected_ellipse = experiment.render_ellipse_response(
            ellipse_response,
            current_path,
        )
        expected_heatmap = experiment.render_heatmap_response(
            heatmap_response,
            current_path,
        )
        expected_precise = experiment.render_heatmap_response(
            precise_response,
            current_path,
        )

        ellipse_native = load_native_map(
            args.experiment_root
            / "rendered_ellipses"
            / "predictions"
            / pair_id
            / "output.nii.gz"
        )
        heatmap_native = load_native_map(
            args.experiment_root
            / "rendered_heatmap"
            / "predictions"
            / pair_id
            / "output.nii.gz"
        )
        precise_native = load_native_map(
            args.experiment_root
            / "rendered_heatmap_precise"
            / "predictions"
            / pair_id
            / "output.nii.gz"
        )
        native_annotation_path = (
            args.experiment_root
            / "rendered_ellipses"
            / "annotations_native"
            / args.annotator_file_name
            / f"pair {pair_number} {args.annotator_file_name}.json"
        )
        native_positive, native_negative = gdl.load_labels_map(
            native_annotation_path,
            native_shape,
        )
        expected_native_positive = np.zeros(native_shape, dtype=bool)
        expected_native_negative = np.zeros(native_shape, dtype=bool)
        for annotation in ellipse_annotations:
            annotation_native_mask = experiment.annotation_to_map(
                annotation,
                current.shape,
            ).T
            if annotation["label"] == "Appearance":
                expected_native_positive |= annotation_native_mask > 0
            else:
                expected_native_negative |= annotation_native_mask < 0

        ellipse_display = ellipse_native.T
        heatmap_display = heatmap_native.T
        precise_display = precise_native.T
        ellipse_error = float(np.max(np.abs(ellipse_display - expected_ellipse)))
        heatmap_error = float(np.max(np.abs(heatmap_display - expected_heatmap)))
        precise_error = float(np.max(np.abs(precise_display - expected_precise)))
        native_positive_iou = binary_iou(
            native_positive != 0,
            expected_native_positive,
        )
        native_negative_iou = binary_iou(
            native_negative != 0,
            expected_native_negative,
        )

        model_path, crop_info_path = loo._resolve_pred(args.model_root, pair_number)
        if model_path is None or crop_info_path is None:
            raise FileNotFoundError(f"Missing ICU model output for {pair_id}")
        model_display = load_model_signed_map(model_path, crop_info_path, native_shape).T
        consensus_display = pathologist_consensus(
            annotation_indices,
            pair_number,
            native_shape,
        ).T

        make_pair_collage(
            pair_number,
            prior,
            current,
            consensus_display,
            model_display,
            ellipse_display,
            heatmap_display,
            precise_display,
            ellipse_response,
            heatmap_response,
            precise_response,
            args.annotator_label,
            args.output_root / "per_pair" / f"{pair_id}_collage.png",
        )
        audit_rows.append(
            {
                "pair_id": pair_id,
                "display_height": current.shape[0],
                "display_width": current.shape[1],
                "native_rows": native_shape[0],
                "native_columns": native_shape[1],
                "is_non_square": current.shape[0] != current.shape[1],
                "ellipse_display_vs_nifti_max_abs_error": ellipse_error,
                "heatmap_display_vs_nifti_max_abs_error": heatmap_error,
                "precision_display_vs_nifti_max_abs_error": precise_error,
                "ellipse_native_json_positive_iou": native_positive_iou,
                "ellipse_native_json_negative_iou": native_negative_iou,
                "status": (
                    "ok"
                    if ellipse_error == 0.0
                    and heatmap_error == 0.0
                    and precise_error == 0.0
                    and native_positive_iou == 1.0
                    and native_negative_iou == 1.0
                    else "error"
                ),
            }
        )

    audit_path = args.output_root / "coordinate_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    failures = [row for row in audit_rows if row["status"] != "ok"]
    summary = {
        "pairs_audited": len(audit_rows),
        "pairs_passed": len(audit_rows) - len(failures),
        "pairs_failed": len(failures),
        "non_square_pairs": [
            row["pair_id"] for row in audit_rows if row["is_non_square"]
        ],
        "coordinate_contract": {
            "response_coordinates": "normalized to each current image independently",
            "display_maps": "native current image transposed exactly as the annotation tool displays it",
            "canvas_annotations": "792x792 annotation-tool coordinates",
            "native_annotations": "native NIfTI coordinates with transpose-aware angle conversion",
            "signed_colors": "red=positive/worsening, green=negative/improving",
        },
        "failures": failures,
    }
    (args.output_root / "coordinate_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} coordinate audit(s) failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
