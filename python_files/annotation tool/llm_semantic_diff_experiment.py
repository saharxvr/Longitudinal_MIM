from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.draw import ellipse


PAIR_RE = re.compile(r"^pair(\d+)$", re.IGNORECASE)
PATHOLOGIES = {
    "Pleural Effusion",
    "Fluid Overload",
    "Consolidation",
    "Pneumothorax",
    "Other",
}
CHANGE_SIGNS = {
    "new": 1.0,
    "increased": 1.0,
    "resolved": -1.0,
    "decreased": -1.0,
}
CONFIDENCE_WEIGHTS = {"low": 0.4, "medium": 0.7, "high": 1.0}
CANVAS_SIZE = 792


def find_pair_dir(annotation_root: Path, pair_number: int) -> Path:
    for batch_number in range(1, 9):
        candidate = annotation_root / f"Pairs{batch_number}" / f"pair{pair_number}"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find pair{pair_number} under {annotation_root}")


def scan_files(pair_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in pair_dir.glob("*.nii.gz")
        if not path.name.endswith("_lung_seg.nii.gz")
    )
    if len(files) != 2:
        raise ValueError(f"Expected exactly two non-segmentation scans in {pair_dir}; found {len(files)}")
    return files


def load_display_array(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata().T
    if data.ndim == 3:
        data = data[:, :, data.shape[2] // 2]
    data = np.asarray(data, dtype=np.float32)
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    if data_max <= data_min:
        raise ValueError(f"Constant-valued image cannot be displayed: {path}")
    return (data - data_min) / (data_max - data_min)


def display_image(path: Path, size: int = CANVAS_SIZE) -> Image.Image:
    data = (load_display_array(path) * 255).astype(np.uint8)
    return Image.fromarray(data, mode="L").resize((size, size), Image.Resampling.BILINEAR)


def create_pair_sheet(prior: Image.Image, current: Image.Image) -> Image.Image:
    header_height = 44
    gap = 12
    sheet = Image.new("RGB", (prior.width + current.width + gap, prior.height + header_height), "black")
    sheet.paste(prior.convert("RGB"), (0, header_height))
    sheet.paste(current.convert("RGB"), (prior.width + gap, header_height))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    draw.text((prior.width // 2, 10), "PRIOR", fill="white", anchor="ma", font=font)
    draw.text(
        (prior.width + gap + current.width // 2, 10),
        "CURRENT",
        fill="white",
        anchor="ma",
        font=font,
    )
    return sheet


def prepare_inputs(annotation_root: Path, output_root: Path, overwrite: bool) -> None:
    inputs_root = output_root / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []

    for pair_number in range(1, 101):
        pair_id = f"pair{pair_number}"
        pair_dir = find_pair_dir(annotation_root, pair_number)
        prior_path, current_path = scan_files(pair_dir)
        pair_output = inputs_root / pair_id
        pair_output.mkdir(parents=True, exist_ok=True)

        prior_output = pair_output / "prior.png"
        current_output = pair_output / "current.png"
        sheet_output = pair_output / "prior_current.png"
        if overwrite or not all(path.exists() for path in (prior_output, current_output, sheet_output)):
            prior = display_image(prior_path)
            current = display_image(current_path)
            prior.save(prior_output)
            current.save(current_output)
            create_pair_sheet(prior, current).save(sheet_output)

        current_nii = nib.load(str(current_path))
        manifest_rows.append(
            {
                "pair_id": pair_id,
                "input_sheet": str(sheet_output.relative_to(output_root)),
                "prior_image": str(prior_output.relative_to(output_root)),
                "current_image": str(current_output.relative_to(output_root)),
                "source_batch": pair_dir.parent.name,
                "current_height": int(current_nii.shape[1]),
                "current_width": int(current_nii.shape[0]),
            }
        )

    with (inputs_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def require_number(value: object, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def validate_common(data: object, expected_pair_id: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Response must be a JSON object")
    if data.get("pair_id") != expected_pair_id:
        raise ValueError(f"Expected pair_id={expected_pair_id!r}; got {data.get('pair_id')!r}")
    if not isinstance(data.get("no_clear_semantic_change"), bool):
        raise ValueError("no_clear_semantic_change must be boolean")
    findings = data.get("findings")
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    if data["no_clear_semantic_change"] != (len(findings) == 0):
        raise ValueError("no_clear_semantic_change must be true exactly when findings is empty")
    return data


def validate_finding(finding: object, index: int) -> dict:
    if not isinstance(finding, dict):
        raise ValueError(f"findings[{index}] must be an object")
    if finding.get("pathology") not in PATHOLOGIES:
        raise ValueError(f"findings[{index}].pathology is not an allowed value")
    if finding.get("pathology") == "Other" and not str(finding.get("pathology_other", "")).strip():
        raise ValueError(f"findings[{index}].pathology_other is required for Other")
    if finding.get("change_type") not in CHANGE_SIGNS:
        raise ValueError(f"findings[{index}].change_type is not an allowed value")
    if finding.get("confidence") not in CONFIDENCE_WEIGHTS:
        raise ValueError(f"findings[{index}].confidence is not an allowed value")
    if not str(finding.get("rationale", "")).strip():
        raise ValueError(f"findings[{index}].rationale is required")
    return finding


def normalized_ellipse_to_annotation(region: dict, finding: dict) -> dict:
    cx = require_number(region.get("cx"), "ellipse.cx", 0.0, 1.0)
    cy = require_number(region.get("cy"), "ellipse.cy", 0.0, 1.0)
    rx = require_number(region.get("rx"), "ellipse.rx", 0.001, 0.5)
    ry = require_number(region.get("ry"), "ellipse.ry", 0.001, 0.5)
    angle = require_number(region.get("angle", 0.0), "ellipse.angle", -180.0, 180.0)
    change_type = finding["change_type"]
    pathology = finding["pathology"]
    return {
        "cx": cx * CANVAS_SIZE,
        "cy": cy * CANVAS_SIZE,
        "rx": rx * CANVAS_SIZE,
        "ry": ry * CANVAS_SIZE,
        "angle": angle,
        "label": "Appearance" if CHANGE_SIGNS[change_type] > 0 else "Disappearance",
        "comment": (
            f"{change_type}; confidence={finding['confidence']}; "
            f"{finding['rationale']}"
        ),
        "tag": pathology,
        "tag_other": str(finding.get("pathology_other", "")).strip() if pathology == "Other" else "",
    }


def annotation_to_map(annotation: dict, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    cx = float(annotation["cx"]) / CANVAS_SIZE * width
    cy = float(annotation["cy"]) / CANVAS_SIZE * height
    rx = float(annotation["rx"]) / CANVAS_SIZE * width
    ry = float(annotation["ry"]) / CANVAS_SIZE * height
    rr, cc = ellipse(
        cy,
        cx,
        ry,
        rx,
        shape=shape,
        rotation=np.deg2rad(float(annotation["angle"])),
    )
    output = np.zeros(shape, dtype=np.float32)
    output[rr, cc] = 1.0 if annotation["label"] == "Appearance" else -1.0
    return output


def canvas_annotation_to_native(annotation: dict, native_shape: tuple[int, int]) -> dict:
    native_rows, native_columns = native_shape
    native = dict(annotation)
    native["cx"] = float(annotation["cx"]) / CANVAS_SIZE * native_rows
    native["cy"] = float(annotation["cy"]) / CANVAS_SIZE * native_columns
    native["rx"] = float(annotation["rx"]) / CANVAS_SIZE * native_rows
    native["ry"] = float(annotation["ry"]) / CANVAS_SIZE * native_columns
    native["angle"] = -float(annotation["angle"])
    return native


def render_ellipse_response(data: dict, current_path: Path) -> tuple[list[dict], np.ndarray]:
    annotations: list[dict] = []
    display_shape = load_display_array(current_path).shape
    signed_map = np.zeros(display_shape, dtype=np.float32)
    for finding_index, raw_finding in enumerate(data["findings"]):
        finding = validate_finding(raw_finding, finding_index)
        regions = finding.get("ellipses")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"findings[{finding_index}].ellipses must be a non-empty list")
        for region in regions:
            if not isinstance(region, dict):
                raise ValueError(f"findings[{finding_index}].ellipses entries must be objects")
            annotation = normalized_ellipse_to_annotation(region, finding)
            annotations.append(annotation)
            region_map = annotation_to_map(annotation, display_shape)
            weight = CONFIDENCE_WEIGHTS[finding["confidence"]]
            positive = region_map > 0
            negative = region_map < 0
            signed_map[positive] = np.maximum(signed_map[positive], weight)
            signed_map[negative] = np.minimum(signed_map[negative], -weight)
    return annotations, signed_map


def render_heatmap_response(data: dict, current_path: Path) -> np.ndarray:
    height, width = load_display_array(current_path).shape
    yy, xx = np.mgrid[0:height, 0:width]
    signed_map = np.zeros((height, width), dtype=np.float32)

    for finding_index, raw_finding in enumerate(data["findings"]):
        finding = validate_finding(raw_finding, finding_index)
        points = finding.get("heatmap_points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"findings[{finding_index}].heatmap_points must be a non-empty list")
        sign = CHANGE_SIGNS[finding["change_type"]]
        confidence_cap = CONFIDENCE_WEIGHTS[finding["confidence"]]
        finding_map = np.zeros_like(signed_map)
        for point in points:
            if not isinstance(point, dict):
                raise ValueError(f"findings[{finding_index}].heatmap_points entries must be objects")
            x = require_number(point.get("x"), "heatmap_point.x", 0.0, 1.0) * width
            y = require_number(point.get("y"), "heatmap_point.y", 0.0, 1.0) * height
            radius = require_number(point.get("radius"), "heatmap_point.radius", 0.005, 0.5)
            strength = require_number(point.get("strength"), "heatmap_point.strength", 0.01, 1.0)
            sigma = radius * min(height, width)
            gaussian = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)))
            finding_map = np.maximum(finding_map, gaussian.astype(np.float32) * strength)
        signed_map += sign * np.minimum(finding_map, confidence_cap)
    return np.clip(signed_map, -1.0, 1.0)


def signed_map_to_rgb(signed_map: np.ndarray) -> Image.Image:
    rgb = np.zeros((*signed_map.shape, 3), dtype=np.uint8)
    positive = signed_map > 0
    negative = signed_map < 0
    rgb[..., 0][positive] = np.round(signed_map[positive] * 255).astype(np.uint8)
    rgb[..., 1][negative] = np.round(-signed_map[negative] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_nifti(display_map: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference = nib.load(str(reference_path))
    nifti_map = display_map.T
    if len(reference.shape) == 3:
        nifti_map = nifti_map[:, :, np.newaxis]
    image = nib.Nifti1Image(nifti_map.astype(np.float32), reference.affine, reference.header)
    image.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def render_responses(
    arm: str,
    annotation_root: Path,
    output_root: Path,
    responses_root: Path,
    annotator_name: str,
    rendered_name: str | None = None,
) -> None:
    if arm not in {"ellipses", "heatmap"}:
        raise ValueError(f"Unsupported arm: {arm}")

    rendered_root = output_root / f"rendered_{rendered_name or arm}"
    annotations_root = rendered_root / "annotations" / annotator_name
    native_annotations_root = rendered_root / "annotations_native" / annotator_name
    predictions_root = rendered_root / "predictions"
    heatmaps_root = rendered_root / "heatmaps"
    output_directories = [annotations_root, predictions_root, heatmaps_root]
    if arm == "ellipses":
        output_directories.append(native_annotations_root)
    for path in output_directories:
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    errors = []
    for pair_number in range(1, 101):
        pair_id = f"pair{pair_number}"
        response_path = responses_root / f"{pair_id}.json"
        try:
            data = validate_common(
                json.loads(response_path.read_text(encoding="utf-8")),
                pair_id,
            )
            pair_dir = find_pair_dir(annotation_root, pair_number)
            prior_path, current_path = scan_files(pair_dir)
            del prior_path

            if arm == "ellipses":
                annotations, signed_map = render_ellipse_response(data, current_path)
                pair_header = (
                    f"{scan_files(pair_dir)[0].name.removesuffix('.nii.gz')} | "
                    f"{current_path.name.removesuffix('.nii.gz')}"
                )
                annotation_payload = [pair_header, *annotations]
                annotation_path = annotations_root / f"pair {pair_number} {annotator_name}.json"
                annotation_path.write_text(
                    json.dumps(annotation_payload, indent=4),
                    encoding="utf-8",
                )
                native_shape = tuple(nib.load(str(current_path)).shape[:2])
                native_annotation_path = (
                    native_annotations_root / f"pair {pair_number} {annotator_name}.json"
                )
                native_annotation_path.write_text(
                    json.dumps(
                        [
                            pair_header,
                            *[
                                canvas_annotation_to_native(annotation, native_shape)
                                for annotation in annotations
                            ],
                        ],
                        indent=4,
                    ),
                    encoding="utf-8",
                )
            else:
                signed_map = render_heatmap_response(data, current_path)
                annotation_path = None
                native_annotation_path = None

            heatmap_path = heatmaps_root / f"{pair_id}_signed_heatmap.png"
            signed_map_to_rgb(signed_map).save(heatmap_path)
            nifti_path = predictions_root / pair_id / "output.nii.gz"
            save_nifti(signed_map, current_path, nifti_path)
            manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "status": "ok",
                    "no_clear_semantic_change": data["no_clear_semantic_change"],
                    "finding_count": len(data["findings"]),
                    "response_json": str(response_path),
                    "annotation_canvas_792_json": str(annotation_path) if annotation_path else "",
                    "annotation_native_json": (
                        str(native_annotation_path) if native_annotation_path else ""
                    ),
                    "signed_heatmap_png": str(heatmap_path),
                    "signed_heatmap_nifti": str(nifti_path),
                    "error": "",
                }
            )
        except Exception as exc:
            errors.append(f"{pair_id}: {exc}")
            manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "status": "error",
                    "no_clear_semantic_change": "",
                    "finding_count": "",
                    "response_json": str(response_path),
                    "annotation_canvas_792_json": "",
                    "annotation_native_json": "",
                    "signed_heatmap_png": "",
                    "signed_heatmap_nifti": "",
                    "error": str(exc),
                }
            )

    manifest_path = rendered_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    if errors:
        raise ValueError(f"{len(errors)} response(s) failed validation. See {manifest_path}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare and render the blinded LLM semantic-diff experiment.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=script_dir,
        help="Directory containing Pairs1 through Pairs8.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "LLM_Pathology_Heatmaps" / "claude_sonnet_5_full_experiment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--overwrite", action="store_true")

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--arm", choices=("ellipses", "heatmap"), required=True)
    render_parser.add_argument("--responses-root", type=Path, required=True)
    render_parser.add_argument("--annotator-name", default="Claude_Sonnet_5")
    render_parser.add_argument(
        "--rendered-name",
        help="Optional output suffix, for example heatmap_precise.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare_inputs(args.annotation_root, args.output_root, args.overwrite)
    else:
        render_responses(
            args.arm,
            args.annotation_root,
            args.output_root,
            args.responses_root,
            args.annotator_name,
            args.rendered_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
