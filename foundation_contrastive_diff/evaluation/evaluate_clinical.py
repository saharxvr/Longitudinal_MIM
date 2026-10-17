"""Frozen comparison-head inference on the 100 clinician-annotated ICU pairs."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from datasets.pair_dataset import _norm01, _to_chw


PHYSICIANS = ("Avi", "Benny", "Sigal", "Smadar", "Nitzan")


def clinical_tools():
    path = (Path(__file__).resolve().parents[2] / "python_files" / "archive" /
            "evaluation" / "Evaluation" / "Observer_Variability_sq_crop.py")
    spec = importlib.util.spec_from_file_location("clinical_ov", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_cases(clinical_root, count=100):
    root = Path(clinical_root)
    tools = clinical_tools()
    roots = [str(root / f"Pairs{number}") for number in range(1, 9)]
    cases = []
    for number in range(1, count + 1):
        matches = sorted({(Path(parent) / prefix).resolve() for parent in roots
                  for prefix in (f"pair{number}", f"Pair{number}")
                  if (Path(parent) / prefix).is_dir()})
        if len(matches) != 1:
            raise ValueError(f"Pair {number}: expected one directory, found {matches}")
        scans = sorted(path for path in matches[0].glob("*.nii.gz")
                       if "_seg" not in path.name.lower())
        if len(scans) != 2:
            raise ValueError(f"Pair {number}: expected exactly two scans, found {scans}")
        annotations = {doctor: tools.resolve_annotation_path(
            str(root / "Annotations"), doctor, number) for doctor in PHYSICIANS}
        cases.append(dict(pair_id=f"pair{number}", prior=str(scans[0]),
                          current=str(scans[1]), annotations=annotations))
    return cases


def prepare_image(array, size, rotation_k):
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite 2D clinical image")
    image = _norm01(_to_chw(array, size))
    return torch.rot90(image, rotation_k, (-2, -1)).contiguous()


def restore_prediction(prediction, shape, rotation_k):
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    prediction = prediction.reshape(1, 1, *prediction.shape[-2:])
    prediction = torch.rot90(prediction, -rotation_k, (-2, -1))
    return F.interpolate(prediction, size=shape, mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def score_masks(predicted, reference):
    result = {}
    intersection = 0
    total = 0
    for direction, prediction, target in zip(("positive", "negative"), predicted, reference):
        overlap = int(np.logical_and(prediction, target).sum())
        target_count = int(target.sum())
        count = int(prediction.sum()) + target_count
        intersection += overlap
        total += count
        result[f"dice_{direction}"] = 2 * overlap / count if target_count else None
        result[f"recall_{direction}"] = overlap / target_count if target_count else None
        result[f"fp_area_if_no_{direction}"] = float(prediction.mean()) if not target_count else None
    active = bool(np.any(reference))
    result["signed_dice"] = 2 * intersection / total if active else None
    result["reference_has_change"] = int(active)
    result["predicted_area"] = float(np.logical_or(*predicted).mean())
    result["fp_area_if_no_change"] = result["predicted_area"] if not active else None
    return result


def load_case(case, tools):
    import nibabel as nib

    prior_nifti, current_nifti = nib.load(case["prior"]), nib.load(case["current"])
    prior = np.asarray(prior_nifti.dataobj, dtype=np.float32).squeeze()
    current = np.asarray(current_nifti.dataobj, dtype=np.float32).squeeze()
    if prior.ndim != 2 or current.ndim != 2:
        raise ValueError(f"{case['pair_id']}: expected two 2D scans")
    if not np.isfinite(prior).all() or not np.isfinite(current).all():
        raise ValueError(f"{case['pair_id']}: nonfinite input")
    annotations = {}
    annotation_warnings = {}
    for doctor, path in case["annotations"].items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            positive, negative = tools.load_labels_map(path, current.shape)
        if caught:
            annotation_warnings[doctor] = sorted({str(warning.message) for warning in caught})
        annotations[doctor] = (positive > 0, negative < 0)
    case["annotation_warnings"] = annotation_warnings
    return prior, current, current_nifti, annotations


def plot_case(prior, current, model_input, scores, predicted, annotations, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.ravel()
    signed = predicted[0].astype(float) - predicted[1].astype(float)
    for axis, image, title in zip(axes[:2], (prior, current), ("Prior", "Current")):
        axis.imshow(image.T, cmap="gray")
        axis.set_title(title)
    axes[2].imshow(model_input, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Current: model input coordinates")
    axes[3].imshow(scores.T, cmap="bwr", vmin=-1, vmax=1)
    axes[3].set_title("Raw signed score [-1,1]")
    axes[4].imshow(current.T, cmap="gray")
    axes[4].imshow(np.ma.masked_where(signed.T == 0, signed.T), cmap="bwr",
                   vmin=-1, vmax=1, alpha=0.55)
    axes[4].set_title("Thresholded prediction / current")
    for axis, (doctor, masks) in zip(axes[5:], annotations.items()):
        axis.imshow(current.T, cmap="gray")
        positive, negative = masks
        mask = positive.astype(float) - negative.astype(float)
        axis.imshow(np.ma.masked_where(mask.T == 0, mask.T), cmap="bwr",
                    vmin=-1, vmax=1, alpha=0.55)
        if np.logical_and(positive, negative).any():
            axis.contour(np.logical_and(positive, negative).T, levels=[0.5], colors="black")
        axis.set_title(doctor)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{out_path.parent.name}: red positive; blue negative; black = both human signs")
    figure.tight_layout()
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def export_case(case, scores, prior, current, model_input, current_nifti, annotations,
                out_dir, threshold):
    import nibabel as nib

    if scores.shape != current.shape or not np.isfinite(scores).all():
        raise ValueError(f"{case['pair_id']}: invalid restored prediction")
    destination = out_dir / case["pair_id"]
    destination.mkdir()
    predicted = (scores > threshold, scores < -threshold)
    header = current_nifti.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(scores.astype(np.float32), current_nifti.affine, header),
             destination / "output.nii.gz")
    signed = predicted[0].astype(np.int8) - predicted[1].astype(np.int8)
    nib.save(nib.Nifti1Image(signed, current_nifti.affine), destination / "mask.nii.gz")
    rows = [dict(pair_id=case["pair_id"], observer=doctor, **score_masks(predicted, masks))
            for doctor, masks in annotations.items()]
    votes = np.asarray(list(annotations.values())).sum(axis=0)
    consensus = (votes[0] >= 3, votes[1] >= 3)
    rows.append(dict(pair_id=case["pair_id"], observer="consensus_3_of_5",
                     **score_masks(predicted, consensus)))
    plot_case(prior, current, model_input, scores, predicted, annotations,
              destination / "review.png")
    (destination / "source.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    return rows


def summarize(rows):
    summary = {}
    for observer in (*PHYSICIANS, "consensus_3_of_5"):
        subset = [row for row in rows if row["observer"] == observer]
        metrics = {}
        for key in ("signed_dice", "dice_positive", "dice_negative", "recall_positive",
                    "recall_negative", "fp_area_if_no_change", "fp_area_if_no_positive",
                    "fp_area_if_no_negative"):
            values = [row[key] for row in subset if row[key] is not None]
            metrics[key] = dict(n=len(values), mean=float(np.mean(values)) if values else None)
        summary[observer] = metrics
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical_root", type=Path, required=True,
                        help="Directory containing Pairs1..Pairs8 and Annotations")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--expected_sha256")
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def run(args):
    if args.num_pairs < 1 or args.num_pairs > 100:
        raise ValueError("num_pairs must be between 1 and 100")
    cases = discover_cases(args.clinical_root, args.num_pairs)
    tools = clinical_tools()
    if args.preflight:
        for case in cases:
            load_case(case, tools)
        print(f"[clinical] preflight passed: {len(cases)} pairs, {5 * len(cases)} annotations")
        flagged = [case["pair_id"] for case in cases if case["annotation_warnings"]]
        print(f"[clinical] annotation raster warnings: {flagged}")
        return
    if args.ckpt is None or args.out_dir is None:
        raise ValueError("Inference requires --ckpt and --out_dir")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Use a new empty output directory; do not mix clinical runs")
    payload = args.ckpt.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("Checkpoint SHA256 does not match the requested clinical experiment")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    del payload
    spec = state["head_spec"]
    config = state.get("run_config", {})
    checkpoint_epoch = state.get("epoch")
    rotation = int(config.get("rotation_k", 0))
    threshold = float(config.get("eval_threshold", 0.1) if args.threshold is None else args.threshold)
    if rotation not in (0, 1, 2, 3) or not 0 < threshold < 1:
        raise ValueError("Invalid checkpoint rotation or prediction threshold")
    from models import FrozenCXRBackbone, build_change_head
    import constants as settings

    head = build_change_head(spec, args.device).eval()
    head.load_state_dict(state["head"])
    del state
    width = spec["kwargs"]["backbone_dim"]
    layers = spec["kwargs"]["patch_dim"] // width
    backbone = FrozenCXRBackbone(
        name=settings.BACKBONE, model_id=settings.RAD_DINO_MODEL,
        backbone_img_size=settings.BACKBONE_IMG_SIZE,
        num_patch_tokens=settings.NUM_PATCH_TOKENS, last_n_layers=layers, freeze=True,
    ).to(args.device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint_sha256=digest, head_spec=spec, run_config=config,
                    rotation_k=rotation, prediction_threshold=threshold,
                    clinical_root=str(args.clinical_root.resolve()), num_pairs=len(cases),
                    coordinate_space="original NIfTI array; output shape/affine match current",
                    preprocessing="training resize/minmax/rotation, no transpose/crop/sharpening",
                    postprocessing="none; mask thresholded AFTER restoration to source resolution",
                    scan_order="lexicographic, matching existing clinical evaluator",
                    evaluation="exploratory transfer; clinician ellipses are not dense pathology GT")
    metadata["checkpoint_epoch"] = checkpoint_epoch
    metadata["annotation_sha256"] = {
        f"{case['pair_id']}/{doctor}": hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for case in cases for doctor, path in case["annotations"].items()}
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    size = int(spec["kwargs"].get("out_size", settings.IMG_SIZE))
    for case in cases:
        prior, current, current_nifti, annotations = load_case(case, tools)
        inputs = [prepare_image(image, size, rotation) for image in (prior, current)]
        features = [backbone(image[None].to(args.device)) for image in inputs]
        prediction, _ = head(features[0]["patch_tokens"], features[1]["patch_tokens"],
                             features[0]["cls_token"], features[1]["cls_token"])
        scores = restore_prediction(prediction[0, 0], current.shape, rotation)
        rows.extend(export_case(case, scores, prior, current, inputs[1][0].numpy(),
                                current_nifti, annotations, args.out_dir, threshold))
        print(f"[clinical] {case['pair_id']} complete", flush=True)
    with (args.out_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(pairs_completed=len(cases), threshold=threshold, observers=summarize(rows))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    links = ["# Clinical Comparison-Head Review", "", "Exploratory clinical transfer, not synthetic Dice.",
             "Raw scores are retained; no small-object cleanup or clinical threshold tuning.", ""]
    links.extend(f"- [{case['pair_id']}]({case['pair_id']}/review.png)" for case in cases)
    (args.out_dir / "index.md").write_text("\n".join(links) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(parse_args())