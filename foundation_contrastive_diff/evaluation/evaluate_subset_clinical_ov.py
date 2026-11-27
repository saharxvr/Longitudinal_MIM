"""ICU100 OV evaluation of validation-selected subset-epoch architecture heads."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects, label
import torch

import constants as C
from evaluation.compare_clinical_models import legacy_summary, read_json, run_legacy
from evaluation.evaluate_clinical import (
    PHYSICIANS, clinical_tools, discover_cases, load_case, prepare_image, restore_prediction,
)
from evaluation.sweep_clinical_postprocessing import STRUCTURE, masks_for, match_counts
from models import FrozenCXRBackbone, build_change_head, checkpoint_feature_options, signed_change_map
from training import train_subset_architecture_ablation as subset
from training import train_shared_architecture_ablation as shared


ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (.1, .25, .5, 1.)
THRESHOLDS = (.1, .25, .5, .75, .9)
CC_CUTOFF = 250
EXPORT_FILES = (
    "summary.json", "protocol.json", "checkpoints.csv", "per_clinician.csv",
    "threshold_summary.csv", "threshold_sweep.png", "data_fraction_f1.png", "index.md",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoints(args, context, fractions=FRACTIONS, seed=42):
    if (not fractions or len(set(fractions)) != len(fractions)
            or any(fraction not in FRACTIONS for fraction in fractions) or seed not in args.seeds):
        raise ValueError("Use available 10/25/50/100% groups and a configured seed")
    models = {}
    for fraction in fractions:
        group_args, group_context, budget = subset.group_setup(args, context, fraction, seed)
        identity = shared.group_identity(group_args, group_context, fraction, seed)[0]
        folder = Path(args.run_root) / shared.group_name(fraction, seed)
        shared.verify_group(folder, identity)
        snapshot = torch.load(folder / "shared_last.pt", map_location="cpu", weights_only=False)
        if snapshot["identity"] != identity or snapshot["epoch"] != args.epochs:
            raise ValueError("Completed group snapshot differs from the selected experiment")
        for arm in shared.base.ARMS:
            branch = snapshot["branches"][arm]
            path = folder / arm / "best.pt"
            payload = path.read_bytes()
            state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            best = max(branch["history"], key=lambda row: row["val"]["target_signed_dice"])
            if (state["epoch"] != best["slot"] or state["best_score"] != best["val"]["target_signed_dice"]
                    or state["head_spec"] != branch["spec"]
                    or state["optimizer_updates"] != best["slot"] * budget["updates_per_epoch"]
                    or set(state["head"]) != set(branch["best_head"])
                    or any(not torch.equal(value, branch["best_head"][name])
                           for name, value in state["head"].items())):
                raise ValueError(f"Best checkpoint does not reproduce validation selection: {arm}/{fraction}")
            config, spec = state["run_config"], state["head_spec"]
            if (config["arm"] != arm or config["seed"] != seed or config["subset_budget"] != budget
                    or config["epoch_budget"] != subset.BUDGET_POLICY
                    or config["group_order"] != subset.GROUP_ORDER
                    or config["rotation_k"] != args.rotation_k
                    or config["flip_horizontal"] != args.flip_horizontal
                    or config["loss_mode"] != "ce_dice" or spec["kwargs"]["output_mode"] != "three_class"
                    or checkpoint_feature_options(spec, config) != dict(last_n_layers=1, selected_layer=9)):
                raise ValueError(f"Checkpoint extraction/configuration differs: {arm}/{fraction}")
            expected_type = "dpt_cross_attention" if arm == "B0" else "dpt_global_attention"
            if spec["type"] != expected_type or spec["kwargs"].get("debias_rank", 0) != (
                    args.debias_rank if arm == "B2" else 0):
                raise ValueError("Checkpoint architecture or projection rank differs")
            head = build_change_head(spec, "cpu").eval()
            head.load_state_dict(state["head"])
            name = f"{arm}_fraction_{fraction:g}_seed_{seed}"
            models[name] = dict(
                head=head, spec=spec, path=path,
                metadata=dict(model=name, arm=arm, patient_fraction=fraction, seed=seed,
                              epoch=state["epoch"], selected_pairs=budget["selected_pairs"],
                              optimizer_updates=state["optimizer_updates"],
                              best_val_signed_dice=state["best_score"],
                              checkpoint_sha256=hashlib.sha256(payload).hexdigest()))
    specs = [model["spec"]["kwargs"] for model in models.values()]
    for key in ("backbone_dim", "patch_dim", "cls_dim", "grid", "out_size"):
        if len({spec[key] for spec in specs}) != 1:
            raise ValueError(f"Selected heads cannot share the same extraction: {key}")
    return models


def input_hashes(cases):
    return {case["pair_id"]: dict(
        prior=file_hash(case["prior"]), current=file_hash(case["current"]),
        annotations={doctor: file_hash(case["annotations"][doctor]) for doctor in PHYSICIANS})
        for case in cases}


def inference_identity(models, cases, context, tools):
    if not cases or len({case["pair_id"] for case in cases}) != len(cases):
        raise ValueError("Require nonempty clinical cases with unique pair IDs")
    return dict(
        checkpoints={name: model["metadata"] for name, model in models.items()},
        inputs=input_hashes(cases), training_provenance=context["provenance"],
        evaluation_code_sha256={
            "evaluate_subset_clinical_ov": file_hash(__file__),
            "clinical_preprocessing": file_hash(prepare_image.__code__.co_filename),
            "postprocessing": file_hash(masks_for.__code__.co_filename),
            "legacy_ov": file_hash(tools.__file__)},
        coordinate_space="original current-image array; unchanged affine; inverse training orientation",
        scan_order="lexicographic, matching the existing clinical evaluator",
        image_size=next(iter(models.values()))["spec"]["kwargs"]["out_size"])


def verify_predictions(directory, identity, cases):
    record = read_json(directory / "complete.json")
    if record["identity"] != identity:
        raise ValueError("Saved clinical inference identity/code/inputs/checkpoints differ")
    expected = {f"{name}/{case['pair_id']}/output.nii.gz"
                for name in identity["checkpoints"] for case in cases}
    if set(record["prediction_sha256"]) != expected:
        raise ValueError("Saved inference does not cover all selected models and pairs")
    for path, digest in record["prediction_sha256"].items():
        if file_hash(directory / path) != digest:
            raise ValueError(f"Saved clinical prediction changed: {path}")
    return record


@torch.inference_mode()
def infer(models, cases, context, tools, directory, identity, device):
    if directory.exists():
        raise ValueError("Use a new inference directory, or explicitly reuse complete saved inference")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no clinical inference started")
    features = context["provenance"]["features"]
    backbone = FrozenCXRBackbone(
        name=C.BACKBONE, model_id=features["model_id"],
        backbone_img_size=features["backbone_image_size"], num_patch_tokens=C.NUM_PATCH_TOKENS,
        last_n_layers=1, selected_layer=9, freeze=True).to(device).eval()
    if backbone.training or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Clinical inference requires a frozen eval-mode backbone")
    if getattr(backbone.backbone.config, "_commit_hash", None) != features["backbone_revision"]:
        raise ValueError("Clinical backbone revision differs from training")
    for model in models.values():
        model["head"].to(device)
    directory.mkdir(parents=True, exist_ok=False)
    prediction_hashes = {}
    for number, case in enumerate(cases, 1):
        prior, current, current_nifti, _ = load_case(case, tools)
        inputs = [prepare_image(image, identity["image_size"], features["rotation_k"],
                                features["flip_horizontal"]) for image in (prior, current)]
        extracted = [backbone(image[None].to(device)) for image in inputs]
        tokens = (extracted[0]["patch_tokens"], extracted[1]["patch_tokens"],
                  extracted[0]["cls_token"], extracted[1]["cls_token"])
        for name, model in models.items():
            output, _ = model["head"](*tokens)
            scores = restore_prediction(
                signed_change_map(output)[0, 0], current.shape,
                features["rotation_k"], features["flip_horizontal"])
            if (scores.shape != current.shape or not np.isfinite(scores).all()
                    or (np.abs(scores) > 1.000001).any()):
                raise ValueError(f"Invalid restored clinical prediction: {name}/{case['pair_id']}")
            destination = directory / name / case["pair_id"]
            destination.mkdir(parents=True)
            path = destination / "output.nii.gz"
            header = current_nifti.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(scores.astype(np.float32), current_nifti.affine, header), path)
            prediction_hashes[path.relative_to(directory).as_posix()] = file_hash(path)
        print(f"[ICU inference] {number}/{len(cases)} pairs; {len(models)} heads", flush=True)
    if input_hashes(cases) != identity["inputs"]:
        raise ValueError("Clinical images/annotations changed during inference")
    for model in models.values():
        if file_hash(model["path"]) != model["metadata"]["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed during clinical inference")
    shared.base.write_json(directory / "complete.json", dict(
        identity=identity, prediction_sha256=prediction_hashes))
    for model in models.values():
        model["head"].to("cpu")
    if str(device).startswith("cuda"):
        backbone.to("cpu")
        torch.cuda.empty_cache()


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def clean_ratios(value):
    if isinstance(value, dict):
        return {key: clean_ratios(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_ratios(item) for item in value]
    if isinstance(value, float):
        if np.isinf(value):
            raise ValueError("Infinite OV metric is not an undefined ratio")
        return finite_or_none(value)
    return value


def score_model(name, cases, tools, inference, local_ov, thresholds, clinical_root):
    totals = np.zeros((len(thresholds), len(PHYSICIANS), 2, 2), dtype=np.int64)
    areas = np.zeros(len(thresholds))
    empty_consensus_areas = [[] for _ in thresholds]
    local_ov.mkdir(parents=True, exist_ok=False)
    rows, details = [], {}
    with TemporaryDirectory(prefix="subset_clinical_ov_") as temporary:
        filtered_roots = [Path(temporary) / f"threshold_{threshold:g}" for threshold in thresholds]
        for root in filtered_roots:
            root.mkdir()
        for case in cases:
            image = nib.load(inference / name / case["pair_id"] / "output.nii.gz")
            scores = np.asarray(image.dataobj, dtype=np.float32).squeeze()
            current = nib.load(case["current"])
            if (scores.ndim != 2 or not np.isfinite(scores).all()
                    or (np.abs(scores) > 1.000001).any()
                    or scores.shape != tuple(size for size in current.shape if size != 1)
                    or not np.allclose(image.affine, current.affine, rtol=0, atol=1e-5)):
                raise ValueError("Clinical score/current geometry or range mismatch")
            references, masks = [], []
            for doctor in PHYSICIANS:
                positive, negative = tools.load_labels_map(case["annotations"][doctor], scores.shape)
                human_masks = (positive > 0, negative < 0)
                masks.append(human_masks)
                reference = []
                for mask in human_masks:
                    components, _ = label(mask, STRUCTURE)
                    reference.append((components, find_objects(components)))
                references.append(reference)
            empty_consensus = not any((np.stack([m[sign] for m in masks]).sum(0) >= 3).any()
                                      for sign in (0, 1))
            for index, threshold in enumerate(thresholds):
                positive, negative = masks_for(scores, f"components_{CC_CUTOFF}", threshold)
                area = float((positive | negative).mean())
                areas[index] += area
                if empty_consensus:
                    empty_consensus_areas[index].append(area)
                filtered = positive.astype(np.float32) - negative.astype(np.float32)
                folder = filtered_roots[index] / case["pair_id"]
                folder.mkdir()
                nib.save(nib.Nifti1Image(filtered, image.affine), folder / "output.nii.gz")
                for sign, mask in enumerate((positive, negative)):
                    prediction, count = label(mask, STRUCTURE)
                    for reader, reference in enumerate(references):
                        totals[index, reader, sign] += match_counts(*reference[sign], prediction, count)
        for index, threshold in enumerate(thresholds):
            destination = local_ov / f"threshold_{threshold:g}"
            print(f"[OV] {name}; threshold={threshold:g}; CC>{CC_CUTOFF}", flush=True)
            run_legacy(tools, filtered_roots[index], clinical_root, destination,
                       len(cases), "original", 0.)
            matrices = read_json(destination / "agreement_matrices.json")
            model_index = matrices["observers"].index("Model")
            for reader, doctor in enumerate(PHYSICIANS):
                row = dict(model=name, threshold=threshold, clinician=doctor)
                human_index = matrices["observers"].index(doctor)
                for sign, counts in (("positive", totals[index, reader, 0]),
                                     ("negative", totals[index, reader, 1]),
                                     ("all", totals[index, reader].sum(0))):
                    matched, unmatched = counts
                    denominator = 2 * matched + unmatched
                    expected = 2 * matched / denominator if denominator else np.nan
                    suffix = {"positive": "pos", "negative": "neg", "all": "all"}[sign]
                    actual = matrices[f"per_label_{suffix}"][model_index][human_index]
                    if not np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True):
                        raise ValueError(f"Actual legacy OV disagrees with filtered-mask counts: {name}/{threshold}")
                    row[f"F1_{sign}"] = finite_or_none(actual)
                rows.append(row)
            empty = empty_consensus_areas[index]
            details[f"{threshold:g}"] = dict(
                ov_summary=clean_ratios(legacy_summary(destination)),
                agreement_matrices=clean_ratios(matrices),
                predicted_area_mean=float(areas[index] / len(cases)),
                empty_consensus_pairs=len(empty),
                empty_consensus_area_mean=float(np.mean(empty)) if empty else None,
                empty_consensus_area_max=max(empty) if empty else None)
    return rows, details


def summarize(models, rows, details):
    summaries = []
    for name, model in models.items():
        for threshold in THRESHOLDS:
            selected = [row for row in rows if row["model"] == name and row["threshold"] == threshold]
            if len(selected) != len(PHYSICIANS):
                raise ValueError("Every operating point requires all five clinician comparisons")
            row = dict(**model["metadata"], threshold=threshold, cc_cutoff=CC_CUTOFF)
            for sign in ("positive", "negative", "all"):
                values = [item[f"F1_{sign}"] for item in selected if item[f"F1_{sign}"] is not None]
                row[f"mean_F1_{sign}"] = statistics.mean(values) if values else None
                row[f"reader_SD_F1_{sign}"] = statistics.stdev(values) if len(values) > 1 else None
            row["predicted_area_mean"] = details[name][f"{threshold:g}"]["predicted_area_mean"]
            summaries.append(row)
    return summaries


def plot_results(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = sorted({row["patient_fraction"] for row in rows})
    figure, axes = plt.subplots(1, len(fractions), figsize=(5 * len(fractions), 4), squeeze=False)
    for axis, fraction in zip(axes[0], fractions):
        for arm in shared.base.ARMS:
            selected = [row for row in rows if row["arm"] == arm and row["patient_fraction"] == fraction]
            axis.plot([row["threshold"] for row in selected],
                      [np.nan if row["mean_F1_all"] is None else row["mean_F1_all"] for row in selected],
                      marker="o", label=arm)
        axis.set(title=f"{fraction:.0%} training patients", xlabel="Signed-score threshold",
                 ylabel="Mean clinician component F1", ylim=(0, 1))
        axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "threshold_sweep.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    for arm in shared.base.ARMS:
        selected = [row for row in rows if row["arm"] == arm and row["threshold"] == .75]
        axis.plot([row["selected_pairs"] for row in selected],
                  [np.nan if row["mean_F1_all"] is None else row["mean_F1_all"] for row in selected],
                  marker="o", label=arm)
    axis.set(xlabel="Actual training pairs", ylabel="Mean clinician component F1",
             title="Fixed threshold 0.75; sign-specific CC >250", ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination / "data_fraction_f1.png", dpi=120)
    plt.close(figure)


def evaluate(models, cases, tools, context, inference_dir, destination, clinical_root,
             device="cuda", reuse=False):
    if destination.exists():
        raise ValueError("Use a new evaluation output directory")
    identity = inference_identity(models, cases, context, tools)
    if reuse:
        record = verify_predictions(inference_dir, identity, cases)
    else:
        infer(models, cases, context, tools, inference_dir, identity, device)
        record = verify_predictions(inference_dir, identity, cases)
    tools.struct = STRUCTURE
    rows, details = [], {}
    local_ov = inference_dir / f"ov_{time.time_ns()}"
    for name in models:
        values, details[name] = score_model(
            name, cases, tools, inference_dir, local_ov / name, THRESHOLDS, clinical_root)
        rows.extend(values)
    if input_hashes(cases) != identity["inputs"]:
        raise ValueError("Clinical inputs/annotations changed during OV evaluation")
    verify_predictions(inference_dir, identity, cases)
    summaries = summarize(models, rows, details)
    destination.mkdir(parents=True, exist_ok=False)
    public_protocol = dict(
        cohort="Repeatedly evaluated ICU100; exploratory, not untouched independent test",
        pairs=len(cases), readers=list(PHYSICIANS), seed=next(iter(models.values()))["metadata"]["seed"],
        thresholds=list(THRESHOLDS), headline_threshold=.75, cc_cutoff=CC_CUTOFF,
        cc_area_space="512x512 equivalent; native cutoff = 250*image_area/512^2",
        comparisons="strict |score| > threshold and component area > cutoff",
        separate_signs=True, connectivity=8, smoothing=False, coordinate_space=identity["coordinate_space"],
        selection="Synthetic validation best checkpoint; never selected on ICU performance",
        matcher="Actual legacy OV and independently checked reference-first greedy one-to-one counts",
        detection_f1="Per-reader pooled component F1; not pixel Dice or legacy boosted per-pair agreement",
        reader_SD="Variability across five clinician scores, not across seeds or a confidence interval",
        postprocessing_warning="Not exact historical Itamar cleanup",
        input_hash=shared.base.json_hash(identity["inputs"]),
        prediction_hash=shared.base.json_hash(record["prediction_sha256"]),
        training_manifest_sha256=context["provenance"]["features"]["manifest_sha256"],
        evaluation_code_sha256=identity["evaluation_code_sha256"],
        undefined_ratios="null; not replaced by zero",
        publication="Aggregates and numeric plots only; scans, case IDs, predictions and weights stay on campus")
    shared.base.write_json(destination / "protocol.json", public_protocol)
    shared.base.write_json(destination / "summary.json", dict(
        protocol=public_protocol, headline=[row for row in summaries if row["threshold"] == .75],
        sweep=summaries, ov=details))
    shared.base.write_csv(destination / "checkpoints.csv", [model["metadata"] for model in models.values()])
    shared.base.write_csv(destination / "per_clinician.csv", rows)
    shared.base.write_csv(destination / "threshold_summary.csv", summaries)
    plot_results(summaries, destination)
    lines = [
        "# Subset-epoch models: ICU100 OV", "",
        "Exploratory reused-cohort evaluation. Checkpoints selected on synthetic validation only.",
        "Headline: threshold 0.75 and separate-sign 8-connected CC >250 at 512x512-equivalent area.",
        "The sweep is exploratory; no operating point is selected as a confirmatory result.",
        "Detection F1 is not pixel Dice or historical boosted per-pair agreement.", "",
        "| Model | Training patients | Best epoch | Mean F1 positive | Mean F1 negative | Mean F1 combined |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["threshold"] == .75:
            values = [f"{row[f'mean_F1_{sign}']:.4f}" if row[f"mean_F1_{sign}"] is not None else "NA"
                      for sign in ("positive", "negative", "all")]
            lines.append(f"| {row['model']} | {row['patient_fraction']:.0%} | {row['epoch']} | "
                         + " | ".join(values) + " |")
    lines.extend(["", "[Exact per-reader F1](per_clinician.csv) | [Sweep table](threshold_summary.csv)",
                  "", "![Threshold sweep](threshold_sweep.png)",
                  "", "![Fixed-operating-point data comparison](data_fraction_f1.png)", ""])
    (destination / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


def publish(destination):
    destination = destination.resolve()
    destination.relative_to((ROOT / "results").resolve())
    if {path.name for path in destination.iterdir()} != set(EXPORT_FILES):
        raise ValueError("Only the explicit aggregate report/plot allowlist may be published")
    files = [destination / name for name in EXPORT_FILES]
    if any(path.is_symlink() or not path.is_file() for path in files) or sum(
            path.stat().st_size for path in files) >= 100 * 1024**2:
        raise ValueError("OV publication requires regular aggregate artifacts below 100 MiB")
    repository = ROOT.parent.resolve()
    if subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repository, text=True).strip() != "main":
        raise ValueError("Campus publication expects main")
    paths = [path.relative_to(repository).as_posix() for path in files]
    subprocess.run(["git", "add", "--", *paths], cwd=repository, check=True)
    subprocess.run(["git", "commit", "--only", "-m", f"Publish {destination.name}", "-m",
                    "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
                    "--", *paths], cwd=repository, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repository, check=True)
    print(f"[ICU publish] pushed {len(paths)} aggregate files; no training-source pull", flush=True)


def check_gpu_job(device):
    if str(device).startswith("cuda") and os.name == "posix":
        jobs = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-af", r"training[.](train_|run_)"],
            capture_output=True, text=True)
        if jobs.returncode == 0:
            raise RuntimeError("Training job detected; use --device cpu or stop the owned job explicitly first")
        if jobs.returncode != 1:
            raise RuntimeError(f"Could not check GPU job ownership: {jobs.stderr}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--clinical-root", required=True, type=Path)
    parser.add_argument("--inference-dir", required=True, type=Path, help="Campus-only prediction storage")
    parser.add_argument("--out-dir", required=True, type=Path, help="Aggregate reports, under study/results to publish")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS), choices=FRACTIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-inference", action="store_true")
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args(argv)
    torch.set_num_threads(1)
    if options.inference_dir.resolve().is_relative_to(ROOT.parent.resolve()):
        raise ValueError("Keep raw clinical prediction storage outside the Git repository")
    if options.publish:
        options.out_dir.resolve().relative_to((ROOT / "results").resolve())
    if not options.reuse_inference:
        check_gpu_job(options.device)
    args = subset.configure(
        shared.base.parse_args(["--config", str(options.config.resolve())]), options.run_root)
    context, _ = subset.report_context(args)
    models = load_checkpoints(args, context, options.fractions, options.seed)
    tools = clinical_tools()
    cases = discover_cases(options.clinical_root, 100)
    with shared.base.run_lock(options.inference_dir.with_suffix(".evaluation.lock")):
        destination = evaluate(
            models, cases, tools, context, options.inference_dir, options.out_dir,
            options.clinical_root, options.device, options.reuse_inference)
    if options.publish:
        publish(destination)


if __name__ == "__main__":
    main()
