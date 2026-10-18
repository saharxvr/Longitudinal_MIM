"""Export exact cache/clinical inputs and measurement references without inference."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import constants as C
from datasets.pair_dataset import CachedPairDataset, LongitudinalPairDataset, _norm01, _to_chw, orient_tensor
from evaluation import evaluate_clinical
from losses import change_map_loss
from training import cache_features


ROOT = Path(__file__).resolve().parents[1]


class CapturedPixels(Exception):
    def __init__(self, pixels):
        super().__init__("Stopped before neural-network inference")
        self.pixels = pixels.detach().cpu()


class PixelSink(torch.nn.Module):
    def forward(self, pixel_values, **kwargs):
        raise CapturedPixels(pixel_values)


class CacheInputCapture:
    def __init__(self, last_n_layers):
        with patch("transformers.AutoModel.from_pretrained", return_value=PixelSink()):
            self.backbone = cache_features._build_backbone("cpu", last_n_layers)
        if list(self.backbone.parameters()):
            raise AssertionError("Input capture must not load model weights")

    def __call__(self, images):
        with torch.no_grad():
            try:
                self.backbone(images)
            except CapturedPixels as captured:
                return captured.pixels
        raise AssertionError("Encoder input capture was not reached")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def values(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def signed_target(gt, threshold):
    return torch.where(gt.abs() > threshold, gt.sign(), torch.zeros_like(gt))


def encoder_preview(pixels, capture):
    restored = pixels * capture.backbone._std[0] + capture.backbone._mean[0]
    return restored.permute(1, 2, 0).clamp(0, 1).numpy()


def save_evidence(folder, tensors, panels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    folder.mkdir(parents=True)
    arrays = {key: values(value) for key, value in tensors.items()}
    np.savez_compressed(folder / "exact_tensors.npz", **arrays)
    for key in ("prior_cache_input", "current_cache_input"):
        image = np.rint(np.clip(arrays[key][0], 0, 1) * 65535).astype(np.uint16)
        Image.fromarray(image).save(folder / f"{key}.png")
    for key in ("prior_encoder_rgb_preview", "current_encoder_rgb_preview"):
        Image.fromarray(np.rint(arrays[key] * 255).astype(np.uint8)).save(folder / f"{key}.png")
    figure, axes = plt.subplots(math.ceil(len(panels) / 4), 4,
                                figsize=(16, 4 * math.ceil(len(panels) / 4)), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, (image, label, signed, background) in zip(axes.flat, panels):
        image = values(image).squeeze()
        if background is not None:
            axis.imshow(values(background).squeeze(), cmap="gray", vmin=0, vmax=1)
            image = np.ma.masked_where(image == 0, image)
        axis.imshow(image, cmap="bwr" if signed else "gray", vmin=-1 if signed else 0,
                    vmax=1, alpha=.6 if background is not None else 1)
        axis.set_title(label, fontsize=10)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .96))
    figure.savefig(folder / "review.png", dpi=120)
    plt.close(figure)
    with np.load(folder / "exact_tensors.npz") as saved:
        for key, array in arrays.items():
            np.testing.assert_array_equal(saved[key], array)


def choose_drr_indices(dataset):
    selected, seen_cases, seen_anomalies = [], set(), set()
    for index, record in enumerate(dataset.records):
        anomaly = record.get("effective_anomaly_type", record.get("anomaly_type", "none"))
        case = record["pair_id"].split("/")[0]
        if anomaly == "none" or not record.get("realized_change", True):
            continue
        if case in seen_cases or anomaly in seen_anomalies:
            continue
        selected.append(index)
        seen_cases.add(case)
        seen_anomalies.add(anomaly)
        if len(selected) == 3:
            return selected
    raise ValueError("Need three realized DRR training examples from distinct cases/anomaly types")


def run(args):
    config = yaml.safe_load(args.config.read_text())
    dataset_root = args.dataset_root or Path(config["dataset_root"])
    cache_dir = args.cache_dir or Path(config["cache_dir"])
    rotation = int(config.get("rotation_k", 0))
    flip = bool(config.get("flip_horizontal", False))
    target_threshold = float(config["target_threshold"])
    gt_threshold = float(config["eval_gt_threshold"])
    if config["target_mode"] != "signed_mask":
        raise ValueError("This evidence export requires the selected signed-mask training configuration")
    manifest = dataset_root / "manifest_split.jsonl"
    expected = dict(backbone=C.RAD_DINO_MODEL, last_n_layers=int(config["feat_last_k"]),
                    rotation_k=rotation, image_size=C.IMG_SIZE, flip_horizontal=flip,
                    manifest_sha256=sha256(manifest))
    cache_config_path = cache_dir / "cache_config.json"
    cache_config = json.loads(cache_config_path.read_text())
    if cache_config != expected:
        raise ValueError("Actual cache metadata does not match this configuration and manifest")
    run_config_path = Path(config["save_folder"]) / "run_config.json"
    training_config_hash = None
    if run_config_path.is_file():
        training_config = json.loads(run_config_path.read_text())
        for key in ("rotation_k", "flip_horizontal", "feat_last_k", "target_mode",
                    "target_threshold", "eval_gt_threshold", "eval_threshold"):
            if training_config.get(key, False if key == "flip_horizontal" else None) != config[key]:
                raise ValueError(f"Active training configuration differs: {key}")
        for key, selected in (("dataset_root", dataset_root), ("cache_dir", cache_dir)):
            if Path(training_config[key]).resolve() != selected.resolve():
                raise ValueError(f"Active training path differs: {key}")
        training_config_hash = sha256(run_config_path)
    destination = args.out_dir or ROOT / "results" / f"cache_input_evidence_{datetime.now():%Y%m%d_%H%M%S}"
    if destination.exists():
        raise ValueError("Use a fresh evidence directory; existing exports will not be overwritten")
    if args.publish:
        destination.resolve().relative_to((ROOT / "results").resolve())
    dataset = LongitudinalPairDataset(str(dataset_root), split="train", img_size=C.IMG_SIZE,
                                      rotation_k=rotation, flip_horizontal=flip)
    indices = choose_drr_indices(dataset)
    selected_records = [dataset.records[index] for index in indices]
    with patch("datasets.pair_dataset._read_manifest", return_value=selected_records):
        cached = CachedPairDataset(str(dataset_root), str(cache_dir), split="train",
                                   rotation_k=rotation, flip_horizontal=flip)
    if len(cached) != 3:
        raise ValueError("All three selected DRR records must exist in the actual cache")
    clinical_numbers = args.clinical_pairs
    if len(clinical_numbers) != 3 or len(set(clinical_numbers)) != 3 or not all(
            1 <= number <= 100 for number in clinical_numbers):
        raise ValueError("Select exactly three distinct clinical pair numbers in 1..100")
    cases = evaluate_clinical.discover_cases(args.clinical_root, max(clinical_numbers))
    capture = CacheInputCapture(expected["last_n_layers"])
    destination.mkdir(parents=True)
    examples = []
    raw_loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    cache_loader = DataLoader(cached, batch_size=1, shuffle=False, num_workers=0)
    for number, (batch, cached_batch, record, index) in enumerate(
            zip(raw_loader, cache_loader, selected_records, indices), 1):
        pair_id = batch["pair_id"][0]
        if cached_batch["pair_id"][0] != pair_id:
            raise AssertionError("Fresh and cached pair ordering differs")
        torch.testing.assert_close(cached_batch["gt_diff"], batch["gt_diff"].half().float(), rtol=0, atol=0)
        gt = cached_batch["gt_diff"]
        target = signed_target(gt, target_threshold)
        loss, _ = change_map_loss(target, gt, target_mode=config["target_mode"],
                                  target_threshold=target_threshold, w_l1=1, w_l2=1,
                                  w_dice=0, w_sign=0, tau=config["tau"],
                                  pos_weight=config["pos_weight"], background_weight=config["background_weight"])
        if loss.item() != 0:
            raise AssertionError("Exported loss target does not match the actual loss implementation")
        checkpoint_gt = signed_target(gt, gt_threshold)
        prior, current = batch["img_prior"], batch["img_curr"]
        prior_pixels, current_pixels = capture(prior)[0], capture(current)[0]
        paths = {key: Path(dataset._abspath(record, key, fallback)) for key, fallback in (
            ("prior_path", "prior.nii.gz"), ("current_path", "current.nii.gz"),
            ("diff_map_path", "diff_map.nii.gz"))}
        source_images = {key: nib.load(path) for key, path in paths.items()}
        source_arrays = {key: np.asarray(image.dataobj, dtype=np.float32).squeeze()
                         for key, image in source_images.items()}
        source_current = source_arrays["current_path"]
        for key, image in source_images.items():
            if source_arrays[key].shape != source_current.shape or not np.array_equal(
                    image.affine, source_images["current_path"].affine):
                raise ValueError(f"{pair_id}: source prior/current/GT geometry differs")
        source_prior = _norm01(torch.from_numpy(source_arrays["prior_path"].copy())).numpy().T
        source_current_display = _norm01(torch.from_numpy(source_current.copy())).numpy().T
        source_gt = source_arrays["diff_map_path"].T
        prior_rgb, current_rgb = encoder_preview(prior_pixels, capture), encoder_preview(current_pixels, capture)
        tensors = dict(prior_cache_input=prior[0], current_cache_input=current[0],
                       prior_pixel_values=prior_pixels, current_pixel_values=current_pixels,
                       prior_encoder_rgb_preview=prior_rgb, current_encoder_rgb_preview=current_rgb,
                       fresh_continuous_gt=batch["gt_diff"][0], cached_continuous_gt=gt[0],
                       actual_loss_target=target[0], checkpoint_reference=checkpoint_gt[0])
        panels = [(source_prior, "Prior: source display (.T)", False, None),
                  (source_current_display, "Current: source display (.T)", False, None),
                  (source_gt, "Continuous GT: source display (.T)", True, None),
                  (source_gt, "Source GT / source current", True, source_current_display),
                  (prior[0], "Prior: exact cache input, no display transpose", False, None),
                  (current[0], "Current: exact cache input, no display transpose", False, None),
                  (gt[0], "Actual cached GT: float16 -> float32", True, None),
                  (gt[0], "Cached GT / exact current input", True, current[0]),
                  (prior_rgb, "Prior: encoder pixels (de-normalized view)", False, None),
                  (current_rgb, "Current: encoder pixels (de-normalized view)", False, None),
                  (target[0], f"Actual loss target: abs(GT) > {target_threshold:g}", True, current[0]),
                  (checkpoint_gt[0], f"Checkpoint reference: abs(GT) > {gt_threshold:g}", True, current[0])]
        relative = f"drr_{number}"
        save_evidence(destination / relative, tensors, panels,
                      f"DRR TRAIN {pair_id} | rotation={rotation}, horizontal flip={flip} | NO MODEL INFERENCE")
        cache_path = cache_dir / pair_id / "feat.pt"
        example = dict(kind="drr_train", pair_id=pair_id, manifest_index=index,
                       folder=relative, source_paths={key: str(path) for key, path in paths.items()},
                       source_sha256={key: sha256(path) for key, path in paths.items()},
                       cached_gt_matches_fresh_half=True, cached_record=str(cache_path),
                       cached_record_sha256=sha256(cache_path),
                       loss_target_verified_by_actual_loss=True,
                       loss_target_pixels=int(torch.count_nonzero(target)),
                       checkpoint_reference_pixels=int(torch.count_nonzero(checkpoint_gt)))
        write_json(destination / relative / "source.json", example)
        examples.append(example)
        print(f"[input evidence] DRR {number}/3: {pair_id}; fresh/cached GT and loss target verified", flush=True)
    tools = evaluate_clinical.clinical_tools()
    for number in clinical_numbers:
        case = cases[number - 1]
        prior_array, current_array, _, annotations = evaluate_clinical.load_case(case, tools)
        prior, current = [evaluate_clinical.prepare_image(array, C.IMG_SIZE, rotation, flip)
                          for array in (prior_array, current_array)]
        for array, actual in ((prior_array, prior), (current_array, current)):
            expected_input = orient_tensor(_norm01(_to_chw(array, C.IMG_SIZE)), rotation, flip)
            torch.testing.assert_close(actual, expected_input, rtol=0, atol=0)
        prior_pixels, current_pixels = capture(prior[None])[0], capture(current[None])[0]
        prior_rgb, current_rgb = encoder_preview(prior_pixels, capture), encoder_preview(current_pixels, capture)
        source_prior = _norm01(torch.from_numpy(prior_array.copy())).numpy().T
        source_current = _norm01(torch.from_numpy(current_array.copy())).numpy().T
        votes = np.asarray(list(annotations.values())).sum(axis=0)
        consensus = (votes[0] >= 3).astype(np.float32) - (votes[1] >= 3).astype(np.float32)
        reference_for_display = orient_tensor(_to_chw(consensus, C.IMG_SIZE), rotation, flip)
        tensors = dict(prior_cache_input=prior, current_cache_input=current,
                       prior_pixel_values=prior_pixels, current_pixel_values=current_pixels,
                       prior_encoder_rgb_preview=prior_rgb, current_encoder_rgb_preview=current_rgb,
                       native_consensus_positive=votes[0] >= 3, native_consensus_negative=votes[1] >= 3,
                       model_space_consensus_for_orientation_only=reference_for_display)
        panels = [(source_prior, "Prior: source display (.T)", False, None),
                  (source_current, "Current: source display (.T)", False, None),
                  (prior, "Prior: exact input, no display transpose", False, None),
                  (current, "Current: exact input, no display transpose", False, None),
                  (prior_rgb, "Prior: encoder pixels (de-normalized view)", False, None),
                  (current_rgb, "Current: encoder pixels (de-normalized view)", False, None),
                  (consensus.T, "3/5 consensus: NATIVE measurement space", True, source_current),
                  (reference_for_display, "Consensus: MODEL space, orientation view ONLY", True, current)]
        for doctor, masks in annotations.items():
            positive, negative = masks
            tensors[f"native_{doctor}_positive"] = positive
            tensors[f"native_{doctor}_negative"] = negative
            signed = positive.astype(np.float32) - negative.astype(np.float32)
            panels.append((signed.T, f"{doctor}: NATIVE measurement space", True, source_current))
        relative = f"clinical_{case['pair_id']}"
        save_evidence(destination / relative, tensors, panels,
                      f"CLINICAL TEST {case['pair_id']} | rotation={rotation}, horizontal flip={flip}\n"
                      "Human ellipses are measurement references, NOT dense DRR GT | NO MODEL INFERENCE")
        example = dict(kind="clinical_test", pair_id=case["pair_id"], folder=relative,
                       source_paths={key: case[key] for key in ("prior", "current")},
                       source_sha256={key: sha256(case[key]) for key in ("prior", "current")},
                       annotation_sha256={doctor: sha256(path) for doctor, path in case["annotations"].items()},
                       annotation_warnings=case.get("annotation_warnings", {}),
                       native_measurement_shape=list(current_array.shape),
                       clinical_loader_matches_cache_transform=True,
                       model_space_reference_is_display_only=True)
        write_json(destination / relative / "source.json", example)
        examples.append(example)
        print(f"[input evidence] clinical {case['pair_id']}: exact input and native annotations exported", flush=True)
    code_paths = ("training/cache_features.py", "datasets/pair_dataset.py", "models/backbone.py",
                  "evaluation/evaluate_clinical.py", "training/train_foundation_diff.py",
                  "losses/change_map_losses.py", "evaluation/export_cache_input_evidence.py")
    report = dict(config_path=str(args.config.resolve()), config_sha256=sha256(args.config), config=config,
                  cache_config=cache_config, cache_config_sha256=sha256(cache_config_path),
                  active_run_config_sha256=training_config_hash,
                  encoder_mean=capture.backbone._mean.flatten().tolist(),
                  encoder_std=capture.backbone._std.flatten().tolist(),
                  encoder_input_size=capture.backbone.backbone_img_size,
                  model_weights_loaded=False, inference_performed=False, embeddings_created=False,
                  existing_cache_modified=False, examples=examples,
                  code_sha256={path: sha256(ROOT / path) for path in code_paths},
                  note="PNG encoder views are de-normalized for visibility; exact normalized tensors are in NPZ. "
                       "Clinical annotations are native-space references, not dense ground truth. "
                       "No prediction maps or historical model outputs are used.")
    write_json(destination / "report.json", report)
    lines = ["# Exact Cache Input and GT Evidence", "",
             "Three synthetic training pairs and three clinical test pairs. No model weights or inference.", "",
             "Use review.png to inspect orientation. exact_tensors.npz preserves the actual float tensors.",
             "Encoder PNGs reverse channel normalization for visibility only.",
             "Training target and checkpoint GT use different cutoffs, labelled separately.",
             "Clinical doctors/consensus are native-coordinate measurement references, not dense GT.", ""]
    for example in examples:
        lines += [f"## {example['kind']}: {example['pair_id']}", "",
                  f"![Exact input and reference views]({example['folder']}/review.png)", "",
                  f"[Exact tensors]({example['folder']}/exact_tensors.npz) | "
                  f"[Source provenance]({example['folder']}/source.json)", ""]
    (destination / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[input evidence] complete: {destination}", flush=True)
    if args.publish:
        from training.run_upright_clinical import publish
        publish(destination)
    return destination


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/last1_upright_mask_thr0p0001_fp075.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--clinical-root", type=Path, default=ROOT.parent / "python_files" / "annotation tool")
    parser.add_argument("--clinical-pairs", type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())