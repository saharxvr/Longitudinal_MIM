"""Train the separate CE/classwise-Dice head, evaluate clinical100, and publish artifacts."""

import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from evaluation import compare_clinical_models, evaluate_clinical
from training.run_upright_clinical import publish, sha256, verify_predictions, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/last1_upright_ce_dice_thr0p001.yaml"
PROBABILITY_FILES = tuple(f"probability_{name}.nii.gz" for name in ("unchanged", "positive", "negative"))
CODE_PATHS = ("training/run_ce_dice_clinical.py", "training/train_foundation_diff.py",
              "training/run_last4_dpt.sh", "datasets/pair_dataset.py", "models/__init__.py",
              "models/dpt_head.py", "models/backbone.py", "losses/change_map_losses.py",
              "evaluation/training_diagnostics.py", "evaluation/evaluate_clinical.py",
              "evaluation/compare_clinical_models.py", "training/run_upright_clinical.py", "constants.py")


def verify_probabilities(clinical_run, cases):
    import nibabel as nib

    hashes = {}
    for case in cases:
        folder = clinical_run / case["pair_id"]
        current = nib.load(case["current"])
        images = [nib.load(folder / name) for name in PROBABILITY_FILES]
        for image in images:
            if image.shape != current.shape or not np.allclose(image.affine, current.affine, rtol=0, atol=1e-5):
                raise ValueError(f"Probability geometry mismatch: {case['pair_id']}")
        values = np.stack([np.asarray(image.dataobj) for image in images])
        signed = np.asarray(nib.load(folder / "output.nii.gz").dataobj)
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=0), 1, atol=1e-5)
                or not np.allclose(signed, values[1] - values[2], atol=1e-6)):
            raise ValueError(f"Invalid probabilities or signed score: {case['pair_id']}")
        hashes[case["pair_id"]] = {name: sha256(folder / name) for name in PROBABILITY_FILES}
    return hashes


def run(config_path, clinical_root, publish_results=False):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    expected = dict(loss_mode="ce_dice", target_mode="signed_mask", target_threshold=.001,
                    rotation_k=3, flip_horizontal=True, head="dpt_cross_attention", feat_last_k=1,
                    checkpoint_metric="target_signed_dice", eval_threshold=.5, eval_gt_threshold=.05)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("This launcher requires the separate upright .001 CE/classwise-Dice recipe")
    run_dir = Path(config["save_folder"]).parent
    allowed_runs = {"run_last1_upright_ce_dice_thr0p001": .01,
                    "run_last1_upright_ce_dice_thr0p001_wd0p05": .05}
    if run_dir.name not in allowed_runs or config.get("weight_decay") != allowed_runs[run_dir.name]:
        raise ValueError("Use the dedicated CE run directory matching its weight decay")
    if (Path(config["save_folder"]) != run_dir / "checkpoints"
            or Path(config["plots_folder"]) != run_dir / "plots"):
        raise ValueError("Keep CE checkpoints and plots inside their dedicated run directory")
    code_hashes = {name: sha256(ROOT / name) for name in CODE_PATHS}
    config_hash = sha256(config_path)
    manifest = Path(config["dataset_root"]) / "manifest_split.jsonl"
    manifest_hash = sha256(manifest)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    cases = evaluate_clinical.discover_cases(clinical_root, 100)
    annotation_hashes = {f"{case['pair_id']}/{doctor}": sha256(path)
                         for case in cases for doctor, path in case["annotations"].items()}
    source_hashes = {f"{case['pair_id']}/{name}": sha256(case[name])
                     for case in cases for name in ("prior", "current")}
    environment = dict(os.environ, HEAD_CONFIG=str(config_path), DATA=config["dataset_root"],
                       CACHE_DIR=config["cache_dir"], RUN=str(run_dir))
    subprocess.run(["bash", "training/run_last4_dpt.sh"], cwd=ROOT, env=environment, check=True)
    checkpoint = run_dir / "checkpoints/best.pt"
    last = torch.load(run_dir / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if last["epoch"] != config["epochs"]:
        raise ValueError("Training has not completed the configured epoch budget")
    for state in (last, best):
        if state["head_spec"]["kwargs"].get("output_mode") != "three_class":
            raise ValueError("Expected a three-logit head checkpoint")
        for key, value in config.items():
            if state["run_config"].get(key) != value:
                raise ValueError(f"Checkpoint configuration mismatch: {key}")
    if (sha256(config_path) != config_hash or sha256(manifest) != manifest_hash
            or any(sha256(ROOT / name) != digest for name, digest in code_hashes.items())):
        raise ValueError("Code/config/manifest changed during training; do not silently mix experiment versions")
    for case in cases:
        for name in ("prior", "current"):
            if sha256(case[name]) != source_hashes[f"{case['pair_id']}/{name}"]:
                raise ValueError("Clinical source changed during training")
        for doctor, path in case["annotations"].items():
            if sha256(path) != annotation_hashes[f"{case['pair_id']}/{doctor}"]:
                raise ValueError("Clinical annotation changed during training")
    best_epoch = best["epoch"]
    del last, best
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clinical_run = run_dir / f"clinical100_{timestamp}"
    checkpoint_hash = sha256(checkpoint)
    evaluate_clinical.run(SimpleNamespace(
        clinical_root=clinical_root, num_pairs=100, preflight=False, ckpt=checkpoint,
        expected_sha256=checkpoint_hash, out_dir=clinical_run, device="cuda", threshold=.5))
    prediction_hashes = verify_predictions(clinical_run, cases)
    probability_hashes = verify_probabilities(clinical_run, cases)
    f1 = clinical_run / "detection_f1"
    compare_clinical_models.run(SimpleNamespace(
        clinical_run=clinical_run, clinical_root=clinical_root, num_pairs=100,
        foundation_only=True, itamar_preds=None, out_dir=f1))
    result_label = run_dir.name.removeprefix("run_last1_")
    destination = ROOT / "results" / f"clinical100_{result_label}_{timestamp}"
    destination.mkdir()
    for name in ("metadata.json", "report.json", "per_pair.csv"):
        shutil.copy2(clinical_run / name, destination / name)
    for case in cases:
        folder = destination / case["pair_id"]
        folder.mkdir()
        for name in ("output.nii.gz", "mask.nii.gz", "source.json", *PROBABILITY_FILES):
            shutil.copy2(clinical_run / case["pair_id"] / name, folder / name)
    (destination / "detection_f1").mkdir()
    for name in ("metadata.json", "report.json", "f1_per_clinician.csv"):
        shutil.copy2(f1 / name, destination / "detection_f1" / name)
    for name in ("metrics.csv", "run_config.json"):
        shutil.copy2(run_dir / "checkpoints" / name, destination / name)
    for epoch in sorted({config["epochs"], best_epoch}):
        for split in ("train", "val"):
            name = f"epoch_{epoch:04d}_{split}.json"
            shutil.copy2(run_dir / "checkpoints/diagnostics" / name, destination / name)
    shutil.copy2(config_path, destination / "training_config.yaml")
    write_json(destination / "verification.json", dict(
        code_commit=code_commit, code_sha256=code_hashes, config_sha256=config_hash,
        manifest_sha256=manifest_hash, checkpoint_sha256=checkpoint_hash, checkpoint_epoch=best_epoch,
        prediction_sha256=prediction_hashes, probability_sha256=probability_hashes,
        clinical_source_sha256=source_hashes, annotation_sha256=annotation_hashes,
        pairs_verified=len(prediction_hashes), class_order=["unchanged", "positive", "negative"],
        signed_score="p_positive-p_negative", encoder_finetuned=False,
        scope="clinical100 exploratory transfer; no held-out synthetic test or threshold tuning"))
    if publish_results:
        publish(destination)
    print(f"[complete] CE/classwise-Dice results: {destination}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clinical-root", type=Path, default=ROOT.parent / "python_files/annotation tool")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    run_dir = Path(config["save_folder"]).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (run_dir / ".ce_pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.config, args.clinical_root, args.publish)


if __name__ == "__main__":
    main()