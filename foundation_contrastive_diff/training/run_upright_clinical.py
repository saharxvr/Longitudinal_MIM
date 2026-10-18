"""Audit corrected DRR/GT orientation, then train, predict clinical100 and publish."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from datasets.pair_dataset import _norm01, _to_chw, orient_tensor
from evaluation import audit_gt, compare_clinical_models, evaluate_clinical


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/last1_upright_mask_thr0p0001_fp075.yaml"
REFERENCE = ROOT / "results/clinical100_cutoff0001_epoch49_predictions/metadata.json"
CODE_PATHS = ("datasets/pair_dataset.py", "training/cache_features.py",
              "training/train_foundation_diff.py", "evaluation/audit_gt.py",
              "evaluation/evaluate_clinical.py", "evaluation/compare_clinical_models.py",
              "models/backbone.py", "constants.py", "training/run_rotated_mask.sh",
              "training/run_last4_dpt.sh", "training/run_upright_clinical.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def approval_hash(provenance):
    return hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    previous = yaml.safe_load((ROOT / "configs/last1_rot90cw_mask_thr0p0001_fp075.yaml").read_text())
    changed = {key for key in previous.keys() | config.keys() if previous.get(key) != config.get(key)}
    if changed != {"cache_dir", "save_folder", "plots_folder", "flip_horizontal"}:
        raise ValueError("This pipeline requires an orientation-only change from the cutoff0.0001 recipe")
    if config["rotation_k"] != 3 or config["flip_horizontal"] is not True:
        raise ValueError("Expected clockwise rotation followed by horizontal correction")
    return config


def clinical_preflight(clinical_root):
    reference = json.loads(REFERENCE.read_text())
    cases = evaluate_clinical.discover_cases(clinical_root, 100)
    tools = evaluate_clinical.clinical_tools()
    source_hashes, annotation_hashes = {}, {}
    for case in cases:
        prior, current, _, _ = evaluate_clinical.load_case(case, tools)
        for name, array in (("prior", prior), ("current", current)):
            actual = evaluate_clinical.prepare_image(array, 512, 3, True)
            expected = _norm01(_to_chw(array, 512)).transpose(-2, -1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            landmarks = torch.zeros(array.shape)
            landmarks[1, 2], landmarks[-3, -2] = 1, -1
            oriented = orient_tensor(landmarks, 3, True)
            restored = evaluate_clinical.restore_prediction(oriented, array.shape, 3, True)
            np.testing.assert_array_equal(restored, landmarks.numpy())
            source_hashes[f"{case['pair_id']}/{name}"] = sha256(case[name])
        for doctor, path in case["annotations"].items():
            key = f"{case['pair_id']}/{doctor}"
            payload = Path(path).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            normalized = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
            if reference["annotation_sha256"].get(key) not in (digest, normalized):
                raise ValueError(f"Clinical reference annotation changed: {key}")
            annotation_hashes[key] = digest
    if len(cases) != 100 or len(annotation_hashes) != 500:
        raise ValueError("Clinical preflight requires all100 pairs and500 annotations")
    return dict(num_pairs=100, source_sha256=source_hashes, annotation_sha256=annotation_hashes)


def preflight(config_path, destination, clinical_root):
    config = load_config(config_path)
    report = audit_gt.run(SimpleNamespace(
        dataset_root=Path(config["dataset_root"]), out_dir=destination,
        splits=["train", "val"], limit=0, seed=config["seed"], img_size=512,
        rotation_k=3, flip_horizontal=True, verify_alignment=True,
        target_threshold=config["target_threshold"], thresholds=[0., .0001, .02, .05],
        component_thresholds=[.0001, .05], overlays_per_group=1, save_arrays=False))
    if report["issues"] or not report["selected_pairs"] or not (
            report["selected_pairs"] == report["audited_pairs"] == report["alignment_verified_pairs"]):
        raise ValueError(f"DRR/GT alignment audit failed; inspect {destination / 'report.json'}")
    provenance = dict(config_sha256=sha256(config_path), audit_sha256=sha256(destination / "report.json"),
                      manifest_sha256=report["manifest_sha256"], clinical=clinical_preflight(clinical_root),
                      code_sha256={name: sha256(ROOT / name) for name in CODE_PATHS},
                      panels_sha256={row["panel"]: sha256(destination / row["panel"])
                                     for row in report["previews"]},
                      limitation="Grid/transform checks do not prove biological GT correctness; review overlays.")
    if not provenance["panels_sha256"]:
        raise ValueError("Orientation approval requires real DRR/GT overlays")
    write_json(destination / "orientation_provenance.json", provenance)
    return approval_hash(provenance)


def validate_approval(config_path, destination, approved_hash):
    provenance = json.loads((destination / "orientation_provenance.json").read_text())
    if not approved_hash or approval_hash(provenance) != approved_hash:
        raise ValueError("Review the real DRR/GT overlays and supply their exact approval hash before training")
    if sha256(config_path) != provenance["config_sha256"]:
        raise ValueError("Training configuration changed since the approved audit")
    if sha256(destination / "report.json") != provenance["audit_sha256"]:
        raise ValueError("Approved audit report changed")
    for name, digest in provenance["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Preprocessing/training code changed after the audit: {name}")
    for name, digest in provenance["panels_sha256"].items():
        if sha256(destination / name) != digest:
            raise ValueError(f"Approved DRR/GT overlay changed: {name}")
    config = load_config(config_path)
    if sha256(Path(config["dataset_root"]) / "manifest_split.jsonl") != provenance["manifest_sha256"]:
        raise ValueError("Dataset manifest changed since approval")
    return provenance


def verify_predictions(clinical_run, cases):
    import nibabel as nib

    metadata = json.loads((clinical_run / "metadata.json").read_text())
    report = json.loads((clinical_run / "report.json").read_text())
    if (metadata["num_pairs"] != 100 or report["pairs_completed"] != 100 or
            metadata["prediction_threshold"] != .5 or metadata["rotation_k"] != 3 or
            metadata.get("flip_horizontal") is not True):
        raise ValueError("Incomplete or incorrectly oriented clinical prediction run")
    if len(cases) != 100 or {case["pair_id"] for case in cases} != {f"pair{number}" for number in range(1, 101)}:
        raise ValueError("Expected the exact clinical100 cohort")
    hashes = {}
    for case in cases:
        folder = clinical_run / case["pair_id"]
        current = nib.load(case["current"])
        output, mask = [nib.load(folder / name) for name in ("output.nii.gz", "mask.nii.gz")]
        for image in (output, mask):
            if image.shape != current.shape or not np.allclose(image.affine, current.affine, rtol=0, atol=1e-5):
                raise ValueError(f"Prediction not in original image coordinates: {case['pair_id']}")
        values = np.asarray(output.dataobj)
        if not np.isfinite(values).all() or np.abs(values).max() > 1.00001:
            raise ValueError(f"Invalid signed scores: {case['pair_id']}")
        expected_mask = (values > .5).astype(np.int8) - (values < -.5).astype(np.int8)
        np.testing.assert_array_equal(np.asarray(mask.dataobj), expected_mask)
        hashes[case["pair_id"]] = sha256(folder / "output.nii.gz")
    return hashes


def publish(destination):
    repository = ROOT.parent.resolve()
    destination = destination.resolve()
    destination.relative_to((ROOT / "results").resolve())
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() or path.stat().st_size >= 95_000_000 for path in files):
        raise ValueError("Publication requires regular artifacts below the Git file-size limit")
    paths = [path.relative_to(repository).as_posix() for path in files]
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    if branch != "main":
        raise ValueError("Campus publication expects the main branch")
    subprocess.run(["git", "pull", "--ff-only"], cwd=repository, check=True)
    subprocess.run(["git", "add", "-f", "--", *paths], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", f"Publish {destination.name}", "--", *paths],
                   cwd=repository, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repository, check=True)
    print(f"[publish] pushed only {len(paths)} files under {destination}", flush=True)


def train_and_predict(config_path, approved_directory, approved_hash, clinical_root):
    config = load_config(config_path)
    provenance = validate_approval(config_path, approved_directory, approved_hash)
    run = Path(config["save_folder"]).parent
    environment = dict(os.environ, HEAD_CONFIG=str(config_path), DATA=config["dataset_root"],
                       CACHE_DIR=config["cache_dir"], RUN=str(run))
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["bash", "training/run_rotated_mask.sh", "--grad_clip", "1.0"],
                   cwd=ROOT, env=environment, check=True)
    last = torch.load(run / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    if last["epoch"] != config["epochs"]:
        raise ValueError("Training did not complete the configured epoch budget; predictions not published")
    checkpoint = run / "checkpoints/best.pt"
    best = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for state in (last, best):
        saved = state["run_config"]
        for key in ("rotation_k", "flip_horizontal", "target_mode", "target_threshold", "eval_threshold", "cache_dir"):
            if saved.get(key) != config[key]:
                raise ValueError(f"Checkpoint configuration mismatch: {key}")
    best_epoch = best["epoch"]
    del last, best
    if clinical_preflight(clinical_root) != provenance["clinical"]:
        raise ValueError("Clinical source scans/annotations changed since orientation approval")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clinical_run = run / f"clinical100_{timestamp}"
    checkpoint_hash = sha256(checkpoint)
    evaluate_clinical.run(SimpleNamespace(
        clinical_root=clinical_root, num_pairs=100, preflight=False, ckpt=checkpoint,
        expected_sha256=checkpoint_hash, out_dir=clinical_run, device="cuda", threshold=.5))
    cases = evaluate_clinical.discover_cases(clinical_root, 100)
    hashes = verify_predictions(clinical_run, cases)
    f1 = clinical_run / "detection_f1"
    compare_clinical_models.run(SimpleNamespace(
        clinical_run=clinical_run, clinical_root=clinical_root, num_pairs=100,
        foundation_only=True, itamar_preds=None, out_dir=f1))
    destination = ROOT / "results" / f"clinical100_upright_cutoff0001_{timestamp}"
    destination.mkdir()
    for name in ("metadata.json", "report.json", "per_pair.csv"):
        shutil.copy2(clinical_run / name, destination / name)
    for case in cases:
        folder = destination / case["pair_id"]
        folder.mkdir()
        for name in ("output.nii.gz", "mask.nii.gz", "source.json"):
            shutil.copy2(clinical_run / case["pair_id"] / name, folder / name)
    (destination / "detection_f1").mkdir()
    for name in ("metadata.json", "report.json", "f1_per_clinician.csv"):
        shutil.copy2(f1 / name, destination / "detection_f1" / name)
    for name in ("metrics.csv", "run_config.json"):
        shutil.copy2(run / "checkpoints" / name, destination / name)
    shutil.copy2(config_path, destination / "training_config.yaml")
    shutil.copy2(run / "input_preview.png", destination / "input_preview.png")
    for epoch in sorted({config["epochs"], best_epoch}):
        for split in ("train", "val"):
            name = f"epoch_{epoch:04d}_{split}.json"
            shutil.copy2(run / "checkpoints/diagnostics" / name, destination / name)
    write_json(destination / "verification.json", dict(
        code_commit=code_commit, approved_preflight=str(approved_directory),
        approved_preflight_sha256=approved_hash, checkpoint_sha256=checkpoint_hash,
        checkpoint_epoch=best_epoch, prediction_sha256=hashes, pairs_verified=len(hashes),
        geometry="original current-image NIfTI shape and affine; inverse flip before inverse rotation",
        scope="100 clinical transfer pairs; no synthetic held-out test evaluation; no encoder fine-tuning"))
    publish(destination)
    print(f"[complete] clinical100 predictions and reports pushed: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clinical-root", type=Path, default=ROOT.parent / "python_files/annotation tool")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--approved-preflight-sha256")
    args = parser.parse_args()
    if args.preflight_only:
        destination = args.preflight_dir or ROOT / "results" / f"orientation_preflight_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            digest = preflight(args.config, destination, args.clinical_root)
        except (ValueError, AssertionError, OSError) as error:
            if (destination / "report.json").is_file():
                write_json(destination / "preflight_failure.json", dict(error=str(error), training_allowed=False))
                publish(destination)
            raise
        publish(destination)
        print(f"[review first] {destination / 'index.md'}\nApproval SHA256: {digest}", flush=True)
    else:
        if not args.preflight_dir or not args.approved_preflight_sha256:
            parser.error("Run --preflight-only, review the overlays, then supply --preflight-dir and --approved-preflight-sha256")
        import fcntl
        run = Path(load_config(args.config)["save_folder"]).parent
        run.mkdir(parents=True, exist_ok=True)
        with (run / ".upright_pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train_and_predict(args.config, args.preflight_dir, args.approved_preflight_sha256, args.clinical_root)


if __name__ == "__main__":
    main()