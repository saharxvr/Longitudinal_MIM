"""Controlled architecture x patient-fraction sweep, with independent subprocess cells."""

import argparse
import copy
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

import constants as C
from datasets.augmentations import PHOTOMETRIC_RECIPE
from datasets.pair_dataset import (
    LongitudinalPairDataset, _read_manifest, audit_patient_splits,
    pair_inventory, patient_subset_indices, record_patient,
)
from models import FrozenCXRBackbone, build_change_head, signed_change_map
from training.positional_debias import NOISE_METHOD, diagnose_features
from training.train_foundation_diff import (
    _tokens_from_batch, cosine_warmup, head_spec, make_diagnostics, objective, parse_args,
)
from training.train_layer_ablation import (
    atomic_save, capture_rng, configure_generators, cpu_weights, restore_rng,
)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("B0", "B1", "B2")
MIN_BENCHMARK_BATCHES = 100
CODE_FILES = (
    "training/train_architecture_data_ablation.py", "training/positional_debias.py",
    "training/train_foundation_diff.py", "training/train_layer_ablation.py",
    "models/backbone.py", "models/dpt_head.py", "models/difference_head.py", "models/__init__.py",
    "datasets/pair_dataset.py", "datasets/augmentations.py", "data_generation/split_dataset.py",
    "losses/change_map_losses.py", "evaluation/training_diagnostics.py", "constants.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(host=socket.gethostname(), command=sys.argv)) + "\n")
    try:
        yield
    finally:
        path.unlink()


def validate_config(args):
    if (args.head != "dpt_cross_attention" or args.feat_last_k != 1
            or args.backbone_layer != 9 or args.feature_normalization != "final_layernorm"
            or not args.no_cache or not args.photometric_augmentation or not args.balanced
            or args.num_workers != 0 or args.overfit or args.resume_from or args.fusion_arms
            or args.loss_mode != "ce_dice" or args.target_mode != "signed_mask"
            or args.checkpoint_metric != "target_signed_dice" or args.dice_warmup_epochs
            or not args.diagnostics or args.diagnostics_every != 1):
        raise ValueError("Use the dedicated online, augmented, block-9 CE/Dice ablation config")
    if (args.w_ce != 1 or args.w_dice != 1 or args.ce_class_weights != [1, 1, 1]
            or (args.w_l1, args.w_l2, args.w_sign) != (0, 0, 0)
            or args.target_threshold != .001 or args.eval_threshold != .5):
        raise ValueError("Keep the matched CE/Dice objective and declared thresholds")
    if (any(type(value) is not int or value < 1
            for value in (args.epochs, args.batch_size, args.accum))
            or not 0 <= args.warmup_epochs < args.epochs
            or args.batch_size * args.accum != 32 or args.grad_clip <= 0
            or not all(math.isfinite(value) for value in (
                args.lr, args.weight_decay, args.grad_clip, args.warmup_epochs))
            or args.lr <= 0 or args.weight_decay < 0):
        raise ValueError("Invalid update budget or optimizer settings; effective batch must be 32")
    if (not args.data_fractions or args.data_fractions != sorted(set(args.data_fractions))
            or args.data_fractions[-1] != 1
            or any(not math.isfinite(f) or not 0 < f <= 1 for f in args.data_fractions)
            or not args.seeds or len(args.seeds) != len(set(args.seeds))
            or any(type(seed) is not int or seed < 0 for seed in args.seeds)
            or args.arms != list(ARMS)):
        raise ValueError("Require unique seeds, nested fractions ending at 100%, and arms B0/B1/B2")
    if (args.eval_gt_threshold != .05 or args.rotation_k != 3 or not args.flip_horizontal
            or not args.diagnostics_components or type(args.diagnostic_seed) is not int
            or args.diagnostic_seed < 0 or type(args.diagnostic_train_pairs) is not int
            or args.diagnostic_train_pairs < 1):
        raise ValueError("Keep upright inputs, raw-GT0.05 component diagnostics and valid noise settings")


def make_backbone(args):
    return FrozenCXRBackbone(
        name=C.BACKBONE, model_id=C.RAD_DINO_MODEL,
        backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
        last_n_layers=1, selected_layer=args.backbone_layer, freeze=True,
    ).to(args.device).eval()


def prepare(args):
    validate_config(args)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not start the GPU sweep")
    manifest = Path(args.dataset_root) / "manifest_split.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError("The controlled sweep requires manifest_split.jsonl")
    records = _read_manifest(args.dataset_root)
    inventory = audit_patient_splits(records)
    datasets = {split: LongitudinalPairDataset(
        args.dataset_root, split=split, img_size=C.IMG_SIZE,
        rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
        for split in ("train", "val", "test")}
    for dataset in datasets.values():
        for record in dataset.records:
            for key, filename in (("prior_path", "prior.nii.gz"), ("current_path", "current.nii.gz"),
                                  ("diff_map_path", "diff_map.nii.gz")):
                if not Path(dataset._abspath(record, key, filename)).is_file():
                    raise FileNotFoundError(f"Missing {key} for selected pair {record['pair_id']}")
    updates = len(datasets["train"]) // (args.batch_size * args.accum)
    if not updates:
        raise ValueError("Full training split must support a complete optimizer update")
    backbone = make_backbone(args)
    if any(parameter.requires_grad for parameter in backbone.parameters()) or backbone.training:
        raise ValueError("Backbone must be frozen and in eval mode")
    model_config = backbone.backbone.config
    feature_provenance = dict(
        manifest_sha256=sha256(manifest),
        model_id=C.RAD_DINO_MODEL, backbone_revision=getattr(model_config, "_commit_hash", None),
        backbone_config=model_config.to_dict(), backbone_layer=args.backbone_layer,
        feature_normalization=args.feature_normalization, image_size=C.IMG_SIZE,
        backbone_image_size=C.BACKBONE_IMG_SIZE, rotation_k=args.rotation_k,
        flip_horizontal=args.flip_horizontal, noise_method=NOISE_METHOD)
    feature_provenance = json.loads(json.dumps(feature_provenance))
    provenance = dict(features=feature_provenance, photometric_recipe=PHOTOMETRIC_RECIPE,
                      runtime=dict(torch=torch.__version__, cuda=torch.version.cuda),
                      code_sha256={name: sha256(ROOT / name) for name in CODE_FILES})
    return dict(datasets=datasets, backbone=backbone, inventory=inventory,
                updates_per_slot=updates, provenance=provenance)


def locked_basis(args, context):
    if (type(args.debias_rank) is not int or not 0 < args.debias_rank < C.BACKBONE_DIM
            or not isinstance(args.diagnostic_sha256, str)
            or len(args.diagnostic_sha256) != 64):
        raise ValueError("Inspect --mode diagnose first, then lock debias_rank and diagnostic_sha256 in config")
    path = Path(args.diagnostic_file)
    if sha256(path) != args.diagnostic_sha256:
        raise ValueError("Debias diagnostic hash changed")
    report = torch.load(path, map_location="cpu", weights_only=False)
    if report["provenance"] != context["provenance"]:
        raise ValueError("Debias diagnostic code/data/backbone/preprocessing provenance differs")
    rows = {row["rank"]: row for row in report["rows"]}
    if args.debias_rank not in rows or report["basis"].shape[1] < args.debias_rank:
        raise ValueError("Choose a rank actually evaluated by the diagnostic")
    return report["basis"][:, :args.debias_rank].contiguous()


def run_diagnostics(args, context):
    path = Path(args.diagnostic_file)
    if any(candidate.exists() for candidate in (path, path.with_suffix(".json"), path.with_suffix(".png"))):
        raise ValueError("Diagnostic artifacts already exist; use a new path rather than overwrite")
    result = diagnose_features(context["backbone"], context["datasets"]["train"],
                               args.diagnostic_ranks, args.diagnostic_seed, args.device,
                               args.diagnostic_train_pairs)
    result["provenance"] = context["provenance"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(result, path)
    report = {key: value for key, value in result.items() if key != "basis"}
    report["diagnostic_sha256"] = sha256(path)
    write_json(path.with_suffix(".json"), report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ranks = [row["rank"] for row in result["rows"]]
    axes[0].plot(ranks, [row["positional_gap"] for row in result["rows"]], marker="o")
    axes[0].set_ylabel("Held-out noise: same-minus-other-position cosine")
    axes[1].plot(ranks, [row["train_energy_retained_mean"] for row in result["rows"]], marker="o")
    axes[1].set_ylabel("Training patch-feature energy retained")
    for axis in axes:
        axis.set_xlabel("Removed rank")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)


def cell_name(arm, fraction, seed):
    return f"{arm}_fraction_{fraction:g}_seed_{seed}"


def cell_identity(args, context, arm, fraction, seed):
    config = {key: value for key, value in vars(args).items()
              if key not in ("config", "resume", "benchmark_batches")}
    indices = patient_subset_indices(context["datasets"]["train"].records, fraction, seed)
    records = [context["datasets"]["train"].records[index] for index in indices]
    selection = dict(
        pair_ids=[record["pair_id"] for record in records],
        patients=sorted({record_patient(record) for record in records}),
        inventory=pair_inventory(records), requested_patient_fraction=fraction,
        actual_pair_fraction=len(records) / len(context["datasets"]["train"]))
    return dict(config=config, provenance=context["provenance"], arm=arm, fraction=fraction,
                seed=seed, selection=selection, updates_per_slot=context["updates_per_slot"]), indices


def make_cell_head(args, arm, basis):
    spec = head_spec(C.BACKBONE_DIM, "dpt_cross_attention", args.dpt_channels, args.loss_mode)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        reference = build_change_head(spec, "cpu")
        if arm == "B0":
            head = reference
        else:
            spec = copy.deepcopy(spec)
            spec["type"] = "dpt_global_attention"
            spec["kwargs"]["debias_rank"] = basis.shape[1] if arm == "B2" else 0
            torch.random.default_generator.manual_seed(args.seed)
            head = build_change_head(spec, "cpu")
            weights = head.state_dict()
            for name, value in reference.state_dict().items():
                if name in weights and weights[name].shape == value.shape:
                    weights[name] = value
            head.load_state_dict(weights)
            if arm == "B2":
                head.set_debias_basis(basis)
    head.z_proj.requires_grad_(False)
    return spec, head.to(args.device)


def make_cell_loaders(args, context, indices, updates=None):
    dataset = context["datasets"]["train"]
    labels = torch.tensor([dataset._anomaly_index(dataset.records[index]) for index in indices])
    counts = torch.bincount(labels, minlength=len(C.ANOMALY_TYPES)).float()
    weights = counts.clamp(min=1).reciprocal()[labels]
    draws = (updates or context["updates_per_slot"]) * args.batch_size * args.accum
    sampler = WeightedRandomSampler(weights, draws, replacement=True)
    train = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                       sampler=sampler, num_workers=0, drop_last=True)
    val = DataLoader(context["datasets"]["val"], batch_size=args.batch_size, num_workers=0)
    test = DataLoader(context["datasets"]["test"], batch_size=args.batch_size, num_workers=0)
    return train, val, test, configure_generators(train, val, args.seed)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def diagnostic_metrics(diagnostics, args):
    report = diagnostics.summary()
    if report["invalid_pairs"]:
        raise ValueError("Invalid pairs in prediction diagnostics")
    rows = {(row["gt_threshold"], row["sign"]): row for row in diagnostics.sweep_rows()
            if row["group"] == "pathology" and row["prediction_threshold"] == args.eval_threshold}
    result = {}
    for cutoff, prefix in ((args.target_threshold, "target"), (args.eval_gt_threshold, "raw")):
        for sign in ("signed", "positive", "negative", "support"):
            row = rows.get((cutoff, sign))
            result[f"{prefix}_{sign}_dice"] = row["micro_dice"] if row else None
        row = rows.get((cutoff, "signed"))
        result[f"{prefix}_macro_signed_dice"] = row["macro_dice"] if row else None
    nuisance = report["groups"]["nuisance"]
    result.update(nuisance_fp_mean=nuisance.get("predicted_area", {}).get("mean"),
                  nuisance_fp_p95=nuisance.get("predicted_area_p95"),
                  nuisance_fp_max=nuisance.get("predicted_area_max"),
                  component_f1=report["groups"]["pathology"].get("component_all_pooled_f1"))
    return result


def active_overlap_counts(prediction, ground_truth, pathology, threshold, gt_threshold):
    positive, negative = ground_truth > gt_threshold, ground_truth < -gt_threshold
    active = (positive | negative).flatten(1).any(1) & pathology.to(ground_truth.device).bool()
    p, n = prediction[active] > threshold, prediction[active] < -threshold
    positive, negative = positive[active], negative[active]
    result = {}
    for sign, predicted, reference in (
            ("positive", p, positive), ("negative", n, negative),
            ("support", p | n, positive | negative)):
        result[sign] = [int((predicted & reference).sum()),
                        int(predicted.sum()), int(reference.sum())]
    result["signed"] = [sum(values) for values in zip(result["positive"], result["negative"])]
    return result


def run_pass(head, backbone, loader, args, optimizer=None, scheduler=None, warmup_updates=0):
    training = optimizer is not None
    head.train(training)
    diagnostics = make_diagnostics(args)
    totals, draws, updates, batches = {}, 0, 0, 0
    counts = {prefix: {sign: [0, 0, 0] for sign in ("positive", "negative", "support", "signed")}
              for prefix in ("target", "raw")}
    synchronize(args.device)
    started = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(loader):
        if training and index == warmup_updates * args.accum:
            synchronize(args.device)
            started = time.perf_counter()
            if str(args.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
        with torch.set_grad_enabled(training):
            tokens = _tokens_from_batch(batch, backbone, args.device,
                                        augment=training and args.photometric_augmentation)
            output, _ = head(*tokens)
            loss, components = objective(output, batch["gt_diff"].to(args.device), args, args.w_dice)
            if not torch.isfinite(output).all() or not torch.isfinite(loss):
                raise ValueError(f"Nonfinite output/loss at batch {index + 1}; no update is skipped")
            prediction = signed_change_map(output).detach()
            ground_truth = batch["gt_diff"].to(args.device)
            diagnostics.update(prediction, ground_truth, batch)
            for prefix, cutoff in (("target", args.target_threshold), ("raw", args.eval_gt_threshold)):
                batch_counts = active_overlap_counts(
                    prediction, ground_truth, batch["is_pathology"], args.eval_threshold, cutoff)
                for sign, values in batch_counts.items():
                    counts[prefix][sign] = [a + b for a, b in zip(counts[prefix][sign], values)]
            if training:
                (loss / args.accum).backward()
                if (index + 1) % args.accum == 0:
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                    if not torch.isfinite(norm):
                        raise ValueError("Nonfinite gradient; cell stopped before optimizer update")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    updates += 1
        for key, value in components.items():
            totals[key] = totals.get(key, 0) + value
        draws += len(batch["pair_id"])
        batches += 1
    if not batches or (training and batches % args.accum):
        raise ValueError("Pass must contain complete accumulation windows")
    synchronize(args.device)
    measured = batches - warmup_updates * args.accum
    if measured <= 0:
        raise ValueError("No measured batches after benchmark warmup")
    metrics = dict(**{f"loss_{key}": value / batches for key, value in totals.items()},
                   **diagnostic_metrics(diagnostics, args), optimizer_updates=updates,
                   sample_draws=draws, seconds=time.perf_counter() - started, measured_batches=measured)
    for prefix, signs in counts.items():
        for sign, (intersection, predicted, reference) in signs.items():
            metrics[f"{prefix}_{sign}_dice"] = (2 * intersection / (predicted + reference)
                                               if reference else None)
    return metrics, diagnostics


def optimizer_state(args, head, updates):
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_warmup(optimizer, int(updates * args.warmup_epochs), updates * args.epochs)
    return optimizer, scheduler


def sweep_identity(args, context):
    return dict(config={key: value for key, value in vars(args).items()
                        if key not in ("config", "resume", "benchmark_batches")},
                provenance=context["provenance"], inventory=context["inventory"],
                updates_per_slot=context["updates_per_slot"])


def run_benchmark(args, context, basis):
    path = Path(args.run_root) / "benchmark.json"
    if path.exists() or (Path(args.run_root) / "sweep.json").exists():
        raise ValueError("Benchmark requires a new run root")
    identity = sweep_identity(args, context)
    args = copy.copy(args)
    args.seed = args.seeds[0]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    _, indices = cell_identity(args, context, "B2", 1., args.seed)
    measured_updates = math.ceil(MIN_BENCHMARK_BATCHES / args.accum)
    warmup = 5
    train, val, _, _ = make_cell_loaders(args, context, indices, measured_updates + warmup)
    _, head = make_cell_head(args, "B2", basis)
    optimizer, scheduler = optimizer_state(args, head, context["updates_per_slot"])
    train_metrics, _ = run_pass(head, context["backbone"], train, args, optimizer, scheduler, warmup)
    val_metrics, _ = run_pass(head, context["backbone"], val, args)
    if val_metrics["target_signed_dice"] is None:
        raise ValueError("Validation has no active signed targets")
    memory = (dict(allocated=torch.cuda.max_memory_allocated(args.device),
                   reserved=torch.cuda.max_memory_reserved(args.device))
              if str(args.device).startswith("cuda") else None)
    write_json(path, dict(identity=identity, training=train_metrics,
                          validation=val_metrics, peak_cuda_bytes=memory,
                          note="Largest B2 arm, full-data sampling, actual augmentation and diagnostics. "
                               "Includes validation; excludes checkpoint writes."))
    print(f"[benchmark] {train_metrics['measured_batches']} measured batches passed; {path}", flush=True)


def verify_completion(folder, identity):
    folder = Path(folder)
    completion = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if completion["identity_sha256"] != json_hash(identity):
        raise ValueError("Completed cell identity differs")
    if completion["optimizer_updates"] != identity["updates_per_slot"] * identity["config"]["epochs"]:
        raise ValueError("Completed cell has the wrong optimizer-update budget")
    for filename, digest in completion["files"].items():
        if sha256(folder / filename) != digest:
            raise ValueError(f"Completed cell artifact changed: {filename}")
    required = {"last.pt", "best.pt", "results.json", "metrics.csv"}
    if set(completion["files"]) != required:
        raise ValueError("Incomplete completion record")
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def run_cell(args, context, arm, fraction, seed, resume=False):
    if arm not in ARMS or fraction not in args.data_fractions or seed not in args.seeds:
        raise ValueError("Cell is not in the configured matrix")
    basis = locked_basis(args, context)
    identity, indices = cell_identity(args, context, arm, fraction, seed)
    folder = Path(args.run_root) / cell_name(arm, fraction, seed)
    if (folder / "complete.json").exists():
        if not resume:
            raise ValueError("Existing completed cell; use --resume or a new run root")
        return verify_completion(folder, identity)
    snapshot = folder / "last.pt"
    if folder.exists() and any(folder.iterdir()) and (not resume or not snapshot.exists()):
        raise ValueError("Existing cell artifacts without an approved compatible resume")
    folder.mkdir(parents=True, exist_ok=True)
    args = copy.copy(args)
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    train, val, test, generators = make_cell_loaders(args, context, indices)
    spec, head = make_cell_head(args, arm, basis)
    optimizer, scheduler = optimizer_state(args, head, context["updates_per_slot"])
    best_score, best_head, best_slot, history, start = -1., None, 0, [], 0
    run_config = dict(vars(args), arm=arm, patient_fraction=fraction, head=spec["type"],
                      total_optimizer_updates=context["updates_per_slot"] * args.epochs)
    if snapshot.exists():
        saved = torch.load(snapshot, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["head_spec"] != spec:
            raise ValueError("Resume code/config/subset/basis/budget mismatch")
        start = saved["epoch"]
        if (not 1 <= start <= args.epochs
                or saved["optimizer_updates"] != start * context["updates_per_slot"]
                or saved["scheduler"]["last_epoch"] != saved["optimizer_updates"]
                or len(saved["history"]) != start
                or not 1 <= saved["best_slot"] <= start):
            raise ValueError("Invalid saved update position")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_score, best_head, best_slot = saved["best_score"], saved["best_head"], saved["best_slot"]
        history = saved["history"]
        restore_rng(saved["rng"], generators)
    write_json(folder / "selection.json", identity["selection"])
    write_json(folder / "run_config.json", run_config)
    write_json(folder / "provenance.json", identity)
    for slot in range(start + 1, args.epochs + 1):
        training, tr_diag = run_pass(head, context["backbone"], train, args, optimizer, scheduler)
        if training["optimizer_updates"] != context["updates_per_slot"]:
            raise ValueError("Cell did not receive its exact optimizer-update budget")
        validation, va_diag = run_pass(head, context["backbone"], val, args)
        score = validation["target_signed_dice"]
        if score is None:
            raise ValueError("Validation has no active signed targets for checkpoint selection")
        if score > best_score:
            best_score, best_head, best_slot = score, cpu_weights(head), slot
        history.append(dict(slot=slot, optimizer_updates=slot * context["updates_per_slot"],
                            lr=optimizer.param_groups[0]["lr"], train=training, val=validation))
        state = dict(head=cpu_weights(head), head_spec=spec, run_config=run_config,
                     epoch=slot, optimizer_updates=slot * context["updates_per_slot"],
                     identity=identity, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=capture_rng(generators), best_score=best_score, best_head=best_head,
                     best_slot=best_slot, history=history)
        if scheduler.last_epoch != state["optimizer_updates"]:
            raise ValueError("Scheduler and successful optimizer updates disagree")
        atomic_save(state, snapshot)
        tr_diag.save(folder / "diagnostics", slot, "train", history[-1])
        va_diag.save(folder / "diagnostics", slot, "val", history[-1])
        print(f"[{folder.name}] slot {slot}/{args.epochs} "
              f"updates={state['optimizer_updates']} val signed Dice={score:.6f}", flush=True)
    head.load_state_dict(best_head)
    atomic_save(dict(head=best_head, head_spec=spec, run_config=run_config, epoch=best_slot,
                     optimizer_updates=best_slot * context["updates_per_slot"],
                     best_score=best_score, best_metric="target_signed_dice", identity=identity),
                folder / "best.pt")
    test_metrics, test_diagnostics = run_pass(head, context["backbone"], test, args)
    if test_metrics["target_signed_dice"] is None:
        raise ValueError("Test split has no active signed targets")
    test_diagnostics.save(folder / "diagnostics", best_slot, "test")
    result = dict(arm=arm, fraction=fraction, seed=seed,
                  inventory=identity["selection"]["inventory"],
                  actual_pair_fraction=identity["selection"]["actual_pair_fraction"],
                  best_slot=best_slot, best_val_signed_dice=best_score, test=test_metrics)
    write_json(folder / "results.json", result)
    rows = [dict(slot=row["slot"], optimizer_updates=row["optimizer_updates"], lr=row["lr"],
                 **{f"{split}_{key}": value for split in ("train", "val")
                    for key, value in row[split].items()}) for row in history]
    write_csv(folder / "metrics.csv", rows)
    write_json(folder / "complete.json", dict(
        identity_sha256=json_hash(identity), optimizer_updates=args.epochs * context["updates_per_slot"],
        files={name: sha256(folder / name) for name in ("last.pt", "best.pt", "results.json", "metrics.csv")}))
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(root, results):
    metrics = ("target_signed_dice", "target_positive_dice", "target_negative_dice",
               "target_macro_signed_dice", "raw_signed_dice", "nuisance_fp_mean", "component_f1")
    rows = [dict(arm=r["arm"], fraction=r["fraction"], seed=r["seed"],
                 patients=r["inventory"]["patients"], pairs=r["inventory"]["pairs"],
                 best_slot=r["best_slot"], **{key: r["test"][key] for key in metrics}) for r in results]
    write_csv(root / "results.csv", rows)
    groups = []
    for arm in ARMS:
        for fraction in sorted({row["fraction"] for row in rows}):
            selected = [row for row in rows if row["arm"] == arm and row["fraction"] == fraction]
            summary = dict(arm=arm, fraction=fraction)
            for key in (*metrics, "patients", "pairs"):
                values = [row[key] for row in selected if row[key] is not None]
                summary[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                    sd=statistics.stdev(values) if len(values) > 1 else None)
            groups.append(summary)
    by_cell = {(row["arm"], row["fraction"], row["seed"]): row for row in rows}
    paired = []
    for row in rows:
        for left, right in (("B1", "B0"), ("B2", "B1")):
            if row["arm"] != left:
                continue
            reference = by_cell[(right, row["fraction"], row["seed"])]
            paired.append(dict(comparison=f"{left}-{right}", fraction=row["fraction"], seed=row["seed"],
                               **{key: row[key] - reference[key]
                                  if row[key] is not None and reference[key] is not None else None
                                  for key in metrics}))
    write_csv(root / "paired_differences.csv", paired)
    write_json(root / "summary.json", dict(groups=groups, cells=len(results),
               note="Mean/sample SD over training seeds (subset and optimization both vary). "
                    "Not reader variability or proof of significance; fractions denote patients. "
                    "B1-B0 combines global attention with image-role embeddings."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric, x_key in zip(
            axes.flat, ("target_signed_dice", "target_signed_dice", "nuisance_fp_mean", "component_f1"),
            ("patients", "pairs", "pairs", "pairs")):
        for arm in ARMS:
            points = [g for g in groups if g["arm"] == arm and g[metric]["mean"] is not None]
            axis.errorbar([g[x_key]["mean"] for g in points], [g[metric]["mean"] for g in points],
                          yerr=[g[metric]["sd"] or 0 for g in points], marker="o", label=arm)
        axis.set_xlabel(f"Actual training {x_key} (mean across seeds)")
        axis.set_ylabel(metric)
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "learning_curves.png", dpi=120)
    plt.close(figure)


def require_benchmark(args, context):
    root = Path(args.run_root)
    identity = sweep_identity(args, context)
    if not (root / "benchmark.json").is_file():
        raise ValueError("Run --mode benchmark after locking the debias diagnostic")
    benchmark = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
    if benchmark["identity"] != identity or benchmark["training"]["measured_batches"] < MIN_BENCHMARK_BATCHES:
        raise ValueError("Run the matching largest-arm benchmark before the sweep")
    return identity


def run_sweep(args, context, resume=False):
    locked_basis(args, context)
    root = Path(args.run_root)
    identity = require_benchmark(args, context)
    manifest = root / "sweep.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing sweep config/code/data/basis differs or --resume is missing")
    else:
        if any(root.glob("B*_fraction_*")):
            raise ValueError("Cell artifacts without a sweep manifest; use a new run root")
        write_json(manifest, identity)
    results = []
    # The parent does not retain an encoder on the GPU while the child runs.
    context["backbone"].to("cpu")
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    for fraction in args.data_fractions:
        for seed in args.seeds:
            for arm in args.arms:
                cell, _ = cell_identity(args, context, arm, fraction, seed)
                folder = root / cell_name(arm, fraction, seed)
                if not (folder / "complete.json").exists():
                    logs = root / "logs"
                    logs.mkdir(exist_ok=True)
                    log = logs / f"{folder.name}_{time.time_ns()}.log"
                    command = [sys.executable, "-u", "-m", "training.train_architecture_data_ablation",
                               "--config", str(Path(args.config).resolve()), "--mode", "cell",
                               "--arm", arm, "--fraction", str(fraction), "--seed", str(seed)]
                    if resume:
                        command.append("--resume")
                    print(f"[sweep] {folder.name}; log={log}", flush=True)
                    with log.open("x", encoding="utf-8") as stream:
                        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                results.append(verify_completion(folder, cell))
    aggregate_results(root, results)
    print(f"[sweep] verified {len(results)} independent cells; {root / 'summary.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("diagnose", "benchmark", "sweep", "cell"), default="sweep")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args(argv)
    args = parse_args(["--config", options.config])
    if options.mode == "cell":
        if options.arm is None or options.fraction is None or options.seed is None:
            parser.error("cell mode requires --arm, --fraction and --seed")
        lock = Path(args.run_root) / f".{cell_name(options.arm, options.fraction, options.seed)}.lock"
    else:
        lock = Path(args.run_root) / ".sweep.lock"
    with run_lock(lock):
        context = prepare(args)
        if options.mode == "diagnose":
            run_diagnostics(args, context)
        elif options.mode == "benchmark":
            run_benchmark(args, context, locked_basis(args, context))
        elif options.mode == "cell":
            require_benchmark(args, context)
            run_cell(args, context, options.arm, options.fraction, options.seed, options.resume)
        else:
            run_sweep(args, context, options.resume)


if __name__ == "__main__":
    main()
