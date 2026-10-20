"""Train independent single-depth heads using one shared frozen encoder pass."""

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import constants as C
from datasets.augmentations import augment_cxr
from models import build_change_head, signed_change_map
from training.train_foundation_diff import (
    cosine_warmup, head_spec, make_diagnostics, make_loaders, objective, parse_args,
    print_prediction_metrics, save_augmentation_preview,
)
from training.run_upright_clinical import publish, write_json


ROOT = Path(__file__).resolve().parents[1]
LOSS_KEYS = ("total", "ce", "dice", "dice_positive", "dice_negative")
CODE_FILES = (
    "training/train_layer_ablation.py", "training/train_foundation_diff.py",
    "training/run_layer_ablation.sh", "training/run_last4_dpt.sh", "models/backbone.py",
    "models/dpt_head.py", "models/__init__.py", "models/difference_head.py",
    "datasets/pair_dataset.py", "datasets/augmentations.py", "losses/change_map_losses.py",
    "evaluation/training_diagnostics.py", "constants.py",
)

def json_record(value):
    """Canonicalize JSON key types before comparing in-memory and saved records."""
    return json.loads(json.dumps(value, allow_nan=False))


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_weights(head):
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def configure_generators(train_loader, val_loader, seed):
    generators = {name: torch.Generator().manual_seed(seed + offset)
                  for offset, name in enumerate(("train", "val", "sampler"), 1)}
    train_loader.generator = generators["train"]
    val_loader.generator = generators["val"]
    train_loader.sampler.generator = generators["sampler"]
    return generators


def capture_rng(generators):
    return dict(torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=np.random.get_state(), python=random.getstate(),
                generators={name: generator.get_state() for name, generator in generators.items()})


def restore_rng(state, generators):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    for name, value in state["generators"].items():
        generators[name].set_state(value)


def build_branches(args, device, steps):
    spec = head_spec(C.BACKBONE_DIM, args.head, args.dpt_channels, args.loss_mode)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        prototype = build_change_head(spec, "cpu")
    if any(isinstance(module, torch.nn.Dropout) and module.p > 0 for module in prototype.modules()):
        raise ValueError("This ablation requires deterministic heads without dropout")
    branches = {}
    for layer in args.ablation_layers:
        head = copy.deepcopy(prototype).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        branches[layer] = dict(
            head=head, optimizer=optimizer,
            scheduler=cosine_warmup(optimizer, int(steps * args.warmup_epochs), steps * args.epochs),
            best_score=-1., best_epoch=0, best_head=None, best_val=None, history=[])
    return spec, branches


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def run_epoch(branches, backbone, loader, args, training, limit=None, warmup=0,
              collect_diagnostics=True):
    """Share image augmentation/extraction; free each branch graph before the next."""
    device = args.device
    losses = {layer: dict.fromkeys(LOSS_KEYS, 0.) for layer in branches}
    counts = {layer: [0, 0, 0] for layer in branches}
    diagnostics = {layer: make_diagnostics(args) if collect_diagnostics else None for layer in branches}
    for branch in branches.values():
        branch["head"].train(training)
        if training:
            branch["optimizer"].zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    steps = 0
    for batch_index, batch in enumerate(loader):
        if limit is not None and batch_index >= limit + warmup:
            break
        if batch_index == warmup:
            synchronize(device)
            started = time.perf_counter()
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            prior, current = batch["img_prior"].to(device), batch["img_curr"].to(device)
            if training and args.photometric_augmentation:
                prior, current = augment_cxr(prior), augment_cxr(current)
            fp = backbone.forward_layers(prior, list(branches))
            fc = backbone.forward_layers(current, list(branches))
        gt = batch["gt_diff"].to(device)
        if not torch.isfinite(gt).all():
            raise ValueError(f"Nonfinite GT in shared batch {batch_index + 1}")
        for layer, branch in branches.items():
            with torch.set_grad_enabled(training):
                output, embedding = branch["head"](
                    fp[layer]["patch_tokens"], fc[layer]["patch_tokens"],
                    fp[layer]["cls_token"], fc[layer]["cls_token"])
                loss, components = objective(output, gt, args, args.w_dice)
                if not torch.isfinite(loss) or not torch.isfinite(output).all():
                    raise ValueError(f"Nonfinite output/loss: layer {layer}, batch {batch_index + 1}")
                pred = signed_change_map(output).detach()
                if diagnostics[layer] is not None:
                    diagnostics[layer].update(pred, gt, batch)
                positive, negative = gt > args.target_threshold, gt < -args.target_threshold
                active = ((positive | negative).flatten(1).any(1)
                          & batch["is_pathology"].to(device).bool())
                p, n = pred[active] > args.eval_threshold, pred[active] < -args.eval_threshold
                counts[layer][0] += int((p & positive[active]).sum() + (n & negative[active]).sum())
                counts[layer][1] += int(p.sum() + n.sum())
                counts[layer][2] += int(positive[active].sum() + negative[active].sum())
                for key in LOSS_KEYS:
                    losses[layer][key] += components[key]
                if training:
                    (loss / args.accum).backward()
                    if (batch_index + 1) % args.accum == 0:
                        norm = torch.nn.utils.clip_grad_norm_(branch["head"].parameters(), args.grad_clip)
                        if not torch.isfinite(norm):
                            raise ValueError(f"Nonfinite gradients: layer {layer}")
                        branch["optimizer"].step()
                        branch["optimizer"].zero_grad(set_to_none=True)
                        branch["scheduler"].step()
                del output, embedding, loss, pred
        steps += 1
        if training and (steps == 1 or steps % 100 == 0):
            print(f"[shared train] {steps} batches | "
                  + " ".join(f"L{layer}={values['total']/steps:.5f}" for layer, values in losses.items()),
                  flush=True)
        del fp, fc
    if steps <= warmup:
        raise ValueError("No measured batches; check dataset and benchmark length")
    if training:
        for branch in branches.values():
            branch["optimizer"].zero_grad(set_to_none=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = {}
    for layer in branches:
        intersection, predicted, reference = counts[layer]
        if not reference:
            raise ValueError(f"No active pathology targets: layer {layer}")
        metrics[layer] = {**{key: value / steps for key, value in losses[layer].items()},
                          "target_signed_dice": 2 * intersection / (predicted + reference)}
    timing = dict(seconds=elapsed, measured_batches=steps - warmup,
                  seconds_per_shared_batch=elapsed / (steps - warmup),
                  optimizer_steps_per_head=steps // args.accum if training else 0,
                  discarded_accumulation_batches=steps % args.accum if training else 0)
    return metrics, diagnostics, timing


def diagnostic_metrics(diagnostics, args):
    matched = next(row for row in diagnostics.sweep_rows()
                   if row["group"] == "pathology" and row["sign"] == "signed"
                   and row["prediction_threshold"] == args.eval_threshold
                   and row["gt_threshold"] == args.target_threshold)
    summary = diagnostics.summary()
    if summary["invalid_pairs"]:
        raise ValueError("Invalid pairs in layer diagnostics")
    nuisance = summary["groups"]["nuisance"]
    return dict(macro_signed_dice=matched["macro_dice"],
                nuisance_fp_mean=nuisance.get("predicted_area", {}).get("mean"),
                nuisance_fp_p95=nuisance.get("predicted_area_p95"),
                nuisance_fp_max=nuisance.get("predicted_area_max"),
                component_f1=summary["groups"]["pathology"].get("component_all_pooled_f1"))


def export_branches(branches, spec, args):
    for layer, branch in branches.items():
        folder = Path(args.save_folder) / f"layer_{layer:02d}"
        folder.mkdir(exist_ok=True)
        config = dict(vars(args), backbone_layer=layer)
        write_json(folder / "run_config.json", config)
        for kind in ("best", "last"):
            best = kind == "best"
            atomic_save(dict(
                head=branch["best_head"] if best else cpu_weights(branch["head"]),
                head_spec=spec, run_config=config,
                epoch=branch["best_epoch"] if best else branch["history"][-1]["epoch"],
                best_score=branch["best_score"], best_metric="target_signed_dice",
                val=branch["best_val"] if best else branch["history"][-1]),
                folder / f"{kind}.pt")
        with (folder / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(branch["history"][0]))
            writer.writeheader()
            writer.writerows(branch["history"])


def validate_args(args):
    if (not args.no_cache or args.feat_last_k != 1 or args.head != "dpt_cross_attention"
            or args.loss_mode != "ce_dice" or args.checkpoint_metric != "target_signed_dice"
            or getattr(args, "feature_normalization", None) != "final_layernorm"
            or args.seed is None or args.num_workers != 0 or not args.balanced
            or args.overfit or args.resume_from or args.dice_warmup_epochs):
        raise ValueError("Use the dedicated online, balanced, seeded CE layer-ablation config; num_workers=0")
    layers = getattr(args, "ablation_layers", [])
    if (not layers or len(set(layers)) != len(layers)
            or any(type(layer) is not int or not 1 <= layer <= 12 for layer in layers)):
        raise ValueError("ablation_layers must contain unique one-based blocks in [1,12]")
    if (args.accum < 1 or args.batch_size < 1 or args.epochs < 1 or args.grad_clip <= 0
            or not 0 <= args.warmup_epochs < args.epochs
            or not 0 < args.eval_threshold < 1 or args.lr <= 0 or args.weight_decay < 0
            or not args.diagnostics or args.diagnostics_every != 1):
        raise ValueError("Invalid budget, accumulation, clipping or diagnostics settings")


def run(args, publish_results=False, benchmark_warmup=10):
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    folder = Path(args.save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    plots = Path(args.plots_folder)
    plots.mkdir(parents=True, exist_ok=True)
    config = json_record({key: value for key, value in vars(args).items()
                          if key not in ("resume", "benchmark_batches")})
    provenance = dict(
        manifest_sha256=hashlib.sha256((Path(args.dataset_root) / "manifest_split.jsonl").read_bytes()).hexdigest(),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES})
    snapshot = folder / "shared_last.pt"
    if not snapshot.exists() and any(folder.glob("layer_*")):
        raise ValueError("Branch artifacts without shared snapshot; use a new run directory")
    if snapshot.exists() and not args.resume and not args.benchmark_batches:
        raise ValueError("Existing shared checkpoint; pass --resume")
    if args.benchmark_batches and snapshot.exists():
        raise ValueError("Benchmark requires a fresh run; do not perturb an existing experiment")
    backbone, train_loader, val_loader = make_loaders(args, args.device)
    provenance.update(model_id=C.RAD_DINO_MODEL,
                      backbone_revision=getattr(backbone.backbone.config, "_commit_hash", None),
                      backbone_config=backbone.backbone.config.to_dict())
    provenance = json_record(provenance)
    if not len(val_loader) or len(train_loader) < args.accum:
        raise ValueError("Need validation data and at least one complete optimizer update")
    generators = configure_generators(train_loader, val_loader, args.seed)
    spec, branches = build_branches(args, args.device, len(train_loader) // args.accum)
    backbone._validate_layers(args.ablation_layers)
    if args.benchmark_batches:
        if args.benchmark_batches < 100 or benchmark_warmup < args.accum or benchmark_warmup % args.accum:
            raise ValueError("Benchmark requires >=100 measured batches and accumulation-aligned warmup")
        if len(train_loader) < args.benchmark_batches + benchmark_warmup:
            raise ValueError("Insufficient full training batches for requested benchmark")
        save_augmentation_preview(train_loader.dataset, plots / "augmentation_preview.png", args.target_threshold)
        _, _, train_time = run_epoch(branches, backbone, train_loader, args, True,
                                     limit=args.benchmark_batches, warmup=benchmark_warmup)
        memory = dict(allocated=torch.cuda.max_memory_allocated(args.device),
                      reserved=torch.cuda.max_memory_reserved(args.device)) if str(args.device).startswith("cuda") else {}
        _, _, val_time = run_epoch(branches, backbone, val_loader, args, False, limit=4)
        report = dict(config=config, provenance=provenance, train=train_time, validation_sample=val_time,
                      peak_cuda_bytes=memory, validation_sample_batches=4,
                      estimated_epoch_seconds=train_time["seconds_per_shared_batch"] * len(train_loader)
                      + val_time["seconds_per_shared_batch"] * len(val_loader),
                      note="Includes diagnostic computation, excludes checkpoint writes; estimate not guaranteed.")
        write_json(folder / "benchmark.json", report)
        print("[benchmark] " + json.dumps(report), flush=True)
        return
    benchmark_path = folder / "benchmark.json"
    if not benchmark_path.is_file():
        raise ValueError("Run the gated >=100-batch benchmark before training")
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark["config"] != config or benchmark["provenance"] != provenance:
        raise ValueError("Benchmark configuration/code/manifest differs; use a new run directory")
    start = 0
    if snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        if state["config"] != config or state["provenance"] != provenance:
            raise ValueError("Resume configuration/code/manifest mismatch")
        for layer, branch in branches.items():
            saved = state["branches"][layer]
            branch["head"].load_state_dict(saved["head"])
            branch["optimizer"].load_state_dict(saved["optimizer"])
            branch["scheduler"].load_state_dict(saved["scheduler"])
            for key in ("best_score", "best_epoch", "best_head", "best_val", "history"):
                branch[key] = saved[key]
        restore_rng(state["rng"], generators)
        start = state["epoch"]
        export_branches(branches, spec, args)
        del state
    write_json(folder / "run_config.json", config)
    write_json(folder / "provenance.json", dict(
        **provenance, code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        head_spec=spec))
    print(f"[layer ablation] blocks={list(branches)} epochs={args.epochs} "
          f"train_batches={len(train_loader)} val_batches={len(val_loader)} start={start}", flush=True)
    for epoch in range(start + 1, args.epochs + 1):
        train, train_diagnostics, train_time = run_epoch(branches, backbone, train_loader, args, True)
        val, val_diagnostics, val_time = run_epoch(branches, backbone, val_loader, args, False)
        for layer, branch in branches.items():
            val[layer].update(diagnostic_metrics(val_diagnostics[layer], args))
            score = val[layer]["target_signed_dice"]
            row = dict(epoch=epoch, lr=branch["optimizer"].param_groups[0]["lr"],
                       **{f"train_{key}": value for key, value in train[layer].items()},
                       **{f"val_{key}": value for key, value in val[layer].items()})
            branch["history"].append(row)
            if score > branch["best_score"]:
                branch.update(best_score=score, best_epoch=epoch,
                              best_head=cpu_weights(branch["head"]), best_val=val[layer])
            directory = folder / f"layer_{layer:02d}" / "diagnostics"
            metadata = dict(backbone_layer=layer, train_timing=train_time, val_timing=val_time,
                            train_loss=train[layer], val_loss=val[layer])
            train_diagnostics[layer].save(directory, epoch, "train", metadata)
            val_diagnostics[layer].save(directory, epoch, "val", metadata)
            print(f"[epoch {epoch:03d}/{args.epochs} layer {layer}] "
                  f"train_loss={train[layer]['total']:.6f} val_loss={val[layer]['total']:.6f} "
                  f"target_signed_dice={score:.6f} best={branch['best_score']:.6f}@{branch['best_epoch']}",
                  flush=True)
            print_prediction_metrics(val_diagnostics[layer], "val", args.target_threshold)
        state = dict(epoch=epoch, config=config, provenance=provenance, rng=capture_rng(generators),
                     branches={layer: {key: value.state_dict() if key in ("head", "optimizer", "scheduler") else value
                                       for key, value in branch.items()} for layer, branch in branches.items()})
        atomic_save(state, snapshot)
        export_branches(branches, spec, args)
        write_json(folder / "summary.json", {
            str(layer): dict(best_epoch=branch["best_epoch"], best_score=branch["best_score"])
            for layer, branch in branches.items()})
        print(f"[saved] shared epoch {epoch}; "
              f"train={train_time['seconds']/60:.1f}min val={val_time['seconds']/60:.1f}min", flush=True)
        del state, train_diagnostics, val_diagnostics
    if publish_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = ROOT / "results" / f"layer_depth_ablation_{timestamp}"
        destination.mkdir()
        for source in folder.rglob("*"):
            if source.is_file() and source.suffix in (".json", ".csv", ".gz") and not source.name.endswith(".pt"):
                target = destination / source.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        shutil.copy2(args.config, destination / "training_config.yaml")
        if (plots / "augmentation_preview.png").exists():
            shutil.copy2(plots / "augmentation_preview.png", destination / "augmentation_preview.png")
        publish(destination)
    print("[complete] layer ablation; clinical inference is a separate validation-selected step.", flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    options, remaining = parser.parse_known_args()
    args = parse_args(remaining)
    folder = Path(args.save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (folder.parent / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, options.publish, options.benchmark_warmup)


if __name__ == "__main__":
    main()
