"""Shared architecture ablation with subset-sized epochs and fractions before seed repeats."""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from datasets.pair_dataset import LongitudinalPairDataset, _read_manifest, audit_patient_splits
from training import train_shared_architecture_ablation as shared


base = shared.base
ROOT = shared.ROOT
SOURCE = "training/train_subset_architecture_ablation.py"
BUDGET_POLICY = "selected_pairs_ceil_effective_batch"
GROUP_ORDER = "all_fractions_for_each_seed"


def configure(args, run_root):
    args = copy.copy(args)
    if Path(run_root).resolve() == Path(args.run_root).resolve():
        raise ValueError("Use a new subset-epoch run root; preserve the original run")
    args.run_root = str(Path(run_root).resolve())
    args.epoch_budget = BUDGET_POLICY
    args.group_order = GROUP_ORDER
    base.validate_config(args)
    return args


def group_order(args):
    return [(fraction, seed) for seed in args.seeds for fraction in args.data_fractions]


def group_setup(args, context, fraction, seed):
    if fraction not in args.data_fractions or seed not in args.seeds:
        raise ValueError("Subset group is not in the configured fraction/seed matrix")
    records = context["datasets"]["train"].records
    indices = base.patient_subset_indices(records, fraction, seed)
    if not indices:
        raise ValueError("Cannot train an empty patient subset")
    effective_batch = args.batch_size * args.accum
    updates = (len(indices) + effective_batch - 1) // effective_batch
    budget = dict(selected_pairs=len(indices), effective_batch=effective_batch,
                  samples_per_epoch=updates * effective_batch, updates_per_epoch=updates,
                  total_updates=args.epochs * updates,
                  warmup_updates=int(args.warmup_epochs * updates), epochs=args.epochs)
    group_args = copy.copy(args)
    group_args.seed = seed
    group_args.subset_budget = budget
    group_context = dict(context, updates_per_slot=updates)
    return group_args, group_context, budget


def add_provenance(context):
    return dict(context, provenance=dict(
        context["provenance"], subset_code_sha256={SOURCE: base.sha256(ROOT / SOURCE)}))


def prepare(args):
    context, basis = shared.prepare(args)
    context = add_provenance(context)
    count = len(context["datasets"]["train"])
    effective_batch = args.batch_size * args.accum
    context["updates_per_slot"] = (count + effective_batch - 1) // effective_batch
    return context, basis


def schedule(args, context):
    rows = []
    for fraction, seed in group_order(args):
        group_args, group_context, budget = group_setup(args, context, fraction, seed)
        identity, _ = shared.group_identity(group_args, group_context, fraction, seed)
        rows.append(dict(patient_fraction=fraction, seed=seed, **budget,
                         selected_patients=identity["selection"]["inventory"]["patients"],
                         actual_pair_fraction=identity["selection"]["actual_pair_fraction"]))
    return rows


def checkpoint_identity(args, context, fraction, seed):
    group_args, group_context, _ = group_setup(args, context, fraction, seed)
    return shared.group_identity(group_args, group_context, fraction, seed)[0]


def run_group(args, context, basis, fraction, seed, resume=False):
    group_args, group_context, budget = group_setup(args, context, fraction, seed)
    print(f"[subset epochs] fraction={fraction:g} seed={seed} "
          f"pairs={budget['selected_pairs']} draws/epoch={budget['samples_per_epoch']} "
          f"updates/epoch/head={budget['updates_per_epoch']} "
          f"total/head={budget['total_updates']}", flush=True)
    return shared.run_group(group_args, group_context, basis, fraction, seed, resume)


def require_benchmark(args, context):
    shared.require_benchmark(args, context)


def run_sweep(args, context, basis, resume=False):
    require_benchmark(args, context)
    root = Path(args.run_root)
    identity = base.sweep_identity(args, context)
    path = root / "sweep_subset_epochs.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing subset-epoch sweep differs or --resume is missing")
    else:
        if (root / "sweep_shared.json").exists() or any(root.glob("fraction_*")):
            raise ValueError("Old/group artifacts in a new subset-epoch run root")
        base.write_json(path, identity)
    base.write_json(root / "subset_schedule.json", dict(
        epoch_budget=BUDGET_POLICY, group_order=GROUP_ORDER, groups=schedule(args, context)))
    context["backbone"].to("cpu")
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    results = []
    for fraction, seed in group_order(args):
        group = checkpoint_identity(args, context, fraction, seed)
        folder = root / shared.group_name(fraction, seed)
        if not (folder / "complete.json").exists():
            logs = root / "logs"
            logs.mkdir(exist_ok=True)
            log = logs / f"{folder.name}_{time.time_ns()}.log"
            command = [
                sys.executable, "-u", "-m", "training.train_subset_architecture_ablation",
                "--config", str(Path(args.config).resolve()), "--run-root", str(root.resolve()),
                "--mode", "group", "--fraction", str(fraction), "--seed", str(seed)]
            if resume:
                command.append("--resume")
            print(f"[subset sweep] {folder.name}; log={log}", flush=True)
            with log.open("x", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        results.extend(shared.verify_group(folder, group))
    base.aggregate_results(root, results)
    print(f"[subset sweep] verified {len(results)} independent trained heads", flush=True)


def report_context(args):
    """Validate identities without constructing a backbone or loading any images."""
    path = Path(args.run_root) / "benchmark_shared.json"
    if not path.is_file():
        raise ValueError("Subset-epoch benchmark is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    provenance = report["identity"]["provenance"]
    hashes = {**provenance["code_sha256"], **provenance["shared_code_sha256"],
              **provenance.get("subset_code_sha256", {})}
    if provenance.get("subset_code_sha256") != {SOURCE: base.sha256(ROOT / SOURCE)}:
        raise ValueError("Report requires the matching subset-epoch benchmark, not an old run")
    for name, digest in hashes.items():
        if base.sha256(ROOT / name) != digest:
            raise ValueError(f"Source changed since the subset benchmark: {name}")
    manifest = Path(args.dataset_root) / "manifest_split.jsonl"
    if base.sha256(manifest) != provenance["features"]["manifest_sha256"]:
        raise ValueError("Manifest changed since the subset benchmark")
    if base.sha256(args.diagnostic_file) != args.diagnostic_sha256:
        raise ValueError("Diagnostic basis changed since the subset benchmark")
    inventory = audit_patient_splits(_read_manifest(args.dataset_root))
    dataset = LongitudinalPairDataset(args.dataset_root, split="train",
                                      rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
    effective_batch = args.batch_size * args.accum
    context = dict(provenance=provenance, inventory=inventory, datasets={"train": dataset},
                   updates_per_slot=(len(dataset) + effective_batch - 1) // effective_batch)
    require_benchmark(args, context)
    return context, report


def export_status(args):
    context, benchmark = report_context(args)
    progress, completed = [], 0
    for fraction, seed in group_order(args):
        group_args, group_context, budget = group_setup(args, context, fraction, seed)
        folder = Path(args.run_root) / shared.group_name(fraction, seed)
        snapshot = folder / "shared_last.pt"
        if not snapshot.is_file():
            continue
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        expected = shared.group_identity(group_args, group_context, fraction, seed)[0]
        if (state["identity"] != expected or not 1 <= state["epoch"] <= args.epochs
                or set(state["branches"]) != set(base.ARMS)):
            raise ValueError("Progress checkpoint differs from its subset/seed/update budget")
        finished = (folder / "complete.json").exists()
        if finished:
            shared.verify_group(folder, expected)
            completed += len(base.ARMS)
        for arm, branch in state["branches"].items():
            if (len(branch["history"]) != state["epoch"]
                    or branch["scheduler"]["last_epoch"] != state["epoch"] * budget["updates_per_epoch"]
                    or not 1 <= branch["best_slot"] <= state["epoch"]):
                raise ValueError(f"Invalid saved subset epoch/update position: {arm}")
            latest = branch["history"][-1]
            best = max(branch["history"], key=lambda row: row["val"]["target_signed_dice"])
            if branch["best_slot"] != best["slot"] or branch["best_score"] != best["val"]["target_signed_dice"]:
                raise ValueError(f"Saved best checkpoint selection disagrees with history: {arm}")
            progress.append(dict(
                arm=arm, patient_fraction=fraction, seed=seed, epoch=state["epoch"],
                budget=budget, optimizer_updates=state["epoch"] * budget["updates_per_epoch"],
                best_epoch=branch["best_slot"], best_val_signed_dice=branch["best_score"],
                best_val=best["val"], latest_train=latest["train"], latest_val=latest["val"],
                shared_epoch_timing=state["shared_timing"], completed=finished))
    configured = len(args.data_fractions) * len(args.seeds) * len(base.ARMS)
    report = dict(
        status="complete" if completed == configured else "progress" if progress else "benchmark",
        completed_models=completed, configured_models=configured, epoch_budget=BUDGET_POLICY,
        group_order=GROUP_ORDER, schedule=schedule(args, context), progress=progress,
        selected_rank=args.debias_rank, diagnostic_sha256=args.diagnostic_sha256,
        manifest_sha256=context["provenance"]["features"]["manifest_sha256"],
        backbone_layer=args.backbone_layer, provenance={
            key: context["provenance"][key] for key in (
                "runtime", "code_sha256", "shared_code_sha256", "subset_code_sha256")},
        benchmark={key: benchmark[key] for key in (
            "training", "validation", "shared_train_timing", "shared_val_timing", "peak_cuda_bytes")},
        note="Balanced replacement draws approximate the selected pair count each epoch, rounded up "
             "by fewer than one effective batch. Not one visit to every selected pair. "
             "Thirty epochs and two warmup epochs per subset; equal updates only within B0/B1/B2. "
             "Fresh heads/optimizers, no continuation of equal-update experiments. "
             "Numeric metrics only; no patient/pair IDs, paths, images, basis or weights.")
    if completed == configured:
        report["aggregate"] = json.loads(
            (Path(args.run_root) / "summary.json").read_text(encoding="utf-8"))
    destination = ROOT / "results" / f"architecture_data_subset_epochs_{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    base.write_json(destination / "summary.json", report)
    if (destination / "summary.json").stat().st_size > 1_000_000:
        raise ValueError("Subset-epoch numeric report exceeds 1 MB")
    print(f"[subset export] {destination / 'summary.json'}", flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing rank/hash-locked YAML, unchanged")
    parser.add_argument("--run-root", required=True, help="Fresh subset-epoch run, never an old run root")
    parser.add_argument("--mode", choices=("benchmark", "sweep", "group", "report"), default="sweep")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args(argv)
    args = configure(base.parse_args(["--config", str(Path(options.config).resolve())]), options.run_root)
    if options.mode == "report":
        destination = export_status(args)
    else:
        if options.mode == "group":
            if options.fraction is None or options.seed is None:
                parser.error("group mode requires --fraction and --seed")
            lock = Path(args.run_root) / f".{shared.group_name(options.fraction, options.seed)}.lock"
        else:
            lock = Path(args.run_root) / ".subset_epochs.lock"
        with base.run_lock(lock):
            context, basis = prepare(args)
            if options.mode == "benchmark":
                if (Path(args.run_root) / "sweep_subset_epochs.json").exists():
                    raise ValueError("Cannot benchmark an existing subset-epoch sweep")
                shared.benchmark(args, context, basis)
            elif options.mode == "group":
                require_benchmark(args, context)
                run_group(args, context, basis, options.fraction, options.seed, options.resume)
                return
            else:
                run_sweep(args, context, basis, options.resume)
            destination = export_status(args)
    if options.publish:
        shared.publish_report(destination)


if __name__ == "__main__":
    main()
