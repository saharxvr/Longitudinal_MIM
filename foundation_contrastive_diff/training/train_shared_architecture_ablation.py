"""Train B0/B1/B2 together using one frozen encoder and identical sampled inputs."""

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from training import train_architecture_data_ablation as base


ROOT = base.ROOT
SOURCE = "training/train_shared_architecture_ablation.py"
MIN_BENCHMARK_BATCHES = 100


def prepare(args):
    context = base.prepare(args)
    basis = base.locked_basis(args, context)
    context["provenance"] = dict(
        context["provenance"], shared_code_sha256={SOURCE: base.sha256(ROOT / SOURCE)})
    return context, basis


def seed_training(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def build_branches(args, basis, updates):
    branches = {}
    for arm in base.ARMS:
        spec, head = base.make_cell_head(args, arm, basis)
        if any(isinstance(module, torch.nn.Dropout) and module.p
               for module in head.modules()):
            raise ValueError("Shared ablation requires deterministic heads without dropout")
        optimizer, scheduler = base.optimizer_state(args, head, updates)
        branches[arm] = dict(head=head, spec=spec, optimizer=optimizer, scheduler=scheduler,
                             best_score=-1., best_slot=0, best_head=None, history=[])
    return branches


def run_pass(branches, backbone, loader, args, training=False, warmup_updates=0):
    if backbone.training or any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("Shared backbone must remain frozen and in eval mode")
    diagnostics = {arm: base.make_diagnostics(args) for arm in branches}
    totals = {arm: {} for arm in branches}
    counts = {arm: {prefix: {sign: [0, 0, 0] for sign in (
        "positive", "negative", "support", "signed")} for prefix in ("target", "raw")}
        for arm in branches}
    areas = {arm: [0, 0, 0] for arm in branches}
    updates = dict.fromkeys(branches, 0)
    for branch in branches.values():
        branch["head"].train(training)
        if training:
            branch["optimizer"].zero_grad(set_to_none=True)
    batches, draws = 0, 0
    base.synchronize(args.device)
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if training and index == warmup_updates * args.accum:
            base.synchronize(args.device)
            started = time.perf_counter()
            if str(args.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(args.device)
        tokens = base._tokens_from_batch(
            batch, backbone, args.device, augment=training and args.photometric_augmentation)
        ground_truth = batch["gt_diff"].to(args.device)
        for arm, branch in branches.items():
            with torch.set_grad_enabled(training):
                output, embedding = branch["head"](*tokens)
                loss, components = base.objective(output, ground_truth, args, args.w_dice)
                if not torch.isfinite(output).all() or not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {arm} output/loss at shared batch {index + 1}")
                prediction = base.signed_change_map(output).detach()
                diagnostics[arm].update(prediction, ground_truth, batch)
                for prefix, cutoff in (("target", args.target_threshold), ("raw", args.eval_gt_threshold)):
                    values = base.active_overlap_counts(
                        prediction, ground_truth, batch["is_pathology"], args.eval_threshold, cutoff)
                    for sign, value in values.items():
                        counts[arm][prefix][sign] = [a + b for a, b in zip(
                            counts[arm][prefix][sign], value)]
                areas[arm][0] += int((prediction > args.eval_threshold).sum())
                areas[arm][1] += int((prediction < -args.eval_threshold).sum())
                areas[arm][2] += prediction.numel()
                if training:
                    (loss / args.accum).backward()
                    if (index + 1) % args.accum == 0:
                        norm = torch.nn.utils.clip_grad_norm_(
                            branch["head"].parameters(), args.grad_clip)
                        if not torch.isfinite(norm):
                            raise ValueError(f"Nonfinite {arm} gradient; shared group stopped")
                        branch["optimizer"].step()
                        branch["optimizer"].zero_grad(set_to_none=True)
                        branch["scheduler"].step()
                        updates[arm] += 1
                for key, value in components.items():
                    totals[arm][key] = totals[arm].get(key, 0) + value
                del output, embedding, loss, prediction
        draws += len(batch["pair_id"])
        batches += 1
        if training and (batches == 1 or batches % 100 == 0):
            print(f"[shared batch {batches}] " + " ".join(
                f"{arm} loss={totals[arm]['total']/batches:.5f}" for arm in branches), flush=True)
        del tokens
    measured = batches - warmup_updates * args.accum
    if measured <= 0 or (training and batches % args.accum):
        raise ValueError("Shared pass requires measured batches and complete accumulation windows")
    base.synchronize(args.device)
    timing = dict(seconds=time.perf_counter() - started, measured_batches=measured,
                  sample_draws=draws, optimizer_updates_per_head=updates)
    metrics = {}
    for arm in branches:
        metrics[arm] = dict(
            **{f"loss_{key}": value / batches for key, value in totals[arm].items()},
            **base.diagnostic_metrics(diagnostics[arm], args),
            optimizer_updates=updates[arm], sample_draws=draws,
            predicted_positive_area=areas[arm][0] / areas[arm][2],
            predicted_negative_area=areas[arm][1] / areas[arm][2])
        for prefix, signs in counts[arm].items():
            for sign, (intersection, predicted, reference) in signs.items():
                metrics[arm][f"{prefix}_{sign}_dice"] = (
                    2 * intersection / (predicted + reference) if reference else None)
    return metrics, diagnostics, timing


def group_name(fraction, seed):
    return f"fraction_{fraction:g}_seed_{seed}"


def group_identity(args, context, fraction, seed):
    return base.cell_identity(args, context, "shared_B0_B1_B2", fraction, seed)


def checkpoint(branch, args, slot, kind):
    best = kind == "best"
    return dict(head=branch["best_head"] if best else base.cpu_weights(branch["head"]),
                head_spec=branch["spec"], run_config=dict(vars(args), head=branch["spec"]["type"]),
                epoch=branch["best_slot"] if best else slot, best_score=branch["best_score"],
                best_metric="target_signed_dice")


def export_branches(folder, branches, args, slot):
    for arm, branch in branches.items():
        directory = folder / arm
        directory.mkdir(exist_ok=True)
        for kind in ("best", "last"):
            payload = checkpoint(branch, args, slot, kind)
            payload["run_config"]["arm"] = arm
            payload["optimizer_updates"] = payload["epoch"] * branch["history"][0]["train"]["optimizer_updates"]
            base.atomic_save(payload, directory / f"{kind}.pt")
        rows = [dict(slot=row["slot"], lr=row["lr"],
                     **{f"{split}_{key}": value for split in ("train", "val")
                        for key, value in row[split].items()}) for row in branch["history"]]
        base.write_csv(directory / "metrics.csv", rows)


def verify_group(folder, identity):
    completion = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    expected_updates = identity["config"]["epochs"] * identity["updates_per_slot"]
    required = {"shared_last.pt", "results.json"}
    required.update(f"{arm}/{name}" for arm in base.ARMS for name in ("best.pt", "last.pt", "metrics.csv"))
    if (completion["identity_sha256"] != base.json_hash(identity)
            or completion["optimizer_updates_per_head"] != dict.fromkeys(base.ARMS, expected_updates)
            or set(completion["files"]) != required):
        raise ValueError("Completed shared group identity/update budget differs")
    for name, digest in completion["files"].items():
        if base.sha256(folder / name) != digest:
            raise ValueError(f"Completed shared artifact changed: {name}")
    return json.loads((folder / "results.json").read_text(encoding="utf-8"))


def run_group(args, context, basis, fraction, seed, resume=False):
    if fraction not in args.data_fractions or seed not in args.seeds:
        raise ValueError("Shared group is not in the configured matrix")
    identity, indices = group_identity(args, context, fraction, seed)
    folder = Path(args.run_root) / group_name(fraction, seed)
    if (folder / "complete.json").exists():
        if not resume:
            raise ValueError("Shared group already completed; use --resume")
        return verify_group(folder, identity)
    snapshot = folder / "shared_last.pt"
    if folder.exists() and any(folder.iterdir()) and (not resume or not snapshot.exists()):
        raise ValueError("Existing shared artifacts without compatible resume checkpoint")
    folder.mkdir(parents=True, exist_ok=True)
    args = copy.copy(args)
    args.seed = seed
    seed_training(seed, args.device)
    train, val, test, generators = base.make_cell_loaders(args, context, indices)
    branches = build_branches(args, basis, context["updates_per_slot"])
    start = 0
    if snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=False)
        start = state["epoch"]
        if (state["identity"] != identity or not 1 <= start <= args.epochs
                or set(state["branches"]) != set(branches)):
            raise ValueError("Shared resume identity/slot/arms mismatch")
        for arm, branch in branches.items():
            saved = state["branches"][arm]
            if (saved["spec"] != branch["spec"] or len(saved["history"]) != start
                    or saved["scheduler"]["last_epoch"] != start * context["updates_per_slot"]
                    or not 1 <= saved["best_slot"] <= start):
                raise ValueError(f"Shared resume checkpoint budget/spec differs: {arm}")
            for key in ("head", "optimizer", "scheduler"):
                branch[key].load_state_dict(saved[key])
            for key in ("best_score", "best_slot", "best_head", "history"):
                branch[key] = saved[key]
        base.restore_rng(state["rng"], generators)
    base.write_json(folder / "selection.json", identity["selection"])
    base.write_json(folder / "provenance.json", identity)
    for slot in range(start + 1, args.epochs + 1):
        training, train_diag, train_time = run_pass(
            branches, context["backbone"], train, args, training=True)
        validation, val_diag, val_time = run_pass(branches, context["backbone"], val, args)
        for arm, branch in branches.items():
            score = validation[arm]["target_signed_dice"]
            if (training[arm]["optimizer_updates"] != context["updates_per_slot"]
                    or branch["scheduler"].last_epoch != slot * context["updates_per_slot"]
                    or score is None):
                raise ValueError(f"Invalid successful-update budget or validation score: {arm}")
            if score > branch["best_score"]:
                branch.update(best_score=score, best_slot=slot, best_head=base.cpu_weights(branch["head"]))
            branch["history"].append(dict(
                slot=slot, lr=branch["optimizer"].param_groups[0]["lr"],
                train=training[arm], val=validation[arm]))
        state = dict(
            epoch=slot, identity=identity, rng=base.capture_rng(generators),
            shared_timing=dict(train=train_time, val=val_time),
            branches={arm: {
                **{key: branch[key] for key in ("spec", "best_score", "best_slot", "best_head", "history")},
                "head": base.cpu_weights(branch["head"]),
                "optimizer": branch["optimizer"].state_dict(),
                "scheduler": branch["scheduler"].state_dict()} for arm, branch in branches.items()})
        base.atomic_save(state, snapshot)
        export_branches(folder, branches, args, slot)
        for arm in branches:
            train_diag[arm].save(folder / arm / "diagnostics", slot, "train")
            val_diag[arm].save(folder / arm / "diagnostics", slot, "val")
        print(f"[shared {folder.name}] slot {slot}/{args.epochs} "
              f"updates/head={slot * context['updates_per_slot']} " + " ".join(
                  f"{arm} val={validation[arm]['target_signed_dice']:.6f}" for arm in branches), flush=True)
    for branch in branches.values():
        branch["head"].load_state_dict(branch["best_head"])
    test_metrics, test_diag, _ = run_pass(branches, context["backbone"], test, args)
    results = []
    for arm, branch in branches.items():
        if test_metrics[arm]["target_signed_dice"] is None:
            raise ValueError(f"No active signed test target: {arm}")
        test_diag[arm].save(folder / arm / "diagnostics", branch["best_slot"], "test")
        results.append(dict(
            arm=arm, fraction=fraction, seed=seed, inventory=identity["selection"]["inventory"],
            actual_pair_fraction=identity["selection"]["actual_pair_fraction"],
            best_slot=branch["best_slot"], best_val_signed_dice=branch["best_score"], test=test_metrics[arm]))
    base.write_json(folder / "results.json", results)
    names = ["shared_last.pt", "results.json"]
    names.extend(f"{arm}/{name}" for arm in branches for name in ("best.pt", "last.pt", "metrics.csv"))
    base.write_json(folder / "complete.json", dict(
        identity_sha256=base.json_hash(identity),
        optimizer_updates_per_head=dict.fromkeys(branches, args.epochs * context["updates_per_slot"]),
        files={name: base.sha256(folder / name) for name in names}))
    return results


def benchmark(args, context, basis):
    root = Path(args.run_root)
    if (root / "benchmark_shared.json").exists() or (root / "sweep_shared.json").exists():
        raise ValueError("All-head benchmark requires a fresh shared run root")
    identity = base.sweep_identity(args, context)
    args = copy.copy(args)
    args.seed = args.seeds[0]
    seed_training(args.seed, args.device)
    _, indices = group_identity(args, context, 1., args.seed)
    measured_updates, warmup = math.ceil(MIN_BENCHMARK_BATCHES / args.accum), 5
    train, val, _, _ = base.make_cell_loaders(args, context, indices, measured_updates + warmup)
    branches = build_branches(args, basis, context["updates_per_slot"])
    training, _, timing = run_pass(
        branches, context["backbone"], train, args, training=True, warmup_updates=warmup)
    validation, _, val_time = run_pass(branches, context["backbone"], val, args)
    if any(values["target_signed_dice"] is None for values in validation.values()):
        raise ValueError("All-head validation needs active signed targets")
    memory = dict(allocated=torch.cuda.max_memory_allocated(args.device),
                  reserved=torch.cuda.max_memory_reserved(args.device)) if str(args.device).startswith("cuda") else None
    base.write_json(root / "benchmark_shared.json", dict(
        identity=identity, training=training, validation=validation,
        shared_train_timing=timing, shared_val_timing=val_time, peak_cuda_bytes=memory,
        note="All three independent heads share augmented inputs and frozen features. "
             "Head graphs are backpropagated sequentially; no model checkpoint saved."))
    print(f"[shared benchmark] passed {timing['measured_batches']} measured batches for all arms", flush=True)


def require_benchmark(args, context):
    path = Path(args.run_root) / "benchmark_shared.json"
    if not path.is_file():
        raise ValueError("Run the new all-head --mode benchmark before shared training")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report["identity"] != base.sweep_identity(args, context)
            or report["shared_train_timing"]["measured_batches"] < MIN_BENCHMARK_BATCHES
            or set(report["training"]) != set(base.ARMS)):
        raise ValueError("Shared benchmark config/code/data/budget differs")


def run_sweep(args, context, basis, resume=False):
    require_benchmark(args, context)
    root = Path(args.run_root)
    identity = base.sweep_identity(args, context)
    path = root / "sweep_shared.json"
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing shared sweep differs or --resume is missing")
    else:
        if any(root.glob("fraction_*")):
            raise ValueError("Group artifacts without a shared sweep manifest")
        base.write_json(path, identity)
    context["backbone"].to("cpu")
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    results = []
    for fraction in args.data_fractions:
        for seed in args.seeds:
            group, _ = group_identity(args, context, fraction, seed)
            folder = root / group_name(fraction, seed)
            if not (folder / "complete.json").exists():
                logs = root / "logs"
                logs.mkdir(exist_ok=True)
                log = logs / f"{folder.name}_{time.time_ns()}.log"
                command = [
                    sys.executable, "-u", "-m", "training.train_shared_architecture_ablation",
                    "--config", str(Path(args.config).resolve()), "--run-root", str(root.resolve()),
                    "--mode", "group", "--fraction", str(fraction), "--seed", str(seed)]
                if resume:
                    command.append("--resume")
                print(f"[shared sweep] {folder.name}; log={log}", flush=True)
                with log.open("x", encoding="utf-8") as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            results.extend(verify_group(folder, group))
    base.aggregate_results(root, results)
    print(f"[shared sweep] verified {len(results)} independent trained heads", flush=True)


def export_status(args):
    root = Path(args.run_root)
    benchmark_report = json.loads((root / "benchmark_shared.json").read_text(encoding="utf-8"))
    expected = base.sha256(ROOT / SOURCE)
    if benchmark_report["identity"]["provenance"]["shared_code_sha256"] != {SOURCE: expected}:
        raise ValueError("Shared source changed since the benchmark")
    config = {key: value for key, value in vars(args).items()
              if key not in ("config", "resume", "benchmark_batches")}
    if benchmark_report["identity"]["config"] != config:
        raise ValueError("Report config differs from the shared benchmark")
    progress, completed = [], 0
    for fraction in args.data_fractions:
        for seed in args.seeds:
            folder = root / group_name(fraction, seed)
            if not (folder / "shared_last.pt").exists():
                continue
            state = torch.load(folder / "shared_last.pt", map_location="cpu", weights_only=False)
            if (state["identity"]["config"] != config
                    or state["identity"]["provenance"] != benchmark_report["identity"]["provenance"]):
                raise ValueError("Progress checkpoint differs from the shared benchmark")
            finished = (folder / "complete.json").exists()
            if finished:
                verify_group(folder, state["identity"])
                completed += 3
            for arm, branch in state["branches"].items():
                latest = branch["history"][-1]
                progress.append(dict(
                    arm=arm, patient_fraction=fraction, seed=seed, slot=state["epoch"],
                    best_slot=branch["best_slot"], best_val_signed_dice=branch["best_score"],
                    latest_train=latest["train"], latest_val=latest["val"], completed=finished))
    provenance = benchmark_report["identity"]["provenance"]
    features = provenance["features"]
    report = dict(
        status="complete" if completed == len(args.data_fractions) * len(args.seeds) * 3 else (
            "progress" if progress else "benchmark"),
        completed_models=completed, configured_models=len(args.data_fractions) * len(args.seeds) * 3,
        selected_rank=args.debias_rank, diagnostic_sha256=args.diagnostic_sha256,
        manifest_sha256=features["manifest_sha256"], backbone_layer=features["backbone_layer"],
        runtime=provenance["runtime"], code_sha256=provenance["code_sha256"],
        shared_code_sha256=provenance["shared_code_sha256"],
        benchmark={key: benchmark_report[key] for key in (
            "training", "validation", "shared_train_timing", "shared_val_timing", "peak_cuda_bytes")},
        progress=progress,
        note="Metrics only: no patient/pair IDs, paths, scans, annotations, basis or weights. "
             "Early all-unchanged predictions are visible in signed Dice and predicted-area metrics; "
             "finite training does not prove learning or clinical benefit.")
    if report["status"] == "complete":
        report["aggregate"] = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    destination = ROOT / "results" / f"architecture_data_shared_{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    base.write_json(destination / "summary.json", report)
    if (destination / "summary.json").stat().st_size > 1_000_000:
        raise ValueError("Shared compact export exceeds 1 MB")
    print(f"[shared export] {destination / 'summary.json'}", flush=True)
    return destination


def publish_report(destination):
    repository = ROOT.parent
    destination.resolve().relative_to((ROOT / "results").resolve())
    report = destination / "summary.json"
    if report.is_symlink() or not report.is_file() or report.stat().st_size > 1_000_000:
        raise ValueError("Publication requires a regular compact summary")
    path = (destination / "summary.json").resolve().relative_to(repository.resolve()).as_posix()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository, text=True).strip()
    if branch != "main":
        raise ValueError("Scoped shared-result publication expects main")
    subprocess.run(["git", "add", "-f", "--", path], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", f"Publish {destination.name}", "-m",
                    "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
                    "--", path], cwd=repository, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repository, check=True)
    print("[shared publish] pushed one compact summary; no pull or training-source update", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing rank/hash-locked YAML")
    parser.add_argument("--run-root", required=True, help="New shared-run directory, not the sequential root")
    parser.add_argument("--mode", choices=("benchmark", "sweep", "group", "report"), default="sweep")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish", action="store_true")
    options = parser.parse_args(argv)
    args = base.parse_args(["--config", str(Path(options.config).resolve())])
    if Path(options.run_root).resolve() == Path(args.run_root).resolve():
        raise ValueError("Preserve the sequential run: use a different shared --run-root")
    args.run_root = str(Path(options.run_root).resolve())
    base.validate_config(args)
    if options.mode == "report":
        destination = export_status(args)
    else:
        if options.mode == "group":
            if options.fraction is None or options.seed is None:
                parser.error("group mode requires --fraction and --seed")
            lock = Path(args.run_root) / f".{group_name(options.fraction, options.seed)}.lock"
        else:
            lock = Path(args.run_root) / ".shared.lock"
        with base.run_lock(lock):
            context, basis = prepare(args)
            if options.mode == "benchmark":
                benchmark(args, context, basis)
            elif options.mode == "group":
                require_benchmark(args, context)
                run_group(args, context, basis, options.fraction, options.seed, options.resume)
                return
            else:
                run_sweep(args, context, basis, options.resume)
            destination = export_status(args)
    if options.publish:
        publish_report(destination)


if __name__ == "__main__":
    main()
