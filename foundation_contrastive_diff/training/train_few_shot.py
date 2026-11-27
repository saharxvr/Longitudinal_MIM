"""Train few-shot change heads (FSDiff + comparators) on frozen RAD-DINO 6/9/12 tokens.

Every optimizer update consumes the same episode for all arms: one frozen extraction of
the support+query images, then each arm's head/loss/optimizer independently (sequential
backward to bound memory). Validation uses the fixed synthetic pathology selection
manifest (held-out tasks); ICU is never used for selection.

Usage (campus):
    python -u -m training.train_few_shot --config configs/few_shot_6_9_12.yaml --benchmark
    python -u -m training.train_few_shot --config configs/few_shot_6_9_12.yaml [--resume]
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from datasets.augmentations import augment_cxr
from datasets.episodes import EpisodeConfig, EpisodeDataset, SceneCorpus, materialize
from evaluation import few_shot_eval, few_shot_manifests
from losses.few_shot_losses import masked_three_class_loss, soft_grid_loss
from models.few_shot_head import MATCHING_TYPES, FewShotChangeHead, count_parameters, grid_labels

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("training/train_few_shot.py", "models/few_shot_head.py", "datasets/episodes.py",
           "losses/few_shot_losses.py", "evaluation/few_shot_eval.py", "evaluation/few_shot_manifests.py")
DEFAULTS = dict(
    arms=["fsdiff", "dcama", "panet", "alpnet", "universeg", "none"], par_arms=["panet"],
    feature_layers=[6, 9, 12], model_id="microsoft/rad-dino", backbone_dim=768, grid=37,
    channels=96, num_heads=4, attention_depth=2, memory_layers=2, task_tokens=4, local_pool=4,
    local_threshold=0.3, token_budget=0, crossblocks=2,
    corpus_tiers=["T1", "T2"], corpus_store=None, max_support=32, queries=4,
    level_weights=[0.45, 0.25, 0.15, 0.15], empty_query_prob=0.15, distractor_query_prob=0.5,
    zero_support_prob=0.05, label_mode="tight", coarse_prob=0.3,
    episodes_per_epoch=2000, epochs=30, accum=1, lr=3e-4, weight_decay=0.01, warmup_epochs=2.0,
    grad_clip=1.0, w_ce=1.0, w_dice=0.0, w_change=0.1, w_par=1.0, target_threshold=0.001,
    eval_threshold=0.5, photometric_augmentation=True, rotation_k=3, flip_horizontal=True,
    precision="bf16", extract_chunk=32, num_workers=4, seed=42, device="cuda",
    val_dataset_root=None, val_store=None, val_cache_pairs=1200, val_shuffle_check=True,
    benchmark_support=[64, 32, 16], benchmark_episodes=12, benchmark_memory_gb=21.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path=None, **overrides):
    values = dict(DEFAULTS)
    if path:
        values.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    values.update({key: value for key, value in overrides.items() if value is not None})
    cfg = SimpleNamespace(**values)
    validate(cfg)
    return cfg


def validate(cfg):
    for key in ("corpus_root", "run_root"):
        if not getattr(cfg, key, None):
            raise ValueError(f"Config requires {key}")
    if not cfg.arms or any(arm not in MATCHING_TYPES for arm in cfg.arms) or len(set(cfg.arms)) != len(cfg.arms):
        raise ValueError(f"Arms must be unique members of {MATCHING_TYPES}")
    if cfg.precision not in ("fp32", "bf16") or cfg.accum < 1 or cfg.epochs < 1:
        raise ValueError("Invalid precision/accumulation/epochs")
    if cfg.label_mode not in ("tight", "coarse_support", "coarse_both"):
        raise ValueError("Invalid label_mode")


def head_spec(cfg, arm):
    return dict(type="few_shot", kwargs=dict(
        matching=arm, backbone_dim=cfg.backbone_dim, feature_layers=list(cfg.feature_layers), grid=cfg.grid,
        channels=cfg.channels, num_heads=cfg.num_heads, attention_depth=cfg.attention_depth,
        memory_layers=cfg.memory_layers, task_tokens=cfg.task_tokens, local_pool=cfg.local_pool,
        local_threshold=cfg.local_threshold, token_budget=cfg.token_budget, crossblocks=cfg.crossblocks))


def cosine_warmup(optimizer, warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_backbone(cfg):
    from models import FrozenCXRBackbone

    backbone = FrozenCXRBackbone(model_id=cfg.model_id, selected_layers=list(cfg.feature_layers), freeze=True)
    return backbone.to(cfg.device).eval()


def build_branches(cfg, updates_per_epoch):
    branches = {}
    total = updates_per_epoch * cfg.epochs
    for arm in cfg.arms:
        torch.manual_seed(cfg.seed)
        spec = head_spec(cfg, arm)
        head = FewShotChangeHead(**spec["kwargs"]).to(cfg.device)
        optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = cosine_warmup(optimizer, int(cfg.warmup_epochs * updates_per_epoch), total)
        branches[arm] = dict(head=head, spec=spec, optimizer=optimizer, scheduler=scheduler,
                             best_score=-1.0, best_epoch=0, history=[], parameters=count_parameters(head))
    return branches


def episode_config(cfg, **overrides):
    values = dict(max_support=cfg.max_support, queries=cfg.queries, level_weights=tuple(cfg.level_weights),
                  empty_query_prob=cfg.empty_query_prob, distractor_query_prob=cfg.distractor_query_prob,
                  zero_support_prob=cfg.zero_support_prob, label_mode=cfg.label_mode,
                  coarse_prob=cfg.coarse_prob, target_threshold=cfg.target_threshold,
                  rotation_k=cfg.rotation_k, flip_horizontal=cfg.flip_horizontal)
    values.update(overrides)
    return EpisodeConfig(**values)


def episode_tokens(batch, extractor, augment):
    """One frozen extraction for supports and queries; photometric augmentation per image."""
    n, queries = batch["n"], batch["query_prior"].shape[0]
    images = torch.cat([batch["support_prior"], batch["query_prior"], batch["support_curr"], batch["query_curr"]])
    if augment:
        images = augment_cxr(images.clamp(0, 1))
    patches, classes = extractor.images(images)
    half = n + queries
    prior, current = patches[:half], patches[half:]
    cls_prior, cls_current = classes[:half], classes[half:]
    query = (prior[n:], current[n:], cls_prior[n:], cls_current[n:])
    support = (prior[:n], current[:n]) if n else None
    return query, support


def train_step(branches, batch, query, support, cfg, training=True, update=True):
    device = cfg.device
    labels = (grid_labels(batch["support_gt"], batch["support_ignore"], cfg.target_threshold, cfg.grid).to(device)
              if support is not None else None)
    query_gt, query_ignore = batch["query_gt"].to(device), batch["query_ignore"].to(device)
    all_labels = None
    if cfg.w_change:
        all_labels = grid_labels(batch["query_all_gt"], batch["query_all_ignore"], cfg.target_threshold,
                                 cfg.grid).to(device)
    results = {}
    for arm, branch in branches.items():
        head = branch["head"]
        head.train(training)
        with torch.set_grad_enabled(training):
            arguments = (*query,) + ((support[0], support[1], labels) if support is not None else ())
            logits, aux = head(*arguments, return_aux=True)
            loss, components = masked_three_class_loss(logits, query_gt, query_ignore, cfg.target_threshold,
                                                       cfg.w_ce, cfg.w_dice)
            if all_labels is not None:
                change = soft_grid_loss(aux["change_logits"], all_labels)
                loss = loss + cfg.w_change * change
                components["change"] = float(change.detach())
            if arm in cfg.par_arms and support is not None and cfg.w_par:
                alignment = head.alignment_loss(aux["query_tokens"], logits, aux["support_tokens"], labels)
                loss = loss + cfg.w_par * alignment
                components["par"] = float(alignment.detach())
            components["total"] = float(loss.detach())
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise ValueError(f"Nonfinite {arm} loss/output")
            if training:
                (loss / cfg.accum).backward()
                if update:
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
                    if not torch.isfinite(norm):
                        raise ValueError(f"Nonfinite {arm} gradient")
                    branch["optimizer"].step()
                    branch["optimizer"].zero_grad(set_to_none=True)
                    branch["scheduler"].step()
        prediction = few_shot_eval.signed_change_map(logits.detach()).cpu()
        counts = np.zeros(6, np.int64)
        for index in range(prediction.shape[0]):
            counts += few_shot_eval.signed_counts(prediction[index], batch["query_gt"][index],
                                                  batch["query_ignore"][index], cfg.eval_threshold,
                                                  cfg.target_threshold)
        results[arm] = (components, counts)
        del logits, aux, loss
    return results


def make_loader(dataset, cfg):
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=cfg.num_workers,
                                       persistent_workers=False)


def run_epoch(branches, dataset, extractor, cfg, epoch):
    dataset.set_epoch(epoch)
    torch.manual_seed(cfg.seed * 1000 + epoch)
    totals = {arm: dict(loss={}, counts=np.zeros(6, np.int64)) for arm in branches}
    for branch in branches.values():
        branch["optimizer"].zero_grad(set_to_none=True)
    started, episodes = time.perf_counter(), 0
    for index, batch in enumerate(make_loader(dataset, cfg)):
        query, support = episode_tokens(batch, extractor, cfg.photometric_augmentation)
        update = (index + 1) % cfg.accum == 0
        for arm, (components, counts) in train_step(branches, batch, query, support, cfg, True, update).items():
            for key, value in components.items():
                totals[arm]["loss"][key] = totals[arm]["loss"].get(key, 0.0) + value
            totals[arm]["counts"] += counts
        episodes += 1
        if episodes == 1 or episodes % 100 == 0:
            print(f"[epoch {epoch} episode {episodes}] n={batch['n']} level={batch['level']} " + " ".join(
                f"{arm}={totals[arm]['loss']['total'] / episodes:.4f}" for arm in branches), flush=True)
    return {arm: dict(**{f"train_{key}": value / max(1, episodes) for key, value in values["loss"].items()},
                      train_pooled_signed_dice=few_shot_eval.pooled_dice(values["counts"]),
                      episodes=episodes, seconds=time.perf_counter() - started)
            for arm, values in totals.items()}


def validation_setup(cfg, run_root):
    path = Path(run_root) / "val_selection_manifest.json"
    if path.is_file():
        manifest = few_shot_manifests.read_manifest(path)
    else:
        manifest = few_shot_manifests.synthetic_manifest(cfg.val_dataset_root, "val", selection=True, seed=cfg.seed)
        few_shot_manifests.write_manifest(manifest, path)
    source = few_shot_eval.SyntheticPairSource(cfg.val_dataset_root, "val", cfg.val_store,
                                               cfg.rotation_k, cfg.flip_horizontal)
    return manifest, source


def validate_branches(branches, manifest, source, extractor, cfg):
    predictors = {arm: few_shot_eval.HeadPredictor(branch["head"], arm != "none", cfg.device)
                  for arm, branch in branches.items()}
    result = few_shot_eval.run_synthetic(manifest, source, extractor, predictors, cfg.eval_threshold,
                                         query_chunk=8, shuffle_check=cfg.val_shuffle_check, seed=cfg.seed)
    for branch in branches.values():
        branch["head"].train()
    return result


def provenance(cfg, corpus, manifest):
    hashed = {key: value for key, value in vars(cfg).items()
              if key not in ("device", "num_workers", "val_cache_pairs", "extract_chunk")}
    return dict(code_sha256={name: sha256(ROOT / name) for name in SOURCES},
                corpus_manifest_sha256=sha256(Path(cfg.corpus_root) / "manifest_scenes.jsonl"),
                corpus_scenes=len(corpus.records), corpus_tiers=list(cfg.corpus_tiers),
                val_manifest_sha256=manifest["meta"]["sha256"] if manifest else None,
                config_hash=json_hash(hashed))


def save_state(path, branches, epoch, cfg, identity):
    state = dict(epoch=epoch, config=vars(cfg), identity=identity, torch_rng=torch.get_rng_state(),
                 branches={arm: dict(head=branch["head"].state_dict(), optimizer=branch["optimizer"].state_dict(),
                                     scheduler=branch["scheduler"].state_dict(), best_score=branch["best_score"],
                                     best_epoch=branch["best_epoch"], history=branch["history"])
                           for arm, branch in branches.items()})
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_state(path, branches, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("Checkpoint identity (config/code/data) differs; use a new run root")
    for arm, saved in state["branches"].items():
        branch = branches[arm]
        branch["head"].load_state_dict(saved["head"])
        branch["optimizer"].load_state_dict(saved["optimizer"])
        branch["scheduler"].load_state_dict(saved["scheduler"])
        for key in ("best_score", "best_epoch", "history"):
            branch[key] = saved[key]
    torch.set_rng_state(state["torch_rng"])
    return state["epoch"]


def write_metrics(run_root, branches):
    rows = [dict(arm=arm, **{key: value for key, value in record.items() if not isinstance(value, dict)})
            for arm, branch in branches.items() for record in branch["history"]]
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with (Path(run_root) / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare(cfg, backbone=None):
    seed_everything(cfg.seed)
    run_root = Path(cfg.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    corpus = SceneCorpus(cfg.corpus_root, tuple(cfg.corpus_tiers), cfg.corpus_store)
    dataset = EpisodeDataset(corpus, episode_config(cfg), cfg.episodes_per_epoch, cfg.seed)
    backbone = backbone if backbone is not None else make_backbone(cfg)
    extractor = few_shot_eval.TokenExtractor(backbone, cfg.device, cfg.precision, cfg.extract_chunk,
                                             cache_items=cfg.val_cache_pairs)
    manifest, source = validation_setup(cfg, run_root) if cfg.val_dataset_root else (None, None)
    updates = cfg.episodes_per_epoch // cfg.accum
    branches = build_branches(cfg, updates)
    identity = provenance(cfg, corpus, manifest)
    return dict(run_root=run_root, corpus=corpus, dataset=dataset, extractor=extractor, manifest=manifest,
                source=source, branches=branches, identity=identity)


def train(cfg, context, resume=False):
    run_root, branches = context["run_root"], context["branches"]
    last = run_root / "last.pt"
    start = 0
    if last.is_file():
        if not resume:
            raise ValueError("Run root has a checkpoint; pass --resume or use a new run root")
        start = load_state(last, branches, context["identity"])
    (run_root / "identity.json").write_text(json.dumps(context["identity"], indent=1) + "\n", encoding="utf-8")
    for epoch in range(start, cfg.epochs):
        train_metrics = run_epoch(branches, context["dataset"], context["extractor"], cfg, epoch)
        validation = (validate_branches(branches, context["manifest"], context["source"], context["extractor"], cfg)
                      if context["manifest"] else None)
        for arm, branch in branches.items():
            record = dict(epoch=epoch + 1, lr=branch["optimizer"].param_groups[0]["lr"], **train_metrics[arm])
            if validation:
                summary = validation["summary"].get(arm, {})
                record["val_selection_score"] = summary.get("selection_score")
                shuffled = validation["summary"].get(f"shuffled:{arm}", {})
                record["val_shuffled_selection_score"] = shuffled.get("selection_score")
                score = record["val_selection_score"]
                if score is not None and score > branch["best_score"]:
                    branch["best_score"], branch["best_epoch"] = score, epoch + 1
                    torch.save(dict(spec=branch["spec"], state_dict=branch["head"].state_dict(), epoch=epoch + 1,
                                    score=score, config=vars(cfg), identity=context["identity"]),
                               run_root / f"best_{arm}.pt")
            branch["history"].append(record)
        if validation:
            (run_root / f"validation_epoch{epoch + 1:03d}.json").write_text(
                json.dumps(dict(summary=validation["summary"], nuisance_fp_area=validation["nuisance_fp_area"]),
                           indent=1) + "\n", encoding="utf-8")
        save_state(last, branches, epoch + 1, cfg, context["identity"])
        write_metrics(run_root, branches)
        print(f"[epoch {epoch + 1}] " + " ".join(
            f"{arm}: train_dice={branch['history'][-1]['train_pooled_signed_dice']} "
            f"val={branch['history'][-1].get('val_selection_score')}" for arm, branch in branches.items()), flush=True)
    summary = {arm: dict(best_epoch=branch["best_epoch"], best_score=branch["best_score"],
                         parameters=branch["parameters"]) for arm, branch in branches.items()}
    (run_root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def benchmark(cfg, context):
    """Memory/time per forced support size for all arms; choose the largest that fits."""
    results = []
    for size in sorted(cfg.benchmark_support, reverse=True):
        local = copy.copy(cfg)
        local.max_support = size
        dataset = EpisodeDataset(context["corpus"], episode_config(local, zero_support_prob=0.0),
                                 cfg.benchmark_episodes, cfg.seed)
        branches = build_branches(local, cfg.benchmark_episodes)
        if str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        status = "ok"
        try:
            for index in range(cfg.benchmark_episodes):
                rng = np.random.default_rng([cfg.seed, 999, index])
                episode = dataset.sampler.sample(rng, support_size=size)
                batch = materialize(context["corpus"], episode, dataset.config, rng)
                query, support = episode_tokens(batch, context["extractor"], cfg.photometric_augmentation)
                train_step(branches, batch, query, support, local, True, True)
        except torch.cuda.OutOfMemoryError:
            status = "oom"
        if str(cfg.device).startswith("cuda"):
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if str(cfg.device).startswith("cuda") else None
        seconds = time.perf_counter() - started
        results.append(dict(max_support=size, status=status, peak_allocated_gb=peak,
                            seconds_per_episode=seconds / cfg.benchmark_episodes,
                            fits=status == "ok" and (peak is None or peak <= cfg.benchmark_memory_gb)))
        del branches
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()
    chosen = max([row["max_support"] for row in results if row["fits"]], default=None)
    report = dict(results=results, chosen_max_support=chosen, arms=list(cfg.arms),
                  parameters={arm: count_parameters(FewShotChangeHead(**head_spec(cfg, arm)["kwargs"]))
                              for arm in cfg.arms}, identity=context["identity"])
    (Path(cfg.run_root) / "benchmark.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def precision_parity(extractor, corpus, cfg, samples=4):
    """Cosine similarity between bf16 and fp32 frozen tokens on a few corpus images."""
    images = torch.cat([torch.stack(corpus.images(row)) for row in range(min(samples, len(corpus.records)))])
    reference = copy.copy(extractor)
    reference.precision = "fp32"
    low, high = extractor.images(images)[0], reference.images(images)[0]
    return float(torch.nn.functional.cosine_similarity(low.flatten(1), high.flatten(1)).min())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--max_support", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, max_support=args.max_support, device=args.device)
    context = prepare(cfg)
    if args.benchmark:
        parity = precision_parity(context["extractor"], context["corpus"], cfg) if cfg.precision == "bf16" else 1.0
        report = benchmark(cfg, context)
        report["bf16_min_token_cosine"] = parity
        (Path(cfg.run_root) / "benchmark.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("results", "chosen_max_support", "bf16_min_token_cosine")},
                         indent=1))
        return 0
    if not (Path(cfg.run_root) / "benchmark.json").is_file():
        raise ValueError("Run --benchmark first and set max_support to the chosen value")
    chosen = json.loads((Path(cfg.run_root) / "benchmark.json").read_text())["chosen_max_support"]
    if chosen is None or cfg.max_support > chosen:
        raise ValueError(f"max_support={cfg.max_support} exceeds the benchmarked limit {chosen}")
    print(json.dumps(train(cfg, context, args.resume), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
