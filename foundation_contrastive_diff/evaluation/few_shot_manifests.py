"""Fixed, seeded few-shot evaluation manifests (identical support draws for every model).

Each entry = (task, n, draw, group): one support set shared by the queries of one patient
group; supports never share a patient with those queries. Every query is evaluated exactly
once per (n, draw). Entries whose pool is too small for n are recorded as unavailable.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_VALUES = (1, 2, 4, 8, 16, 32, 64)
SYNTHETIC_DIRECTIONS = ("appearance", "disappearance")


def manifest_hash(manifest):
    payload = json.dumps(manifest["entries"], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def partition(patients, groups, rng):
    unique = sorted(set(patients))
    order = rng.permutation(len(unique))
    return {unique[index]: position % groups for position, index in enumerate(order)}


def build_manifest(task_queries, task_supports, n_values=N_VALUES, draws=10, groups=4, seed=0,
                   nuisance=None, nuisance_per_task=0, max_queries=None):
    """task_queries/task_supports: {task: [(item_id, patient), ...]}; nuisance: [(item_id, patient)]."""
    rng = np.random.default_rng(seed)
    entries, unavailable = [], []
    for task in sorted(task_queries):
        queries = list(task_queries[task])
        if max_queries and len(queries) > max_queries:
            queries = [queries[index] for index in sorted(rng.choice(len(queries), max_queries, replace=False))]
        extra = []
        if nuisance and nuisance_per_task:
            count = min(nuisance_per_task, len(nuisance))
            extra = [nuisance[index] for index in sorted(rng.choice(len(nuisance), count, replace=False))]
        groups_of = partition([patient for _, patient in queries + extra], groups, rng)
        pool = list(task_supports[task])
        for n in n_values:
            for draw in range(draws):
                for group in range(groups):
                    members = [item for item, patient in queries if groups_of[patient] == group]
                    nuisance_members = [item for item, patient in extra if groups_of[patient] == group]
                    if not members and not nuisance_members:
                        continue
                    excluded = {patient for patient, assigned in groups_of.items() if assigned == group}
                    excluded_items = set(members) | set(nuisance_members)
                    candidates = [item for item, patient in pool
                                  if patient not in excluded and item not in excluded_items]
                    if len(candidates) < n:
                        unavailable.append(dict(task=task, n=n, draw=draw, group=group,
                                                available=len(candidates)))
                        continue
                    support = [candidates[index] for index in rng.choice(len(candidates), n, replace=False)]
                    entries.append(dict(task=task, n=n, draw=draw, group=group, support=support,
                                        queries=members, nuisance=nuisance_members))
    manifest = dict(entries=entries, unavailable=unavailable,
                    meta=dict(n_values=list(n_values), draws=draws, groups=groups, seed=seed,
                              nuisance_per_task=nuisance_per_task, max_queries=max_queries))
    manifest["meta"]["sha256"] = manifest_hash(manifest)
    return manifest


def synthetic_task_pools(records, split="val"):
    """Validation tasks = realized pathology x direction; nuisance = realized-none pairs."""
    from datasets.pair_dataset import record_patient

    pools, nuisance = defaultdict(list), []
    for record in records:
        if record.get("split") != split:
            continue
        item = (record["pair_id"], record_patient(record))
        anomaly = record.get("effective_anomaly_type", record.get("anomaly_type", "none"))
        if not record.get("realized_change", anomaly != "none") or anomaly == "none":
            nuisance.append(item)
        elif record.get("direction") in SYNTHETIC_DIRECTIONS:
            pools[f"{anomaly}|{record['direction']}"].append(item)
    return dict(pools), nuisance


def synthetic_manifest(dataset_root, split="val", selection=False, seed=0, **overrides):
    """Pre-declared manifests: selection (n 1/4/16, R=3, <=64 queries/task) or final (all, R=10)."""
    from datasets.pair_dataset import _read_manifest

    pools, nuisance = synthetic_task_pools(_read_manifest(dataset_root), split)
    settings = (dict(n_values=(1, 4, 16), draws=3, groups=4, max_queries=64, nuisance_per_task=16)
                if selection else dict(n_values=N_VALUES, draws=10, groups=4, max_queries=None,
                                       nuisance_per_task=64))
    settings.update(overrides)
    manifest = build_manifest(pools, pools, seed=seed, nuisance=nuisance, **settings)
    manifest["meta"].update(kind="synthetic_selection" if selection else "synthetic_final", split=split,
                            tasks={task: len(items) for task, items in pools.items()},
                            nuisance_pairs=len(nuisance))
    return manifest


def icu_manifest(pair_ids, support_pool, patients=None, n_values=N_VALUES, draws=10, groups=10, seed=0):
    """Leave-group-out supports from the other ICU pairs (one task: the annotation mission)."""
    patients = patients or {pair: pair for pair in pair_ids}
    queries = {"icu": [(pair, patients[pair]) for pair in pair_ids]}
    supports = {"icu": [(pair, patients[pair]) for pair in support_pool]}
    manifest = build_manifest(queries, supports, n_values=n_values, draws=draws, groups=groups, seed=seed)
    manifest["meta"].update(kind="icu", pairs=len(pair_ids), support_pool=len(support_pool))
    return manifest


def write_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return path


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest_hash(manifest) != manifest["meta"]["sha256"]:
        raise ValueError(f"Episode manifest hash mismatch: {path}")
    return manifest
