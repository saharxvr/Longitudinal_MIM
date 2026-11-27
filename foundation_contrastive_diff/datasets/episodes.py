"""Selective few-shot episodes over the supervoxel scene corpus.

An episode = one task tau + n support scenes + B query scenes (all from different patients
than the supports). Targets mark ONLY the query changes that match tau; other changes in
the same scene are unlabeled distractors. Opposite-sign overlaps between tau and distractor
changes (or within tau) are ignored. Supports always contain >=1 tau change; a fraction of
queries contain none (empty target).

Task levels (granularity sampled per episode):
    fine         family + GT sign             (e.g. heart_resize, positive)
    family       family, either sign
    compartment  compartment + GT sign
    any          every realized change        (the ICU-like generic mission; allows n = 0)

Support label styles (decision deferred, see FEW_SHOT_DIFF_PLAN.md 4.2):
    tight            exact per-change GT
    coarse_support   with probability p the whole support set uses coarse labels
    coarse_both      ... and so do the query targets of that episode
Coarse variants: stored ellipses (default), bbox, dilate, erode, drop-one-component.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import ndimage as ndi

from data_generation.supervoxel_render import EIGHT_CONNECTED, rasterize_ellipses
from datasets.image_store import ArrayStore
from datasets.pair_dataset import _load_nii_2d, _norm01, _to_chw, orient_tensor

LEVELS = ("fine", "family", "compartment", "any")
LABEL_MODES = ("tight", "coarse_support", "coarse_both")
COARSE_VARIANTS = dict(ellipse=0.6, bbox=0.15, dilate=0.1, erode=0.05, drop=0.1)
SIZE = 512


@dataclass(frozen=True)
class Task:
    level: str
    family: Optional[str] = None
    compartment: Optional[str] = None
    sign: Optional[str] = None

    def matches(self, change):
        if not change.get("realized"):
            return False
        if self.level == "any":
            return True
        if self.level == "fine":
            return change["family"] == self.family and change["gt_sign"] == self.sign
        if self.level == "family":
            return change["family"] == self.family
        if self.level == "compartment":
            return change["compartment"] == self.compartment and change["gt_sign"] == self.sign
        raise ValueError(f"Unknown task level {self.level}")

    @property
    def key(self):
        return "|".join(str(value) for value in (self.level, self.family, self.compartment, self.sign))

    @staticmethod
    def from_change(level, change):
        if level == "fine":
            return Task("fine", family=change["family"], sign=change["gt_sign"])
        if level == "family":
            return Task("family", family=change["family"])
        if level == "compartment":
            return Task("compartment", compartment=change["compartment"], sign=change["gt_sign"])
        return Task("any")


@dataclass
class EpisodeConfig:
    max_support: int = 32
    queries: int = 4
    level_weights: tuple = (0.45, 0.25, 0.15, 0.15)
    empty_query_prob: float = 0.15
    distractor_query_prob: float = 0.5
    zero_support_prob: float = 0.05
    label_mode: str = "tight"
    coarse_prob: float = 0.3
    target_threshold: float = 0.001
    rotation_k: int = 3
    flip_horizontal: bool = True

    def validate(self):
        if self.label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}")
        if (self.max_support < 1 or self.queries < 1 or len(self.level_weights) != len(LEVELS)
                or not all(0 <= value <= 1 for value in (self.empty_query_prob, self.distractor_query_prob,
                                                         self.zero_support_prob, self.coarse_prob))):
            raise ValueError("Invalid episode configuration")


class SceneCorpus:
    """Scene manifest + optional float16 store; restricted to allowed corpus tiers."""

    def __init__(self, root, allowed_tiers=("T1", "T2"), store=None):
        self.root = Path(root)
        lines = (self.root / "manifest_scenes.jsonl").read_text(encoding="utf-8").splitlines()
        allowed = set(allowed_tiers)
        self.records = [record for record in (json.loads(line) for line in lines if line.strip())
                        if set(record["corpus_tiers"]) <= allowed and record.get("split") == "train"]
        if not self.records:
            raise ValueError("No scenes for the allowed tiers")
        store = store if store is not None else (self.root / "_store")
        self.store = ArrayStore(store) if Path(store).is_dir() else None
        self.patients = [record["patient_key"] for record in self.records]
        self._ellipses: Dict[int, list] = {}
        self.index: Dict[str, Dict[str, List[int]]] = {level: {} for level in LEVELS}
        self.tasks: Dict[str, Task] = {}
        for row, record in enumerate(self.records):
            for change in record["changes"]:
                if not change.get("realized"):
                    continue
                for level in LEVELS:
                    task = Task.from_change(level, change)
                    scenes = self.index[level].setdefault(task.key, [])
                    if not scenes or scenes[-1] != row:
                        scenes.append(row)
                    self.tasks[task.key] = task

    def images(self, row):
        record = self.records[row]
        if self.store is not None:
            return self.store.pair(record["scene_id"])
        folder = self.root / record["scene_dir"]
        return tuple(_norm01(_to_chw(_load_nii_2d(str(folder / record["files"][name])), SIZE))
                     for name in ("prior", "current"))

    def maps(self, row):
        record = self.records[row]
        if self.store is not None:
            return self.store.scene_maps(record["scene_id"])
        folder = self.root / record["scene_dir"]
        return torch.from_numpy(np.load(folder / record["files"]["changes"])["maps"].astype(np.float32))

    def ellipses(self, row):
        if row not in self._ellipses:
            record = self.records[row]
            path = self.root / record["scene_dir"] / record["files"]["ellipses"]
            self._ellipses[row] = json.loads(path.read_text(encoding="utf-8"))
        return self._ellipses[row]


def task_target(maps, changes, task, threshold=0.001):
    """Signed target of tau, ignore mask (opposite-sign overlaps), and the all-change target."""
    if len(changes) != maps.shape[0]:
        raise ValueError("Change records and maps disagree")
    selected = torch.tensor([task.matches(change) for change in changes], dtype=torch.bool)
    realized = torch.tensor([bool(change.get("realized")) for change in changes], dtype=torch.bool)
    shape = maps.shape[1:]

    def combine(mask):
        if not bool(mask.any()):
            return torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)
        chosen = maps[mask]
        positive, negative = (chosen > threshold).any(0), (chosen < -threshold).any(0)
        return chosen.sum(0).clamp(-1, 1), positive & negative

    target, conflict = combine(selected)
    others, _ = combine(realized & ~selected)
    everything, all_conflict = combine(realized)
    opposite = (((target > threshold) & (others < -threshold))
                | ((target < -threshold) & (others > threshold)))
    return target, conflict | opposite, everything, all_conflict


def coarse_target(target, ellipses, changes, task, variant, rng, shape=(SIZE, SIZE)):
    """Coarse signed label (+-0.5) in the saved frame; ignore marks pixels with both signs."""
    if variant == "ellipse" or variant == "drop":
        items = [ellipse for change, group in zip(changes, ellipses) if task.matches(change)
                 for ellipse in group]
        if variant == "drop" and len(items) > 1:
            items.pop(int(rng.integers(len(items))))
        positive, negative = rasterize_ellipses(items, shape)
    else:
        positive, negative = (target > 0).numpy(), (target < 0).numpy()
        if variant == "bbox":
            positive, negative = (_boxes(mask) for mask in (positive, negative))
        elif variant == "dilate":
            radius = int(rng.integers(5, 16))
            positive, negative = (ndi.binary_dilation(mask, iterations=radius) if mask.any() else mask
                                  for mask in (positive, negative))
        elif variant == "erode":
            radius = int(rng.integers(2, 7))
            positive, negative = (ndi.binary_erosion(mask, iterations=radius) for mask in (positive, negative))
        else:
            raise ValueError(f"Unknown coarse variant {variant}")
    signed = torch.from_numpy(positive.astype(np.float32) * 0.5 - negative.astype(np.float32) * 0.5)
    return signed, torch.from_numpy(positive & negative)


def _boxes(mask):
    labels, count = ndi.label(mask, EIGHT_CONNECTED)
    output = np.zeros_like(mask)
    for region in ndi.find_objects(labels):
        if region is not None:
            output[region] = True
    return output


def log_uniform_int(rng, maximum):
    return min(maximum, max(1, int(math.floor(math.exp(rng.uniform(0, math.log(maximum + 1)))))))


class EpisodeSampler:
    def __init__(self, corpus: SceneCorpus, config: EpisodeConfig):
        config.validate()
        self.corpus, self.config = corpus, config
        self.eligible = {level: [key for key, rows in corpus.index[level].items()
                                 if len({corpus.patients[row] for row in rows}) >= 2]
                         for level in LEVELS}
        if not any(self.eligible.values()):
            raise ValueError("No task has positives from at least two patients")
        self.distractor_rows = [row for row, record in enumerate(corpus.records)
                                if record.get("realized_changes", 0) >= 2]

    def sample(self, rng, support_size=None, _depth=0):
        corpus, config = self.corpus, self.config
        weights = np.array([w if self.eligible[level] else 0 for w, level in zip(config.level_weights, LEVELS)])
        level = LEVELS[int(rng.choice(len(LEVELS), p=weights / weights.sum()))]
        key = self.eligible[level][int(rng.integers(len(self.eligible[level])))]
        task = corpus.tasks[key]
        positives = corpus.index[level][key]
        positive_set = set(positives)
        queries = []
        for _ in range(config.queries):
            if rng.random() < config.empty_query_prob:
                row = None
                for _attempt in range(20):
                    candidate = int(rng.integers(len(corpus.records)))
                    if candidate not in positive_set:
                        row = candidate
                        break
                if row is None:
                    row = positives[int(rng.integers(len(positives)))]
            elif self.distractor_rows and rng.random() < config.distractor_query_prob:
                candidates = [row for row in positives if row in self._distractor_set()]
                row = (candidates[int(rng.integers(len(candidates)))] if candidates
                       else positives[int(rng.integers(len(positives)))])
            else:
                row = positives[int(rng.integers(len(positives)))]
            queries.append(row)
        query_patients = {corpus.patients[row] for row in queries}
        pool = [row for row in positives if corpus.patients[row] not in query_patients]
        if support_size is None:
            zero = level == "any" and rng.random() < config.zero_support_prob
            support_size = 0 if zero else log_uniform_int(rng, config.max_support)
        support_size = min(support_size, len(pool))
        if support_size == 0 and level != "any":
            if _depth > 50:
                raise ValueError("Could not sample supports from patients disjoint from the queries")
            return self.sample(rng, None, _depth + 1)
        supports = [pool[index] for index in rng.permutation(len(pool))[:support_size]]
        coarse = config.label_mode != "tight" and rng.random() < config.coarse_prob
        return dict(task=task, supports=supports, queries=queries, coarse=coarse,
                    coarse_variant=_choice(rng, COARSE_VARIANTS) if coarse else None)

    def _distractor_set(self):
        if not hasattr(self, "_distractors"):
            self._distractors = set(self.distractor_rows)
        return self._distractors


def _choice(rng, weights):
    names = list(weights)
    probabilities = np.array([weights[name] for name in names], np.float64)
    return names[int(rng.choice(len(names), p=probabilities / probabilities.sum()))]


def orient(config, tensor):
    return orient_tensor(tensor, config.rotation_k, config.flip_horizontal)


def materialize(corpus: SceneCorpus, episode, config: EpisodeConfig, rng):
    """Load and orient tensors; returns the dict consumed by the few-shot trainer."""
    task = episode["task"]

    def item(row, role):
        record = corpus.records[row]
        prior, current = corpus.images(row)
        maps = corpus.maps(row)
        target, ignore, everything, all_ignore = task_target(maps, record["changes"], task,
                                                             config.target_threshold)
        coarse = episode["coarse"] and (role == "support" or config.label_mode == "coarse_both")
        if coarse:
            target, ignore = coarse_target(target, corpus.ellipses(row), record["changes"], task,
                                           episode["coarse_variant"], rng)
        tensors = [prior, current, target[None], ignore[None], everything[None], all_ignore[None]]
        return [orient(config, tensor) for tensor in tensors]

    supports = [item(row, "support") for row in episode["supports"]]
    queries = [item(row, "query") for row in episode["queries"]]

    def stack(items, position, dtype=torch.float32):
        if not items:
            return torch.zeros((0, 1, SIZE, SIZE), dtype=dtype)
        return torch.stack([entry[position] for entry in items]).to(dtype)

    return dict(
        task=task.key, level=task.level, n=len(supports), coarse=bool(episode["coarse"]),
        coarse_variant=episode["coarse_variant"] or "",
        support_prior=stack(supports, 0), support_curr=stack(supports, 1),
        support_gt=stack(supports, 2), support_ignore=stack(supports, 3, torch.bool),
        query_prior=stack(queries, 0), query_curr=stack(queries, 1),
        query_gt=stack(queries, 2), query_ignore=stack(queries, 3, torch.bool),
        query_all_gt=stack(queries, 4), query_all_ignore=stack(queries, 5, torch.bool),
        support_ids=[corpus.records[row]["scene_id"] for row in episode["supports"]],
        query_ids=[corpus.records[row]["scene_id"] for row in episode["queries"]])


class EpisodeDataset(torch.utils.data.Dataset):
    """Deterministic episodes: (seed, epoch, index) -> one task batch. Use batch_size=None."""

    def __init__(self, corpus, config, episodes_per_epoch, seed=42):
        self.sampler = EpisodeSampler(corpus, config)
        self.corpus, self.config = corpus, config
        self.episodes_per_epoch, self.seed, self.epoch = int(episodes_per_epoch), int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, int(index)])
        return materialize(self.corpus, self.sampler.sample(rng), self.config, rng)
