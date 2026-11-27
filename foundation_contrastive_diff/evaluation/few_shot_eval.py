"""Few-shot evaluation core: pair sources, frozen token extraction, synthetic and ICU scoring.

Predictors share one interface (see ``predict``). ``uses_support=False`` predictors (the
supervised arm A, FSDiff/none with n = 0) are scored once at n = 0, not per support size.
All models see identical support draws because the manifest is fixed and hashed.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict

import numpy as np
import torch

from datasets.pair_dataset import _to_chw, orient_tensor
from losses.few_shot_losses import masked_three_class_loss
from models import signed_change_map
from models.few_shot_head import grid_labels

SIZE = 512
METRICS = ("one_to_many", "one_to_one_presentation", "one_to_one_legacy")


# ---------------------------------------------------------------------------- sources
class SyntheticPairSource:
    """Synthetic pairs (pair store or NIfTI files) in the model frame."""

    def __init__(self, dataset_root, split="val", store=None, rotation_k=3, flip_horizontal=True):
        from datasets.image_store import ArrayStore
        from datasets.pair_dataset import LongitudinalPairDataset

        self.store = ArrayStore(store) if store else None
        self.dataset = None if self.store else LongitudinalPairDataset(dataset_root, split=split)
        self.positions = ({record["pair_id"]: index for index, record in enumerate(self.dataset.records)}
                          if self.dataset else None)
        self.rotation_k, self.flip = rotation_k, flip_horizontal

    def item(self, key, mode=None):
        if self.store is not None:
            prior, current = self.store.pair(key)
            gt = self.store.pair_map(key)
        else:
            sample = self.dataset[self.positions[key]]
            prior, current, gt = sample["img_prior"], sample["img_curr"], sample["gt_diff"]
        oriented = [orient_tensor(tensor, self.rotation_k, self.flip) for tensor in (prior, current, gt)]
        return (*oriented, torch.zeros((1, SIZE, SIZE), dtype=torch.bool))


class ICUPairSource:
    """ICU-100 pairs with consensus or per-reader support GT (ellipses) in the model frame."""

    def __init__(self, clinical_root, rotation_k=3, flip_horizontal=True, consensus_k=3, count=100):
        from evaluation.evaluate_clinical import (PHYSICIANS, clinical_tools, discover_cases,
                                                  load_case, prepare_image)

        self.tools = clinical_tools()
        self.cases = {case["pair_id"]: case for case in discover_cases(clinical_root, count)}
        self.rotation_k, self.flip, self.consensus_k = rotation_k, flip_horizontal, consensus_k
        self._load_case, self._prepare, self.readers = load_case, prepare_image, tuple(PHYSICIANS)
        self.cache = {}

    def load(self, key):
        if key not in self.cache:
            prior, current, _, annotations = self._load_case(dict(self.cases[key]), self.tools)
            images = tuple(self._prepare(image, SIZE, self.rotation_k, self.flip) for image in (prior, current))
            self.cache[key] = dict(images=images, annotations=annotations, shape=current.shape)
        return self.cache[key]

    def consensus(self, key, k=None):
        annotations = self.load(key)["annotations"]
        k = k or self.consensus_k
        return tuple(np.stack([masks[sign] for masks in annotations.values()]).sum(0) >= k for sign in (0, 1))

    def native_masks(self, key, mode):
        if mode == "consensus":
            return self.consensus(key)
        if mode.startswith("consensus_"):
            return self.consensus(key, int(mode.rsplit("_", 1)[1]))
        if mode.startswith("reader:"):
            return self.load(key)["annotations"][mode.split(":", 1)[1]]
        raise ValueError(f"Unknown ICU support mode {mode}")

    def _model_frame(self, mask):
        return orient_tensor(_to_chw(mask.astype(np.float32), SIZE) > 0.5, self.rotation_k, self.flip)

    def item(self, key, mode="consensus"):
        loaded = self.load(key)
        positive, negative = (self._model_frame(mask) for mask in self.native_masks(key, mode))
        signed = positive.float() * 0.5 - negative.float() * 0.5
        return loaded["images"][0], loaded["images"][1], signed, positive & negative

    def nonempty(self, key, mode="consensus"):
        return any(mask.any() for mask in self.native_masks(key, mode))


# ---------------------------------------------------------------------------- tokens
class TokenExtractor:
    """Frozen multi-depth RAD-DINO tokens with a pair-level LRU cache (fp16 on CPU)."""

    def __init__(self, backbone, device="cpu", precision="fp32", chunk=32, cache_items=256):
        if backbone.training or any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Few-shot evaluation requires a frozen eval-mode backbone")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16")
        self.backbone, self.device, self.precision = backbone, device, precision
        self.chunk, self.cache_items, self.cache = chunk, cache_items, OrderedDict()

    @torch.no_grad()
    def images(self, images):
        patches, classes = [], []
        use_bf16 = self.precision == "bf16" and str(self.device).startswith("cuda")
        for start in range(0, images.shape[0], self.chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                output = self.backbone(images[start:start + self.chunk].to(self.device))
            patches.append(output["patch_tokens"].float())
            classes.append(output["cls_token"].float())
        return torch.cat(patches), torch.cat(classes)

    def pairs(self, keys, loader):
        """(p_prior, p_curr, cls_prior, cls_curr) on the device for item keys."""
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        fresh = {}
        if missing:
            loaded = [loader(key) for key in missing]
            patches, classes = self.images(torch.cat([torch.stack([item[0] for item in loaded]),
                                                      torch.stack([item[1] for item in loaded])]))
            count = len(missing)
            for index, key in enumerate(missing):
                fresh[key] = (patches[index], patches[count + index], classes[index], classes[count + index])
                if self.cache_items:
                    self.cache[key] = tuple(tensor.to("cpu", torch.float16) for tensor in fresh[key])
        result = [fresh[key] if key in fresh else self.cache[key] for key in keys]
        for key in keys:
            if key in self.cache:
                self.cache.move_to_end(key)
        while len(self.cache) > self.cache_items:
            self.cache.popitem(last=False)
        return tuple(torch.stack([item[part].to(self.device, torch.float32) for item in result])
                     for part in range(4))


# ---------------------------------------------------------------------------- predictors
class HeadPredictor:
    def __init__(self, head, uses_support=True, device="cpu"):
        self.head, self.uses_support, self.device = head.to(device).eval(), uses_support, device

    @torch.no_grad()
    def __call__(self, query, support=None, labels=None, targets=None):
        if self.uses_support and support is not None:
            output = self.head(*query, support[0], support[1], labels)
        else:
            output = self.head(*query)
        return signed_change_map(output[0] if isinstance(output, tuple) else output)


class FineTunePredictor:
    """Few-shot fine-tuning baseline: adapt a copy of a supervised head on the supports."""

    uses_support = True

    def __init__(self, head, steps=20, lr=1e-4, threshold=0.001, device="cpu"):
        self.base, self.steps, self.lr, self.threshold, self.device = head.to(device), steps, lr, threshold, device

    def __call__(self, query, support=None, labels=None, targets=None):
        head = copy.deepcopy(self.base).train()
        if support is not None and targets is not None and self.steps:
            optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=self.lr)
            for _ in range(self.steps):
                output = head(*support)
                logits = output[0] if isinstance(output, tuple) else output
                loss, _ = masked_three_class_loss(logits, targets[0], targets[1], self.threshold)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.no_grad():
            output = head(*query)
        return signed_change_map(output[0] if isinstance(output, tuple) else output)


def predict(predictor, query, support, labels, targets):
    if not predictor.uses_support:
        return predictor(query).cpu()
    if isinstance(predictor, FineTunePredictor):
        return predictor(query, support, labels, targets).cpu()
    return predictor(query, support[:2], labels).cpu()


class SupportBundle:
    """Support tokens, grid labels and full-resolution targets for one manifest entry."""

    def __init__(self, keys, extractor, loader):
        self.tokens = extractor.pairs(keys, loader)
        items = [loader(key) for key in keys]
        gt = torch.stack([item[2] for item in items])
        ignore = torch.stack([item[3] for item in items])
        grid = int(round(self.tokens[0].shape[1] ** 0.5))
        self.labels = grid_labels(gt, ignore, grid=grid).to(extractor.device)
        self.targets = (gt.to(extractor.device), ignore.to(extractor.device))


# ---------------------------------------------------------------------------- synthetic scoring
def signed_counts(prediction, target, ignore, threshold=0.5, cutoff=0.001):
    valid = ~ignore
    counts = []
    for predicted, reference in ((prediction > threshold, target > cutoff),
                                 (prediction < -threshold, target < -cutoff)):
        predicted, reference = predicted & valid, reference & valid
        counts += [int((predicted & reference).sum()), int(predicted.sum()), int(reference.sum())]
    return counts


def pooled_dice(counts):
    if not counts[2] + counts[5]:
        return None
    return 2 * (counts[0] + counts[3]) / (counts[1] + counts[2] + counts[4] + counts[5])


def run_synthetic(manifest, source, extractor, predictors, threshold=0.5, cutoffs=(0.001, 0.05),
                  query_chunk=8, shuffle_check=False, seed=0):
    """Pooled signed Dice per (predictor, task, n, draw); nuisance FP area; support-shuffle check."""
    totals = defaultdict(lambda: np.zeros(6, np.int64))
    nuisance_area = defaultdict(list)
    zero_done = set()
    rng = np.random.default_rng(seed)
    by_n = defaultdict(list)
    for entry in manifest["entries"]:
        by_n[entry["n"]].append(entry)
    loader = source.item
    for entry in manifest["entries"]:
        variants = [("", SupportBundle(entry["support"], extractor, loader))]
        if shuffle_check:
            others = [other for other in by_n[entry["n"]] if other["task"] != entry["task"]]
            if others:
                other = others[int(rng.integers(len(others)))]
                variants.append(("shuffled:", SupportBundle(other["support"], extractor, loader)))
        for kind, keys in (("task", entry["queries"]), ("nuisance", entry["nuisance"])):
            for start in range(0, len(keys), query_chunk):
                chunk = keys[start:start + query_chunk]
                query = extractor.pairs(chunk, loader)
                items = [loader(key) for key in chunk]
                for name, predictor in predictors.items():
                    zero = not predictor.uses_support
                    done_key = (name, kind, tuple(chunk), entry["task"])
                    for prefix, bundle in variants:
                        if zero and (prefix or done_key in zero_done):
                            continue
                        if prefix and isinstance(predictor, FineTunePredictor):
                            continue
                        prediction = predict(predictor, query, bundle.tokens, bundle.labels, bundle.targets)
                        n, draw = (0, 0) if zero else (entry["n"], entry["draw"])
                        for index, item in enumerate(items):
                            if kind == "nuisance":
                                area = float((prediction[index].abs() > threshold).float().mean())
                                nuisance_area[(prefix + name, entry["task"], n)].append(area)
                                continue
                            for cutoff in cutoffs:
                                totals[(prefix + name, entry["task"], n, draw, cutoff)] += signed_counts(
                                    prediction[index], item[2], item[3], threshold, cutoff)
                    if zero:
                        zero_done.add(done_key)
    return summarize_synthetic(totals, nuisance_area, cutoffs)


def summarize_synthetic(totals, nuisance_area, cutoffs, selection_n=(1, 4, 16)):
    rows = [dict(predictor=name, task=task, n=n, draw=draw, gt_cutoff=cutoff,
                 pooled_signed_dice=pooled_dice(counts), counts=counts.tolist())
            for (name, task, n, draw, cutoff), counts in sorted(totals.items(), key=lambda item: str(item[0]))]
    summary = {}
    for name in sorted({row["predictor"] for row in rows}):
        result = {}
        sizes = sorted({row["n"] for row in rows if row["predictor"] == name})
        for cutoff in cutoffs:
            for n in sizes:
                draws = defaultdict(list)
                for row in rows:
                    if (row["predictor"] == name and row["n"] == n and row["gt_cutoff"] == cutoff
                            and row["pooled_signed_dice"] is not None):
                        draws[row["draw"]].append(row["pooled_signed_dice"])
                macro = [float(np.mean(values)) for values in draws.values()]
                if macro:
                    result[f"n{n}_gt{cutoff:g}"] = dict(
                        task_macro_dice_mean=float(np.mean(macro)),
                        task_macro_dice_sd=float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
                        draws=len(macro))
        chosen = [n for n in sizes if n in selection_n] or sizes
        values = [result[f"n{n}_gt{cutoffs[0]:g}"]["task_macro_dice_mean"] for n in chosen
                  if f"n{n}_gt{cutoffs[0]:g}" in result]
        if values:
            result["selection_score"] = float(np.mean(values))
        summary[name] = result
    nuisance = {f"{name}|{task}|n{n}": dict(mean=float(np.mean(values)), p95=float(np.quantile(values, .95)),
                                            max=float(np.max(values)), pairs=len(values))
                for (name, task, n), values in nuisance_area.items()}
    return dict(rows=rows, summary=summary, nuisance_fp_area=nuisance)


# ---------------------------------------------------------------------------- ICU scoring
def icu_terms(scores, references, threshold=0.75, cc_cutoff=250):
    """[refs, 2 signs, 6] presentation/legacy component terms (see rescore_subset_clinical_many)."""
    from evaluation.rescore_subset_clinical_many import comparison_terms, components
    from evaluation.sweep_clinical_postprocessing import masks_for

    positive, negative = masks_for(scores, f"components_{cc_cutoff}", threshold)
    predicted = (components(positive), components(negative))
    terms = np.zeros((len(references), 2, 6), np.int64)
    for reader, masks in enumerate(references):
        for sign in (0, 1):
            terms[reader, sign] = comparison_terms(predicted[sign], components(masks[sign]))
    return terms, float((positive | negative).mean())


def vector_scores(terms):
    """Vectorized rescore_subset_clinical_many.scores over [..., 6] terms."""
    predicted, reference, matched_pred, matched_ref, presentation, legacy = np.moveaxis(terms, -1, 0)
    denominator = (predicted + reference).astype(np.float64)
    empty = denominator == 0
    safe = np.where(empty, 1, denominator)
    return dict(one_to_many=np.where(empty, 1.0, (matched_pred + matched_ref) / safe),
                one_to_one_presentation=np.where(empty, 1.0, 2 * presentation / safe),
                one_to_one_legacy=np.where(empty, np.nan, 2 * legacy / safe))


def run_icu(manifest, source, extractor, predictors, support_mode="consensus", threshold=0.75,
            cc_cutoff=250, query_chunk=8, ensemble=True, retrieval=False):
    """results[predictor][(n, draw)][pair] = terms [readers+consensus, 2, 6]."""
    from evaluation.evaluate_clinical import restore_prediction

    pairs = sorted({key for entry in manifest["entries"] for key in entry["queries"]})
    references = {key: [source.load(key)["annotations"][reader] for reader in source.readers]
                  + [source.consensus(key)] for key in pairs}
    results = {name: defaultdict(dict) for name in predictors}
    areas = {name: defaultdict(list) for name in predictors}
    sums, counts, zero_done = {}, defaultdict(int), set()

    def loader(key):
        return source.item(key, support_mode)

    def score(name, n, draw, key, signed):
        native = restore_prediction(signed[0], source.load(key)["shape"], source.rotation_k, source.flip)
        terms, area = icu_terms(native, references[key], threshold, cc_cutoff)
        results[name][(n, draw)][key] = terms
        areas[name][(n, draw)].append(area)

    for entry in manifest["entries"]:
        bundle = SupportBundle(entry["support"], extractor, loader)
        for start in range(0, len(entry["queries"]), query_chunk):
            chunk = entry["queries"][start:start + query_chunk]
            query = extractor.pairs(chunk, loader)
            for name, predictor in predictors.items():
                if not predictor.uses_support and (name, tuple(chunk)) in zero_done:
                    continue
                prediction = predict(predictor, query, bundle.tokens, bundle.labels, bundle.targets)
                for index, key in enumerate(chunk):
                    if not predictor.uses_support:
                        score(name, 0, 0, key, prediction[index])
                        continue
                    score(name, entry["n"], entry["draw"], key, prediction[index])
                    if ensemble:
                        value = prediction[index].to(torch.float16)
                        slot = (name, entry["n"], key)
                        sums[slot] = value if slot not in sums else sums[slot] + value
                        counts[slot] += 1
                if not predictor.uses_support:
                    zero_done.add((name, tuple(chunk)))
    for (name, n, key), total in sums.items():
        score(name, n, "ensemble", key, total.float() / counts[(name, n, key)])
    if retrieval:
        _retrieval(manifest, source, extractor, predictors, loader, score, pairs)
    return dict(results=results, areas=areas, pairs=pairs,
                reference_names=list(source.readers) + ["consensus"])


def _retrieval(manifest, source, extractor, predictors, loader, score, pairs):
    """Supports = nearest pool pairs by frozen CLS-difference cosine, excluding the query's group."""
    pool = sorted({key for entry in manifest["entries"] for key in entry["support"]})
    group_of = {key: entry["group"] for entry in manifest["entries"] for key in entry["queries"]}
    keys = sorted(set(pool) | set(pairs))
    tokens = extractor.pairs(keys, loader)
    difference = torch.nn.functional.normalize(tokens[3] - tokens[2], dim=-1)
    position = {key: index for index, key in enumerate(keys)}
    for n in manifest["meta"]["n_values"]:
        for key in pairs:
            allowed = [item for item in pool if item != key and group_of.get(item) != group_of[key]]
            if len(allowed) < n:
                continue
            similarity = difference[[position[item] for item in allowed]] @ difference[position[key]]
            chosen = [allowed[index] for index in similarity.argsort(descending=True)[:n].tolist()]
            bundle = SupportBundle(chosen, extractor, loader)
            query = extractor.pairs([key], loader)
            for name, predictor in predictors.items():
                if predictor.uses_support and not isinstance(predictor, FineTunePredictor):
                    score(name, n, "retrieval", key,
                          predict(predictor, query, bundle.tokens, bundle.labels, bundle.targets)[0])


def _reader_f1(terms, readers, weights=None):
    """Mean-over-readers F1 per metric; terms [pairs, refs, 2, 6]; weights [B, pairs] for bootstrap."""
    collapsed = terms.sum(2)  # [pairs, refs, 6]
    summed = collapsed.sum(0) if weights is None else np.einsum("bp,prk->brk", weights, collapsed)
    values = vector_scores(summed[..., readers, :])
    return {metric: np.nanmean(values[metric], axis=-1) for metric in METRICS}


def summarize_icu(run, bootstrap=10000, seed=20261127):
    """Per (predictor, n): draw mean/SD, ensemble, retrieval; 95% pair-bootstrap CI of draw means."""
    readers = [index for index, name in enumerate(run["reference_names"]) if name != "consensus"]
    pairs = run["pairs"]
    weights = None
    if bootstrap:
        samples = np.random.default_rng(seed).integers(0, len(pairs), size=(bootstrap, len(pairs)))
        weights = np.stack([np.bincount(sample, minlength=len(pairs)) for sample in samples]).astype(np.float64)
    output = {}
    for predictor, by_key in run["results"].items():
        grouped = defaultdict(dict)
        for (n, draw), per_pair in by_key.items():
            grouped[n][draw] = per_pair
        for n, draws in grouped.items():
            row = {}
            regular = [draws[draw] for draw in sorted(draw for draw in draws if not isinstance(draw, str))]
            special = {draw: [draws[draw]] for draw in draws if isinstance(draw, str)}
            for label, selected in ([("draws", regular)] if regular else []) + list(special.items()):
                if any(len(per_pair) != len(pairs) for per_pair in selected):
                    raise ValueError(f"{predictor} n={n} {label}: not every ICU pair was scored")
                stacks = [np.stack([per_pair[key] for key in pairs]) for per_pair in selected]
                point = [_reader_f1(terms, readers) for terms in stacks]
                consensus = [vector_scores(terms[:, -1].sum((0, 1))) for terms in stacks]
                summary = {}
                for metric in METRICS:
                    values = [float(item[metric]) for item in point]
                    summary[metric] = dict(mean=float(np.mean(values)),
                                           sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                           consensus=float(np.nanmean([item[metric] for item in consensus])))
                if weights is not None:
                    boot = [_reader_f1(terms, readers, weights) for terms in stacks]
                    for metric in METRICS:
                        distribution = np.mean([item[metric] for item in boot], axis=0)
                        summary[metric]["ci95"] = [float(np.quantile(distribution, .025)),
                                                   float(np.quantile(distribution, .975))]
                summary["draws"] = len(selected)
                row[label] = summary
            output[f"{predictor}|n{n}"] = row
    return output


def paired_difference(run, first, second, n, metric="one_to_many", bootstrap=10000, seed=20261127):
    """Bootstrap CI of (first at n) - (second at n, or at n = 0 if it is support-free)."""
    readers = [index for index, name in enumerate(run["reference_names"]) if name != "consensus"]
    pairs = run["pairs"]

    def stacks(name, size):
        keys = [key for key in run["results"][name] if key[0] == size and not isinstance(key[1], str)]
        if not keys:
            keys = [key for key in run["results"][name] if key[0] == 0 and not isinstance(key[1], str)]
        return [np.stack([run["results"][name][key][pair] for pair in pairs]) for key in keys]

    samples = np.random.default_rng(seed).integers(0, len(pairs), size=(bootstrap, len(pairs)))
    weights = np.stack([np.bincount(sample, minlength=len(pairs)) for sample in samples]).astype(np.float64)

    def value(terms_list, sample_weights=None):
        return np.mean([_reader_f1(terms, readers, sample_weights)[metric] for terms in terms_list], axis=0)

    a, b = stacks(first, n), stacks(second, n)
    boot = value(a, weights) - value(b, weights)
    return dict(difference=float(value(a) - value(b)),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
