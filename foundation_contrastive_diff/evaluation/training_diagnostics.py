"""Detached, streaming diagnostics; these values never contribute to gradients."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def gt_statistics(values, thresholds):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    result = dict(pixels=int(values.size), nonfinite=int((~finite).sum()))
    if not finite.all():
        return result
    magnitude = np.abs(values)
    active = magnitude > 0
    nonzero = magnitude[active]
    result.update(nonzero_pixels=int(active.sum()), positive_pixels=int((values > 0).sum()),
                  negative_pixels=int((values < 0).sum()), zero_fraction=float((~active).mean()),
                  peak=float(magnitude.max()), mean_abs=float(magnitude.mean()),
                  signed_mean=float(values.mean()))
    percentiles = (1, 5, 25, 50, 75, 95, 99)
    quantiles = np.percentile(nonzero, percentiles) if nonzero.size else [None] * len(percentiles)
    for percentile, quantile in zip(percentiles, quantiles):
        result[f"nonzero_abs_p{percentile:02d}"] = float(quantile) if quantile is not None else None
    for threshold in thresholds:
        kept = magnitude > threshold
        prefix = f"cutoff_{threshold:g}"
        result[f"{prefix}_pixels"] = int(kept.sum())
        result[f"{prefix}_area"] = float(kept.mean())
        result[f"{prefix}_support_retained"] = ratio(kept.sum(), active.sum())
        result[f"{prefix}_mass_retained"] = ratio(magnitude[kept].sum(), nonzero.sum())
        result[f"{prefix}_entirely_removed"] = int(bool(active.any()) and not kept.any())
    return result


def overlap_counts(prediction, reference):
    intersection = int((prediction & reference).sum())
    predicted = int(prediction.sum())
    actual = int(reference.sum())
    return dict(tp=intersection, fp=predicted - intersection, fn=actual - intersection,
                predicted=predicted, reference=actual)


def overlap_scores(counts):
    tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
    return dict(precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                dice=ratio(2 * tp, 2 * tp + fp + fn) if tp + fn else None,
                iou=ratio(tp, tp + fp + fn) if tp + fn else None)


def component_counts(prediction, reference):
    from scipy.ndimage import label

    predicted, pred_count = label(prediction, structure=np.ones((3, 3)))
    actual, ref_count = label(reference, structure=np.ones((3, 3)))
    corroborated = len(np.unique(predicted[reference & (predicted > 0)]))
    available_predictions = predicted.copy()
    matched = 0
    for component in range(1, ref_count + 1):
        overlaps, sizes = np.unique(available_predictions[actual == component], return_counts=True)
        nonzero = overlaps > 0
        if nonzero.any():
            chosen = overlaps[nonzero][np.argsort(sizes[nonzero])[::-1][0]]
            available_predictions[available_predictions == chosen] = 0
            matched += 1
    return dict(predicted=pred_count, reference=ref_count, matched=matched,
                corroborated=corroborated, unmatched=pred_count - corroborated,
                f1=ratio(2 * matched, pred_count + ref_count) if ref_count else None)


class PredictionDiagnostics:
    def __init__(self, prediction_thresholds, gt_thresholds, primary_prediction=0.5,
                 primary_gt=0.05, target_mode="continuous", target_threshold=0.05,
                 tau=0.02, pos_weight=10.0, background_weight=1.0, components=False):
        self.prediction_thresholds = sorted(set([primary_prediction, *prediction_thresholds]))
        self.gt_thresholds = sorted(set([primary_gt, *gt_thresholds]))
        self.primary_prediction = primary_prediction
        self.primary_gt = primary_gt
        self.target_mode = target_mode
        self.target_threshold = target_threshold
        self.tau = tau
        self.pos_weight = pos_weight
        self.background_weight = background_weight
        self.components = components
        self.rows = []
        self.sweeps = defaultdict(lambda: defaultdict(float))
        self.invalid_pairs = 0
        self.invalid_pair_ids = []

    def update(self, prediction, ground_truth, batch):
        import constants as constants

        predictions = prediction.detach().float().cpu().numpy()
        targets = ground_truth.detach().float().cpu().numpy()
        for index, (pred, raw) in enumerate(zip(predictions, targets)):
            pred, raw = pred.reshape(pred.shape[-2:]), raw.reshape(raw.shape[-2:])
            if not (np.isfinite(pred).all() and np.isfinite(raw).all()):
                self.invalid_pairs += 1
                self.invalid_pair_ids.append(batch.get("pair_id", [""] * len(predictions))[index])
                continue
            kind = int(batch["anomaly_type"][index]) if "anomaly_type" in batch else 0
            anomaly = constants.ANOMALY_TYPES[kind]
            pathology = int(batch["is_pathology"][index])
            group = "pathology" if pathology else "nuisance"
            pair_id = batch.get("pair_id", [""] * len(predictions))[index]
            change_group = batch.get("change_group_id", [""] * len(predictions))[index]
            has_positive, has_negative = bool((raw > self.primary_gt).any()), bool((raw < -self.primary_gt).any())
            direction = ("mixed" if has_positive and has_negative else "positive" if has_positive
                         else "negative" if has_negative else "empty")
            target = (np.where(np.abs(raw) > self.target_threshold, np.sign(raw), 0)
                      if self.target_mode == "signed_mask" else raw)
            foreground = np.abs(target) > self.tau
            weight = np.where(foreground, 1 + self.pos_weight, self.background_weight)
            difference = pred - target
            raw_support = raw != 0
            row = dict(pair_id=pair_id, change_group_id=change_group, anomaly_type=anomaly,
                       gt_direction=direction, is_pathology=pathology,
                       raw_gt_area=float(raw_support.mean()), raw_gt_peak=float(np.abs(raw).max()),
                       target_area=float(foreground.mean()),
                       gt_support_removed=ratio((raw_support & ~foreground).sum(), raw_support.sum()),
                       raw_mae=float(np.abs(pred - raw).mean()), raw_mse=float(((pred - raw) ** 2).mean()),
                       sign_gt_mae=float(np.abs(pred - np.sign(raw)).mean()),
                       sign_gt_mse=float(((pred - np.sign(raw)) ** 2).mean()),
                       target_mae=float(np.abs(difference).mean()), target_mse=float((difference ** 2).mean()),
                       weighted_l1=float((weight * np.abs(difference)).mean()),
                       weighted_l2=float((weight * difference ** 2).mean()),
                       foreground_l1_contribution=float((weight * np.abs(difference) * foreground).mean()),
                       background_l1_contribution=float((weight * np.abs(difference) * ~foreground).mean()),
                       foreground_l2_contribution=float((weight * difference ** 2 * foreground).mean()),
                       background_l2_contribution=float((weight * difference ** 2 * ~foreground).mean()),
                       foreground_mae=float(np.abs(difference[foreground]).mean()) if foreground.any() else None,
                       background_mae=float(np.abs(difference[~foreground]).mean()) if (~foreground).any() else None,
                       pred_abs_mean=float(np.abs(pred).mean()), pred_peak=float(np.abs(pred).max()),
                       pred_saturation_area=float((np.abs(pred) > 0.95).mean()),
                       sign_penalty=float(np.maximum(-pred * target, 0)[foreground].mean()) if foreground.any() else None)
            for percentile, value in zip((50, 95, 99), np.percentile(np.abs(pred), [50, 95, 99])):
                row[f"pred_abs_p{percentile}"] = float(value)
            soft_intersection = (np.abs(pred) * foreground).sum()
            row["soft_support_dice"] = (ratio(2 * soft_intersection, np.abs(pred).sum() + foreground.sum())
                                        if foreground.any() else None)
            for pred_threshold in self.prediction_thresholds:
                positive, negative = pred > pred_threshold, pred < -pred_threshold
                for gt_threshold in self.gt_thresholds:
                    ref_positive, ref_negative = raw > gt_threshold, raw < -gt_threshold
                    counts = {
                        "positive": overlap_counts(positive, ref_positive),
                        "negative": overlap_counts(negative, ref_negative),
                        "support": overlap_counts(positive | negative, ref_positive | ref_negative)}
                    counts["signed"] = {key: counts["positive"][key] + counts["negative"][key]
                                        for key in counts["positive"]}
                    primary = pred_threshold == self.primary_prediction and gt_threshold == self.primary_gt
                    for sign, values in counts.items():
                        scores = overlap_scores(values)
                        for label in ("overall", group, f"anomaly:{anomaly}", f"direction:{direction}"):
                            accumulator = self.sweeps[(label, pred_threshold, gt_threshold, sign)]
                            accumulator["pairs"] += 1
                            for name, value in values.items():
                                accumulator[name] += value
                            for name, value in scores.items():
                                if value is not None:
                                    accumulator[f"{name}_sum"] += value
                                    accumulator[f"{name}_n"] += 1
                            accumulator["active_reference_pairs"] += int(values["reference"] > 0)
                            accumulator["empty_prediction_pairs"] += int(values["predicted"] == 0)
                            accumulator["hit_pairs"] += int(values["tp"] > 0)
                            accumulator["fp_area_sum"] += values["fp"] / pred.size
                        if primary:
                            row.update({f"{sign}_{name}": value for name, value in {**values, **scores}.items()})
                    if primary:
                        row["wrong_sign_pixels"] = int(((positive & ref_negative) | (negative & ref_positive)).sum())
                        row["wrong_sign_fraction_on_overlap"] = ratio(
                            row["wrong_sign_pixels"], row["wrong_sign_pixels"] + counts["signed"]["tp"])
                        row["predicted_area"] = float((positive | negative).mean())
                        row["reference_area"] = float((ref_positive | ref_negative).mean())
                        if self.components:
                            for sign, predicted, reference in (("positive", positive, ref_positive),
                                                               ("negative", negative, ref_negative)):
                                row.update({f"component_{sign}_{key}": value for key, value in
                                            component_counts(predicted, reference).items()})
                            for name in ("predicted", "reference", "matched", "corroborated", "unmatched"):
                                row[f"component_all_{name}"] = (row[f"component_positive_{name}"] +
                                                                row[f"component_negative_{name}"])
                            row["component_all_f1"] = (ratio(2 * row["component_all_matched"],
                                                            row["component_all_predicted"] + row["component_all_reference"])
                                                       if row["component_all_reference"] else None)
            self.rows.append(row)

    def summary(self):
        groups = {}
        categories = {f"anomaly:{row['anomaly_type']}" for row in self.rows}
        categories.update(f"direction:{row['gt_direction']}" for row in self.rows)
        for name in ("overall", "pathology", "nuisance", *sorted(categories)):
            selected = [row for row in self.rows if name == "overall" or
                        name == ("pathology" if row["is_pathology"] else "nuisance") or
                        name == f"anomaly:{row['anomaly_type']}" or name == f"direction:{row['gt_direction']}"]
            values = dict(pairs=len(selected), unique_pairs=len({row["pair_id"] for row in selected}),
                          unique_change_groups=len({row["change_group_id"] for row in selected if row["change_group_id"]}))
            if selected:
                for key in selected[0]:
                    if key in ("pair_id", "change_group_id", "anomaly_type", "gt_direction", "is_pathology"):
                        continue
                    valid = [row[key] for row in selected if row[key] is not None]
                    values[key] = dict(mean=float(np.mean(valid)) if valid else None, n=len(valid))
                area = [row["predicted_area"] for row in selected]
                values["predicted_area_p95"] = float(np.percentile(area, 95))
                values["predicted_area_max"] = max(area)
                if self.components:
                    for sign in ("positive", "negative", "all"):
                        count = lambda key: sum(row[f"component_{sign}_{key}"] for row in selected)
                        values[f"component_{sign}_hmdr"] = ratio(count("corroborated"), count("predicted"))
                        values[f"component_{sign}_udpp"] = count("unmatched") / len(selected)
                        values[f"component_{sign}_pooled_f1"] = ratio(2 * count("matched"), count("predicted") + count("reference"))
            groups[name] = values
        return dict(groups=groups, invalid_pairs=self.invalid_pairs,
                    invalid_pair_ids=self.invalid_pair_ids,
                    primary_prediction_threshold=self.primary_prediction, primary_gt_threshold=self.primary_gt,
                    components_enabled=self.components,
                    component_reference="synthetic GT, reference-first largest-overlap greedy 8-connected matching; no boost",
                    note="Train statistics are online sampled batches, not a fixed-model full-train evaluation. Undefined ratios are null.")

    def sweep_rows(self):
        rows = []
        for (group, pred_threshold, gt_threshold, sign), values in sorted(self.sweeps.items()):
            row = dict(group=group, prediction_threshold=pred_threshold, gt_threshold=gt_threshold, sign=sign,
                       **{key: value for key, value in values.items() if not key.endswith(("_sum", "_n"))})
            row.update({f"micro_{key}": value for key, value in overlap_scores(values).items()})
            for name in ("precision", "recall", "dice", "iou"):
                row[f"macro_{name}"] = ratio(values[f"{name}_sum"], values[f"{name}_n"])
                row[f"{name}_valid_pairs"] = values[f"{name}_n"]
            row["mean_fp_area"] = ratio(values["fp_area_sum"], values["pairs"])
            rows.append(row)
        return rows

    def save(self, directory, epoch, split, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"epoch_{epoch:04d}_{split}"
        report = dict(epoch=epoch, split=split, **self.summary(), metadata=metadata or {})
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for suffix, rows in ((".csv.gz", self.rows), ("_thresholds.csv.gz", self.sweep_rows())):
            if rows:
                with gzip.open(str(stem) + suffix, "wt", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        return report