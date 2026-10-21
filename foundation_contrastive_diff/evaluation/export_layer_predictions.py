"""Export selected synthetic validation maps from the matched layer-depth heads."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from pathlib import Path


LAYERS = (6, 9, 12)
ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = (
    "anomaly_type", "is_pathology", "change_group_id",
    "positive_reference", "negative_reference", "signed_reference",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_diagnostics(run):
    summary = json.loads((run / "summary.json").read_text())
    rows = {}
    for layer in LAYERS:
        epoch = summary[str(layer)]["best_epoch"]
        stem = run / f"layer_{layer:02d}" / "diagnostics" / f"epoch_{epoch:04d}_val"
        report = json.loads(stem.with_suffix(".json").read_text())
        if (report["invalid_pairs"] or report["primary_prediction_threshold"] != 0.5
                or report["primary_gt_threshold"] != 0.05
                or report["epoch"] != epoch or report["metadata"]["backbone_layer"] != layer):
            raise ValueError(f"Invalid diagnostic metadata for block {layer}")
        with gzip.open(str(stem) + ".csv.gz", "rt", newline="") as stream:
            records = list(csv.DictReader(stream))
        rows[layer] = {row["pair_id"]: row for row in records}
        if (not records or len(rows[layer]) != len(records) or "" in rows[layer]
                or len(records) != report["groups"]["overall"]["pairs"]):
            raise ValueError(f"Missing or duplicate diagnostic pairs for block {layer}")
        for row in records:
            for field in ("positive_reference", "negative_reference", "signed_reference",
                          "predicted_area", "signed_dice"):
                if row[field] == "" and field == "signed_dice" and float(row["signed_reference"]) == 0:
                    continue
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Nonfinite {field}: block {layer}, {row['pair_id']}")
    first = rows[6]
    for layer in LAYERS:
        if rows[layer].keys() != first.keys():
            raise ValueError(f"Layer pair IDs differ for block {layer}")
        for pair, row in first.items():
            if any(rows[layer][pair][key] != row[key] for key in IDENTITY_FIELDS):
                raise ValueError(f"Layer reference mismatch: block {layer}, {pair}")
    return summary, rows


def select_cases(rows):
    first = rows[6]
    active = [pair for pair, row in first.items()
              if int(row["is_pathology"]) and float(row["signed_reference"]) > 0]
    if not active:
        raise ValueError("No pathology pairs with active reference")
    selected, used_groups = [], set()

    def take(pair, reason):
        identity = first[pair]["change_group_id"] or pair
        if identity in used_groups:
            return False
        used_groups.add(identity)
        selected.append(dict(
            pair_id=pair, anomaly_type=first[pair]["anomaly_type"],
            change_group_id=first[pair]["change_group_id"], selection=reason,
            diagnostic_dice={
                str(layer): float(rows[layer][pair]["signed_dice"])
                if rows[layer][pair]["signed_dice"] else None for layer in LAYERS},
        ))
        return True

    generator = random.Random(42)
    for anomaly in sorted({first[pair]["anomaly_type"] for pair in active}):
        group = [pair for pair in active if first[pair]["anomaly_type"] == anomaly]
        for layer in LAYERS:
            def lead(pair):
                return float(rows[layer][pair]["signed_dice"]) - max(
                    float(rows[other][pair]["signed_dice"]) for other in LAYERS if other != layer)

            count = 0
            for pair in sorted(group, key=lambda pair: (-lead(pair), pair)):
                if lead(pair) > 0.05 and take(pair, f"block_{layer}_leads_both_by_more_than_0.05"):
                    count += 1
                    if count == 2:
                        break
        controls = sorted(group)
        generator.shuffle(controls)
        for pair in controls:
            if take(pair, "seeded_pathology_control"):
                break
    nuisance = [pair for pair, row in first.items() if not int(row["is_pathology"])]
    count = 0
    for pair in sorted(nuisance, key=lambda pair: (
            -max(float(rows[layer][pair]["predicted_area"]) for layer in LAYERS), pair)):
        if take(pair, "high_nuisance_predicted_area"):
            count += 1
            if count == 4:
                break
    if not selected or len(selected) > 32:
        raise ValueError("Unexpected export size; expected at most 32 cases")
    return selected


def validate_checkpoint(state, selection, layer):
    config = state["run_config"]
    if (config["backbone_layer"] != layer
            or config["feature_normalization"] != "final_layernorm"
            or not config["no_cache"]
            or state["epoch"] != selection["best_epoch"]
            or not math.isclose(state["best_score"], selection["best_score"],
                                rel_tol=0, abs_tol=1e-12)):
        raise ValueError(f"Checkpoint metadata mismatch for block {layer}")
    return {key: value for key, value in config.items() if key != "backbone_layer"}


def save_case(out, case, arrays, rows, image_size):
    import numpy as np

    expected_keys = {"gt_diff", *(f"pred_layer_{layer:02d}" for layer in LAYERS)}
    if set(arrays) != expected_keys or any(
            array.shape != (image_size, image_size) or array.dtype != np.float32
            or not np.isfinite(array).all() or (np.abs(array) > 1.000001).any()
            for array in arrays.values()):
        raise ValueError(f"Invalid prediction maps: {case['pair_id']}")
    reference = arrays["gt_diff"]
    ref_positive, ref_negative = reference > 0.05, reference < -0.05
    ref_count = int(ref_positive.sum() + ref_negative.sum())
    pair = case["pair_id"]
    if (int(ref_positive.sum()) != int(float(rows[6][pair]["positive_reference"]))
            or int(ref_negative.sum()) != int(float(rows[6][pair]["negative_reference"]))):
        raise ValueError(f"Export GT differs from validation diagnostics: {pair}")
    actual_dice, actual_area = {}, {}
    for layer in LAYERS:
        prediction = arrays[f"pred_layer_{layer:02d}"]
        positive, negative = prediction > 0.5, prediction < -0.5
        predicted = int(positive.sum() + negative.sum())
        actual_area[str(layer)] = predicted / prediction.size
        if abs(actual_area[str(layer)] - float(rows[layer][pair]["predicted_area"])) > 0.001:
            raise ValueError(f"Predicted area does not reproduce diagnostics: block {layer}, {pair}")
        if ref_count:
            overlap = int((positive & ref_positive).sum() + (negative & ref_negative).sum())
            dice = 2 * overlap / (predicted + ref_count)
            expected = float(rows[layer][pair]["signed_dice"])
            if abs(dice - expected) > 0.001:
                raise ValueError(f"Prediction does not reproduce diagnostics: block {layer}, {pair}")
            actual_dice[str(layer)] = dice
        else:
            actual_dice[str(layer)] = None
    filename = hashlib.sha256(pair.encode()).hexdigest()[:24] + ".npz"
    np.savez_compressed(out / filename, **arrays)
    case.update(file=filename, sha256=file_sha256(out / filename),
                export_signed_dice=actual_dice, export_predicted_area=actual_area)


def export_maps(run, out):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    import constants as C
    from datasets import LongitudinalPairDataset
    from models import FrozenCXRBackbone, build_change_head, signed_change_map

    if out.exists():
        raise ValueError(f"Use a new export directory: {out}")
    summary, rows = load_diagnostics(run)
    selected = select_cases(rows)
    provenance = json.loads((run / "provenance.json").read_text())
    for name, expected in provenance["code_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Source differs from training: {name}; do not bypass provenance checks")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no export was started")
    heads, checkpoints, configs, specs = {}, {}, {}, {}
    for layer in LAYERS:
        path = run / f"layer_{layer:02d}" / "best.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        configs[layer] = validate_checkpoint(state, summary[str(layer)], layer)
        specs[layer] = state["head_spec"]
        heads[layer] = build_change_head(state["head_spec"], "cuda").eval()
        heads[layer].load_state_dict(state["head"])
        checkpoints[str(layer)] = dict(epoch=state["epoch"], sha256=file_sha256(path))
    if not configs[6] == configs[9] == configs[12] or not specs[6] == specs[9] == specs[12]:
        raise ValueError("Layer configurations or head architectures differ")
    config = configs[6]
    manifest = Path(config["dataset_root"]) / "manifest_split.jsonl"
    if file_sha256(manifest) != provenance["manifest_sha256"]:
        raise ValueError("Dataset manifest hash differs from training")
    dataset = LongitudinalPairDataset(
        config["dataset_root"], split="val", img_size=C.IMG_SIZE,
        rotation_k=config["rotation_k"], flip_horizontal=config["flip_horizontal"])
    indices = {row["pair_id"]: index for index, row in enumerate(dataset.records)}
    if len(indices) != len(dataset.records) or indices.keys() != rows[6].keys():
        raise ValueError("Dataset does not match the diagnostic validation split")
    backbone = FrozenCXRBackbone(
        name=C.BACKBONE, model_id=provenance["model_id"],
        backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
        last_n_layers=1, freeze=True).to("cuda").eval()
    if getattr(backbone.backbone.config, "_commit_hash", None) != provenance["backbone_revision"]:
        raise ValueError("Cached backbone revision differs from training")
    loader = DataLoader(
        Subset(dataset, [indices[row["pair_id"]] for row in selected]),
        batch_size=2, shuffle=False, num_workers=0)
    out.mkdir(parents=True, exist_ok=False)
    lookup = {row["pair_id"]: row for row in selected}
    with torch.inference_mode():
        for batch in loader:
            prior = backbone.forward_layers(batch["img_prior"].to("cuda"), list(LAYERS))
            current = backbone.forward_layers(batch["img_curr"].to("cuda"), list(LAYERS))
            predictions = {}
            for layer in LAYERS:
                score, _ = heads[layer](
                    prior[layer]["patch_tokens"], current[layer]["patch_tokens"],
                    prior[layer]["cls_token"], current[layer]["cls_token"])
                predictions[layer] = signed_change_map(score).float().cpu().numpy()[:, 0]
            gt = batch["gt_diff"].float().numpy()[:, 0]
            for index, pair in enumerate(batch["pair_id"]):
                arrays = {"gt_diff": gt[index], **{
                    f"pred_layer_{layer:02d}": predictions[layer][index] for layer in LAYERS}}
                save_case(out, lookup[pair], arrays, rows, C.IMG_SIZE)
                print(f"[export] {pair}", flush=True)
    if any("file" not in row for row in selected):
        raise ValueError("Incomplete export")
    metadata = dict(
        cases=selected, checkpoints=checkpoints, training_provenance=provenance,
        exporter_sha256=file_sha256(Path(__file__)), run_config=config, split="val",
        dtype="float32", raw_unfiltered_scores=True,
        orientation="upright dataset coordinates; identical for GT and all predictions",
        diagnostic_prediction_threshold=0.5, diagnostic_gt_threshold=0.05,
        reproduction_absolute_tolerance=0.001,
        selection_note="Enriched case review, not representative performance or distinct patients",
        contents="Predictions and synthetic GT only; no source images, model weights, or caches",
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
    )
    (out / "manifest.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in out.iterdir())
    if size > 100 * 1024 ** 2:
        raise ValueError(f"Export exceeds 100 MiB; not staged or pushed: {size}")
    print(f"Export complete: {len(selected)} cases, {size / 1024**2:.1f} MiB in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    export_maps(args.checkpoints, args.out_dir)


if __name__ == "__main__":
    main()
