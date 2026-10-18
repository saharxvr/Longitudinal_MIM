"""
RQ1 step 2 — precompute frozen backbone features for every pair (one-time).

Because the backbone is frozen, we run it over each (prior, current) pair ONCE and cache
the patch tokens + [CLS] + the GT diff + labels to disk. Head training then reads only the
cache (no backbone, no NIfTI) -> fast, fits any GPU, and decouples from RAD-DINO.

Cache layout (one file per pair, mirroring pair_id):
    <cache_dir>/<case>/pair<i>/variant<v>/feat.pt   (torch.save, fp16 tokens)
      {
        'p_prior':  [N, D] fp16,  'cls_prior': [D] fp16,
        'p_curr':   [N, D] fp16,  'cls_curr':  [D] fp16,
        'gt_diff':  [1, H, W] fp16,
        'anomaly_type': int, 'direction': int, 'is_pathology': int,
        'change_group_id': str, 'pair_id': str,
      }

Resumable: existing feat.pt files are skipped.

Usage (on the cluster GPU):
    python training/cache_features.py -o /cs/.../fcd_train --cache_dir /cs/.../fcd_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_FCD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FCD_ROOT not in sys.path:
    sys.path.insert(0, _FCD_ROOT)

import constants as C  # noqa: E402
from models.backbone import FrozenCXRBackbone  # noqa: E402
from datasets.pair_dataset import LongitudinalPairDataset, _read_manifest  # noqa: E402


def slice_last_layer_cache(dataset_root, source_dir, cache_dir, split=None, limit=0):
    """Write compact final-layer slices without rerunning or modifying the encoder cache."""
    source_dir, cache_dir = Path(source_dir).resolve(), Path(cache_dir).resolve()
    if source_dir == cache_dir:
        raise ValueError("Source and destination cache directories must differ")
    records = _read_manifest(dataset_root)
    if split is not None:
        records = [record for record in records if record.get("split") == split]
    if not records:
        raise ValueError("No manifest records selected")
    if limit:
        records = records[:limit]
    written, skipped = 0, 0
    for index, record in enumerate(records, 1):
        source_path = source_dir / record["pair_id"] / "feat.pt"
        cached = torch.load(source_path, map_location="cpu", weights_only=True)
        for key in ("p_prior", "p_curr"):
            tensor = cached[key]
            if tuple(tensor.shape) != (C.NUM_PATCH_TOKENS, 4 * C.BACKBONE_DIM):
                raise ValueError(f"Expected last-four {key} in {source_path}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite {key} in {source_path}")
            cached[key] = tensor[..., -C.BACKBONE_DIM:].clone().contiguous()
        cached["_feature_source"] = dict(kind="last4_final_slice", path=str(source_path))
        destination = cache_dir / record["pair_id"] / "feat.pt"
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            for key, value in cached.items():
                actual = existing.get(key)
                matches = (isinstance(actual, torch.Tensor) and torch.equal(actual, value)
                           if isinstance(value, torch.Tensor) else actual == value)
                if not matches:
                    raise ValueError(f"Existing cache differs at {key}: {destination}")
            skipped += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                torch.save(cached, temporary_path)
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            written += 1
        if index % 250 == 0 or index == len(records):
            print(f"[cache slice] {index}/{len(records)} written={written} verified-existing={skipped}", flush=True)
    return {"written": written, "skipped": skipped}


def _build_backbone(device: str, last_n_layers: int) -> FrozenCXRBackbone:
    return FrozenCXRBackbone(
        name=C.BACKBONE,
        model_id=getattr(C, "RAD_DINO_MODEL", "microsoft/rad-dino"),
        backbone_img_size=C.BACKBONE_IMG_SIZE,
        num_patch_tokens=C.NUM_PATCH_TOKENS,
        last_n_layers=last_n_layers,
        freeze=True,
    ).to(device).eval()


def prepare_native_cache(args):
    cache_dir = Path(args.cache_dir or C.FEATURE_CACHE_DIR).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.dataset_root) / "manifest_split.jsonl"
    if not manifest.is_file():
        manifest = Path(args.dataset_root) / "manifest.jsonl"
    expected = dict(backbone=C.RAD_DINO_MODEL, last_n_layers=args.last_n_layers,
                    rotation_k=args.rotation_k, image_size=C.IMG_SIZE,
                    flip_horizontal=args.flip_horizontal,
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    metadata = cache_dir / "cache_config.json"
    if metadata.is_file():
        actual = json.loads(metadata.read_text())
        actual.setdefault("flip_horizontal", False)
        if actual != expected:
            raise ValueError("Cache configuration differs; use a new cache directory")
    elif any(cache_dir.iterdir()):
        if args.rotation_k or args.flip_horizontal:
            raise ValueError("Rotated cache requires a new empty directory or matching cache_config.json")
    else:
        metadata.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    return cache_dir


def save_input_preview(dataset, path, target_threshold=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices, seen = [], set()
    for index, record in enumerate(getattr(dataset, "records", [])):
        case = record["pair_id"].split("/")[0]
        if case not in seen:
            indices.append(index)
            seen.add(case)
        if len(indices) == 3:
            break
    if not indices:
        indices = list(range(min(3, len(dataset))))
    count = len(indices)
    if not count:
        raise ValueError("Cannot preview an empty dataset")
    figure, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for row, index in enumerate(indices):
        sample = dataset[index]
        gt = sample["gt_diff"][0]
        mask = torch.where(gt.abs() > target_threshold, gt.sign(), torch.zeros_like(gt))
        for axis, values, title in zip(axes[row],
                                      (sample["img_prior"][0], sample["img_curr"][0], gt, mask),
                                      ("Prior encoder input", "Current encoder input", "Continuous GT",
                                       f"Signed GT mask @{target_threshold:g}")):
            axis.imshow(values, cmap="gray" if "input" in title else "bwr",
                        vmin=0 if "input" in title else -1, vmax=1)
            axis.set_title(title)
            axis.axis("off")
    figure.suptitle(f"Dataset inputs before frozen-backbone resize/normalization; rotation_k={sample['rotation_k']}; "
                   f"horizontal flip={sample.get('flip_horizontal', False)}")
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print(f"[cache] input preview: {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-o", "--dataset_root", required=True, help="Folder with manifest[_split].jsonl + pair tree.")
    p.add_argument("--cache_dir", default=None, help="Where to write feat.pt files (default constants.FEATURE_CACHE_DIR).")
    p.add_argument("--split", default=None, help="Cache only this split (default: all).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--last_n_layers", type=int, default=C.LAST_N_LAYERS,
                   help="concat the last N backbone layers (4 for the DPT/multi-scale cache)")
    p.add_argument("--rotation_k", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--flip_horizontal", action="store_true")
    p.add_argument("--preview_path", default="", help="Save dataset input/GT preview before feature extraction")
    p.add_argument("--preview_threshold", type=float, default=0.05)
    p.add_argument("--preview_only", action="store_true", help="Save preview without loading the backbone")
    p.add_argument("--slice_last_layer_from", default="",
                   help="CPU-only: copy final 768 dims from this last-four cache; requires --last_n_layers 1")
    p.add_argument("--limit", type=int, default=0, help="Cache at most N pairs (0 = all); for smoke tests.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.slice_last_layer_from:
        if args.rotation_k or args.flip_horizontal:
            raise SystemExit("Rotated scans require native feature extraction, not cache slicing")
        if not args.cache_dir or args.last_n_layers != 1:
            raise SystemExit("Cache slicing requires explicit --cache_dir and --last_n_layers 1")
        slice_last_layer_cache(args.dataset_root, args.slice_last_layer_from,
                               args.cache_dir, args.split, args.limit)
        return 0
    cache_dir = str(prepare_native_cache(args))

    ds = LongitudinalPairDataset(args.dataset_root, split=args.split, img_size=C.IMG_SIZE,
                                 rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
    if args.preview_path or args.preview_only:
        save_input_preview(ds, args.preview_path or Path(cache_dir) / "input_preview.png", args.preview_threshold)
    if args.preview_only:
        return 0
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    backbone = _build_backbone(args.device, args.last_n_layers)

    print(f"[cache] {len(ds)} pairs -> {cache_dir} on {args.device}; layers={args.last_n_layers}, rotation_k={args.rotation_k}", flush=True)
    done = 0
    for batch in loader:
        pair_ids = batch["pair_id"]
        # Skip a whole batch only if every file already exists.
        out_paths = [os.path.join(cache_dir, pid.replace("/", os.sep), "feat.pt") for pid in pair_ids]
        if all(os.path.isfile(p) for p in out_paths):
            done += len(pair_ids)
            continue

        with torch.no_grad():
            fp = backbone(batch["img_prior"].to(args.device))
            fc = backbone(batch["img_curr"].to(args.device))

        for i, (pid, out_path) in enumerate(zip(pair_ids, out_paths)):
            if os.path.isfile(out_path):
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            payload = {
                    "p_prior": fp["patch_tokens"][i].half().cpu(),
                    "cls_prior": fp["cls_token"][i].half().cpu(),
                    "p_curr": fc["patch_tokens"][i].half().cpu(),
                    "cls_curr": fc["cls_token"][i].half().cpu(),
                    "gt_diff": batch["gt_diff"][i].half().cpu(),
                    "anomaly_type": int(batch["anomaly_type"][i]),
                    "direction": int(batch["direction"][i]),
                    "is_pathology": int(batch["is_pathology"][i]),
                    "change_group_id": batch["change_group_id"][i],
                    "pair_id": pid,
                    "rotation_k": args.rotation_k,
                    "flip_horizontal": args.flip_horizontal,
                }
            with tempfile.NamedTemporaryFile(dir=os.path.dirname(out_path), suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                torch.save(payload, temporary_path)
                os.replace(temporary_path, out_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            done += 1
            if args.limit and done >= args.limit:
                print(f"[cache] hit --limit {args.limit}; stopping.", flush=True)
                return 0

        if done % (args.batch_size * 25) < args.batch_size:
            print(f"[cache] {done}/{len(ds)}", flush=True)

    print(f"[cache] done: {done} pairs cached in {cache_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
