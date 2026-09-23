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
import os
import sys

import torch
from torch.utils.data import DataLoader

_FCD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FCD_ROOT not in sys.path:
    sys.path.insert(0, _FCD_ROOT)

import constants as C  # noqa: E402
from models.backbone import FrozenCXRBackbone  # noqa: E402
from datasets.pair_dataset import LongitudinalPairDataset  # noqa: E402


def _build_backbone(device: str) -> FrozenCXRBackbone:
    return FrozenCXRBackbone(
        name=C.BACKBONE,
        model_id=getattr(C, "RAD_DINO_MODEL", "microsoft/rad-dino"),
        backbone_img_size=C.BACKBONE_IMG_SIZE,
        num_patch_tokens=C.NUM_PATCH_TOKENS,
        last_n_layers=C.LAST_N_LAYERS,
        freeze=True,
    ).to(device).eval()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-o", "--dataset_root", required=True, help="Folder with manifest[_split].jsonl + pair tree.")
    p.add_argument("--cache_dir", default=None, help="Where to write feat.pt files (default constants.FEATURE_CACHE_DIR).")
    p.add_argument("--split", default=None, help="Cache only this split (default: all).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--limit", type=int, default=0, help="Cache at most N pairs (0 = all); for smoke tests.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = os.path.abspath(args.cache_dir or C.FEATURE_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)

    ds = LongitudinalPairDataset(args.dataset_root, split=args.split, img_size=C.IMG_SIZE)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    backbone = _build_backbone(args.device)

    print(f"[cache] {len(ds)} pairs -> {cache_dir} on {args.device}", flush=True)
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
            torch.save(
                {
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
                },
                out_path,
            )
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
