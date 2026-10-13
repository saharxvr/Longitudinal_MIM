"""
Cache a SUBSET of frozen backbone features for the cheap backbone-fitness screen.

Writes feat.pt per pair (same schema as training/cache_features.py) for a small deterministic
subset of train+val, so the existing evaluation/feature_change_correlation.py and
evaluation/linear_probe.py run unchanged against --cache_dir (pass --grid for non-37 grids).

Only the subset pairs get a feat.pt, so CachedPairDataset naturally serves just the subset.

Run (on the GPU node):
    python -m screening.cache_backbone_subset -o $DATA \
        --backbone dinov2_imagenet --cache_dir /cs/.../screen/dinov2 \
        --n_train 2500 --n_val 1500
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from datasets.pair_dataset import LongitudinalPairDataset
from screening.backbones import build_backbone


def _subset(ds, n, seed=0):
    if n <= 0 or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


@torch.no_grad()
def cache_split(bb, root, split, n, cache_dir, device):
    ds = LongitudinalPairDataset(root, split=split)
    sub = _subset(ds, n)
    loader = DataLoader(sub, batch_size=4, shuffle=False, num_workers=4)
    done = 0
    for batch in loader:
        op = bb(batch["img_prior"].to(device))
        oc = bb(batch["img_curr"].to(device))
        pp, cp = op["patch_tokens"].cpu(), op["cls_token"].cpu()
        pc, cc = oc["patch_tokens"].cpu(), oc["cls_token"].cpu()
        bs = pp.shape[0]
        for b in range(bs):
            pid = batch["pair_id"][b]
            out = os.path.join(cache_dir, pid.replace("/", os.sep), "feat.pt")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save({
                "p_prior": pp[b].half(), "cls_prior": cp[b].half(),
                "p_curr": pc[b].half(), "cls_curr": cc[b].half(),
                "gt_diff": batch["gt_diff"][b].half(),
                "anomaly_type": int(batch["anomaly_type"][b]),
                "direction": int(batch["direction"][b]),
                "is_pathology": int(batch["is_pathology"][b]),
                "change_group_id": batch["change_group_id"][b],
                "pair_id": pid,
            }, out)
        done += bs
        if done % 200 < bs:
            print(f"  [{split}] cached {done}/{len(sub)}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--dataset_root", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--n_val", type=int, default=1500)
    args = ap.parse_args()

    device = C.DEVICE
    bb = build_backbone(args.backbone).to(device)
    print(f"[screen-cache] backbone={args.backbone} grid={bb.grid} dim={bb.dim} "
          f"-> {args.cache_dir}", flush=True)
    nt = cache_split(bb, args.dataset_root, "train", args.n_train, args.cache_dir, device)
    nv = cache_split(bb, args.dataset_root, "val", args.n_val, args.cache_dir, device)
    print(f"[screen-cache] done. train {nt}  val {nv}  grid {bb.grid}\n"
          f"  next: python -m evaluation.feature_change_correlation -o {args.dataset_root} "
          f"--cache_dir {args.cache_dir} --split val --feat_last_k 0 --grid {bb.grid}\n"
          f"        python -m evaluation.linear_probe -o {args.dataset_root} "
          f"--cache_dir {args.cache_dir} --feat_last_k 0 --grid {bb.grid} --epochs 12", flush=True)


if __name__ == "__main__":
    main()
