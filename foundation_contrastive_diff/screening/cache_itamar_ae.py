"""
Standalone subset-cache for Itamar's reconstruction autoencoder (backbone-fitness screen).

Itamar's model tree (`refactored_modules.longitudinal_v2`, legacy `models.py`) re-defines
`constants`/`models`, which clash with this study's modules. So this script is deliberately
standalone: it imports NO foundation_contrastive_diff code. Run it from python_files/ so that
`import constants`, `import models`, `refactored_modules` all resolve to Itamar's project.

It reads the SAME fcd manifest + NIfTIs directly, encodes each image with Itamar's frozen
encoder, and writes feat.pt in the schema the fcd probe/correlation scripts expect:
    encode: model.backbone.encoder(x) -> encoded_bn -> [B,640,32,32]
            -> tokens [B,1024,640], grid 32, CLS = global avg-pool.

Then run the fcd probe/correlation against the produced cache with --grid 32 --feat_last_k 0.

Usage (from the cluster, CWD = python_files/, in Itamar's env or one with his deps):
    cd /cs/.../python_files
    python /cs/.../foundation_contrastive_diff/screening/cache_itamar_ae.py \
        --dataset_root $DATA --cache_dir /cs/.../screen/itamar_ae \
        --ckpt /cs/casmip/itamar_sab/.../Checkpoint_id32_Epoch8_...L1L2_GN.pt \
        --n_train 2500 --n_val 1500

CAVEAT: feeds images as norm01 [0,1] 512x512 1-channel. If Itamar's encoder expects a
different normalization, the ABSOLUTE scores shift but the RELATIVE ranking vs other backbones
still holds. Validate on a handful of pairs first.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def _add_code_root(code_root: str = "") -> str:
    """Put Itamar's project (the dir CONTAINING refactored_modules) on sys.path, ahead of all
    else, so `refactored_modules` / `constants` / `models` resolve to his code, not fcd's.
    Tries --code_root, then repo-relative python_files, then CWD."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))          # .../repos
    candidates = [c for c in (code_root,
                              os.path.join(repo, "python_files"),
                              os.getcwd()) if c]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "refactored_modules")):
            sys.path.insert(0, c)
            print(f"[itamar] code_root = {c}", flush=True)
            return c
    raise ModuleNotFoundError(
        "Cannot find 'refactored_modules'. Tried:\n  " + "\n  ".join(candidates) +
        "\nPass --code_root /path/to/dir_containing_refactored_modules")

# Must match foundation_contrastive_diff/constants.py ordering so labels line up downstream.
ANOMALY_TYPES = ['none', 'consolidation', 'pleural_effusion', 'pneumothorax', 'fluid_overload']
DIRECTION_TYPES = ['none', 'appearance', 'disappearance']


def _read_manifest(root):
    for name in ("manifest_split.jsonl", "manifest.jsonl"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return [json.loads(l) for l in f if l.strip()]
    raise FileNotFoundError(f"no manifest in {root}")


def _load_nii_2d(path):
    import nibabel as nib
    return np.squeeze(np.asarray(nib.load(path).dataobj, dtype=np.float32))


def _to_chw(arr, size=512):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim != 2:
        t = t.reshape(t.shape[-2], t.shape[-1])
    t = t[None, None]
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0]


def _norm01(t):
    mn, mx = t.amin(), t.amax()
    return (t - mn) / (mx - mn + 1e-8)


def _abspath(root, rec, key, fallback):
    p = rec.get(key)
    if p and os.path.isabs(p) and os.path.isfile(p):
        return p
    return os.path.join(root, rec["pair_id"].replace("/", os.sep), fallback)


def _subset(recs, n, seed=0):
    if n <= 0 or n >= len(recs):
        return recs
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(recs), generator=g)[:n].tolist()
    return [recs[i] for i in idx]


def build_itamar(ckpt_path, device):
    from refactored_modules.longitudinal_v2 import LongitudinalMIMDeformV2
    model = LongitudinalMIMDeformV2()
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model_dict"] if isinstance(ck, dict) and "model_dict" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[itamar] loaded ckpt (missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def encode(model, img):
    """img [B,1,512,512] -> tokens [B,1024,640], cls [B,640] (grid 32)."""
    feats = model.encode_image(img)            # [B,640,32,32]
    b, d, h, w = feats.shape
    patch = feats.flatten(2).transpose(1, 2)   # [B, h*w, d]
    cls = feats.mean(dim=(2, 3))               # pseudo-CLS
    return patch, cls


@torch.no_grad()
def cache_split(model, root, recs, cache_dir, device):
    done = 0
    for rec in recs:
        pid = rec.get("pair_id", "")
        if not pid:
            continue
        out = os.path.join(cache_dir, pid.replace("/", os.sep), "feat.pt")
        if os.path.isfile(out):
            done += 1
            continue
        prior = _norm01(_to_chw(_load_nii_2d(_abspath(root, rec, "prior_path", "prior.nii.gz"))))
        curr = _norm01(_to_chw(_load_nii_2d(_abspath(root, rec, "current_path", "current.nii.gz"))))
        gt = _to_chw(_load_nii_2d(_abspath(root, rec, "diff_map_path", "diff_map.nii.gz"))).clamp(-1, 1)

        pp, cp = encode(model, prior[None].to(device))
        pc, cc = encode(model, curr[None].to(device))

        a = rec.get("effective_anomaly_type", rec.get("anomaly_type", "none"))
        dct = rec.get("direction", "none")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({
            "p_prior": pp[0].half().cpu(), "cls_prior": cp[0].half().cpu(),
            "p_curr": pc[0].half().cpu(), "cls_curr": cc[0].half().cpu(),
            "gt_diff": gt.half(),
            "anomaly_type": ANOMALY_TYPES.index(a) if a in ANOMALY_TYPES else 0,
            "direction": DIRECTION_TYPES.index(dct) if dct in DIRECTION_TYPES else 0,
            "is_pathology": int(bool(rec.get("realized_change",
                                             rec.get("effective_anomaly_type", "none") != "none"))),
            "change_group_id": rec.get("change_group_id", ""),
            "pair_id": pid,
        }, out)
        done += 1
        if done % 200 == 0:
            print(f"  cached {done}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--n_val", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code_root", default="",
                    help="dir containing Itamar's refactored_modules (default: repo python_files/)")
    args = ap.parse_args()

    _add_code_root(args.code_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_itamar(args.ckpt, device)
    recs = _read_manifest(args.dataset_root)
    tr = _subset([r for r in recs if r.get("split") == "train"], args.n_train, args.seed)
    va = _subset([r for r in recs if r.get("split") == "val"], args.n_val, args.seed)
    print(f"[itamar] caching train {len(tr)} + val {len(va)} -> {args.cache_dir}", flush=True)
    nt = cache_split(model, args.dataset_root, tr, args.cache_dir, device)
    nv = cache_split(model, args.dataset_root, va, args.cache_dir, device)
    print(f"[itamar] done. train {nt} val {nv}. grid 32.\n"
          f"  next (fcd env): python -m evaluation.feature_change_correlation -o {args.dataset_root} "
          f"--cache_dir {args.cache_dir} --split val --feat_last_k 0 --grid 32\n"
          f"                  python -m evaluation.linear_probe -o {args.dataset_root} "
          f"--cache_dir {args.cache_dir} --feat_last_k 0 --grid 32 --epochs 12", flush=True)


if __name__ == "__main__":
    main()
