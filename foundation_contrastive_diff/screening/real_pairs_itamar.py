"""
Itamar's reconstruction AE on the REAL radiologist-annotated pairs (confound check).

Standalone twin of screening/real_pairs_correlation.py for Itamar's encoder, which can't share
the fcd backbone registry (its `constants`/`models` clash with this study's). Run from any CWD;
it puts python_files on sys.path to import his model. Imports NO foundation_contrastive_diff code.

Purpose: Itamar WON the synthetic screen (cos 0.731) but that is confounded by CT-level leakage +
DRR domain-match. On REAL CXRs he is OUT-of-domain (he trained on DRRs), so his cos AUROC should
DROP toward chance. The gap between his synthetic 0.731 and his real AUROC quantifies the confound.

Encoder: model.backbone.encoder(x) -> encoded_bn -> [B,640,32,32] -> tokens, grid 32.

Run (fcd env, from the repo):
    python screening/real_pairs_itamar.py --root "../python_files/annotation tool" \
        --ckpt /cs/labs/josko/itamar_sab/.../Checkpoint_id45_..._L1L2_GN.pt
"""

import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

CANVAS = 792
_EXCLUDE_PARTS = {"new", "old", "second_look"}
_EXCLUDE_PREFIX = ("ver", "ov_", "mini_ov", "presentation")


def _add_code_root(code_root=""):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    for c in [c for c in (code_root, os.path.join(repo, "python_files"), os.getcwd()) if c]:
        if os.path.isdir(os.path.join(c, "refactored_modules")):
            sys.path.insert(0, c)
            print(f"[itamar] code_root = {c}", flush=True)
            return c
    raise ModuleNotFoundError("refactored_modules not found; pass --code_root")


# ---- doctor-mask helpers (identical semantics to real_pairs_correlation) --------------------
def _load_img(path):
    import nibabel as nib
    arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32).T
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)


def _is_change(e):
    lab = str(e.get("label", "")).lower()
    if lab.startswith(("appear", "disappear")):
        return True
    if lab.startswith("persist"):
        sc = str(e.get("size_change", "")).lower()
        ic = str(e.get("intensity_change", "")).lower()
        return sc in ("increase", "decrease") or ic in ("increase", "decrease")
    return True


def _rasterize(ellipses, size=CANVAS):
    mask = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for e in ellipses:
        if not _is_change(e):
            continue
        cx, cy, rx, ry = e["cx"], e["cy"], max(e["rx"], 1.0), max(e["ry"], 1.0)
        th = math.radians(e.get("angle", 0.0))
        dx, dy = xx - cx, yy - cy
        xr = dx * math.cos(th) + dy * math.sin(th)
        yr = -dx * math.sin(th) + dy * math.cos(th)
        mask[(xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0] = 1.0
    return mask


def _excluded(path):
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    return any(p in _EXCLUDE_PARTS for p in parts) or any(p.startswith(_EXCLUDE_PREFIX) for p in parts)


def _image_index(root):
    idx = {}
    for pairs_dir in glob(os.path.join(root, "Pairs*")):
        for p in glob(os.path.join(pairs_dir, "**", "*.nii.gz"), recursive=True):
            base = os.path.basename(p)[:-len(".nii.gz")]
            if not base.endswith("_lung_seg"):
                idx[base] = p
    return idx


def build_pairs(annot_dirs, img_index):
    masks = {}
    for annot in annot_dirs:
        for jp in glob(os.path.join(annot, "**", "*.json"), recursive=True):
            if _excluded(jp):
                continue
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if (not isinstance(data, list) or not data or not isinstance(data[0], str)
                    or "|" not in data[0]):
                continue
            parts = [p.strip() for p in data[0].split("|")]
            if len(parts) < 2 or parts[0] not in img_index or parts[1] not in img_index:
                continue
            ell = [it for it in data[1:] if isinstance(it, dict)]
            if not ell:
                continue
            key = (img_index[parts[0]], img_index[parts[1]])
            m = _rasterize(ell)
            masks[key] = np.maximum(masks[key], m) if key in masks else m
    return [(pp, cp, m) for (pp, cp), m in masks.items() if m.max() > 0]


# ---- metrics (self-contained; no fcd import) -----------------------------------------------
def _rank(x):
    order = torch.argsort(x)
    r = torch.empty_like(x)
    r[order] = torch.arange(x.numel(), dtype=x.dtype)
    return r


def _auroc(scores, labels):
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank(scores) + 1.0
    return (float(ranks[pos].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---- Itamar encoder -------------------------------------------------------------------------
def build_itamar(ckpt_path, device):
    from refactored_modules.longitudinal_v2 import LongitudinalMIMDeformV2
    model = LongitudinalMIMDeformV2()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_dict"] if isinstance(ck, dict) and "model_dict" in ck else ck
    bb_state = {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in state.items()}
    missing, _ = model.backbone.load_state_dict(bb_state, strict=False)
    enc_missing = [k for k in missing if k.startswith(("encoder.", "encoded_bn."))]
    print(f"[itamar] encoder/bn missing {len(enc_missing)}", flush=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../python_files/annotation tool")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--code_root", default="")
    args = ap.parse_args()

    _add_code_root(args.code_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_itamar(args.ckpt, device)

    img_index = _image_index(args.root)
    annot_dirs = [os.path.join(args.root, "Annotations"), os.path.join(args.root, "Annotations_Pnimit")]
    pairs = build_pairs(annot_dirs, img_index)
    print(f"[real-pairs/itamar] images {len(img_index)} | pairs {len(pairs)} | grid 32", flush=True)

    cos_all, lbl_all = [], []
    for pp, cp, dmask in pairs:
        pimg = torch.from_numpy(_load_img(pp))[None, None].to(device)
        cimg = torch.from_numpy(_load_img(cp))[None, None].to(device)
        # Itamar's CNN encoder scales with input -> stretch to 512x512 (grid 32), as in training.
        pimg = F.interpolate(pimg, size=(512, 512), mode="bilinear", align_corners=False)
        cimg = F.interpolate(cimg, size=(512, 512), mode="bilinear", align_corners=False)
        fp = model.encode(pimg).flatten(2).transpose(1, 2)[0]     # [1024,640]
        fc = model.encode(cimg).flatten(2).transpose(1, 2)[0]
        cos = 1.0 - F.cosine_similarity(fp, fc, dim=-1)
        gt = F.adaptive_avg_pool2d(torch.from_numpy(dmask)[None, None].to(device), 32).reshape(-1)
        cos_all.append(cos.cpu()); lbl_all.append((gt > 0).float().cpu())
    cos_all = torch.cat(cos_all); lbl_all = torch.cat(lbl_all)
    print(f"\nitamar_ae (real pairs)  grid 32  chg% {100*float(lbl_all.mean()):.1f}  "
          f"cosAUROC {_auroc(cos_all, lbl_all):.3f}", flush=True)
    print("  compare: synthetic 0.731 -> drop quantifies the leakage/domain confound; "
          "RAD-DINO real = 0.614", flush=True)


if __name__ == "__main__":
    main()
