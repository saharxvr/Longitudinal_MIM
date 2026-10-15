"""
Gold-standard, leakage-free screen: correlate each FROZEN backbone's feature movement with
REAL radiologist annotations on ALL annotated test pairs (ICU `Annotations` + `Annotations_Pnimit`).

Why this is the clean test (vs the synthetic-DRR screen):
    * Real patients -> no CT-level leakage (Itamar's oracle advantage vanishes).
    * Real CXRs -> no synthetic-DRR domain match; every backbone is judged on real images.
    * GT = where DOCTORS marked change -> the actual RQ1 clinical claim.

Doctors drew labeled ellipses (Appearance / Disappearance / Persistence[+size/intensity change])
on a 792x792 canvas the annotation tool built as `nib.load(img).get_fdata().T` -> min-max norm ->
resize((792,792)) (a plain STRETCH). We reproduce that mapping so masks align with features. Pair
images are resolved by the JSON header ("prior | current") against a global name->folder index over
all Pairs* dirs, so it is robust to the ICU vs PNIMIT folder/number naming.

AUROC of per-patch cos feature-movement separating doctor-change patches from the rest, pooled
over all pairs (training-free). Union of all doctors across both sets.

Run (GPU node, fcd env):
    python -m screening.real_pairs_correlation --root "../python_files/annotation tool" \
        --backbones rad_dino clip_vitb16 pubmedclip biomedclip sam_vitb medsam
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from screening.backbones import build_backbone, GRIDS
from evaluation.feature_change_correlation import _auroc, _spearman

CANVAS = 792
_EXCLUDE_PARTS = {"new", "old", "second_look"}
_EXCLUDE_PREFIX = ("ver", "ov_", "mini_ov", "presentation")


def _load_img(path):
    """Match the annotation tool: get_fdata().T, middle slice if 3D, min-max to [0,1]."""
    import nibabel as nib
    arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32).T
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)      # [H,W] in [0,1]


def _is_change(e):
    """A real change region? Appearance/Disappearance always; Persistence only if size/intensity changed."""
    lab = str(e.get("label", "")).lower()
    if lab.startswith(("appear", "disappear")):
        return True
    if lab.startswith("persist"):
        sc = str(e.get("size_change", "")).lower()
        ic = str(e.get("intensity_change", "")).lower()
        return sc in ("increase", "decrease") or ic in ("increase", "decrease")
    return True  # unknown labels: count conservatively as change


def _rasterize(ellipses, size=CANVAS):
    """|change| map from change ellipses on a size x size canvas."""
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
    if any(p in _EXCLUDE_PARTS for p in parts):
        return True
    return any(p.startswith(_EXCLUDE_PREFIX) for p in parts)


def _image_index(root):
    """basename (no _lung_seg, no .nii.gz) -> full path, over every Pairs* dir under root."""
    idx = {}
    for pairs_dir in glob(os.path.join(root, "Pairs*")):
        for p in glob(os.path.join(pairs_dir, "**", "*.nii.gz"), recursive=True):
            base = os.path.basename(p)[:-len(".nii.gz")]
            if base.endswith("_lung_seg"):
                continue
            idx[base] = p
    return idx


@torch.no_grad()
def screen_backbone(name, pairs, device, tau_frac=0.0):
    bb = build_backbone(name).to(device).eval()
    grid = GRIDS.get(name, getattr(bb, "grid", C.FEATURE_GRID))
    cos_all, lbl_all = [], []
    used = 0
    for (prior_path, curr_path, dmask) in pairs:
        pimg = torch.from_numpy(_load_img(prior_path))[None, None].to(device)   # [1,1,H,W]
        cimg = torch.from_numpy(_load_img(curr_path))[None, None].to(device)
        op, oc = bb(pimg), bb(cimg)
        pp, pc = op["patch_tokens"][0], oc["patch_tokens"][0]                    # [N,D]
        cos = 1.0 - F.cosine_similarity(pp, pc, dim=-1)                          # [N]
        dm = torch.from_numpy(dmask)[None, None].to(device)                      # [1,1,792,792]
        gt = F.adaptive_avg_pool2d(dm, grid).reshape(-1)                         # [N]
        cos_all.append(cos.cpu()); lbl_all.append((gt > tau_frac).float().cpu())
        used += 1
    cos_all = torch.cat(cos_all); lbl_all = torch.cat(lbl_all)
    frac = float(lbl_all.mean())
    return {
        "grid": grid, "pairs": used, "patches": cos_all.numel(), "change_frac": frac,
        "auroc": _auroc(cos_all, lbl_all),
        "spearman": _spearman(cos_all, lbl_all),
    }


def build_pairs(root, annot_dirs, img_index):
    """Union all doctors' change masks per pair -> list of (prior_path, current_path, |change| 792)."""
    masks = {}  # key: (prior_path, current_path) -> np.float32 [792,792]
    n_json = 0
    for annot in annot_dirs:
        for jp in glob(os.path.join(annot, "**", "*.json"), recursive=True):
            if _excluded(jp):
                continue
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if (not isinstance(data, list) or len(data) < 1 or not isinstance(data[0], str)
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
            n_json += 1
    pairs = [(pp, cp, m) for (pp, cp), m in masks.items() if m.max() > 0]
    return pairs, n_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../python_files/annotation tool",
                    help="annotation-tool dir containing Pairs*/ and Annotations*/")
    ap.add_argument("--annot_dirs", nargs="+", default=None,
                    help="annotation roots (default: <root>/Annotations and <root>/Annotations_Pnimit)")
    ap.add_argument("--backbones", nargs="+",
                    default=["rad_dino", "clip_vitb16", "pubmedclip", "biomedclip", "sam_vitb", "medsam"])
    args = ap.parse_args()

    annot_dirs = args.annot_dirs or [os.path.join(args.root, "Annotations"),
                                     os.path.join(args.root, "Annotations_Pnimit")]
    img_index = _image_index(args.root)
    pairs, n_json = build_pairs(args.root, annot_dirs, img_index)
    print(f"[real-pairs] images indexed {len(img_index)} | annotation files used {n_json} | "
          f"unique annotated pairs {len(pairs)}", flush=True)
    if not pairs:
        print("  no pairs - check --root (needs Pairs*/ + Annotations*/)", flush=True)
        return

    print(f"\n{'backbone':<18} {'grid':>4} {'chg%':>6} {'cosAUROC':>9} {'spearman':>9}")
    for name in args.backbones:
        try:
            r = screen_backbone(name, pairs, device=C.DEVICE)
            print(f"{name:<18} {r['grid']:>4} {100*r['change_frac']:>5.1f} "
                  f"{r['auroc']:>9.3f} {r['spearman']:>+9.3f}", flush=True)
        except Exception as e:
            print(f"{name:<18}  SKIP ({type(e).__name__}: {str(e)[:60]})", flush=True)


if __name__ == "__main__":
    main()
