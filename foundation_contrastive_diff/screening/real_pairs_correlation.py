"""
Gold-standard, leakage-free screen: correlate each FROZEN backbone's feature movement with
REAL radiologist annotations on the PNIMIT test pairs.

Why this is the clean test (vs the synthetic-DRR screen):
    * Real patients -> no CT-level leakage (Itamar's oracle advantage vanishes).
    * Real CXRs -> no synthetic-DRR domain match; every backbone is judged on real images.
    * GT = where DOCTORS marked change -> the actual RQ1 clinical claim.

Doctors annotated labeled ellipses (Appearance / Disappearance) on a 792x792 canvas built by the
annotation tool as `nib.load(img).get_fdata().T` -> min-max norm -> resize((792,792)) (a plain
STRETCH). We reproduce that exact mapping so the ellipse masks align with the backbone features.

For each frozen backbone we report AUROC of per-patch cos feature-movement separating
doctor-annotated change patches from the rest (training-free), pooled over all pairs.

Run (GPU node, fcd env):
    python -m screening.real_pairs_correlation \
        --pairs_root "python_files/annotation tool/Pairs_PNIMIT_1_pairs" \
        --annot_root "python_files/annotation tool/Annotations_Pnimit/JSON_only_named" \
        --backbones rad_dino clip_vitb16 pubmedclip biomedclip sam_vitb medsam
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from screening.backbones import build_backbone, GRIDS
from evaluation.feature_change_correlation import _auroc, _pearson, _spearman

CANVAS = 792
DOCTORS_DEFAULT = ["Benny", "Nathalia", "Thaer"]


def _load_img(path):
    """Match the annotation tool: get_fdata().T, middle slice if 3D, min-max to [0,1]."""
    import nibabel as nib
    arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32).T
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)      # [H,W] in [0,1]


def _rasterize(ellipses, size=CANVAS):
    """Signed change map from labeled ellipses on a size x size canvas (Appearance=+, Disappearance=-)."""
    mask = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for e in ellipses:
        sign = 1.0 if str(e.get("label", "")).lower().startswith("appear") else \
               -1.0 if str(e.get("label", "")).lower().startswith("disappear") else 1.0
        cx, cy, rx, ry = e["cx"], e["cy"], max(e["rx"], 1.0), max(e["ry"], 1.0)
        th = math.radians(e.get("angle", 0.0))
        dx, dy = xx - cx, yy - cy
        xr = dx * math.cos(th) + dy * math.sin(th)
        yr = -dx * math.sin(th) + dy * math.cos(th)
        inside = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
        mask[inside] = sign
    return mask


def _doctor_mask(pair_stem, annot_root, doctors):
    """Union of all doctors' signed ellipse masks for a pair; returns (|change| 792x792, prior, current)."""
    mag = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    prior = current = None
    n_doc = 0
    for d in doctors:
        jp = os.path.join(annot_root, d, pair_stem + ".json")
        if not os.path.isfile(jp):
            continue
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            continue
        header = data[0]
        if prior is None and isinstance(header, str) and "|" in header:
            parts = [p.strip() for p in header.split("|")]
            prior, current = parts[0], parts[1]
        ell = [it for it in data[1:] if isinstance(it, dict)]
        mag = np.maximum(mag, np.abs(_rasterize(ell)))
        n_doc += 1
    return mag, prior, current, n_doc


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


def build_pairs(pairs_root, annot_root, doctors):
    """One entry per annotated pair: (prior_nii, current_nii, doctor |change| mask)."""
    stems = set()
    for d in doctors:
        ddir = os.path.join(annot_root, d)
        if os.path.isdir(ddir):
            stems.update(os.path.splitext(f)[0] for f in os.listdir(ddir) if f.endswith(".json"))
    out = []
    for stem in sorted(stems):
        mag, prior, current, n_doc = _doctor_mask(stem, annot_root, doctors)
        if prior is None or n_doc == 0 or mag.max() == 0:
            continue
        folder = os.path.join(pairs_root, "pair_" + stem[len("Pair_"):])
        pp = os.path.join(folder, prior + ".nii.gz")
        cp = os.path.join(folder, current + ".nii.gz")
        if os.path.isfile(pp) and os.path.isfile(cp):
            out.append((pp, cp, mag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_root", required=True)
    ap.add_argument("--annot_root", required=True)
    ap.add_argument("--doctors", nargs="+", default=DOCTORS_DEFAULT)
    ap.add_argument("--backbones", nargs="+",
                    default=["rad_dino", "clip_vitb16", "pubmedclip", "biomedclip", "sam_vitb", "medsam"])
    args = ap.parse_args()

    device = C.DEVICE
    pairs = build_pairs(args.pairs_root, args.annot_root, args.doctors)
    print(f"[real-pairs] {len(pairs)} annotated pairs (doctors: {', '.join(args.doctors)})", flush=True)
    if not pairs:
        print("  no pairs found — check --pairs_root / --annot_root", flush=True)
        return

    print(f"\n{'backbone':<18} {'grid':>4} {'chg%':>6} {'cosAUROC':>9} {'spearman':>9}")
    for name in args.backbones:
        try:
            r = screen_backbone(name, pairs, device)
            print(f"{name:<18} {r['grid']:>4} {100*r['change_frac']:>5.1f} "
                  f"{r['auroc']:>9.3f} {r['spearman']:>+9.3f}", flush=True)
        except Exception as e:  # keep going if one backbone fails to load
            print(f"{name:<18}  SKIP ({type(e).__name__}: {str(e)[:60]})", flush=True)


if __name__ == "__main__":
    main()
