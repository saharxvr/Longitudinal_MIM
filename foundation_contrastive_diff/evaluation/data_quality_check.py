"""
Data-quality / learnability diagnostic for the synthetic change maps (RQ1).

Before investing in a bigger head, check whether the GT itself is learnable at the
backbone's spatial resolution. For a sample of pairs it reports, per anomaly type:

    * change_frac   : fraction of pixels that are "change" (|GT| > tau)
    * mag_median    : median |GT| on change pixels (signal strength; faint GT caps Dice)
    * res_ceiling   : BEST achievable hard-Dice if the model perfectly predicted the
                      PATCH-AVERAGED GT (i.e. the finest a grid-sized feature map can
                      represent). This is the architectural upper bound at this resolution.

If res_ceiling is itself ~0.7, then ~0.7 is the data/resolution limit, not a model
failure — and a finer-resolution head is the only way past it.

Run:
    python -m evaluation.data_quality_check -o /path/to/fcd_train --split val --limit 300
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from datasets import LongitudinalPairDataset
from evaluation.change_detection_metrics import dice_score


def resolution_ceiling_dice(gt, grid, img_size, tau):
    """Best hard-Dice if the model predicted the patch-averaged GT (grid-res limit)."""
    g = gt.unsqueeze(0)                                      # [1,1,H,W]
    coarse = F.adaptive_avg_pool2d(g, (grid, grid))         # what a grid-sized map can hold
    recon = F.interpolate(coarse, size=(img_size, img_size), mode="bilinear", align_corners=False)[0]
    m_gt = (gt.abs() > tau).float()
    m_rec = (recon.abs() > tau).float()
    return dice_score(m_rec, m_gt)


def main():
    p = argparse.ArgumentParser(description="synthetic change-map data-quality check")
    p.add_argument("-o", "--dataset_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--tau", type=float, default=0.05, help="|GT| threshold for 'change'")
    p.add_argument("--grid", type=int, default=C.FEATURE_GRID, help="backbone token grid (37)")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    ds = LongitudinalPairDataset(args.dataset_root, split=args.split, img_size=C.IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.num_workers)

    per = defaultdict(lambda: {"frac": [], "mag": [], "ceil": [], "realized": 0})
    n = 0
    for batch in loader:
        gt = batch["gt_diff"][0]                            # [1,H,W]
        a = int(batch["anomaly_type"][0])
        name = C.ANOMALY_TYPES[a] if a < len(C.ANOMALY_TYPES) else str(a)
        mask = gt.abs() > args.tau
        frac = float(mask.float().mean())
        per[name]["frac"].append(frac)
        if mask.any():
            per[name]["mag"].append(float(gt.abs()[mask].median()))
            per[name]["realized"] += 1
        per[name]["ceil"].append(resolution_ceiling_dice(gt, args.grid, C.IMG_SIZE, args.tau))
        n += 1
        if n >= args.limit:
            break

    def mean(x):
        return sum(x) / max(1, len(x))

    print(f"[data-check] split={args.split}  n={n}  tau={args.tau}  grid={args.grid}x{args.grid}\n")
    print(f"{'anomaly':16s} {'n':>5s} {'realized':>9s} {'change_frac':>12s} {'mag_median':>11s} {'res_ceiling':>12s}")
    all_ceil = []
    for name, d in per.items():
        all_ceil += d["ceil"]
        print(f"{name:16s} {len(d['ceil']):5d} {d['realized']:9d} "
              f"{mean(d['frac']):12.4f} {mean(d['mag']):11.3f} {mean(d['ceil']):12.3f}")
    print(f"\n  OVERALL mean resolution-ceiling Dice = {mean(all_ceil):.3f}")
    print("\nInterpretation:")
    print("  - res_ceiling ~0.7  -> 0.7 is the RESOLUTION limit; model at ~0.71 is already near it.")
    print("  - res_ceiling ~0.95 -> plenty of headroom; the model, not the data, is the limit.")
    print("  - low mag_median (<0.2) -> change signal is faint; hard-Dice will read low regardless.")


if __name__ == "__main__":
    main()
