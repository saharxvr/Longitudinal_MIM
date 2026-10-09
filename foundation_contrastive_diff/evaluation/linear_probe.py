"""
Diagnostic #1: linear probe on the frozen features — is the change info LINEARLY accessible?

Replaces the whole 9.46M-param D-GLoRI head with a single per-patch linear map:

    delta = p_curr - p_prior            # [1369, D]  (FUSION_MODE='diff')
    logit = delta @ W + b               # [1369, 1]  -> reshape 37x37 -> bilinear up to 512

That's the *entire* model: one Linear layer, backbone frozen, bilinear upsample (also linear),
so the predicted map is a strictly linear function of the cached features. Trained on the full
split with the same L1(+L2) loss, evaluated with the same change-only metrics as evaluate_rq1.

Interpretation (compare probe Dice to the full-head ~0.46):
    * probe Dice ~= full head  -> the head architecture is NOT the bottleneck; the ceiling is
      set by the features (frozen RAD-DINO) -> adapt / replace the backbone.
    * probe Dice << full head  -> the info is in the features but nonlinearly tangled ->
      a better head (DPT) is the right lever.

Run:
    python -m evaluation.linear_probe -o $DATA --cache_dir $CACHE4 \
        --feat_last_k 1 --epochs 15 --w_l2 1.0 --threshold 0.05
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from datasets import CachedPairDataset
from losses import change_map_loss
from evaluation.change_detection_metrics import dice_score, iou_score, directional_sensitivity


class LinearProbe(nn.Module):
    """Single per-patch linear map from the prior->current token difference to a scalar."""

    def __init__(self, in_dim: int, grid: int, out_size: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.grid = grid
        self.out_size = out_size

    def forward(self, p_prior, p_curr):               # [B, N, D]
        delta = p_curr - p_prior
        y = self.fc(delta)                            # [B, N, 1]
        b = y.shape[0]
        y = y.reshape(b, 1, self.grid, self.grid)     # [B, 1, 37, 37]
        y = F.interpolate(y, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return y


def _slice(t, feat_dim):
    return t if feat_dim is None else t[..., -feat_dim:]


@torch.no_grad()
def evaluate(model, loader, device, feat_dim, thr):
    model.eval()
    dices, ious, spos, sneg, nuis = [], [], [], [], []
    for batch in loader:
        pp = _slice(batch["p_prior"].to(device), feat_dim)
        pc = _slice(batch["p_curr"].to(device), feat_dim)
        gt = batch["gt_diff"].to(device)
        pred = model(pp, pc)
        pm = (pred.abs() > thr).float()
        gm = (gt.abs() > thr).float()
        is_path = batch["is_pathology"]
        for b in range(pred.shape[0]):
            if int(is_path[b]) == 0:
                nuis.append(float(pm[b].mean()))
                continue
            dices.append(dice_score(pm[b], gm[b]))
            ious.append(iou_score(pm[b], gm[b]))
            d = directional_sensitivity(pred[b:b + 1], gt[b:b + 1], thr)
            if (gt[b] > thr).any():
                spos.append(d["sensitivity_positive"])
            if (gt[b] < -thr).any():
                sneg.append(d["sensitivity_negative"])
    m = lambda x: float(sum(x) / max(1, len(x)))
    return m(dices), m(ious), m(spos), m(sneg), m(nuis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--dataset_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--feat_last_k", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_l2", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()

    device = C.DEVICE
    feat_dim = args.feat_last_k * C.BACKBONE_DIM if args.feat_last_k > 0 else None
    in_dim = feat_dim if feat_dim is not None else C.BACKBONE_DIM * C.LAST_N_LAYERS

    tr = CachedPairDataset(args.dataset_root, args.cache_dir, split="train")
    va = CachedPairDataset(args.dataset_root, args.cache_dir, split="val")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LinearProbe(in_dim, C.FEATURE_GRID, C.IMG_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[linear probe] in_dim {in_dim}  params {sum(p.numel() for p in model.parameters())}  "
          f"train {len(tr)}  val {len(va)}", flush=True)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for batch in tl:
            pp = _slice(batch["p_prior"].to(device), feat_dim)
            pc = _slice(batch["p_curr"].to(device), feat_dim)
            gt = batch["gt_diff"].to(device)
            pred = model(pp, pc)
            loss, _ = change_map_loss(pred, gt, w_l1=args.w_l1, w_dice=0.0, w_sign=0.0,
                                      w_l2=args.w_l2, pos_weight=args.pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        dice, iou, sp, sn, fp = evaluate(model, vl, device, feat_dim, args.threshold)
        best = max(best, dice)
        print(f"epoch {ep:02d}/{args.epochs} | train {tot/len(tl):.4f} | "
              f"val dice {dice:.4f} iou {iou:.4f} sens+ {sp:.3f} sens- {sn:.3f} fp {fp:.4f}",
              flush=True)

    print(f"\n[linear probe] best val dice = {best:.4f}  (compare to full-head ~0.46)")


if __name__ == "__main__":
    main()
