"""
Change-map loss for RQ1 — handles the sparse, signed GT difference map.

The GT is mostly zero (no change) with small signed regions (appear = +, resolve = -).
A plain L1 collapses to predicting all zeros, so we combine three terms:

    L = w_l1 * weighted_L1        (magnitude; change pixels upweighted)
      + w_dice * (1 - soft_dice)  (|pred| vs change mask; fights the all-zero collapse)
      + w_sign * sign_penalty     (correct direction on change pixels: appear vs resolve)

pred and gt are both [B, 1, H, W] in [-1, 1].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def change_map_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float = 0.02,
    w_l1: float = 1.0,
    w_dice: float = 1.0,
    w_sign: float = 0.5,
    pos_weight: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (total_loss, components dict)."""
    change_mask = (gt.abs() > tau).float()                       # [B,1,H,W]

    # 1) Weighted L1 on magnitude — change pixels count much more than background.
    weight = 1.0 + pos_weight * change_mask
    l1 = (weight * (pred - gt).abs()).mean()

    # 2) Soft Dice between |pred| and the change mask (per-sample, then averaged).
    pmag = pred.abs()
    dims = (1, 2, 3)
    inter = (pmag * change_mask).sum(dims)
    denom = pmag.sum(dims) + change_mask.sum(dims)
    dice = 1.0 - ((2 * inter + eps) / (denom + eps))
    dice = dice.mean()

    # 3) Direction penalty: on change pixels, penalize sign(pred) != sign(gt).
    sign_pen = (F.relu(-pred * gt) * change_mask).sum() / (change_mask.sum() + eps)

    total = w_l1 * l1 + w_dice * dice + w_sign * sign_pen
    return total, {
        "l1": float(l1.detach()),
        "dice": float(dice.detach()),
        "sign": float(sign_pen.detach()),
        "total": float(total.detach()),
    }
