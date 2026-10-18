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

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def three_class_change_map_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    target_threshold: float = 0.001,
    w_ce: float = 1.0,
    w_dice: float = 1.0,
    class_weights: tuple = (1.0, 1.0, 1.0),
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CE plus batchwise foreground-class Dice; classes: unchanged, positive, negative."""
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError("Three-class loss requires [B,3,H,W] raw logits")
    if gt.shape != (logits.shape[0], 1, *logits.shape[2:]):
        raise ValueError("GT must have shape [B,1,H,W] matching logits")
    if (not 0 <= target_threshold < 1 or w_ce <= 0 or w_dice < 0 or eps <= 0
            or not all(math.isfinite(value) for value in (w_ce, w_dice, eps))):
        raise ValueError("Invalid three-class loss coefficients or threshold")
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=torch.float32)
    if weights.shape != (3,) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("All three CE class weights must be finite and positive")
    if not torch.isfinite(gt).all() or (gt.abs() > 1).any():
        raise ValueError("GT must be finite and in [-1,1]")
    values = gt[:, 0].float()
    target = torch.zeros_like(values, dtype=torch.long)
    target[values > target_threshold] = 1
    target[values < -target_threshold] = 2
    ce = F.cross_entropy(logits.float(), target, weight=weights)
    probabilities = logits.float().softmax(dim=1)[:, 1:]
    reference = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2)[:, 1:].float()
    intersection = (probabilities * reference).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + reference.sum((0, 2, 3))
    class_losses = 1 - (2 * intersection + eps) / (denominator + eps)
    dice = class_losses.mean()
    total = w_ce * ce + w_dice * dice
    return total, {
        "ce": float(ce.detach()),
        "dice": float(dice.detach()),
        "dice_positive": float(class_losses[0].detach()),
        "dice_negative": float(class_losses[1].detach()),
        "total": float(total.detach()),
    }


def change_map_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    tau: float = 0.02,
    w_l1: float = 1.0,
    w_dice: float = 1.0,
    w_sign: float = 0.5,
    w_l2: float = 0.0,
    pos_weight: float = 10.0,
    eps: float = 1e-6,
    target_mode: str = "continuous",
    target_threshold: float = 0.05,
    background_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (total_loss, components dict).

    Set w_dice=w_sign=0, w_l2=1 for an Itamar-style pure L1+L2 reconstruction loss.
    """
    if target_mode not in ("continuous", "signed_mask"):
        raise ValueError("target_mode must be continuous or signed_mask")
    if not 0 <= target_threshold < 1 or not 0 < background_weight <= 1:
        raise ValueError("Invalid target_threshold or background_weight")
    if target_mode == "signed_mask":
        gt = torch.where(gt.abs() > target_threshold, gt.sign(), torch.zeros_like(gt))
    change_mask = (gt.abs() > tau).float()                       # [B,1,H,W]

    # 1) Weighted L1 on magnitude — change pixels count much more than background.
    weight = background_weight + (1.0 + pos_weight - background_weight) * change_mask
    l1 = (weight * (pred - gt).abs()).mean()

    # 1b) Weighted L2 (MSE) — reconstruction term (Itamar-style).
    l2 = (weight * (pred - gt) ** 2).mean()

    # 2) Soft Dice between |pred| and the change mask (per-sample, then averaged).
    pmag = pred.abs()
    dims = (1, 2, 3)
    inter = (pmag * change_mask).sum(dims)
    denom = pmag.sum(dims) + change_mask.sum(dims)
    dice = 1.0 - ((2 * inter + eps) / (denom + eps))
    dice = dice.mean()

    # 3) Direction penalty: on change pixels, penalize sign(pred) != sign(gt).
    sign_pen = (F.relu(-pred * gt) * change_mask).sum() / (change_mask.sum() + eps)

    total = w_l1 * l1 + w_l2 * l2 + w_dice * dice + w_sign * sign_pen
    return total, {
        "l1": float(l1.detach()),
        "l2": float(l2.detach()),
        "dice": float(dice.detach()),
        "sign": float(sign_pen.detach()),
        "total": float(total.detach()),
    }
