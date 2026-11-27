"""Losses for selective few-shot change maps (ignore-aware three-class CE/Dice)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def class_targets(signed, ignore=None, threshold=0.001):
    """[B,1,H,W] signed map -> [B,H,W] long classes (0 unchanged, 1 positive, 2 negative)."""
    values = signed[:, 0].float()
    target = torch.zeros_like(values, dtype=torch.long)
    target[values > threshold] = 1
    target[values < -threshold] = 2
    if ignore is not None:
        target[ignore[:, 0].bool()] = IGNORE_INDEX
    return target


def masked_three_class_loss(logits, signed, ignore=None, threshold=0.001, w_ce=1.0, w_dice=0.0,
                            class_weights=(1.0, 1.0, 1.0), eps=1e-6):
    if logits.ndim != 4 or logits.shape[1] != 3 or signed.shape != (logits.shape[0], 1, *logits.shape[2:]):
        raise ValueError("Expected [B,3,H,W] logits and [B,1,H,W] signed targets")
    target = class_targets(signed, ignore, threshold)
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=torch.float32)
    valid = target != IGNORE_INDEX
    if not bool(valid.any()):
        zero = logits.sum() * 0
        return zero, dict(ce=0.0, dice=0.0, total=0.0)
    ce = F.cross_entropy(logits.float(), target, weight=weights, ignore_index=IGNORE_INDEX)
    total = w_ce * ce
    dice_value = logits.new_zeros(())
    if w_dice:
        probabilities = logits.float().softmax(1)[:, 1:] * valid[:, None]
        reference = F.one_hot(target.clamp_min(0), 3).permute(0, 3, 1, 2)[:, 1:].float() * valid[:, None]
        intersection = (probabilities * reference).sum((0, 2, 3))
        denominator = probabilities.sum((0, 2, 3)) + reference.sum((0, 2, 3))
        dice_value = (1 - (2 * intersection + eps) / (denominator + eps)).mean()
        total = total + w_dice * dice_value
    return total, dict(ce=float(ce.detach()), dice=float(dice_value.detach()), total=float(total.detach()))


def soft_grid_loss(logits, soft_targets):
    """Soft-label CE between [B,N,3] token logits and [B,N,3] grid label fractions."""
    if logits.shape != soft_targets.shape:
        raise ValueError("Grid logits and soft targets must match")
    return -(soft_targets * logits.float().log_softmax(-1)).sum(-1).mean()
