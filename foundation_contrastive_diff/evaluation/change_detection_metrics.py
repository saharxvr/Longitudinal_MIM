"""
Change-detection metrics for evaluating the difference map (RQ1).

Metrics:
    - dice / iou on thresholded change regions
    - sensitivity for positive (new) and negative (resolved) changes
    - Pairwise Agreement Index (PAI) vs. radiologist annotations (parent-project metric)
"""

import torch


def _binarize(diff_map: torch.Tensor, threshold: float = 0.1):
    """Split a signed change map into positive/negative change masks."""
    pos = (diff_map > threshold).float()
    neg = (diff_map < -threshold).float()
    return pos, neg


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def directional_sensitivity(pred_diff: torch.Tensor, gt_diff: torch.Tensor, threshold: float = 0.1):
    """Sensitivity (recall) for positive and negative changes separately."""
    pred_pos, pred_neg = _binarize(pred_diff, threshold)
    gt_pos, gt_neg = _binarize(gt_diff, threshold)

    def recall(p, g, eps=1e-6):
        return ((p * g).sum() / (g.sum() + eps)).item()

    return {
        "sensitivity_positive": recall(pred_pos, gt_pos),
        "sensitivity_negative": recall(pred_neg, gt_neg),
    }


def pairwise_agreement_index(pred_diff: torch.Tensor, radiologist_diff: torch.Tensor):
    """Placeholder for PAI vs. radiologist annotations (parent-project metric).

    TODO(Phase 2): port the PAI implementation from the parent project's evaluation code.
    """
    raise NotImplementedError("Port PAI from parent project evaluation — see RESEARCH_PLAN.md.")
