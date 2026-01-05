"""
Evaluation Metrics
==================

Functions for computing evaluation metrics like Dice coefficient,
connected component analysis, and detection metrics.

Usage:
------
    from utils.metrics import dice_coefficient
    
    pred = torch.tensor([...])
    target = torch.tensor([...])
    score = dice_coefficient(pred, target, threshold=0.5)
"""

import torch
import numpy as np
from typing import Optional, Tuple


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient between prediction and target.
    
    The Dice coefficient measures overlap between two binary masks:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask, can be probabilities or binary.
        Shape: (B, C, H, W) or (B, H, W) or (H, W)
    target : torch.Tensor
        Ground truth binary mask, same shape as pred.
    threshold : float, default=0.5
        Threshold to binarize predictions if not already binary.
    smooth : float, default=1e-6
        Smoothing factor to avoid division by zero.
        
    Returns
    -------
    torch.Tensor
        Dice coefficient, scalar or per-batch tensor.
        
    Examples
    --------
    >>> pred = torch.rand(1, 1, 256, 256)
    >>> target = (torch.rand(1, 1, 256, 256) > 0.5).float()
    >>> dice = dice_coefficient(pred, target)
    """
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten spatial dimensions
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Compute Dice
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def dice_coefficient_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient per sample in a batch.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predictions, shape (B, C, H, W) or (B, H, W)
    target : torch.Tensor
        Targets, same shape as pred.
    threshold : float
        Binarization threshold.
    smooth : float
        Smoothing factor.
        
    Returns
    -------
    torch.Tensor
        Dice scores, shape (B,)
    """
    batch_size = pred.shape[0]
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten all but batch dimension
    pred_flat = pred_binary.view(batch_size, -1)
    target_flat = target_binary.view(batch_size, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU / Jaccard index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted mask.
    target : torch.Tensor
        Ground truth mask.
    threshold : float
        Binarization threshold.
    smooth : float
        Smoothing factor.
        
    Returns
    -------
    torch.Tensor
        IoU score.
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou
