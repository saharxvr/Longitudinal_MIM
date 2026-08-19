"""Evaluation utilities for foundation-based contrastive difference detection."""

from .change_detection_metrics import (
    dice_score,
    iou_score,
    directional_sensitivity,
)
from .latent_space_analysis import (
    collect_embeddings,
    silhouette,
    linear_probe_accuracy,
    plot_tsne,
)

__all__ = [
    "dice_score",
    "iou_score",
    "directional_sensitivity",
    "collect_embeddings",
    "silhouette",
    "linear_probe_accuracy",
    "plot_tsne",
]
