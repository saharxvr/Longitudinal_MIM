"""Loss functions for foundation-based contrastive difference detection."""

from .contrastive_losses import (
    supervised_contrastive_loss,
    triplet_loss,
    orthogonality_loss,
)
from .change_map_losses import change_map_loss

__all__ = [
    "supervised_contrastive_loss",
    "triplet_loss",
    "orthogonality_loss",
    "change_map_loss",
]
