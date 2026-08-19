"""
Disentangled projection heads.

Maps the difference embedding `z` into two subspaces:
    - z_path : pathology-relevant change subspace (supervised by anomaly-type labels)
    - z_nuis : nuisance / acquisition change subspace (positioning, exposure, noise)

An orthogonality constraint between the two subspaces encourages disentanglement so the
model can explicitly factor out clinically irrelevant differences (see RESEARCH_PLAN.md
Phase 3 and orthogonality_loss in losses/contrastive_losses.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2-normalize for contrastive losses.
        return F.normalize(self.net(x), dim=-1)


class DisentangledProjectionHeads(nn.Module):
    """Two projection heads splitting `z` into pathology and nuisance subspaces."""

    def __init__(self, embed_dim: int = 256, proj_dim: int = 128, use_disentanglement: bool = True):
        super().__init__()
        self.use_disentanglement = use_disentanglement
        self.path_head = _ProjectionMLP(embed_dim, proj_dim)
        self.nuis_head = _ProjectionMLP(embed_dim, proj_dim) if use_disentanglement else None

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, embed_dim] difference embedding.
        Returns:
            z_path: [B, proj_dim]
            z_nuis: [B, proj_dim] or None
        """
        z_path = self.path_head(z)
        z_nuis = self.nuis_head(z) if self.nuis_head is not None else None
        return z_path, z_nuis
