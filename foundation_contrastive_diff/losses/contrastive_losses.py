"""
Contrastive and disentanglement losses for the difference embedding.

Implements:
    - supervised_contrastive_loss (SupCon): pulls together embeddings of the same
      change category (anomaly type / pathology-vs-nuisance), pushes apart different ones.
    - triplet_loss: anchor/positive/negative trios as an alternative/addition.
    - orthogonality_loss: enforces independence between pathology and nuisance subspaces.

Labels come from the synthetic DRR pipeline (it knows what change was introduced).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Args:
        embeddings: [B, D] L2-normalized embeddings.
        labels: [B] integer class labels (e.g. anomaly type).
        temperature: softmax temperature.
    Returns:
        scalar loss.
    """
    device = embeddings.device
    batch_size = embeddings.shape[0]

    # Cosine similarity logits.
    sim = embeddings @ embeddings.t() / temperature
    # Numerical stability.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.t()).float().to(device)
    # Remove self-comparisons.
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask
    logits_mask = 1.0 - self_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_per_sample = positive_mask.sum(dim=1)
    # Avoid div-by-zero for samples with no positive pair.
    valid = pos_per_sample > 0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / pos_per_sample[valid]

    if mean_log_prob_pos.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return -mean_log_prob_pos.mean()


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Standard triplet margin loss on L2-normalized embeddings."""
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin)


def orthogonality_loss(z_path: torch.Tensor, z_nuis: torch.Tensor) -> torch.Tensor:
    """Penalize correlation between pathology and nuisance subspaces.

    Uses the mean squared cosine similarity between paired embeddings as a soft
    orthogonality constraint.
    """
    if z_nuis is None:
        return torch.zeros((), device=z_path.device)
    cos = F.cosine_similarity(z_path, z_nuis, dim=-1)
    return (cos ** 2).mean()
