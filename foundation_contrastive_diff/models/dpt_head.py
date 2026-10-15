"""DPT-style change decoder for concatenated, adjacent last-four ViT layers."""

import torch
from torch import nn
from torch.nn import functional as F

from .difference_head import GroupNormCG


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            GroupNormCG(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            GroupNormCG(8, channels),
        )

    def forward(self, features):
        return features + self.layers(features.contiguous())


class DPTDifferenceHead(nn.Module):
    """Separate layer reassembly and residual top-down fusion; backbone stays frozen."""

    def __init__(self, backbone_dim=768, patch_dim=3072, cls_dim=768,
                 grid=37, channels=96, embed_dim=256, out_size=512,
                 out_range=(-1.0, 1.0)):
        super().__init__()
        if patch_dim != 4 * backbone_dim:
            raise ValueError("DPT head requires exactly four concatenated backbone layers")
        if channels < 8 or channels % 8:
            raise ValueError("DPT channels must be a positive multiple of eight")
        if grid < 2:
            raise ValueError("DPT grid must be at least two")
        self.backbone_dim = backbone_dim
        self.patch_dim = patch_dim
        self.grid = grid
        self.out_size = out_size
        self.out_range = out_range
        self.sizes = (grid * 4, grid * 2, grid, grid // 2)
        self.projections = nn.ModuleList([
            nn.Conv2d(backbone_dim, channels, 1) for _ in range(4)
        ])
        self.reassembly = nn.ModuleList([
            nn.ConvTranspose2d(channels, channels, 4, stride=4),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.Identity(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
        ])
        self.lateral = nn.ModuleList([ResidualConv(channels) for _ in range(4)])
        self.refine = nn.ModuleList([ResidualConv(channels) for _ in range(4)])
        self.cls_proj = nn.Linear(cls_dim, channels)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
        )
        self.z_proj = nn.Linear(channels * 2, embed_dim)

    def forward(self, p_prior, p_curr, cls_prior=None, cls_curr=None):
        expected = (self.grid * self.grid, self.patch_dim)
        if p_prior.shape != p_curr.shape or tuple(p_prior.shape[1:]) != expected:
            raise ValueError(f"Expected matching [batch, {expected[0]}, {expected[1]}] tokens")
        batch_size = p_prior.shape[0]
        streams = (p_curr - p_prior).split(self.backbone_dim, dim=-1)
        pyramid = []
        for stream, project, reassemble, size in zip(
                streams, self.projections, self.reassembly, self.sizes):
            features = stream.transpose(1, 2).reshape(
                batch_size, self.backbone_dim, self.grid, self.grid).contiguous()
            features = reassemble(project(features))
            features = F.interpolate(features, size=(size, size), mode="bilinear",
                                     align_corners=False).contiguous()
            pyramid.append(features)
        global_change = pyramid[-1].new_zeros(batch_size, self.cls_proj.out_features)
        if cls_prior is not None and cls_curr is not None:
            global_change = self.cls_proj(cls_curr - cls_prior)
        fused = None
        for index in range(3, -1, -1):
            lateral = self.lateral[index](pyramid[index])
            if fused is None:
                fused = lateral + global_change[:, :, None, None]
            else:
                fused = F.interpolate(fused, size=lateral.shape[-2:], mode="bilinear",
                                      align_corners=False).contiguous() + lateral
            fused = self.refine[index](fused)
        embedding = self.z_proj(torch.cat((fused.mean(dim=(-2, -1)), global_change), dim=1))
        logits = self.output(fused.contiguous())
        logits = F.interpolate(logits, size=(self.out_size, self.out_size),
                               mode="bilinear", align_corners=False)
        signed_map = torch.tanh(logits)
        low, high = self.out_range
        if (low, high) != (-1.0, 1.0):
            signed_map = low + (signed_map + 1.0) * 0.5 * (high - low)
        return signed_map, embedding