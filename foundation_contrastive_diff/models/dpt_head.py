"""DPT-style change decoding and explicit multi-depth temporal fusion."""

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


class _DPTDecoder(nn.Module):
    """Reassemble one or four projected maps with the shared signed-change decoder."""

    def __init__(self, cls_dim=768,
                 grid=37, channels=96, embed_dim=256, out_size=512,
                 out_range=(-1.0, 1.0), num_layers=4, output_mode="signed"):
        super().__init__()
        self._validate_decoder(grid, channels, num_layers, output_mode)
        self.output_mode = output_mode
        self.num_layers = num_layers
        self.grid = grid
        self.out_size = out_size
        self.out_range = out_range
        self.sizes = (grid * 4, grid * 2, grid, grid // 2)
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
            nn.Conv2d(channels // 2, 3 if output_mode == "three_class" else 1, 1),
        )
        self.z_proj = nn.Linear(channels * 2, embed_dim)

    @staticmethod
    def _validate_decoder(grid, channels, num_layers, output_mode):
        if num_layers not in (1, 4):
            raise ValueError("DPT decoder requires one or four projected streams")
        if channels < 8 or channels % 8:
            raise ValueError("DPT channels must be a positive multiple of eight")
        if grid < 2:
            raise ValueError("DPT grid must be at least two")
        if output_mode not in ("signed", "three_class"):
            raise ValueError("Output mode must be signed or three_class")

    def forward(self, projected, cls_prior=None, cls_curr=None):
        if (len(projected) != self.num_layers or any(
                features.ndim != 4 or features.shape[1:] != (
                    self.cls_proj.out_features, self.grid, self.grid)
                for features in projected)):
            raise ValueError("Unexpected number or shape of projected DPT streams")
        batch_size = projected[0].shape[0]
        pyramid = []
        if self.num_layers == 1:
            projected = projected * 4
        for features, reassemble, size in zip(
            projected, self.reassembly, self.sizes):
            features = reassemble(features)
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
        if self.output_mode == "three_class":
            return logits, embedding
        signed_map = torch.tanh(logits)
        low, high = self.out_range
        if (low, high) != (-1.0, 1.0):
            signed_map = low + (signed_map + 1.0) * 0.5 * (high - low)
        return signed_map, embedding


class DPTDifferenceHead(_DPTDecoder):
    """Separate layer reassembly and residual top-down fusion; backbone stays frozen."""

    def __init__(self, backbone_dim=768, patch_dim=3072, cls_dim=768,
                 grid=37, channels=96, embed_dim=256, out_size=512,
                 out_range=(-1.0, 1.0), num_layers=4, output_mode="signed"):
        if num_layers not in (1, 4) or patch_dim != num_layers * backbone_dim:
            raise ValueError("DPT patch dimension must match one or four backbone layers")
        self._validate_decoder(grid, channels, num_layers, output_mode)
        projections = nn.ModuleList([
            nn.Conv2d(backbone_dim, channels, 1) for _ in range(num_layers)
        ])
        super().__init__(cls_dim, grid, channels, embed_dim, out_size,
                         out_range, num_layers, output_mode)
        self.backbone_dim = backbone_dim
        self.patch_dim = patch_dim
        self.projections = projections

    def _project_streams(self, p_prior, p_curr):
        batch_size = p_prior.shape[0]
        streams = (p_curr - p_prior).split(self.backbone_dim, dim=-1)
        return [project(stream.transpose(1, 2).reshape(
            batch_size, self.backbone_dim, self.grid, self.grid).contiguous())
            for stream, project in zip(streams, self.projections)]

    def forward(self, p_prior, p_curr, cls_prior=None, cls_curr=None):
        expected = (self.grid * self.grid, self.patch_dim)
        if p_prior.shape != p_curr.shape or tuple(p_prior.shape[1:]) != expected:
            raise ValueError(f"Expected matching [batch, {expected[0]}, {expected[1]}] tokens")
        return super().forward(self._project_streams(p_prior, p_curr), cls_prior, cls_curr)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, prior, current):
        norm_prior, norm_current = self.norm(prior), self.norm(current)
        prior_context = self.attention(
            norm_prior, norm_current, norm_current, need_weights=False)[0]
        current_context = self.attention(
            norm_current, norm_prior, norm_prior, need_weights=False)[0]
        prior, current = prior + prior_context, current + current_context
        return (prior + self.ffn(self.ffn_norm(prior)),
                current + self.ffn(self.ffn_norm(current)))


def _temporal_difference(prior, current, project, blocks, grid):
    batch_size, _, width = prior.shape
    prior, current = [
        project(tokens.transpose(1, 2).reshape(batch_size, width, grid, grid).contiguous())
        .flatten(2).transpose(1, 2) for tokens in (prior, current)
    ]
    for block in blocks:
        prior, current = block(prior, current)
    return (current - prior).transpose(1, 2).reshape(
        batch_size, -1, grid, grid).contiguous()


class DPTCrossAttentionHead(DPTDifferenceHead):
    """Bidirectional temporal attention before signed fusion and DPT reassembly."""

    def __init__(self, *args, channels=96, num_heads=4, attention_depth=2, **kwargs):
        super().__init__(*args, channels=channels, **kwargs)
        if num_heads < 1 or channels % num_heads:
            raise ValueError("Attention heads must be positive and divide DPT channels")
        if attention_depth < 1:
            raise ValueError("Attention depth must be positive")
        self.temporal_blocks = nn.ModuleList([
            nn.ModuleList([BidirectionalCrossAttention(channels, num_heads)
                           for _ in range(attention_depth)]) for _ in range(self.num_layers)
        ])

    def _project_streams(self, p_prior, p_curr):
        return [_temporal_difference(prior, current, project, blocks, self.grid)
                for prior, current, project, blocks in zip(
                    p_prior.split(self.backbone_dim, dim=-1),
                    p_curr.split(self.backbone_dim, dim=-1),
                    self.projections, self.temporal_blocks)]


class DPTLayerFusionHead(_DPTDecoder):
    """Fuse three explicitly selected depths before or after temporal attention."""

    def __init__(self, backbone_dim=768, patch_dim=2304, cls_dim=2304,
                 feature_layers=(6, 9, 12), fusion_mode="early",
                 grid=37, channels=96, embed_dim=256, out_size=512,
                 out_range=(-1.0, 1.0), num_heads=4, attention_depth=2,
                 output_mode="three_class"):
        if (len(feature_layers) != 3 or any(
                type(layer) is not int or not 1 <= layer <= 12 for layer in feature_layers)
                or backbone_dim < 1 or patch_dim != 3 * backbone_dim or cls_dim != patch_dim):
            raise ValueError("Fusion requires three ordered depth sources and matching patch/CLS dimensions")
        if fusion_mode not in ("early", "late"):
            raise ValueError("Fusion mode must be early or late")
        if num_heads < 1 or channels % num_heads or attention_depth < 1:
            raise ValueError("Invalid fusion attention heads or depth")
        super().__init__(cls_dim, grid, channels, embed_dim, out_size,
                         out_range, num_layers=1, output_mode=output_mode)
        self.backbone_dim = backbone_dim
        self.patch_dim = patch_dim
        self.feature_layers = tuple(feature_layers)
        self.fusion_mode = fusion_mode
        streams = 1 if fusion_mode == "early" else 3
        self.projections = nn.ModuleList([
            nn.Conv2d(patch_dim if fusion_mode == "early" else backbone_dim, channels, 1)
            for _ in range(streams)
        ])
        self.temporal_blocks = nn.ModuleList([
            nn.ModuleList([BidirectionalCrossAttention(channels, num_heads)
                           for _ in range(attention_depth)]) for _ in range(streams)
        ])
        self.fusion = nn.Identity() if fusion_mode == "early" else nn.Conv2d(3 * channels, channels, 1)
        self.z_proj.requires_grad_(False)

    def forward(self, p_prior, p_curr, cls_prior=None, cls_curr=None):
        expected = (self.grid * self.grid, self.patch_dim)
        if (p_prior.ndim != 3 or p_prior.shape != p_curr.shape
                or tuple(p_prior.shape[1:]) != expected):
            raise ValueError(f"Expected matching [batch, {expected[0]}, {expected[1]}] fusion tokens")
        if (cls_prior is None or cls_curr is None or cls_prior.shape != cls_curr.shape
                or tuple(cls_prior.shape) != (p_prior.shape[0], self.patch_dim)):
            raise ValueError("Fusion requires concatenated own-depth prior/current CLS tokens")
        priors = (p_prior,) if self.fusion_mode == "early" else p_prior.split(self.backbone_dim, dim=-1)
        currents = (p_curr,) if self.fusion_mode == "early" else p_curr.split(self.backbone_dim, dim=-1)
        differences = [_temporal_difference(prior, current, project, blocks, self.grid)
                       for prior, current, project, blocks in zip(
                           priors, currents, self.projections, self.temporal_blocks)]
        fused = self.fusion(differences[0] if self.fusion_mode == "early"
                            else torch.cat(differences, dim=1))
        return super().forward([fused], cls_prior, cls_curr)