"""
D-GLoRI difference head — GLoRI re-purposed for longitudinal change detection.

Sits on top of the frozen CheXFound backbone and is the ONLY trainable component. It
applies the **Global and Local Representations Integration (GLoRI)** idea to the *change*
between two CXRs instead of to a single image.

Inputs are patch tokens + [CLS] for the prior and current images (from FrozenCXRBackbone):
    P_prior, P_curr : [B, N, D]   (N = 1024 patch tokens on a 32x32 grid, D = 1024)
    CLS_prior, CLS_curr : [B, D]  global representations

Pipeline (see RQ1_PLAN.md section 2):
    local   : dP   = fuse(P_prior, P_curr)            # default: P_curr - P_prior
    global  : dCLS = CLS_curr - CLS_prior

    D-GLoRI on dP:
      - linear-embed dP -> D_GLoRI
      - fine-grained branch  : M change-type queries + adaptive-temperature cross-attention
      - coarse-grained branch: pyramid patch merging (8x8 / 4x4 / 2x2 pooling)
      - integrate global dCLS via skip connection

Two outputs:
    1. difference_map : UPerNet-style decoder upsamples the enriched 32x32 grid -> 512x512
                        signed change map in [-1, +1] (positive = new, negative = resolved).
    2. embedding `z`  : global+local change vector for the RQ2 contrastive / disentangle heads.

Fusion modes (constants.FUSION_MODE) for the LOCAL branch:
    - 'diff'            : P_curr - P_prior
    - 'concat'          : token-wise concat -> linear
    - 'cross_attention' : current queries attend to prior tokens
"""

import torch
import torch.nn as nn


class AdaptiveTemperatureAttention(nn.Module):
    """Fine-grained local branch: cross-attention with per-query adaptive temperature.

    Change-type queries attend to the projected patch-difference tokens. An adaptive
    temperature (MLP(avg-pool tokens) -> tanh -> exp) sharpens/smooths each query's
    attention to focus on small or diffuse change regions (CheXFound GLoRI, Eq. 2).
    """

    def __init__(self, dim: int, num_queries: int, num_heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temp_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, num_queries))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO(Phase 1): apply tau to the attention logits (custom scaled-dot-product).
        b = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, tokens, tokens)
        return out  # [B, M, dim] fine-grained local change features


class PyramidPatchMerging(nn.Module):
    """Coarse-grained local branch: multi-scale average pooling of patch-diff tokens.

    Merges 8x8 / 4x4 / 2x2 adjacent tokens, projects + upsamples back, and concatenates
    to provide coarse-grained contextual change features (CheXFound GLoRI, Sec. 3.2.2).
    """

    def __init__(self, dim: int, grid: int = 32):
        super().__init__()
        self.dim = dim
        self.grid = grid
        # TODO(Phase 1): implement 8x8/4x4/2x2 avg-pool + linear(D/3) + upsample + concat -> D.

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # TODO(Phase 1): return coarse-grained local features over the 32x32 grid.
        return tokens


class UPerNetDecoder(nn.Module):
    """UPerNet-style decoder: 32x32 enriched grid -> 512x512 signed change map.

    CheXFound itself trained a UPerNet decoder on frozen features for segmentation; we
    reuse the idea to produce the dense difference map.
    """

    def __init__(self, in_dim: int, grid: int = 32, out_range: tuple = (-1.0, 1.0)):
        super().__init__()
        self.grid = grid
        self.out_range = out_range
        # TODO(Phase 1): proper UPerNet (PPM + FPN). Placeholder progressive upsampler below.
        self.decode = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),   # 32->128
            nn.Conv2d(256, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),   # 128->512
            nn.Conv2d(64, 1, 1), nn.Tanh(),
        )

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        return self.decode(grid_features)


class DifferenceHead(nn.Module):
    """D-GLoRI: integrate global + local change representations into a map and an embedding."""

    def __init__(
        self,
        backbone_dim: int = 1024,
        d_glori: int = 768,
        num_change_queries: int = 7,
        num_heads: int = 8,
        grid: int = 32,
        embed_dim: int = 256,
        fusion_mode: str = "diff",
        use_adaptive_temperature: bool = True,
        use_pyramid_patch_merging: bool = True,
        integrate_global_cls: bool = True,
        out_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.grid = grid
        self.integrate_global_cls = integrate_global_cls

        # Local patch-difference projection -> D_GLoRI.
        in_dim = backbone_dim * 2 if fusion_mode == "concat" else backbone_dim
        self.embed_tokens = nn.Sequential(nn.Linear(in_dim, d_glori), nn.ReLU())

        # Local-feature branches.
        self.fine = AdaptiveTemperatureAttention(d_glori, num_change_queries, num_heads) \
            if use_adaptive_temperature else None
        self.coarse = PyramidPatchMerging(d_glori, grid) if use_pyramid_patch_merging else None

        # Global [CLS] difference skip.
        self.global_proj = nn.Linear(backbone_dim, d_glori) if integrate_global_cls else None

        # Dense decoder for the difference map (operates on the enriched 32x32 grid).
        self.decoder = UPerNetDecoder(d_glori, grid, out_range)

        # Global+local change embedding for RQ2.
        self.embed = nn.Linear(d_glori, embed_dim)

    # ------------------------------------------------------------------
    def _fuse_local(self, p_prior: torch.Tensor, p_curr: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "diff":
            return p_curr - p_prior
        if self.fusion_mode == "concat":
            return torch.cat([p_prior, p_curr], dim=-1)
        if self.fusion_mode == "cross_attention":
            # TODO(Phase 1): current queries attend to prior tokens.
            return p_curr - p_prior
        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, C] -> [B, C, grid, grid]."""
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid, self.grid)

    # ------------------------------------------------------------------
    def forward(self, p_prior, p_curr, cls_prior=None, cls_curr=None):
        """
        Args:
            p_prior, p_curr: [B, N, D] frozen patch tokens (prior / current).
            cls_prior, cls_curr: [B, D] global [CLS] tokens (optional).
        Returns:
            difference_map: [B, 1, 512, 512] signed change map.
            embedding:      [B, embed_dim] global+local change embedding.
        """
        d_tokens = self.embed_tokens(self._fuse_local(p_prior, p_curr))  # [B, N, D_GLoRI]

        # Coarse-grained local features feed the dense decoder grid.
        grid_tokens = self.coarse(d_tokens) if self.coarse is not None else d_tokens
        difference_map = self.decoder(self._to_grid(grid_tokens))

        # Global+local change embedding: pooled local + global [CLS] difference.
        pooled_local = d_tokens.mean(dim=1)
        if self.fine is not None:
            pooled_local = pooled_local + self.fine(d_tokens).mean(dim=1)
        if self.integrate_global_cls and cls_prior is not None and cls_curr is not None:
            pooled_local = pooled_local + self.global_proj(cls_curr - cls_prior)
        embedding = self.embed(pooled_local)

        return difference_map, embedding
