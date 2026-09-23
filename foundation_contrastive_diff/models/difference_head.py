"""
D-GLoRI difference head — GLoRI re-purposed for longitudinal change detection.

Sits on top of the frozen foundation backbone (RAD-DINO by default) and is the ONLY
trainable component. It applies the **Global and Local Representations Integration
(GLoRI)** idea to the *change* between two CXRs.

Inputs are patch tokens + [CLS] for prior and current (from FrozenCXRBackbone):
    P_prior, P_curr    : [B, N, D]   (N = grid*grid patch tokens, D = backbone dim)
    CLS_prior, CLS_curr: [B, D]      global representations

Pipeline:
    local  : dP   = fuse(P_prior, P_curr)     (default P_curr - P_prior) -> embed -> grid
    coarse : pyramid pooling (multi-scale context)
    global : dCLS = CLS_curr - CLS_prior skip-connected into the grid
    queries: M change-queries cross-attend the grid (adaptive temperature) and are
             written back onto the grid, so they influence the dense map (and seed z).

Outputs:
    difference_map : [B, 1, out, out] signed change map, Tanh -> [-1, +1].
    embedding z    : [B, embed_dim] global+local change vector (used by RQ2/RQ3).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveTemperatureAttention(nn.Module):
    """Change-query cross-attention with per-query adaptive temperature (GLoRI).

    M learnable queries attend to grid tokens. A per-query temperature tau = exp(MLP(q))
    scales the attention logits, letting each query sharpen (small focal change) or soften
    (diffuse change) its attention. Returns query features and the (head-averaged)
    attention map for write-back / visualization.
    """

    def __init__(self, dim: int, num_heads: int = 8, adaptive_temp: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.h = num_heads
        self.dh = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.adaptive = adaptive_temp
        if adaptive_temp:
            self.temp_mlp = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        B, M, _ = q.shape
        N = kv.shape[1]
        Q = self.q_proj(q).view(B, M, self.h, self.dh).transpose(1, 2)   # [B,h,M,dh]
        K = self.k_proj(kv).view(B, N, self.h, self.dh).transpose(1, 2)  # [B,h,N,dh]
        V = self.v_proj(kv).view(B, N, self.h, self.dh).transpose(1, 2)
        logits = (Q @ K.transpose(-1, -2)) / math.sqrt(self.dh)          # [B,h,M,N]
        if self.adaptive:
            temp = self.temp_mlp(q).clamp(-4, 4).exp()                   # [B,M,1] positive
            logits = logits / temp.unsqueeze(1)                          # broadcast over heads
        attn = logits.softmax(dim=-1)                                    # [B,h,M,N]
        out = (attn @ V).transpose(1, 2).reshape(B, M, self.h * self.dh)  # [B,M,dim]
        return self.out(out), attn.mean(dim=1)                          # [B,M,dim], [B,M,N]


class PyramidContext(nn.Module):
    """PSPNet/UPerNet-style pyramid pooling: multi-scale coarse context over the grid."""

    def __init__(self, dim: int, scales=(1, 2, 3, 6)):
        super().__init__()
        branch = dim // len(scales)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(dim, branch, 1, bias=False),
                nn.GroupNorm(8, branch), nn.GELU(),
            ) for s in scales
        ])
        self.project = nn.Sequential(
            nn.Conv2d(dim + branch * len(scales), dim, 1, bias=False),
            nn.GroupNorm(8, dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x]
        for st in self.stages:
            feats.append(F.interpolate(st(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class UPerNetDecoder(nn.Module):
    """Progressive upsampler: [B, dim, grid, grid] -> [B, 1, out, out], Tanh."""

    def __init__(self, in_dim: int, out_size: int = 512, out_range=(-1.0, 1.0)):
        super().__init__()
        self.out_size = out_size
        self.lo, self.hi = out_range
        chans = [in_dim, 384, 192, 96, 48]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(8, cout), nn.GELU(),
            ))
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Conv2d(chans[-1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        x = torch.tanh(self.final(x))                       # [-1, 1]
        if (self.lo, self.hi) != (-1.0, 1.0):
            x = self.lo + (x + 1.0) * 0.5 * (self.hi - self.lo)
        return x


class DifferenceHead(nn.Module):
    """D-GLoRI: integrate global + local change into a signed heatmap and an embedding."""

    def __init__(
        self,
        backbone_dim: int = 768,
        d_glori: int = 768,
        num_change_queries: int = 8,
        num_heads: int = 8,
        grid: int = 37,
        embed_dim: int = 256,
        fusion_mode: str = "diff",
        use_adaptive_temperature: bool = True,
        use_pyramid_patch_merging: bool = True,
        integrate_global_cls: bool = True,
        out_size: int = 512,
        out_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.grid = grid
        self.integrate_global_cls = integrate_global_cls

        in_dim = backbone_dim * 2 if fusion_mode == "concat" else backbone_dim
        self.embed_tokens = nn.Sequential(nn.Linear(in_dim, d_glori), nn.GELU())
        if fusion_mode == "cross_attention":
            self.cross = nn.MultiheadAttention(backbone_dim, num_heads, batch_first=True)

        self.cls_proj = nn.Linear(backbone_dim, d_glori) if integrate_global_cls else None
        self.pyramid = PyramidContext(d_glori) if use_pyramid_patch_merging else None

        self.queries = nn.Parameter(torch.randn(num_change_queries, d_glori) * 0.02)
        self.attn = AdaptiveTemperatureAttention(d_glori, num_heads, use_adaptive_temperature)

        z_in = d_glori * 2 if integrate_global_cls else d_glori
        self.z_proj = nn.Sequential(nn.Linear(z_in, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

        self.decoder = UPerNetDecoder(d_glori, out_size, out_range)

    # ------------------------------------------------------------------
    def _fuse_local(self, p_prior, p_curr):
        if self.fusion_mode == "diff":
            return p_curr - p_prior
        if self.fusion_mode == "concat":
            return torch.cat([p_prior, p_curr], dim=-1)
        if self.fusion_mode == "cross_attention":
            aligned, _ = self.cross(p_curr, p_prior, p_prior)   # current attends to prior
            return p_curr - aligned
        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def _to_grid(self, tokens):
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, self.grid, self.grid)

    # ------------------------------------------------------------------
    def forward(self, p_prior, p_curr, cls_prior=None, cls_curr=None):
        B = p_prior.shape[0]
        g = self.embed_tokens(self._fuse_local(p_prior, p_curr))        # [B,N,d]

        dcls = None
        if self.integrate_global_cls and cls_prior is not None and cls_curr is not None:
            dcls = self.cls_proj(cls_curr - cls_prior)                  # [B,d]
            g = g + dcls.unsqueeze(1)                                   # global skip

        grid = self._to_grid(g)                                        # [B,d,H,W]
        if self.pyramid is not None:
            grid = self.pyramid(grid)                                  # coarse context
        tokens = grid.flatten(2).transpose(1, 2)                       # [B,N,d]

        # Change-queries attend the grid, then are written back so they shape the map.
        q = self.queries.unsqueeze(0).expand(B, -1, -1)               # [B,M,d]
        q_out, attn = self.attn(q, tokens)                            # [B,M,d], [B,M,N]
        writeback = attn.transpose(1, 2) @ q_out                      # [B,N,d]
        enriched = self._to_grid(tokens + writeback)                 # [B,d,H,W]

        difference_map = self.decoder(enriched)                      # [B,1,out,out]

        z_local = q_out.mean(dim=1)                                  # [B,d]
        z_in = torch.cat([z_local, dcls], dim=-1) if dcls is not None else z_local
        embedding = self.z_proj(z_in)                                # [B,embed_dim]

        return difference_map, embedding
