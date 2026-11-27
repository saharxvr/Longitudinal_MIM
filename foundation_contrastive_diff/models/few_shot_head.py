"""Few-shot (in-context) signed change heads on frozen multi-depth RAD-DINO tokens.

One call segments the query pairs of ONE task: every query in the batch shares the same
support set of ``n >= 0`` pairs with signed ground truth. All matching modules are
permutation invariant over supports and accept any ``n`` (``n = 0`` falls back to learned
null context, i.e. a generic all-change predictor).

Matching variants (identical pair encoder, fusion width and DPT decoder):
    fsdiff    dense label attention + memory attention + prototypes + task-token FiLM
    dcama     dense label attention only (Shi et al., ECCV 2022)
    panet     global masked-average prototypes (Wang et al., ICCV 2019; PAR via alignment_loss)
    alpnet    global + adaptive local prototypes (Ouyang et al., ECCV 2020)
    universeg CrossBlock query/support interaction averaged over supports (Butoi et al., ICCV 2023)
    none      no support context (supervised generic baseline)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_head import BidirectionalCrossAttention, _DPTDecoder

MATCHING_TYPES = ("fsdiff", "dcama", "panet", "alpnet", "universeg", "none")
LABEL_CHANNELS = 3  # unchanged, positive, negative (same order as three-class logits)


def grid_labels(gt, ignore=None, threshold=0.001, grid=37):
    """Signed [n,1,H,W] GT -> [n, grid*grid, 3] (unchanged, positive, negative) area fractions.

    Ignored pixels are excluded from the fractions; fully ignored cells become unchanged.
    """
    if gt.ndim != 4 or gt.shape[1] != 1:
        raise ValueError("Support GT must have shape [n,1,H,W]")
    valid = torch.ones_like(gt, dtype=torch.bool) if ignore is None else ~ignore.bool()
    positive = ((gt > threshold) & valid).float()
    negative = ((gt < -threshold) & valid).float()
    pooled = F.adaptive_avg_pool2d(torch.cat((positive, negative, valid.float()), 1), grid)
    denominator = pooled[:, 2:3].clamp_min(1e-6)
    signs = torch.where(pooled[:, 2:3] > 0, pooled[:, :2] / denominator, torch.zeros_like(pooled[:, :2]))
    signs = signs.clamp(0, 1)
    unchanged = (1 - signs.sum(1, keepdim=True)).clamp_min(0)
    return torch.cat((unchanged, signs), 1).flatten(2).transpose(1, 2).contiguous()


class MultiHeadAttention(nn.Module):
    """Projected attention through SDPA (memory-efficient kernels for long support memories)."""

    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads:
            raise ValueError("Attention heads must divide channels")
        self.num_heads = num_heads
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    def _split(self, tensor):
        batch, length, channels = tensor.shape
        return tensor.view(batch, length, self.num_heads, channels // self.num_heads).transpose(1, 2)

    def forward(self, query, key, value):
        batch = query.shape[0]
        q = self._split(self.query(query))
        k = self._split(self.key(key)).expand(batch, -1, -1, -1)
        v = self._split(self.value(value)).expand(batch, -1, -1, -1)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.output(attended.transpose(1, 2).reshape(batch, query.shape[1], -1))


class MemoryLayer(nn.Module):
    """Query tokens read a support memory whose values carry label embeddings (SAM 2 style)."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.memory_norm = nn.LayerNorm(channels)
        self.attention = MultiHeadAttention(channels, num_heads)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                 nn.Linear(4 * channels, channels))

    def forward(self, tokens, memory_keys, memory_values):
        tokens = tokens + self.attention(self.norm(tokens), self.memory_norm(memory_keys)[None],
                                         memory_values[None])
        return tokens + self.ffn(self.ffn_norm(tokens))


class LabelAttention(nn.Module):
    """DCAMA: query-to-support attention whose values are the support label fractions."""

    def __init__(self, channels, num_heads):
        super().__init__()
        if channels % num_heads or channels // num_heads < LABEL_CHANNELS:
            raise ValueError("Label attention head dimension must hold the three label channels")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.output = nn.Linear(num_heads * LABEL_CHANNELS, channels)

    def forward(self, query_tokens, support_tokens, labels):
        batch, length, _ = query_tokens.shape
        q = self.query(self.norm(query_tokens)).view(batch, length, self.num_heads, -1).transpose(1, 2)
        k = self.key(self.norm(support_tokens)).view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = F.pad(labels, (0, self.head_dim - LABEL_CHANNELS))[None, None].expand(
            batch, self.num_heads, -1, -1)
        aggregated = F.scaled_dot_product_attention(q, k.expand(batch, -1, -1, -1), v)
        aggregated = aggregated[..., :LABEL_CHANNELS].transpose(1, 2).reshape(batch, length, -1)
        return self.output(aggregated)


class TaskEncoder(nn.Module):
    """Per-support pooling by attention (PMA) then a set layer; mean over supports."""

    def __init__(self, channels, num_heads, num_tokens):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)
        self.null = nn.Parameter(torch.zeros(num_tokens, channels))
        self.pool = MultiHeadAttention(channels, num_heads)
        self.pool_norm = nn.LayerNorm(channels)
        self.set_layer = nn.TransformerEncoderLayer(
            channels, num_heads, 4 * channels, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)

    def forward(self, support_tokens):
        if support_tokens is None:
            return self.null
        count = support_tokens.shape[0]
        seeds = self.seeds[None].expand(count, -1, -1)
        normalized = self.pool_norm(support_tokens)
        summaries = seeds + _batched_attention(self.pool, seeds, normalized, normalized)
        summaries = self.set_layer(summaries.reshape(1, -1, summaries.shape[-1]))
        return summaries.view(count, -1, summaries.shape[-1]).mean(0)


def _batched_attention(attention, query, key, value):
    """Run MultiHeadAttention where key/value have their own batch dimension."""
    q = attention._split(attention.query(query))
    k = attention._split(attention.key(key))
    v = attention._split(attention.value(value))
    attended = F.scaled_dot_product_attention(q, k, v)
    return attention.output(attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class CrossBlock(nn.Module):
    """UniverSeg CrossBlock on the token grid: pairwise query/support convs, support mean."""

    def __init__(self, channels, update_support=True):
        super().__init__()
        self.cross = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU())
        self.support = (nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
                        if update_support else None)

    def forward(self, query, supports):
        batch, channels, height, width = query.shape
        if supports.ndim == 4:
            supports = supports[None].expand(batch, -1, -1, -1, -1)
        count = supports.shape[1]
        paired = torch.cat((query[:, None].expand(-1, count, -1, -1, -1), supports), 2)
        interaction = self.cross(paired.reshape(batch * count, 2 * channels, height, width))
        interaction = interaction.view(batch, count, channels, height, width)
        if self.support is None:
            return query + interaction.mean(1), None
        updated = self.support(interaction.reshape(batch * count, channels, height, width))
        return query + interaction.mean(1), updated.view(batch, count, channels, height, width)


class FewShotChangeHead(nn.Module):
    """Frozen-token pair encoder + support matching + existing three-class DPT decoder."""

    def __init__(self, matching="fsdiff", backbone_dim=768, feature_layers=(6, 9, 12), grid=37,
                 channels=96, num_heads=4, attention_depth=2, memory_layers=2, task_tokens=4,
                 local_pool=4, local_threshold=0.3, token_budget=0, out_size=512, embed_dim=256,
                 crossblocks=2):
        super().__init__()
        if matching not in MATCHING_TYPES:
            raise ValueError(f"Unknown matching type {matching!r}; choose from {MATCHING_TYPES}")
        if (not feature_layers or channels % num_heads or attention_depth < 1 or memory_layers < 0
                or task_tokens < 1 or local_pool < 1 or not 0 < local_threshold <= 1
                or token_budget < 0 or crossblocks < 1):
            raise ValueError("Invalid few-shot head configuration")
        self.matching = matching
        self.backbone_dim = backbone_dim
        self.feature_layers = tuple(feature_layers)
        self.patch_dim = backbone_dim * len(self.feature_layers)
        self.grid = grid
        self.channels = channels
        self.local_pool = local_pool
        self.local_threshold = local_threshold
        self.token_budget = token_budget

        self.projection = nn.Conv2d(self.patch_dim, channels, 1)
        self.temporal_blocks = nn.ModuleList([
            BidirectionalCrossAttention(channels, num_heads) for _ in range(attention_depth)])
        self.pair_mix = nn.Sequential(nn.Linear(3 * channels, channels), nn.LayerNorm(channels))
        self.change_logits = nn.Linear(channels, LABEL_CHANNELS)

        uses = dict(fsdiff=("label", "memory", "prototype", "task"), dcama=("label",),
                    panet=("prototype",), alpnet=("prototype",), universeg=("crossblock",),
                    none=())[matching]
        self.uses = set(uses)
        if self.uses & {"label", "memory"}:
            self.position = nn.Parameter(torch.randn(grid * grid, channels) * 0.02)
        if self.uses & {"memory", "crossblock", "task"}:
            self.label_embedding = nn.Linear(LABEL_CHANNELS, channels)
        self.local_prototypes = matching in ("fsdiff", "alpnet")
        fused = channels
        if "label" in self.uses:
            self.label_attention = LabelAttention(channels, num_heads)
            fused += channels
        if "memory" in self.uses:
            self.memory = nn.ModuleList([MemoryLayer(channels, num_heads) for _ in range(memory_layers)])
            fused += channels
        if "prototype" in self.uses:
            self.prototype_scale = nn.Parameter(torch.tensor(10.0))
            self.prototype_embedding = nn.Linear(2 * LABEL_CHANNELS, channels)
            fused += channels
        if "crossblock" in self.uses:
            self.crossblocks = nn.ModuleList([CrossBlock(channels, index + 1 < crossblocks)
                                              for index in range(crossblocks)])
            self.null_cross = nn.Parameter(torch.zeros(channels, 1, 1))
        if "task" in self.uses:
            self.task_encoder = TaskEncoder(channels, num_heads, task_tokens)
            self.film = nn.Linear(task_tokens * channels, 2 * channels)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.fusion = nn.Sequential(nn.Conv2d(fused, channels, 1), nn.GELU(),
                                    nn.Conv2d(channels, channels, 1))
        self.decoder = _DPTDecoder(cls_dim=self.patch_dim, grid=grid, channels=channels,
                                   embed_dim=embed_dim, out_size=out_size, num_layers=1,
                                   output_mode="three_class")
        self.decoder.z_proj.requires_grad_(False)

    # ------------------------------------------------------------------ encoding
    def encode_pairs(self, p_prior, p_curr):
        """[B, N, patch_dim] prior/current tokens -> [B, N, C] pair tokens."""
        expected = (self.grid * self.grid, self.patch_dim)
        if p_prior.ndim != 3 or p_prior.shape != p_curr.shape or tuple(p_prior.shape[1:]) != expected:
            raise ValueError(f"Expected matching [batch, {expected[0]}, {expected[1]}] tokens")
        batch = p_prior.shape[0]

        def project(tokens):
            grid = tokens.transpose(1, 2).reshape(batch, self.patch_dim, self.grid, self.grid)
            return self.projection(grid.contiguous()).flatten(2).transpose(1, 2)

        prior, current = project(p_prior), project(p_curr)
        for block in self.temporal_blocks:
            prior, current = block(prior, current)
        return self.pair_mix(torch.cat((current - prior, current, prior), -1))

    def _select_tokens(self, tokens, labels):
        """Flatten supports to a memory; optionally keep all changed tokens + random background."""
        tokens = tokens.reshape(-1, tokens.shape[-1])
        labels = labels.reshape(-1, LABEL_CHANNELS)
        if not self.token_budget or tokens.shape[0] <= self.token_budget:
            return tokens, labels
        changed = labels[:, 0] < 1 - 1e-6
        changed_index = changed.nonzero().flatten()
        background_index = (~changed).nonzero().flatten()
        if changed_index.numel() > self.token_budget:
            changed_index = changed_index[torch.randperm(changed_index.numel(), device=tokens.device)[:self.token_budget]]
        remaining = self.token_budget - changed_index.numel()
        background_index = background_index[torch.randperm(background_index.numel(), device=tokens.device)[:remaining]]
        index = torch.cat((changed_index, background_index))
        return tokens[index], labels[index]

    # ------------------------------------------------------------------ matching
    def _prototype_maps(self, query, supports, labels):
        """Cosine maps to global (and local) unchanged/positive/negative prototypes: [B, N, 6]."""
        weights = labels.reshape(-1, LABEL_CHANNELS)
        flat = supports.reshape(-1, supports.shape[-1])
        mass = weights.sum(0)
        prototypes = (weights.T @ flat) / mass.clamp_min(1e-6)[:, None]
        normalized_query = F.normalize(query, dim=-1)
        global_maps = normalized_query @ F.normalize(prototypes, dim=-1).T
        global_maps = global_maps.masked_fill(mass[None, None] <= 1e-6, 0.)
        local_maps = global_maps
        if self.local_prototypes:
            count = supports.shape[0]
            grid_features = supports.transpose(1, 2).reshape(count, -1, self.grid, self.grid)
            grid_labels_ = labels.transpose(1, 2).reshape(count, LABEL_CHANNELS, self.grid, self.grid)
            pooled = F.avg_pool2d(grid_features, self.local_pool, ceil_mode=True)
            fractions = F.avg_pool2d(grid_labels_, self.local_pool, ceil_mode=True)
            regions = F.normalize(pooled.flatten(2).transpose(1, 2).reshape(-1, pooled.shape[1]), dim=-1)
            fractions = fractions.flatten(2).transpose(1, 2).reshape(-1, LABEL_CHANNELS)
            similarity = normalized_query @ regions.T
            maps = []
            for label in range(LABEL_CHANNELS):
                valid = fractions[:, label] >= self.local_threshold
                if bool(valid.any()):
                    maps.append(similarity.masked_fill(~valid[None, None], -2.).amax(-1))
                else:
                    maps.append(global_maps[..., label])
            local_maps = torch.stack(maps, -1)
        return self.prototype_scale * torch.cat((global_maps, local_maps), -1)

    def _to_grid(self, tokens):
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, self.grid, self.grid).contiguous()

    def forward(self, q_prior, q_curr, q_cls_prior, q_cls_curr, s_prior=None, s_curr=None,
                s_labels=None, return_aux=False):
        """Return three-class logits [B,3,out,out] (and auxiliary tensors if requested).

        Supports: tokens [n, N, patch_dim] and grid labels [n, N, 3] (see grid_labels);
        pass None (or n = 0) for the no-support condition.
        """
        has_support = s_prior is not None and s_prior.shape[0] > 0
        if has_support and (s_labels is None or s_curr is None or s_labels.shape != (
                s_prior.shape[0], self.grid * self.grid, LABEL_CHANNELS)):
            raise ValueError("Supports need prior/current tokens and [n, N, 3] grid labels")
        batch = q_prior.shape[0]
        if has_support:
            encoded = self.encode_pairs(torch.cat((q_prior, s_prior)), torch.cat((q_curr, s_curr)))
            query, supports = encoded[:batch], encoded[batch:]
        else:
            query, supports = self.encode_pairs(q_prior, q_curr), None
        position = self.position[None] if hasattr(self, "position") else 0.
        features = [query]
        if "label" in self.uses or "memory" in self.uses:
            if has_support:
                memory, memory_labels = self._select_tokens(supports + position, s_labels)
            if "label" in self.uses:
                features.append(self.label_attention(query + position, memory, memory_labels)
                                if has_support else torch.zeros_like(query))
            if "memory" in self.uses:
                read = query + position
                if has_support:
                    values = memory + self.label_embedding(memory_labels)
                    for layer in self.memory:
                        read = layer(read, memory, values)
                    features.append(read)
                else:
                    features.append(torch.zeros_like(query))
        if "prototype" in self.uses:
            maps = (self._prototype_maps(query, supports, s_labels) if has_support
                    else query.new_zeros(batch, query.shape[1], 2 * LABEL_CHANNELS))
            features.append(self.prototype_embedding(maps))
        fused = self.fusion(self._to_grid(torch.cat(features, -1)))
        if "crossblock" in self.uses:
            if has_support:
                support_grid = self._to_grid(supports + self.label_embedding(s_labels))
                for block in self.crossblocks:
                    fused, support_grid = block(fused, support_grid)
            else:
                fused = fused + self.null_cross
        if "task" in self.uses:
            context = self.task_encoder(
                supports + self.label_embedding(s_labels) if has_support else None)
            gamma, beta = self.film(context.reshape(1, -1)).chunk(2, -1)
            fused = fused * (1 + gamma[..., None, None]) + beta[..., None, None]
        logits, _ = self.decoder([fused], q_cls_prior, q_cls_curr)
        if not return_aux:
            return logits
        return logits, dict(query_tokens=query, support_tokens=supports,
                            change_logits=self.change_logits(query))

    # ------------------------------------------------------------------ auxiliary losses
    def alignment_loss(self, query_tokens, query_logits, support_tokens, support_labels):
        """PANet prototype alignment: query predictions define prototypes that must segment supports."""
        if support_tokens is None:
            return query_tokens.new_zeros(())
        probabilities = F.adaptive_avg_pool2d(query_logits.float().softmax(1), self.grid)
        probabilities = probabilities.flatten(2).transpose(1, 2)
        prototypes = probabilities.transpose(1, 2) @ query_tokens / probabilities.sum(1)[..., None].clamp_min(1e-6)
        similarity = getattr(self, "prototype_scale", 10.0) * (
            F.normalize(support_tokens.reshape(-1, support_tokens.shape[-1]), dim=-1)
            @ F.normalize(prototypes, dim=-1).transpose(1, 2))
        targets = support_labels.reshape(-1, LABEL_CHANNELS)[None].expand(similarity.shape[0], -1, -1)
        return -(targets * similarity.log_softmax(-1)).sum(-1).mean()


def build_few_shot_head(spec, device="cpu"):
    if spec.get("type") != "few_shot":
        raise ValueError("Few-shot head spec must have type 'few_shot'")
    return FewShotChangeHead(**spec["kwargs"]).to(device)


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def log_uniform_support_size(maximum, generator=None):
    """Sample n in [1, maximum] log-uniformly (UniverSeg-style support curriculum)."""
    if maximum < 1:
        raise ValueError("Maximum support size must be positive")
    value = torch.rand((), generator=generator).item()
    return min(maximum, max(1, int(math.floor(math.exp(value * math.log(maximum + 1))))))
