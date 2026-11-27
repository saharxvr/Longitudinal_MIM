"""Training-free few-shot baselines on frozen RAD-DINO 6/9/12 difference tokens.

prototype   PANet/FSSDINO-style: masked-average unchanged/positive/negative prototypes of the
            support token differences; cosine softmax on the query differences.
knn         DCAMA without training / Matcher-style dense matching: every query token
            aggregates support label fractions with softmax(cosine / temperature) weights.
Both output p(positive) - p(negative) on the 37x37 grid, bilinearly upsampled to 512x512.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SIZE = 512


def difference_features(prior, current, layers=3):
    """Per-depth L2-normalized (current - prior) token differences, concatenated."""
    chunks = (current - prior).chunk(layers, dim=-1)
    return torch.cat([F.normalize(chunk, dim=-1) for chunk in chunks], -1) / layers ** 0.5


def _to_map(probabilities):
    batch, tokens, _ = probabilities.shape
    grid = int(round(tokens ** 0.5))
    if grid * grid != tokens:
        raise ValueError("Tokens must form a square grid")
    grid = probabilities.transpose(1, 2).reshape(batch, 3, grid, grid)
    grid = F.interpolate(grid, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return grid[:, 1:2] - grid[:, 2:3]


class PrototypePredictor:
    uses_support = True

    def __init__(self, scale=20.0, layers=3):
        self.scale, self.layers = scale, layers

    @torch.no_grad()
    def __call__(self, query, support=None, labels=None, targets=None):
        query_features = difference_features(query[0], query[1], self.layers)
        support_features = difference_features(support[0], support[1], self.layers)
        weights = labels.reshape(-1, 3)
        mass = weights.sum(0)
        prototypes = weights.T @ support_features.reshape(-1, support_features.shape[-1])
        prototypes = F.normalize(prototypes / mass.clamp_min(1e-6)[:, None], dim=-1)
        logits = self.scale * F.normalize(query_features, dim=-1) @ prototypes.T
        logits = logits.masked_fill(mass[None, None] <= 1e-6, float("-inf"))
        return _to_map(logits.softmax(-1))


class KNNPredictor:
    uses_support = True

    def __init__(self, temperature=0.05, layers=3, chunk=4096):
        self.temperature, self.layers, self.chunk = temperature, layers, chunk

    @torch.no_grad()
    def __call__(self, query, support=None, labels=None, targets=None):
        query_features = F.normalize(difference_features(query[0], query[1], self.layers), dim=-1)
        support_features = F.normalize(difference_features(support[0], support[1], self.layers), dim=-1)
        keys = support_features.reshape(-1, support_features.shape[-1])
        values = labels.reshape(-1, 3)
        outputs = []
        for item in query_features:
            maximum = torch.full((item.shape[0], 1), float("-inf"), device=item.device)
            numerator = torch.zeros(item.shape[0], 3, device=item.device)
            denominator = torch.zeros(item.shape[0], 1, device=item.device)
            for start in range(0, keys.shape[0], self.chunk):
                logits = item @ keys[start:start + self.chunk].T / self.temperature
                block_max = logits.amax(-1, keepdim=True)
                new_max = torch.maximum(maximum, block_max)
                rescale = torch.exp(maximum - new_max)
                exponent = torch.exp(logits - new_max)
                numerator = numerator * rescale + exponent @ values[start:start + self.chunk]
                denominator = denominator * rescale + exponent.sum(-1, keepdim=True)
                maximum = new_max
            outputs.append(numerator / denominator)
        return _to_map(torch.stack(outputs))


def build_baselines(names=("prototype", "knn")):
    available = dict(prototype=PrototypePredictor, knn=KNNPredictor)
    return {f"training_free_{name}": available[name]() for name in names}
