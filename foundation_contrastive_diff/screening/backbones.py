"""
Backbone registry for the cheap backbone-fitness screen.

Each backbone exposes the SAME interface so the subset-cache script is backbone-agnostic:

    bb = build_backbone(name)          # frozen, eval
    out = bb(img)                      # img [B,1,H,W] in [0,1]
    out["patch_tokens"]  -> [B, N, D]  # N = grid*grid
    out["cls_token"]     -> [B, D]     # real CLS if available, else global avg-pool
    bb.grid, bb.dim

Covers the non-import-conflicting backbones (rad_dino, dinov2_imagenet, torchxrayvision).
Itamar's reconstruction AE has its own standalone cache script (cache_itamar_ae.py) because
its module tree re-defines `constants`/`models` and clashes with this study's modules.

All backbones are frozen; this is a read-only feature extractor for screening.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_FCD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FCD_ROOT not in sys.path:
    sys.path.insert(0, _FCD_ROOT)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _RadDino(nn.Module):
    grid = 37

    def __init__(self):
        super().__init__()
        from models.backbone import FrozenCXRBackbone
        self.bb = FrozenCXRBackbone(name="rad_dino", last_n_layers=1, freeze=True)
        self.dim = 768

    @torch.no_grad()
    def forward(self, img):
        return self.bb(img)


class _Dinov2ImageNet(nn.Module):
    """General DINOv2 ViT-B/14 (ImageNet) — control for 'is it RAD-DINO or DINO in general'."""
    grid = 37

    def __init__(self, img_size=518):
        super().__init__()
        self.img_size = img_size
        self.dim = 768
        self.net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False
        self.register_buffer("_m", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_s", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, img):
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        img = (img - self._m) / self._s
        out = self.net.forward_features(img)
        return {"patch_tokens": out["x_norm_patchtokens"], "cls_token": out["x_norm_clstoken"]}


class _TorchXrayVision(nn.Module):
    """Supervised CXR DenseNet121 (all-datasets). Low grid (7) but intensity-sensitive."""
    grid = 7

    def __init__(self, weights="densenet121-res224-all", img_size=224):
        super().__init__()
        import torchxrayvision as xrv
        self.img_size = img_size
        self.dim = 1024
        self.net = xrv.models.DenseNet(weights=weights)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, img):
        if img.shape[1] == 3:
            img = img.mean(1, keepdim=True)
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        img = (2.0 * img - 1.0) * 1024.0                      # [0,1] -> xrv [-1024,1024]
        fmap = F.relu(self.net.features(img))                 # [B,1024,7,7]
        b, d, h, w = fmap.shape
        patch = fmap.flatten(2).transpose(1, 2)               # [B,49,1024]
        cls = fmap.mean(dim=(2, 3))                            # global avg-pool pseudo-CLS
        return {"patch_tokens": patch, "cls_token": cls}


_REGISTRY = {
    "rad_dino": _RadDino,
    "dinov2_imagenet": _Dinov2ImageNet,
    "torchxrayvision": _TorchXrayVision,
}


def build_backbone(name: str) -> nn.Module:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; options: {list(_REGISTRY)} (itamar_ae uses cache_itamar_ae.py)")
    return _REGISTRY[name]().eval()
