from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import IMG_SIZE, FEATURE_SIZE


def _device_type(device: torch.device | str) -> str:
    dev = str(device)
    return "cuda" if dev.startswith("cuda") else "cpu"


def _base_grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a normalized [-1,1] sampling grid of shape [B,H,W,2]."""
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1)  # [H,W,2]
    return grid.unsqueeze(0).repeat(b, 1, 1, 1)


def warp_image(img: torch.Tensor, flow_px: torch.Tensor, *, mode: str = "bilinear") -> torch.Tensor:
    """Warp an image using a flow field.

    Args:
        img: [B,C,H,W]
        flow_px: [B,2,H,W] in *pixel* units (dx, dy).

    Returns:
        warped: [B,C,H,W]
    """
    b, c, h, w = img.shape
    assert flow_px.shape == (b, 2, h, w), f"Expected flow [B,2,H,W], got {tuple(flow_px.shape)}"

    # Convert pixel displacement to normalized grid displacement.
    # grid_sample expects normalized coords in [-1,1].
    dx = flow_px[:, 0] / ((w - 1) / 2.0)
    dy = flow_px[:, 1] / ((h - 1) / 2.0)
    flow_norm = torch.stack([dx, dy], dim=-1)  # [B,H,W,2]

    grid = _base_grid(b, h, w, img.device, img.dtype) + flow_norm
    warped = F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)
    return warped


class DeformationHead(nn.Module):
    """Predicts a 2D flow field at feature resolution.

    Input:  concatenated features [B, 2*C, Hf, Wf]
    Output: flow [B, 2, Hf, Wf] (dx, dy in *feature-pixel* units)
    """

    def __init__(self, in_channels: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReconstructionHead(nn.Module):
    """Decode encoder features [B,640,32,32] back to an image [B,1,512,512].

    This is intentionally lightweight and independent from the change-map decoder.
    Output uses Sigmoid so it matches the [0,1] min-max normalized inputs.
    """

    def __init__(self, in_channels: int = 640, base_channels: int = 256):
        super().__init__()

        def up(in_ch: int, out_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
            )

        c1 = base_channels
        c2 = base_channels // 2
        c3 = base_channels // 4
        c4 = base_channels // 8

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, c1), num_channels=c1),
            nn.ReLU(inplace=True),
            up(c1, c2),  # 32->64
            up(c2, c3),  # 64->128
            up(c3, c4),  # 128->256
            up(c4, c4),  # 256->512
            nn.Conv2d(c4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


@dataclass
class DeformOutputs:
    flow_feat: torch.Tensor  # [B,2,FEATURE_SIZE,FEATURE_SIZE]
    flow_img: torch.Tensor  # [B,2,IMG_SIZE,IMG_SIZE]
    prior_warped: torch.Tensor  # [B,1,IMG_SIZE,IMG_SIZE]


class LongitudinalMIMDeformV2(nn.Module):
    """V2 model wrapper that keeps the legacy forward contract.

    - `forward(bls, fus)` returns the same output as `LongitudinalMIMModelBig`:
        change map [B,1,512,512]

    - Additional methods enable staged deformation training using triplets.
    """

    def __init__(
        self,
        *,
        use_mask_token: bool = True,
        dec: int = 6,
        patch_dec: bool = False,
        use_pos_embed: bool = False,
        use_technical_bottleneck: bool = False,
        enable_deformation: bool = True,
        enable_reconstruction: bool = True,
    ):
        super().__init__()

        # Import lazily to avoid heavy dependency initialization at module-import time.
        try:
            from models import LongitudinalMIMModelBig
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Failed to import legacy `models.py`. The v2 model reuses the existing base encoder, "
                "which depends on the `transformers` package (HuggingFace). Install it (e.g., `pip install transformers`) "
                "or run in the same environment where your original training works."
            ) from e

        self.backbone = LongitudinalMIMModelBig(
            use_mask_token=use_mask_token,
            dec=dec,
            patch_dec=patch_dec,
            use_pos_embed=use_pos_embed,
            use_technical_bottleneck=use_technical_bottleneck,
        )

        self.enable_deformation = enable_deformation
        if enable_deformation:
            # Backbone encoder outputs FEATURE_CHANNELS (=640) at FEATURE_SIZE (=32).
            self.deform_head = DeformationHead(in_channels=2 * 640, hidden=128)
        else:
            self.deform_head = None

        self.enable_reconstruction = enable_reconstruction
        self.recon_head = ReconstructionHead(in_channels=640) if enable_reconstruction else None

    # -------------------------
    # Legacy forward (unchanged)
    # -------------------------

    def forward(self, bls: torch.Tensor, fus: torch.Tensor) -> torch.Tensor:
        return self.backbone(bls, fus)

    # -------------------------
    # Helpers for staged training
    # -------------------------

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single image to [B,640,32,32] using the shared backbone encoder+bn."""
        feats = self.backbone.encoder(x)
        feats = self.backbone.encoded_bn(feats)
        return feats

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from encoder features.

        Args:
            x: [B,1,512,512] (typically min-max normalized)
        Returns:
            recon: [B,1,512,512] in [0,1]
        """
        if not self.enable_reconstruction or self.recon_head is None:
            raise RuntimeError("Reconstruction head disabled. Instantiate with enable_reconstruction=True.")
        feats = self.backbone.encoded_bn(self.backbone.encoder(x))
        return self.recon_head(feats)

    def predict_deformation(self, prior: torch.Tensor, intermediate: torch.Tensor) -> DeformOutputs:
        """Predict flow to warp `prior` toward `intermediate`.

        This does not change the main `forward()` output contract.

        Returns:
            DeformOutputs with flow at feature and image resolution plus warped prior.
        """
        if not self.enable_deformation or self.deform_head is None:
            raise RuntimeError("Deformation head disabled. Instantiate with enable_deformation=True.")

        # Feature-level flow
        feat_p = self.backbone.encoded_bn(self.backbone.encoder(prior))
        feat_i = self.backbone.encoded_bn(self.backbone.encoder(intermediate))

        flow_feat = self.deform_head(torch.cat([feat_p, feat_i], dim=1))  # [B,2,32,32]

        # Upsample to image resolution. Scale displacement accordingly.
        flow_img = F.interpolate(flow_feat, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        scale = IMG_SIZE / float(FEATURE_SIZE)
        flow_img = flow_img * scale

        prior_warped = warp_image(prior, flow_img)
        return DeformOutputs(flow_feat=flow_feat, flow_img=flow_img, prior_warped=prior_warped)
