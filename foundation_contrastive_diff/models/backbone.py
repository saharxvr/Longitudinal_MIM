"""
Frozen chest X-ray foundation backbone.

Default: **RAD-DINO** (`microsoft/rad-dino`, Microsoft) — a frozen DINOv2 ViT-B/14
pretrained on chest X-rays, loaded via HuggingFace `transformers`. Both prior and current
images pass through the SAME frozen instance (Siamese); only the difference head trains.

RAD-DINO facts that drive this wrapper:
    - ViT-Base, patch 14, input 518x518 -> 37x37 = 1369 patch tokens + 1 [CLS], dim 768.
    - DINOv2 self-supervised; may carry register tokens between [CLS] and patch tokens
      (handled by taking the LAST NUM_PATCH_TOKENS tokens).

Input adaptation: grayscale DRR [B,1,H,W] in [0,1] -> repeat to 3 channels, resize to
518, normalize with the model's processor mean/std.

Output:
    forward(img) -> dict(patch_tokens [B, N, D*last_n], cls_token [B, D])
        N = NUM_PATCH_TOKENS (1369), D = BACKBONE_DIM (768).

CheXFound (arXiv:2502.05142) is kept as ablation E and will slot into `_build_backbone`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenCXRBackbone(nn.Module):
    """Frozen foundation feature extractor for CXR images (RAD-DINO by default)."""

    def __init__(
        self,
        name: str = "rad_dino",
        model_id: str = "microsoft/rad-dino",
        checkpoint: str = "",
        config: str = "",
        backbone_img_size: int = 518,
        num_patch_tokens: int = 1369,
        last_n_layers: int = 1,
        freeze: bool = True,
    ):
        super().__init__()
        self.name = name
        self.model_id = model_id
        self.backbone_img_size = int(backbone_img_size)
        self.num_patch_tokens = int(num_patch_tokens)
        self.last_n_layers = int(last_n_layers)

        self.backbone, mean, std = self._build_backbone(name, model_id, checkpoint, config)
        # Normalization buffers (move with .to(device)).
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------
    def _build_backbone(self, name, model_id, checkpoint, config):
        """Instantiate the chosen pretrained backbone; return (module, mean, std)."""
        if name == "rad_dino":
            from transformers import AutoImageProcessor, AutoModel

            model = AutoModel.from_pretrained(model_id)
            try:
                proc = AutoImageProcessor.from_pretrained(model_id)
                mean = list(proc.image_mean)
                std = list(proc.image_std)
            except Exception:
                # RAD-DINO processor defaults if unavailable offline.
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            return model, mean, std

        # chexfound / imagenet_vit / parent_efficientnet: ablations, added later.
        raise NotImplementedError(
            f"Backbone '{name}' not implemented yet — RQ1 first run uses 'rad_dino'."
        )

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):  # noqa: D401
        """Keep the frozen backbone in eval mode regardless of module.train()."""
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        """[B,1|3,H,W] in [0,1] -> [B,3,518,518] normalized for RAD-DINO."""
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        if img.shape[-1] != self.backbone_img_size or img.shape[-2] != self.backbone_img_size:
            img = F.interpolate(
                img, size=(self.backbone_img_size, self.backbone_img_size),
                mode="bilinear", align_corners=False,
            )
        return (img - self._mean) / self._std

    def forward(self, img: torch.Tensor) -> dict:
        """Extract frozen patch tokens + [CLS] from a single CXR.

        Args:
            img: [B, 1, H, W] CXR in [0, 1].
        Returns:
            patch_tokens: [B, NUM_PATCH_TOKENS, BACKBONE_DIM * last_n_layers]
            cls_token:    [B, BACKBONE_DIM]
        """
        x = self._prep(img)
        need_hidden = self.last_n_layers > 1
        out = self.backbone(pixel_values=x, output_hidden_states=need_hidden)

        n = self.num_patch_tokens
        if need_hidden:
            # Concat patch tokens from the last N transformer layers (ablation C).
            layers = out.hidden_states[-self.last_n_layers:]
            patch = torch.cat([h[:, -n:, :] for h in layers], dim=-1)
            cls = out.hidden_states[-1][:, 0, :]
        else:
            hs = out.last_hidden_state          # [B, 1(+regs)+N, D]
            patch = hs[:, -n:, :]               # last N tokens = patches (robust to registers)
            cls = hs[:, 0, :]                   # token 0 = [CLS]
        return {"patch_tokens": patch, "cls_token": cls}
