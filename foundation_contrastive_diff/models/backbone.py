"""
Frozen chest X-ray foundation backbone (CheXFound).

Wraps the pretrained **CheXFound** model (Yang et al., IEEE TMI 2025,
arXiv:2502.05142, github.com/RPIDIAL/CheXFound, MIT) and exposes a Siamese-friendly
feature extractor whose weights stay frozen. Both prior and current images pass through
the SAME instance (shared weights). Only the downstream difference head is trained.

CheXFound facts that drive this wrapper:
    - ViT-Large, patch 16, input 512x512 -> 32x32 = 1024 patch tokens + 1 [CLS].
    - Embedding dim D_model = 1024.
    - Pretrained via DINOv2 self-distillation (MIM loss wt 3, [CLS] align wt 1) on CXR-987K.
    - Downstream adaptation concatenates patch tokens from the LAST 4 layers.
    - Released weights: teacher_checkpoint.pth. Requires PyTorch 2.0 + xFormers 0.0.18 + Linux.

Environment: this study runs on the school Linux PCs with CUDA GPUs (the same machines
used to generate the synthetic DRRs), so CheXFound's Linux + xFormers + GPU requirements
are satisfied natively. Because the backbone is frozen, the recommended workflow is still
to PRECOMPUTE patch tokens once (see training/cache_features.py) and train the head on
cached tensors purely for speed/memory efficiency, not as a platform workaround.

Output:
    forward(img) -> dict(patch_tokens [B, N, D], cls_token [B, D])
        where N = NUM_PATCH_TOKENS (1024), D = BACKBONE_DIM (1024).

Alternative backbones for ablation (constants.BACKBONE):
    'rad_dino', 'imagenet_vit', 'parent_efficientnet'.
"""

import torch
import torch.nn as nn


class FrozenCXRBackbone(nn.Module):
    """Frozen CheXFound feature extractor for CXR images."""

    def __init__(
        self,
        name: str = "chexfound",
        checkpoint: str = "",
        config: str = "",
        last_n_layers: int = 4,
        freeze: bool = True,
    ):
        super().__init__()
        self.name = name
        self.last_n_layers = last_n_layers
        self.backbone = self._build_backbone(name, checkpoint, config)

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------
    def _build_backbone(self, name: str, checkpoint: str, config: str) -> nn.Module:
        """Instantiate the chosen pretrained backbone.

        TODO(Phase 1): implement each option.
          - chexfound: vendor github.com/RPIDIAL/CheXFound under third_party/, build the
            ViT-L/16 model from `config`, load `teacher_checkpoint.pth`, and configure it to
            return intermediate tokens from the last `last_n_layers` layers + [CLS].
          - rad_dino / imagenet_vit / parent_efficientnet: ablation baselines (RQ1, ablation E).
        """
        raise NotImplementedError(
            f"Backbone '{name}' not implemented yet — see RQ1_PLAN.md step 1.1."
        )

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        """Disable gradients and keep BN/stat layers in eval mode."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):  # noqa: D401
        """Override so the frozen backbone never leaves eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> dict:
        """Extract frozen CheXFound features from a single CXR.

        Args:
            img: [B, 1, 512, 512] normalized CXR tensor.
        Returns:
            dict with:
                patch_tokens: [B, N, D]  (N=1024 last-4-layer concat tokens, D=1024)
                cls_token:    [B, D]     global [CLS] representation
        """
        return self.backbone(img)
