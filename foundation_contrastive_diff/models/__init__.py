"""Models for foundation-based contrastive difference detection."""

from .backbone import FrozenCXRBackbone, checkpoint_feature_options, concatenate_layer_features
from .difference_head import DifferenceHead
from .dpt_head import DPTCrossAttentionHead, DPTDifferenceHead, DPTLayerFusionHead
from .projection_heads import DisentangledProjectionHeads


def build_change_head(spec, device="cpu"):
    head_types = {"dglori": DifferenceHead, "dpt": DPTDifferenceHead,
                  "dpt_cross_attention": DPTCrossAttentionHead, "dpt_layer_fusion": DPTLayerFusionHead}
    if spec["type"] not in head_types:
        raise ValueError(f"Unknown head type: {spec['type']}")
    return head_types[spec["type"]](**spec["kwargs"]).to(device)


def signed_change_map(output):
    """Preserve regression maps; convert class logits to p(positive)-p(negative)."""
    if output.ndim != 4 or output.shape[1] not in (1, 3):
        raise ValueError("Expected a one-channel signed map or three class logits")
    if output.shape[1] == 1:
        return output
    probabilities = output.float().softmax(dim=1)
    return probabilities[:, 1:2] - probabilities[:, 2:3]


__all__ = [
    "FrozenCXRBackbone",
    "DifferenceHead",
    "DPTDifferenceHead",
    "DPTCrossAttentionHead",
    "DPTLayerFusionHead",
    "checkpoint_feature_options",
    "concatenate_layer_features",
    "build_change_head",
    "signed_change_map",
    "DisentangledProjectionHeads",
]
