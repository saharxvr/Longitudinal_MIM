"""Models for foundation-based contrastive difference detection."""

from .backbone import FrozenCXRBackbone
from .difference_head import DifferenceHead
from .dpt_head import DPTCrossAttentionHead, DPTDifferenceHead
from .projection_heads import DisentangledProjectionHeads


def build_change_head(spec, device="cpu"):
    head_types = {"dglori": DifferenceHead, "dpt": DPTDifferenceHead,
                  "dpt_cross_attention": DPTCrossAttentionHead}
    if spec["type"] not in head_types:
        raise ValueError(f"Unknown head type: {spec['type']}")
    return head_types[spec["type"]](**spec["kwargs"]).to(device)


__all__ = [
    "FrozenCXRBackbone",
    "DifferenceHead",
    "DPTDifferenceHead",
    "DPTCrossAttentionHead",
    "build_change_head",
    "DisentangledProjectionHeads",
]
