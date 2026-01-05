"""
Models Module - Neural Network Architectures for Longitudinal CXR Analysis.

This module contains all neural network components organized hierarchically:
- utils: Model utility functions (parameter counting, weight freezing)
- blocks: Basic building blocks (convolution blocks, channel expansion)
- encoders: Encoder architectures (EfficientNet-based, feature embeddings)
- decoders: Decoder architectures (multi-scale reconstruction decoders)
- bottleneck: Transformer and compression bottleneck modules
- detection: Downstream classification and detection models
- longitudinal: Full longitudinal MIM models for change detection

Architecture Overview:
    Input Images (BL, FU) → Encoder → Bottleneck → Difference Processing → Decoder → Change Map

Key Components:
    - EfficientNetMiniEncoder: Pretrained EfficientNet-B7 feature extractor
    - TransformerBranch: ViT-based transformer for global context
    - ConvBlockBranch: CNN branch for local features
    - Decoder6: Multi-scale upsampling decoder with GroupNorm

Example Usage:
    >>> from core.models import LongitudinalMIMModel
    >>> model = LongitudinalMIMModel(dec=6)
    >>> change_map = model(baseline_image, followup_image)
"""

# Utility functions
from core.models.utils import (
    count_parameters,
    freeze_and_unfreeze,
    weights_check,
)

# Basic building blocks
from core.models.blocks import (
    ChannelExpansionLayer,
    ChannelExpansionBlock,
    SamplingConvBlock,
    ConvBlockBranch,
    ClippedRelu,
)

# Encoder components
from core.models.encoders import (
    EfficientNetMiniEncoder,
    FeatureEmbeddings,
)

# Bottleneck modules
from core.models.bottleneck import (
    TransformerBranch,
    FeatureCompression,
    BottleneckBlock,
    ContrastiveBottleneckBlock,
    BottleneckEncoder,
    EncodingBottleneck,
    DownstreamBottleneck,
    TechnicalBottleneck,
)

# Decoder architectures
from core.models.decoders import (
    DecoderBlock,
    DecoderBlockNoTranspose,
    Decoder1,
    Decoder2,
    Decoder4,
    Decoder5,
    Decoder6,
    PatchDecoder6,
)

# Detection and downstream models
from core.models.detection import (
    Discriminator,
    DownstreamHeads,
    DetectionContrastiveModel,
)

# Full longitudinal models
from core.models.longitudinal import (
    MaskedReconstructionModel,
    LongitudinalMIMModel,
    LongitudinalMIMModelBig,
    LongitudinalMIMModelTest,
    LongitudinalMIMModelBigTransformer,
)

__all__ = [
    # Utils
    "count_parameters",
    "freeze_and_unfreeze", 
    "weights_check",
    # Blocks
    "ChannelExpansionLayer",
    "ChannelExpansionBlock",
    "SamplingConvBlock",
    "ConvBlockBranch",
    "ClippedRelu",
    # Encoders
    "EfficientNetMiniEncoder",
    "FeatureEmbeddings",
    # Bottleneck
    "TransformerBranch",
    "FeatureCompression",
    "BottleneckBlock",
    "ContrastiveBottleneckBlock",
    "BottleneckEncoder",
    "EncodingBottleneck",
    "DownstreamBottleneck",
    "TechnicalBottleneck",
    # Decoders
    "DecoderBlock",
    "DecoderBlockNoTranspose",
    "Decoder1",
    "Decoder2",
    "Decoder4",
    "Decoder5",
    "Decoder6",
    "PatchDecoder6",
    # Detection
    "Discriminator",
    "DownstreamHeads",
    "DetectionContrastiveModel",
    # Longitudinal
    "MaskedReconstructionModel",
    "LongitudinalMIMModel",
    "LongitudinalMIMModelBig",
    "LongitudinalMIMModelTest",
    "LongitudinalMIMModelBigTransformer",
]
