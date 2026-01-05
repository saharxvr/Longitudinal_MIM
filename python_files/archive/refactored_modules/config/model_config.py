"""
Model Architecture Configuration
================================

Neural network architecture parameters for the Longitudinal CXR Analysis models.

Architecture Overview:
----------------------
The main model uses a dual-branch encoder-decoder architecture:

1. CNN Branch (EfficientNet-B7):
   - Extracts multi-scale local features
   - Output at specific block indices (EFF_NET_BLOCK_IDXS)
   
2. Transformer Branch (ViT):
   - Processes features for global context
   - Uses patch embeddings and self-attention

3. Bottleneck:
   - Combines CNN and transformer features
   - Feature compression and fusion

4. Decoder:
   - Reconstructs images or generates difference maps
   - Skip connections from encoder

Sections:
---------
1. EfficientNet Configuration
2. Feature Dimensions
3. Transformer Configuration
4. Normalization Settings
5. Model Variants

Usage:
------
    from config.model_config import EMBED_DIM, NUM_HEADS, EFF_NET_BLOCK_IDXS
"""

# =============================================================================
# EFFICIENTNET BACKBONE CONFIGURATION
# =============================================================================

EFF_NET_BLOCK_IDXS: tuple = (0, 4, 11, 18, 28, 38, 51, 52)
"""
Indices of EfficientNet-B7 blocks from which to extract features.
Used for multi-scale feature extraction and skip connections.

Block index mapping (approximate):
- 0: Input stem (3 -> 64 channels, 1/2 resolution)
- 4: MBConv block 1 output (64 -> 32, 1/2)
- 11: MBConv block 2 output (32 -> 48, 1/4)
- 18: MBConv block 3 output (48 -> 80, 1/8)  
- 28: MBConv block 4 output (80 -> 160, 1/16)
- 38: MBConv block 5 output (160 -> 224, 1/16)
- 51: MBConv block 6 output (224 -> 384, 1/32)
- 52: Final block (384 -> 640, 1/32)

Used by: models.py - EfficientNetMiniEncoder
"""

# =============================================================================
# FEATURE DIMENSIONS
# =============================================================================

IMG_SIZE: int = 512
"""
Input image size (square).
All chest X-rays are resized to IMG_SIZE x IMG_SIZE.
Chosen as power of 2 for efficient downsampling.
Used by: datasets.py, augmentations.py, all model inputs
"""

FEATURE_SIZE: int = 32
"""
Spatial size of feature maps at bottleneck (IMG_SIZE / 16).
After 4 downsampling stages: 512 -> 256 -> 128 -> 64 -> 32
Used by: models.py - encoder output, decoder input
"""

FEATURE_CHANNELS: int = 640
"""
Number of channels in EfficientNet-B7 final block output.
This is the channel dimension entering the bottleneck.
Fixed by EfficientNet-B7 architecture.
Used by: models.py - BottleneckEncoder input
"""

INTER_CHANNELS: int = 704
"""
Intermediate channel count in bottleneck processing.
Chosen to be divisible by NUM_HEADS (704 / 4 = 176).
Used for attention layers and feature fusion.
Used by: models.py - BottleneckBlock, FeatureCompression
"""

HIDDEN_CHANNELS: int = 768
"""
Hidden dimension matching ViT-Base embedding size.
Allows direct feature fusion with transformer branch.
ViT-Base uses 768-dim embeddings throughout.
Used by: models.py - TransformerBranch, cross-attention
"""

EMBED_DIM: int = 256
"""
Embedding dimension for compressed feature representations.
Used in bottleneck after channel compression.
Smaller than HIDDEN_CHANNELS for efficiency.
Used by: models.py - FeatureEmbeddings, position encoding
"""

NUM_HEADS: int = 4
"""
Number of attention heads in custom transformer layers.
INTER_CHANNELS must be divisible by NUM_HEADS.
704 / 4 = 176 dims per head.
Used by: models.py - multi-head attention in bottleneck
"""

# =============================================================================
# TRANSFORMER/PATCH CONFIGURATION
# =============================================================================

PATCHES_IN_SPATIAL_DIM: int = 16
"""
Number of patches per spatial dimension at bottleneck.
Total patches = PATCHES_IN_SPATIAL_DIM^2 = 256
Matches ViT-Base patch grid for 224x224 with 16x16 patches.
Used by: models.py - patch embedding, position encoding
"""

PATCHES_NUM: int = PATCHES_IN_SPATIAL_DIM ** 2  # 256
"""
Total number of patches (16 x 16 = 256).
Each patch represents a PATCH_SIZE x PATCH_SIZE region.
Used by: models.py - sequence length for transformer
"""

PATCH_SIZE: int = (FEATURE_SIZE ** 2) // PATCHES_NUM  # 4
"""
Size of each patch in feature space.
32^2 / 256 = 4 (2x2 feature map region per patch).
Used by: models.py - patch flattening/unflattening
"""

# =============================================================================
# NORMALIZATION CONFIGURATION
# =============================================================================

USE_BN: bool = False
"""
Whether to use Batch Normalization (True) or Group Normalization (False).
Group Norm preferred for small batch sizes typical in medical imaging.
GN is batch-size independent, more stable for batch_size < 16.
Used by: models.py - all normalization layers
"""

GROUP_NORM_GROUPS: int = 32
"""
Number of groups for Group Normalization.
Channels are divided into this many groups, each normalized separately.
32 is standard choice (works well when channels divisible by 32).
Used by: models.py - nn.GroupNorm layers
"""

BATCH_NORM_EPS: float = 1e-5
"""
Epsilon value for numerical stability in normalization.
Added to denominator to prevent division by zero.
Standard value for both BatchNorm and GroupNorm.
"""

BATCH_NORM_MOMENTUM: float = 0.08
"""
Momentum for running statistics in Batch Normalization.
Lower momentum = more stable but slower adaptation.
Only used when USE_BN=True.
"""

# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

INIT_WEIGHTS: bool = True
"""
Whether to initialize model weights with custom initialization.
True: Use truncated normal initialization (std=INIT_STD)
False: Use PyTorch defaults
Used by: models.py - model constructors
"""

INIT_STD: float = 0.02
"""
Standard deviation for truncated normal weight initialization.
0.02 is common for transformers and works well with layer norm.
Used by: models.py - Linear, Conv2d weight init
"""

# =============================================================================
# MODEL VARIANT FLAGS
# =============================================================================

USE_MASK_TOKEN: bool = False
"""
Whether to use learnable mask tokens in masked image modeling.
True: Replace masked patches with learned token embedding
False: Zero out masked patches
Used by: models.py - MaskedReconstructionModel
"""

USE_POS_EMBED: bool = False
"""
Whether to add positional embeddings to patch features.
True: Add learnable 2D position encoding
False: Rely on CNN's implicit position encoding
Used by: models.py - FeatureEmbeddings
"""

USE_PATCH_DEC: bool = False
"""
Whether to use patch-based decoder (vs convolutional decoder).
True: Transformer-style patch prediction
False: CNN upsampling decoder
Used by: models.py - decoder selection in longitudinal model
"""

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

# Feature dimensions for pooling variants
USE_GLOBAL_POOLING: bool = False
"""
Whether to use global average pooling for classification.
True: Pool to single vector per image
False: Keep spatial dimensions
Used by: models.py - classification heads
"""

GLOBAL_POOLING_FEATURES: int = 4 * FEATURE_CHANNELS  # 2560
"""
Feature dimension after global pooling (4 * 640 = 2560).
Used when USE_GLOBAL_POOLING=True.
"""

NON_GLOBAL_POOLING_FEATURES: int = 2 * (FEATURE_SIZE ** 2)  # 2048
"""
Feature dimension without global pooling (2 * 32^2 = 2048).
Used when USE_GLOBAL_POOLING=False.
"""

LATENT_FEATURES: int = 128
"""
Dimension of latent space for contrastive learning.
Projection head maps features to this dimension.
Used by: models.py - contrastive projection head
"""
