"""
Encoder Architectures for Feature Extraction.

Contains encoder models that transform input images into feature representations:
- EfficientNetMiniEncoder: Pretrained EfficientNet-B7 backbone with selective layer extraction
- FeatureEmbeddings: Converts spatial features to sequence embeddings for transformers

The EfficientNet encoder is the primary feature extractor, providing rich multi-scale
features that are further processed by bottleneck modules.
"""

import torch
import torch.nn as nn
from transformers import EfficientNetModel

from config import (
    EFFICIENT_NET_PRETRAINED_PATH,
    EFF_NET_BLOCK_IDXS,
    FEATURE_SIZE,
    FEATURE_CHANNELS,
    HIDDEN_CHANNELS,
    GROUP_NORM_GROUPS,
    INIT_WEIGHTS,
    INIT_STD,
)


class EfficientNetMiniEncoder(nn.Module):
    """
    Pretrained EfficientNet-B7 encoder with selective block extraction.
    
    Loads a pretrained EfficientNet-B7 model and extracts features from
    specified blocks. Supports channel expansion for multi-channel inputs
    and optional dropout for regularization.
    
    Args:
        inp: Number of input channels (default 1 for grayscale CXR)
        exp: Expansion ratio for first channel expansion (default 3.0)
        exp2: Expansion ratio for second channel expansion (default 1.0, no expansion)
        block_idxs: Which EfficientNet blocks to extract features from
        dropout: Whether to add spatial dropout after encoder
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        InputConv (if inp > 1) → EfficientNet Blocks[block_idxs] → Dropout (optional)
    
    Feature Dimensions:
        - Input: (B, inp, 512, 512)
        - Output: (B, 640, 32, 32) with default block_idxs
    
    Note:
        The model uses pretrained ImageNet weights from HuggingFace.
        For grayscale inputs, the first conv is modified to accept 1 channel.
    
    Example:
        >>> encoder = EfficientNetMiniEncoder(inp=1, dropout=True)
        >>> x = torch.randn(2, 1, 512, 512)
        >>> features = encoder(x)  # Shape: (2, 640, 32, 32)
    """
    
    def __init__(
        self,
        inp: int = 1,
        exp: float = 3.,
        exp2: float = 1.,
        block_idxs: tuple = EFF_NET_BLOCK_IDXS,
        dropout: bool = False,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.block_idxs = block_idxs
        
        # Load pretrained EfficientNet-B7
        efficientnet_model = EfficientNetModel.from_pretrained(EFFICIENT_NET_PRETRAINED_PATH)
        
        # Input projection for multi-channel inputs
        if inp != 1:
            mid_channels = int(inp * exp)
            out_channels = int(mid_channels * exp2)
            
            self.input_proj = nn.Sequential(
                nn.Conv2d(inp, mid_channels, kernel_size=3, padding='same', bias=False),
                nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding='same', bias=False),
                nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels),
                nn.ReLU(),
            )
            
            # Modify first conv to accept projected channels
            original_conv = efficientnet_model.embeddings.convolution
            new_first_conv = nn.Conv2d(
                out_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            efficientnet_model.embeddings.convolution = new_first_conv
        else:
            self.input_proj = None
            # Modify for single-channel input
            original_conv = efficientnet_model.embeddings.convolution
            new_first_conv = nn.Conv2d(
                1, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Initialize with averaged RGB weights
            with torch.no_grad():
                new_first_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
            efficientnet_model.embeddings.convolution = new_first_conv
        
        # Extract only needed components
        self.embeddings = efficientnet_model.embeddings
        
        # Select only the blocks we need
        self.encoder_blocks = nn.ModuleList([
            efficientnet_model.encoder.blocks[i] for i in block_idxs
        ])
        
        # Optional dropout
        self.dropout = nn.Dropout2d(p=0.2) if dropout else None
        
        if init_weights and self.input_proj is not None:
            self._init_weights()
    
    def _init_weights(self):
        if self.input_proj is not None:
            for module in self.input_proj.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(module, nn.GroupNorm):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Feature tensor of shape (B, 640, H/16, W/16)
        """
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        x = self.embeddings(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class FeatureEmbeddings(nn.Module):
    """
    Convert spatial feature maps to sequence embeddings for transformers.
    
    Flattens 2D feature maps and adds learnable positional embeddings
    for use with transformer architectures.
    
    Args:
        embedding_layers: Number of linear projection layers
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Flatten(H×W) → [Linear → LayerNorm → ReLU] × embedding_layers + PosEmbed
    
    Input/Output:
        - Input: (B, C, H, W) spatial features
        - Output: (B, H*W, C) sequence embeddings with positional info
    
    Example:
        >>> embedder = FeatureEmbeddings(embedding_layers=1)
        >>> spatial_features = torch.randn(2, 320, 8, 8)  # After downsampling
        >>> sequence = embedder(spatial_features)  # Shape: (2, 64, 320)
    """
    
    def __init__(
        self,
        embedding_layers: int = 1,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        # Positional embeddings for transformer
        num_patches = (FEATURE_SIZE // 2) ** 2  # After 2x downsampling
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches, HIDDEN_CHANNELS))
        
        # Linear projections
        layers = []
        for _ in range(embedding_layers):
            layers.extend([
                nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
                nn.LayerNorm(HIDDEN_CHANNELS),
                nn.ReLU(),
            ])
        self.linear_proj = nn.Sequential(*layers) if layers else nn.Identity()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=INIT_STD)
        for module in self.linear_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert spatial features to sequence embeddings.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
        
        Returns:
            Sequence tensor of shape (B, H*W, C)
        """
        batch_size = x.shape[0]
        
        # Flatten spatial dimensions: (B, C, H, W) → (B, C, H*W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings
        x = x + self.pos_emb
        
        # Apply linear projections
        x = self.linear_proj(x)
        
        return x
