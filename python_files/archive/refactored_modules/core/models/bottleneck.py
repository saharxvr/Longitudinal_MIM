"""
Bottleneck and Transformer Modules.

Contains bottleneck architectures that process encoded features:
- TransformerBranch: ViT-based transformer for global context modeling
- FeatureCompression: Reduces feature dimensions before transformer
- BottleneckBlock: Combined CNN and transformer processing
- BottleneckEncoder: Full encoder with bottleneck
- TechnicalBottleneck: Compression bottleneck for latent representations

The bottleneck modules are critical for learning global relationships
and compressing information before the decoder stage.
"""

import torch
import torch.nn as nn
from transformers import ViTConfig

from config import (
    FEATURE_CHANNELS,
    FEATURE_SIZE,
    HIDDEN_CHANNELS,
    INTER_CHANNELS,
    PATCHES_IN_SPATIAL_DIM,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    GROUP_NORM_GROUPS,
    USE_BN,
    BATCH_NORM_MOMENTUM,
    INIT_WEIGHTS,
    INIT_STD,
)

from core.models.blocks import (
    ConvBlockBranch,
    SamplingConvBlock,
)
from core.models.encoders import (
    EfficientNetMiniEncoder,
    FeatureEmbeddings,
)


class TransformerBranch(nn.Module):
    """
    Vision Transformer branch for global context modeling.
    
    Processes flattened feature maps through transformer encoder layers,
    capturing long-range dependencies that CNNs may miss.
    
    Args:
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Features → FeatureEmbeddings → TransformerEncoder → Reshape
    
    Input/Output:
        - Input: (B, 640, 32, 32) spatial features
        - Output: (B, 640, 32, 32) with global context
    
    The transformer processes features as a sequence of patches,
    allowing attention across all spatial positions.
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        # Downsample and create sequence embeddings
        self.downsample = SamplingConvBlock(
            num_convs=2,
            in_channels=FEATURE_CHANNELS,
            out_channels=INTER_CHANNELS,
            sampling_type='down'
        )
        self.embeddings = FeatureEmbeddings(embedding_layers=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_CHANNELS,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_CHANNELS * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        
        # Upsample back to feature resolution
        self.upsample = SamplingConvBlock(
            num_convs=2,
            in_channels=HIDDEN_CHANNELS,
            out_channels=FEATURE_CHANNELS,
            sampling_type='up'
        )
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        for name, param in self.transformer.named_parameters():
            if 'weight' in name and 'linear' in name:
                nn.init.xavier_uniform_(param, gain=1.0)
            elif 'in_proj_weight' in name or 'out_proj.weight' in name:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through transformer.
        
        Args:
            x: Feature tensor (B, C, H, W)
        
        Returns:
            Processed features with global context (B, C, H, W)
        """
        batch_size = x.shape[0]
        
        # Downsample and embed
        x = self.downsample(x)
        x = self.embeddings(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Reshape back to spatial
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        
        # Upsample to original resolution
        x = self.upsample(x)
        
        return x


class FeatureCompression(nn.Module):
    """
    Compresses feature channels before bottleneck processing.
    
    Uses 1x1 convolutions to reduce channel dimensions while
    preserving spatial information.
    
    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Conv1x1 → GroupNorm → ReLU
    """
    
    def __init__(
        self,
        in_channels: int = FEATURE_CHANNELS,
        out_channels: int = HIDDEN_CHANNELS,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels)
        self.relu = nn.ReLU()
        
        if init_weights:
            nn.init.kaiming_normal_(self.compress.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.compress(x)))


class BottleneckBlock(nn.Module):
    """
    Combined CNN and transformer bottleneck block.
    
    Processes features through both CNN (local) and transformer (global)
    branches, then combines the results.
    
    Args:
        transformer: Pre-configured transformer encoder module
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Input → [TransformerBranch, ConvBranch] → Add → ConvBlock
    
    The dual-branch design captures both local texture details (CNN)
    and global structural relationships (transformer).
    """
    
    def __init__(self, transformer: nn.Module, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        # Downsampling and embedding for transformer
        self.down_sample = SamplingConvBlock(
            num_convs=2,
            in_channels=FEATURE_CHANNELS,
            out_channels=INTER_CHANNELS,
            sampling_type='down'
        )
        self.embeddings = FeatureEmbeddings(embedding_layers=1)
        
        # Transformer branch
        self.transformer = transformer
        
        # Upsampling from transformer
        self.up_sample = SamplingConvBlock(
            num_convs=2,
            in_channels=HIDDEN_CHANNELS,
            out_channels=FEATURE_CHANNELS,
            sampling_type='up'
        )
        
        # CNN branch
        self.conv_branch = ConvBlockBranch()
        
        # Final processing
        self.final_conv = ConvBlockBranch()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Transformer branch
        trans_x = self.down_sample(x)
        trans_x = self.embeddings(trans_x)
        trans_x = self.transformer(trans_x)
        trans_x = trans_x.transpose(1, 2).contiguous()
        trans_x = trans_x.view(batch_size, HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        trans_x = self.up_sample(trans_x)
        
        # CNN branch
        conv_x = self.conv_branch(x)
        
        # Combine and process
        combined = trans_x + conv_x
        output = self.final_conv(combined)
        
        return output


class ContrastiveBottleneckBlock(nn.Module):
    """
    Bottleneck block for contrastive learning.
    
    Similar to BottleneckBlock but designed for contrastive representation
    learning, with appropriate architecture for producing embeddings.
    
    Args:
        transformer: Pre-configured transformer encoder layer
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Same as BottleneckBlock but optimized for contrastive objectives.
    """
    
    def __init__(self, transformer: nn.Module, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        self.down_sample = SamplingConvBlock(
            num_convs=2,
            in_channels=FEATURE_CHANNELS,
            out_channels=INTER_CHANNELS,
            sampling_type='down'
        )
        self.embeddings = FeatureEmbeddings(embedding_layers=1)
        self.transformer = transformer
        self.up_sample = SamplingConvBlock(
            num_convs=2,
            in_channels=HIDDEN_CHANNELS,
            out_channels=FEATURE_CHANNELS,
            sampling_type='up'
        )
        self.conv_branch = ConvBlockBranch()
        self.final_conv = ConvBlockBranch()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Transformer path
        trans_x = self.down_sample(x)
        trans_x = self.embeddings(trans_x)
        trans_x = self.transformer(trans_x)
        trans_x = trans_x.transpose(1, 2).contiguous()
        trans_x = trans_x.view(batch_size, HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        trans_x = self.up_sample(trans_x)
        
        # CNN path
        conv_x = self.conv_branch(x)
        
        # Combine
        combined = trans_x + conv_x
        output = self.final_conv(combined)
        
        return output


class BottleneckEncoder(nn.Module):
    """
    Full encoder with bottleneck for masked image modeling.
    
    Combines EfficientNet feature extraction with bottleneck processing
    for self-supervised pre-training tasks.
    
    Args:
        in_channels: Number of input image channels
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        EfficientNetEncoder → BottleneckBlock (with ViT)
    
    Returns both bottleneck output and encoder output for skip connections.
    """
    
    def __init__(self, in_channels: int = 1, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        # Feature encoder
        if in_channels == 1:
            self.encoder = EfficientNetMiniEncoder(dropout=False)
        else:
            self.encoder = EfficientNetMiniEncoder(inp=in_channels, exp=4., dropout=False)
        
        # Create ViT encoder for bottleneck
        from transformers import ViTEncoder
        vit_config = ViTConfig(
            hidden_size=HIDDEN_CHANNELS,
            num_attention_heads=NUM_HEADS,
            num_hidden_layers=NUM_LAYERS,
            intermediate_size=HIDDEN_CHANNELS * 4,
        )
        transformer = ViTEncoder(vit_config)
        
        self.bottleneck = BottleneckBlock(transformer)
    
    def forward(
        self, 
        x: torch.Tensor, 
        get_encoded: bool = False
    ) -> tuple:
        """
        Encode input through encoder and bottleneck.
        
        Args:
            x: Input image tensor (B, C, H, W)
            get_encoded: Whether to also return encoder features
        
        Returns:
            If get_encoded=True: (bottleneck_features, encoder_features)
            Else: bottleneck_features
        """
        encoded = self.encoder(x)
        bottleneck_out = self.bottleneck(encoded)
        
        if get_encoded:
            return bottleneck_out, encoded
        return bottleneck_out


class EncodingBottleneck(nn.Module):
    """
    Lightweight encoding bottleneck for downstream tasks.
    
    Uses ConvBlockBranch and BottleneckBlock for feature processing
    with higher dropout for regularization.
    
    Architecture:
        Dropout2d(0.35) → ConvBlockBranch → BottleneckBlock
    """
    
    def __init__(self):
        super().__init__()
        
        from transformers import ViTConfig, ViTEncoder
        
        vit_config = ViTConfig(
            hidden_size=HIDDEN_CHANNELS,
            num_attention_heads=NUM_HEADS,
            num_hidden_layers=NUM_LAYERS,
        )
        transformer = ViTEncoder(vit_config)
        
        self.dropout2d = nn.Dropout2d(p=0.35)
        self.encoding_layers = nn.Sequential(
            ConvBlockBranch(),
            BottleneckBlock(transformer)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoding_layers(self.dropout2d(x))


class DownstreamBottleneck(nn.Module):
    """
    Bottleneck for downstream classification tasks.
    
    Uses standard PyTorch TransformerEncoderLayer instead of ViT
    for more efficient processing in classification heads.
    
    Architecture:
        Dropout2d(0.35) → ConvBlock → ContrastiveBottleneck → ConvBlock
    """
    
    def __init__(self):
        super().__init__()
        
        self.dropout2d = nn.Dropout2d(p=0.35)
        
        transformer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_CHANNELS,
            nhead=12,
            dropout=0.0,
            batch_first=True
        )
        
        self.contrastive_encoding_layers = nn.Sequential(
            ConvBlockBranch(),
            ContrastiveBottleneckBlock(transformer),
            ConvBlockBranch()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.contrastive_encoding_layers(self.dropout2d(x))


class TechnicalBottleneck(nn.Module):
    """
    Information compression bottleneck for latent representations.
    
    Compresses spatial features to a low-dimensional latent vector,
    useful for visualization, clustering, or auxiliary objectives.
    
    Architecture:
        Down: Conv(640→64) → Conv(64→2) → Flatten → Linear(2048→512)
        Up: Linear(512→2048) → Unflatten → Conv(2→64) → Conv(64→640)
    
    Returns both reconstructed features and the latent representation.
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        # Downsampling path
        down_conv1 = nn.Conv2d(640, 64, kernel_size=3, padding='same', bias=False)
        down_conv2 = nn.Conv2d(64, 2, kernel_size=3, padding='same', bias=False)
        
        # Upsampling path
        up_conv1 = nn.Conv2d(2, 64, kernel_size=3, padding='same', bias=False)
        up_conv2 = nn.Conv2d(64, 640, kernel_size=3, padding='same', bias=False)
        
        flatten = nn.Flatten()
        unflatten = nn.Unflatten(dim=1, unflattened_size=(2, 32, 32))
        
        # Normalization layers
        if USE_BN:
            down_norm1 = nn.BatchNorm2d(64, momentum=BATCH_NORM_MOMENTUM)
            down_norm2 = nn.BatchNorm2d(2, momentum=BATCH_NORM_MOMENTUM)
            up_norm2 = nn.BatchNorm2d(64, momentum=BATCH_NORM_MOMENTUM)
            up_norm3 = nn.BatchNorm2d(640, momentum=BATCH_NORM_MOMENTUM)
        else:
            down_norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=64)
            down_norm2 = nn.GroupNorm(num_groups=2, num_channels=2)
            up_norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=64)
            up_norm3 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=640)
        
        down_norm3 = nn.BatchNorm1d(512, momentum=BATCH_NORM_MOMENTUM)
        up_norm1 = nn.BatchNorm1d(2048, momentum=BATCH_NORM_MOMENTUM)
        
        relu = nn.ReLU()
        
        down_lin = nn.Linear(2048, 512)
        up_lin = nn.Linear(512, 2048)
        
        self.down_block = nn.Sequential(
            down_conv1, relu, down_norm1,
            down_conv2, relu, down_norm2,
            flatten, down_lin, relu, down_norm3
        )
        self.up_block = nn.Sequential(
            up_lin, relu, up_norm1,
            unflatten,
            up_conv1, relu, up_norm2,
            up_conv2, relu, up_norm3
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Compress and reconstruct features.
        
        Args:
            x: Feature tensor (B, 640, 32, 32)
        
        Returns:
            Tuple of (reconstructed_features, latent_vector)
            - reconstructed: (B, 640, 32, 32)
            - latent: (B, 512) detached for auxiliary use
        """
        latent = self.down_block(x)
        reconstructed = self.up_block(latent)
        
        return reconstructed, latent.detach()
