"""
Basic Neural Network Building Blocks.

Contains fundamental layers and blocks used throughout the model architecture:
- Channel expansion/compression layers for feature dimension control
- Convolutional blocks with normalization
- Sampling blocks for up/down-sampling feature maps

These blocks serve as building blocks for encoders, decoders, and bottlenecks.
"""

import torch
import torch.nn as nn

from config import (
    FEATURE_CHANNELS,
    GROUP_NORM_GROUPS,
    USE_BN,
    BATCH_NORM_MOMENTUM,
    INIT_WEIGHTS,
    INIT_STD,
)


class ChannelExpansionLayer(nn.Module):
    """
    Pointwise convolution layer for channel dimension changes.
    
    Expands or compresses feature channels using 1x1 or larger kernel
    convolutions with GroupNorm and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        expand_ratio: Multiplier for output channels (e.g., 0.5 to halve, 2.0 to double)
        kernel: Convolution kernel size (default 1 for pointwise)
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Conv2d(in_ch, out_ch) → GroupNorm → ReLU
    
    Example:
        >>> layer = ChannelExpansionLayer(640, expand_ratio=0.5)  # 640 → 320
        >>> x = torch.randn(2, 640, 32, 32)
        >>> out = layer(x)  # Shape: (2, 320, 32, 32)
    """
    
    def __init__(
        self,
        in_channels: int,
        expand_ratio: float,
        kernel: int = 1,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.out_channels = int(in_channels * expand_ratio)
        padding = 'same' if kernel > 1 else 0
        
        self.conv = nn.Conv2d(
            in_channels, 
            self.out_channels, 
            kernel_size=kernel, 
            padding=padding, 
            bias=False
        )
        self.norm = nn.GroupNorm(
            num_groups=GROUP_NORM_GROUPS, 
            num_channels=self.out_channels
        )
        self.relu = nn.ReLU()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class ChannelExpansionBlock(nn.Module):
    """
    Two-stage channel expansion block with configurable kernels.
    
    Performs gradual channel expansion through two convolution stages,
    useful for feature extraction in early encoder layers.
    
    Args:
        in_channels: Number of input channels
        expand_ratio_1: Channel multiplier for first stage
        expand_ratio_2: Channel multiplier for second stage (applied to stage 1 output)
        first_kernel: Kernel size for first convolution
        second_kernel: Kernel size for second convolution
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Conv(k1) → GroupNorm → ReLU → Conv(k2) → GroupNorm → ReLU
    
    Example:
        >>> block = ChannelExpansionBlock(1, expand_ratio_1=8., expand_ratio_2=8.)
        >>> x = torch.randn(2, 1, 512, 512)  # Single channel input
        >>> out = block(x)  # Shape: (2, 64, 512, 512)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        expand_ratio_1: float = 8.,
        expand_ratio_2: float = 8.,
        first_kernel: int = 3,
        second_kernel: int = 3,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        mid_channels = int(in_channels * expand_ratio_1)
        out_channels = int(mid_channels * expand_ratio_2)
        
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, 
            kernel_size=first_kernel, 
            padding='same', 
            bias=False
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(GROUP_NORM_GROUPS, mid_channels), 
            num_channels=mid_channels
        )
        
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, 
            kernel_size=second_kernel, 
            padding='same', 
            bias=False
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(GROUP_NORM_GROUPS, out_channels), 
            num_channels=out_channels
        )
        
        self.relu = nn.ReLU()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0.0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        return x


class SamplingConvBlock(nn.Module):
    """
    Convolution block with spatial up/down-sampling.
    
    Combines multiple 3x3 convolutions with strided convolution/transposed
    convolution for spatial resolution changes.
    
    Args:
        num_convs: Number of same-resolution convolutions before sampling
        in_channels: Input channel count
        out_channels: Output channel count
        sampling_type: 'up' for transposed conv, 'down' for strided conv
        init_weights: Whether to apply custom weight initialization
    
    Architecture (down):
        [Conv3x3 → GroupNorm → ReLU] × num_convs → StridedConv(k=4, s=2)
        
    Architecture (up):
        [Conv3x3 → GroupNorm → ReLU] × num_convs → TransposedConv(k=4, s=2)
    
    Example:
        >>> down_block = SamplingConvBlock(2, 640, 320, 'down')
        >>> x = torch.randn(2, 640, 32, 32)
        >>> out = down_block(x)  # Shape: (2, 320, 16, 16)
    """
    
    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        out_channels: int,
        sampling_type: str = 'down',
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        if sampling_type not in ['up', 'down']:
            raise ValueError(f"sampling_type must be 'up' or 'down', got {sampling_type}")
        
        layers = []
        current_channels = in_channels
        
        # Same-resolution convolutions
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding='same', bias=False)
            )
            layers.append(nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels))
            layers.append(nn.ReLU())
            current_channels = out_channels
        
        # Sampling convolution
        if sampling_type == 'down':
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
        else:  # up
            layers.append(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            )
        layers.append(nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels))
        layers.append(nn.ReLU())
        
        self.block = nn.Sequential(*layers)
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        for module in self.block.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlockBranch(nn.Module):
    """
    Convolutional processing block with optional residual connection.
    
    Standard building block for feature processing in bottleneck and 
    encoder stages. Uses Conv → Norm → ReLU pattern.
    
    Args:
        channels: Number of channels (same for input/output)
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Conv3x3 → GroupNorm/BatchNorm → ReLU
    
    Note:
        Uses GroupNorm by default (controlled by USE_BN config).
        BatchNorm can be enabled for training with larger batch sizes.
    
    Example:
        >>> block = ConvBlockBranch(channels=640)
        >>> x = torch.randn(2, 640, 32, 32)
        >>> out = block(x)  # Shape: (2, 640, 32, 32)
    """
    
    def __init__(
        self, 
        channels: int = FEATURE_CHANNELS, 
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding='same', bias=False)
        
        if USE_BN:
            self.norm = nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=channels)
        
        self.relu = nn.ReLU()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class ClippedRelu(nn.Module):
    """
    ReLU activation with upper bound clipping.
    
    Clips output to [0, max_value] range. Useful for constraining
    decoder outputs to valid intensity ranges.
    
    Args:
        max_value: Maximum output value (default 1.0 for normalized images)
    
    Example:
        >>> relu = ClippedRelu(max_value=1.0)
        >>> x = torch.randn(2, 1, 512, 512) * 2  # Values outside [0, 1]
        >>> out = relu(x)  # Clipped to [0, 1]
    """
    
    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0, max=self.max_value)
