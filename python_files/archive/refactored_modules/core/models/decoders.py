"""
Decoder Architectures for Image Reconstruction.

Contains decoder modules that upsample bottleneck features back to image resolution:
- DecoderBlock: Standard upsampling block with transposed convolution
- DecoderBlockNoTranspose: Upsampling block using interpolation instead
- Decoder1-6: Complete decoder architectures with varying depths
- PatchDecoder6: Specialized decoder for patch-based reconstruction

The decoders reconstruct difference/change maps from encoded features,
with Decoder6 being the primary architecture used for longitudinal analysis.
"""

import torch
import torch.nn as nn

from config import (
    GROUP_NORM_GROUPS,
    USE_BN,
    BATCH_NORM_MOMENTUM,
    INIT_WEIGHTS,
    INIT_STD,
)


class DecoderBlock(nn.Module):
    """
    Standard decoder upsampling block.
    
    Combines transposed convolution for upsampling with regular convolution
    for feature refinement. Supports optional sigmoid output for final layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        deconv_kernel_size: Kernel size for transposed convolution (default 4)
        deconv_stride: Stride for transposed convolution (default 2 for 2x upsample)
        conv_kernel_size: Kernel size for refinement convolution (default 3)
        sigmoid: Whether to apply sigmoid activation at output
        mult: Multiplier before sigmoid (for scaling output range)
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        TransposedConv → GroupNorm → ReLU → Conv → GroupNorm → [Sigmoid × mult | ReLU]
    
    Example:
        >>> block = DecoderBlock(640, 512)
        >>> x = torch.randn(2, 640, 32, 32)
        >>> out = block(x)  # Shape: (2, 512, 64, 64)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv_kernel_size: int = 4,
        deconv_stride: int = 2,
        conv_kernel_size: int = 3,
        sigmoid: bool = False,
        mult: float = 1.,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.sigmoid = sigmoid
        self.mult = mult
        
        # Calculate padding for transposed conv to achieve exact 2x upsampling
        deconv_padding = (deconv_kernel_size - deconv_stride) // 2
        
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=deconv_kernel_size,
            stride=deconv_stride,
            padding=deconv_padding,
            bias=False
        )
        
        if USE_BN:
            self.norm1 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm1 = nn.GroupNorm(
                num_groups=min(GROUP_NORM_GROUPS, out_channels),
                num_channels=out_channels
            )
        
        self.conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=conv_kernel_size,
            padding='same',
            bias=False
        )
        
        if USE_BN:
            self.norm2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm2 = nn.GroupNorm(
                num_groups=min(GROUP_NORM_GROUPS, out_channels),
                num_channels=out_channels
            )
        
        self.relu = nn.ReLU()
        self.sigmoid_fn = nn.Sigmoid()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0.0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.deconv(x)))
        x = self.norm2(self.conv(x))
        
        if self.sigmoid:
            x = self.sigmoid_fn(self.mult * x)
        else:
            x = self.relu(x)
        
        return x


class DecoderBlockNoTranspose(nn.Module):
    """
    Decoder block without transposed convolution.
    
    Uses same-resolution convolutions only (no spatial upsampling).
    Useful for feature refinement without resolution change or when
    upsampling is handled separately.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Conv3x3 → GroupNorm → ReLU → Conv3x3 → GroupNorm → ReLU
    
    Example:
        >>> block = DecoderBlockNoTranspose(384, 256)
        >>> x = torch.randn(2, 384, 64, 64)
        >>> out = block(x)  # Shape: (2, 256, 64, 64) - same spatial size
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.norm1 = nn.GroupNorm(
            num_groups=min(GROUP_NORM_GROUPS, out_channels),
            num_channels=out_channels
        )
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
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


class Decoder1(nn.Module):
    """
    Minimal single-layer decoder.
    
    Fastest decoder using single transposed convolution for 16x upsampling.
    Used for quick experiments or when reconstruction quality is less critical.
    
    Architecture:
        TransposedConv(640→1, k=16, s=16) → Sigmoid
    
    Input: (B, 640, 32, 32) → Output: (B, 1, 512, 512)
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(640, 1, kernel_size=16, stride=16, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        if init_weights:
            nn.init.kaiming_normal_(self.deconv.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.deconv(x))


class Decoder2(nn.Module):
    """
    Two-stage decoder with 4x upsampling per stage.
    
    Architecture:
        DecoderBlock(640→320, s=4) → DecoderBlock(320→1, s=4, sigmoid)
    
    Input: (B, 640, 32, 32) → Output: (B, 1, 512, 512)
    """
    
    def __init__(self):
        super().__init__()
        
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(640, 320, deconv_kernel_size=4, deconv_stride=4),
            DecoderBlock(320, 1, deconv_kernel_size=4, deconv_stride=4, sigmoid=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_blocks(x)


class Decoder4(nn.Module):
    """
    Four-stage decoder with 2x upsampling per stage.
    
    Balanced between speed and reconstruction quality.
    
    Architecture:
        Dropout2d → DecoderBlock(640→480) → DecoderBlock(480→320) →
        DecoderBlock(320→160) → DecoderBlock(160→1, k=5, sigmoid)
    
    Input: (B, 640, 32, 32) → Output: (B, 1, 512, 512)
    """
    
    def __init__(self):
        super().__init__()
        
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(640, 480),
            DecoderBlock(480, 320),
            DecoderBlock(320, 160),
            DecoderBlock(160, 1, sigmoid=True, conv_kernel_size=5)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout2d(x)
        return self.decoder_blocks(x)


class Decoder5(nn.Module):
    """
    Five-stage decoder with dropout regularization.
    
    Architecture:
        DecoderBlock(640→512) → Dropout2d(0.4) → DecoderBlock(512→384) →
        DecoderBlock(384→256, s=1, k=5) → Dropout2d(0.3) →
        DecoderBlock(256→128) → DecoderBlock(128→1, sigmoid, mult=5)
    
    Input: (B, 640, 32, 32) → Output: (B, 1, 512, 512)
    
    Note:
        The mult=5 on final sigmoid helps with gradient flow for difference maps.
    """
    
    def __init__(self):
        super().__init__()
        
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(640, 512),
            nn.Dropout2d(0.4),
            DecoderBlock(512, 384),
            DecoderBlock(384, 256, deconv_kernel_size=1, deconv_stride=1, conv_kernel_size=5),
            nn.Dropout2d(0.3),
            DecoderBlock(256, 128),
            DecoderBlock(128, 1, sigmoid=True, mult=5.)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_blocks(x)


class Decoder6(nn.Module):
    """
    Six-stage decoder with Tanh output (primary decoder).
    
    The main decoder architecture for longitudinal change detection.
    Uses Tanh activation to output signed change values in [-1, 1].
    
    Architecture:
        DecoderBlock(640→512) → DecoderBlock(512→384) →
        DecoderBlock(384→256, s=1, k=5) → DecoderBlock(256→128) →
        DecoderBlock(128→64) → Conv(64→1) → GroupNorm → Tanh
    
    Input: (B, 640, 32, 32) → Output: (B, 1, 512, 512) in range [-1, 1]
    
    Note:
        Tanh output allows representing both positive (new findings) and
        negative (resolved findings) changes in longitudinal comparison.
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(640, 512),
            DecoderBlock(512, 384),
            DecoderBlock(384, 256, deconv_kernel_size=1, deconv_stride=1, conv_kernel_size=5),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Tanh()
        )
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        # Xavier init for output conv to work well with Tanh
        nn.init.xavier_uniform_(self.decoder_blocks[5].weight, gain=5/3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_blocks(x)


class PatchDecoder6(nn.Module):
    """
    Patch-based decoder for masked image modeling.
    
    Similar to Decoder6 but without spatial upsampling in later stages,
    designed for reconstructing masked patches at the feature resolution.
    
    Architecture:
        DecoderBlock(640→512) → Dropout2d(0.2) → DecoderBlock(512→384) →
        DecoderBlockNoTranspose(384→256) → Dropout2d(0.2) →
        DecoderBlockNoTranspose(256→128) → DecoderBlockNoTranspose(128→1) →
        Conv(1→1) → GroupNorm → Sigmoid(2×)
    
    Note:
        Uses scaled sigmoid (2× input) for sharper reconstruction outputs.
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(640, 512),
            nn.Dropout2d(0.2),
            DecoderBlock(512, 384),
            DecoderBlockNoTranspose(384, 256),
            nn.Dropout2d(0.2),
            DecoderBlockNoTranspose(256, 128),
            DecoderBlockNoTranspose(128, 1),
            nn.Conv2d(1, 1, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1)
        )
        self.sig = nn.Sigmoid()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.decoder_blocks[7].weight, mean=0.0, std=INIT_STD)
        nn.init.constant_(self.decoder_blocks[8].weight, 1.0)
        nn.init.constant_(self.decoder_blocks[8].bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_blocks(x)
        decoded = self.sig(2. * decoded)
        return decoded
