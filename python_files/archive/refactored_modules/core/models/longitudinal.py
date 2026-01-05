"""
Longitudinal Models for Change Detection.

Contains the main model architectures for longitudinal CXR analysis:
- MaskedReconstructionModel: Self-supervised masked image modeling
- LongitudinalMIMModel: Primary model for detecting changes between timepoints
- LongitudinalMIMModelBig: Extended model with dual processing paths
- LongitudinalMIMModelTest: Lightweight test model
- LongitudinalMIMModelBigTransformer: Model with transformer branches

These models process pairs of baseline (BL) and followup (FU) chest X-rays
to predict difference maps highlighting clinically significant changes.
"""

from math import sqrt
from itertools import chain

import torch
import torch.nn as nn

from config import (
    IMG_SIZE,
    FEATURE_CHANNELS,
    HIDDEN_CHANNELS,
    INTER_CHANNELS,
    PATCHES_IN_SPATIAL_DIM,
    USE_MASK_TOKEN,
    USE_POS_EMBED,
    MASK_PATCH_SIZE,
    INIT_WEIGHTS,
    INIT_STD,
)

from core.models.blocks import (
    ChannelExpansionLayer,
    ChannelExpansionBlock,
    ConvBlockBranch,
    SamplingConvBlock,
)
from core.models.encoders import (
    EfficientNetMiniEncoder,
    FeatureEmbeddings,
)
from core.models.decoders import (
    Decoder1,
    Decoder4,
    Decoder5,
    Decoder6,
    PatchDecoder6,
)
from core.models.bottleneck import BottleneckEncoder, TechnicalBottleneck


class MaskedReconstructionModel(nn.Module):
    """
    Self-supervised masked image modeling (MIM) model.
    
    Pre-training model that learns to reconstruct masked patches,
    following the MAE (Masked Autoencoder) paradigm.
    
    Args:
        use_mask_token: Whether to use learnable mask tokens
        dec: Decoder version to use (1, 4, or 5)
        in_channels: Number of input image channels
        patch_size: Size of patches for masking
        use_pos_embed: Whether to add positional embeddings
    
    Architecture:
        (Optional: PosEmbed + MaskToken) → BottleneckEncoder → Decoder
    
    Pre-training Objective:
        Reconstruct original image from partially masked input.
        Only patches marked as masked contribute to the loss.
    
    Example:
        >>> model = MaskedReconstructionModel(dec=5)
        >>> masked_input = apply_mask(image, mask)
        >>> reconstructed = model(masked_input)
    """
    
    def __init__(
        self,
        use_mask_token: bool = USE_MASK_TOKEN,
        dec: int = 5,
        in_channels: int = 4,
        patch_size: int = MASK_PATCH_SIZE,
        use_pos_embed: bool = USE_POS_EMBED
    ):
        super().__init__()
        
        # Positional embeddings
        if use_pos_embed:
            n_patches = (IMG_SIZE // patch_size) ** 2
            self.pos_emb = nn.Parameter(torch.zeros(1, in_channels * (patch_size ** 2), n_patches))
            self.patch_size = patch_size
        else:
            self.pos_emb = None
        
        # Learnable mask token
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, patch_size ** 2, 1))
        else:
            self.mask_token = None
        
        # Encoder with bottleneck
        self.encoder_bottleneck = BottleneckEncoder(in_channels=in_channels)
        
        # Select decoder
        if dec == 1:
            self.decoder = Decoder1()
        elif dec == 4:
            self.decoder = Decoder4()
        elif dec == 5:
            self.decoder = Decoder5()
        else:
            raise NotImplementedError(f"Decoder {dec} not implemented")
    
    def get_mask_token(self):
        """Get the learnable mask token."""
        return self.mask_token
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from (possibly masked) input.
        
        Args:
            x: Input image (B, C, H, W), potentially with masked patches
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        post_x = x
        
        # Add positional embeddings if enabled
        if self.pos_emb is not None:
            h, w = x.shape[-2:]
            unfolded = nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
            unfolded = unfolded + self.pos_emb
            post_x = nn.functional.fold(
                unfolded,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                output_size=(h, w)
            )
        
        # Encode and decode
        bn_encoded, enc_encoded = self.encoder_bottleneck(post_x, get_encoded=True)
        encoded = bn_encoded + enc_encoded
        decoded = self.decoder(encoded)
        
        return decoded


class LongitudinalMIMModel(nn.Module):
    """
    Primary model for longitudinal change detection (MODEL THAT WORKED FOR id9).
    
    Processes baseline and followup CXR images to produce a difference map
    highlighting clinically significant changes between the two timepoints.
    
    Args:
        use_mask_token: Whether to use learnable mask tokens
        dec: Decoder version (1, 4, 5, or 6)
        patch_dec: Use patch-based decoder variant
        patch_size: Size of patches for masking
        use_pos_embed: Whether to add positional embeddings to inputs
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        BL, FU → SharedEncoder → EncodedBN → (FU - BL) → DiffProcessing → Decoder → ChangeMap
    
    Key Design Choices:
        - Shared encoder for both timepoints (weight sharing)
        - Feature subtraction to compute differences
        - Decoder6 with Tanh for signed change values
    
    Output:
        Change map in range [-1, 1] where:
        - Positive values: New findings in followup
        - Negative values: Resolved findings
        - Zero: No change
    
    Example:
        >>> model = LongitudinalMIMModel(dec=6)
        >>> baseline = torch.randn(2, 1, 512, 512)
        >>> followup = torch.randn(2, 1, 512, 512)
        >>> change_map = model(baseline, followup)  # Shape: (2, 1, 512, 512)
    """
    
    def __init__(
        self,
        use_mask_token: bool = USE_MASK_TOKEN,
        dec: int = 6,
        patch_dec: bool = False,
        patch_size: int = MASK_PATCH_SIZE,
        use_pos_embed: bool = USE_POS_EMBED,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        # Positional embeddings for inputs
        self.use_pos_emb = use_pos_embed
        if self.use_pos_emb:
            self.pos_emb_bl = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.pos_emb_fu = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.patch_size = patch_size
        else:
            self.pos_emb_bl = None
            self.pos_emb_fu = None
        
        # Mask token (optional)
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, patch_size ** 2, 1))
        else:
            self.mask_token = None
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        # Shared encoder for both timepoints
        self.encoder = EfficientNetMiniEncoder(dropout=False)
        
        # Bottleneck processing
        self.encoded_bn = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        )
        
        # Difference processing
        self.diff_processing = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        )
        
        # Select decoder
        if dec == 1:
            self.decoder = Decoder1()
        elif dec == 4:
            self.decoder = Decoder4()
        elif dec == 5:
            self.decoder = Decoder5()
        elif dec == 6:
            if patch_dec:
                self.decoder = PatchDecoder6()
            else:
                self.decoder = Decoder6()
        else:
            raise NotImplementedError(f"Decoder {dec} not implemented")
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        if self.use_pos_emb:
            nn.init.trunc_normal_(self.pos_emb_bl.data, mean=0.0, std=INIT_STD)
            nn.init.trunc_normal_(self.pos_emb_fu.data, mean=0.0, std=INIT_STD)
    
    def get_mask_token(self):
        """Get the learnable mask token."""
        return self.mask_token
    
    def forward(self, bl: torch.Tensor, fu: torch.Tensor) -> torch.Tensor:
        """
        Compute change map between baseline and followup.
        
        Args:
            bl: Baseline image (B, 1, H, W)
            fu: Followup image (B, 1, H, W)
        
        Returns:
            Change map (B, 1, H, W) with values in [-1, 1]
        """
        # Add positional embeddings
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        
        # Encode both timepoints with shared encoder
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        
        # Process through bottleneck
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        
        # Compute feature difference
        encoded_diff = encoded_fu - encoded_bl
        
        # Process difference features
        features = self.diff_processing(encoded_diff)
        features = self.dropout(features)
        
        # Decode to change map
        decoded = self.decoder(features)
        
        return decoded


class LongitudinalMIMModelBig(nn.Module):
    """
    Extended longitudinal model with dual processing paths.
    
    Enhanced version with additional bottleneck processing stage
    and optional technical bottleneck for latent representations.
    
    Args:
        use_mask_token: Whether to use learnable mask tokens
        dec: Decoder version (1, 4, 5, or 6)
        patch_dec: Use patch-based decoder variant
        patch_size: Size of patches for masking
        use_pos_embed: Whether to add positional embeddings
        init_weights: Whether to apply custom weight initialization
        use_technical_bottleneck: Enable latent compression bottleneck
    
    Architecture:
        BL, FU → SharedEncoder → EncodedBN → EncodedBN2 →
        (FU - BL) + (FU2 - BL2) → DiffProcessing → Decoder
    
    Additional Features:
        - Dual bottleneck paths for multi-scale processing
        - Optional technical bottleneck for latent extraction
        - Support for returning latent representations
    """
    
    def __init__(
        self,
        use_mask_token: bool = USE_MASK_TOKEN,
        dec: int = 6,
        patch_dec: bool = False,
        patch_size: int = MASK_PATCH_SIZE,
        use_pos_embed: bool = USE_POS_EMBED,
        init_weights: bool = INIT_WEIGHTS,
        use_technical_bottleneck: bool = False
    ):
        super().__init__()
        
        self.use_pos_emb = use_pos_embed
        if self.use_pos_emb:
            self.pos_emb_bl = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.pos_emb_fu = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.patch_size = patch_size
        else:
            self.pos_emb_bl = None
            self.pos_emb_fu = None
        
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, patch_size ** 2, 1))
        else:
            self.mask_token = None
        
        # Shared encoder
        self.encoder = EfficientNetMiniEncoder(dropout=False)
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        # Primary bottleneck
        self.encoded_bn = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        )
        
        # Secondary bottleneck
        self.encoded_bn2 = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        )
        
        # Feature upsampling
        self.features_upsampling = nn.Sequential(
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        )
        
        # Difference processing stages
        self.diff_processing = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        )
        
        self.diff_processing2 = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        )
        
        # Decoder selection
        if dec == 1:
            self.decoder = Decoder1()
        elif dec == 4:
            self.decoder = Decoder4()
        elif dec == 5:
            self.decoder = Decoder5()
        elif dec == 6:
            if patch_dec:
                self.decoder = PatchDecoder6()
            else:
                self.decoder = Decoder6()
        else:
            raise NotImplementedError(f"Decoder {dec} not implemented")
        
        self.return_latent = False
        self.use_technical_bottleneck = use_technical_bottleneck
        if self.use_technical_bottleneck:
            self.technical_bottleneck = TechnicalBottleneck()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        if self.use_pos_emb:
            nn.init.trunc_normal_(self.pos_emb_bl.data, mean=0.0, std=INIT_STD)
            nn.init.trunc_normal_(self.pos_emb_fu.data, mean=0.0, std=INIT_STD)
    
    def get_mask_token(self):
        return self.mask_token
    
    def forward(self, bl: torch.Tensor, fu: torch.Tensor):
        """
        Compute change map with dual processing paths.
        
        Args:
            bl: Baseline image (B, 1, H, W)
            fu: Followup image (B, 1, H, W)
        
        Returns:
            Change map (B, 1, H, W) or latent vector if return_latent=True
        """
        # Add positional embeddings
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        
        # Encode
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        
        # Primary bottleneck
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        
        # Secondary bottleneck
        encoded_bl2 = self.encoded_bn2(encoded_bl)
        encoded_fu2 = self.encoded_bn2(encoded_fu)
        
        # Primary difference
        encoded_diff = encoded_fu - encoded_bl
        features = self.diff_processing(encoded_diff)
        features = self.dropout(features)
        
        # Secondary difference
        encoded_diff2 = encoded_fu2 - encoded_bl2
        encoded_diff2 = self.dropout(encoded_diff2)
        
        # Combine differences
        features = features + encoded_diff2
        features = self.diff_processing2(features)
        
        # Optional latent extraction
        if not self.use_technical_bottleneck and self.return_latent:
            return torch.flatten(features, 1, -1)
        
        features = self.dropout(features)
        
        if self.use_technical_bottleneck:
            features, latent = self.technical_bottleneck(features)
            if self.return_latent:
                return latent
        
        # Decode
        decoded = self.decoder(features)
        
        return decoded


class LongitudinalMIMModelTest(nn.Module):
    """
    Lightweight test model for quick experiments.
    
    Simplified architecture using channel expansion blocks instead of
    pretrained EfficientNet. Useful for debugging and rapid prototyping.
    
    Architecture:
        BL, FU → ChannelExpansion(1→64) → (FU - BL) → ConvBlock → Conv(1) → Tanh
    """
    
    def __init__(
        self,
        patch_size: int = MASK_PATCH_SIZE,
        use_pos_embed: bool = USE_POS_EMBED,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.mask_token = None
        
        # Lightweight encoders
        self.encoder_bl = nn.Sequential(
            ChannelExpansionBlock(in_channels=1, expand_ratio_1=4., expand_ratio_2=4., first_kernel=5, second_kernel=5),
            ChannelExpansionBlock(in_channels=16, expand_ratio_1=2., expand_ratio_2=2., first_kernel=3, second_kernel=3),
        )
        
        self.encoder_fu = nn.Sequential(
            ChannelExpansionBlock(in_channels=1, expand_ratio_1=4., expand_ratio_2=4., first_kernel=5, second_kernel=5),
            ChannelExpansionBlock(in_channels=16, expand_ratio_1=2., expand_ratio_2=2., first_kernel=3, second_kernel=3),
        )
        
        # Difference processing
        self.diff_enc = nn.Sequential(
            ConvBlockBranch(channels=64),
            ChannelExpansionLayer(64, 0.25)
        )
        
        self.out_layer = nn.Conv2d(16, 1, kernel_size=1)
        self.tanh = nn.Tanh()
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_layer.weight, gain=5/3)
    
    def get_mask_token(self):
        return self.mask_token
    
    def forward(self, bl: torch.Tensor, fu: torch.Tensor) -> torch.Tensor:
        bl_enc = self.encoder_bl(bl)
        fu_enc = self.encoder_fu(fu)
        feat = fu_enc - bl_enc
        feat = self.diff_enc(feat)
        feat = self.tanh(self.out_layer(feat))
        return feat


class LongitudinalMIMModelBigTransformer(nn.Module):
    """
    Longitudinal model with transformer processing branches.
    
    Extended version that incorporates transformer encoder layers
    for global context modeling in both the encoding and difference
    processing stages.
    
    Args:
        use_mask_token: Whether to use learnable mask tokens
        dec: Decoder version (1, 4, 5, or 6)
        patch_dec: Use patch-based decoder variant
        patch_size: Size of patches for masking
        use_pos_embed: Whether to add positional embeddings
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        BL, FU → Encoder → EncodedBN → TransformerBN →
        (FU - BL) processed through CNN + Transformer branches →
        Decoder → ChangeMap
    """
    
    def __init__(
        self,
        use_mask_token: bool = USE_MASK_TOKEN,
        dec: int = 6,
        patch_dec: bool = False,
        patch_size: int = MASK_PATCH_SIZE,
        use_pos_embed: bool = USE_POS_EMBED,
        init_weights: bool = INIT_WEIGHTS
    ):
        super().__init__()
        
        self.use_pos_emb = use_pos_embed
        if self.use_pos_emb:
            self.pos_emb_bl = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.pos_emb_fu = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
            self.patch_size = patch_size
        else:
            self.pos_emb_bl = None
            self.pos_emb_fu = None
        
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, patch_size ** 2, 1))
        else:
            self.mask_token = None
        
        # Shared encoder
        self.encoder = EfficientNetMiniEncoder(dropout=False)
        
        self.dropout = nn.Dropout2d(p=0.1)
        
        # Embedding layers for transformer
        self.emb1 = nn.Sequential(
            SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
            FeatureEmbeddings(1)
        )
        
        self.emb2 = nn.Sequential(
            SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
            FeatureEmbeddings(1)
        )
        
        # Transformer encoder layer
        trans_enc_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_CHANNELS,
            nhead=8,
            dropout=0.0,
            batch_first=True
        )
        
        # Primary bottleneck (CNN)
        self.encoded_bn = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        )
        
        # Feature upsampling paths
        self.features_upsampling1 = nn.Sequential(
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        )
        
        self.features_upsampling2 = nn.Sequential(
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        )
        
        # Secondary bottleneck (Transformer)
        self.encoded_bn2 = nn.Sequential(
            self.emb1,
            nn.TransformerEncoder(trans_enc_layer, num_layers=2),
        )
        
        # Difference processing (CNN path)
        self.diff_processing1 = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        )
        
        # Difference processing (Transformer path)
        self.diff_processing2 = nn.Sequential(
            self.emb2,
            nn.TransformerEncoder(trans_enc_layer, num_layers=2),
        )
        
        # Final difference processing
        self.diff_processing3 = nn.Sequential(
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        )
        
        # Decoder selection
        if dec == 1:
            self.decoder = Decoder1()
        elif dec == 4:
            self.decoder = Decoder4()
        elif dec == 5:
            self.decoder = Decoder5()
        elif dec == 6:
            if patch_dec:
                self.decoder = PatchDecoder6()
            else:
                self.decoder = Decoder6()
        else:
            raise NotImplementedError(f"Decoder {dec} not implemented")
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        if self.use_pos_emb:
            nn.init.trunc_normal_(self.pos_emb_bl.data, mean=0.0, std=INIT_STD)
            nn.init.trunc_normal_(self.pos_emb_fu.data, mean=0.0, std=INIT_STD)
        
        # Initialize transformer weights
        for name, param in chain(
            self.encoded_bn2[1].named_parameters(),
            self.diff_processing2[1].named_parameters()
        ):
            if 'weight' in name and 'linear' in name:
                nn.init.xavier_uniform_(param, gain=sqrt(2.))
            elif 'in_proj_weight' in name or 'out_proj.weight' in name:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
    
    def get_mask_token(self):
        return self.mask_token
    
    def forward(self, bl: torch.Tensor, fu: torch.Tensor) -> torch.Tensor:
        """
        Compute change map using CNN + Transformer branches.
        
        Args:
            bl: Baseline image (B, 1, H, W)
            fu: Followup image (B, 1, H, W)
        
        Returns:
            Change map (B, 1, H, W) with values in [-1, 1]
        """
        # Add positional embeddings
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        
        # Encode with shared encoder
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        
        # Primary bottleneck (CNN)
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        
        # Secondary bottleneck (Transformer)
        encoded_bl2 = self.encoded_bn2(encoded_bl)
        encoded_fu2 = self.encoded_bn2(encoded_fu)
        
        # Reshape transformer outputs
        encoded_bl2 = encoded_bl2.permute(0, 2, 1).contiguous()
        encoded_bl2 = encoded_bl2.view(encoded_bl2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_bl2 = self.features_upsampling1(encoded_bl2)
        
        encoded_fu2 = encoded_fu2.permute(0, 2, 1).contiguous()
        encoded_fu2 = encoded_fu2.view(encoded_fu2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_fu2 = self.features_upsampling1(encoded_fu2)
        
        # CNN difference path
        encoded_diff = encoded_fu - encoded_bl
        features = self.diff_processing1(encoded_diff)
        features = self.dropout(features)
        
        # Transformer difference path
        encoded_diff2 = encoded_fu2 - encoded_bl2
        encoded_diff2 = self.diff_processing2(encoded_diff2)
        encoded_diff2 = encoded_diff2.permute(0, 2, 1).contiguous()
        encoded_diff2 = encoded_diff2.view(encoded_diff2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_diff2 = self.features_upsampling1(encoded_diff2)
        encoded_diff2 = self.dropout(encoded_diff2)
        
        # Combine paths
        features = features + encoded_diff2
        features = self.diff_processing3(features)
        features = self.dropout(features)
        
        # Decode
        decoded = self.decoder(features)
        
        return decoded
