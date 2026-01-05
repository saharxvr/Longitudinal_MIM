"""
Neural network architectures for Longitudinal CXR Analysis.

This module contains the model architectures for change detection:

Main Models (used in training):
-------------------------------
- LongitudinalMIMModel: Primary model for longitudinal change detection
  Architecture: Shared EfficientNet encoder → Bottleneck → Decoder → Change map
  
- LongitudinalMIMModelBig: Extended version with dual bottleneck paths
  Used in current training configuration.

Supporting Components:
----------------------
Encoders:
- EfficientNetMiniEncoder: Pretrained EfficientNet-B7 feature extractor

Bottleneck:
- BottleneckBlock: CNN + Transformer dual-branch feature processing
- BottleneckEncoder: Full bottleneck with encoding/decoding
- TransformerBranch: ViT-based attention processing

Decoders:
- Decoder6: 6-stage upsampling decoder (32→512) with Tanh output
- Decoder5: 5-stage alternative decoder

Building Blocks:
- ChannelExpansionLayer/Block: Channel dimension expansion
- SamplingConvBlock: Downsampling/upsampling conv blocks
- DecoderBlock: Single decoder stage with skip connections

Model Output:
-------------
All longitudinal models output a change map in range [-1, +1]:
- Positive values: New findings in followup
- Negative values: Resolved findings
- Zero: No change
"""

import torch
import torch.nn as nn
import torchvision.models
from transformers import EfficientNetModel, EfficientNetConfig, ViTForMaskedImageModeling, ViTConfig
from transformers.models.vit.modeling_vit import ViTEncoder
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetEncoder
from constants import *
from math import sqrt
from itertools import chain


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_and_unfreeze(to_freeze: list[nn.Module], to_unfreeze: list[nn.Module]):
    """
    Freeze and unfreeze model parameters for transfer learning.
    
    Args:
        to_freeze: List of modules to freeze (requires_grad=False)
        to_unfreeze: List of modules to unfreeze (requires_grad=True)
    """
    for module in to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    for module in to_unfreeze:
        for param in module.parameters():
            param.requires_grad = True


def weights_check(model):
    """
    Check model weights for max values and NaN.
    
    Returns:
        Tuple of (max_weight, has_nan, nan_weights_list)
    """
    max_weights = []
    has_nan = False
    nan_weights = []
    for param in model.parameters():
        max_weights.append(torch.max(torch.abs(param.data)).item())
        if torch.any(torch.isnan(param.data)).item():
            has_nan = True
            nan_weights.append(param)
    max_weights = torch.tensor(max_weights, dtype=torch.float32)
    max_weight = torch.max(max_weights).item()
    return max_weight, has_nan, nan_weights


def check_if_all_weights_change(model1, model2):
    """Check if any weights differ between two models."""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() == 0:
            return False
    return True


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ChannelExpansionLayer(nn.Module):
    """Convolutional layer that expands channel dimensions."""
    
    def __init__(self, in_channels, out_multiplier, init_weights=INIT_WEIGHTS, kernel=3):
        super().__init__()
        out_channels = int(in_channels * out_multiplier)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding='same', bias=False)
        if USE_BN:
            self.norm = nn.BatchNorm2d(out_channels, eps=BATCH_NORM_EPS, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm = nn.GroupNorm(num_groups=min(out_channels, GROUP_NORM_GROUPS), num_channels=out_channels)
        self.silu = nn.SiLU()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=INIT_STD)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        features = self.norm(features)
        features = self.silu(features)
        return features


class ChannelExpansionBlock(nn.Module):
    def __init__(self, in_channels=1, expand_ratio_1=8., expand_ratio_2=8., first_kernel=7, second_kernel=3):
        super().__init__()
        self.expansion_layer1 = ChannelExpansionLayer(in_channels, expand_ratio_1, kernel=first_kernel)
        self.expansion_layer2 = ChannelExpansionLayer(int(in_channels * expand_ratio_1), expand_ratio_2, kernel=second_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.expansion_layer1(x)
        features = self.expansion_layer2(features)
        return features


class EfficientNetMiniEncoder(nn.Module):
    def __init__(self, block_idxs=EFF_NET_BLOCK_IDXS, inp=1, exp=8., exp2=8., dropout=False):
        super().__init__()
        self.input_expansion_block = ChannelExpansionBlock(in_channels=inp, expand_ratio_1=exp, expand_ratio_2=exp2)
        # self.feature_expansion_block = ChannelExpansionBlock(in_channels=32, expand_ratio_1=1.5, expand_ratio_2=1)

        # eff_model = EfficientNetModel.from_pretrained(EFFICIENTNET_B7_PRETRAINED_PATH)
        config = EfficientNetConfig()
        config.batch_norm_eps = BATCH_NORM_EPS
        config.batch_norm_momentum = BATCH_NORM_MOMENTUM
        eff_model = EfficientNetModel(config)
        # print(eff_model)
        # block_idxs = [0, 4, 11, 18, 28, 38, 51, 52]
        # block_idxs = [11, 18, 28, 38, 51, 52]
        enc = eff_model.encoder
        if dropout:
            self.blocks = []
            for i, idx in enumerate(block_idxs):
                if i % 3 == 0:
                    self.blocks.append(nn.Dropout2d(0.2))
                self.blocks.append(enc.blocks[idx])
            self.blocks = nn.Sequential(*self.blocks)
        else:
            self.blocks = nn.Sequential(*[enc.blocks[idx] for idx in block_idxs])
        # self.blocks = nn.Sequential(*([enc.blocks[0], self.feature_expansion_block] + [enc.blocks[idx] for idx in block_idxs]))

        self.silu = nn.SiLU()

    def forward(self, x):
        features = self.input_expansion_block(x)
        features = self.blocks(features)
        features = self.silu(features)
        return features


class FeatureEmbeddings(nn.Module):
    def __init__(self, kernel_size, in_channels=INTER_CHANNELS, out_channels=HIDDEN_CHANNELS, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.patch_embeddings = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=True)
        self.positional_encodings = nn.Parameter(torch.zeros(1, PATCHES_NUM, out_channels))
        # self.dropout = nn.Dropout(p=0.1)

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embeddings.weight, mean=0.0, std=INIT_STD)
        nn.init.zeros_(self.patch_embeddings.bias)
        nn.init.trunc_normal_(self.positional_encodings.data, mean=0.0, std=INIT_STD)

    def forward(self, x: torch.Tensor):
        emb = self.patch_embeddings(x)
        emb = emb.flatten(2)
        emb = emb.transpose(-1, -2)
        emb = emb + self.positional_encodings
        # emb = self.dropout(emb)

        return emb


class SamplingConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, sampling_type: str, init_weights=INIT_WEIGHTS):
        super().__init__()

        assert sampling_type in {'up', 'down'}

        if sampling_type == 'down':
            self.sampling = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=False)
        else:
            self.sampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=False)

        if USE_BN:
            self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels)

        self.relu = nn.ReLU()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.sampling.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor):
        features = self.sampling(x)
        features = self.norm(features)
        features = self.relu(features)

        return features


class TransformerBranch(nn.Module):
    def __init__(self, transformer_enc):
        super().__init__()

        # kernel_size = int(int(PATCH_SIZE ** 0.5) ** 0.5)  # downsample -> patch_embedding | 2 upsampling layers

        self.downsample = SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down')
        self.embeddings = FeatureEmbeddings(1)
        self.transformer_encoder = transformer_enc
        self.upsample1 = SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up')
        self.upsample2 = SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up')

    def forward(self, x: torch.Tensor):
        features = self.downsample(x)
        emb = self.embeddings(features)
        if type(self.transformer_encoder) is ViTEncoder:
            features = self.transformer_encoder(emb).last_hidden_state
        else:
            features = self.transformer_encoder(emb)
        features = features.permute(0, 2, 1)
        features = features.contiguous().view(features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # features = features.unflatten(2, (PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM))
        features = self.upsample1(features)
        features = self.upsample2(features)
        return features


def init_sequential(seq, sec_relu=False):
    nn.init.kaiming_normal_(seq[0].weight, mode='fan_in', nonlinearity='relu')
    nn.init.constant_(seq[1].weight, 1.0)
    nn.init.constant_(seq[1].bias, 0.0)

    if sec_relu:
        nn.init.kaiming_normal_(seq[3].weight, mode='fan_in', nonlinearity='relu')
    else:
        nn.init.normal_(seq[3].weight, mean=0.0, std=INIT_STD)
    nn.init.constant_(seq[4].weight, 1.0)
    nn.init.constant_(seq[4].bias, 0.0)


class ConvBlockBranch(nn.Module):
    def __init__(self, channels=FEATURE_CHANNELS, kernel_size=3, init_weights=INIT_WEIGHTS):
        super().__init__()

        conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding='same', bias=False)
        conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding='same', bias=False)
        if USE_BN:
            norm1 = nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOMENTUM)
            norm2 = nn.BatchNorm2d(channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=channels)
            norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=channels)
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(conv1, norm1, self.relu, conv2, norm2)

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        init_sequential(self.conv_block, sec_relu=True)

    def forward(self, x: torch.Tensor):
        features = self.conv_block(x)
        features = self.relu(features + x)

        return features


class FeatureCompression(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super().__init__()

        if USE_BN:
            depth_norm = nn.BatchNorm2d(FEATURE_CHANNELS, momentum=BATCH_NORM_MOMENTUM)
            conv_norm1 = nn.BatchNorm2d(2 * FEATURE_CHANNELS, momentum=BATCH_NORM_MOMENTUM)
            conv_norm2 = nn.BatchNorm2d(FEATURE_CHANNELS, momentum=BATCH_NORM_MOMENTUM)
            depthwise_norm1 = nn.BatchNorm2d(3 * FEATURE_CHANNELS, momentum=BATCH_NORM_MOMENTUM)
            depthwise_norm2 = nn.BatchNorm2d(FEATURE_CHANNELS, momentum=BATCH_NORM_MOMENTUM)
        else:
            depth_norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=FEATURE_CHANNELS)
            conv_norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=2 * FEATURE_CHANNELS)
            conv_norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=FEATURE_CHANNELS)
            depthwise_norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=3 * FEATURE_CHANNELS)
            depthwise_norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=FEATURE_CHANNELS)

        self.depth_comp = nn.Sequential(*[nn.Conv2d(3 * FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=1, bias=False),
                                          depth_norm])

        self.conv_comp = nn.Sequential(*[nn.Conv2d(3 * FEATURE_CHANNELS, 2 * FEATURE_CHANNELS, kernel_size=3, padding='same', bias=False),
                                         conv_norm1,
                                         nn.ReLU(),
                                         nn.Conv2d(2 * FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=3, padding='same', bias=False),
                                         conv_norm2
                                         ])

        self.depthwise_spatial_comp = nn.Sequential(*[nn.Conv2d(3 * FEATURE_CHANNELS, 3 * FEATURE_CHANNELS, kernel_size=7, padding='same', bias=False, groups=3 * FEATURE_CHANNELS),
                                                      depthwise_norm1,
                                                      nn.ReLU(),
                                                      nn.Conv2d(3 * FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=7, padding='same', bias=False),
                                                      depthwise_norm2,
                                                      ])

        self.silu = nn.SiLU()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.depth_comp[0].weight, mean=0.0, std=INIT_STD)
        nn.init.constant_(self.depth_comp[1].weight, val=1.0)
        nn.init.constant_(self.depth_comp[1].bias, val=0.0)
        init_sequential(self.conv_comp)
        init_sequential(self.depthwise_spatial_comp)

    def forward(self, x: torch.Tensor):
        depth_out = self.depth_comp(x)
        conv_out = self.conv_comp(x)
        spatial_out = self.depthwise_spatial_comp(x)
        features = self.silu(depth_out + conv_out + spatial_out)

        return features


class BottleneckBlock(nn.Module):
    def __init__(self, transformer_enc):
        super().__init__()

        self.transformer_block = TransformerBranch(transformer_enc)
        self.conv_block = ConvBlockBranch()

        self.feature_compression = FeatureCompression()

    def forward(self, x: torch.Tensor):
        transformer_out = self.transformer_block(x)
        conv_out = self.conv_block(x)

        concat_features = torch.cat([transformer_out, conv_out, x], dim=1)
        compressed_features = self.feature_compression(concat_features)

        return compressed_features


class ContrastiveBottleneckBlock(nn.Module):
    def __init__(self, transformer_enc, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.transformer_block = TransformerBranch(transformer_enc)
        self.conv_block = ConvBlockBranch()

        conv_norm1 = nn.GroupNorm(GROUP_NORM_GROUPS, FEATURE_CHANNELS)
        conv_norm2 = nn.GroupNorm(GROUP_NORM_GROUPS, FEATURE_CHANNELS)

        self.feature_compression = nn.Sequential(*[
            nn.Conv2d(3 * FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=3, padding='same', bias=False),
            conv_norm1,
            nn.ReLU(),
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS, kernel_size=3, padding='same', bias=False),
            conv_norm2,
            nn.SiLU()
        ])

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        init_sequential(self.feature_compression)

    def forward(self, x: torch.Tensor):
        transformer_out = self.transformer_block(x)
        conv_out = self.conv_block(x)

        compressed_features = self.feature_compression(torch.cat([transformer_out, conv_out, x], dim=1))
        return compressed_features


class BottleneckEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # ViT_model = ViTForMaskedImageModeling.from_pretrained(VIT_PRETRAINED_PATH_16_224)
        # transformer = ViT_model.vit.encoder

        conf = ViTConfig()
        transformer = ViTEncoder(conf)

        if in_channels == 1:
            self.encoder = EfficientNetMiniEncoder()
        elif in_channels == 2:
            self.encoder = EfficientNetMiniEncoder(inp=2, exp=4., dropout=True)
        elif in_channels == 4:
            self.encoder = EfficientNetMiniEncoder(inp=4, exp=4., exp2=4.)
        else:
            raise("Invalid input channels num to MaskedReconstructionModel")

        self.dropout2d = nn.Dropout2d(p=0.3)

        self.bottleneck = nn.Sequential(*[ConvBlockBranch(),
                                          nn.Dropout2d(p=0.4),
                                          BottleneckBlock(transformer),
                                          nn.Dropout2d(p=0.4),
                                          ConvBlockBranch(),
                                          ])

    def forward(self, x: torch.Tensor, get_encoded=False):
        encoded = self.encoder(x)
        dropped = self.dropout2d(encoded)
        features = self.bottleneck(dropped)

        if get_encoded:
            return features, encoded

        return features


class ClippedRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.clip(self.relu(x), max=1.0)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, deconv_kernel_size=2, deconv_stride=2, conv_kernel_size=3, sigmoid=False, clipped_relu=False, mult=1.0, init_weights=INIT_WEIGHTS):
        super().__init__()

        mid_channels = (in_channels + out_channels) // 2

        self.mult = mult

        self.deconv = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=deconv_kernel_size, stride=deconv_stride, bias=False)
        if USE_BN:
            self.norm1 = nn.BatchNorm2d(mid_channels, momentum=BATCH_NORM_MOMENTUM)
            self.norm2 = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            self.norm1 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS // 2, num_channels=mid_channels)
            if out_channels > 1:
                self.norm2 = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS // 2, num_channels=out_channels)
            else:
                self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.conv = nn.Conv2d(mid_channels, out_channels, kernel_size=conv_kernel_size, padding='same', bias=False)
        self.relu = nn.ReLU()
        if sigmoid:
            self.final_act = nn.Sigmoid()
        elif clipped_relu:
            self.final_act = ClippedRelu()
        else:
            self.final_act = self.relu

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0.0)
        if type(self.final_act) == nn.Sigmoid:
            nn.init.normal_(self.conv.weight, mean=0.0, std=INIT_STD)
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0.0)

    def forward(self, x: torch.Tensor):
        decoded = self.deconv(x)
        decoded = self.norm1(decoded)
        decoded = self.relu(decoded)
        decoded = self.conv(decoded)
        decoded = self.norm2(decoded)
        decoded = self.final_act(self.mult * decoded)

        return decoded


class DecoderBlockNoTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, sigmoid=False, clipped_relu=False, mult=1.0, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.mult = mult

        if USE_BN:
            self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_MOMENTUM)
        else:
            if out_channels > 1:
                self.norm = nn.GroupNorm(num_groups=GROUP_NORM_GROUPS, num_channels=out_channels)
            else:
                self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding='same', bias=False)
        self.relu = nn.ReLU()
        if sigmoid:
            self.final_act = nn.Sigmoid()
        elif clipped_relu:
            self.final_act = ClippedRelu()
        else:
            self.final_act = self.relu

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        if type(self.final_act) == nn.Sigmoid:
            nn.init.normal_(self.conv.weight, mean=0.0, std=INIT_STD)
        else:
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x: torch.Tensor):
        decoded = self.conv(x)
        decoded = self.norm(decoded)
        decoded = self.final_act(self.mult * decoded)

        return decoded


class Decoder5(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 512),
                                              nn.Dropout2d(0.4),
                                              DecoderBlock(512, 384),
                                              DecoderBlock(384, 256, deconv_kernel_size=1, deconv_stride=1, conv_kernel_size=5),
                                              nn.Dropout2d(0.3),
                                              DecoderBlock(256, 128),
                                              DecoderBlock(128, 1, sigmoid=True, mult=5.)
                                              ])

    def forward(self, x: torch.Tensor):
        decoded = self.decoder_blocks(x)

        return decoded


class Decoder6(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 512),
                                              # nn.Dropout2d(0.1),
                                              DecoderBlock(512, 384),
                                              DecoderBlock(384, 256, deconv_kernel_size=1, deconv_stride=1, conv_kernel_size=5),
                                              # nn.Dropout2d(0.1),
                                              DecoderBlock(256, 128),
                                              DecoderBlock(128, 64),
                                              nn.Conv2d(64, 1, kernel_size=3, padding='same', bias=False),
                                              nn.GroupNorm(num_groups=1, num_channels=1),
                                              nn.Tanh()
                                              # nn.ReLU()
                                              ])
        # self.sig = nn.Sigmoid()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        # nn.init.normal_(self.decoder_blocks[7].weight, mean=0.0, std=INIT_STD)
        nn.init.xavier_uniform_(self.decoder_blocks[5].weight, gain=5/3)
        # nn.init.kaiming_normal_(self.decoder_blocks[5].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.decoder_blocks[8].weight, 1.0)
        # nn.init.constant_(self.decoder_blocks[8].bias, 0.0)

    def forward(self, x: torch.Tensor):
        decoded = self.decoder_blocks(x)
        # decoded = self.sig(2. * decoded)

        return decoded


class PatchDecoder6(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 512),
                                              nn.Dropout2d(0.2),
                                              DecoderBlock(512, 384),
                                              DecoderBlockNoTranspose(384, 256),
                                              nn.Dropout2d(0.2),
                                              DecoderBlockNoTranspose(256, 128),
                                              DecoderBlockNoTranspose(128, 1),
                                              nn.Conv2d(1, 1, kernel_size=3, padding='same', bias=False),
                                              nn.GroupNorm(num_groups=1, num_channels=1)
                                              ])
        self.sig = nn.Sigmoid()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.decoder_blocks[7].weight, mean=0.0, std=INIT_STD)
        nn.init.constant_(self.decoder_blocks[8].weight, 1.0)
        nn.init.constant_(self.decoder_blocks[8].bias, 0.0)

    def forward(self, x: torch.Tensor):
        decoded = self.decoder_blocks(x)
        decoded = self.sig(2. * decoded)

        return decoded


class Decoder4(nn.Module):
    def __init__(self):
        super().__init__()

        # self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 512),
        #                                       DecoderBlock(512, 384),
        #                                       DecoderBlock(384, 256),
        #                                       DecoderBlock(256, 128),
        #                                       DecoderBlock(128, 1, deconv_kernel_size=1, deconv_stride=1, conv_kernel_size=5)
        #                                       ])
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 480),
                                              DecoderBlock(480, 320),
                                              DecoderBlock(320, 160),
                                              DecoderBlock(160, 1, sigmoid=True, conv_kernel_size=5)
                                              ])

    def forward(self, x: torch.Tensor):
        x = self.dropout2d(x)
        decoded = self.decoder_blocks(x)

        return decoded


class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_blocks = nn.Sequential(*[DecoderBlock(640, 320, deconv_kernel_size=4, deconv_stride=4),
                                              DecoderBlock(320, 1, deconv_kernel_size=4, deconv_stride=4, sigmoid=True)
                                              ])

    def forward(self, x: torch.Tensor):
        decoded = self.decoder_blocks(x)

        return decoded


class Decoder1(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(640, 1, kernel_size=16, stride=16, bias=False)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.deconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.sigmoid(self.deconv(x))


class MaskedReconstructionModel(nn.Module):
    def __init__(self, use_mask_token=USE_MASK_TOKEN, dec=5, in_channels=4, patch_size=MASK_PATCH_SIZE, use_pos_embed=USE_POS_EMBED):
        super().__init__()

        if use_pos_embed:
            n_patches = (IMG_SIZE // patch_size) ** 2
            self.pos_emb = nn.Parameter(torch.zeros(1, in_channels * (patch_size ** 2), n_patches))
            self.patch_size = patch_size
        else:
            self.pos_emb = None

        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, patch_size ** 2, 1))
        else:
            self.mask_token = None

        self.encoder_bottleneck = BottleneckEncoder(in_channels=in_channels)

        if dec == 1:
            self.decoder = Decoder1()
        elif dec == 4:
            self.decoder = Decoder4()
        elif dec == 5:
            self.decoder = Decoder5()
        else:
            raise NotImplementedError()

    def get_mask_token(self):
        return self.mask_token

    def forward(self, x: torch.Tensor):
        post_x = x
        if self.pos_emb is not None:
            h, w = x.shape[-2:]
            unfolded = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size) + self.pos_emb
            post_x = torch.nn.functional.fold(unfolded, kernel_size=self.patch_size, stride=self.patch_size, output_size=(h, w))
        bn_encoded, enc_encoded = self.encoder_bottleneck(post_x, get_encoded=True)
        encoded = bn_encoded + enc_encoded
        decoded = self.decoder(encoded)

        return decoded


class Discriminator(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super(Discriminator, self).__init__()

        self.encoder = EfficientNetMiniEncoder(block_idxs=EFF_NET_BLOCK_IDXS[:-1])
        self.bottleneck = nn.Sequential(*[
            ConvBlockBranch(),
            nn.Conv2d(FEATURE_CHANNELS, 2, kernel_size=1),
            nn.GroupNorm(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * FEATURE_SIZE * FEATURE_SIZE, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(),
            nn.Linear(128, 1)
        ])

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.bottleneck[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bottleneck[2].weight, 1.0)
        nn.init.constant_(self.bottleneck[2].bias, 0.0)
        nn.init.kaiming_normal_(self.bottleneck[5].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bottleneck[6].weight, 1.0)
        nn.init.constant_(self.bottleneck[6].bias, 0.0)
        nn.init.normal_(self.bottleneck[8].weight, mean=0.0, std=INIT_STD)

    def forward(self, x):
        return self.bottleneck(self.encoder(x))


class DownstreamHeads(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super(DownstreamHeads, self).__init__()

        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout(p=0.2)

        self.reduce = nn.Sequential(*[
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS // 2, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(GROUP_NORM_GROUPS, FEATURE_CHANNELS // 2),
            nn.ReLU(),
            nn.Conv2d(FEATURE_CHANNELS // 2, 2, kernel_size=1),
            nn.GroupNorm(2, 2),
            nn.ReLU(),
            nn.Flatten()
        ])
        # self.lin_block = nn.Sequential(*[
        #     nn.Linear(NON_GLOBAL_POOLING_FEATURES, 1024),
        #     nn.LayerNorm(normalized_shape=1024),
        #     nn.ReLU()
        # ])
        # self.projection = nn.Linear(1024, LATENT_FEATURES)
        # self.pred_head = nn.Sequential(*[
        #     nn.Linear(1024, 256),
        #     nn.LayerNorm(normalized_shape=256),
        #     nn.ReLU(),
        #     nn.Linear(256, CUR_LABELS_NUM)
        # ])
        self.early_projection = nn.Linear(2048, LATENT_FEATURES)
        self.lin_block = nn.Sequential(*[
            nn.Linear(NON_GLOBAL_POOLING_FEATURES, 1024),
            nn.LayerNorm(normalized_shape=1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU()
        ])
        self.late_projection = nn.Linear(256, LATENT_FEATURES)
        self.pred_head = nn.Sequential(*[
            nn.Linear(256, CUR_LABELS_NUM)
        ])

        self.use_contrastive = False

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.reduce[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.reduce[1].weight, val=1.0)
        nn.init.constant_(self.reduce[1].bias, val=0.0)
        nn.init.kaiming_normal_(self.reduce[3].weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.reduce[4].weight, val=1.0)
        nn.init.constant_(self.reduce[4].bias, val=0.0)
        nn.init.kaiming_normal_(self.lin_block[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.early_projection.weight, mean=0.0, std=INIT_STD)
        nn.init.normal_(self.late_projection.weight, mean=0.0, std=INIT_STD)
        nn.init.kaiming_normal_(self.pred_head[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.lin_block[3].weight, mean=0.0, std=INIT_STD)

    def forward(self, x):
        x = self.dropout1(x)
        x_reduced = self.reduce(x)
        x = self.dropout2(x_reduced)
        x = self.lin_block(x)
        preds = self.pred_head(x)
        if not self.use_contrastive:
            return preds, None
        else:
            late_projected = self.late_projection(x)
            return preds, late_projected

    def set_use_contrastive(self, val):
        self.use_contrastive = val


class EncodingBottleneck(nn.Module):
    def __init__(self):
        super(EncodingBottleneck, self).__init__()

        conf = ViTConfig()
        transformer = ViTEncoder(conf)

        self.dropout2d = nn.Dropout2d(p=0.35)

        self.encoding_layers = nn.Sequential(*[
            ConvBlockBranch(),
            BottleneckBlock(transformer)
        ])

    def forward(self, x):
        features = self.encoding_layers(self.dropout2d(x))
        return features


class DownstreamBottleneck(nn.Module):
    def __init__(self):
        super(DownstreamBottleneck, self).__init__()

        self.dropout2d = nn.Dropout2d(p=0.35)

        transformer = nn.TransformerEncoderLayer(d_model=HIDDEN_CHANNELS, nhead=12, dropout=0.0, batch_first=True)

        self.contrastive_encoding_layers = nn.Sequential(*[
            ConvBlockBranch(),
            ContrastiveBottleneckBlock(transformer),
            ConvBlockBranch()
        ])

    def forward(self, x):
        features = self.contrastive_encoding_layers(self.dropout2d(x))
        return features


class DetectionContrastiveModel(nn.Module):
    def __init__(self, in_channels=1):
        super(DetectionContrastiveModel, self).__init__()

        if in_channels == 1:
            self.encoder = EfficientNetMiniEncoder(dropout=True)
        elif in_channels == 2:
            self.encoder = EfficientNetMiniEncoder(inp=2, exp=4., dropout=True)
        elif in_channels == 4:
            self.encoder = EfficientNetMiniEncoder(inp=4, exp=4., exp2=4., dropout=True)
        else:
            raise "Invalid input channels num to DetectionContrastiveModel"

        self.encoding_bottleneck = EncodingBottleneck()
        self.e_b = nn.ModuleList([self.encoder, self.encoding_bottleneck])
        self.downstream_bottleneck = DownstreamBottleneck()
        self.heads = DownstreamHeads()

    def detection_stage(self):
        self.heads.set_use_contrastive(False)
        freeze_and_unfreeze([], [self.encoder, self.encoding_bottleneck])

    def contrastive_detection_stage(self):
        self.heads.set_use_contrastive(True)
        freeze_and_unfreeze([self.encoder, self.encoding_bottleneck], [])

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoding_bottleneck(x)
        x = self.downstream_bottleneck(x)
        x = self.heads(x)
        return x

    def enc_forward(self, x):
        x = self.encoder(x)
        x = self.encoding_bottleneck(x)
        return x

    def pred_forward(self, x):
        x = self.downstream_bottleneck(x)
        x = self.heads(x)
        return x


class TechnicalBottleneck(nn.Module):
    def __init__(self, init_weights=INIT_WEIGHTS):
        super().__init__()

        down_conv1 = nn.Conv2d(640, 64, kernel_size=3, padding='same', bias=False)
        down_conv2 = nn.Conv2d(64, 2, kernel_size=3, padding='same', bias=False)
        up_conv1 = nn.Conv2d(2, 64, kernel_size=3, padding='same', bias=False)
        up_conv2 = nn.Conv2d(64, 640, kernel_size=3, padding='same', bias=False)

        flatten = nn.Flatten()
        unflatten = nn.Unflatten(dim=1, unflattened_size=(2, 32, 32))

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

        self.down_block = nn.Sequential(down_conv1, relu, down_norm1, down_conv2, relu, down_norm2, flatten, down_lin, relu, down_norm3)
        self.up_block = nn.Sequential(up_lin, relu, up_norm1, unflatten, up_conv1, relu, up_norm2, up_conv2, relu, up_norm3)

    def forward(self, x: torch.Tensor):
        shrunk = self.down_block(x)
        expanded = self.up_block(shrunk)

        return expanded, shrunk.detach()


# =============================================================================
# EXTENDED LONGITUDINAL MODEL (CURRENT TRAINING CONFIG)
# =============================================================================

class LongitudinalMIMModelBig(nn.Module):
    """
    Extended longitudinal model with dual bottleneck paths.
    
    This is the model currently used in training (see longitudinal_MIM_training.py).
    Extends LongitudinalMIMModel with:
    - Additional bottleneck processing paths
    - Optional technical bottleneck for compression
    - Dropout layers for regularization
    
    Architecture:
        1. Shared EfficientNet-B7 encoder
        2. Dual conv bottleneck paths (encoded_bn, encoded_bn2)
        3. Dual diff processing paths (diff_processing, diff_processing2)
        4. Combined feature processing before decoder
        5. Decoder6 with Tanh output
    
    Args:
        use_mask_token: Whether to use learnable mask token
        dec: Decoder version (5 or 6)
        patch_dec: Whether to use patch decoder
        patch_size: Patch size for masking
        use_pos_embed: Whether to use positional embeddings
        init_weights: Whether to initialize weights
        use_technical_bottleneck: Whether to add compression bottleneck
    
    Input:
        bl: Baseline CXR [B, 1, 512, 512]
        fu: Followup CXR [B, 1, 512, 512]
    
    Output:
        Change map [B, 1, 512, 512] with Tanh activation (range [-1, +1])
    """
    
    def __init__(self, use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=False, patch_size=MASK_PATCH_SIZE, use_pos_embed=USE_POS_EMBED, init_weights=INIT_WEIGHTS,
                 use_technical_bottleneck=False):
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

        # self.encoder_bl = EfficientNetMiniEncoder(dropout=False)
        # self.encoder_fu = EfficientNetMiniEncoder(dropout=False)
        self.encoder = EfficientNetMiniEncoder(dropout=False)

        # self.bl_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        # self.fu_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        self.dropout = nn.Dropout2d(p=0.1)

        self.encoded_bn = nn.Sequential(*[
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        ])

        self.encoded_bn2 = nn.Sequential(*[
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        ])

        # self.bl_emb = nn.Sequential(*[
        #     SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
        #     FeatureEmbeddings(1)
        # ])
        #
        # self.fu_emb = nn.Sequential(*[
        #     SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
        #     FeatureEmbeddings(1)
        # ])

        # ViT_model = ViTForMaskedImageModeling.from_pretrained(VIT_PRETRAINED_PATH_16_224)
        # self.transformer_encoder = ViT_model.vit.encoder

        # conf = ViTConfig()
        # transformer_enc = ViTEncoder(conf)
        # self.transformer_encoder = transformer_enc

        # trans_enc_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_CHANNELS, nhead=12, dropout=0.0, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(trans_enc_layer, num_layers=3)

        # self.enc = nn.ModuleList([self.encoder, self.encoder_bl])

        # self.bl_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])
        #
        # self.fu_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])

        self.features_upsampling = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        ])

        # self.channel_compression1 = ChannelExpansionLayer(2 * FEATURE_CHANNELS, 0.5, kernel=5)
        # self.channel_compression2 = ChannelExpansionLayer(3 * FEATURE_CHANNELS, 1 / 3, kernel=5)

        self.diff_processing = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        ])

        self.diff_processing2 = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        ])

        # self.features_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2)
        #     ConvBlockBranch()
        # ])

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
            raise NotImplementedError()

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

        # for name, param in self.transformer_encoder.named_parameters():
        #     if 'weight' in name and 'linear' in name:
        #         nn.init.xavier_uniform_(param, gain=sqrt(2.))
        #     elif 'in_proj_weight' in name or 'out_proj.weight' in name:
        #         nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

    def get_mask_token(self):
        return self.mask_token

    def forward(self, bl: torch.Tensor, fu: torch.tensor):
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        # encoded_bl = self.encoder_bl(bl_pos)
        # encoded_fu = self.encoder_fu(fu_pos)
        # encoded_bl = self.bl_bn(encoded_bl)
        # encoded_fu = self.fu_bn(encoded_fu)
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        encoded_bl2 = self.encoded_bn2(encoded_bl)
        encoded_fu2 = self.encoded_bn2(encoded_fu)
        # emb_bl = self.bl_emb(encoded_bl)
        # emb_fu = self.fu_emb(encoded_fu)
        # # embeddings = torch.cat([emb_bl, emb_fu], dim=-2)
        # embeddings = emb_fu - emb_bl
        # features = self.transformer_encoder(embeddings)
        #
        # features = features.permute(0, 2, 1)
        # features = features.contiguous().view(features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # features = self.features_upsampling(features)

        # bl_features = features[:, :features.shape[1] // 2, :]
        # bl_features = bl_features.permute(0, 2, 1)
        # bl_features = bl_features.contiguous().view(bl_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # bl_features = self.bl_upsampling(bl_features)
        #
        # fu_features = features[:, features.shape[1] // 2:, :]
        # fu_features = fu_features.permute(0, 2, 1)
        # fu_features = fu_features.contiguous().view(fu_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # fu_features = self.fu_upsampling(fu_features)

        encoded_diff = encoded_fu - encoded_bl

        # features = torch.cat([features, encoded_diff], dim=1)
        # features = self.channel_compression1(features)
        # features = features + encoded_diff
        features = encoded_diff
        # features = self.features_processing(features)
        # features = torch.cat([features, encoded_bl, encoded_fu], dim=1)

        # processed_encoded_diff = self.diff_processing(encoded_diff)
        features = self.diff_processing(features)
        features = self.dropout(features)

        encoded_diff2 = encoded_fu2 - encoded_bl2
        encoded_diff2 = self.dropout(encoded_diff2)

        features = features + encoded_diff2

        features = self.diff_processing2(features)

        if not self.use_technical_bottleneck and self.return_latent:
            latent = torch.flatten(features, 1, -1)
            return latent

        features = self.dropout(features)

        # features = self.channel_compression2(features)
        # features = features + processed_encoded_diff
        # features = self.features_bn(features)

        if self.use_technical_bottleneck:
            features, latent = self.technical_bottleneck(features)

            if self.return_latent:
                return latent

        decoded = self.decoder(features)

        # outputs = torch.clamp(fu + decoded, min=0., max=1.)

        return decoded

        # encoded_bl = self.bl_bn(self.encoder_bl(bl))
        # encoded_fu = self.fu_bn(self.encoder_fu(fu))
        #
        # features = torch.cat([encoded_bl, encoded_fu], dim=1)
        # features = self.channel_compression1(features)
        # features = self.features_bn(features)
        #
        # decoded = self.decoder(features)
        # return decoded


class LongitudinalMIMModelTest(nn.Module):
    def __init__(self, patch_size=MASK_PATCH_SIZE, use_pos_embed=USE_POS_EMBED, init_weights=INIT_WEIGHTS):
        super().__init__()

        self.mask_token = None

        # self.use_pos_emb = use_pos_embed
        # if self.use_pos_emb:
        #     self.pos_emb_bl = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
        #     self.pos_emb_fu = nn.Parameter(torch.zeros(1, IMG_SIZE, IMG_SIZE))
        #     self.patch_size = patch_size
        # else:
        #     self.pos_emb_bl = None
        #     self.pos_emb_fu = None
        #
        # self.encoder_bl = nn.Sequential(*[
        #     ChannelExpansionBlock(in_channels=1, expand_ratio_1=8., expand_ratio_2=8., first_kernel=5, second_kernel=5),
        #     ChannelExpansionBlock(in_channels=64, expand_ratio_1=2., expand_ratio_2=2., first_kernel=5, second_kernel=5),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256)
        # ])
        #
        # self.encoder_fu = nn.Sequential(*[
        #     ChannelExpansionBlock(in_channels=1, expand_ratio_1=8., expand_ratio_2=8., first_kernel=5, second_kernel=5),
        #     ChannelExpansionBlock(in_channels=64, expand_ratio_1=2., expand_ratio_2=2., first_kernel=5, second_kernel=5),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256),
        #     ConvBlockBranch(channels=256)
        # ])

        self.encoder_bl = nn.Sequential(*[
            ChannelExpansionBlock(in_channels=1, expand_ratio_1=4., expand_ratio_2=4., first_kernel=5, second_kernel=5),
            ChannelExpansionBlock(in_channels=16, expand_ratio_1=2., expand_ratio_2=2., first_kernel=3, second_kernel=3),
        ])

        self.encoder_fu = nn.Sequential(*[
            ChannelExpansionBlock(in_channels=1, expand_ratio_1=4., expand_ratio_2=4., first_kernel=5, second_kernel=5),
            ChannelExpansionBlock(in_channels=16, expand_ratio_1=2., expand_ratio_2=2., first_kernel=3, second_kernel=3),
        ])

        self.diff_enc = nn.Sequential(*[
            ConvBlockBranch(channels=64),
            ChannelExpansionLayer(64, 0.25)
        ])

        self.out_layer = nn.Conv2d(16, 1, kernel_size=1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_layer.weight, gain=5 / 3)
        # nn.init.normal_(self.out_layer.weight, mean=0.0, std=INIT_STD)

    def get_mask_token(self):
        return self.mask_token

    def forward(self, bl: torch.Tensor, fu: torch.tensor):
        bl_enc = self.encoder_bl(bl)
        fu_enc = self.encoder_fu(fu)
        feat = fu_enc - bl_enc
        feat = self.diff_enc(feat)
        feat = self.tanh(self.out_layer(feat))

        return feat


# =============================================================================
# MAIN LONGITUDINAL MODEL (WORKING VERSION)
# =============================================================================

class LongitudinalMIMModel(nn.Module):
    """
    Primary model for longitudinal CXR change detection.
    
    Architecture:
        1. Shared EfficientNet-B7 encoder for both baseline and followup
        2. Conv bottleneck blocks for feature processing
        3. Feature difference computation (FU - BL)
        4. Decoder to reconstruct change map at original resolution
    
    This is the model that worked for experiment id9.
    
    Args:
        use_mask_token: Whether to use learnable mask token (for MIM pretraining)
        dec: Decoder version to use (5 or 6, default 6)
        patch_dec: Whether to use patch-based decoder
        patch_size: Size of patches for masking
        use_pos_embed: Whether to add learnable positional embeddings
        init_weights: Whether to initialize weights with custom scheme
    
    Input:
        bl: Baseline CXR [B, 1, 512, 512]
        fu: Followup CXR [B, 1, 512, 512]
    
    Output:
        Change map [B, 1, 512, 512] with Tanh activation (range [-1, +1])
        Positive = new findings, Negative = resolved findings
    """
    
    def __init__(self, use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=False, patch_size=MASK_PATCH_SIZE, use_pos_embed=USE_POS_EMBED, init_weights=INIT_WEIGHTS):
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

        self.dropout = nn.Dropout2d(p=0.1)

        # self.encoder_bl = EfficientNetMiniEncoder(dropout=False)
        # self.encoder_fu = EfficientNetMiniEncoder(dropout=False)
        self.encoder = EfficientNetMiniEncoder(dropout=False)

        # self.bl_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        # self.fu_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        self.encoded_bn = nn.Sequential(*[
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        ])

        # self.bl_emb = nn.Sequential(*[
        #     SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
        #     FeatureEmbeddings(1)
        # ])
        #
        # self.fu_emb = nn.Sequential(*[
        #     SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
        #     FeatureEmbeddings(1)
        # ])

        # ViT_model = ViTForMaskedImageModeling.from_pretrained(VIT_PRETRAINED_PATH_16_224)
        # self.transformer_encoder = ViT_model.vit.encoder

        # conf = ViTConfig()
        # transformer_enc = ViTEncoder(conf)
        # self.transformer_encoder = transformer_enc

        # trans_enc_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_CHANNELS, nhead=12, dropout=0.0, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(trans_enc_layer, num_layers=3)

        # self.enc = nn.ModuleList([self.encoder, self.encoder_bl])

        # self.bl_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])
        #
        # self.fu_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])

        # self.features_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])
        #
        # self.channel_compression1 = ChannelExpansionLayer(2 * FEATURE_CHANNELS, 0.5, kernel=5)
        # # self.channel_compression2 = ChannelExpansionLayer(3 * FEATURE_CHANNELS, 1 / 3, kernel=5)
        #
        # self.features_processing = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch()
        # ])

        self.diff_processing = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        ])

        # self.features_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2)
        #     ConvBlockBranch()
        # ])

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
            raise NotImplementedError()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        if self.use_pos_emb:
            nn.init.trunc_normal_(self.pos_emb_bl.data, mean=0.0, std=INIT_STD)
            nn.init.trunc_normal_(self.pos_emb_fu.data, mean=0.0, std=INIT_STD)

        # for name, param in self.transformer_encoder.named_parameters():
        #     if 'weight' in name and 'linear' in name:
        #         nn.init.xavier_uniform_(param, gain=sqrt(2.))
        #     elif 'in_proj_weight' in name or 'out_proj.weight' in name:
        #         nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

    def get_mask_token(self):
        return self.mask_token

    def forward(self, bl: torch.Tensor, fu: torch.tensor):
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        # encoded_bl = self.encoder_bl(bl_pos)
        # encoded_fu = self.encoder_fu(fu_pos)
        # encoded_bl = self.bl_bn(encoded_bl)
        # encoded_fu = self.fu_bn(encoded_fu)
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        # emb_bl = self.bl_emb(encoded_bl)
        # emb_fu = self.fu_emb(encoded_fu)
        # # embeddings = torch.cat([emb_bl, emb_fu], dim=-2)
        # embeddings = emb_fu - emb_bl
        # features = self.transformer_encoder(embeddings)
        #
        # features = features.permute(0, 2, 1)
        # features = features.contiguous().view(features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # features = self.features_upsampling(features)

        # bl_features = features[:, :features.shape[1] // 2, :]
        # bl_features = bl_features.permute(0, 2, 1)
        # bl_features = bl_features.contiguous().view(bl_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # bl_features = self.bl_upsampling(bl_features)
        #
        # fu_features = features[:, features.shape[1] // 2:, :]
        # fu_features = fu_features.permute(0, 2, 1)
        # fu_features = fu_features.contiguous().view(fu_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # fu_features = self.fu_upsampling(fu_features)

        encoded_diff = encoded_fu - encoded_bl

        # features = torch.cat([features, encoded_diff], dim=1)
        # features = self.channel_compression1(features)
        # features = features + encoded_diff
        features = encoded_diff
        # features = self.features_processing(features)
        # features = torch.cat([features, encoded_bl, encoded_fu], dim=1)

        # processed_encoded_diff = self.diff_processing(encoded_diff)
        features = self.diff_processing(features)
        features = self.dropout(features)

        # features = self.channel_compression2(features)
        # features = features + processed_encoded_diff
        # features = self.features_bn(features)

        decoded = self.decoder(features)

        # outputs = torch.clamp(fu + decoded, min=0., max=1.)

        return decoded

        # encoded_bl = self.bl_bn(self.encoder_bl(bl))
        # encoded_fu = self.fu_bn(self.encoder_fu(fu))
        #
        # features = torch.cat([encoded_bl, encoded_fu], dim=1)
        # features = self.channel_compression1(features)
        # features = self.features_bn(features)
        #
        # decoded = self.decoder(features)
        # return decoded


#################


class LongitudinalMIMModelBigTransformer(nn.Module):
    def __init__(self, use_mask_token=USE_MASK_TOKEN, dec=6, patch_dec=False, patch_size=MASK_PATCH_SIZE, use_pos_embed=USE_POS_EMBED, init_weights=INIT_WEIGHTS):
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

        # self.encoder_bl = EfficientNetMiniEncoder(dropout=False)
        # self.encoder_fu = EfficientNetMiniEncoder(dropout=False)
        self.encoder = EfficientNetMiniEncoder(dropout=False)

        # self.bl_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        # self.fu_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        # ])

        self.dropout = nn.Dropout2d(p=0.1)

        self.emb1 = nn.Sequential(*[
            SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
            FeatureEmbeddings(1)
        ])

        self.emb2 = nn.Sequential(*[
            SamplingConvBlock(2, FEATURE_CHANNELS, INTER_CHANNELS, sampling_type='down'),
            FeatureEmbeddings(1)
        ])

        # ViT_model = ViTForMaskedImageModeling.from_pretrained(VIT_PRETRAINED_PATH_16_224)
        # self.transformer_encoder = ViT_model.vit.encoder

        # conf = ViTConfig()
        # transformer_enc = ViTEncoder(conf)
        # self.transformer_encoder = transformer_enc

        trans_enc_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_CHANNELS, nhead=8, dropout=0.0, batch_first=True)

        self.encoded_bn = nn.Sequential(*[
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch(),
        ])

        # self.enc = nn.ModuleList([self.encoder, self.encoder_bl])

        # self.bl_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])
        #
        # self.fu_upsampling = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
        #     SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        # ])

        self.features_upsampling1 = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        ])

        self.features_upsampling2 = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            SamplingConvBlock(1, HIDDEN_CHANNELS, INTER_CHANNELS, sampling_type='up'),
            SamplingConvBlock(2, INTER_CHANNELS, FEATURE_CHANNELS, sampling_type='up'),
        ])

        self.encoded_bn2 = nn.Sequential(*[
            self.emb1,
            nn.TransformerEncoder(trans_enc_layer, num_layers=2),
        ])

        # self.channel_compression1 = ChannelExpansionLayer(2 * FEATURE_CHANNELS, 0.5, kernel=5)
        # self.channel_compression2 = ChannelExpansionLayer(3 * FEATURE_CHANNELS, 1 / 3, kernel=5)

        self.diff_processing1 = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        ])

        self.diff_processing2 = nn.Sequential(*[
            self.emb2,
            nn.TransformerEncoder(trans_enc_layer, num_layers=2),
        ])

        self.diff_processing3 = nn.Sequential(*[
            # nn.Dropout2d(0.2),
            ConvBlockBranch(),
            ConvBlockBranch(),
            ConvBlockBranch()
        ])

        # self.features_bn = nn.Sequential(*[
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2),
        #     ConvBlockBranch(),
        #     # nn.Dropout2d(0.2)
        #     ConvBlockBranch()
        # ])

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
            raise NotImplementedError()

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        if self.use_pos_emb:
            nn.init.trunc_normal_(self.pos_emb_bl.data, mean=0.0, std=INIT_STD)
            nn.init.trunc_normal_(self.pos_emb_fu.data, mean=0.0, std=INIT_STD)

        for name, param in chain(self.encoded_bn2[1].named_parameters(), self.diff_processing2[1].named_parameters()):
            if 'weight' in name and 'linear' in name:
                nn.init.xavier_uniform_(param, gain=sqrt(2.))
            elif 'in_proj_weight' in name or 'out_proj.weight' in name:
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

    def get_mask_token(self):
        return self.mask_token

    def forward(self, bl: torch.Tensor, fu: torch.tensor):
        if self.pos_emb_bl is not None:
            bl_pos = bl + self.pos_emb_bl
            fu_pos = fu + self.pos_emb_fu
        else:
            bl_pos = bl
            fu_pos = fu
        # encoded_bl = self.encoder_bl(bl_pos)
        # encoded_fu = self.encoder_fu(fu_pos)
        # encoded_bl = self.bl_bn(encoded_bl)
        # encoded_fu = self.fu_bn(encoded_fu)
        encoded_bl = self.encoder(bl_pos)
        encoded_fu = self.encoder(fu_pos)
        encoded_bl = self.encoded_bn(encoded_bl)
        encoded_fu = self.encoded_bn(encoded_fu)
        encoded_bl2 = self.encoded_bn2(encoded_bl)
        encoded_fu2 = self.encoded_bn2(encoded_fu)
        # emb_bl = self.bl_emb(encoded_bl)
        # emb_fu = self.fu_emb(encoded_fu)
        # # embeddings = torch.cat([emb_bl, emb_fu], dim=-2)
        # embeddings = emb_fu - emb_bl
        # features = self.transformer_encoder(embeddings)
        #
        encoded_bl2 = encoded_bl2.permute(0, 2, 1)
        encoded_bl2 = encoded_bl2.contiguous().view(encoded_bl2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_bl2 = self.features_upsampling1(encoded_bl2)

        encoded_fu2 = encoded_fu2.permute(0, 2, 1)
        encoded_fu2 = encoded_fu2.contiguous().view(encoded_fu2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_fu2 = self.features_upsampling1(encoded_fu2)

        # bl_features = features[:, :features.shape[1] // 2, :]
        # bl_features = bl_features.permute(0, 2, 1)
        # bl_features = bl_features.contiguous().view(bl_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # bl_features = self.bl_upsampling(bl_features)
        #
        # fu_features = features[:, features.shape[1] // 2:, :]
        # fu_features = fu_features.permute(0, 2, 1)
        # fu_features = fu_features.contiguous().view(fu_features.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        # fu_features = self.fu_upsampling(fu_features)

        encoded_diff = encoded_fu - encoded_bl

        # features = torch.cat([features, encoded_diff], dim=1)
        # features = self.channel_compression1(features)
        # features = features + encoded_diff
        features = encoded_diff
        # features = self.features_processing(features)
        # features = torch.cat([features, encoded_bl, encoded_fu], dim=1)

        # processed_encoded_diff = self.diff_processing(encoded_diff)
        features = self.diff_processing1(features)
        features = self.dropout(features)

        encoded_diff2 = encoded_fu2 - encoded_bl2
        encoded_diff2 = self.diff_processing2(encoded_diff2)
        encoded_diff2 = encoded_diff2.permute(0, 2, 1)
        encoded_diff2 = encoded_diff2.contiguous().view(encoded_diff2.shape[0], HIDDEN_CHANNELS, PATCHES_IN_SPATIAL_DIM, PATCHES_IN_SPATIAL_DIM)
        encoded_diff2 = self.features_upsampling1(encoded_diff2)
        encoded_diff2 = self.dropout(encoded_diff2)

        features = features + encoded_diff2

        features = self.diff_processing3(features)
        features = self.dropout(features)

        # features = self.channel_compression2(features)
        # features = features + processed_encoded_diff
        # features = self.features_bn(features)

        decoded = self.decoder(features)

        # outputs = torch.clamp(fu + decoded, min=0., max=1.)

        return decoded

        # encoded_bl = self.bl_bn(self.encoder_bl(bl))
        # encoded_fu = self.fu_bn(self.encoder_fu(fu))
        #
        # features = torch.cat([encoded_bl, encoded_fu], dim=1)
        # features = self.channel_compression1(features)
        # features = self.features_bn(features)
        #
        # decoded = self.decoder(features)
        # return decoded
