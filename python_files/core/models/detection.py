"""
Detection and Downstream Classification Models.

Contains models for downstream tasks like disease detection and classification:
- Discriminator: Adversarial discriminator for GAN-based training
- DownstreamHeads: Classification heads for multi-label disease prediction
- DetectionContrastiveModel: Full model combining encoder + heads for detection

These models leverage pretrained encoders from self-supervised learning
for fine-tuning on labeled classification datasets.
"""

import torch
import torch.nn as nn

from config import (
    FEATURE_CHANNELS,
    FEATURE_SIZE,
    HIDDEN_CHANNELS,
    LATENT_FEATURES,
    NON_GLOBAL_POOLING_FEATURES,
    CUR_LABELS_NUM,
    GROUP_NORM_GROUPS,
    INIT_WEIGHTS,
    INIT_STD,
)

from core.models.blocks import ConvBlockBranch
from core.models.encoders import EfficientNetMiniEncoder
from core.models.bottleneck import EncodingBottleneck, DownstreamBottleneck
from core.models.utils import freeze_and_unfreeze


class Discriminator(nn.Module):
    """
    Adversarial discriminator for GAN-based training.
    
    Used in adversarial training to distinguish real from generated images.
    Shares the EfficientNet encoder architecture with the generator.
    
    Architecture:
        EfficientNetEncoder → ConvBlock → Conv(1x1) → Flatten → Linear(128) → Linear(1)
    
    Output: Single logit for real/fake classification
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        from config import EFF_NET_BLOCK_IDXS
        
        self.encoder = EfficientNetMiniEncoder(block_idxs=EFF_NET_BLOCK_IDXS[:-1])
        
        self.bottleneck = nn.Sequential(
            ConvBlockBranch(),
            nn.Conv2d(FEATURE_CHANNELS, 2, kernel_size=1),
            nn.GroupNorm(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * FEATURE_SIZE * FEATURE_SIZE, 128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input as real or fake.
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            Logit tensor (B, 1)
        """
        return self.bottleneck(self.encoder(x))


class DownstreamHeads(nn.Module):
    """
    Classification heads for multi-label disease detection.
    
    Processes encoded features through reduction and linear layers
    to produce disease predictions. Supports contrastive learning mode.
    
    Args:
        init_weights: Whether to apply custom weight initialization
    
    Architecture:
        Dropout2d → Conv(reduce) → Flatten → Linear(1024) → Linear(256) → [Predictions, Projection]
    
    Outputs:
        - predictions: (B, num_labels) disease logits
        - projection: (B, latent_features) for contrastive loss (if enabled)
    
    Training Modes:
        - Detection: use_contrastive=False, returns only predictions
        - Contrastive: use_contrastive=True, returns predictions + projections
    """
    
    def __init__(self, init_weights: bool = INIT_WEIGHTS):
        super().__init__()
        
        self.dropout1 = nn.Dropout2d(p=0.3)
        self.dropout2 = nn.Dropout(p=0.2)
        
        # Spatial reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS // 2, kernel_size=3, padding='same', bias=False),
            nn.GroupNorm(GROUP_NORM_GROUPS, FEATURE_CHANNELS // 2),
            nn.ReLU(),
            nn.Conv2d(FEATURE_CHANNELS // 2, 2, kernel_size=1),
            nn.GroupNorm(2, 2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Projection for contrastive learning
        self.early_projection = nn.Linear(2048, LATENT_FEATURES)
        
        # Classification pathway
        self.lin_block = nn.Sequential(
            nn.Linear(NON_GLOBAL_POOLING_FEATURES, 1024),
            nn.LayerNorm(normalized_shape=1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU()
        )
        
        self.late_projection = nn.Linear(256, LATENT_FEATURES)
        
        self.pred_head = nn.Sequential(
            nn.Linear(256, CUR_LABELS_NUM)
        )
        
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
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Generate predictions and optional projections.
        
        Args:
            x: Encoded features (B, C, H, W)
        
        Returns:
            Tuple of (predictions, projections)
            - predictions: (B, num_labels)
            - projections: (B, latent_features) or None if not contrastive
        """
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
    
    def set_use_contrastive(self, val: bool):
        """Enable or disable contrastive learning mode."""
        self.use_contrastive = val


class DetectionContrastiveModel(nn.Module):
    """
    Full detection model with contrastive learning support.
    
    Combines encoder, encoding bottleneck, downstream bottleneck, and
    classification heads for disease detection with optional contrastive
    representation learning.
    
    Args:
        in_channels: Number of input image channels (1, 2, or 4)
    
    Architecture:
        Encoder → EncodingBottleneck → DownstreamBottleneck → Heads
    
    Training Stages:
        1. detection_stage(): Train heads only, freeze encoder
        2. contrastive_detection_stage(): Enable contrastive, unfreeze encoder
    
    Example:
        >>> model = DetectionContrastiveModel(in_channels=1)
        >>> model.detection_stage()  # For supervised fine-tuning
        >>> preds, _ = model(x)
        >>> 
        >>> model.contrastive_detection_stage()  # For contrastive+detection
        >>> preds, projections = model(x)
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Select encoder based on input channels
        if in_channels == 1:
            self.encoder = EfficientNetMiniEncoder(dropout=True)
        elif in_channels == 2:
            self.encoder = EfficientNetMiniEncoder(inp=2, exp=4., dropout=True)
        elif in_channels == 4:
            self.encoder = EfficientNetMiniEncoder(inp=4, exp=4., exp2=4., dropout=True)
        else:
            raise ValueError(f"Invalid input channels: {in_channels}. Must be 1, 2, or 4.")
        
        self.encoding_bottleneck = EncodingBottleneck()
        self.e_b = nn.ModuleList([self.encoder, self.encoding_bottleneck])
        self.downstream_bottleneck = DownstreamBottleneck()
        self.heads = DownstreamHeads()
    
    def detection_stage(self):
        """
        Configure for detection-only training.
        
        Freezes encoder and encoding bottleneck, enables only classification.
        Use for supervised fine-tuning with frozen backbone.
        """
        self.heads.set_use_contrastive(False)
        freeze_and_unfreeze(
            to_freeze=[],
            to_unfreeze=[self.encoder, self.encoding_bottleneck]
        )
    
    def contrastive_detection_stage(self):
        """
        Configure for contrastive + detection training.
        
        Freezes encoder (pretrained), enables contrastive projections.
        Use for semi-supervised learning with contrastive objective.
        """
        self.heads.set_use_contrastive(True)
        freeze_and_unfreeze(
            to_freeze=[self.encoder, self.encoding_bottleneck],
            to_unfreeze=[]
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass through all components.
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            Tuple of (predictions, projections)
        """
        x = self.encoder(x)
        x = self.encoding_bottleneck(x)
        x = self.downstream_bottleneck(x)
        x = self.heads(x)
        return x
    
    def enc_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only forward pass.
        
        Useful for extracting features without classification.
        
        Args:
            x: Input image (B, C, H, W)
        
        Returns:
            Encoded features (B, C, H', W')
        """
        x = self.encoder(x)
        x = self.encoding_bottleneck(x)
        return x
    
    def pred_forward(self, x: torch.Tensor) -> tuple:
        """
        Prediction-only forward pass.
        
        Useful when encoder features are pre-computed.
        
        Args:
            x: Encoded features (B, C, H, W)
        
        Returns:
            Tuple of (predictions, projections)
        """
        x = self.downstream_bottleneck(x)
        x = self.heads(x)
        return x
