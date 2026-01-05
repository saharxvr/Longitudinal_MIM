"""
Training Configuration
======================

Hyperparameters for training the Longitudinal CXR Analysis models.

Training Phases:
----------------
1. Masked Image Modeling (MIM) Pretraining:
   - Self-supervised learning on unlabeled CXR
   - Mask patches and reconstruct
   
2. Longitudinal MIM Training:
   - Learn to predict difference maps between image pairs
   - Uses synthetic DRR pairs with known changes

3. Detection/Classification Fine-tuning:
   - Supervised training on labeled datasets
   - Contrastive learning for better representations

Sections:
---------
1. Batch and Optimization
2. Learning Rate Schedule
3. Loss Function Flags
4. Loss Weights (Lambdas)
5. Masking Configuration
6. Epoch Milestones
7. Contrastive Learning

Usage:
------
    from config.training_config import BATCH_SIZE, MAX_LR, USE_L1, LAMBDA_L1_ALL
"""

# =============================================================================
# BATCH AND OPTIMIZATION
# =============================================================================

BATCH_SIZE: int = 4
"""
Number of samples per training batch for MIM pretraining.
Small due to large image size (512x512) and model size.
Effective batch = BATCH_SIZE * UPDATE_EVERY_BATCHES = 48
Used by: longitudinal_MIM_training.py, datasets.py DataLoader
"""

UPDATE_EVERY_BATCHES: int = 12
"""
Gradient accumulation steps before optimizer update.
Simulates larger batch size without memory increase.
Effective batch size = BATCH_SIZE * UPDATE_EVERY_BATCHES
Used by: longitudinal_MIM_training.py training loop
"""

DETECTION_BATCH_SIZE: int = 8
"""
Batch size for detection/classification training.
Can be larger than MIM since no decoder needed.
Used by: detection training scripts
"""

DETECTION_UPDATE_EVERY_BATCHES: int = 5
"""
Gradient accumulation for detection training.
Effective batch = 8 * 5 = 40
"""

CONTRASTIVE_BATCH_SIZE: int = 128
"""
Batch size for contrastive learning.
Larger batches improve contrastive learning performance.
Requires more negatives for effective learning.
Used by: contrastive learning training
"""

# =============================================================================
# LEARNING RATE CONFIGURATION
# =============================================================================

MAX_LR: float = 6e-4
"""
Maximum learning rate for cosine annealing schedule.
Reached after warmup period.
Good default for AdamW with transformers.
Used by: longitudinal_MIM_training.py - OneCycleLR scheduler
"""

WEIGHT_DECAY: float = 1e-2
"""
L2 regularization coefficient for AdamW optimizer.
Helps prevent overfitting, especially with large models.
0.01 is standard for transformers.
Used by: all training scripts - optimizer configuration
"""

# =============================================================================
# LOSS FUNCTION FLAGS
# =============================================================================
# Enable/disable different loss components.
# Multiple losses can be combined for training.

USE_L1: bool = True
"""
Enable L1 (Mean Absolute Error) reconstruction loss.
Robust to outliers, produces sharper reconstructions.
Primary loss for image reconstruction tasks.
Used by: longitudinal_MIM_training.py, utils.py
"""

USE_L2: bool = True
"""
Enable L2 (Mean Squared Error) reconstruction loss.
Penalizes large errors more than L1.
Often combined with L1 for balanced training.
Used by: longitudinal_MIM_training.py
"""

USE_FOURIER: bool = False
"""
Enable Fourier domain loss.
Compares frequency components of predicted vs target.
Helps preserve high-frequency details (edges, textures).
Used by: utils.py - masked_patches_fourier_loss
"""

USE_PERC_STYLE: bool = False
"""
Enable perceptual and style losses (VGG-based).
Uses pretrained VGG features to compare semantic content.
Perceptual: feature similarity, Style: Gram matrix similarity.
Used by: longitudinal_MIM_training.py, extra/vgg_losses.py
"""

USE_GAN: bool = False
"""
Enable adversarial (GAN) loss.
Trains discriminator to distinguish real vs generated.
Can improve visual quality but harder to train.
Used by: utils.py - masked_patches_GAN_loss
"""

USE_SSIM: bool = False
"""
Enable Structural Similarity Index (SSIM) loss.
Compares luminance, contrast, and structure.
Good for preserving perceptual quality.
Used by: utils.py - masked_patches_SSIM_loss
"""

# =============================================================================
# LOSS WEIGHTS (LAMBDAS)
# =============================================================================
# Coefficients for combining multiple loss terms.
# Total loss = sum(lambda_i * loss_i)

LAMBDA_L1_ALL: float = 1.0
"""
Weight for L1 loss on entire image reconstruction.
Applied to full output vs target comparison.
Used by: longitudinal_MIM_training.py
"""

LAMBDA_L1_MASKED: float = 1.0
"""
Weight for L1 loss specifically on masked regions.
Focuses learning on reconstruction ability.
Used by: utils.py - masked_patches_l1_loss
"""

LAMBDA_L2: float = 1.0
"""
Weight for L2 (MSE) reconstruction loss.
Usually equal to L1 weight for balanced training.
Used by: longitudinal_MIM_training.py
"""

LAMBDA_FOURIER: float = 0.05
"""
Weight for Fourier domain loss on full image.
Small weight since Fourier values can be large.
Used by: longitudinal_MIM_training.py
"""

LAMBDA_FOURIER_MASKED: float = 0.3
"""
Weight for Fourier loss on masked regions only.
Higher than full-image to focus on masked reconstruction.
Used by: utils.py - masked_patches_fourier_loss
"""

LAMBDA_GAN: float = 0.5
"""
Weight for adversarial generator loss.
Balanced to prevent mode collapse.
Used by: longitudinal_MIM_training.py (when USE_GAN=True)
"""

LAMBDA_P: float = 2.0
"""
Weight for perceptual loss (VGG feature matching).
Higher weight for semantic preservation.
Used by: longitudinal_MIM_training.py (when USE_PERC_STYLE=True)
"""

LAMBDA_S: float = 2.0
"""
Weight for style loss (VGG Gram matrix matching).
Preserves texture statistics.
Used by: longitudinal_MIM_training.py (when USE_PERC_STYLE=True)
"""

LAMBDA_SSIM: float = 1.5
"""
Weight for SSIM loss.
Slightly higher to emphasize structural similarity.
Used by: longitudinal_MIM_training.py (when USE_SSIM=True)
"""

# =============================================================================
# MASKING CONFIGURATION
# =============================================================================
# Settings for masked image modeling (MIM)

MASK_PATCH_SIZE: int = 128
"""
Size of patches to mask during MIM training.
Larger patches = harder reconstruction task.
128 = 1/4 of image size, masks significant regions.
Used by: augmentations.py, utils.py loss functions
"""

MASK_MODE: str = 'single'
"""
Masking strategy for MIM.
'single': One contiguous masked region
'random': Multiple random patches
'grid': Regular grid pattern
Used by: augmentations.py - mask generation
"""

INIT_MASK_PROB: float = 0.05
"""
Initial masking probability at training start.
Low initial value for curriculum learning.
Gradually increases during training.
Used by: utils.py - MaskProbScheduler
"""

MAX_MASK_PROB: float = 0.7
"""
Maximum masking probability during training.
Reached after warmup period, held during main training.
70% is aggressive but effective for CXR.
Used by: utils.py - MaskProbScheduler
"""

END_MASK_PROB: float = 0.7
"""
Final masking probability at training end.
Can decrease for fine-tuning phase.
Currently same as MAX for continued challenge.
Used by: utils.py - MaskProbScheduler
"""

SIGMAS: list = []
"""
Gaussian blur sigma values for Canny edge augmentation.
Empty list = no Canny edges used.
Non-empty = adds edge channels to input.
Used by: augmentations.py, model input channels calculation
"""

MASKED_IN_CHANNELS: int = len(SIGMAS) + 1 if isinstance(SIGMAS, list) else 1
"""
Number of input channels for masked reconstruction.
1 (grayscale) + number of edge channels if using Canny.
Derived from SIGMAS configuration.
"""

USE_CANNY: bool = isinstance(SIGMAS, list) and len(SIGMAS) > 0
"""
Whether Canny edge detection is used.
True if SIGMAS list is non-empty.
Adds edge information to guide reconstruction.
"""

# =============================================================================
# TRAINING PHASE EPOCHS
# =============================================================================

MASKED_RECONSTRUCTION_EPOCHS: int = 30
"""
Number of epochs for masked reconstruction pretraining.
30 epochs typically sufficient for convergence.
Used by: MIM training scripts
"""

LONGITUDINAL_MIM_EPOCHS: int = 10
"""
Number of epochs for longitudinal MIM training.
Shorter since built on pretrained encoder.
Used by: longitudinal_MIM_training.py
"""

CONTRASTIVE_TRAINING_EPOCHS: int = 40
"""
Number of epochs for contrastive learning phase.
Longer training helps learn discriminative features.
Used by: contrastive learning scripts
"""

GAN_START_EPOCH: int = 13
"""
Epoch to start GAN training (if USE_GAN=True).
Delayed start for stable initial training.
Generator trained alone first, then with discriminator.
"""

DETECTION_START_EPOCH: int = 1
"""
Epoch to start detection head training.
Usually from beginning (epoch 1).
Used by: multi-task training scripts
"""

CONTRASTIVE_START_EPOCH: int = 38
"""
Epoch to start contrastive loss.
After initial supervised training.
Helps with representation learning.
"""

NORM_START_EPOCH: int = 55
"""
Epoch to start normalization loss.
Late addition for stable training.
Regularizes feature magnitudes.
"""

# =============================================================================
# CONTRASTIVE LEARNING SETTINGS
# =============================================================================

TEMPERATURE: float = 0.1
"""
Temperature parameter for contrastive loss (InfoNCE).
Lower temperature = sharper probability distribution.
0.1 is common for SimCLR-style losses.
Used by: contrastive loss computation
"""

USE_NEWER_CONTRASTIVE_LOSS: bool = True
"""
Use updated contrastive loss implementation.
True: Improved multi-label handling
False: Original implementation
Used by: contrastive training
"""

OLD_LOSS_C: float = 0.5
"""
Coefficient for old-style contrastive loss.
Only used when USE_NEWER_CONTRASTIVE_LOSS=False.
Blending factor between loss variants.
"""

USE_DEDISORDER: bool = False
"""
Enable DeDiSorder regularization.
Encourages disentangled representations.
Experimental feature.
"""

CONT_LAMBDAS: dict = {'bce': 1., 'cont': 0.01, 'norm': 0.1}
"""
Loss weights for multi-task contrastive training.
- bce: Binary cross-entropy for classification
- cont: Contrastive loss weight
- norm: Feature normalization regularization
Used by: contrastive training scripts
"""

DETECTION_IN_CHANNELS: int = 1
"""
Number of input channels for detection model.
1 for grayscale CXR images.
Used by: detection model initialization
"""
