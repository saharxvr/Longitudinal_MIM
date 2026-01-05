"""
Path Configuration
==================

All file system paths used throughout the project.
Supports environment variable overrides for portability.

Environment Variables:
----------------------
THESIS_PROJECT_ROOT : str
    Override the base project directory.
    Default: '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/'

THESIS_DATA_ROOT : str  
    Override the data storage root (if different from project root).
    Default: Same as PROJECT_FOLDER

Sections:
---------
1. Base Directories: Project root and main data folders
2. Dataset Paths: MIMIC-CXR, CXR-14, PadChest, VinDr, etc.
3. Model Checkpoints: Saved model paths
4. Pretrained Models: HuggingFace model identifiers

Usage:
------
    from config.paths import MIMIC_FOLDER, LOAD_PATH
    
    # Or with environment override:
    # export THESIS_PROJECT_ROOT=/my/custom/path
"""

import os

# =============================================================================
# BASE DIRECTORIES
# =============================================================================

PROJECT_FOLDER: str = os.environ.get(
    'THESIS_PROJECT_ROOT',
    '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/'
)
"""
Root directory for the entire project.
Contains: saved_models/, data folders, output directories.
Override with THESIS_PROJECT_ROOT environment variable.
"""

DATA_ROOT: str = os.environ.get('THESIS_DATA_ROOT', PROJECT_FOLDER)
"""
Root directory for all datasets.
Usually same as PROJECT_FOLDER but can be separate for large storage.
Override with THESIS_DATA_ROOT environment variable.
"""

# =============================================================================
# MIMIC-CXR DATASET PATHS
# =============================================================================

MIMIC_FOLDER: str = os.path.join(DATA_ROOT, 'physionet.org')
"""
MIMIC-CXR dataset root directory.
Contains: files/, other_files/, supine_pairs/
Source: https://physionet.org/content/mimic-cxr/
"""

MIMIC_OTHER_AP: str = os.path.join(MIMIC_FOLDER, 'other_files/')
"""
MIMIC-CXR processed AP (anterior-posterior) view images.
Contains NIfTI files organized by patient/study.
"""

MIMIC_SUPINE: str = os.path.join(MIMIC_FOLDER, 'supine_pairs/')
"""
MIMIC-CXR supine image pairs for longitudinal analysis.
Contains paired baseline/follow-up studies.
"""

# =============================================================================
# CHEST X-RAY 14 (NIH) DATASET PATHS
# =============================================================================

CXR14_FOLDER: str = os.path.join(DATA_ROOT, 'ChestX-ray14')
"""
NIH Chest X-ray 14 dataset root directory.
Contains: images/, Data_Entry_2017.csv
Source: https://nihcc.app.box.com/v/ChestXray-NIHCC
Used for: Contrastive learning, classification pretraining
"""

CXR14_CL_TRAIN: str = os.path.join(CXR14_FOLDER, 'contrastive_learning/train')
"""
CXR-14 contrastive learning training split.
Contains NIfTI converted images for self-supervised training.
"""

CXR14_CL_VAL: str = os.path.join(CXR14_FOLDER, 'contrastive_learning/val')
"""
CXR-14 contrastive learning validation split.
"""

CXR14_CL_TEST: str = os.path.join(CXR14_FOLDER, 'contrastive_learning/test')
"""
CXR-14 contrastive learning test split.
"""

# =============================================================================
# OTHER DATASET PATHS
# =============================================================================

PNEUMONIA_FOLDER: str = os.path.join(DATA_ROOT, 'pneumonia_normal')
"""
Pneumonia/Normal binary classification dataset.
Contains: pneumonia/, normal/ subdirectories
Used for: Binary classification baseline
"""

PNEUMONIA_DS_NORMAL: str = os.path.join(PNEUMONIA_FOLDER, 'normal_nibs')
"""
Normal (healthy) chest X-rays in NIfTI format.
"""

PADCHEST_FOLDER: str = os.path.join(DATA_ROOT, 'PadChest')
"""
PadChest dataset root directory.
Contains: images/, PADCHEST_chest_x_ray_images_labels.csv
Source: https://bimcv.cipf.es/bimcv-projects/padchest/
Used for: Multi-label classification, external validation
"""

VINDR_FOLDER: str = os.path.join(DATA_ROOT, 'VinDrCXR')
"""
VinDr-CXR dataset root directory.
Contains: train/, test/, annotations/
Source: https://vindr.ai/datasets/cxr
Used for: External validation, localization tasks
"""

# =============================================================================
# MODEL CHECKPOINT PATHS
# =============================================================================

LOAD_PATH: str = os.path.join(
    PROJECT_FOLDER,
    'saved_models/MIM/Checkpoint_id0_MaskGrad_Sig5_Expert_Perc_1Channel_single128_Sched_NoFinding_Decoder5_Augs_Eff_ViT_Epoch30_MaskToken_MS-SSIM_L1_PosEmb_GN.pt'
)
"""
Default checkpoint path for Masked Image Modeling (MIM) pretrained model.
Architecture: EfficientNet-B7 encoder + ViT bottleneck + CNN decoder
Training: 30 epochs with MS-SSIM and L1 losses
Used by: models.py for initializing encoder weights
"""

DETECTION_LOAD_PATH: str = os.path.join(
    PROJECT_FOLDER,
    'saved_models/Base/Detection_Dropout_InitializedEncoderBottleneck_1Channels_PretrainedEffViT_16_224_GN.pt'
)
"""
Checkpoint for detection/classification downstream model.
Built on MIM pretrained encoder with classification head.
Used by: Detection tasks, contrastive learning fine-tuning
"""

LONGITUDINAL_LOAD_PATH: str = os.path.join(
    PROJECT_FOLDER,
    'saved_models/Base/Longitudinal_InitializedEnc_PretrainedEff_16_224_GN.pt'
)
"""
Checkpoint for longitudinal change detection model.
Dual-encoder architecture for comparing baseline/follow-up pairs.
Used by: Evaluation/Prediction.py, DRRs_test_*.py
"""

# =============================================================================
# HUGGINGFACE PRETRAINED MODEL IDENTIFIERS
# =============================================================================

EFFICIENTNET_B7_PRETRAINED_PATH: str = "google/efficientnet-b7"
"""
HuggingFace identifier for EfficientNet-B7 pretrained on ImageNet.
Used as: CNN feature extraction backbone in encoder
Input size: 600x600 (we resize to 512x512)
Output features: Multi-scale feature maps at indices defined by EFF_NET_BLOCK_IDXS
"""

VIT_PRETRAINED_PATH_16_224: str = 'google/vit-base-patch16-224'
"""
HuggingFace identifier for Vision Transformer (ViT-Base) with 16x16 patches.
Pretrained on ImageNet-21k at 224x224 resolution.
Used as: Transformer branch in bottleneck for global context
Patch size: 16x16, embedding dim: 768, heads: 12, layers: 12
"""

VIT_PRETRAINED_PATH_32_384: str = 'google/vit-base-patch32-384'
"""
Alternative ViT configuration with 32x32 patches at 384x384.
Larger patches = faster but less fine-grained attention.
Currently not used (VIT_PRETRAINED_PATH_16_224 preferred).
"""
