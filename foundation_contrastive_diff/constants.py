"""
Configuration constants for the Foundation-Based Difference Detection study.

Follow-up to the parent Longitudinal CXR project. Keeps this study's hyperparameters
self-contained so it can evolve independently from python_files/constants.py.
"""

import torch

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# IMAGE / FEATURE DIMENSIONS  (RAD-DINO ViT-B/14 @ 518; our DRR/GT stay at 512)
# =============================================================================
IMG_SIZE = 512                  # our DRR / GT / decoder-output resolution
BACKBONE_IMG_SIZE = 518         # RAD-DINO input size (backbone wrapper resizes internally)
PATCH_SIZE = 14                 # ViT-B/14 patch size
FEATURE_GRID = BACKBONE_IMG_SIZE // PATCH_SIZE   # 37 x 37 patch grid
NUM_PATCH_TOKENS = FEATURE_GRID ** 2             # 1369 patch tokens (+ 1 [CLS])
BACKBONE_DIM = 768              # ViT-B embedding dim (per layer)
LAST_N_LAYERS = 1               # first try: last layer only (last-4 concat = ablation C)

EMBED_DIM = 256                 # Difference embedding dimensionality (z)
PROJ_DIM = 128                  # Projection-head output dimensionality

# =============================================================================
# BACKBONE
# =============================================================================
# One of: 'rad_dino' (default, HuggingFace), 'chexfound', 'imagenet_vit', 'parent_efficientnet'
BACKBONE = 'rad_dino'
RAD_DINO_MODEL = 'microsoft/rad-dino'   # HuggingFace model id (frozen CXR DINOv2 ViT-B/14)
CHEXFOUND_CHECKPOINT = ''       # path to teacher_checkpoint.pth (ablation E, later)
CHEXFOUND_CONFIG = ''           # path to CheXFound config yaml
FREEZE_BACKBONE = True          # Keep foundation weights frozen
ALLOW_LAST_BLOCK_ADAPTER = False  # Optional shallow adapter / LoRA on last block

# Frozen backbone -> precompute patch tokens once and train the head on cached tensors.
USE_FEATURE_CACHE = True
FEATURE_CACHE_DIR = './feature_cache'

# =============================================================================
# DIFFERENCE HEAD  (D-GLoRI: GLoRI re-purposed for change detection)
# =============================================================================
# Local fusion of prior/current patch tokens: 'diff' | 'concat' | 'cross_attention'
FUSION_MODE = 'diff'
D_GLORI = 768                   # GLoRI embedding dim (CheXFound default)
GLORI_NUM_HEADS = 8             # Multi-head cross-attention heads
# Generic latent change queries (NOT per-anomaly classes). The head output is a single
# signed change heatmap; these queries are an internal attention mechanism, so their count
# is a free hyperparameter and need not match len(ANOMALY_TYPES).
NUM_CHANGE_QUERIES = 8
USE_ADAPTIVE_TEMPERATURE = True # Fine-grained local-feature branch
USE_PYRAMID_PATCH_MERGING = True# Coarse-grained local-feature branch (8x8/4x4/2x2 pooling)
INTEGRATE_GLOBAL_CLS = True     # Skip-connect global [CLS] difference
DECODER_OUT_RANGE = (-1.0, 1.0)  # Signed change map (positive=new, negative=resolved)

# =============================================================================
# CONTRASTIVE / DISENTANGLEMENT
# =============================================================================
USE_CONTRASTIVE = True
USE_DISENTANGLEMENT = True
CONTRASTIVE_LOSS = 'supcon'      # 'supcon' | 'triplet'
SUPCON_TEMPERATURE = 0.1
TRIPLET_MARGIN = 0.5

# =============================================================================
# LOSS WEIGHTS  (L_total = sum of weighted terms)
# =============================================================================
LAMBDA_SEG = 1.0
LAMBDA_CONTRASTIVE = 0.5
LAMBDA_ORTHOGONALITY = 0.1
LAMBDA_DIRECTION = 0.0           # appearance vs. disappearance (optional)

# =============================================================================
# TRAINING
# =============================================================================
BATCH_SIZE = 8
MAX_LR = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 50
UPDATE_EVERY_BATCHES = 4         # gradient accumulation

# =============================================================================
# CHANGE-TYPE TAXONOMY  (supervision from the synthetic DRR pipeline)
# =============================================================================
# NOTE: devices and projection-angle (positioning) changes are rendered across ALL
# anomaly types but are NOT change classes — they are nuisance and ignored (masked out
# of the GT difference map, excluded from supervision). Cardiomegaly is also excluded
# because the synthetic pipeline does not produce cardiomegaly *changes*.
ANOMALY_TYPES = [
    'none',
    'consolidation',
    'pleural_effusion',
    'pneumothorax',
    'fluid_overload',
]
DIRECTION_TYPES = ['none', 'appearance', 'disappearance']
# 'nuisance' = non-clinical change (device insertion/removal, positioning/angle, exposure);
# these are ignored rather than detected.
PATHOLOGY_VS_NUISANCE = ['nuisance', 'pathology']

# =============================================================================
# PATHS  (fill in for your environment)
# =============================================================================
TRAIN_PAIRS_DIR = ''             # synthetic DRR pairs (BL/FU + GT diff + labels)
VAL_PAIRS_DIR = ''
REAL_TEST_PAIRS_DIR = ''         # ICU / PNIMIT annotated longitudinal pairs
SAVE_FOLDER = './checkpoints'
PLOTS_FOLDER = './plots'
