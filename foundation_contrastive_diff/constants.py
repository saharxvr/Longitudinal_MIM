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
# IMAGE / FEATURE DIMENSIONS  (CheXFound ViT-L/16 @ 512)
# =============================================================================
IMG_SIZE = 512                  # Input CXR size (pixels) — CheXFound is trained at 512
PATCH_SIZE = 16                 # ViT-L/16 patch size
FEATURE_GRID = IMG_SIZE // PATCH_SIZE   # 32 x 32 patch grid
NUM_PATCH_TOKENS = FEATURE_GRID ** 2    # 1024 patch tokens (+ 1 [CLS])
BACKBONE_DIM = 1024             # ViT-L embedding dim (D_model)
LAST_N_LAYERS = 4               # Concatenate patch tokens from last N layers (CheXFound default)

EMBED_DIM = 256                 # Difference embedding dimensionality (z)
PROJ_DIM = 128                  # Projection-head output dimensionality

# =============================================================================
# BACKBONE  (CheXFound — arXiv:2502.05142, github.com/RPIDIAL/CheXFound, MIT)
# =============================================================================
# One of: 'chexfound' (default), 'rad_dino', 'imagenet_vit', 'parent_efficientnet'
BACKBONE = 'chexfound'
CHEXFOUND_CHECKPOINT = ''       # path to teacher_checkpoint.pth (Google Drive)
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
