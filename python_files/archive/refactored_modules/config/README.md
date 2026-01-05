# Configuration Module (`config/`)

Centralized configuration for the Longitudinal CXR Analysis project. All hyperparameters, paths, and settings are documented and organized by category.

## Module Structure

```
config/
├── __init__.py          # Re-exports all config values
├── device.py            # GPU/CPU device selection
├── paths.py             # Dataset and output paths
├── model_config.py      # Model architecture parameters
├── training_config.py   # Training hyperparameters
└── data_config.py       # Dataset labels and mappings
```

## Files Overview

### `device.py`
Automatic device selection with memory-based GPU preference.

```python
from config import DEVICE

model = model.to(DEVICE)
```

### `paths.py`
All hardcoded paths in one place. Supports environment variable overrides.

```python
from config import MIMIC_CXR_PATH, CXR14_FOLDER, CHECKPOINTS_DIR

# Override via environment:
# export MIMIC_CXR_PATH=/new/path
```

**Key paths:**
- `MIMIC_CXR_PATH` - MIMIC-CXR dataset
- `CXR14_FOLDER` - NIH ChestX-ray14
- `PADCHEST_FOLDER` - PadChest dataset
- `CHECKPOINTS_DIR` - Model checkpoints
- `EFFICIENT_NET_PRETRAINED_PATH` - Pretrained weights

### `model_config.py`
Neural network architecture parameters.

```python
from config import EMBED_DIM, NUM_HEADS, FEATURE_CHANNELS

# Architecture constants:
# - EMBED_DIM: Transformer embedding dimension
# - NUM_HEADS: Attention heads
# - EFF_NET_BLOCK_IDXS: Which EfficientNet blocks to use
```

### `training_config.py`
Training hyperparameters and loss configurations.

```python
from config import (
    BATCH_SIZE, MAX_LR, 
    USE_MASKED_LOSS, PERCEPTUAL_LOSS_WEIGHT
)
```

**Key parameters:**
- Learning rate scheduling
- Loss function weights
- Masking configuration for MIM

### `data_config.py`
Dataset label definitions and mappings.

```python
from config import LABELS, CSVS_TO_LABEL_MAPPING

# LABELS: List of disease labels
# CSVS_TO_LABEL_MAPPING: Maps dataset CSVs to their labels
```

## Usage

Import what you need:

```python
# Import specific values
from config import DEVICE, BATCH_SIZE, MIMIC_CXR_PATH

# Or import by category
from config.model_config import *
from config.training_config import *
```

## Adding New Configuration

1. Identify the appropriate file (paths, model, training, data)
2. Add the constant with a descriptive docstring
3. Export it in `__init__.py`

Example:
```python
# In training_config.py
NEW_PARAM: int = 42
"""
Description of what this parameter controls.

Used in:
    - training_script.py
    - evaluation.py
"""
```
