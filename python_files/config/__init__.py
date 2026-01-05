"""
Configuration Package for Longitudinal CXR Analysis Project
============================================================

This package contains all configuration settings organized by domain:

Modules:
--------
- paths: File system paths and directories
- model_config: Neural network architecture parameters
- training_config: Training hyperparameters and loss settings
- data_config: Data processing, augmentation, and label configurations

Usage:
------
    # Import all configs
    from config import *
    
    # Import specific modules
    from config.paths import PROJECT_FOLDER, MIMIC_FOLDER
    from config.model_config import EMBED_DIM, NUM_HEADS
    from config.training_config import BATCH_SIZE, MAX_LR
    from config.data_config import IMG_SIZE, CUR_LABELS

Environment Variables:
----------------------
The paths module supports environment variable overrides:
    - THESIS_PROJECT_ROOT: Override base project directory
    - THESIS_DATA_ROOT: Override data storage directory

Author: Longitudinal CXR Analysis Team
"""

from config.device import DEVICE
from config.paths import *
from config.model_config import *
from config.training_config import *
from config.data_config import *
