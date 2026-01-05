"""
Utils Package - Shared Utilities
================================

Centralized utility functions used across the project.

Modules:
--------
- metrics: Evaluation metrics (dice coefficient, etc.)
- visualization: Plotting and alpha map generation
- io_utils: File I/O for NIfTI, DICOM
- image_processing: Histogram equalization, image transforms
- schedulers: Learning rate and mask probability schedulers
- losses: Loss function utilities

Usage:
------
    from utils.metrics import dice_coefficient
    from utils.visualization import create_alpha_map, plot_diff_on_image
    from utils.io_utils import save_as_nifti, load_nifti
"""

from utils.metrics import dice_coefficient
from utils.visualization import create_alpha_map
from utils.io_utils import save_as_nifti, load_nifti
from utils.schedulers import MaskProbScheduler
