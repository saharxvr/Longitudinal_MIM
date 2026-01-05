"""
Data Preprocessing Utilities.

This module contains utilities for preprocessing CXR and CT datasets:
- Image format conversion (PNG, DICOM, NIfTI)
- Dataset CSV creation and filtering
- Pair creation for longitudinal studies
- Image cropping and resizing

Most functions operate on specific dataset paths and may need
modification for different data locations.

Supported Datasets:
- MIMIC-CXR
- CXR-14 (ChestX-ray14/NIH)
- PadChest
- VinDr-CXR
- CheXpert
- Pneumonia/Normal dataset
"""

from data_prep.io_operations import (
    convert_png_to_nifti,
    convert_dicom_to_nifti,
    convert_cases_to_nib,
    resize_images,
)

from data_prep.dataset_csv import (
    create_no_finding_csv,
    create_specific_abnormalities_csv,
    filter_relevant_images,
)

from data_prep.pair_creation import (
    create_longitudinal_pairs,
    days_between,
    is_within_n_days,
)

from data_prep.image_processing import (
    crop_edges,
    normalize_intensity,
    invert_photometric,
)

__all__ = [
    # IO
    "convert_png_to_nifti",
    "convert_dicom_to_nifti",
    "convert_cases_to_nib",
    "resize_images",
    # CSV
    "create_no_finding_csv",
    "create_specific_abnormalities_csv",
    "filter_relevant_images",
    # Pairs
    "create_longitudinal_pairs",
    "days_between",
    "is_within_n_days",
    # Image
    "crop_edges",
    "normalize_intensity",
    "invert_photometric",
]
