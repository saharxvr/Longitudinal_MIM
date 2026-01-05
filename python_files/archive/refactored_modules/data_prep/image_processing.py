"""
Image Processing Utilities for CXR Preprocessing.

Functions for image-level preprocessing operations like
edge cropping, intensity normalization, and photometric inversion.
"""

import os
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

import torch
import nibabel as nib
import pydicom

# Standard affine matrix for medical images
AFFINE_DCM = np.array([
    [-0.139, 0, 0, 0],
    [0, -0.139, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def crop_edges(
    image: np.ndarray,
    threshold_ratio: float = 0.1,
    min_crop: int = 5
) -> np.ndarray:
    """
    Crop black edges from a CXR image.
    
    Detects and removes the black borders that often appear in
    digitized radiographs.
    
    Args:
        image: 2D numpy array
        threshold_ratio: Pixels below this fraction of max are considered black
        min_crop: Minimum pixels to always crop from each edge
    
    Returns:
        Cropped image array
    
    Example:
        >>> img = nib.load("cxr.nii.gz").get_fdata()
        >>> cropped = crop_edges(img)
    """
    threshold = np.max(image) * threshold_ratio
    
    # Find non-black regions
    mask = image > threshold
    
    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return image  # Return original if all black
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    # Apply minimum crop
    row_min = max(min_crop, row_min)
    row_max = min(image.shape[0] - min_crop, row_max)
    col_min = max(min_crop, col_min)
    col_max = min(image.shape[1] - min_crop, col_max)
    
    return image[row_min:row_max+1, col_min:col_max+1]


def normalize_intensity(
    image: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 255.0,
    clip_percentile: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize image intensity to a target range.
    
    Args:
        image: Input image array
        target_min: Minimum output value
        target_max: Maximum output value
        clip_percentile: Optional (low, high) percentiles for clipping
                        before normalization (e.g., (1, 99))
    
    Returns:
        Normalized image
    
    Example:
        >>> img = normalize_intensity(img, clip_percentile=(1, 99))
    """
    if clip_percentile is not None:
        low = np.percentile(image, clip_percentile[0])
        high = np.percentile(image, clip_percentile[1])
        image = np.clip(image, low, high)
    
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.full_like(image, (target_max + target_min) / 2)
    
    normalized = (image - img_min) / (img_max - img_min)
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def invert_photometric(
    image: np.ndarray,
    max_value: float = 255.0
) -> np.ndarray:
    """
    Invert image intensity (for MONOCHROME1 to MONOCHROME2 conversion).
    
    In MONOCHROME1, minimum pixel value = white.
    In MONOCHROME2, minimum pixel value = black.
    
    Args:
        image: Input image array
        max_value: Maximum intensity value
    
    Returns:
        Inverted image
    """
    return max_value - image


def batch_invert_monochrome1(
    dicom_folder: str,
    image_folder: str,
    output_folder: str
) -> None:
    """
    Batch invert MONOCHROME1 images based on DICOM metadata.
    
    Reads DICOM files to check PhotometricInterpretation, then
    inverts corresponding NIfTI images if needed.
    
    Args:
        dicom_folder: Folder containing DICOM files
        image_folder: Folder containing NIfTI images
        output_folder: Output folder for inverted images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for dcm_name in tqdm(os.listdir(dicom_folder)):
        dcm_path = os.path.join(dicom_folder, dcm_name)
        
        try:
            dcm = pydicom.dcmread(dcm_path)
            
            if not hasattr(dcm, 'PhotometricInterpretation'):
                continue
            
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                # Find corresponding NIfTI
                nifti_name = dcm_name.rsplit('.', 1)[0] + '.nii.gz'
                nifti_path = os.path.join(image_folder, nifti_name)
                
                if not os.path.exists(nifti_path):
                    continue
                
                # Invert and save
                data = nib.load(nifti_path).get_fdata()
                inverted = 255 - data
                
                output_path = os.path.join(output_folder, nifti_name)
                nif = nib.Nifti1Image(inverted, AFFINE_DCM)
                nib.save(nif, output_path)
                
        except Exception as e:
            print(f"Error processing {dcm_name}: {e}")


def preprocess_padchest_images(
    input_folder: str,
    output_folder: str,
    target_size: Tuple[int, int] = (512, 512),
    min_std: float = 5.0
) -> int:
    """
    Preprocess PadChest images: crop edges and resize.
    
    Args:
        input_folder: Input images folder
        output_folder: Output folder
        target_size: Target image size
        min_std: Minimum standard deviation (skip low-variance images)
    
    Returns:
        Number of processed images
    """
    import torchvision.transforms as tf
    
    os.makedirs(output_folder, exist_ok=True)
    resize = tf.Resize(target_size)
    
    processed = 0
    
    for name in tqdm(os.listdir(input_folder)):
        if not name.endswith('.nii.gz'):
            continue
        
        input_path = os.path.join(input_folder, name)
        output_path = os.path.join(output_folder, name)
        
        data = nib.load(input_path).get_fdata()
        
        # Skip low-variance images
        if np.std(data) < min_std:
            print(f"Skipping low-variance image: {name}")
            continue
        
        # Crop edges
        cropped = crop_edges(data)
        
        # Resize if needed
        if cropped.shape != target_size:
            cropped = torch.tensor(cropped[None, ...])
            cropped = np.array(resize(cropped).squeeze(0))
        
        # Save
        cropped = cropped.astype(np.uint8)
        nif = nib.Nifti1Image(cropped, AFFINE_DCM)
        nib.save(nif, output_path)
        processed += 1
    
    return processed
