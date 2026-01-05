"""
Image Processing Utilities
==========================

Functions for image preprocessing like histogram equalization,
normalization, and common image transformations.

Usage:
------
    from utils.image_processing import histogram_equalization, normalize_image
"""

import numpy as np
import torch
from typing import Tuple, Optional


def histogram_equalization(
    image: np.ndarray,
    number_bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform histogram equalization on an image.
    
    Redistributes pixel intensities to achieve a more uniform histogram,
    enhancing contrast in the image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, any shape (will be flattened internally).
    number_bins : int, default=256
        Number of histogram bins.
        
    Returns
    -------
    image_equalized : np.ndarray
        Equalized image, same shape as input.
    cdf : np.ndarray
        Cumulative distribution function used for transformation.
        
    Examples
    --------
    >>> img = np.random.rand(256, 256)
    >>> eq_img, cdf = histogram_equalization(img)
    
    References
    ----------
    http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    """
    # Compute histogram
    image_histogram, bins = np.histogram(
        image.flatten(), 
        number_bins, 
        density=True
    )
    
    # Compute cumulative distribution function
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1]  # Normalize
    
    # Use linear interpolation to map pixels
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape), cdf


def normalize_image(
    image: torch.Tensor,
    method: str = 'minmax'
) -> torch.Tensor:
    """
    Normalize image tensor.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor.
    method : str, default='minmax'
        Normalization method:
        - 'minmax': Scale to [0, 1]
        - 'zscore': Zero mean, unit variance
        - 'imagenet': ImageNet mean/std normalization
        
    Returns
    -------
    torch.Tensor
        Normalized image.
    """
    if method == 'minmax':
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    
    elif method == 'imagenet':
        # ImageNet normalization (for RGB)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        if image.device != mean.device:
            mean = mean.to(image.device)
            std = std.to(image.device)
        return (image - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def crop_to_content(
    arr: np.ndarray,
    return_coords: bool = False
) -> np.ndarray:
    """
    Crop array to non-edge content.
    
    Removes border regions that contain only min or max values.
    Useful for removing padding/background from medical images.
    
    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    return_coords : bool, default=False
        Whether to return crop coordinates.
        
    Returns
    -------
    arr : np.ndarray
        Cropped array.
    coords : tuple (only if return_coords=True)
        (min_row, min_col, max_row, max_col) crop boundaries.
    """
    min_v = np.min(arr)
    max_v = np.max(arr)
    
    # Find coordinates that aren't min or max
    content_mask = np.logical_and(arr > min_v, arr < max_v)
    content_coords = np.argwhere(content_mask)
    
    if len(content_coords) == 0:
        # No content found, return as-is
        if return_coords:
            return arr, (0, 0, arr.shape[0], arr.shape[1])
        return arr
    
    min_c = np.min(content_coords, axis=0)
    max_c = np.max(content_coords, axis=0)
    
    cropped = arr[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1]
    
    if return_coords:
        return cropped, (min_c[0], min_c[1], max_c[0]+1, max_c[1]+1)
    return cropped


def scale_and_suppress(
    image: torch.Tensor,
    upper_bound: float = 0.3,
    scale_factor: float = 1.0
) -> torch.Tensor:
    """
    Scale image values and suppress small values exponentially.
    
    Applies exponential scaling to emphasize larger values while
    suppressing noise/small values. Useful for difference maps.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image (can have positive and negative values).
    upper_bound : float, default=0.3
        Maximum output magnitude.
    scale_factor : float, default=1.0
        Exponential scaling factor.
        
    Returns
    -------
    torch.Tensor
        Scaled image with same sign as input.
    """
    abs_img = image.abs()
    abs_max = torch.max(abs_img).item()
    new_max = min(abs_max, upper_bound)
    
    # Exponential scaling
    scaled = torch.exp(scale_factor * abs_img) - 1
    scaled = scaled / (torch.max(scaled) / new_max)
    
    # Restore sign
    return scaled * torch.sign(image)
