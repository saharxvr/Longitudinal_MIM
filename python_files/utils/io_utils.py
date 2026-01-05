"""
File I/O Utilities
==================

Functions for loading and saving medical images (NIfTI, DICOM).

Usage:
------
    from utils.io_utils import save_as_nifti, load_nifti, load_scan
"""

import os
import numpy as np
import nibabel as nib
import torch
from typing import Union, Optional, Tuple

# Default affine matrix for NIfTI files (identity)
DEFAULT_AFFINE = np.eye(4)


def save_as_nifti(
    data: Union[np.ndarray, torch.Tensor],
    output_path: str,
    affine: Optional[np.ndarray] = None
) -> None:
    """
    Save array as NIfTI file (.nii.gz).
    
    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Image data to save. Can be 2D or 3D.
    output_path : str
        Output file path. Should end with .nii.gz or .nii
    affine : np.ndarray, optional
        4x4 affine transformation matrix. 
        Default is identity matrix.
        
    Examples
    --------
    >>> img = np.random.rand(256, 256)
    >>> save_as_nifti(img, 'output.nii.gz')
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    if affine is None:
        affine = DEFAULT_AFFINE
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)


def load_nifti(
    file_path: str,
    return_affine: bool = False,
    as_tensor: bool = False,
    dtype: Optional[torch.dtype] = None
) -> Union[np.ndarray, torch.Tensor, Tuple]:
    """
    Load NIfTI file.
    
    Parameters
    ----------
    file_path : str
        Path to NIfTI file (.nii or .nii.gz).
    return_affine : bool, default=False
        Whether to also return the affine matrix.
    as_tensor : bool, default=False
        Whether to return as torch.Tensor instead of np.ndarray.
    dtype : torch.dtype, optional
        Data type for tensor conversion.
        
    Returns
    -------
    data : np.ndarray or torch.Tensor
        Image data.
    affine : np.ndarray (only if return_affine=True)
        4x4 affine transformation matrix.
        
    Examples
    --------
    >>> img = load_nifti('scan.nii.gz')
    >>> img, affine = load_nifti('scan.nii.gz', return_affine=True)
    >>> tensor = load_nifti('scan.nii.gz', as_tensor=True)
    """
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    
    if as_tensor:
        data = torch.tensor(data)
        if dtype:
            data = data.to(dtype)
    
    if return_affine:
        return data, nifti_img.affine
    return data


def load_scan(
    file_path: str,
    add_channel: bool = True,
    normalize: bool = False
) -> torch.Tensor:
    """
    Load medical scan as tensor, ready for model input.
    
    Parameters
    ----------
    file_path : str
        Path to NIfTI file.
    add_channel : bool, default=True
        Whether to add channel dimension at front.
    normalize : bool, default=False
        Whether to normalize to [0, 1].
        
    Returns
    -------
    torch.Tensor
        Scan data as tensor.
        Shape: (1, H, W) if add_channel else (H, W)
    """
    data = load_nifti(file_path, as_tensor=True, dtype=torch.float32)
    
    if normalize:
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    if add_channel:
        data = data.unsqueeze(0)
    
    return data


def get_mimic_path(
    subject: Union[str, int],
    study: Union[str, int],
    dicom: str,
    mimic_folder: str,
    add_cropped: bool = False
) -> str:
    """
    Construct MIMIC-CXR file path from subject/study/dicom identifiers.
    
    Parameters
    ----------
    subject : str or int
        Patient subject ID.
    study : str or int
        Study ID.
    dicom : str
        DICOM filename.
    mimic_folder : str
        Root MIMIC folder path.
    add_cropped : bool, default=False
        Whether to add '_cropped.nii.gz' suffix.
        
    Returns
    -------
    str
        Full file path.
    """
    subject = str(subject)
    study = str(study)
    
    if not add_cropped:
        return f"{mimic_folder}/other_files/p{subject[:2]}/p{subject}/s{study}/{dicom}"
    return f"{mimic_folder}/other_files/p{subject[:2]}/p{subject}/s{study}/{dicom}_cropped.nii.gz"


def check_file_exists(path: str) -> bool:
    """Check if file exists at given path."""
    return os.path.exists(path)
