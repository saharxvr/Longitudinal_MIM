"""
Image I/O Operations for Dataset Preprocessing.

Functions for converting images between formats (PNG, DICOM, NIfTI)
and for resizing/transforming images for model input.
"""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

import torch
import nibabel as nib
import pydicom

from config.paths import CXR14_FOLDER


# Standard affine for medical images (0.139mm pixel spacing)
AFFINE_DCM = np.array([
    [-0.139, 0, 0, 0],
    [0, -0.139, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def convert_png_to_nifti(
    input_path: str,
    output_path: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None
) -> str:
    """
    Convert a PNG image to NIfTI format.
    
    Args:
        input_path: Path to input PNG file
        output_path: Path for output NIfTI file. If None, uses same name with .nii.gz
        target_size: Optional (width, height) to resize to
    
    Returns:
        Path to created NIfTI file
    
    Example:
        >>> convert_png_to_nifti("image.png", target_size=(512, 512))
    """
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.nii.gz'
    
    img = Image.open(input_path).convert("L")
    arr = np.array(img.getdata()).astype(np.uint8).reshape((img.height, img.width)).T
    
    if target_size is not None:
        from torchvision.transforms import Resize
        resize_transform = Resize(target_size)
        arr = torch.tensor(arr[None, ...])
        arr = np.array(resize_transform(arr).squeeze(0))
    
    nif = nib.Nifti1Image(arr, AFFINE_DCM)
    nib.save(nif, output_path)
    
    return output_path


def convert_dicom_to_nifti(
    input_path: str,
    output_path: Optional[str] = None,
    invert_if_monochrome1: bool = True
) -> str:
    """
    Convert a DICOM file to NIfTI format.
    
    Args:
        input_path: Path to input DICOM file
        output_path: Path for output NIfTI file
        invert_if_monochrome1: Whether to invert MONOCHROME1 images
    
    Returns:
        Path to created NIfTI file
    
    Note:
        Handles photometric interpretation - MONOCHROME1 images are
        inverted to match MONOCHROME2 convention.
    """
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.nii.gz'
    
    dcm = pydicom.dcmread(input_path)
    data = dcm.pixel_array.astype(float)
    
    # Normalize to 0-255
    data_min, data_max = np.min(data), np.max(data)
    if data_max > data_min:
        data = 255 * (data - data_min) / (data_max - data_min)
    
    # Handle photometric interpretation
    if invert_if_monochrome1:
        if hasattr(dcm, 'PhotometricInterpretation'):
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                data = 255 - data
    
    data = data.astype(np.uint8).T
    nif = nib.Nifti1Image(data, AFFINE_DCM)
    nib.save(nif, output_path)
    
    return output_path


def convert_cases_to_nib(
    input_folder: str,
    output_folder: Optional[str] = None,
    extensions: Tuple[str, ...] = ('.png', '.PNG', '.jpg', '.jpeg')
) -> None:
    """
    Batch convert image files to NIfTI format.
    
    Args:
        input_folder: Folder containing images
        output_folder: Output folder. If None, saves in input_folder
        extensions: File extensions to process
    """
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_folder)):
        if not any(filename.endswith(ext) for ext in extensions):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_name = filename.rsplit('.', 1)[0] + '.nii.gz'
        output_path = os.path.join(output_folder, output_name)
        
        img = Image.open(input_path).convert("L")
        arr = np.array(img.getdata()).reshape((img.height, img.width)).T
        arr = arr.astype(np.uint8)
        
        nif = nib.Nifti1Image(arr, AFFINE_DCM)
        nib.save(nif, output_path)


def resize_images(
    input_folder: str,
    output_folder: str,
    target_size: Tuple[int, int] = (512, 512),
    include_segmentations: bool = False,
    seg_folder: Optional[str] = None,
    seg_output_folder: Optional[str] = None
) -> None:
    """
    Resize NIfTI images to a target size.
    
    Args:
        input_folder: Folder with input images
        output_folder: Folder for resized images
        target_size: (width, height) target size
        include_segmentations: Whether to also resize segmentation masks
        seg_folder: Input folder for segmentations
        seg_output_folder: Output folder for segmentations
    
    Note:
        Segmentation masks use nearest-neighbor interpolation to
        preserve label values.
    """
    import torchvision.transforms.v2 as v2
    
    os.makedirs(output_folder, exist_ok=True)
    
    resize = v2.Resize(target_size, antialias=True)
    resize_seg = v2.Resize(target_size, antialias=False)  # Nearest-neighbor for masks
    
    for filename in tqdm(os.listdir(input_folder)):
        if not filename.endswith('.nii.gz'):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        img = torch.tensor(nib.load(input_path).get_fdata().T[None, ...])
        img = np.array(resize(img).squeeze()).astype(np.uint8).T
        
        nif = nib.Nifti1Image(img, AFFINE_DCM)
        nib.save(nif, output_path)
    
    # Handle segmentations
    if include_segmentations and seg_folder is not None:
        os.makedirs(seg_output_folder, exist_ok=True)
        
        for filename in tqdm(os.listdir(seg_folder)):
            if not filename.endswith('.nii.gz'):
                continue
            
            input_path = os.path.join(seg_folder, filename)
            output_path = os.path.join(seg_output_folder, filename)
            
            img = torch.tensor(nib.load(input_path).get_fdata().T[None, ...])
            img = np.array(resize_seg(img).squeeze()).astype(np.uint8).T
            
            nif = nib.Nifti1Image(img, AFFINE_DCM)
            nib.save(nif, output_path)


def batch_convert_cxr14_to_nifti(images_folder: str) -> None:
    """
    Convert all CXR-14 PNG images to NIfTI format.
    
    Args:
        images_folder: Path to CXR-14 images folder
    """
    files = glob(os.path.join(images_folder, '*.png'))
    
    for i, filepath in enumerate(tqdm(files)):
        convert_png_to_nifti(filepath)
        os.remove(filepath)
        
        if i % 1000 == 0:
            print(f'Converted {i+1} images')
