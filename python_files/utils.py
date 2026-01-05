"""
Utility functions for Longitudinal CXR Analysis.

This module provides helper functions for:
- Image processing (cropping, masking)
- Learning rate scheduling (mask probability)
- Loss computation helpers
- Lung segmentation utilities
- Visualization helpers
"""

import os.path
from math import e

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from constants import (
    DEVICE, 
    MIMIC_FOLDER, 
    MASK_PATCH_SIZE,
    INIT_MASK_PROB, 
    MAX_MASK_PROB, 
    END_MASK_PROB
)

# Binary structure for morphological operations
STRUCT = ndi.generate_binary_structure(2, 2)


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def crop_out_edges(arr: np.ndarray, get_coords: bool = False):
    """
    Crop black/white borders from an image.
    
    Removes uniform intensity regions at the edges of the image,
    commonly found in chest X-rays due to collimation.
    
    Args:
        arr: 2D numpy array (image)
        get_coords: If True, also return crop coordinates
        
    Returns:
        Cropped array, or (cropped_array, y_min, x_min, y_max, x_max) if get_coords=True
    """
    min_v = np.min(arr)
    max_v = np.max(arr)
    coord_bounds_to_keep = np.argwhere(np.logical_and(arr > min_v, arr < max_v))
    min_c = np.min(coord_bounds_to_keep, axis=0)
    max_c = np.max(coord_bounds_to_keep, axis=0)
    arr = arr[min_c[0]: max_c[0] + 1, min_c[1]: max_c[1] + 1]
    
    if get_coords:
        return arr, min_c[0], min_c[1], max_c[0] + 1, max_c[1] + 1
    return arr


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_mimic_path(subject: str, study: str, dicom: str, add_cropped: bool = False) -> str:
    """
    Construct the file path for a MIMIC-CXR image.
    
    Args:
        subject: Patient ID (e.g., '10000032')
        study: Study ID
        dicom: DICOM filename
        add_cropped: If True, append '_cropped.nii.gz' suffix
        
    Returns:
        Full path to the image file
    """
    if not add_cropped:
        return f'{MIMIC_FOLDER}/other_files/p{str(subject)[:2]}/p{subject}/s{study}/{dicom}'
    return f'{MIMIC_FOLDER}/other_files/p{str(subject)[:2]}/p{subject}/s{study}/{dicom}_cropped.nii.gz'


def check_existence_of_mimic_path(path: str) -> bool:
    """Check if a MIMIC-CXR image file exists."""
    return os.path.exists(path)


# =============================================================================
# MASK PROBABILITY SCHEDULER
# =============================================================================

class MaskProbScheduler:
    """
    Scheduler for mask probability during Masked Image Modeling training.
    
    Implements a multi-phase schedule:
    1. Initial phase: constant low probability
    2. Ramp-up phase: linearly increase to max probability
    3. Max phase: constant max probability
    4. Ramp-down phase: linearly decrease to end probability
    5. Final phase: constant end probability
    
    Args:
        epochs: Total training epochs
        steps_per_epoch: Number of steps per epoch
        init_val: Initial mask probability
        max_val: Maximum mask probability
        end_val: Final mask probability
        perc_on_start: Fraction of training at initial value
        perc_on_slope: Fraction for first ramp-up
        perc_on_max: Fraction at maximum value
        perc_on_slope2: Fraction for ramp-down
    """
    
    def __init__(
        self, 
        epochs: int, 
        steps_per_epoch: int, 
        init_val: float = INIT_MASK_PROB, 
        max_val: float = MAX_MASK_PROB, 
        end_val: float = END_MASK_PROB, 
        perc_on_start: float = 0.05, 
        perc_on_slope: float = 0.2, 
        perc_on_max: float = 0.4, 
        perc_on_slope2: float = 0.1
    ):
        total_steps = epochs * steps_per_epoch
        self.init_val = init_val
        self.max_val = max_val
        self.end_val = end_val
        
        # Phase thresholds
        self.th1 = perc_on_start * total_steps
        self.th2 = self.th1 + perc_on_slope * total_steps
        self.th3 = self.th2 + perc_on_max * total_steps
        self.th4 = self.th3 + perc_on_slope2 * total_steps
        
        # Slopes for linear phases
        self.slope1 = (max_val - init_val) / (self.th2 - self.th1) if self.th2 > self.th1 else 0
        self.slope2 = (end_val - max_val) / (self.th4 - self.th3) if self.th4 > self.th3 else 0
        self.cur_step = 0

    def get_step(self) -> int:
        """Get current step number."""
        return self.cur_step

    def set_step(self, step: int):
        """Set current step number."""
        self.cur_step = step

    def calc_cur_val(self) -> float:
        """Calculate mask probability for current step."""
        if self.cur_step <= self.th1:
            return self.init_val
        elif self.th1 < self.cur_step <= self.th2:
            return (self.cur_step - self.th1) * self.slope1 + self.init_val
        elif self.th2 < self.cur_step <= self.th3:
            return self.max_val
        elif self.th3 < self.cur_step <= self.th4:
            return (self.cur_step - self.th3) * self.slope2 + self.max_val
        else:
            return self.end_val

    def step(self) -> float:
        """Advance one step and return the mask probability."""
        val = self.calc_cur_val()
        self.cur_step += 1
        return val


# =============================================================================
# LOSS COMPUTATION HELPERS
# =============================================================================

def masked_patches_l1_loss(
    outputs: torch.Tensor, 
    inputs: torch.Tensor, 
    mask: torch.Tensor, 
    patch_size: int = MASK_PATCH_SIZE
) -> torch.Tensor | None:
    """
    Compute L1 loss only on masked patches.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        inputs: Ground truth [B, C, H, W]
        mask: Binary mask where 1=unmasked, 0=masked [B, num_patches]
        patch_size: Size of each patch
        
    Returns:
        L1 loss on masked regions, or None if no masked patches
    """
    inv_mask = 1. - mask
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    
    f_inputs = F.unfold(inputs, kernel_size=patch_size, stride=patch_size)
    masked_inputs = f_inputs * inv_mask
    
    loss = F.l1_loss(masked_outputs, masked_inputs, reduction='sum')
    loss = loss / (torch.sum(inv_mask) * (patch_size ** 2))
    return loss


def fourier(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D Fourier transform magnitude (log scale).
    
    Args:
        x: Input tensor [B, C, H, W]
        
    Returns:
        Log-magnitude of FFT, centered [B, C, H, W//2+1]
    """
    b, c, h, w = x.shape
    f = torch.fft.rfft2(x.to(torch.float32))
    f = f.abs() + 1e-6
    f = f.log()
    f = torch.roll(f, shifts=(int(h/2), int(w/2)), dims=(2, 3))
    return f


def masked_patches_fourier_loss(
    outputs: torch.Tensor, 
    inputs_fourier: torch.Tensor, 
    mask: torch.Tensor, 
    patch_size: int = MASK_PATCH_SIZE
) -> torch.Tensor | None:
    """
    Compute Fourier domain loss on masked patches.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        inputs_fourier: Pre-computed Fourier of ground truth
        mask: Binary mask
        patch_size: Size of each patch
        
    Returns:
        L1 loss in Fourier domain, or None if no masked patches
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs = masked_outputs + unmasked_outputs
    outputs = F.fold(outputs, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    
    loss = F.l1_loss(fourier(outputs), inputs_fourier)
    return loss


def masked_patches_SSIM_loss(
    outputs: torch.Tensor, 
    inputs: torch.Tensor, 
    mask: torch.Tensor, 
    ssim1, 
    ssim2, 
    patch_size: int = MASK_PATCH_SIZE
):
    """
    Compute SSIM loss on masked patches.
    
    Args:
        outputs: Model predictions [B, C, H, W]
        inputs: Ground truth [B, C, H, W]
        mask: Binary mask
        ssim1: First SSIM metric (e.g., SSIM)
        ssim2: Second SSIM metric (e.g., MS-SSIM)
        patch_size: Size of each patch
        
    Returns:
        Tuple of (ssim_loss, ms_ssim_loss), or None if no masked patches
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs = masked_outputs + unmasked_outputs
    outputs = F.fold(outputs, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    
    loss1 = 1 - ssim1(outputs, inputs)
    loss2 = 1 - ssim2(outputs, inputs)
    return loss1, loss2


def masked_patches_GAN_loss(
    outputs: torch.Tensor, 
    mask: torch.Tensor, 
    disc, 
    gan_loss, 
    patch_size: int = MASK_PATCH_SIZE
) -> torch.Tensor | None:
    """
    Compute GAN generator loss on masked patches.
    
    Args:
        outputs: Generator predictions [B, C, H, W]
        mask: Binary mask
        disc: Discriminator network
        gan_loss: GAN loss function (e.g., BCEWithLogitsLoss)
        patch_size: Size of each patch
        
    Returns:
        Generator loss, or None if no masked patches
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs = masked_outputs + unmasked_outputs
    outputs = F.fold(outputs, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    
    disc_fake_outputs = disc(outputs)
    real_labels = torch.ones_like(disc_fake_outputs, device=disc_fake_outputs.device)
    loss = gan_loss(disc_fake_outputs, real_labels)
    return loss


# =============================================================================
# LUNG SEGMENTATION UTILITIES
# =============================================================================

def get_sep_lung_masks(seg: np.ndarray, ret_right_then_left: bool = False):
    """
    Separate left and right lung masks from a combined lung segmentation.
    
    Uses connected component analysis to identify individual lungs.
    Handles edge cases with warnings.
    
    Args:
        seg: Binary lung segmentation mask
        ret_right_then_left: If True, return (right_lung, left_lung) based on x-position
        
    Returns:
        Tuple of (lung1_mask, lung2_mask) as boolean arrays
    """
    label, _ = ndi.label(seg, structure=STRUCT)
    ccs, counts = np.unique(label, return_counts=True)
    num_ccs = len(ccs)
    
    # Handle edge cases
    if num_ccs == 2:
        print("Warning: Only one lung found.")
        lung1 = label == ccs[1]
        lung1_coords = np.argwhere(lung1)
        top1 = lung1_coords[0]
        top2 = (top1[0], seg.shape[-1] - top1[1])
        label[top2[0]: top2[0] + 3, top2[1]] = 2
        lung2 = label == 2
        return lung1, lung2
        
    if num_ccs == 1:
        print("Warning: No lungs could be found.")
        y = seg.shape[-2] // 8
        lung1_x = (seg.shape[-1] * 2) // 5
        lung2_x = (seg.shape[-1] * 3) // 5
        label[y: y + 3, lung1_x] = 1
        label[y: y + 3, lung2_x] = 2
        lung1 = label == 1
        lung2 = label == 2
        return lung1, lung2
        
    if num_ccs > 3:
        print("Warning: More than 2 connectivity components found. Using 2 largest ones.")
        ocs = list(zip(ccs, counts))
        sorted_ocs = sorted(ocs, key=lambda x: x[1], reverse=True)
        label1 = sorted_ocs[1][0]
        label2 = sorted_ocs[2][0]
    else:
        label1 = 1
        label2 = 2
    
    lung1 = label == label1
    lung2 = label == label2
    
    # Optionally order by x-position (right then left)
    if ret_right_then_left:
        reg1 = lung1 == 1
        count1 = reg1.sum()
        y_center1, x_center1 = np.argwhere(reg1).sum(0) / count1
        
        reg2 = lung2 == 1
        count2 = reg2.sum()
        y_center2, x_center2 = np.argwhere(reg2).sum(0) / count2
        
        if x_center1 > x_center2:
            return lung2, lung1
        return lung1, lung2
    
    return lung1, lung2


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def scale_and_suppress_non_max(
    cur_img: torch.Tensor, 
    new_upper_bound: float = 0.3, 
    scale_fac: float = 1.0
) -> torch.Tensor:
    """
    Scale image values exponentially to suppress low values.
    
    Useful for enhancing visualization of difference maps.
    
    Args:
        cur_img: Input image tensor
        new_upper_bound: Maximum value after scaling
        scale_fac: Exponential scaling factor
        
    Returns:
        Scaled image with same sign as input
    """
    abs_img = cur_img.abs()
    abs_max = torch.max(abs_img).item()
    new_max = min(abs_max, new_upper_bound)
    scaled_img = torch.exp(scale_fac * abs_img) - 1
    scaled_img = scaled_img / (torch.max(scaled_img) / new_max)
    cur_img = scaled_img * torch.sign(cur_img)
    return cur_img


def get_max_inpaint_diff_val(init_max: float) -> float:
    """
    Calculate expected maximum difference value for inpainting.
    
    Empirically derived formula for normalizing inpainted regions.
    
    Args:
        init_max: Initial maximum value
        
    Returns:
        Expected maximum difference value
    """
    return -0.34 + 0.725 * (1 - e ** (-5.4 * init_max))


def generate_alpha_map(x: torch.Tensor) -> torch.Tensor:
    """
    Generate an alpha (transparency) map from a difference image.
    
    Creates a map where higher absolute values have higher opacity,
    useful for overlaying difference maps on base images.
    
    Args:
        x: Difference map tensor
        
    Returns:
        Alpha map in range [0, 1]
    """
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val
    alphas_map = alphas_map.squeeze()
    return alphas_map
