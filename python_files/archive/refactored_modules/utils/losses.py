"""
Loss Function Utilities
=======================

Custom loss functions for image reconstruction, including
masked patch losses, Fourier domain losses, and GAN losses.

Usage:
------
    from utils.losses import masked_l1_loss, fourier_loss
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def masked_patches_l1_loss(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 128
) -> Optional[torch.Tensor]:
    """
    Compute L1 loss only on masked patch regions.
    
    For Masked Image Modeling, we only want to compute loss on
    the regions that were masked, not the visible regions.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions, shape (B, C, H, W).
    inputs : torch.Tensor
        Ground truth images, shape (B, C, H, W).
    mask : torch.Tensor
        Binary mask where 1=visible, 0=masked.
        Shape (B, num_patches) where num_patches = (H/patch_size)^2
    patch_size : int, default=128
        Size of each patch.
        
    Returns
    -------
    torch.Tensor or None
        L1 loss on masked regions, or None if no masked regions.
    """
    # inv_mask selects masked (to-be-predicted) regions
    inv_mask = 1. - mask
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    # Unfold into patches
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    
    f_inputs = F.unfold(inputs, kernel_size=patch_size, stride=patch_size)
    masked_inputs = f_inputs * inv_mask
    
    # Compute normalized L1 loss
    loss = F.l1_loss(masked_outputs, masked_inputs, reduction='sum')
    loss = loss / (torch.sum(inv_mask) * (patch_size ** 2))
    
    return loss


def fourier_transform_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D Fourier transform magnitude (log scale).
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape (B, C, H, W).
        
    Returns
    -------
    torch.Tensor
        Log magnitude of FFT, centered.
    """
    b, c, h, w = x.shape
    
    # Compute real FFT
    f = torch.fft.rfft2(x.to(torch.float32))
    f = f.abs() + 1e-6
    f = f.log()
    
    # Center the spectrum
    f = torch.roll(f, shifts=(int(h/2), int(w/2)), dims=(2, 3))
    
    return f


def masked_patches_fourier_loss(
    outputs: torch.Tensor,
    inputs_fourier: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 128
) -> Optional[torch.Tensor]:
    """
    Compute Fourier domain loss on masked regions.
    
    Compares frequency content of predicted vs target images.
    Helps preserve high-frequency details (edges, textures).
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions, shape (B, C, H, W).
    inputs_fourier : torch.Tensor
        Pre-computed Fourier transform of inputs.
    mask : torch.Tensor
        Binary mask (1=visible, 0=masked).
    patch_size : int
        Patch size for reconstruction.
        
    Returns
    -------
    torch.Tensor or None
        Fourier L1 loss, or None if no masked regions.
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    # Reconstruct full image with masked outputs and visible inputs
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs_combined = masked_outputs + unmasked_outputs
    outputs_combined = F.fold(
        outputs_combined, 
        kernel_size=patch_size, 
        stride=patch_size, 
        output_size=(h, w)
    )
    
    # Compare in Fourier domain
    loss = F.l1_loss(fourier_transform_2d(outputs_combined), inputs_fourier)
    
    return loss


def masked_patches_ssim_loss(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    ssim_fn,
    msssim_fn,
    patch_size: int = 128
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute SSIM and MS-SSIM loss on masked regions.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Model predictions.
    inputs : torch.Tensor
        Ground truth images.
    mask : torch.Tensor
        Binary mask.
    ssim_fn : callable
        SSIM computation function.
    msssim_fn : callable
        Multi-scale SSIM computation function.
    patch_size : int
        Patch size.
        
    Returns
    -------
    tuple or None
        (ssim_loss, msssim_loss) or None if no masked regions.
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    # Reconstruct with masked outputs
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs_combined = masked_outputs + unmasked_outputs
    outputs_combined = F.fold(
        outputs_combined,
        kernel_size=patch_size,
        stride=patch_size,
        output_size=(h, w)
    )
    
    # Compute losses (1 - SSIM since higher SSIM is better)
    ssim_loss = 1 - ssim_fn(outputs_combined, inputs)
    msssim_loss = 1 - msssim_fn(outputs_combined, inputs)
    
    return ssim_loss, msssim_loss


def masked_patches_gan_loss(
    outputs: torch.Tensor,
    mask: torch.Tensor,
    discriminator,
    gan_loss_fn,
    patch_size: int = 128
) -> Optional[torch.Tensor]:
    """
    Compute GAN generator loss on masked regions.
    
    The generator tries to fool the discriminator into thinking
    the reconstructed image is real.
    
    Parameters
    ----------
    outputs : torch.Tensor
        Generator outputs.
    mask : torch.Tensor
        Binary mask.
    discriminator : nn.Module
        Discriminator network.
    gan_loss_fn : callable
        GAN loss function (e.g., BCEWithLogitsLoss).
    patch_size : int
        Patch size.
        
    Returns
    -------
    torch.Tensor or None
        Generator loss, or None if no masked regions.
    """
    inv_mask = 1. - mask
    b, c, h, w = outputs.size()
    
    if torch.sum(inv_mask.detach()).item() == 0:
        return None
    
    # Reconstruct image
    f_outputs = F.unfold(outputs, kernel_size=patch_size, stride=patch_size)
    masked_outputs = f_outputs * inv_mask
    unmasked_outputs = (f_outputs * mask).detach()
    outputs_combined = masked_outputs + unmasked_outputs
    outputs_combined = F.fold(
        outputs_combined,
        kernel_size=patch_size,
        stride=patch_size,
        output_size=(h, w)
    )
    
    # Get discriminator output and compute loss
    disc_fake = discriminator(outputs_combined)
    real_labels = torch.ones_like(disc_fake, device=disc_fake.device)
    loss = gan_loss_fn(disc_fake, real_labels)
    
    return loss
