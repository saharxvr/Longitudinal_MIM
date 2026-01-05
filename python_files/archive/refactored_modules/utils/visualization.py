"""
Visualization Utilities
=======================

Functions for creating visualizations, alpha maps, and plotting
difference maps overlaid on chest X-rays.

Usage:
------
    from utils.visualization import create_alpha_map, plot_diff_on_image
    
    diff_map = model_output - baseline
    alpha = create_alpha_map(diff_map)
    plot_diff_on_image(diff_map, current_image, 'output.png')
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple, Union
import scipy.ndimage as ndi


# Colormap for difference visualization (blue=decrease, red=increase)
try:
    differential_grad = plt.cm.get_cmap('RdBu_r')
except:
    differential_grad = plt.colormaps['RdBu_r']


def create_alpha_map(
    x: torch.Tensor,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    min_threshold: float = 0.07,
    squeeze: bool = True
) -> torch.Tensor:
    """
    Generate alpha (transparency) map from difference values.
    
    Creates a transparency map where larger absolute values have
    higher opacity. Used for overlaying difference maps on images.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor (typically a difference map).
        Can be any shape, usually (H, W) or (1, H, W) or (B, 1, H, W).
    min_alpha : float, default=0.0
        Minimum alpha value (fully transparent).
    max_alpha : float, default=1.0
        Maximum alpha value (fully opaque).
    min_threshold : float, default=0.07
        Minimum value for max normalization to avoid division issues.
    squeeze : bool, default=True
        Whether to squeeze singleton dimensions from output.
        
    Returns
    -------
    torch.Tensor
        Alpha map with values in [min_alpha, max_alpha].
        Same spatial shape as input (optionally squeezed).
        
    Examples
    --------
    >>> diff = torch.randn(1, 1, 256, 256) * 0.1
    >>> alpha = create_alpha_map(diff)
    >>> alpha.shape
    torch.Size([256, 256])
    """
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), min_threshold)
    
    # Normalize to [0, 1]
    alpha_map = x_abs / max_val
    
    # Scale to [min_alpha, max_alpha]
    alpha_map = alpha_map * (max_alpha - min_alpha) + min_alpha
    
    if squeeze:
        alpha_map = alpha_map.squeeze()
    
    return alpha_map


def connected_component_analysis(
    binary_mask: np.ndarray,
    min_size: int = 0,
    connectivity: int = 2
) -> Tuple[np.ndarray, int]:
    """
    Perform connected component analysis on binary mask.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask, shape (H, W).
    min_size : int, default=0
        Minimum component size to keep. Components smaller than this
        are removed.
    connectivity : int, default=2
        Connectivity for labeling (1=4-connectivity, 2=8-connectivity).
        
    Returns
    -------
    labeled : np.ndarray
        Labeled array where each connected component has unique integer.
    num_components : int
        Number of connected components found.
        
    Examples
    --------
    >>> mask = np.array([[1,1,0,0], [1,0,0,1], [0,0,1,1]])
    >>> labeled, n = connected_component_analysis(mask)
    >>> n
    3
    """
    struct = ndi.generate_binary_structure(2, connectivity)
    labeled, num_components = ndi.label(binary_mask, structure=struct)
    
    if min_size > 0 and num_components > 0:
        # Remove small components
        component_sizes = ndi.sum(binary_mask, labeled, range(1, num_components + 1))
        small_components = np.where(np.array(component_sizes) < min_size)[0] + 1
        for comp in small_components:
            labeled[labeled == comp] = 0
        # Relabel
        labeled, num_components = ndi.label(labeled > 0, structure=struct)
    
    return labeled, num_components


def plot_diff_on_image(
    diff_map: Union[torch.Tensor, np.ndarray],
    image: Union[torch.Tensor, np.ndarray],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    cmap_image: str = 'gray',
    show: bool = False
) -> Optional[plt.Figure]:
    """
    Plot difference map overlaid on base image.
    
    Creates a visualization where the base image is shown in grayscale
    and the difference map is overlaid with transparency based on
    magnitude. Blue indicates decrease, red indicates increase.
    
    Parameters
    ----------
    diff_map : torch.Tensor or np.ndarray
        Difference map to overlay. Shape (H, W).
    image : torch.Tensor or np.ndarray
        Base image. Shape (H, W).
    output_path : str, optional
        Path to save figure. If None, figure is not saved.
    title : str, optional
        Title for the plot.
    figsize : tuple, default=(8, 8)
        Figure size in inches.
    cmap_image : str, default='gray'
        Colormap for base image.
    show : bool, default=False
        Whether to display the plot.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if show=False, else None.
        
    Examples
    --------
    >>> img = torch.rand(256, 256)
    >>> diff = torch.randn(256, 256) * 0.1
    >>> plot_diff_on_image(diff, img, 'output.png')
    """
    # Convert to numpy if needed
    if isinstance(diff_map, torch.Tensor):
        diff_map = diff_map.detach().cpu().numpy()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Squeeze dimensions
    diff_map = np.squeeze(diff_map)
    image = np.squeeze(image)
    
    # Create alpha map
    diff_tensor = torch.tensor(diff_map)
    alphas = create_alpha_map(diff_tensor).numpy()
    
    # Create normalization centered at 0
    vmin = min(np.min(diff_map), -0.01)
    vmax = max(np.max(diff_map), 0.01)
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap=cmap_image)
    im = ax.imshow(diff_map, alpha=alphas, cmap=differential_grad, norm=divnorm)
    plt.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
    ax.set_axis_off()
    
    if title:
        ax.set_title(title)
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
        plt.close()
        return None
    
    return fig


def plot_comparison(
    images: list,
    titles: Optional[list] = None,
    output_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'gray'
) -> Optional[plt.Figure]:
    """
    Plot multiple images side by side for comparison.
    
    Parameters
    ----------
    images : list
        List of images to plot.
    titles : list, optional
        Titles for each subplot.
    output_path : str, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size. Default based on number of images.
    cmap : str, default='gray'
        Colormap for images.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.squeeze(img)
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    
    fig.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
