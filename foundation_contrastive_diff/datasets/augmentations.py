"""Image-only nuisance augmentation; signed pathology labels stay unchanged."""

import torch


PHOTOMETRIC_RECIPE = {
    "probability": 0.8,
    "brightness_delta": 0.05,
    "contrast_range": [0.9, 1.1],
    "gamma_range": [0.9, 1.1],
    "noise_std_max": 0.01,
}


@torch.no_grad()
def augment_cxr(images):
    """Independently perturb each [B,1,H,W] image without moving any pixels."""
    if images.ndim != 4 or images.shape[1] != 1 or not images.is_floating_point():
        raise ValueError("Photometric augmentation requires floating [B,1,H,W] images")
    if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError("Photometric augmentation requires finite images in [0,1]")
    shape = (images.shape[0], 1, 1, 1)

    def uniform(low, high):
        return torch.empty(shape, device=images.device, dtype=images.dtype).uniform_(low, high)

    apply = uniform(0, 1) < PHOTOMETRIC_RECIPE["probability"]
    delta = PHOTOMETRIC_RECIPE["brightness_delta"]
    brightness = uniform(-delta, delta)
    contrast = uniform(*PHOTOMETRIC_RECIPE["contrast_range"])
    gamma = uniform(*PHOTOMETRIC_RECIPE["gamma_range"])
    noise_std = uniform(0, PHOTOMETRIC_RECIPE["noise_std_max"])
    mean = images.mean(dim=(-2, -1), keepdim=True)
    augmented = ((images - mean) * contrast + mean + brightness).clamp(0, 1).pow(gamma)
    augmented = (augmented + torch.randn_like(images) * noise_std).clamp(0, 1)
    return torch.where(apply, augmented, images)
