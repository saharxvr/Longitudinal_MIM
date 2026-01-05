import torch
import torchvision.transforms as tf
import torch.fft as fft
import torch.nn as nn
import kornia
import numpy as np
from typing import Tuple

import models
from constants import *
from typing import Optional
import torch.nn.functional as F
from skimage import feature
from utils import fourier


class BatchPreprocessing:
    def __init__(self, clip_limit=(1.5, 2.5), grid_size=(8, 8), clahe_prob=1., rand_crop_prob=0.15, use_fourier=USE_FOURIER, use_canny=USE_CANNY) -> None:
        super().__init__()
        self.grid_size = grid_size
        if type(clip_limit) == tuple:
            self.clip_add = clip_limit[0]
            self.clip_mult = clip_limit[1] - clip_limit[0]
            self.get_clip_limit = self.get_range_clip_limit
        else:
            self.clip_limit = clip_limit
            self.get_clip_limit = self.get_const_clip_limit
        self.clahe_prob = clahe_prob
        self.rand_crop_prob = rand_crop_prob
        self.resize = tf.Resize((512, 512))
        self.rand_crop = tf.RandomCrop(360)
        self.use_fourier = use_fourier
        self.use_canny = use_canny

    def get_const_clip_limit(self):
        return self.clip_limit

    def get_range_clip_limit(self):
        return (torch.rand(1) * self.clip_mult + self.clip_add).item()

    def __call__(self, img: torch.Tensor):
        # Normalize to [0, 1]
        img = img.to(torch.float16)
        img = img.view(img.shape[0], 1, -1)
        max_vals = torch.amax(img, dim=2, keepdim=True)
        min_vals = torch.amin(img, dim=2, keepdim=True)
        img = (img - min_vals) / (max_vals - min_vals)
        img = img.view(img.shape[0], 1, IMG_SIZE, IMG_SIZE)

        # Apply CLAHE at set probability
        apply_clahe = torch.rand(1).item() < self.clahe_prob
        if apply_clahe:
            clip_limit = self.get_clip_limit()
            img = kornia.enhance.equalize_clahe(img, clip_limit, self.grid_size)

        apply_rand_crop = torch.rand(1).item() < self.rand_crop_prob
        if apply_rand_crop:
            img = self.rand_crop(img)
            img = self.resize(img)

        if self.use_fourier:
            fouriers = fft.rfft2(img.to(torch.float32))
            return img, fouriers

        return img, None


# TODO: INCLUDE OTHER TRANSFORMS FOR FUTURE RUNS
class BatchPreprocessingImagewise(BatchPreprocessing):
    def __init__(self, clip_limit=(1.5, 2.5), grid_size=(8, 8), clahe_prob=1., rand_crop_prob=0.1, use_fourier=USE_FOURIER, use_canny=USE_CANNY, sigmas=SIGMAS, blur_prob=0.0):
        super(BatchPreprocessingImagewise, self).__init__(clip_limit=clip_limit, grid_size=grid_size, clahe_prob=clahe_prob, rand_crop_prob=rand_crop_prob, use_fourier=use_fourier, use_canny=use_canny)
        self.sigmas = sigmas

        self.gaussian_blur = tf.GaussianBlur(kernel_size=(15, 15), sigma=(1.8, 4.0))
        self.blur_prob = blur_prob

    def __call__(self, img: torch.Tensor):
        img = img.to(DEVICE)

        batch_size = img.shape[0]

        img = img.to(torch.float16)
        img = img.contiguous().view(img.shape[0], 1, -1)
        max_vals = torch.amax(img, dim=2, keepdim=True)
        min_vals = torch.amin(img, dim=2, keepdim=True)
        img = (img - min_vals) / (max_vals - min_vals)
        img = img.view(img.shape[0], 1, IMG_SIZE, IMG_SIZE)

        clahe_mask = torch.rand((batch_size, 1, 1, 1)) > self.clahe_prob
        clahe_mask = clahe_mask.float().to(DEVICE)
        crop_mask = torch.rand((batch_size, 1, 1, 1)) > self.rand_crop_prob
        crop_mask = crop_mask.float().to(DEVICE)
        blur_mask = torch.rand((batch_size, 1, 1, 1)) > self.blur_prob
        blur_mask = blur_mask.float().to(DEVICE)

        clip_limit = self.get_clip_limit()
        clahed_images = kornia.enhance.equalize_clahe(img, clip_limit, self.grid_size) * (1. - clahe_mask)
        img = img * clahe_mask
        img = img + clahed_images

        cropped_images = self.resize(self.rand_crop(img)) * (1. - crop_mask)
        img = img * crop_mask
        img = img + cropped_images

        blurred_images = self.gaussian_blur(img) * (1. - blur_mask)
        img = img * blur_mask
        img = img + blurred_images

        fouriers = None
        cannys = None

        if self.use_canny:
            cannys = []
            for im in img:
                canny_ims = []
                for sigma in self.sigmas:
                    canny_im = torch.tensor(feature.canny(im[0].cpu().numpy() * 256, sigma=sigma))[None, ...]
                    canny_ims.append(canny_im)
                canny_ims = torch.cat(canny_ims, dim=0).to(DEVICE)
                cannys.append(canny_ims)
            cannys = torch.stack(cannys, dim=0)

        if self.use_fourier:
            fouriers = fourier(img.to(torch.float32))

        return img, fouriers, cannys


def generate_masked_images(inputs, mask_probability, patch_size=MASK_PATCH_SIZE, mask_token: Optional[nn.Parameter]=None):
    batch_size, channels, height, width = inputs.size()

    # Unfold the input images into patches
    unfolded = F.unfold(inputs, kernel_size=patch_size, stride=patch_size)

    # Randomly generate mask for each image in the batch
    mask = torch.rand((batch_size, 1, unfolded.shape[2]))
    n_patches = int(height / patch_size)
    bias = torch.cat([torch.linspace(start=1.1, end=0.9, steps=n_patches//2), torch.linspace(start=0.9, end=1.1, steps=n_patches//2)])
    bias = torch.outer(bias, bias)[None, None, ...]
    bias = F.unfold(bias, kernel_size=1, stride=1).repeat(batch_size, 1, 1)
    mask = mask * bias
    mask = mask > mask_probability
    mask = mask.float().to(DEVICE)

    # Apply mask to patches
    masked_patches = unfolded * mask

    if mask_token is not None:
        mask_token_expanded = mask_token.expand(batch_size, patch_size ** 2, unfolded.shape[2]) * (1. - mask)
        masked_patches = masked_patches + mask_token_expanded

    # Fold the masked patches back into images
    masked_inputs = F.fold(masked_patches, output_size=(height, width), kernel_size=patch_size, stride=patch_size)

    return masked_inputs, mask


def generate_single_patch_masked_images(inputs, patch_size=MASK_PATCH_SIZE, mask_token: Optional[nn.Parameter]=None):
    batch_size, channels, height, width = inputs.size()

    # Unfold the input images into patches
    unfolded = F.unfold(inputs, kernel_size=patch_size, stride=patch_size)

    # Randomly generate mask for each image in the batch
    mask = torch.ones((batch_size, 1, unfolded.shape[2]), dtype=torch.float)
    n_patches_dim = int(height / patch_size)
    bias = torch.cat([torch.linspace(start=0.8, end=1.2, steps=n_patches_dim // 2), torch.linspace(start=1.2, end=0.8, steps=n_patches_dim // 2)])
    bias = np.array(torch.outer(bias, bias).flatten())
    bias = bias / bias.sum()
    indices = torch.tensor(np.random.choice(unfolded.shape[2], size=(batch_size, 1, 1), p=bias))
    # indices = torch.randint(0, unfolded.shape[2], (batch_size, 1, 1))
    mask.scatter_(2, indices, 0.)
    mask = mask.to(DEVICE)

    # Apply mask to patches
    masked_patches = unfolded * mask

    if mask_token is not None:
        mask_token_expanded = mask_token.expand(batch_size, patch_size ** 2, unfolded.shape[2]) * (1. - mask)
        masked_patches = masked_patches + mask_token_expanded

    # Fold the masked patches back into images
    masked_inputs = F.fold(masked_patches, output_size=(height, width), kernel_size=patch_size, stride=patch_size)

    return masked_inputs, mask


def generate_checkerboard_masked_images(inputs, patch_size=MASK_PATCH_SIZE, mask_token: Optional[nn.Parameter]=None):
    assert inputs.ndim == 4
    b, c, h, w = inputs.size()

    unfolded = F.unfold(inputs, kernel_size=patch_size, stride=patch_size).unsqueeze(1).transpose(-2, -1)
    spa = int(unfolded.shape[-2] ** 0.5)
    mask_var_1 = torch.eye(2)
    checkerboard_mask_1 = mask_var_1.repeat(spa // 2, spa // 2).flatten().to(unfolded.device)
    checkerboard_mask_1 = checkerboard_mask_1.view(1, 1, unfolded.shape[-2], 1).expand(b, -1, -1, -1)
    checkerboard_mask_2 = 1. - checkerboard_mask_1
    var_mask = (torch.rand((b, 1, 1, 1)) > 0.5).float().to(unfolded.device)
    mask = checkerboard_mask_1 * var_mask + checkerboard_mask_2 * (1. - var_mask)
    masked_patches = unfolded * mask

    if mask_token is not None:
        mask_token_expanded = mask_token.unsqueeze(0).expand(b, -1, -1, -1).transpose(-2, -1)
        masked_patches = masked_patches + mask_token_expanded * (1. - mask)

    masked_patches = masked_patches.transpose(-2, -1).squeeze(1)
    masked_inputs = F.fold(masked_patches, kernel_size=patch_size, stride=patch_size, output_size=(h, w))
    return masked_inputs, mask


def generate_masked_variations(inputs: torch.Tensor, patch_size: int=MASK_PATCH_SIZE, mask_type: Optional[str]=None, canny_im: Optional[torch.Tensor]=None, mask_token: Optional[nn.Parameter]=None):
    assert inputs.ndim == 4
    b, c, h, w = inputs.size()

    unfolded = F.unfold(inputs, kernel_size=patch_size, stride=patch_size).unsqueeze(1).transpose(-2, -1)
    if mask_type is None:
        vars_num = unfolded.shape[-2]
        mask_variations = (~torch.eye(vars_num, dtype=torch.bool, device=unfolded.device)).float()
    elif mask_type == 'checkerboard':
        vars_num = 2
        spa = int(unfolded.shape[-2] ** 0.5)
        mask_var1 = torch.eye(2).repeat(spa // 2, spa // 2).flatten()
        mask_var2 = (1. - torch.eye(2)).repeat(spa // 2, spa // 2).flatten()
        mask_variations = torch.stack([mask_var1, mask_var2]).to(unfolded.device)
    else:
        raise NotImplementedError()
    mask_variations = mask_variations.view(1, mask_variations.shape[0], mask_variations.shape[1], 1)
    variations = unfolded * mask_variations

    if mask_token is not None:
        spatial = int(mask_token.shape[1] ** 0.5)
        to_repeat = int(patch_size / spatial)
        mask_token = mask_token.view(spatial, spatial)
        assert to_repeat >= 1
        if to_repeat > 1:
            mask_token = mask_token.repeat((to_repeat, to_repeat))
        mask_token = mask_token.view(1, 1, 1, -1)
        variations = variations + (mask_token * (1 - mask_variations))

    variations = variations.transpose(-2, -1)

    if canny_im is not None:
        variations_arr = [torch.cat([F.fold(var, output_size=(h, w), kernel_size=patch_size, stride=patch_size), canny_im[i].unsqueeze(0).expand(vars_num, -1, -1, -1)], dim=1) for i, var in enumerate(variations)]
    else:
        variations_arr = [F.fold(var, output_size=(h, w), kernel_size=patch_size, stride=patch_size) for var in variations]

    # if canny_im is not None:
    #     canny_im = canny_im.unsqueeze(1)
    #     canny_im = canny_im.expand(-1, variations_arr[0].shape[0], -1, -1, -1)
    #     variations_arr = [torch.cat([var, canny_im], dim=1) for var in variations_arr]

    # variations = torch.stack(variations_arr)

    return variations_arr


def combine_variations_predictions(outputs: torch.Tensor, mask_type: Optional[str]=None, patch_size: int = MASK_PATCH_SIZE):
    assert outputs.ndim == 4
    b, c, h, w = outputs.size()
    # assert b == (h * w) / (patch_size ** 2)

    unfolded = F.unfold(outputs, kernel_size=patch_size, stride=patch_size).transpose(-2, -1)
    if mask_type is None:
        mask_variations = torch.eye(b, dtype=torch.bool, device=unfolded.device)
        mask_variations = mask_variations.view(mask_variations.shape[0], mask_variations.shape[1], 1)
        combined = torch.masked_select(unfolded, mask_variations).view(1, unfolded.shape[1], unfolded.shape[-1])
        combined = combined.transpose(-2, -1)
        combined = F.fold(combined, output_size=(h, w), kernel_size=patch_size, stride=patch_size)
        return combined
    elif mask_type == 'checkerboard':
        spa = int(unfolded.shape[-2] ** 0.5)
        mask_var1 = (1. - torch.eye(2)).repeat(spa // 2, spa // 2).flatten()
        mask_var2 = torch.eye(2).repeat(spa // 2, spa // 2).flatten()
        mask_variations = torch.stack([mask_var1, mask_var2]).to(unfolded.device).bool()
        mask_variations = mask_variations.view(mask_variations.shape[0], mask_variations.shape[1], 1)
        unfolded = unfolded * mask_variations
        combined = (unfolded[0] + unfolded[1])[None, ...]
        combined = combined.transpose(-2, -1)
        combined = F.fold(combined, output_size=(h, w), kernel_size=patch_size, stride=patch_size)
        return combined


def dedisorder_batch(inputs: torch.tensor, cannys: torch.tensor, mim_model: models.MaskedReconstructionModel, patch_size=MASK_PATCH_SIZE):
    mask_token = mim_model.get_mask_token()
    mask_vars = generate_masked_variations(inputs, patch_size=patch_size, mask_type=None, canny_im=cannys, mask_token=mask_token)
    batch_cat = []
    for var in mask_vars:
        var_cat = []
        for var_chunk in torch.split(var, 8):
            out_chunk = mim_model(var_chunk)
            var_cat.append(out_chunk)
        var = torch.cat(var_cat, dim=0)
        combined_var = combine_variations_predictions(var, mask_type=None, patch_size=patch_size)
        batch_cat.append(combined_var)
    dedisordered_batch = torch.cat(batch_cat, dim=0)
    return dedisordered_batch


# import nibabel as nib
# import kornia
# # import torch
# import matplotlib.pyplot as plt
# # pre = BatchPreprocessingImagewise(use_fourier=False, use_canny=False)
# img = torch.tensor(nib.load('00011101_004.nii.gz').get_fdata()).T[None, None, ...]
# img = img.to(torch.float16)
# img = img.view(img.shape[0], 1, -1)
# max_vals = torch.amax(img, dim=2, keepdim=True)
# min_vals = torch.amin(img, dim=2, keepdim=True)
# img = (img - min_vals) / (max_vals - min_vals)
# img = img.view(img.shape[0], 1, IMG_SIZE, IMG_SIZE)
# img = kornia.enhance.equalize_clahe(img, 5., (8, 8))
# plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
# plt.show()
# im2 = torch.tensor(nib.load('remove_cxr14/00000061_009.nii.gz').get_fdata()).T[None, None, ...]
# im = torch.cat([im1, im2])
# mask_t = nn.Parameter(torch.linspace(0, 128, steps=128*128, dtype=torch.float).view(1, 128*128, 1))
# im_mask = generate_checkerboard_masked_images(im, patch_size=128, mask_token=None)
#
# fig, ax = plt.subplots(2)
# ax[0].imshow(im_mask[0][0].detach().numpy(), cmap='gray')
# ax[1].imshow(im_mask[1][0].detach().numpy(), cmap='gray')
# plt.show()
