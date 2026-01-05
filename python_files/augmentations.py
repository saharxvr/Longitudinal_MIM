"""
Data augmentation transforms for Longitudinal CXR Analysis.

This module provides augmentation transforms for training, including:
- Geometric transforms (B-spline, affine, rotation, scaling)
- Intensity transforms (CLAHE, color jitter)
- Paired transforms (for baseline-followup pairs)
- Synthetic abnormality generation

Main Transform Classes:
-----------------------
- RandomFlipBLWithFU: Randomly swap baseline/followup images
- RandomChannelsFlip: Flip channel order
- CropResizeWithMaskTransform: Crop to lung region and resize
- RandomScaleTranslationTransform: Random scale and translation
- RandomBsplineAndSimilarityWithMaskTransform: B-spline deformation
- RandomAffineWithMaskTransform: Random affine transformation
- RescaleValuesTransform: Normalize to [0, 1]
- RandomIntensityTransform: CLAHE and color jitter
- PairwiseRandomIntensityTransform: Paired intensity augmentation
- RandomAbnormalizationTransform: Synthetic abnormality overlay

Used by:
    - datasets.py (LongitudinalMIMDataset)
"""

import random

import torch
import gryds
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2 as v2
from constants import *
import kornia
from skimage import feature
from utils import get_sep_lung_masks
from scipy.ndimage import distance_transform_edt


# =============================================================================
# MASK UTILITIES
# =============================================================================

def get_bounding_rect(mask: torch.tensor):
    # Order of return: y_min, x_min, y_max, x_max
    coord_bounds_to_keep = torch.argwhere(mask == 1.)
    min_c = torch.amin(coord_bounds_to_keep, dim=0)
    max_c = torch.amax(coord_bounds_to_keep, dim=0)
    return min_c[-2:], max_c[-2:]


def get_mask_center_prop(mask: torch.tensor):
    min_c, max_c = get_bounding_rect(mask)
    return 0.5 * (min_c + max_c) / mask.shape[-1]


def get_mask_center_coord(mask: torch.tensor):
    min_c, max_c = get_bounding_rect(mask)
    return (min_c + max_c) // 2


def left_lower_ver(grid_y, grid_x, biases):
    next_ops = {right_lower_ver, left_hor, right_hor, upper_ver_sym, upper_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.35
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[grid_y_add.shape[0] // 2:, grid_y_add.shape[1] // 2:] = val
    grid_y_add = grid_y_add * biases[0]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def right_lower_ver(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, left_hor, right_hor, upper_ver_sym, upper_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.35
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[grid_y_add.shape[0] // 2:, :grid_y_add.shape[1] // 2] = val
    grid_y_add = grid_y_add * biases[0]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def left_hor(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, right_lower_ver, right_hor, upper_ver_sym, upper_ver_ver, lower_ver_sym, lower_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.18
    grid_x_add = torch.zeros_like(grid_x)
    grid_x_add[:, grid_x_add.shape[1] // 2:] = val
    grid_x_add = grid_x_add * biases[2]
    grid_x += grid_x_add
    return grid_y, grid_x, next_ops


def right_hor(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, right_lower_ver, left_hor, upper_ver_sym, upper_ver_ver, lower_ver_sym, lower_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.18
    grid_x_add = torch.zeros_like(grid_x)
    grid_x_add[:, :grid_x_add.shape[1] // 2] = val
    grid_x_add = grid_x_add * biases[2]
    grid_x += grid_x_add
    return grid_y, grid_x, next_ops


def upper_ver_sym(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, right_lower_ver, left_hor, right_hor, lower_ver_sym, lower_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.2
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[:grid_y_add.shape[0] // 2, :] = val
    grid_y_add = grid_y_add * biases[0]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def lower_ver_sym(grid_y, grid_x, biases):
    next_ops = {left_hor, right_hor, upper_ver_sym, upper_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.2
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[grid_y_add.shape[0] // 2:, :] = val
    grid_y_add = grid_y_add * biases[0]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def upper_ver_ver(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, right_lower_ver, left_hor, right_hor, lower_ver_sym, lower_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.16
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[:grid_y_add.shape[0] // 2, :] = val
    grid_y_add = grid_y_add * biases[1]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def lower_ver_ver(grid_y, grid_x, biases):
    next_ops = {left_hor, right_hor, upper_ver_sym, upper_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.16
    grid_y_add = torch.zeros_like(grid_y)
    grid_y_add[grid_y_add.shape[0] // 2:, :] = val
    grid_y_add = grid_y_add * biases[1]
    grid_y += grid_y_add
    return grid_y, grid_x, next_ops


def left_lower_upper_opposite_hor(grid_y, grid_x, biases):
    next_ops = {right_lower_ver, right_hor, right_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.3
    grid_x_add = torch.zeros_like(grid_x)
    grid_x_add[:grid_x_add.shape[0] // 2, grid_x_add.shape[1] // 2:] = val
    grid_x_add[grid_x_add.shape[0] // 2:, grid_x_add.shape[1] // 2:] = -val
    grid_x_add = grid_x_add * biases[0]
    grid_x += grid_x_add
    return grid_y, grid_x, next_ops


def right_lower_upper_opposite_hor(grid_y, grid_x, biases):
    next_ops = {left_lower_ver, left_hor, left_lower_upper_opposite_hor, finish}
    val = (random.random() - 0.5) * 0.3
    grid_x_add = torch.zeros_like(grid_x)
    grid_x_add[:grid_x_add.shape[0] // 2, :grid_x_add.shape[1] // 2] = val
    grid_x_add[grid_x_add.shape[0] // 2:, :grid_x_add.shape[1] // 2] = -val
    grid_x_add = grid_x_add * biases[0]
    grid_x += grid_x_add
    return grid_y, grid_x, next_ops


def finish(grid_y, grid_x, biases):
    return grid_y, grid_x, set()


def get_bspline_tf_of_random_composition(shape: tuple) -> gryds.BSplineTransformation:
    grid_y = torch.zeros(shape)
    grid_x = torch.zeros(shape)

    ops_probs = {finish: 0.2,
                 left_lower_ver: 0.1, right_lower_ver: 0.1,
                 left_hor: 0.1, right_hor: 0.1,
                 upper_ver_sym: 0.05, upper_ver_ver: 0.05, lower_ver_sym: 0.05, lower_ver_ver: 0.05,
                 left_lower_upper_opposite_hor: 0.1, right_lower_upper_opposite_hor: 0.1}
    pos_grid_ops = {finish, left_lower_ver, right_lower_ver, left_hor, right_hor, upper_ver_sym, upper_ver_ver, lower_ver_sym, lower_ver_ver, left_lower_upper_opposite_hor, right_lower_upper_opposite_hor}

    bias = torch.cat([torch.linspace(start=1.0, end=0.0, steps=grid_y.shape[0] // 2), torch.linspace(start=0.0, end=1.0, steps=grid_y.shape[0] // 2)])[None, ...]
    bias_sym = bias.T @ bias
    bias_hor = bias.repeat((bias.shape[0], 1))
    bias_ver = bias.T.repeat((1, bias.shape[1]))
    biases = [bias_sym, bias_ver, bias_hor]

    while pos_grid_ops:
        ops_list = list(pos_grid_ops)
        probs_list = np.array([ops_probs[c_op] for c_op in ops_list])
        probs_list = probs_list / np.sum(probs_list)
        op = np.random.choice(ops_list, size=None, p=probs_list)
        grid_y, grid_x, new_ops = op(grid_y, grid_x, biases)
        pos_grid_ops.intersection_update(new_ops)

    bspline = gryds.BSplineTransformation([grid_y, grid_x], order=1)
    return bspline


# =============================================================================
# PAIRED IMAGE TRANSFORMS
# =============================================================================

class RandomFlipBLWithFU:
    """Randomly swap baseline and followup images (with their masks)."""
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, bl, fu, bl_mask=None, fu_mask=None):
        if random.random() < self.p:
            return fu, bl, fu_mask, bl_mask
        return bl, fu, bl_mask, fu_mask


class RandomChannelsFlip:
    """Randomly flip channel order for both baseline and followup."""
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, bl, fu):
        if random.random() < self.p:
            bl = torch.flip(bl, dims=(0,))
            fu = torch.flip(fu, dims=(0,))
        return bl, fu


# =============================================================================
# GEOMETRIC TRANSFORMS
# =============================================================================

class CropResizeWithMaskTransform:
    # def __init__(self, min_sub: int = 30, max_add: torch.tensor = torch.tensor([60, 30])):
    def __init__(self):
        self.resize = v2.Resize((IMG_SIZE, IMG_SIZE))
        self.add_max = torch.tensor([50, 20])

    def __call__(self, img: torch.tensor):
        mask = torch.round(img[-1])
        # img = img[0:-1]
        min_c, max_c = get_bounding_rect(mask)
        min_c = (min_c - random.randint(10, 45)).clamp_min(0)
        max_c = (max_c + torch.randint(-20, 16, size=(2,)) + self.add_max).clamp_max(img.shape[-1])
        cropped_img = img[:, min_c[0]: max_c[0], min_c[1]: max_c[1]]
        resized_img = self.resize(cropped_img)

        mask = torch.round(resized_img[-1:])
        resized_img = resized_img[0:-1]

        return resized_img, mask


class RandomScaleTranslationTransform:
    def __init__(self, scale_y_p=0.35, scale_x_p=0.35, trans_y_p=0.35, trans_x_p=0.35, scale_mult=0.3, transl_mult=0.12):
        self.scale_y_p = scale_y_p
        self.scale_x_p = scale_x_p
        self.trans_y_p = trans_y_p
        self.trans_x_p = trans_x_p

        self.scale_mult = scale_mult
        self.transl_mult = transl_mult

    def __call__(self, def_img):
        scale = [1., 1.]
        scale_y = ((random.random() - 0.5) * self.scale_mult) * (random.random() < self.scale_y_p)
        scale[0] += scale_y
        scale_x = ((random.random() - 0.5) * self.scale_mult) * (random.random() < self.scale_x_p)
        scale[1] += scale_x

        transl = [0., 0.]
        transl_y = ((random.random() - 0.5) * self.transl_mult) * (random.random() < self.trans_y_p)
        transl[0] = transl_y
        transl_x = ((random.random() - 0.5) * self.transl_mult) * (random.random() < self.trans_x_p)
        transl[1] = transl_x

        if scale != [1., 1.] or transl != [0., 0.]:
            interpolator = gryds.MultiChannelInterpolator(def_img, data_format='channels_first', mode='nearest')
            scale_transl_tf = gryds.AffineTransformation(ndim=2, translation=transl, scaling=scale)
            def_img = torch.tensor(interpolator.transform(scale_transl_tf))

        return def_img


class RandomBsplineAndSimilarityWithMaskTransform:
    def __init__(self, rot_p=0.5, scale_y_p=0.35, scale_x_p=0.35, trans_x_p=0.35, trans_y_p=0.35):
        self.rot_p = rot_p
        self.scale_transl_tf = RandomScaleTranslationTransform(scale_y_p=scale_y_p, scale_x_p=scale_x_p, trans_y_p=trans_y_p, trans_x_p=trans_x_p)
        self.crop_resize = CropResizeWithMaskTransform()

    def __call__(self, img):
        interpolator = gryds.MultiChannelInterpolator(img, data_format='channels_first', mode='nearest')
        bspline_tf = get_bspline_tf_of_random_composition((img.shape[-2] // 8, img.shape[-1] // 8))
        tfs = [bspline_tf]

        if random.random() < self.rot_p:
            mask = img[-1]
            mask_cent = get_mask_center_prop(mask)
            angle = (random.random() - 0.5) * torch.pi / 5
            rot_tf = gryds.AffineTransformation(ndim=2, center=mask_cent, angles=(angle,))
            tfs.append(rot_tf)

        def_img = torch.tensor(interpolator.transform(*tfs))
        def_mask = def_img[-1]
        def_img = self.scale_transl_tf(def_img)
        def_img = torch.cat([def_img, def_mask.unsqueeze(0)], dim=0)
        # def_img[-1] = def_mask
        # cropped_resized_img, mask = self.crop_resize(def_img)
        cropped_resized_img, _ = self.crop_resize(def_img)
        mask = torch.round(cropped_resized_img[-1:])
        cropped_resized_img = cropped_resized_img[0: -1]

        return cropped_resized_img, mask


class RandomAffineWithMaskTransform:
    def __init__(self, scale_y_p=0.4, scale_x_p=0.4, trans_x_p=0.4, trans_y_p=0.4):
        # self.random_perspective = v2.RandomPerspective(distortion_scale=0.35, p=0.9)
        self.random_affine = v2.RandomAffine(degrees=30, shear=15)
        self.scale_transl_tf = RandomScaleTranslationTransform(scale_y_p=scale_y_p, scale_x_p=scale_x_p, trans_y_p=trans_y_p, trans_x_p=trans_x_p,
                                                               scale_mult=0.15, transl_mult=0.05)
        # self.crop_resize = CropResizeWithMaskTransform(10, torch.tensor([20, 10]))
        self.crop_resize = CropResizeWithMaskTransform()

    def __call__(self, img):
        # def_img = self.random_perspective(img)
        def_img = self.random_affine(img)
        def_mask = def_img[-1]
        def_img = self.scale_transl_tf(def_img)
        def_img = torch.cat([def_img, def_mask.unsqueeze(0)], dim=0)
        # def_img[-1] = def_mask
        # cropped_resized_img, mask = self.crop_resize(def_img)
        cropped_resized_img, _ = self.crop_resize(def_img)
        mask = torch.round(cropped_resized_img[-1:])
        cropped_resized_img = cropped_resized_img[0: -1]

        return cropped_resized_img, mask


class RescaleValuesTransform:
    def __call__(self, img):
        channels, height, width = img.size()

        img = img.to(torch.float)
        img = img.reshape(channels, -1)
        max_vals = torch.amax(img, dim=-1, keepdim=True)
        min_vals = torch.amin(img, dim=-1, keepdim=True)
        img = (img - min_vals) / (max_vals - min_vals)
        img = img.reshape(channels, height, width)

        return img


class RandomIntensityTransform:
    def __init__(self, clahe_p=0.5, clahe_clip_limit=(0.75, 2.5), blur_p=0.0, jitter_p=0.4):
        self.clahe_p = clahe_p
        if type(clahe_clip_limit) == tuple:
            self.clip_add = clahe_clip_limit[0]
            self.clip_mult = clahe_clip_limit[1] - clahe_clip_limit[0]
            self.get_clip_limit = self.get_range_clip_limit
        else:
            self.clahe_clip_limit = clahe_clip_limit
            self.get_clip_limit = self.get_const_clip_limit

        self.jitter_p = jitter_p
        self.color_jitter = v2.ColorJitter(brightness=0.12, contrast=0.1, saturation=None, hue=None)
        # self.gaussian_blur = v2.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 1.0))
        # self.blur_p = blur_p

    def get_const_clip_limit(self):
        return self.clahe_clip_limit

    def get_range_clip_limit(self):
        return random.random() * self.clip_mult + self.clip_add

    def __call__(self, img):
        if random.random() < self.jitter_p:
            img = self.color_jitter(img)

        if random.random() < self.clahe_p:
            img = kornia.enhance.equalize_clahe(img, clip_limit=self.get_clip_limit(), grid_size=(8, 8))

        # if random.random() < self.blur_p:
        #     img = self.gaussian_blur(img)

        return img


class PairwiseRandomIntensityTransform:
    def __init__(self, clahe_p=0.625, clahe_range=(0.75, 2.5), clahe_var=0.15):
        self.clahe_p = clahe_p
        self.clahe_val_mult = clahe_range[1] - clahe_range[0]
        self.clahe_val_add = clahe_range[0]
        self.clahe_var_mult = clahe_var * 2.
        self.clahe_var_add = - clahe_var

    def __call__(self, img1, img2):
        if random.random() < self.clahe_p:
            clip_limit = random.random() * self.clahe_val_mult + self.clahe_val_add
            clip_dif = random.random() * self.clahe_var_mult + self.clahe_var_add
            img1 = kornia.enhance.equalize_clahe(img1, clip_limit=clip_limit, grid_size=(8, 8))
            img2 = kornia.enhance.equalize_clahe(img2, clip_limit=clip_limit + clip_dif, grid_size=(8, 8))

        return img1, img2


class RandomAbnormalizationTransform:
    def __init__(self, lung_abnormalities=True, devices=True, size=768, none_chance_to_update=0.13):
        self.resize = v2.Resize((size, size))
        self.deform_resize = v2.Resize((size // 8, size // 8))
        resize_bicubic = v2.Resize((size, size), interpolation=v2.InterpolationMode.BICUBIC)
        resize_nearest = v2.Resize((size, size), interpolation=v2.InterpolationMode.NEAREST)
        self.device_resizes = [resize_bicubic, resize_nearest]
        self.perspective = v2.RandomPerspective(distortion_scale=0.6)

        self.abnormalities_general = []
        self.abnormalities_local = []

        if lung_abnormalities:
            self.abnormalities_general.extend([self.masses, self.lesions, self.general_opacity, self.disordered_opacity, self.contour_deformation])
            self.abnormalities_local.extend([self.masses, self.lesions, self.general_opacity, self.disordered_opacity, self.contour_deformation])
        if devices:
            self.abnormalities_general.extend([self.devices])
            self.abnormalities_local.extend([self.devices])

        self.is_lung_abnormalities = lung_abnormalities
        self.is_devices = devices

        # self.abnormalities_general = [self.effusion_opacity2]
        # self.abnormalities_local = [self.effusion_opacity2]

        # if lung_abnormalities and devices:
        #     self.update_on_devices = False
        # else:
        #     self.update_on_devices = True

        self.none_chance_to_update = none_chance_to_update

        self.base_map = torch.arange(size).repeat(size, 1).T
        self.interp_range = np.arange(768)

    def __call__(self, img, mask):
        # TODO: TO REMOVE
        if len(self.abnormalities_general) == 0:
            return img

        mask_center = get_mask_center_coord(mask)

        mask_center_0 = mask_center[0].item()
        mask_center_1 = mask_center[1].item()
        top_right = [(0, mask_center_0), (0, mask_center_1)]
        top_left = [(0, mask_center_0), (mask_center_1, img.shape[-1])]
        bottom_right = [(mask_center_0, img.shape[-2]), (0, mask_center_1)]
        bottom_left = [(mask_center_0, img.shape[-2]), (mask_center_1, img.shape[-1])]
        right = [(0, img.shape[-2]), (0, mask_center_1)]
        left = [(0, img.shape[-2]), (mask_center_1, img.shape[-1])]
        both = [(0, img.shape[-2]), (0, img.shape[-1])]
        none = None

        regions_to_attributes_dict = {
            'top_left': [top_left, {'bottom_left', 'top_right', 'bottom_right', 'right', 'none'}, 0.1125, self.abnormalities_local],
            'top_right': [top_right, {'bottom_right', 'top_left', 'bottom_left', 'left', 'none'}, 0.1125, self.abnormalities_local],
            'bottom_left': [bottom_left, {'top_left', 'top_right', 'bottom_right', 'right', 'none'}, 0.1125, self.abnormalities_general],
            'bottom_right': [bottom_right, {'top_right', 'top_left', 'bottom_left', 'left', 'none'}, 0.1125, self.abnormalities_general],
            'left': [left, {'top_right', 'bottom_right', 'right', 'none'}, 0.15, self.abnormalities_general],
            'right': [right, {'top_left', 'bottom_left', 'left', 'none'}, 0.15, self.abnormalities_general],
            'both': [both, set(), 0.15, self.abnormalities_general],
            # 'none': [none, set(), 0.075, self.abnormalities_general],
            'none': [none, set(), 0.09, self.abnormalities_general],
        }
        all_regions = {'top_left', 'top_right', 'bottom_left', 'bottom_right', 'left', 'right', 'both'}
        cur_pos_regions = {'top_left', 'top_right', 'bottom_left', 'bottom_right', 'left', 'right', 'both', 'none'}

        if self.is_lung_abnormalities:
            if random.random() < 0.9:
                modifications_probs_dict = {
                    self.masses: 1.,
                    self.lesions: 0.75,
                    self.general_opacity: 1.,
                    self.disordered_opacity: 1.,
                    self.contour_deformation: 0.3
                }
                if self.devices in self.abnormalities_general:
                    modifications_probs_dict[self.devices] = 0.175
                ab_decay_fac = 1.
                # reg_decay_fac = 0.25
                # TODO: REPLACE
                # reg_decay_fac = 0.075
                reg_decay_fac = 0.9
            else:
                modifications_probs_dict = {
                    self.masses: 0.05,
                    self.lesions: 0.05,
                    self.general_opacity: 1.,
                    self.disordered_opacity: 0.05,
                    self.contour_deformation: 0.075
                }
                if self.devices in self.abnormalities_general:
                    modifications_probs_dict[self.devices] = 0.03
                ab_decay_fac = 0.9
                reg_decay_fac = 0.75
        elif self.is_devices:
            modifications_probs_dict = {self.devices: 1.}
            #TODO: REPLACE
            # ab_decay_fac = 0.65
            ab_decay_fac = 1.
            reg_decay_fac = 0.75
        else:
            raise NotImplementedError()

        #TODO: TO REMOVE
        modifications_probs_dict = {k: modifications_probs_dict[k] for k in modifications_probs_dict.keys() if k in self.abnormalities_general}

        abs_list = list(modifications_probs_dict)

        while True:
            regions_list = list(cur_pos_regions)
            probs_list = np.array([regions_to_attributes_dict[c_reg_name][2] for c_reg_name in regions_list])
            probs_list = probs_list / np.sum(probs_list)
            reg_name = np.random.choice(regions_list, size=None, p=probs_list)
            if reg_name == 'none':
                break
            reg = regions_to_attributes_dict[reg_name][0]

            ab_probs_list = np.array([modifications_probs_dict[ab_name] for ab_name in abs_list])
            ab_probs_list = ab_probs_list / np.sum(ab_probs_list)
            abnormality = np.random.choice(abs_list, size=None, p=ab_probs_list)

            # abnormality = random.choice(regions_to_attributes_dict[reg_name][3])
            old_img = img.clone()
            img = abnormality(img, mask, reg, mask_center)
            img = torch.maximum(img, old_img)

            # if self.update_on_devices or abnormality.__name__ != 'devices' or reg_name == 'none':
            #     cur_pos_regions.intersection_update(regions_to_attributes_dict[reg_name][1])
            # regions_to_attributes_dict[reg_name][2] *= 0.4
            # regions_to_attributes_dict['none'][2] = self.none_chance_to_update
            # if abnormality.__name__ == 'general_opacity' and c == 4:
            #     modifications_probs_dict[self.general_opacity] = 0.
            #     modifications_probs_dict[self.contour_deformation] = 1.
            if abnormality.__name__ == 'contour_deformation':
                if reg_name in {'top_left', 'bottom_left', 'left'}:
                    cur_pos_regions.intersection_update({'top_right', 'bottom_right', 'right', 'none'})
                elif reg_name in {'top_right', 'bottom_right', 'right'}:
                    cur_pos_regions.intersection_update({'top_left', 'bottom_left', 'left', 'none'})
            else:
                for r in all_regions:
                    if r not in regions_to_attributes_dict[reg_name][1]:
                        regions_to_attributes_dict[reg_name][2] *= reg_decay_fac
            modifications_probs_dict[abnormality] *= ab_decay_fac
            # regions_to_attributes_dict['none'][2] += 0.11
            regions_to_attributes_dict['none'][2] += self.none_chance_to_update

        img = torch.clip(img, min=0, max=255)
        return img

    def masses(self, img, mask, region, _):
        if not region:
            return img

        opacity = torch.zeros_like(img)
        # TODO: REMOVE PARAMS
        # Area + Power
        ab_power = random.random() * 24. + 8.
        # ab_power = self.ab_power
        opacity[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = ab_power

        # Erosion of mask
        # erod_mask = -torch.max_pool2d(-mask, kernel_size=11, stride=1, padding=5)
        # opacity = opacity * erod_mask
        opacity = opacity * mask

        # Softening of edges
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=11, stride=1, padding=5)

        # adding noise
        frequency_scale = random.random() * 32 + 32
        noise = (torch.rand(1, int(img.shape[-2] / frequency_scale), int(img.shape[-1] / frequency_scale)) * 1.2) ** 5
        # noise = self.noise
        noise = self.resize(torch.round(noise))
        inv_size = random.random() * 0.4 + 0.45
        # inv_size = self.inv_size
        noise[noise < inv_size] = 0.
        opacity *= noise

        blur_fac = random.randint(0, 5) * 2 + 5
        # blur_fac = self.blur_fac
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)

        opacity *= mask

        img = img + opacity
        return img

    def lesions(self, img, mask, region, _):
        if not region:
            return img

        opacity = torch.zeros_like(img)

        # Area + Power
        ab_power = random.random() * 25. + 35.
        opacity[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = ab_power

        # Erosion of mask
        erod_mask = -torch.max_pool2d(-mask, kernel_size=11, stride=1, padding=5)
        opacity = opacity * erod_mask

        # Softening of edges
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=11, stride=1, padding=5)

        # adding noise
        frequency_scale = random.random() * 6. + 6.
        sparsity_exp = random.random() * 7. + 18.
        sparsity_mult = random.random() * 0.04 + 0.96
        noise = (torch.rand(1, int(img.shape[-2] / frequency_scale), int(img.shape[-1] / frequency_scale)) * sparsity_mult) ** sparsity_exp
        noise = self.resize(torch.round(noise))
        inv_size = random.random() * 0.5 + 0.5
        noise[noise > inv_size] = 1.
        noise[noise < 0.2] = 0.
        opacity *= noise
        blur_fac = random.randint(0, 4) * 2 + 5
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
        opacity *= mask

        img = img + opacity
        return img

    def general_opacity(self, img, mask, region, mask_center):
        if not region:
            return img

        def deform(opac, msk):
            x_freq = random.random() * 45 + 15
            y_freq = random.random() * 45 + 15
            x_deform_intensity = random.random() * 0.09 + 0.07
            y_deform_intensity = random.random() * 0.09 + 0.07
            grid_x = self.deform_resize((torch.rand(1, int(opac.shape[-2] / x_freq), int(opac.shape[-1] / x_freq)) - 0.5) * x_deform_intensity).squeeze(0)
            grid_y = self.deform_resize((torch.rand(1, int(opac.shape[-2] / y_freq), int(opac.shape[-1] / y_freq)) - 0.5) * y_deform_intensity).squeeze(0)
            bspline = gryds.BSplineTransformation([grid_y, grid_x], order=1)
            interpolator = gryds.Interpolator(opac.squeeze(0))
            opac = torch.tensor(interpolator.transform(bspline)).unsqueeze(0)
            # blr_fac = random.randint(0, 5) * 2 + 11
            # opac = torch.nn.functional.avg_pool2d(opac, kernel_size=blr_fac, stride=1, padding=blr_fac // 2)
            opac *= msk
            return opac

        cent_mult = random.random() * 0.15 + 0.875
        center_val = img[0, mask_center[0], mask_center[1]] * cent_mult

        opacity = torch.zeros_like(img).float()

        # Area + Power
        opacity[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = center_val

        # Dilation of mask
        dial_mask = torch.max_pool2d(mask, kernel_size=9, stride=1, padding=4)

        op_mask = (dial_mask * opacity).bool()
        #
        # op_mask = (mask * opacity).bool()

        # adding noise
        frequency_scale = random.random() * 90 + 60
        # frequency_scale = 150
        noise = torch.rand(1, int(img.shape[-2] / frequency_scale), int(img.shape[-1] / frequency_scale))
        noise = self.resize(torch.round(noise))

        if random.random() < 0.6:
            mult_frequency_scale1 = random.random() * 20 + 50
            mult_frequency_scale2 = random.random() * 20 + 50
            mult_noise1 = torch.rand(1, int(img.shape[-2] / mult_frequency_scale1), int(img.shape[-1] / mult_frequency_scale1))
            mult_noise2 = torch.rand(1, int(img.shape[-2] / mult_frequency_scale2), int(img.shape[-1] / mult_frequency_scale2))
            # mult_noise = mult_noise * 1. + 0.2
            # mult_noise = - 0.8 * (mult_noise ** 2) + 2 * mult_noise + 0.1
            mult_noise1 = mult_noise1 + 0.175
            mult_noise2 = mult_noise2 + 0.175
            mult_noise1 = self.resize(mult_noise1)
            mult_noise2 = self.resize(mult_noise2)
            opacity_coef = mult_noise1 * mult_noise2
            opacity_coef = torch.clamp_max(opacity_coef, 1.)
        else:
            opacity_coef = random.random() * 0.55 + 0.25
        # immc = plt.imshow(mult_noise.squeeze())
        # cb = plt.colorbar(immc)
        # plt.show()
        # plt.close()
        #
        # opacity *= mult_noise
        # plt.imshow(opacity.squeeze(), cmap='gray')
        # plt.show()
        # exit()

        float_img = img.float()

        inv_size = random.random() * 0.35 + 0.1
        cond = torch.logical_or(noise < inv_size, ~op_mask)
        if random.random() < 0.75:
            opacity[cond] = 0.
            opacity = deform(opacity, op_mask)
            cond = opacity < 0.5
        blur_fac = random.randint(0, 3) * 2 + 5
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
        opacity[cond] = float_img[cond]
        # opacity[cond] = float_img[cond]
        blur_fac = random.randint(0, 9) * 2 + 17
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
        opacity[cond] = float_img[cond]

        # mask = torch.max_pool2d(mask.int(), kernel_size=5, stride=1, padding=2)
        # opacity *= mask

        # img[opacity > 0] = ((1. - opacity_coef) * img + opacity_coef * opacity)[opacity > 0]
        img = (1. - opacity_coef) * img + opacity_coef * opacity
        return img

    def disordered_opacity(self, img, mask, region, _):
        if not region:
            return img

        opacity = torch.zeros_like(img)

        # Area + Power
        ab_power = random.random() * 10. + 18.5
        opacity[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = ab_power

        opacity = opacity * mask

        sparsity = random.random() * 1.45 + 0.05

        bias = torch.cat([torch.linspace(start=0.2, end=1.52, steps=img.shape[-1] // 2), torch.linspace(start=1.52, end=0.2, steps=img.shape[-1] // 2)])[None, ...] ** sparsity
        bias_sym = bias.T @ bias

        # adding noise
        frequency_scale1 = random.random() * 24 + 8
        frequency_scale2 = random.random() * 24 + 8
        noise = torch.rand(1, int(img.shape[-2] / frequency_scale1), int(img.shape[-1] / frequency_scale1)) * 1.2
        noise2 = torch.rand(1, int(img.shape[-2] / frequency_scale2), int(img.shape[-1] / frequency_scale2)) * 1.2
        noise = self.resize(torch.round(noise))
        noise2 = self.resize(torch.round(noise2))
        noise = noise * noise2
        noise = noise * bias_sym
        if random.random() < 0.33:
            inv_size = random.random() * 0.3 + 0.45
            noise[noise < inv_size] = 0.
        noise[noise >= 1.5] = 1.5
        noise = torch.nn.functional.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
        opacity *= noise

        blur_fac = random.randint(0, 4) * 2 + 5
        opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)

        opacity *= mask

        img = img + opacity
        return img

    def contour_deformation(self, img, mask, region, mask_center):
        if not region:
            return img

        if region[1][1] < img.shape[-1]:
            sides = [0]
        elif region[1][0] > 0:
            sides = [1]
        else:
            sides = [0, 1]

        def deform_map(c_map, freq_range=40, freq_min=60, int_range=0.16, int_min=0.08):
            x_freq = random.random() * freq_range + freq_min
            y_freq = random.random() * freq_range + freq_min
            x_deform_intensity = random.random() * int_range + int_min
            y_deform_intensity = random.random() * int_range + int_min
            def_grid_x = self.deform_resize((torch.rand(1, int(img.shape[-2] / x_freq), int(img.shape[-1] / x_freq)) - 0.5) * x_deform_intensity).squeeze(0)
            def_grid_y = self.deform_resize((torch.rand(1, int(img.shape[-2] / y_freq), int(img.shape[-1] / y_freq)) - 0.5) * y_deform_intensity).squeeze(0)
            bspline = gryds.BSplineTransformation([def_grid_y, def_grid_x], order=1)
            interpolator = gryds.Interpolator(c_map)
            return torch.tensor(interpolator.transform(bspline))

        def generate_grids_with_dist_map(c_dialed_lung):
            dist_map, coords_map = distance_transform_edt(c_dialed_lung.squeeze(), return_indices=True)
            dist_map = torch.tensor(dist_map)
            coords_map = torch.tensor(coords_map).float()

            y_vals = self.base_map
            x_vals = self.base_map.T

            y_dist = (y_vals - coords_map[0])
            x_dist = (x_vals - coords_map[1])

            y_abs_dist = y_dist.abs()
            x_abs_dist = x_dist.abs()

            max_y_dist = y_abs_dist.max()
            max_x_dist = x_abs_dist.max()

            y_dist += (dial_ker_size // 2) * y_dist / max_y_dist
            x_dist += (dial_ker_size // 2) * x_dist / max_x_dist

            c_grid_y = -y_dist / 768
            c_grid_x = -x_dist / 768
            dist_map /= 768

            freq1 = int((random.random() ** 3) * 21 + 4)
            freq2 = int((random.random() ** 3) * 21 + 4)
            noise3 = self.resize(torch.rand((1, freq1, freq2)) + 0.5).squeeze()
            noise3 = deform_map(noise3)

            def_dist_map = dist_map * noise3

            mag_y = random.random() * 0.04 + 0.02
            mag_x = random.random() * 0.04 + 0.02
            c_grid_y[def_dist_map >= mag_y] = 0
            c_grid_x[def_dist_map >= mag_x] = 0

            return c_grid_y, c_grid_x

        def generate_grids_from_rules():
            if random.random() < 0.7:
                c_trig = False
                y_mag = (random.random() ** 2) * 0.325 + 0.1 * (random.random() > 0.15)
                c_grid_y = torch.cat([torch.linspace(start=0, end=0, steps=img.shape[-1] // 4),
                                      torch.linspace(start=y_mag, end=0., steps=img.shape[-1] * 3 // 4)])[None, ...]
                c_grid_y = c_grid_y.repeat((img.shape[-2], 1)).T

                x_mag = (random.random() ** 3) * 0.12 + 0.025 * (random.random() > 0.45)
                c_grid_x = torch.cat([torch.linspace(start=0, end=0, steps=img.shape[-1] // 4), torch.linspace(start=x_mag, end=0., steps=img.shape[-1] * 3 // 4)])[None, ...]
                c_grid_x = c_grid_x.repeat((img.shape[-2], 1))
                if side == 1:
                    c_grid_x = -torch.flip(c_grid_x, dims=(-1,))

                c_grid_y = deform_map(c_grid_y)
                c_grid_y = deform_map(c_grid_y, freq_range=20, freq_min=20, int_range=0.05, int_min=0.05)
                c_grid_x = deform_map(c_grid_x, int_range=0.08, int_min=0.075)
            else:
                c_trig = True
                def_steps = int(img.shape[-1] * (random.random() * 0.3 + 0.25))
                comp_steps = img.shape[-1] - def_steps
                mag = random.random() * 0.1 + 0.1
                c_grid_y = torch.cat([torch.linspace(start=0, end=0, steps=comp_steps),
                                      torch.linspace(start=mag, end=0., steps=def_steps)])[None, ...]
                c_grid_y = c_grid_y.repeat((img.shape[-2], 1)).T
                c_grid_x = torch.zeros_like(c_grid_y)

                c_grid_y = deform_map(c_grid_y)
                c_grid_y = deform_map(c_grid_y, freq_range=20, freq_min=20, int_range=0.05, int_min=0.05)

            return c_grid_y, c_grid_x, c_trig

        lungs = get_sep_lung_masks(mask.squeeze(), ret_right_then_left=True)

        for side in sides:
            trig = False

            cur_lung = torch.tensor(lungs[side]).unsqueeze(0).float()
            dial_ker_size = random.randint(0, 6) * 2 + 11
            dialed_lung = torch.max_pool2d(cur_lung, kernel_size=dial_ker_size, stride=1, padding=dial_ker_size // 2)

            x = random.random()
            if x <= 0.35:
                grid_y, grid_x, trig = generate_grids_from_rules()
            elif 0.35 < x < 0.65:
                grid_y, grid_x = generate_grids_with_dist_map(dialed_lung)
            else:
                dist_map_grid_y, dist_map_grid_x = generate_grids_with_dist_map(dialed_lung)
                rules_grid_y, rules_grid_x, trig = generate_grids_from_rules()
                grid_y = torch.zeros_like(dist_map_grid_y)
                grid_x = torch.zeros_like(dist_map_grid_x)

                y_cond = rules_grid_y.abs() >= dist_map_grid_y.abs()
                grid_y[y_cond] = rules_grid_y[y_cond]
                grid_y[~y_cond] = dist_map_grid_y[~y_cond]

                x_cond = rules_grid_x.abs() >= dist_map_grid_x.abs()
                grid_x[x_cond] = rules_grid_x[x_cond]
                grid_x[~x_cond] = dist_map_grid_x[~x_cond]

            grid_x = dialed_lung.squeeze() * grid_x
            grid_y = dialed_lung.squeeze() * grid_y

            bspline = gryds.BSplineTransformation([grid_y, grid_x], order=1)
            img_mask = torch.cat([img, cur_lung], dim=0)
            interpolator = gryds.MultiChannelInterpolator(img_mask, data_format='channels_first', mode='nearest')
            def_img_mask = torch.tensor(interpolator.transform(bspline))

            def_img = def_img_mask[0].unsqueeze(0)
            def_mask = def_img_mask[1].unsqueeze(0)
            opening_coef = random.randint(0, 7) * 2 + 5
            def_mask = torch.max_pool2d(-torch.max_pool2d(-def_mask, kernel_size=opening_coef, stride=1, padding=opening_coef // 2), kernel_size=opening_coef, stride=1, padding=opening_coef // 2)

            def_cond = def_mask > 0.8
            double_dialed_mask = torch.max_pool2d(dialed_lung, kernel_size=dial_ker_size, stride=1, padding=dial_ker_size // 2)
            def_change_mask = torch.max_pool2d((~def_cond) * cur_lung, kernel_size=dial_ker_size, stride=1, padding=dial_ker_size // 2)
            dial_mask = double_dialed_mask - cur_lung
            dial_change_mask = dial_mask * torch.max_pool2d(def_change_mask, kernel_size=dial_ker_size, stride=1, padding=dial_ker_size // 2)
            # def_change_mask *= dialed_right_lung

            def_change_cond = def_change_mask > 0.8
            blur_im_fac = random.randint(0, 7) * 2 + 17
            blur_im = v2.functional.gaussian_blur(img.float(), kernel_size=blur_im_fac)
            blur_def_fac = random.randint(0, 30) * 2 + 45
            blur_def = v2.functional.gaussian_blur(def_img, kernel_size=blur_def_fac)
            def_img[def_change_cond] = blur_def[def_change_cond]
            def_coef = random.random() * 0.65 + 0.35
            def_img[def_change_cond] = def_coef * def_img[def_change_cond] + (1 - def_coef) * blur_im[def_change_cond]

            # dial_mean = torch.mean(img[dial_mask > 0.8])
            dial_mean = torch.mean(img[dial_mask > 0.8])
            def_mean = torch.mean(def_img[def_change_cond])

            def_img[def_change_cond] -= def_mean - dial_mean
            def_img = def_img.float()
            if trig:
                def_img[def_cond] = img.float()[def_cond]
            else:
                def_img[def_mask == 1] = img.float()[def_mask == 1]

            blur_def_im_fac = random.randint(0, 12) * 2 + 31
            blur_def_img = v2.functional.gaussian_blur(def_img, kernel_size=blur_def_im_fac)
            def_img[dial_change_mask == 1] = blur_def_img[dial_change_mask == 1]
            img = def_img

        return img

    def combined_opacity(self, img, mask, region, mask_center):
        if not region:
            return img

        opacity1 = torch.zeros_like(img)

        # Area + Power
        ab_power = random.random() * 40. + 25.
        opacity1[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = ab_power

        opacity1 = opacity1 * mask.bool()

        sparsity = random.random() * 1.75 + 0.75
        dist = random.random() * 0.325 + 0.05

        bias = torch.cat([torch.linspace(start=dist, end=1., steps=img.shape[-1] // 2), torch.linspace(start=1., end=dist, steps=img.shape[-1] // 2)])[None, ...] ** sparsity
        bias_sym = bias.T @ bias
        im_cent = img.shape[-2] // 2, img.shape[-1] // 2
        op_mask = opacity1.bool().int()
        min_b, max_b = get_bounding_rect(op_mask)
        # root_coord = int(random.random() * (region[0][1] - region[0][0]) + region[0][0]), int(random.random() * (region[1][1] - region[1][0]) + region[1][0])
        root_coord = int(random.random() * (max_b[0] - min_b[0]) + min_b[0]), int(random.random() * (max_b[1] - min_b[1]) + min_b[1])

        bias_sym = torch.roll(bias_sym, shifts=(root_coord[0] - im_cent[0], root_coord[1] - im_cent[1]), dims=(-2, -1))
        # print(f'region {region}')
        # print(f'center {im_cent}')
        # print(f'root coord {root_coord}')
        # print(f'new roll coord {im_cent[0] - root_coord[0], im_cent[1] - root_coord[1]}')

        # adding noise
        frequency_scale1 = random.random() * 16 + 12
        frequency_scale2 = random.random() * 16 + 12
        noise1_1 = torch.rand(1, int(img.shape[-2] / frequency_scale1), int(img.shape[-1] / frequency_scale1))
        noise1_2 = torch.rand(1, int(img.shape[-2] / frequency_scale2), int(img.shape[-1] / frequency_scale2))
        noise1_1 = self.resize(torch.round(noise1_1)) * 1.75
        noise1_2 = self.resize(torch.round(noise1_2)) * 1.75
        noise1 = noise1_1 * noise1_2
        noise1 = noise1 * bias_sym
        # inv_size = random.random() * 0.3 + 0.45
        # noise1[noise1 < inv_size] = 0.
        noise1[noise1 >= 1.125] = 1.125
        noise1 = torch.nn.functional.avg_pool2d(noise1, kernel_size=5, stride=1, padding=2)
        opacity1 *= noise1

        blur_fac = random.randint(0, 4) * 2 + 9
        opacity1 = torch.nn.functional.avg_pool2d(opacity1, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
        img_dis = img + opacity1

        #########################

        cent_mult = random.random() * 0.08 + 0.95
        center_val = img[0, mask_center[0], mask_center[1]] * cent_mult

        opacity2 = torch.zeros_like(img)

        # Area + Power
        opacity2[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = center_val

        # Dilation of mask
        dial_mask = torch.max_pool2d(mask, kernel_size=9, stride=1, padding=4)

        op_mask = (dial_mask * opacity2).bool()

        # op_mask = (mask * opacity2).bool()

        # adding noise
        frequency_scale = random.random() * 90 + 60
        noise2 = torch.rand(1, int(img.shape[-2] / frequency_scale), int(img.shape[-1] / frequency_scale))
        noise2 = self.resize(torch.round(noise2))
        inv_size = random.random() * 0.225 + 0.075
        # inv_size = 0.
        cond = torch.logical_or(noise2 < inv_size, ~op_mask)
        opacity2[cond] = img[cond]
        blur_fac = random.randint(0, 4) * 2 + 31
        opacity2 = torch.nn.functional.avg_pool2d(opacity2, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
        opacity2[cond] = img[cond]
        # mask = torch.max_pool2d(mask.int(), kernel_size=5, stride=1, padding=2)
        # opacity2 *= mask

        img_coef = random.random() * 0.3 + 0.15
        gen_opacity_coef = random.random() * 0.3 + 0.3
        dis_opacity_coef = 1. - (img_coef + gen_opacity_coef)
        img[opacity2 > 0] = (img_coef * img + gen_opacity_coef * opacity2 + dis_opacity_coef * img_dis)[opacity2 > 0]

        return img

    def effusion_opacity(self, img, mask, region, mask_center):
        if not region:
            return img

        min_c, max_c = get_bounding_rect(mask)

        cent_mult = random.random() * 0.05 + 0.975
        # cent_mult = 1.
        # center_val = img[0, mask_center[0], mask_center[1]] * cent_mult
        if max_c[0] + 10 < img.shape[-2]:
            mask_width = max_c[1] - min_c[1]
            sample_coord = (max_c[0] + 10, min_c[1] + mask_width * 3 // 20)
        else:
            sample_coord = (mask_center[0], mask_center[1])
        center_val = img[0, sample_coord[0], sample_coord[1]] * cent_mult

        opacity = torch.ones_like(img) * center_val

        if region[0][0] == 0:
            mask_height = max_c[0] - min_c[0]
        else:
            mask_height = mask_center[0] - min_c[0] + 20
        sin_bias = max_c[0] - (1 - (random.random() * 0.5 + 0.5) * random.random()) * mask_height
        waves_num = random.random() * 3.5 + 0.5
        if random.random() < 0.6:
            amp_mult = torch.tensor(np.interp(self.interp_range, np.array([0, 191, 383, 575, 767]), (np.random.random(5) * 30 + 10) * (np.random.randint(0, 2) * 2 - 1)))
        else:
            amp_mult = random.random() * 25 + 15
        sin_wave = amp_mult * torch.sin(torch.linspace(0., waves_num * 2. * torch.pi, img.shape[-1])) + sin_bias
        mask_map = (self.base_map > sin_wave)
        if region[1][0] != 0:
            mask_map[:, :mask_center[1]] = 0.
        elif region[1][1] != img.shape[-1]:
            mask_map[:, mask_center[1]:] = 0.
        c_mask = mask * mask_map

        # Dilation of mask
        erod_mask = (-torch.max_pool2d(-c_mask, kernel_size=9, stride=1, padding=4)).bool()
        # dial_mask[0, :mask_center[0]] = 0.

        dialed_mask = torch.max_pool2d(c_mask, kernel_size=23, stride=1, padding=11).bool()
        dialed_mask = dialed_mask * mask_map
        dial_mask = torch.max_pool2d(dialed_mask.float(), kernel_size=23, stride=1, padding=11).bool()
        dial_mask = dial_mask * mask_map
        # dialed_mask[0, :mask_center[0]] = 0.
        op_mask = (dialed_mask * opacity).bool()

        opacity_coef = random.random() * 0.05 + 0.75

        cond = ~op_mask
        opacity[cond] = img[cond]

        img = (1. - opacity_coef) * img + opacity_coef * opacity
        diff_mask = torch.logical_and(dial_mask, ~erod_mask).bool()
        img[diff_mask] = torch.nn.functional.avg_pool2d(img, kernel_size=33, stride=1, padding=16)[diff_mask]
        return img

    # def effusion_opacity2(self, img, mask, region, mask_center):
    #     if not region:
    #         return img
    #
    #     min_c, max_c = get_bounding_rect(mask)
    #
    #     cent_mult = random.random() * 0.05 + 0.975
    #     # cent_mult = 1.
    #     # center_val = img[0, mask_center[0], mask_center[1]] * cent_mult
    #     if max_c[0] + 10 < img.shape[-2]:
    #         mask_width = max_c[1] - min_c[1]
    #         sample_coord = (max_c[0] + 10, min_c[1] + mask_width * 3 // 20)
    #     else:
    #         sample_coord = (mask_center[0], mask_center[1])
    #     center_val = img[0, sample_coord[0], sample_coord[1]] * cent_mult
    #
    #     opacity = torch.ones_like(img) * center_val
    #
    #     if region[0][0] == 0:
    #         mask_height = max_c[0] - min_c[0]
    #     else:
    #         mask_height = mask_center[0] - min_c[0] + 20
    #     sin_bias = max_c[0] - (1 - (random.random() * 0.5 + 0.5) * random.random()) * mask_height
    #     waves_num = random.random() * 3.5 + 0.5
    #     if random.random() < 0.6:
    #         amp_mult = torch.tensor(np.interp(self.interp_range, np.array([0, 191, 383, 575, 767]), (np.random.random(5) * 30 + 10) * (np.random.randint(0, 2) * 2 - 1)))
    #     else:
    #         amp_mult = random.random() * 25 + 15
    #     sin_wave = amp_mult * torch.sin(torch.linspace(0., waves_num * 2. * torch.pi, img.shape[-1])) + sin_bias
    #     mask_map = (self.base_map > sin_wave)
    #     if region[1][0] != 0:
    #         mask_map[:, :mask_center[1]] = 0.
    #     elif region[1][1] != img.shape[-1]:
    #         mask_map[:, mask_center[1]:] = 0.
    #     c_mask = mask * mask_map
    #
    #     # Dilation of mask
    #     erod_mask = (-torch.max_pool2d(-c_mask, kernel_size=9, stride=1, padding=4)).bool()
    #     # dial_mask[0, :mask_center[0]] = 0.
    #
    #     dialed_mask = torch.max_pool2d(c_mask, kernel_size=23, stride=1, padding=11).bool()
    #     dialed_mask = dialed_mask * mask_map
    #     dial_mask = torch.max_pool2d(dialed_mask.float(), kernel_size=23, stride=1, padding=11).bool()
    #     dial_mask = dial_mask * mask_map
    #     # dialed_mask[0, :mask_center[0]] = 0.
    #     op_mask = (dialed_mask * opacity).bool()
    #
    #     opacity_coef = random.random() * 0.05 + 0.75
    #
    #     cond = ~op_mask
    #     opacity[cond] = img[cond]
    #
    #     img = (1. - opacity_coef) * img + opacity_coef * opacity
    #     diff_mask = torch.logical_and(dial_mask, ~erod_mask).bool()
    #     img[diff_mask] = torch.nn.functional.avg_pool2d(img, kernel_size=33, stride=1, padding=16)[diff_mask]
    #     return img

    def devices(self, img, mask, region, _):
        inv_size = random.random() * 0.375 + 0.125
        freq_x = int(random.random() * 36. + 27.)
        freq_y = int(random.random() * 36. + 27.)
        sparsity = random.random() * 0.0075 + 0.9815
        im_coef = random.random() * 0.55 + 0.2

        resize_max_idx = len(self.device_resizes) - 1
        idx = random.randint(0, resize_max_idx)
        resize = self.device_resizes[idx]

        mask = torch.nn.functional.max_pool2d(mask, kernel_size=11, stride=1, padding=5)

        orig_noise = torch.rand((1, int(img.shape[-2] / freq_y), int(img.shape[-1] / freq_x)))
        orig_noise[orig_noise > sparsity] = 1.
        orig_noise[orig_noise < sparsity] = 0.
        noise = resize(orig_noise)
        noise[noise > inv_size] = 1.
        noise[noise < inv_size] = 0.

        if random.random() < 0.425:
            noise2 = self.device_resizes[random.randint(0, resize_max_idx)](orig_noise)
            if random.random() < 0.66:
                size_mult = random.random() * 0.5 + 0.75
                noise2[noise2 > size_mult * inv_size] = 1.
                noise2[noise2 < size_mult * inv_size] = 0.
                shift_x = int(random.random() * 12.5 + 12.5)
                shift_y = int(random.random() * 12.5 + 12.5)
                shifts = [shift_y, shift_x]
                if random.random() < 0.85:
                    shifts[random.randint(0, 1)] = 0
                noise2 = torch.roll(noise2, shifts, dims=(-2, -1))
                if random.random() < 0.5:
                    noise = torch.logical_and(noise.bool(), ~noise2.bool())
                else:
                    noise = torch.logical_or(noise.bool(), noise2.bool())
            else:
                size_mult = random.random() + 2.
                noise2[noise2 > size_mult * inv_size] = 1.
                noise2[noise2 < size_mult * inv_size] = 0.
                val_mult = min(1., random.random() * 0.9 + 0.4)
                noise2 *= val_mult
                # noise = torch.logical_and(noise.bool(), ~noise2.bool())
                noise = noise - noise2
            noise = noise.float()
            # plt.imshow(noise.squeeze(), cmap='gray')
            # plt.show()
            # plt.imshow(noise2.squeeze(), cmap='gray')
            # plt.show()
            # exit()

        if random.random() < 0.35:
            mult_freq_x = int(random.random() * 8. + 8.)
            mult_freq_y = int(random.random() * 8. + 8.)
            mult_noise = torch.rand((1, int(img.shape[-2] / mult_freq_y), int(img.shape[-1] / mult_freq_x)))
            mult_noise = (mult_noise + 0.6).clamp_max_(1.)
            mult_noise = self.resize(mult_noise)
            noise *= mult_noise

        noise = self.perspective(noise)
        noise = noise * mask
        noise *= 255.

        img[noise > 0.] = (im_coef * img + (1. - im_coef) * noise).to(img.dtype)[noise > 0.]
        return img


def det_general_opacity(img, mask, region, mask_center, resize):
    if not region:
        return img

    center_val = img[0, mask_center[0], mask_center[1]] * 0.975

    opacity = torch.zeros_like(img)

    # Area + Power
    opacity[0, region[0][0]: region[0][1], region[1][0]: region[1][1]] = center_val
    noise = torch.rand((1, int(img.shape[-2] / 32), int(img.shape[-1] / 32))) * 0.5 + 0.5
    noise = resize(noise)
    opacity *= noise

    op_mask = (mask * opacity).bool()

    cond = ~op_mask
    opacity[cond] = img[cond]
    blur_fac = 15
    opacity = torch.nn.functional.avg_pool2d(opacity, kernel_size=blur_fac, stride=1, padding=blur_fac // 2)
    opacity[cond] = img[cond]

    opacity_coef = 0.55
    img[opacity > 0] = ((1. - opacity_coef) * img + opacity_coef * opacity)[opacity > 0]
    return img


def get_covering_variants(im, mask, func=det_general_opacity):
    mask_cent = get_mask_center_coord(mask)
    min_c, max_c = get_bounding_rect(mask)
    h = (max_c[0] - min_c[0]).item()
    w = (max_c[1] - min_c[1]).item()
    s_h = min_c[0].item()
    s_w = min_c[1].item()

    resize = v2.Resize((im.shape[-2], im.shape[-1]))

    variants = []
    for i in range(11):
        c_region_h = [(s_h + h * max(0, i - 1) // 10, s_h + h * min(10, i + 1) // 10), (0, im.shape[-1])]
        c_im_h = func(im.clone(), mask, c_region_h, mask_cent, resize)
        variants.append(c_im_h)
        c_region_w = [(0, im.shape[-2]), (s_w + w * max(0, i - 1) // 10, s_w + w * min(10, i + 1) // 10)]
        c_im_w = func(im.clone(), mask, c_region_w, mask_cent, resize)
        variants.append(c_im_w)
    # plt.imshow(variants[6].squeeze().cpu(), cmap='gray')
    # plt.savefig('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cover_test.png')
    # exit()
    return variants


@torch.no_grad()
def generate_pred_with_uncertainty(bl, bl_seg, fu, fu_seg, model, random_geometric_tf, rescale_tf, random_intensity_tf, mask_crop_tf):
    orig_bl = bl.clone().view(1, IMG_SIZE, IMG_SIZE).cpu()
    orig_fu = fu.clone().view(1, IMG_SIZE, IMG_SIZE).cpu()
    orig_bl_seg = bl_seg.view(1, IMG_SIZE, IMG_SIZE).cpu()
    orig_fu_seg = fu_seg.view(1, IMG_SIZE, IMG_SIZE).cpu()

    outputs = []

    model.train()
    for j in range(40):
        c_bl = random_intensity_tf(rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0]))
        c_fu = random_intensity_tf(rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0]))

        c_bl = c_bl.unsqueeze(0).cuda()
        c_fu = c_fu.unsqueeze(0).cuda()

        outs = model(c_bl, c_fu).squeeze()
        outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
        # outs[torch.logical_and(outs < 0.02, outs > -0.02)] = 0.
        outs[:30, -30:] = 0.

        outputs.append(outs[None, ...])

    outputs = torch.cat(outputs)
    outs_mean = torch.mean(outputs, dim=0)
    outs_mean_abs = outs_mean.abs()
    outs_std = torch.std(outputs, dim=0)
    outs_r = 0.91 + 1 / (0.5 + torch.exp(40. * outs_mean_abs))
    outs_mean_div = outs_mean_abs * outs_r
    outs_uncertainty = outs_std / outs_mean_div
    # outs_uncertainty[outs_mean_abs < 0.005] = 0.
    outs_uncertainty[torch.logical_and(outs_mean_abs < 0.005, outs_std < 0.005)] = 0.
    # outs_uncertainty[torch.logical_and(outs_mean.abs() < 0.005, outs_std >= 0.0075)] = 2.
    outs_uncertainty.clamp_max_(2.)
    outs_mean[outs_mean_abs < 0.02] = 0.

    return outs_mean, outs_uncertainty


if __name__ == '__main__':
    from datasets import LongitudinalMIMDataset

    ds = LongitudinalMIMDataset(['images'])
    bl, fu, fu_gt, fu_mask = ds[0]

    plt.imshow(bl.squeeze(), cmap='gray')
    plt.title(f"Final BL image")
    plt.show()

    plt.imshow(fu.squeeze(), cmap='gray')
    plt.title(f"Final FU image")
    plt.show()

    plt.imshow(fu_gt.squeeze(), cmap='gray')
    plt.title(f"Difference map")
    plt.show()

    exit()
#
#     # fold = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/test'
#     p1 = '00000067_000.nii.gz'
#     im1 = nib.load(p1).get_fdata().T
#     p2 = '00000067_000_seg.nii.gz'
#     im2 = nib.load(p2).get_fdata().T
#
#     im1 = torch.tensor(im1[None, ...])
#     im2 = torch.tensor(im2[None, ...])
#
#     im = torch.cat([im1, im2], dim=0)
#     #
#     # im_interpolator = gryds.MultiChannelInterpolator(im, data_format='channels_first', mode='nearest')
#     #
#     # cent = get_mask_center_prop(im[1])
#     # print(cent)
#     # tf = gryds.AffineTransformation(ndim=2, center=(0.5, 0.5), scaling=(1.15, 1.15), angles=(np.pi / 12,), translation=(-0.1, 0.1,))
#     # #
#     # # tf = get_bspline_tf_of_random_composition((im.shape[-2] // 8, im.shape[-1] // 8))
#     # def_im = im_interpolator.transform(tf)
#     a = RandomAbnormalizationTransform()
#     # def_im = a(im1.clone(), im2)
#
#     b = RescaleValuesTransform()
#
#     def_im = a(im1.clone(), im2)
#     def_im = b(def_im)
#     # def_im = kornia.enhance.equalize_clahe(def_im, clip_limit=c, grid_size=(8,8))
#
#     im1 = b(im1)
#     # im1 = kornia.enhance.equalize_clahe(im1, clip_limit=2.2, grid_size=(8,8))
#
#     # ax = plt.gca()
#     # imm = ax.imshow(opacity.squeeze().numpy(), cmap='gray')
#     # cbar = ax.figure.colorbar(imm, ax=ax)
#     # plt.show()
#     # exit()
#     # print(def_im)
#
#     fig, axs = plt.subplots(2, 2)
#     axs[0,0].imshow(im1[0], cmap='gray')
#     axs[0,1].imshow(def_im[0, :, :], cmap='gray')
#     imm = axs[1,0].imshow((def_im[0] - im1[0]).squeeze(), cmap='gray')
#     cbar = axs[1,0].figure.colorbar(imm, ax=axs[1,0])
#     # plt.savefig('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ab_plot.png')
#     plt.show()
