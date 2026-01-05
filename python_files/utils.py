import os.path

from constants import *
import torch.nn.functional as F
import torch
import numpy as np
import scipy.ndimage as ndi
from math import e


STRUCT = ndi.generate_binary_structure(2, 2)


def crop_out_edges(arr: np.ndarray, get_coords=False):
    min_v = np.min(arr)
    max_v = np.max(arr)
    coord_bounds_to_keep = np.argwhere(np.logical_and(arr > min_v, arr < max_v))
    min_c = np.min(coord_bounds_to_keep, axis=0)
    max_c = np.max(coord_bounds_to_keep, axis=0)
    arr = arr[min_c[0]: max_c[0] + 1, min_c[1]: max_c[1] + 1]
    if get_coords:
        return arr, min_c[0], min_c[1], max_c[0] + 1, max_c[1] + 1
    return arr


def get_mimic_path(subject, study, dicom, add_cropped=False):
    if not add_cropped:
        return MIMIC_FOLDER + f'/other_files/p{str(subject)[:2]}/p{subject}/s{study}/{dicom}'
    return MIMIC_FOLDER + f'/other_files/p{str(subject)[:2]}/p{subject}/s{study}/{dicom}_cropped.nii.gz'


def check_existence_of_mimic_path(path):
    return os.path.exists(path)


class MaskProbScheduler:
    def __init__(self, epochs, steps_per_epoch, init_val=INIT_MASK_PROB, max_val=MAX_MASK_PROB, end_val=END_MASK_PROB, perc_on_start=0.05, perc_on_slope=0.2, perc_on_max=0.4, perc_on_slope2=0.1):
        total_steps = epochs * steps_per_epoch
        self.init_val = init_val
        self.max_val = max_val
        self.end_val = end_val
        # self.mult_fac = max_val - init_val
        self.th1 = perc_on_start * total_steps
        self.th2 = self.th1 + perc_on_slope * total_steps
        self.th3 = self.th2 + perc_on_max * total_steps
        self.th4 = self.th3 + perc_on_slope2 * total_steps
        self.slope1 = (max_val - init_val) / (self.th2 - self.th1)
        self.slope2 = (end_val - max_val) / (self.th4 - self.th3)
        self.cur_step = 0

    def get_step(self):
        return self.cur_step

    def set_step(self, step):
        self.cur_step = step

    def calc_cur_val(self):
        if self.cur_step <= self.th1:
            val = self.init_val
        elif self.th1 < self.cur_step <= self.th2:
            val = (self.cur_step - self.th1) * self.slope1 + self.init_val
        elif self.th2 < self.cur_step <= self.th3:
            val = self.max_val
        elif self.th3 < self.cur_step <= self.th4:
            val = (self.cur_step - self.th3) * self.slope2 + self.max_val
            # val = rand().item() * self.mult_fac + self.init_val
        else:
            val = self.end_val
        return val

    def step(self):
        val = self.calc_cur_val()
        self.cur_step += 1
        return val


def masked_patches_l1_loss(outputs, inputs, mask, patch_size=MASK_PATCH_SIZE):
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


def fourier(x):  # 2D Fourier transform
    b, c, h, w = x.shape
    f = torch.fft.rfft2(x.to(torch.float32))
    f = f.abs() + 1e-6
    f = f.log()
    f = torch.roll(f, shifts=(int(h/2), int(w/2)), dims=(2, 3))
    return f


def masked_patches_fourier_loss(outputs, inputs_fourier, mask, patch_size=MASK_PATCH_SIZE):
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


def masked_patches_SSIM_loss(outputs, inputs, mask, ssim1, ssim2, patch_size=MASK_PATCH_SIZE):
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


def masked_patches_GAN_loss(outputs, mask, disc, gan_loss, patch_size=MASK_PATCH_SIZE):
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


def get_sep_lung_masks(seg, ret_right_then_left=False):
    label, _ = ndi.label(seg, structure=STRUCT)
    ccs, counts = np.unique(label, return_counts=True)
    num_ccs = len(ccs)

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


def scale_and_suppress_non_max(cur_img, new_upper_bound=0.3, scale_fac: float = 1.):
    abs_img = cur_img.abs()
    abs_max = torch.max(abs_img).item()
    new_max = min(abs_max, new_upper_bound)
    scaled_img = torch.exp(scale_fac * abs_img) - 1
    scaled_img = scaled_img / (torch.max(scaled_img) / new_max)
    cur_img = scaled_img * torch.sign(cur_img)
    return cur_img


def get_max_inpaint_diff_val(init_max):
    # return -0.372 + 0.625 * (1 - e ** (-7.126 * init_max))
    # return -0.3 + 0.625 * (1 - e ** (-5.4 * init_max))
    return -0.34 + 0.725 * (1 - e ** (-5.4 * init_max))


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val
    alphas_map = alphas_map.squeeze()

    return alphas_map
