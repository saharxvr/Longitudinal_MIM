import os

import numpy as np

from models import *
import torch
import nibabel as nib
from datasets import LongitudinalMIMDataset
from augmentations import *
import matplotlib.pyplot as plt
from matplotlib import colors
import torchvision.transforms.v2 as v2
from extra.case_filtering import AFFINE_DCM
from utils import get_sep_lung_masks, STRUCT
from skimage.measure import regionprops
from scipy.ndimage import label


def generate_alpha_map(x: torch.Tensor):
    # alphas_map = outs.abs()
    # alphas_map = (alphas_map - torch.min(alphas_map)) / (torch.max(alphas_map) - torch.min(alphas_map))
    x_abs = x.abs()
    #TODO: PREVIOUSLY
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val

    # max_val = 0.2
    # alphas_map = torch.sqrt(x_abs.clamp_max_(0.2)) / (max_val ** 0.5)

    # alphas_map[x_abs < 0.03] *= 0.75
    # alphas_map[x_abs < 0.015] *= 0.5
    # if torch.max(x_abs).item() < 0.06:
        # low_vals = (x_abs < 0.03).float()
        # low_vals = x_abs * low_vals
        # low_vals = (low_vals - torch.min(low_vals)) / (torch.max(low_vals) - torch.min(low_vals))
        # alphas_map[low_vals > 0] =
    # print(alphas_map)
    # alphas_map = 2.352941 * alphas_map - 0.1764706
    # alphas_map = alphas_map.clamp(0., 1.)
    # alphas_map[x < 0.04] = 0.

    # alphas_map = 1.027689 + (-0.00003443215 - 1.027689) / (1 + (alphas_map / 0.07866591) ** 279.162) ** 0.005007674
    # alphas_map = 0.981258 + (-0.008780828 - 0.981258) / (1 + (alphas_map / 0.1749422) ** 3.518648)
    # alphas_map = alphas_map.clamp(0., 1.)

    return alphas_map


@torch.no_grad()
def regular_pred(bl, bl_seg, fu, fu_seg):
    bl = clahe_tf(rescale_tf(mask_crop_tf(torch.cat([bl, bl_seg], dim=0))[0]))
    fu = clahe_tf(rescale_tf(mask_crop_tf(torch.cat([fu, fu_seg], dim=0))[0]))

    bl = bl.unsqueeze(0).cuda()
    fu = fu.unsqueeze(0).cuda()
    bl_seg = bl_seg.unsqueeze(0).cuda()
    fu_seg = fu_seg.unsqueeze(0).cuda()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(bl.squeeze().cpu(), cmap='gray')
    ax[0].set_title('BL')
    ax[1].imshow(fu.squeeze().cpu(), cmap='gray')
    ax[1].set_title('FU')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'{c_dir}/inputs.png')
    ax[0].clear()
    ax[1].clear()
    plt.cla()
    plt.clf()
    plt.close()

    for i, model in enumerate(models):
        outs = model(bl, fu).squeeze().cpu()
        # outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.

        outs_clone = outs.clone()
        outs_clone[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
        # imm1 = ax[1,0].imshow(outs_clone, alpha=1., cmap='gray')
        # cbar1 = ax[1,0].figure.colorbar(imm1, ax=ax[1,0])
        # ax[1, 0].set_title('Heatmap Prediction')
        plt.imshow(fu.squeeze().cpu(), cmap='gray')

        max_likely_val = torch.max(outs_clone)
        min_likely_val = torch.min(outs_clone)
        outs_clamped = outs.clone()
        outs_clamped = outs_clamped.clamp(min_likely_val, max_likely_val)

        alphas = generate_alpha_map(outs_clamped)
        # alphas = outs.abs()
        # alphas = (alphas - torch.min(alphas)) / (torch.max(alphas) - torch.min(alphas))

        imm2 = plt.imshow(outs_clamped, alpha=alphas)
        cbar2 = plt.colorbar(imm2, fraction=0.05, pad=0.04)
        plt.title(f'Model {i} output')

        plt.axis('off')

        plt.savefig(f'{c_dir}/model_{i}_output.png')
        plt.clf()
        plt.cla()
        plt.close()


@torch.no_grad()
def TTA_pred(bl, bl_seg, fu, fu_seg):
    orig_bl = bl.clone()
    orig_fu = fu.clone()
    orig_bl_seg = bl_seg.clone()
    orig_fu_seg = fu_seg.clone()

    cropped_bl = rescale_tf(mask_crop_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0])
    cropped_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cropped_bl.squeeze().cpu(), cmap='gray')
    ax[0].set_title('Prior')
    ax[1].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
    ax[1].set_title('Current')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'{c_dir}/inputs.png')
    ax[0].clear()
    ax[1].clear()
    plt.cla()
    plt.clf()
    plt.close()

    to_ret = []

    for i, model in enumerate(models):
        if model == 0:
            continue

        print(f'Using model {i}')

        cur_dir = c_dir + f'/model{i}'
        os.makedirs(cur_dir, exist_ok=True)

        outputs = []
        model.eval()

        for j in range(40):
            c_bl = random_intensity_tf(rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0]))
            # c_bl = rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0])
            c_fu = random_intensity_tf(rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0]))
            # c_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0])

            # c_bl = cropped_bl.unsqueeze(0).cuda()
            c_bl = c_bl.unsqueeze(0).cuda()
            # c_fu = cropped_fu.unsqueeze(0).cuda()
            c_fu = c_fu.unsqueeze(0).cuda()
            # c_bl_seg = orig_bl_seg.unsqueeze(0).cuda()
            # c_fu_seg = orig_fu_seg.unsqueeze(0).cuda()

            outs = model(c_bl, c_fu).squeeze().cpu()
            outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
            # outs[torch.logical_or(outs < 0.02, outs > -0.02)] = 0.
            if i + 1 >= 3:
                outs[:30, -30:] = 0.
            outputs.append(outs[None, ...])
            continue

        outputs = torch.cat(outputs)
        outs_mean = torch.mean(outputs, dim=0)
        outs_std = torch.std(outputs, dim=0)
        outs_mean_abs = outs_mean.abs()
        outs_r = 0.92 + 1 / (0.5 + torch.exp(40. * outs_mean_abs))
        outs_mean_div = outs_mean_abs * outs_r
        outs_uncertainty = outs_std / outs_mean_div
        # outs_uncertainty[outs_mean.abs() < 0.005] = 0.
        outs_uncertainty[torch.logical_and(outs_mean_abs < 0.005, outs_std < 0.005)] = 0.
        # outs_uncertainty[torch.logical_and(outs_mean.abs() < 0.005, outs_std >= 0.0075)] = 2.
        outs_uncertainty.clamp_max_(2.)
        # outs_mean[outs_mean_abs < 0.01] = 0.

        mean_alphas = generate_alpha_map(outs_mean)
        # std_alphas = generate_alpha_map(outs_std)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        divnorm = colors.TwoSlopeNorm(vmin=torch.min(outs_mean).item(), vcenter=0., vmax=torch.max(outs_mean).item())
        imm1 = ax[0].imshow(outs_mean.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
        cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax[0])
        ax[0].set_title('Mean outputs')
        ax[0].set_axis_off()
        # ax[1].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        imm2 = ax[1].imshow(outs_uncertainty.squeeze().cpu(), cmap='seismic')
        cbar2 = plt.colorbar(imm2, fraction=0.05, pad=0.04, ax=ax[1])
        ax[1].set_title('Uncertainty')
        ax[1].set_axis_off()
        fig.tight_layout()
        plt.savefig(f'{cur_dir}/mean_outputs_with_uncertainty.png')
        ax[0].clear()
        ax[1].clear()
        # fig.gcf()
        plt.cla()
        plt.clf()
        plt.close()

        plt.imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        imm3 = plt.imshow(outs_mean.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
        cbar3 = plt.colorbar(imm3, fraction=0.05, pad=0.04)
        low_confidence = (outs_uncertainty > 1).float().squeeze().cpu()
        plt.imshow(low_confidence, cmap='seismic', alpha=low_confidence)
        plt.axis('off')
        plt.title("Low Confidence overlaid on Output")
        plt.tight_layout()
        plt.savefig(f'{cur_dir}/mean_outputs_with_low_confidence_overlaid.png')
        plt.cla()
        plt.clf()
        plt.close()

        to_ret.append(outs_mean)
    return to_ret


@torch.no_grad()
def Monte_Carlo_pred(bl, bl_seg, fu, fu_seg, effusions_data=None, mid_y=None, plots_suffix=''):
    orig_bl = bl.clone()
    orig_fu = fu.clone()
    orig_bl_seg = bl_seg.clone()
    orig_fu_seg = fu_seg.clone()

    cropped_bl = rescale_tf(mask_crop_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0])
    cropped_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0])

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cropped_bl.squeeze().cpu(), cmap='gray')
    ax[0].set_title('Prior')
    ax[1].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
    ax[1].set_title('Current')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'{c_dir}/inputs{plots_suffix}.png')
    ax[0].clear()
    ax[1].clear()
    plt.cla()
    plt.clf()
    plt.close()

    to_ret = []

    for i, model in enumerate(models):
        if model == 0:
            continue
        print(f'Using model {i}')

        cur_dir = c_dir + f'/model{i}'
        os.makedirs(cur_dir, exist_ok=True)

        outputs = []
        model.train()

        for j in range(40):
            c_bl = random_intensity_tf(rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0]))
            # c_bl = rescale_tf(random_geometric_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0])
            c_fu = random_intensity_tf(rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0]))
            # c_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0])

            # c_bl = cropped_bl.unsqueeze(0).cuda()
            c_bl = c_bl.unsqueeze(0).cuda()
            # c_fu = cropped_fu.unsqueeze(0).cuda()
            c_fu = c_fu.unsqueeze(0).cuda()
            # c_bl_seg = orig_bl_seg.unsqueeze(0).cuda()
            # c_fu_seg = orig_fu_seg.unsqueeze(0).cuda()

            outs = model(c_bl, c_fu).squeeze().cpu()
            outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
            # outs[torch.logical_or(outs < 0.02, outs > -0.02)] = 0.
            if i + 1 >= 3:
                outs[:30, -30:] = 0.
            if i + 1 >= 7:
                outs[-40:, -40:] = 0.
            outputs.append(outs[None, ...])
            continue

        outputs = torch.cat(outputs)
        outs_mean = torch.mean(outputs, dim=0)
        outs_std = torch.std(outputs, dim=0)
        outs_mean_abs = outs_mean.abs()
        outs_r = 0.92 + 1 / (0.5 + torch.exp(40. * outs_mean_abs))
        outs_mean_div = outs_mean_abs * outs_r
        outs_uncertainty = outs_std / outs_mean_div
        # outs_uncertainty[outs_mean.abs() < 0.005] = 0.
        outs_uncertainty[torch.logical_and(outs_mean_abs < 0.005, outs_std < 0.005)] = 0.
        # outs_uncertainty[torch.logical_and(outs_mean.abs() < 0.005, outs_std >= 0.0075)] = 2.
        outs_uncertainty.clamp_max_(2.)
        # outs_mean[outs_mean_abs < 0.01] = 0.

        mean_alphas = generate_alpha_map(outs_mean)
        # std_alphas = generate_alpha_map(outs_std)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        divnorm = colors.TwoSlopeNorm(vmin=torch.min(outs_mean).item(), vcenter=0., vmax=torch.max(outs_mean).item())
        imm1 = ax[0].imshow(outs_mean.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
        cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax[0])
        if mid_y is not None:
            ax[0].plot([0, outs_mean.shape[-1]], [mid_y, mid_y], color='pink', linestyle='dashed', linewidth=1.)
        if effusions_data is not None:
            xs, ys, c_colors = effusions_data
            effs_len = len(c_colors)
            for k in range(effs_len):
                cur_xs = xs[k]
                cur_ys = ys[k]
                color = c_colors[k]
                ax[0].plot(cur_xs, cur_ys, color=color, marker='_', linestyle='dashed')
        ax[0].set_title('Mean outputs')
        ax[0].set_axis_off()
        # ax[1].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        imm2 = ax[1].imshow(outs_uncertainty.squeeze().cpu(), cmap='seismic')
        cbar2 = plt.colorbar(imm2, fraction=0.05, pad=0.04, ax=ax[1])
        if mid_y is not None:
            ax[1].plot([0, outs_mean.shape[-1]], [mid_y, mid_y], color='pink', linestyle='dashed', linewidth=1.)
        ax[1].set_title('Uncertainty')
        ax[1].set_axis_off()
        fig.tight_layout()
        plt.savefig(f'{cur_dir}/mean_outputs_with_uncertainty{plots_suffix}.png')
        ax[0].clear()
        ax[1].clear()
        # fig.gcf()
        plt.cla()
        plt.clf()
        plt.close()

        plt.imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        imm3 = plt.imshow(outs_mean.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
        cbar3 = plt.colorbar(imm3, fraction=0.05, pad=0.04)
        low_confidence = (outs_uncertainty > 1).float().squeeze().cpu()
        plt.imshow(low_confidence, cmap='seismic', alpha=low_confidence)
        plt.axis('off')
        plt.title("Low Confidence overlaid on Output")
        plt.tight_layout()
        plt.savefig(f'{cur_dir}/mean_outputs_with_low_confidence_overlaid{plots_suffix}.png')
        plt.cla()
        plt.clf()
        plt.close()

        to_ret.append((outs_mean, low_confidence))
    return to_ret


@torch.no_grad()
def generate_mask(bl, bl_seg, fu, fu_seg, other, other_seg, save_nif=True, ret_mask=False):
    orig_bl = bl.clone()
    orig_fu = fu.clone()
    orig_other = other.clone()
    orig_bl_seg = bl_seg.clone()
    orig_fu_seg = fu_seg.clone()
    orig_other_seg = other_seg.clone()

    cropped_bl = rescale_tf(mask_crop_tf(torch.cat([orig_bl, orig_bl_seg], dim=0))[0])
    cropped_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0])
    cropped_other = rescale_tf(mask_crop_tf(torch.cat([orig_other, orig_other_seg], dim=0))[0])

    if save_nif:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cropped_bl.squeeze().cpu(), cmap='gray')
        ax[0].set_title('BL')
        ax[1].imshow(cropped_fu.squeeze().cpu(), cmap='gray')
        ax[1].set_title('FU')
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        fig.tight_layout()
        plt.savefig(f'{c_dir}/inputs.png')
        ax[0].clear()
        ax[1].clear()
        plt.cla()
        plt.clf()
        plt.close()

    c_fu = rescale_tf(mask_crop_tf(torch.cat([orig_fu, orig_fu_seg], dim=0))[0]).unsqueeze(0).cuda()
    # c_fu = rescale_tf(resize_tf(orig_fu)).unsqueeze(0).cuda()

    assert len(models_mask) > 0
    for i, model in enumerate(models_mask):
        if model == 0:
            continue

        print(f'Using model {i}')

        cur_dir = c_dir + f'/model{i}'
        os.makedirs(cur_dir, exist_ok=True)

        outputs = []
        model.eval()

        bl_vars = get_covering_variants(orig_bl, orig_bl_seg)
        bl_vars = [rescale_tf(mask_crop_tf(torch.cat([v, orig_bl_seg]))[0]).unsqueeze(0).cuda() for v in bl_vars]
        bl_vars.append(cropped_bl.unsqueeze(0).cuda())
        bl_vars.append(cropped_other.unsqueeze(0).cuda())

        for var in bl_vars:
            outs = model(var, c_fu).squeeze()
            outs[torch.logical_or(outs > 0.9, outs < -0.9)] = 0.
            if i + 1 >= 3:
                outs[:30, -30:] = 0.
            outputs.append(outs[None, ...])
            continue

        # outputs = [o.abs() > 0.065 for o in outputs]
        # last_out = torch.logical_or(outputs[-1] > 0.075, outputs[-1] < -0.06)
        outputs = [torch.logical_or(o > 0.045, o < -0.035) for o in outputs]
        # outputs[-1] = last_out
        outputs = torch.cat(outputs)
        fu_mask = torch.any(outputs, dim=0, keepdim=True).float()
        # fu_mask = -torch.nn.functional.max_pool2d(-fu_mask, kernel_size=7, stride=1, padding=3)
        # fu_mask = torch.nn.functional.max_pool2d(fu_mask, kernel_size=7, stride=1, padding=3)
        # fu_mask = torch.nn.functional.max_pool2d(fu_mask, kernel_size=35, stride=1, padding=17)
        # fu_mask = -torch.nn.functional.max_pool2d(-fu_mask, kernel_size=35, stride=1, padding=17)

        fu_mask = torch.nn.functional.max_pool2d(fu_mask, kernel_size=49, stride=1, padding=24)
        fu_mask = -torch.nn.functional.max_pool2d(-fu_mask, kernel_size=49, stride=1, padding=24)
        fu_mask = -torch.nn.functional.max_pool2d(-fu_mask, kernel_size=11, stride=1, padding=5)
        fu_mask = torch.nn.functional.max_pool2d(fu_mask, kernel_size=11, stride=1, padding=5)
        fu_mask[0, :, fu_mask.shape[-1] // 2] = 0.

        # imm = plt.imshow(fu_mask.squeeze().cpu())
        # plt.colorbar(imm, fraction=0.05, pad=0.04)
        # plt.savefig('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cover_test.png')
        # exit()
        if save_nif:
            fu_nif = nib.Nifti1Image(c_fu.cpu().numpy().squeeze().T, AFFINE_DCM)
            nib.save(fu_nif, cur_dir + '/im.nii.gz')
            mask_nif = nib.Nifti1Image(fu_mask.int().cpu().numpy().T, AFFINE_DCM)
            nib.save(mask_nif, cur_dir + '/mask.nii.gz')

    if ret_mask:
        return fu_mask


def detect_effusions(bl_mask, fu_mask, ret_split_fu_masks=False):
    def determine_left_and_right(lung1, lung2, props1, props2):
        if props1.centroid[1] > props2.centroid[1]:
            return lung2, lung1, props2, props1
        return lung1, lung2, props1, props2

    bl_lung1, bl_lung2 = get_sep_lung_masks(bl_mask.cpu().numpy().squeeze())
    fu_lung1, fu_lung2 = get_sep_lung_masks(fu_mask.cpu().numpy().squeeze())
    bl_1_props = regionprops(bl_lung1.astype(np.uint8))[0]
    bl_2_props = regionprops(bl_lung2.astype(np.uint8))[0]
    fu_1_props = regionprops(fu_lung1.astype(np.uint8))[0]
    fu_2_props = regionprops(fu_lung2.astype(np.uint8))[0]

    bl_right_lung, bl_left_lung, bl_right_props, bl_left_props = determine_left_and_right(bl_lung1, bl_lung2, bl_1_props, bl_2_props)
    fu_right_lung, fu_left_lung, fu_right_props, fu_left_props = determine_left_and_right(fu_lung1, fu_lung2, fu_1_props, fu_2_props)

    bl_right_len = bl_right_props.axis_major_length
    bl_left_len = bl_left_props.axis_major_length
    fu_right_len = fu_right_props.axis_major_length
    fu_left_len = fu_left_props.axis_major_length

    bl_ratio = bl_right_len / bl_left_len
    fu_ratio = fu_right_len / fu_left_len
    cross_ratio = fu_ratio / bl_ratio

    right_len_dif = fu_right_len - bl_right_len
    left_len_dif = fu_left_len - bl_left_len
    abs_right_len_dif = abs(right_len_dif)
    abs_left_len_dif = abs(left_len_dif)

    print(f'BL ratio: {bl_ratio}')
    print(f'FU ratio: {fu_ratio}')
    print(f'Cross ratio: {cross_ratio}')
    print(f'Right len diff: {right_len_dif}')
    print(f'Left len diff: {left_len_dif}')
    if 0.8 < cross_ratio < 1.25 and min(abs_right_len_dif, abs_left_len_dif) <= 60 and max(abs_right_len_dif, abs_left_len_dif) <= 110:
        if ret_split_fu_masks:
            fu_right_lung = torch.tensor(fu_right_lung).float()
            fu_left_lung = torch.tensor(fu_left_lung).float()
            return None, (fu_right_lung, fu_left_lung)
        return None

    print("Effusion detected!!!")
    found.append((bl_name, fu_name))

    opts = [(fu_right_lung, fu_right_props, 1 / cross_ratio, right_len_dif), (fu_left_lung, fu_left_props, cross_ratio, left_len_dif)]
    final_x_coords = []
    final_y_coords = []
    final_colors = []

    if min(abs_right_len_dif, abs_left_len_dif) > 60:
        print('Len difference for both lungs is large.')
        for j in range(2):
            eff_lung, eff_props, _, eff_len_dif = opts[j]
            x_coord = int(eff_props.centroid[1])
            y_coord_1 = np.max(np.where(eff_lung)[0]).item()
            y_coord_2 = max(min(y_coord_1 - eff_len_dif, bl_mask.shape[-1]), 0)
            color = 'orange' if eff_len_dif < 0 else 'purple'
            final_x_coords.append((x_coord, x_coord))
            final_y_coords.append((y_coord_1, y_coord_2))
            final_colors.append(color)
    else:
        opts = sorted(opts, key=lambda x: abs(x[3]))
        eff_lung, eff_props, eff_diff_ratio, _ = opts[1]
        eff_len = eff_props.axis_major_length * abs(1 - eff_diff_ratio)
        dirc, color = (1, 'orange') if eff_diff_ratio > 1 else (-1, 'purple')
        x_coord = int(eff_props.centroid[1])
        y_coord_1 = np.max(np.where(eff_lung)[0]).item()
        # tops_diff = abs(np.min(np.where(opts[0][0])[0]).item() - np.min(np.where(eff_lung)[0]).item())
        # if tops_diff > 62:
        #     y_coord_1 -= tops_diff
        y_coord_2 = max(min(y_coord_1 + dirc * eff_len, bl_mask.shape[-1]), 0)
        final_x_coords.append((x_coord, x_coord))
        final_y_coords.append((y_coord_1, y_coord_2))
        final_colors.append(color)

    if ret_split_fu_masks:
        fu_right_lung = torch.tensor(fu_right_lung).float()
        fu_left_lung = torch.tensor(fu_left_lung).float()
        return (final_x_coords, final_y_coords, final_colors), (fu_right_lung, fu_left_lung)
    return final_x_coords, final_y_coords, final_colors


def calc_mid_y(sep_lungs):
    right_lung, left_lung = sep_lungs
    right_ys = torch.where(right_lung)[0]
    left_ys = torch.where(left_lung)[0]
    right_mid_y = (min(right_ys) + max(right_ys)).item() // 2
    left_mid_y = (min(left_ys) + max(left_ys)).item() // 2
    mid_y = max(right_mid_y, left_mid_y)
    return mid_y


def generate_findings_report(c_outputs, sep_lungs, effusions_data, mid_y):
    def upper_half_intersection(mask):
        return np.sum(mask * upper_mask[c_props[j].slice])

    def lower_half_intersection(mask):
        return np.sum(mask * lower_mask[c_props[j].slice])

    def upper_low_conf_intersection(mask):
        return np.sum(mask * np_low_conf_map[c_props[j].slice] * upper_mask[c_props[j].slice])

    def lower_low_conf_intersection(mask):
        return np.sum(mask * np_low_conf_map[c_props[j].slice] * lower_mask[c_props[j].slice])

    def write_changes(difference_mag: str, data: dict, c_region_area: float) -> str:
        if c_region_area == 0:
            return ""
        rel_diff_area = data['cum_area'] / c_region_area
        if data['num'] == 0 or rel_diff_area == 0 or (difference_mag == 'minor' and rel_diff_area < 0.05):
            return ""
        rel_low_conf_area = data['cum_low_conf_area'] / data['cum_area']
        if data['num'] == 1:
            amount = f'One {difference_mag} change.'
        elif 2 <= data['num'] <= 5:
            amount = f'A few {difference_mag} changes.'
        else:
            amount = f'Multiple {difference_mag} changes.'
        extent = f'{"Expansive" if rel_diff_area > 0.2 else "Localized"} changed area.'
        conf = f'{"High" if rel_low_conf_area < 0.33 else "Low"} confidence.'
        changes_text = f'\t\t\t{amount} {extent} {conf}\n'
        return changes_text

    print("Generating findings report")

    diff_map, low_conf_map = c_outputs
    right_lung, left_lung = sep_lungs
    if effusions_data:
        xs, ys, c_colors = effusions_data
    else:
        xs, ys, c_colors = [], [], []

    both_lungs = torch.logical_or(right_lung, left_lung)
    both_xs = torch.where(both_lungs)[1]
    mid_x = (min(both_xs) + max(both_xs)).item() // 2

    # right_ys = torch.where(right_lung)[0]
    # left_ys = torch.where(left_lung)[0]
    # right_mid_y = (min(right_ys) + max(right_ys)).item() // 2
    # left_mid_y = (min(left_ys) + max(left_ys)).item() // 2
    # mid_y = max(right_mid_y, left_mid_y)

    right_lung = torch.nn.functional.max_pool2d(right_lung.unsqueeze(0), kernel_size=35, stride=1, padding=17).squeeze().numpy()
    left_lung = torch.nn.functional.max_pool2d(left_lung.unsqueeze(0), kernel_size=35, stride=1, padding=17).squeeze().numpy()

    np_low_conf_map = low_conf_map.cpu().numpy()
    upper_mask = np.zeros_like(np_low_conf_map)
    upper_mask[..., :mid_y, :] = 1.
    lower_mask = np.zeros_like(np_low_conf_map)
    lower_mask[..., mid_y:, :] = 1.

    diff_map_abs = diff_map.abs()
    diff_map[diff_map_abs < 0.02] = 0.
    np_diff_map = diff_map.cpu().numpy()
    diff_map_pos_mask = diff_map > 0.
    diff_map_neg_mask = diff_map < 0.
    diff_map_neutral_mask = (np_diff_map == 0.)
    ccs_pos, ccs_pos_num = label(diff_map_pos_mask.cpu(), STRUCT)
    pos_regionprops = regionprops(ccs_pos, intensity_image=np_diff_map, extra_properties=(upper_low_conf_intersection, lower_low_conf_intersection, upper_half_intersection, lower_half_intersection))
    ccs_neg, ccs_neg_num = label(diff_map_neg_mask.cpu(), STRUCT)
    neg_regionprops = regionprops(ccs_neg, intensity_image=np_diff_map, extra_properties=(upper_low_conf_intersection, lower_low_conf_intersection, upper_half_intersection, lower_half_intersection))
    pos_regionprops_dict = {}
    neg_regionprops_dict = {}
    c_props = pos_regionprops
    for j in range(len(pos_regionprops)):
        pos_regionprops_dict[pos_regionprops[j].label] = (pos_regionprops[j].intensity_max, pos_regionprops[j].upper_half_intersection, pos_regionprops[j].lower_half_intersection, pos_regionprops[j].upper_low_conf_intersection, pos_regionprops[j].lower_low_conf_intersection)
    c_props = neg_regionprops
    for j in range(len(neg_regionprops)):
        neg_regionprops_dict[neg_regionprops[j].label] = (neg_regionprops[j].intensity_min, neg_regionprops[j].upper_half_intersection, neg_regionprops[j].lower_half_intersection, neg_regionprops[j].upper_low_conf_intersection, neg_regionprops[j].lower_low_conf_intersection)
    # pos_regionprops = {pos_regionprops[j].label: (pos_regionprops[j].intensity_max, pos_regionprops[j].upper_half_intersection, pos_regionprops[j].lower_half_intersection, pos_regionprops[j].upper_low_conf_intersection, pos_regionprops[j].lower_low_conf_intersection) for j in range(len(pos_regionprops))}
    # neg_regionprops = {neg_regionprops[j].label: (neg_regionprops[j].intensity_min, neg_regionprops[j].upper_half_intersection, neg_regionprops[j].lower_half_intersection, neg_regionprops[j].upper_low_conf_intersection, neg_regionprops[j].lower_low_conf_intersection) for j in range(len(neg_regionprops))}

    effs = {'right': None, 'left': None}
    len_effs = len(xs)
    for j in range(len_effs):
        if xs[j][0] <= mid_x:
            if effs['right'] is not None:
                raise 'Found 2 effusions for the right lung'
            effs['right'] = c_colors[j]
        else:
            if effs['left'] is not None:
                raise 'Found 2 effusions for the left lung'
            effs['left'] = c_colors[j]

    locs = {
        'Right': {'Upper': ((0, mid_y), right_lung, None), 'Lower': ((mid_y, diff_map.shape[-2]), right_lung, effs['right'])},
        'Left': {'Upper': ((0, mid_y), left_lung, None), 'Lower': ((mid_y, diff_map.shape[-2]), left_lung, effs['left'])}
    }

    report_text = 'Findings Report\n\n'
    for side in ['Right', 'Left']:
        report_text += f'Patient\'s {side} lung:\n'
        c_locs = locs[side]
        for lobe in ['Upper', 'Lower']:
            report_text += f'\t{lobe} lobe:\n'
            y_limits, c_lung, eff_type = c_locs[lobe]
            region_mask = np.zeros_like(np_low_conf_map)
            region_mask[..., y_limits[0]: y_limits[1], :] = 1.

            cur_region = c_lung * region_mask
            region_area = np.sum(cur_region).item()
            # low_conf_area = np.sum(cur_region * np_low_conf_map).item()
            cur_neutral_reg = cur_region * diff_map_neutral_mask
            neutral_area = np.sum(cur_neutral_reg).item()
            neutral_low_conf_area = np.sum(cur_neutral_reg * np_low_conf_map)
            if neutral_area > 0:
                neutral_area_ratio = neutral_low_conf_area / neutral_area
            else:
                neutral_area_ratio = 0

            pos_region = ccs_pos * cur_region
            pos_vals = np.unique(pos_region)
            if 0 in pos_vals:
                pos_vals = pos_vals[1:]

            neg_region = ccs_neg * cur_region
            neg_vals = np.unique(neg_region)
            if 0 in neg_vals:
                neg_vals = neg_vals[1:]

            text_to_add = ""

            for diff_type, props, vals in zip(['pos', 'neg'], [pos_regionprops_dict, neg_regionprops_dict], [pos_vals, neg_vals]):
                cur_data = {
                    'minor': {'num': 0, 'cum_area': 0, 'cum_low_conf_area': 0},
                    'mild': {'num': 0, 'cum_area': 0, 'cum_low_conf_area': 0},
                    'major': {'num': 0, 'cum_area': 0, 'cum_low_conf_area': 0},
                }
                for val in vals:
                    int_max_abs, upper_inter, lower_inter, upper_low_conf_inter, lower_low_conf_inter = props[val]
                    int_max_abs = abs(int_max_abs)
                    cur_inter, other_inter, cur_low_conf_inter = (upper_inter, lower_inter, upper_low_conf_inter) if lobe == 'Upper' else (lower_inter, upper_inter, lower_low_conf_inter)
                    rel_lobe_area = cur_inter / (cur_inter + other_inter)
                    if rel_lobe_area < 0.2:
                        continue
                    if int_max_abs < 0.06:
                        c_dict = cur_data['minor']
                    elif 0.06 <= int_max_abs < 0.11:
                        c_dict = cur_data['mild']
                    else:
                        c_dict = cur_data['major']
                    c_dict['num'] += 1
                    c_dict['cum_area'] += cur_inter
                    c_dict['cum_low_conf_area'] += cur_low_conf_inter

                major_text = write_changes('major', cur_data['major'], region_area)
                mild_text = write_changes('mild', cur_data['mild'], region_area)
                minor_text = write_changes('minor', cur_data['minor'], region_area)

                if any([major_text, mild_text, minor_text]):
                    diff_title = f'{"Intensified" if diff_type == "pos" else "Subsided"} changes inside the lung field:'
                    text_to_add += f'\t\t{diff_title}\n' \
                                   f'{major_text}{mild_text}{minor_text}\n'

            if text_to_add:
                report_text += text_to_add
            else:
                report_text += "\t\tNo meaningful changes detected inside the lung field.\n\n"

            if neutral_area_ratio > 0.35:
                report_text += "\t\tLow confidence in unchanged region.\n\n"

            if eff_type:
                report_text += f"\t{'Increase' if eff_type == 'orange' else 'Decrease'} in lung elevation detected.\n\n"

    return report_text


if __name__ == '__main__':
    with torch.no_grad():
        # model_path1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id9_Epoch2_Longitudinal_FullImg_SmallNet_DiffEncs_DiffGT_1Channel_single128_Sched_Decoder6_Eff_ViT_MaskToken_L1L2_GN.pt.pt'
        # model1 = LongitudinalMIMModel(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # # model_path2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id14_FTid9_Epoch2_Longitudinal_FullImg_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt.pt'
        # # model2 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id15_FTid9_Epoch3_Longitudinal_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model2 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path3 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id16_FTid9_Epoch1_Longitudinal_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model3 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path4 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id16_FTid9_Epoch2_Longitudinal_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model4 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path5 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id16_FTid9_Epoch3_Longitudinal_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model5 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path6 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id18_FTid16_Epoch2_Longitudinal_Devices2_Dropout_ExtendedConvNet_DiffEncs_DiffGT_BothAbs_NoDiffAbProb_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model6 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path7 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id20_FTid16_Epoch4_Longitudinal_Effusion_Devices2_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model7 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path8 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id21_FTid9_Epoch3_Longitudinal_Transformer_Effusion_Devices2_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model8 = LongitudinalMIMModelBigTransformer(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path9 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id22_Retrain_Epoch4_Longitudinal_Intrapulmonary_Changes_Devices2_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model9 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path10 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id23_Retrain_Epoch4_Longitudinal_Intrapulmonary_ContourDeform_Devices2_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model10 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path11 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id24_Finetune_id23_Epoch1_Longitudinal_LessEntities_Intrapulmonary_ContourDeform_Devices2_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model11 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path12 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id28_Epoch2_Longitudinal_Overlay_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model12 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path13 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id27_Epoch2_Longitudinal_Overlay_Inpaint_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model13 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path14 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id29_Epoch3_Longitudinal_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model14 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        # model_path15 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id31_Epoch3_Longitudinal_DeviceInvariant_DRRs_Overlay_Inpaint_MoreData_MoreEntities_NoUnrelated_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model15 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()
        model_path16 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id35_Epoch10_Longitudinal_MoreFT_MassesRotationInvariance_TrainSet_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        model16 = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).cuda()

        model_paths = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, model_path16]
        # model_paths = [model_path1, model_path2, model_path3, model_path4]
        models = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, model16]
        # models = [model1, model2, model3, model4]

        # model_paths_mask = [0, 0, 0, 0, model_path5, 0, 0, 0]
        # model_paths = [model_path1, model_path2, model_path3, model_path4]
        # models_mask = [0, 0, 0, 0, model5, 0, 0, 0]
        # models = [model1, model2, model3, model4]

        for model_path, model in zip(model_paths, models):
            if model == 0:
                continue
            checkpoint_dict = torch.load(model_path)
            model.load_state_dict(checkpoint_dict['model_dict'], strict=False)
            # model.eval()  # TODO: NOTE

        # for model_path, model in zip(model_paths_mask, models_mask):
        #     if model == 0:
        #         continue
        #     checkpoint_dict = torch.load(model_path)
        #     model.load_state_dict(checkpoint_dict['model_dict'], strict=False)

        # ds = LongitudinalMIMDataset(['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR/train', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14/images',
        #                             '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/images'])
        # idx = random.randint(0, len(ds))
        # bl, fu, bl_seg, fu_seg = ds[idx]

        # differential_grad = colors.LinearSegmentedColormap.from_list('differential_grad', (
        #     # Edit this gradient at https://eltos.github.io/gradient/#0:440A57-12.5:5628A5-25:1256D4-37.5:119AB9-49:07C8C3-50:FFFFFF-51:00A64E-62.5:3CC647-75:60F132-87.5:A8FF3E-100:ECF800
        #     (0.000, (0.267, 0.039, 0.341)),
        #     (0.125, (0.337, 0.157, 0.647)),
        #     (0.250, (0.071, 0.337, 0.831)),
        #     (0.375, (0.067, 0.604, 0.725)),
        #     (0.490, (0.027, 0.784, 0.765)),
        #     (0.500, (1.000, 1.000, 1.000)),
        #     (0.510, (0.000, 0.651, 0.306)),
        #     (0.625, (0.235, 0.776, 0.278)),
        #     (0.750, (0.376, 0.945, 0.196)),
        #     (0.875, (0.659, 1.000, 0.243)),
        #     (1.000, (0.925, 0.973, 0.000))))

        # differential_grad = colors.LinearSegmentedColormap.from_list('differential_grad', (
        #     # Edit this gradient at https://eltos.github.io/gradient/#0:440A57-12.5:5628A5-25:1256D4-37.5:119AB9-49:07C8C3-50:FFFFFF-51:00A64E-62.5:3CC647-75:60F132-87.5:A8FF3E-100:ECF800
        #     (0.000, (0.016, 1.000, 0.000)),
        #     (0.175, (0.302, 1.000, 0.290)),
        #     (0.350, (0.639, 1.000, 0.631)),
        #     (0.446, (0.800, 1.000, 0.792)),
        #     (0.500, (1.000, 1.000, 1.000)),
        #     (0.554, (1.000, 0.816, 0.816)),
        #     (0.675, (1.000, 0.584, 0.588)),
        #     (0.825, (1.000, 0.322, 0.310)),
        #     (1.000, (1.000, 0.000, 0.008))
        # ))

        differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
            # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
            (0.000, (0.235, 1.000, 0.239)),
            (0.400, (0.000, 1.000, 0.702)),
            (0.500, (1.000, 0.988, 0.988)),
            (0.600, (1.000, 0.604, 0.000)),
            (1.000, (0.682, 0.000, 0.000))))

        resize_tf = v2.Resize((512, 512))
        random_perspective_tf = RandomAffineWithMaskTransform()
        random_bspline_tf = RandomBsplineAndSimilarityWithMaskTransform()
        random_geometric_tf = v2.RandomChoice([random_perspective_tf, random_bspline_tf], p=[0.2, 0.8])
        # random_intensity_tf = RandomIntensityTransform(clahe_p=0.25, clahe_clip_limit=(0.75, 2.5), blur_p=0., jitter_p=0.35)
        random_intensity_tf = RandomIntensityTransform(clahe_p=0.25, clahe_clip_limit=(0.75, 1.25), blur_p=0., jitter_p=0.35)
        rescale_tf = RescaleValuesTransform()
        clahe_tf = RandomIntensityTransform(clahe_p=0., clahe_clip_limit=2.2, blur_p=0.0, jitter_p=0.)
        mask_crop_tf = CropResizeWithMaskTransform()

        found = []

        # data_folder_bl = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14'
        # data_folder_fu = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14'
        # data_folder_bl = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal'
        # data_folder_fu = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_sigal'
        data_folder_bl = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_nitzan'
        data_folder_fu = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_nitzan'
        # data_folder_bl = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/BennyCases/Neonatal'
        # data_folder_fu = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/BennyCases/Neonatal'

        # data_folder_bl = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14'
        # data_folder_fu = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/cases_nitzan'

        healthy_scan_name = '00000067_000.nii.gz'
        healthy_scan = torch.tensor(nib.load('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14' + f'/images/{healthy_scan_name}').get_fdata().T[None, ...])
        healthy_scan_seg_name = '00000067_000_seg.nii.gz'
        healthy_scan_seg = torch.tensor(nib.load('/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ChestX-ray14' + f'/images_segs/{healthy_scan_seg_name}').get_fdata().T[None, ...])
        # healthy_scan = rescale_tf(mask_crop_tf(torch.cat([healthy_scan, healthy_scan_seg], dim=0))[0])
        #
        # fu_names = ['00000001_002.nii.gz', '00000003_001.nii.gz', '00000003_007.nii.gz', '00000005_001.nii.gz', '00000005_005.nii.gz',
        #             '00000008_001.nii.gz', '00000011_003.nii.gz', '00000011_008.nii.gz', '00000013_001.nii.gz', '00000013_004.nii.gz', '00000013_016.nii.gz', '00000013_019.nii.gz',
        #             '00000013_019.nii.gz', '00000013_032.nii.gz', '00000013_046.nii.gz', '00000017_001.nii.gz', '00000020_002.nii.gz', '00000021_001.nii.gz',
        #             '00000022_001.nii.gz', '00000023_004.nii.gz', '00000030_001.nii.gz', '00000032_001.nii.gz', '00000032_002.nii.gz',
        #             '00000032_022.nii.gz', '00000032_028.nii.gz', '00000032_048.nii.gz', '00000032_060.nii.gz', '00000034_001.nii.gz', '00000035_001.nii.gz',
        #             '00000038_006.nii.gz', '00000039_003.nii.gz', '00000040_003.nii.gz', '00000041_002.nii.gz', '00000041_006.nii.gz',
        #             '00000042_008.nii.gz', '00000044_001.nii.gz', '00000047_004.nii.gz', '00000049_002.nii.gz', '00000050_003.nii.gz',
        #             '00000052_001.nii.gz', '00000053_001.nii.gz', '00000054_009.nii.gz', '00000056_001.nii.gz', '00000057_003.nii.gz',
        #             '00000059_001.nii.gz', '00000061_005.nii.gz', '00000061_011.nii.gz', '00000061_025.nii.gz', '00000067_002.nii.gz',
        #             '00000071_007.nii.gz', '00000073_001.nii.gz', '00000073_009.nii.gz', '00000075_001.nii.gz', '00000078_002.nii.gz',
        #             '00000080_002.nii.gz', '00000080_005.nii.gz', '00000084_001.nii.gz']

        # bl_names = ['00000013_019.nii.gz']
        # fu_names = ['00000013_006.nii.gz']

        # fu_names = [name for pair in zip(bl_names, fu_names) for name in pair]
        # fu_names.remove(healthy_scan_name)
        # # fu_names.remove('00000067_002.nii.gz')
        # bl_names = [healthy_scan_name for _ in range(len(fu_names))]

        # bl_names = ['p1_1.nii.gz', 'p1_2.nii.gz', 'p1_1.nii.gz', 'p2_1.nii.gz', 'p2_2.nii.gz', 'p2_3.nii.gz', 'p2_1.nii.gz', 'p2_1.nii.gz', 'p2_2.nii.gz', 'p3_1.nii.gz', 'p4_1.nii.gz', 'p4_2.nii.gz', 'p4_1.nii.gz']
        # fu_names = ['p1_2.nii.gz', 'p1_3.nii.gz', 'p1_3.nii.gz', 'p2_2.nii.gz', 'p2_3.nii.gz', 'p2_4.nii.gz', 'p2_3.nii.gz', 'p2_4.nii.gz', 'p2_4.nii.gz', 'p3_2.nii.gz', 'p4_2.nii.gz', 'p4_3.nii.gz', 'p4_3.nii.gz']

        # bl_names = [f'{i}A.nii.gz' for i in range(1, 25)]
        # fu_names = [f'{i}B.nii.gz' for i in range(1, 25)]
        #
        # bl_names = ['10A.nii.gz']
        # fu_names = ['10B.nii.gz']

        # bl_names = [f'{i}A.nii.gz' for i in range(25, 38)] + ['26B.nii.gz', '36B.nii.gz', '36C.nii.gz']
        # fu_names = [f'{i}B.nii.gz' for i in range(25, 38)] + ['26C.nii.gz', '36C.nii.gz', '36D.nii.gz']

        bl_names = []
        fu_names = []

        # bl_names.extend(['A1.nii.gz', 'A2.nii.gz', 'A3.nii.gz', 'A4.nii.gz', 'A5.nii.gz', 'A6.nii.gz', 'B1.nii.gz', 'B2.nii.gz', 'B3.nii.gz',
        #             'C1.nii.gz', 'C2.nii.gz', 'C3.nii.gz', 'D1.nii.gz', 'D2.nii.gz', 'D3.nii.gz', 'D4.nii.gz', 'E1.nii.gz', 'E2.nii.gz',
        #             'F1.nii.gz', 'F2.nii.gz', 'F3.nii.gz', 'F4.nii.gz', 'G1.nii.gz', 'H1.nii.gz', 'H2.nii.gz', 'I1.nii.gz', 'I2.nii.gz',
        #             'I3.nii.gz', 'J1.nii.gz', 'J2.nii.gz', 'J3.nii.gz', 'J4.nii.gz'])
        # fu_names.extend(['A2.nii.gz', 'A3.nii.gz', 'A4.nii.gz', 'A5.nii.gz', 'A6.nii.gz', 'A7.nii.gz', 'B2.nii.gz', 'B3.nii.gz', 'B4.nii.gz',
        #             'C2.nii.gz', 'C3.nii.gz', 'C4.nii.gz', 'D2.nii.gz', 'D3.nii.gz', 'D4.nii.gz', 'D5.nii.gz', 'E2.nii.gz', 'E3.nii.gz',
        #             'F2.nii.gz', 'F3.nii.gz', 'F4.nii.gz', 'F5.nii.gz', 'G2.nii.gz', 'H2.nii.gz', 'H3.nii.gz', 'I2.nii.gz', 'I3.nii.gz',
        #             'I4.nii.gz', 'J2.nii.gz', 'J3.nii.gz', 'J4.nii.gz', 'J5.nii.gz'])
        #
        # bl_names.extend(['AA1.nii.gz', 'AA2.nii.gz', 'BB1.nii.gz', 'CC1.nii.gz', 'DD1.nii.gz', 'DD2.nii.gz'])
        # fu_names.extend(['AA2.nii.gz', 'AA3.nii.gz', 'BB2.nii.gz', 'CC2.nii.gz', 'DD2.nii.gz', 'DD3.nii.gz'])

        bl_names.extend([f'C_{c}1.nii.gz' for c in map(chr, range(ord('A'), ord('T')+1))])
        fu_names.extend([f'C_{c}2.nii.gz' for c in map(chr, range(ord('A'), ord('T')+1))])

        bl_names.extend(['C_B2.nii.gz'])
        fu_names.extend(['C_B1.nii.gz'])

        # bl_names = ['D2.nii.gz']
        # fu_names = ['D1.nii.gz']

        # fu_names = sorted(list(set(bl_names).union(set(fu_names))))
        # bl_names = ['00000067_002.nii.gz' for _ in fu_names]

        # bl_names = ['B2.nii.gz']
        # fu_names = ['B3.nii.gz']

        # bl_names = sorted(os.listdir(f'{data_folder_bl}/images'))[:-1]
        # fu_names = sorted(os.listdir(f'{data_folder_fu}/images'))[1:]

        folder_suffix = ''

        for bl_name, fu_name in zip(bl_names, fu_names):
            # assert bl_name != fu_name
            # assert bl_name.split('_')[0] == fu_name.split('_')[0]
            bl_path = f'{data_folder_bl}/images{folder_suffix}/{bl_name}'
            fu_path = f'{data_folder_fu}/images{folder_suffix}/{fu_name}'

            if folder_suffix == '' and not os.path.exists(bl_path):
                print(f"Path {bl_path} does not exists. Trying with another suffix...")
                folder_suffix = '2'
                bl_path = f'{data_folder_bl}/images{folder_suffix}/{bl_name}'
                fu_path = f'{data_folder_fu}/images{folder_suffix}/{fu_name}'

            assert os.path.exists(bl_path)
            assert os.path.exists(fu_path)

        for k, (c_bl_name, c_fu_name) in enumerate(zip(bl_names, fu_names)):
            # pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/MonteCarloDropout_TTA_with_effs/geo_int_bl_int_fu/{c_bl_name.split(".")[0]}_{c_fu_name.split(".")[0]}'
            # pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/MaskGeneration/{c_bl_name.split(".")[0]}_{c_fu_name.split(".")[0]}'
            # pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/sigal_cases_tests/MonteCarloDropout_TTA/{c_bl_name.split(".")[0]}_{c_fu_name.split(".")[0]}'
            pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/nitzan_cases_tests/MonteCarloDropout_TTA/{c_bl_name.split(".")[0]}_{c_fu_name.split(".")[0]}'
            # pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/MonteCarloDropout_TTA/{c_bl_name.split(".")[0]}_{c_fu_name.split(".")[0]}'
            os.makedirs(pair_dir_path, exist_ok=True)

            # bls = [c_bl_name, c_fu_name]
            # fus = [c_fu_name, c_bl_name]

            bls = [c_bl_name]
            fus = [c_fu_name]

            # if k % 2 == 0:
            #     cur_pair_preds = []
            #     pair_names = []
            # pair_names.append(c_fu_name)

            for bl_name, fu_name in zip(bls, fus):
                print(f'Working on bl: {bl_name}, fu: {fu_name}')

                c_dir = f'{pair_dir_path}/Prior_{bl_name.split(".")[0]}_Current_{fu_name.split(".")[0]}'
                os.makedirs(c_dir, exist_ok=True)

                # data_folder = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/VinDrCXR'
                # bl_name = '00000023_000.nii.gz'
                # bl_name = '002a34c58c5b758217ed1f584ccbcfe9.nii.gz'
                bl_seg_name = f'{bl_name.split(".")[0]}_seg.nii.gz'
                # fu_name = '00000023_002.nii.gz'
                # fu_name = '004f33259ee4aef671c2b95d54e4be68.nii.gz'
                fu_seg_name = f'{fu_name.split(".")[0]}_seg.nii.gz'
                bl_path = f'{data_folder_bl}/images{folder_suffix}/{bl_name}'
                # bl_path = f'{data_folder}/test/{bl_name}'
                bl_seg_path = f'{data_folder_bl}/images{folder_suffix}_segs/{bl_seg_name}'
                # bl_seg_path = f'{data_folder}/test_segs/{bl_seg_name}'
                fu_path = f'{data_folder_fu}/images{folder_suffix}/{fu_name}'
                # fu_path = f'{data_folder}/test/{fu_name}'
                fu_seg_path = f'{data_folder_fu}/images{folder_suffix}_segs/{fu_seg_name}'
                # fu_seg_path = f'{data_folder}/test_segs/{fu_seg_name}'

                bl = torch.tensor(nib.load(bl_path).get_fdata().T[None, ...])
                bl_seg = torch.tensor(nib.load(bl_seg_path).get_fdata().T[None, ...])
                fu = torch.tensor(nib.load(fu_path).get_fdata().T[None, ...])
                fu_seg = torch.tensor(nib.load(fu_seg_path).get_fdata().T[None, ...])

                bl_seg = bl_seg.view_as(bl)
                fu_seg = fu_seg.view_as(fu)

                # fu_msk = generate_mask(healthy_scan, healthy_scan_seg, fu, fu_seg, bl, bl_seg, save_nif=False, ret_mask=True)
                # bl_msk = generate_mask(healthy_scan, healthy_scan_seg, bl, bl_seg, fu, fu_seg, save_nif=False, ret_mask=True)
                # c_effusions_data, split_lungs = detect_effusions(bl_msk, fu_msk, ret_split_fu_masks=True)
                # c_mid_y = calc_mid_y(split_lungs)

                # regular_pred(bl, bl_seg, fu, fu_seg)
                # TTA_pred(bl, bl_seg, fu, fu_seg)
                # outputs = Monte_Carlo_pred(bl, bl_seg, fu, fu_seg, effusions_data=c_effusions_data, mid_y=None)
                mean_outputs_with_prior = Monte_Carlo_pred(bl, bl_seg, fu, fu_seg, effusions_data=None, mid_y=None, plots_suffix='_real')[0][0]
                # mean_outputs_with_healthy = Monte_Carlo_pred(healthy_scan, healthy_scan_seg, fu, fu_seg, effusions_data=None, mid_y=None, plots_suffix='_healthy')[0][0]
                # generate_mask(bl, bl_seg, fu, fu_seg)

                ###
                # Opacity Classification (!!!)
                ###
                # disappeared_opacities = (mean_outputs_with_healthy < 0.015) & ((mean_outputs_with_prior - mean_outputs_with_healthy) < -0.035)
                # appeared_opacities = ((mean_outputs_with_prior > 0.05) | (mean_outputs_with_healthy > 0.05)) & ((mean_outputs_with_prior - mean_outputs_with_healthy).abs() < 0.035)
                # # persisting_opacities = (mean_outputs_with_healthy.abs() > 0.05) & (~appeared_opacities) & (~disappeared_opacities)
                # persisting_opacities = (mean_outputs_with_healthy > 0.05) & (~appeared_opacities) & (~disappeared_opacities)
                #
                # cur_dir = c_dir + f'/model5'
                #
                # cur_cropped_fu = rescale_tf(mask_crop_tf(torch.cat([fu, fu_seg], dim=0))[0])
                # plt.imshow(cur_cropped_fu.squeeze().cpu(), cmap='gray')
                # persisting_opacities = persisting_opacities.float().squeeze().cpu()
                # appeared_opacities = appeared_opacities.float().squeeze().cpu()
                # disappeared_opacities = disappeared_opacities.float().squeeze().cpu()
                # plt.imshow(persisting_opacities, cmap='Wistia', alpha=persisting_opacities * 0.3)
                # plt.imshow(appeared_opacities, cmap='rainbow', alpha=appeared_opacities * 0.3)
                # plt.imshow(disappeared_opacities, cmap='brg', alpha=disappeared_opacities * 0.3)
                # plt.axis('off')
                # plt.title('Opacity classification overlaid on Current')
                # plt.tight_layout()
                # plt.savefig(f'{cur_dir}/opacity_classification.png')
                # plt.clf()
                # plt.close()

                """
                plt.imshow(cropped_fu.squeeze().cpu(), cmap='gray')
                imm3 = plt.imshow(outs_mean.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
                cbar3 = plt.colorbar(imm3, fraction=0.05, pad=0.04)
                low_confidence = (outs_uncertainty > 1).float().squeeze().cpu()
                plt.imshow(low_confidence, cmap='seismic', alpha=low_confidence)
                plt.axis('off')
                plt.title("Low Confidence overlaid on Output")
                plt.tight_layout()
                plt.savefig(f'{cur_dir}/mean_outputs_with_low_confidence_overlaid{plots_suffix}.png')
                plt.cla()
                plt.clf()
                plt.close()
                """

                # if len(outputs) > 1:
                #     raise 'Trying to generate findings reports using more than one model.'
                #
                # model_outputs = outputs[0]
                # report = generate_findings_report(model_outputs, split_lungs, c_effusions_data, c_mid_y)
                #
                # with open(f'{c_dir}/findings_report.txt', 'w') as f:
                #     f.write(report)

                # if fu_name == '00000067_000.nii.gz':
                #     for n in range(len(preds)):
                #         preds[n][preds[n] > 0.] = 0.
                #     cur_pair_preds.append(preds)

            # if k % 2 == 1:
            #     print(f'Indirectly comparing pair {" & ".join(pair_names)}')
            #     indirect_pair_dir_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/plots/Longitudinal_MIM/tests/MonteCarloDropout_TTA/comparison_with_healthy/comparison_of_{pair_names[0].split(".")[0]}_{pair_names[1].split(".")[0]}_through_healthy_00000067_002'
            #     path_im1_fu = f'{indirect_pair_dir_path}/BL_{pair_names[1].split(".")[0]}_FU_{pair_names[0].split(".")[0]}'
            #     path_im2_fu = f'{indirect_pair_dir_path}/BL_{pair_names[0].split(".")[0]}_FU_{pair_names[1].split(".")[0]}'
            #     p1m4 = path_im1_fu + '/model4'
            #     p2m4 = path_im2_fu + '/model4'
            #
            #     im1_pred_model4 = -cur_pair_preds[0][0] + cur_pair_preds[1][0]
            #     im2_pred_model4 = -im1_pred_model4
            #
            #     for p, pred in zip([p1m4, p2m4], [im1_pred_model4, im2_pred_model4]):
            #         os.makedirs(p, exist_ok=True)
            #         plt.imshow(healthy_scan.squeeze(), cmap='gray')
            #         alphas = generate_alpha_map(pred)
            #         imm = plt.imshow(pred.squeeze(), alpha=alphas)
            #         cbar = plt.colorbar(imm, fraction=0.05, pad=0.04)
            #         plt.title('Mean outputs - Comparison through healthy')
            #         plt.axis('off')
            #         plt.tight_layout()
            #         plt.savefig(f'{p}/mean_outputs.png')
            #         plt.cla()
            #         plt.clf()
            #         plt.close()

        print(found)








