import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torchvision.transforms.v2 as v2
from models import *
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import colors
from constants import *
import kornia
from torchvision.transforms.v2.functional import adjust_sharpness
from scipy.ndimage import label


# One time use function
def organize_ICU_cases_into_pairs():
    ...


def remove_small_ccs(im, min_count=60):
    ccs, ccs_num = label(im.cpu().squeeze(), struct)
    ccs = torch.from_numpy(ccs).to(im.device)
    unique_vals, unique_counts = torch.unique(ccs, return_counts=True)
    unique_vals = unique_vals[unique_counts > min_count]
    ccs = ccs[None, None, ...]
    ccs_indic = torch.isin(ccs, unique_vals)
    new_im = im * ccs_indic
    new_ccs = ccs * ccs_indic
    ccs_num = torch.numel(unique_vals) - 1
    return new_im.squeeze(), new_ccs, ccs_num


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val

    # alphas_map = 1.027689 + (-0.00003443215 - 1.027689) / (1 + (alphas_map / 0.07866591) ** 279.162) ** 0.005007674
    # alphas_map = 0.981258 + (-0.008780828 - 0.981258) / (1 + (alphas_map / 0.1749422) ** 3.518648)
    # alphas_map = alphas_map.clamp(0., 1.)

    return alphas_map


def plot_pair(prior, current, out_dir):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(prior.squeeze().cpu(), cmap='gray')
    ax[0].set_title('Prior')
    ax[1].imshow(current.squeeze().cpu(), cmap='gray')
    ax[1].set_title('Current')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'{out_dir}/inputs.png')
    ax[0].clear()
    ax[1].clear()
    plt.cla()
    plt.clf()
    plt.close()


def plot_output_on_current(current, output, out_dir, suffix=''):
    current = current.squeeze().cpu()
    output = output.squeeze().cpu()

    plt.imshow(current, cmap='gray')

    alphas = generate_alpha_map(output)
    divnorm = colors.TwoSlopeNorm(vmin=min(torch.min(output).item(), -0.01), vcenter=0., vmax=max(torch.max(output).item(), 0.01))

    imm1 = plt.imshow(output, alpha=alphas, cmap=differential_grad, norm=divnorm)
    cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04)

    plt.title(f'Model output')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'{out_dir}/model_output{suffix}.png')

    plt.clf()
    plt.cla()
    plt.close()


@torch.no_grad()
def pred(prior, current, model):
    prior = prior.unsqueeze(0).to(DEVICE)
    current = current.unsqueeze(0).to(DEVICE)

    out = model(prior, current).squeeze().cpu()

    return out


def preprocess(img: torch.Tensor, boundary_seg: torch.Tensor, crop_pad_val=15, crop_seg=False):
    assert img.shape == boundary_seg.shape, (img.shape, boundary_seg.shape)

    boundary_seg_coords = boundary_seg.nonzero().T
    x_min, x_max = torch.min(boundary_seg_coords[-2]), torch.max(boundary_seg_coords[-2])
    y_min, y_max = torch.min(boundary_seg_coords[-1]), torch.max(boundary_seg_coords[-1])

    x_min = max(x_min.item() - crop_pad_val, 0)
    y_min = max(y_min.item() - crop_pad_val, 0)
    x_max = min(x_max.item() + crop_pad_val, img.shape[-2] - 1)
    y_max = min(y_max.item() + crop_pad_val, img.shape[-1] - 1)

    img = img[..., x_min: x_max, y_min: y_max]
    img = resize(img[None, ...])

    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))

    # Best: No clahe. sharpness = 4. Normal resize

    # img = kornia.enhance.equalize_clahe(img, clip_limit=0.9, grid_size=(8, 8))
    img = adjust_sharpness(img, sharpness_factor=4.)

    # img = adjust_sharpness(img, sharpness_factor=7.)

    img = torch.clamp(img, 0., 1.)

    if crop_seg:
        boundary_seg = boundary_seg[..., x_min: x_max, y_min: y_max]
        boundary_seg = resize(boundary_seg[None, ...])

        return img, boundary_seg

    return img


def postprocess(out):
    out[torch.logical_or(out > 0.75, out < -0.75)] = 0.

    for s in [-0.15, -0.12, -0.09, -0.06, -0.03, -0, 0.03, 0.06, 0.09, 0.12]:
        c_map = torch.logical_and(out > s, out < s + 0.05)
        if torch.sum(c_map) > 512 * 512 * 0.25:
            out[c_map] = 0

    bin_out = (out > 0).float() - (out < 0).float()
    bin_out, _, __ = remove_small_ccs(bin_out, min_count=250)
    out[bin_out == 0] = 0

    # Best: < 0.02, > -0.0075

    # out[out.abs() < 0.02] = 0
    out[torch.logical_and(out < 0.02, out > -0.0075)] = 0

    # out = out.numpy()
    # out_binary = out.copy()
    # out_binary[out_binary > 0] = 1
    # out_binary[out_binary < 0] = -1
    # ccs, ccs_num = label(out_binary)
    # vals, counts = np.unique(ccs, return_counts=True)

    # out_pos = out * (out > 0)
    # out_neg = out * (out < 0)
    #
    # out_pos = torch.exp(out_pos)
    # out_pos = (out_pos - out_pos.min()) / (out_pos.max() - out_pos.min())
    #
    # out_neg = torch.exp(-out_neg)
    # out_neg = (out_neg - out_neg.min()) / (out_neg.max() - out_neg.min())
    # out_neg = -out_neg
    #
    # out[out > 0] = out_pos[out > 0]
    # out[out < 0] = out_neg[out < 0]

    return out


def main():
    with torch.no_grad():
        model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id45_Epoch3_Longitudinal_AllEntities_DEVICES_FT_Cons_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id42_Epoch15_Longitudinal_AllEntities_DEVICES_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id38_Epoch10_Longitudinal_ConsolidationRotationInvariance_MoreEpochs_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id39_Epoch15_Longitudinal_PleuralEffusionRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id40_Epoch12_Longitudinal_PneumothoraxRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id41_Epoch12_Longitudinal_FluidOverloadRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        model = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).to(DEVICE)

        checkpoint_dict = torch.load(model_path)
        model.load_state_dict(checkpoint_dict['model_dict'], strict=True)
        model.eval()

        preds_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ICU_cases/test_predictions/all_entities_model'
        pairs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ICU_cases/test_pairs'
        segs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ICU_cases/segs'

        for pair in sorted(os.listdir(pairs_dir)):
            print(f'Working on {pair}')

            pair_d = f'{pairs_dir}/{pair}'
            pred_d = f'{preds_dir}/{pair}'
            os.makedirs(pred_d, exist_ok=True)

            prior_n, current_n = sorted([n for n in os.listdir(pair_d)])
            prior_p, current_p = f'{pair_d}/{prior_n}', f'{pair_d}/{current_n}'

            prior_n = prior_n[:-7]
            current_n = current_n[:-7]

            prior_seg_p = f'{segs_dir}/{prior_n}_seg.nii.gz'
            current_seg_p = f'{segs_dir}/{current_n}_seg.nii.gz'

            prior_nif = nib.load(prior_p)
            aff = prior_nif.affine
            prior = torch.tensor(prior_nif.get_fdata().T, dtype=torch.float32)
            current = torch.tensor(nib.load(current_p).get_fdata().T, dtype=torch.float32)

            prior_seg = torch.tensor(nib.load(prior_seg_p).get_fdata().T)
            current_seg = torch.tensor(nib.load(current_seg_p).get_fdata().T)

            prior = preprocess(prior, prior_seg, crop_pad_val=15, crop_seg=False)
            current = preprocess(current, current_seg, crop_pad_val=15, crop_seg=False)

            plot_pair(prior, current, pred_d)

            output = pred(prior, current, model)

            output = postprocess(output)

            output_nif = nib.Nifti1Image(output.numpy().T, aff)
            nib.save(output_nif, f'{pred_d}/output.nii.gz')

            plot_output_on_current(current, output, pred_d)
            plot_output_on_current(current, (output > 0).float() - (output < 0).float(), pred_d, suffix='_bin')


if __name__ == '__main__':
    resize = v2.Resize((512, 512))
    struct = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))
    main()
