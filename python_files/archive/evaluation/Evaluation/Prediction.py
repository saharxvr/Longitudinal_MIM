import os
import sys
import argparse
import re

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
python_files_dir = os.path.normpath(os.path.join(current_dir, '../../..'))
if python_files_dir not in sys.path:
    sys.path.append(python_files_dir)

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


def preprocess_no_seg(img: torch.Tensor):
    img = resize(img[None, ...])
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    img = adjust_sharpness(img, sharpness_factor=4.)
    img = torch.clamp(img, 0., 1.)
    return img


def _pair_sort_key(path_or_name: str):
    name = os.path.basename(path_or_name)
    nums = re.findall(r'\d+', name)
    if nums:
        return int(nums[-1])
    return float('inf')


def collect_pair_dirs(pairs_roots):
    pair_dirs = []
    for root in pairs_roots:
        if not os.path.isdir(root):
            print(f'Skipping missing pairs root: {root}')
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                pair_dirs.append(p)

    pair_dirs = sorted(pair_dirs, key=_pair_sort_key)
    return pair_dirs


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


ROI_NAMES = [
    "full_image", "lungs", "lungs_heart", "lungs_mediastinum", "full_thorax",
    "lungs_margin5", "lungs_med_margin5", "lungs_convex_hull",
]


def load_roi_mask(roi_masks_dir, roi_name, pair_name):
    """Load a precomputed ROI mask. Returns None for full_image."""
    if roi_name == "full_image" or roi_masks_dir is None:
        return None
    mask_path = os.path.join(roi_masks_dir, roi_name, pair_name, "mask.nii.gz")
    if not os.path.exists(mask_path):
        return None
    arr = nib.load(mask_path).get_fdata()
    arr = np.squeeze(arr)
    return (arr > 0).astype(np.float32)


def apply_roi_mask(img_tensor, mask_np):
    """Multiply a preprocessed [1, 1, H, W] tensor by an ROI mask (resized to match)."""
    if mask_np is None:
        return img_tensor
    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    mask_t = resize(mask_t)
    mask_t = (mask_t > 0.5).float().to(img_tensor.device)
    return img_tensor * mask_t


def crop_to_roi(img_raw, mask_np, crop_pad_val=15):
    """Crop [H,W] image to ROI bbox + padding, resize to 512x512, normalize & sharpen.
    Returns [1, 512, 512] tensor (same shape as preprocess_no_seg)."""
    if mask_np is None:
        return preprocess_no_seg(img_raw)
    mask_t = torch.from_numpy(mask_np).float()
    coords = mask_t.nonzero(as_tuple=False)
    if len(coords) == 0:
        return preprocess_no_seg(img_raw)
    x_min, x_max = coords[:, 0].min().item(), coords[:, 0].max().item()
    y_min, y_max = coords[:, 1].min().item(), coords[:, 1].max().item()
    # Pad & clamp to image bounds
    x_min = max(x_min - crop_pad_val, 0)
    y_min = max(y_min - crop_pad_val, 0)
    x_max = min(x_max + crop_pad_val, img_raw.shape[-2] - 1)
    y_max = min(y_max + crop_pad_val, img_raw.shape[-1] - 1)
    cropped = img_raw[..., x_min:x_max, y_min:y_max]
    cropped = resize(cropped[None, ...])  # [H',W'] -> [1,512,512]
    cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min() + 1e-8)
    cropped = adjust_sharpness(cropped, sharpness_factor=4.)
    cropped = torch.clamp(cropped, 0., 1.)
    return cropped


def main(use_segs=False, model_path=None, preds_dir=None, pairs_roots=None, segs_dir=None,
         roi_masks_dir=None, roi_names=None, roi_mode='mask'):
    with torch.no_grad():
        if model_path is None:
            model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id45_Epoch3_Longitudinal_AllEntities_DEVICES_FT_Cons_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id42_Epoch15_Longitudinal_AllEntities_DEVICES_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id38_Epoch10_Longitudinal_ConsolidationRotationInvariance_MoreEpochs_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id39_Epoch15_Longitudinal_PleuralEffusionRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id40_Epoch12_Longitudinal_PneumothoraxRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        # model_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id41_Epoch12_Longitudinal_FluidOverloadRotationInvariance_Sharpen_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt'
        model = LongitudinalMIMModelBig(dec=6, use_pos_embed=USE_POS_EMBED, patch_size=MASK_PATCH_SIZE).to(DEVICE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

        checkpoint_dict = torch.load(model_path)
        model.load_state_dict(checkpoint_dict['model_dict'], strict=True)
        model.eval()

        if preds_dir is None:
            preds_dir = os.path.normpath(os.path.join(current_dir, '../../../annotation tool/predictions'))
        if pairs_roots is None:
            pairs_roots = [
                os.path.normpath(os.path.join(current_dir, '../../../annotation tool/Pairs5')),
                os.path.normpath(os.path.join(current_dir, '../../../annotation tool/Pairs6')),
                os.path.normpath(os.path.join(current_dir, '../../../annotation tool/Pairs7')),
                os.path.normpath(os.path.join(current_dir, '../../../annotation tool/Pairs8')),
            ]
        if segs_dir is None:
            local_segs_dir = os.path.normpath(os.path.join(current_dir, '../../../annotation tool/segs'))
            legacy_segs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ICU_cases/segs'
            if os.path.isdir(local_segs_dir):
                segs_dir = local_segs_dir
            elif os.path.isdir(legacy_segs_dir):
                segs_dir = legacy_segs_dir
            else:
                segs_dir = local_segs_dir
                print(f'Warning: segmentation directory not found. Expected one of: {local_segs_dir}, {legacy_segs_dir}')

        os.makedirs(preds_dir, exist_ok=True)

        pair_dirs = collect_pair_dirs(pairs_roots)
        rois_to_run = roi_names if roi_names else [None]  # None = original (no ROI)

        for pair_d in pair_dirs:
            pair = os.path.basename(pair_d)
            print(f'Working on {pair}')

            nii_files = sorted([n for n in os.listdir(pair_d) if n.endswith('.nii.gz')])
            if len(nii_files) < 2:
                print(f'Skipping {pair}: fewer than 2 .nii.gz files in {pair_d}')
                continue
            prior_n, current_n = nii_files[0], nii_files[1]
            prior_p, current_p = f'{pair_d}/{prior_n}', f'{pair_d}/{current_n}'

            prior_n = prior_n[:-7]
            current_n = current_n[:-7]

            prior_nif = nib.load(prior_p)
            aff = prior_nif.affine
            prior_raw = torch.tensor(prior_nif.get_fdata().T, dtype=torch.float32)
            current_raw = torch.tensor(nib.load(current_p).get_fdata().T, dtype=torch.float32)

            if use_segs:
                prior_seg_p = f'{segs_dir}/{prior_n}_seg.nii.gz'
                current_seg_p = f'{segs_dir}/{current_n}_seg.nii.gz'

                if not os.path.exists(prior_seg_p):
                    raise FileNotFoundError(f'Missing prior seg file: {prior_seg_p}')
                if not os.path.exists(current_seg_p):
                    raise FileNotFoundError(f'Missing current seg file: {current_seg_p}')

                prior_seg = torch.tensor(nib.load(prior_seg_p).get_fdata().T)
                current_seg = torch.tensor(nib.load(current_seg_p).get_fdata().T)

                prior_pp = preprocess(prior_raw, prior_seg, crop_pad_val=15, crop_seg=False)
                current_pp = preprocess(current_raw, current_seg, crop_pad_val=15, crop_seg=False)
            else:
                prior_pp = preprocess_no_seg(prior_raw)
                current_pp = preprocess_no_seg(current_raw)

            for roi_name in rois_to_run:
                if roi_name is not None:
                    pred_d = f'{preds_dir}/{roi_mode}/{roi_name}/{pair}'
                else:
                    pred_d = f'{preds_dir}/{pair}'
                os.makedirs(pred_d, exist_ok=True)

                if os.path.exists(f'{pred_d}/output.nii.gz'):
                    continue  # skip already computed

                # Apply ROI to images
                if roi_name is not None:
                    roi_mask = load_roi_mask(roi_masks_dir, roi_name, pair)
                    if roi_name != 'full_image' and roi_mask is None:
                        print(f'  [SKIP] {roi_name}/{pair}: mask not found')
                        continue
                    if roi_mode == 'crop':
                        prior = crop_to_roi(prior_raw, roi_mask)
                        current = crop_to_roi(current_raw, roi_mask)
                    else:  # mask
                        prior = apply_roi_mask(prior_pp, roi_mask)
                        current = apply_roi_mask(current_pp, roi_mask)
                else:
                    prior = prior_pp
                    current = current_pp

                plot_pair(prior, current, pred_d)

                output = pred(prior, current, model)

                output = postprocess(output)

                output_nif = nib.Nifti1Image(output.numpy().T, aff)
                nib.save(output_nif, f'{pred_d}/output.nii.gz')

                plot_output_on_current(current, output, pred_d)
                plot_output_on_current(current, (output > 0).float() - (output < 0).float(), pred_d, suffix='_bin')

                if roi_name is not None:
                    print(f'  {roi_name} done')


def parse_args():
    parser = argparse.ArgumentParser()
    segs_group = parser.add_mutually_exclusive_group()
    segs_group.add_argument('--use-segs', dest='use_segs', action='store_true')
    segs_group.add_argument('--no-segs', dest='use_segs', action='store_false')
    parser.set_defaults(use_segs=True)
    parser.add_argument('--model-path', type=str, default=None, help='Path to model checkpoint (.pt).')
    parser.add_argument('--preds-dir', type=str, default=None, help='Output predictions directory.')
    parser.add_argument('--pairs-roots', nargs='+', default=None, help='One or more roots containing pair directories.')
    parser.add_argument('--segs-dir', type=str, default=None, help='Segmentation directory. If omitted, local path is used with legacy fallback.')
    parser.add_argument('--roi-masks-dir', type=str, default=None, help='Directory with precomputed ROI masks. Enables ROI mode.')
    parser.add_argument('--roi-names', nargs='+', default=None,
                        help=f'Which ROIs to run. Default (when --roi-masks-dir given): all {len(ROI_NAMES)}. Options: {ROI_NAMES}')
    parser.add_argument('--roi-mode', type=str, default='mask', choices=['mask', 'crop'],
                        help='mask = zero outside ROI, crop = crop to ROI bbox & resize to 512.')
    return parser.parse_args()


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
    args = parse_args()
    roi_names = args.roi_names
    if roi_names is None and args.roi_masks_dir is not None:
        roi_names = ROI_NAMES
    main(use_segs=args.use_segs, model_path=args.model_path, preds_dir=args.preds_dir,
         pairs_roots=args.pairs_roots, segs_dir=args.segs_dir,
         roi_masks_dir=args.roi_masks_dir, roi_names=roi_names, roi_mode=args.roi_mode)
