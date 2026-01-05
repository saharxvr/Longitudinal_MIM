import os
import shutil

import torch
import torchvision.transforms.v2 as v2
import random
from mediffusion import DiffusionModule
import nibabel as nib
from torchvision.transforms.v2 import Resize, InterpolationMode
from matplotlib import colors
from utils import get_sep_lung_masks
from augmentations import get_bounding_rect
import monai as mn
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from extra.alignment import AFFINE_DCM
from constants import DEVICE
from functools import partialmethod
import gryds


def plot_sample(orig_im, new_im, orig_mask, new_diff_img, labels_str='', c_id=1):
    divnorm = colors.TwoSlopeNorm(vmin=torch.min(new_diff_img).item(), vcenter=0., vmax=torch.max(new_diff_img).item())

    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0, 0].imshow(orig_im.T.squeeze().cpu(), cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(new_im.T.squeeze().cpu(), cmap='gray')
    ax[0, 1].set_title(f'Inpainted: {labels_str}')
    ax[1, 0].imshow(orig_mask.T.squeeze().cpu(), cmap='gray')
    ax[1, 0].set_title('Inpainting Mask')
    ax[1, 1].imshow(new_im.T.squeeze().cpu(), cmap='gray')
    mean_alphas = generate_alpha_map(new_diff_img.T)
    imm = ax[1, 1].imshow(new_diff_img.T.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
    ax[1, 1].set_title('Difference map')
    fig.colorbar(imm, fraction=0.05, pad=0.04, ax=ax[1, 1], shrink=0.7)
    ax[0, 0].set_axis_off()
    ax[0, 1].set_axis_off()
    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/tests/diffusion_output{c_id}.png')
    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()
    plt.cla()
    plt.clf()
    plt.close()


def scale_and_suppress_non_max(cur_img, new_upper_bound=0.2, scale_fac: float = 1.):
    abs_img = cur_img.abs()
    abs_max = torch.max(abs_img).item()
    new_max = min(abs_max, new_upper_bound)
    scaled_img = torch.exp(scale_fac * abs_img) - 1
    scaled_img = scaled_img / (torch.max(scaled_img) / new_max)
    cur_img = scaled_img * torch.sign(cur_img)
    return cur_img


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val
    alphas_map = alphas_map.squeeze()

    return alphas_map


def load_img_data(img_path, lungs_path, heart_path=None):
    im = torch.tensor(nib.load(img_path).get_fdata()).float().squeeze()[None, ...]
    im = resize_512(im).unsqueeze(0)

    lungs = torch.tensor(nib.load(lungs_path).get_fdata()).squeeze()[None, ...]
    lungs = resize_512(lungs).unsqueeze(0)

    if heart_path is None:
        return im, lungs

    heart = torch.tensor(nib.load(heart_path).get_fdata()).squeeze()[None, ...]
    heart = resize_512(heart).unsqueeze(0)

    return im, lungs, heart


def save_pair_data(im1, im2, diff_im):
    ...


def get_random_choice_of_lungs(lungs, single_lung_chance=0.333, side_flag=None):
    if side_flag is None:
        lung_flag = random.random()
    else:
        lung_flag = side_flag
    if lung_flag < single_lung_chance:
        lung1, lung2 = get_sep_lung_masks(lungs.T.squeeze(), ret_right_then_left=True)  # Note: Need to transpose if sides need to be ordered correctly
        seg = torch.tensor(lung1.T)[None, None, ...]
    elif lung_flag < 2 * single_lung_chance:
        lung1, lung2 = get_sep_lung_masks(lungs.T.squeeze(), ret_right_then_left=True)  # Note: Need to transpose if sides need to be ordered correctly
        seg = torch.tensor(lung2.T)[None, None, ...]
    else:
        seg = lungs
    return seg


def inpaint_img(orig_im, inpaint_mask, inpaint_labels, classifier_cond_scale, noise=None, scale_back=True):
    orig_im = scale(orig_im)
    model_kwargs = {"cls": inpaint_labels.unsqueeze(0).to(DEVICE)}

    if noise is None:
        noise = torch.randn(1, 1, 512, 512)

    inpainted_img = model.predict(
        noise.to(DEVICE),
        model_kwargs=model_kwargs,
        classifier_cond_scale=classifier_cond_scale,
        inference_protocol="DDIM100",
        original_image=orig_im.to(DEVICE),
        mask=inpaint_mask.to(DEVICE)
    )[0]

    if scale_back:
        inpainted_img = inv_scale(inpainted_img)

    return inpainted_img.unsqueeze(0)


def im_to_seg_path(im_path: str, get_heart=False):
    lungs_path = im_path.replace('images_new_new', 'images_segs_lungs')
    lungs_path = lungs_path.replace('.nii.gz', '_seg.nii.gz')

    if get_heart:
        hearts_path = im_path.replace('images_new_new', 'images_segs_heart')
        hearts_path = hearts_path.replace('.nii.gz', '_seg.nii.gz')

        return lungs_path, hearts_path

    return lungs_path


def get_random_no_finding_path():
    return no_finding_paths[random.randint(0, no_finding_len - 1)]


def get_random_image_path():
    return df.iloc[random.randint(0, df_len - 1)]['Path']


def create_unrelated_healthy_pair(pair_path):
    im_path1 = get_random_no_finding_path()
    lungs_path1 = im_to_seg_path(im_path1)
    im_path2 = get_random_no_finding_path()
    lungs_path2 = im_to_seg_path(im_path2)

    os.symlink(im_path1, f'{pair_path}/im1.nii.gz')
    os.symlink(im_path2, f'{pair_path}/im2.nii.gz')
    os.symlink(lungs_path1, f'{pair_path}/seg1.nii.gz')
    os.symlink(lungs_path2, f'{pair_path}/seg2.nii.gz')

    diff_im = torch.zeros(im_size)
    torch.save(diff_im, f'{pair_path}/difference_map.pt')


def create_inpainted_healthy_pair(pair_path):
    # if random.random() < 0.475:
    #     im_path = get_random_no_finding_path()
    #     healthy = True
    # else:
    #     im_path = get_random_image_path()
    #     healthy = False
    im_path = get_random_image_path()
    healthy = False

    lungs_path = im_to_seg_path(im_path)

    im, lungs = load_img_data(im_path, lungs_path)

    dilation_kernel_size = random.randint(1, 3) * 2 + 1
    lungs = torch.nn.functional.max_pool2d(lungs, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size // 2)
    seg = get_random_choice_of_lungs(lungs)

    cls_label = torch.zeros(12, dtype=torch.float)
    cls_label[0] = 1.

    classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
    im1 = inpaint_img(im, seg, cls_label, classifier_cond_scale, scale_back=True)
    # im1 = resize_reduce(resize_enlarge(im1.squeeze(0)))

    if not healthy:
        classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
        im2 = inpaint_img(im, seg, cls_label, classifier_cond_scale, scale_back=True)
        # im2 = resize_reduce(resize_enlarge(im2.squeeze(0)))
    else:
        im2 = im

    idx1 = random.randint(1, 2)
    idx2 = 3 - idx1

    nif1 = nib.Nifti1Image(im1.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif1, f'{pair_path}/im{idx1}.nii.gz')
    os.symlink(lungs_path, f'{pair_path}/seg{idx1}.nii.gz')
    os.symlink(lungs_path, f'{pair_path}/seg{idx2}.nii.gz')

    if not healthy:
        nif2 = nib.Nifti1Image(im2.squeeze().cpu().numpy(), AFFINE_DCM)
        nib.save(nif2, f'{pair_path}/im{idx2}.nii.gz')
    else:
        os.symlink(im_path, f'{pair_path}/im{idx2}.nii.gz')

    diff_im = torch.zeros(im_size)
    nif_diff = nib.Nifti1Image(diff_im.numpy(), AFFINE_DCM)
    nib.save(nif_diff, f'{pair_path}/difference_map.nii.gz')


def disordered_img_inpainting(im, lungs, return_seg=False, side_flag=None):
    seg = get_random_choice_of_lungs(lungs, single_lung_chance=0.3, side_flag=side_flag).float()

    dilated = False
    pool_flag = random.random()
    if pool_flag < 0.15:
        kernel_size = random.randint(1, 3) * 2 + 1
        seg = torch.nn.functional.max_pool2d(seg, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        dilated = True
    elif pool_flag < 0.7:
        kernel_size = random.randint(1, 15) * 2 + 1
        seg = -torch.nn.functional.max_pool2d(-seg, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    if dilated or random.random() > 0.75:
        frequency_scale = random.random() * 35 + 25
        th = (random.random() * 0.447) ** 2 + 0.45
        overlay = torch.rand(1, 1, int(im.shape[-2] / frequency_scale), int(im.shape[-1] / frequency_scale))
        overlay = resize_512(overlay)
        overlay[overlay > th] = 1.
        overlay[overlay <= th] = 0.
        seg = seg * overlay

    cls_label = torch.rand(12, dtype=torch.float)
    cls_label = (cls_label > label_th_arr_intra_pul).float()

    classifier_cond_scale = random.random() * 4.5 + 7.5

    inpainted_im = inpaint_img(im, seg, cls_label, classifier_cond_scale, scale_back=True)

    if return_seg:
        return inpainted_im, seg
    return inpainted_im


def create_inpainted_disordered_abnormal_pair(pair_path):
    im_path = get_random_image_path()
    lungs_path = im_to_seg_path(im_path)
    im, lungs = load_img_data(im_path, lungs_path)
    im = inv_scale(im)

    inpainted_im, seg1 = disordered_img_inpainting(im, lungs, return_seg=True)
    both_inpainted = False

    if random.random() < 0.66:
        idx1 = random.randint(1, 2)
        idx2 = 3 - idx1
        imgs = [0, 0]
        imgs[idx1 - 1] = inpainted_im
        imgs[idx2 - 1] = im
        seg = seg1
    else:
        inpainted_im2, seg2 = disordered_img_inpainting(im, lungs, return_seg=True)
        seg = (seg1 + seg2).clamp_max(1.)
        idx1 = 1
        idx2 = 2
        imgs = [inpainted_im, inpainted_im2]
        both_inpainted = True

    diff_im = imgs[1] - imgs[0]
    # diff_im = scale_and_suppress_non_max(diff_im, 0.175, scale_fac=1.5)
    diff_im *= seg
    # diff_im[diff_im.abs() < 0.02] = 0

    nif1 = nib.Nifti1Image(inpainted_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif1, f'{pair_path}/im{idx1}.nii.gz')

    os.symlink(lungs_path, f'{pair_path}/seg{idx1}.nii.gz')
    os.symlink(lungs_path, f'{pair_path}/seg{idx2}.nii.gz')

    if both_inpainted:
        nif2 = nib.Nifti1Image(imgs[idx2 - 1].squeeze().cpu().numpy(), AFFINE_DCM)
        nib.save(nif2, f'{pair_path}/im{idx2}.nii.gz')
    else:
        os.symlink(im_path, f'{pair_path}/im{idx2}.nii.gz')

    nif_diff = nib.Nifti1Image(diff_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif_diff, f'{pair_path}/difference_map.nii.gz')


def random_effusion_height_limit():
    init_val = random.random()
    return 0.01365 + 2.47792 * init_val - 4.51568 * (init_val ** 2) + 3.01045 * (init_val ** 3)


def create_random_lower_mask_right_lung(seg):
    x_freq = random.random() * 45 + 15
    y_freq = random.random() * 45 + 15
    x_deform_intensity = random.random() * 0.15 + 0.1
    y_deform_intensity = random.random() * 0.15 + 0.1
    grid_x = deform_resize((torch.rand(1, int(seg.shape[-2] / x_freq), int(seg.shape[-1] / x_freq)) - 0.5) * x_deform_intensity).squeeze()
    grid_y = deform_resize((torch.rand(1, int(seg.shape[-2] / y_freq), int(seg.shape[-1] / y_freq)) - 0.5) * y_deform_intensity).squeeze()
    bspline = gryds.BSplineTransformation([grid_y, grid_x], order=1)

    (y_min, x_min), (y_max, x_max) = get_bounding_rect(seg)
    seg = seg.float()

    orig_seg = seg.clone()

    seg_outer_side = torch.argmax(seg, dim=1, keepdim=True)
    height_limit_perc = random_effusion_height_limit()
    seg[:int(y_min + (y_max - y_min) * height_limit_perc), :] = 0.

    index_map = torch.arange(0, seg.shape[-2]).repeat(seg.shape[-1], 1)
    update_map = torch.logical_and(index_map >= seg_outer_side, index_map < x_max)
    mask_rows = torch.any(seg > 0, dim=1)
    seg[mask_rows] = update_map[mask_rows].float()

    interpolator = gryds.Interpolator(seg)
    seg = torch.tensor(interpolator.transform(bspline))
    seg = torch.round(seg)

    seg[index_map < seg_outer_side] = 0.

    if random.random() < 0.35:
        frequency_scale = random.random() * 35 + 25
        th = random.random() * 0.35 + 0.45
        overlay = torch.rand(1, int(seg.shape[-2] / frequency_scale), int(seg.shape[-1] / frequency_scale))
        overlay = resize_512(overlay).squeeze()
        overlay[overlay > th] = 1.
        overlay[overlay <= th] = 0.
        seg = ((orig_seg * overlay) + seg).clamp_max(1.)

    return seg


def create_random_lower_mask(seg, side_flag=None):
    seg = seg.squeeze().T
    right_lung, left_lung = get_sep_lung_masks(seg.numpy(), ret_right_then_left=True)
    right_lung = torch.tensor(right_lung)
    left_lung = torch.tensor(left_lung)

    if side_flag is None:
        lungs_flag = random.random()
    else:
        lungs_flag = side_flag

    if lungs_flag < 0.37:
        seg = create_random_lower_mask_right_lung(right_lung)
    elif lungs_flag < 0.74:
        left_lung = torch.flip(left_lung, dims=(1,))
        seg = create_random_lower_mask_right_lung(left_lung)
        seg = torch.flip(seg, dims=(1,))
    else:
        seg1 = create_random_lower_mask_right_lung(right_lung)

        left_lung = torch.flip(left_lung, dims=(1,))
        seg2 = create_random_lower_mask_right_lung(left_lung)
        seg2 = torch.flip(seg2, dims=(1,))

        seg = (seg1 + seg2).clamp_max(1.)

    seg = seg.T
    seg = seg[None, None, ...]
    return seg


def lower_img_inpainting(im, lungs):
    cls_label = torch.rand(12, dtype=torch.float)
    cls_label = (cls_label > label_th_arr_lower).float()

    classifier_cond_scale = random.random() * 4.5 + 7.5

    inpainted_im = inpaint_img(im, lungs, cls_label, classifier_cond_scale, scale_back=True)

    return inpainted_im


def create_inpainted_lower_abnormal_pair(pair_path):
    im_path = get_random_image_path()
    lungs_path = im_to_seg_path(im_path)
    im, lungs = load_img_data(im_path, lungs_path)
    im = inv_scale(im)

    seg1 = create_random_lower_mask(lungs)
    inpainted_im1 = lower_img_inpainting(im, seg1)
    both_inpainted = False

    if random.random() < 0.75:
        idx1 = random.randint(1, 2)
        idx2 = 3 - idx1
        imgs = [0, 0]
        imgs[idx1 - 1] = inpainted_im1
        imgs[idx2 - 1] = im
        seg = seg1
    else:
        seg2 = create_random_lower_mask(lungs)
        inpainted_im2 = lower_img_inpainting(im, seg2)
        seg = (seg1 + seg2).clamp_max(1.)
        idx1 = 1
        idx2 = 2
        imgs = [inpainted_im1, inpainted_im2]
        both_inpainted = True

    diff_im = imgs[1] - imgs[0]
    # diff_im = scale_and_suppress_non_max(diff_im, 0.175, scale_fac=1.5)
    diff_im *= seg
    # diff_im[diff_im.abs() < 0.02] = 0

    nif1 = nib.Nifti1Image(inpainted_im1.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif1, f'{pair_path}/im{idx1}.nii.gz')

    os.symlink(lungs_path, f'{pair_path}/seg{idx1}.nii.gz')
    os.symlink(lungs_path, f'{pair_path}/seg{idx2}.nii.gz')

    if both_inpainted:
        nif2 = nib.Nifti1Image(imgs[idx2 - 1].squeeze().cpu().numpy(), AFFINE_DCM)
        nib.save(nif2, f'{pair_path}/im{idx2}.nii.gz')
    else:
        os.symlink(im_path, f'{pair_path}/im{idx2}.nii.gz')

    nif_diff = nib.Nifti1Image(diff_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif_diff, f'{pair_path}/difference_map.nii.gz')


def create_inpainted_healthy_abnormal_pair(pair_path, abnormal_type='disordered'):
    assert abnormal_type in ['disordered', 'lower']

    im_path = get_random_image_path()
    lungs_path = im_to_seg_path(im_path)
    im, lungs = load_img_data(im_path, lungs_path)
    im = inv_scale(im)

    sides = [0.2, 0.5]
    healthy_side_idx = random.randint(0, len(sides) - 1)
    healthy_side = sides[healthy_side_idx]
    abnormal_side = sides[len(sides) - 1 - healthy_side_idx]

    ###
    dilation_kernel_size = random.randint(0, 3) * 2 + 1
    seg_healthy = torch.nn.functional.max_pool2d(lungs, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size // 2)
    seg_healthy = get_random_choice_of_lungs(seg_healthy, side_flag=healthy_side)

    cls_label = torch.zeros(12, dtype=torch.float)
    cls_label[0] = 1.

    classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
    inpainted_healthy_im1 = inpaint_img(im, seg_healthy, cls_label, classifier_cond_scale, scale_back=True)

    dilation_kernel_size = random.randint(0, 3) * 2 + 1
    seg_healthy = torch.nn.functional.max_pool2d(lungs, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size // 2)
    seg_healthy = get_random_choice_of_lungs(seg_healthy, side_flag=healthy_side)

    classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
    inpainted_healthy_im2 = inpaint_img(im, seg_healthy, cls_label, classifier_cond_scale, scale_back=True)
    ###

    imgs = [inpainted_healthy_im1, inpainted_healthy_im2]
    to_inpaint_abnormal_idx = random.randint(0, len(imgs) - 1)
    to_inpaint_abnormal = imgs[to_inpaint_abnormal_idx]
    other_im = imgs[len(imgs) - 1 - to_inpaint_abnormal_idx]

    if abnormal_type == 'disordered':
        inpainted_abnormal_im, seg_abnormal = disordered_img_inpainting(to_inpaint_abnormal, lungs, return_seg=True, side_flag=abnormal_side)
    else:
        seg_abnormal = create_random_lower_mask(lungs, side_flag=abnormal_side)
        inpainted_abnormal_im = lower_img_inpainting(to_inpaint_abnormal, seg_abnormal)

    # ###
    # dilation_kernel_size = random.randint(0, 3) * 2 + 1
    # seg_healthy = torch.nn.functional.max_pool2d(lungs, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size // 2)
    # seg_healthy = get_random_choice_of_lungs(seg_healthy, side_flag=healthy_side)
    #
    # cls_label = torch.zeros(12, dtype=torch.float)
    # cls_label[0] = 1.
    #
    # classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
    # inpainted_healthy_im = inpaint_img(to_inpaint_healthy, seg_healthy, cls_label, classifier_cond_scale, scale_back=True)
    # ###

    # if not healthy:
    #     dilation_kernel_size = random.randint(0, 3) * 2 + 1
    #     seg_healthy = torch.nn.functional.max_pool2d(lungs, kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size // 2)
    #     seg_healthy = get_random_choice_of_lungs(seg_healthy, side_flag=healthy_side)
    #
    #     classifier_cond_scale = (random.random() ** 2.) * 3.5 + 4.
    #     other_im = inpaint_img(other_im, seg_healthy, cls_label, classifier_cond_scale, scale_back=True)

    order = random.randint(0, 1)
    if order == 1:
        diff_im = (other_im - inpainted_abnormal_im) * seg_abnormal
        idx_abnormal = 1
        idx_other = 2
    else:
        diff_im = (inpainted_abnormal_im - other_im) * seg_abnormal
        idx_abnormal = 2
        idx_other = 1

    nif_healthy = nib.Nifti1Image(other_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif_healthy, f'{pair_path}/im{idx_other}.nii.gz')

    nif_other = nib.Nifti1Image(inpainted_abnormal_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif_other, f'{pair_path}/im{idx_abnormal}.nii.gz')

    os.symlink(lungs_path, f'{pair_path}/seg{idx_other}.nii.gz')
    os.symlink(lungs_path, f'{pair_path}/seg{idx_abnormal}.nii.gz')

    nif_diff = nib.Nifti1Image(diff_im.squeeze().cpu().numpy(), AFFINE_DCM)
    nib.save(nif_diff, f'{pair_path}/difference_map.nii.gz')


def create_Chexpert_df():
    def Chexpert_csv_to_path(c_row):
        c_path = c_row['Path']
        c_path_split = c_path.split('/')
        # split_type = c_path_split[1]
        img_path = '/'.join(c_path_split[2:]).split('.')[0] + '.nii.gz'
        c_file = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_new_new/{img_path}'
        # c_file = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/{split_type}/images/{img_path}'
        return c_file

    train_labels_csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/train.csv'
    valid_labels_csv_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/valid/valid.csv'

    c_path_col = 'Path'
    c_labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
              'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Support Devices']
    drop_labels = ['Pleural Other', 'Fracture']
    train_df = pd.read_csv(train_labels_csv_path)
    valid_df = pd.read_csv(valid_labels_csv_path)
    c_df = pd.concat([train_df, valid_df])
    c_df = c_df[c_df.apply(lambda r: r[c_path_col] not in {'CheXpert-v1.0/train/patient05271/study5/view2_frontal.jpg', 'CheXpert-v1.0/train/patient48043/study1/view2_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient09797/study4/view1_frontal.jpg', 'CheXpert-v1.0/train/patient12362/study1/view1_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient25979/study8/view1_frontal.jpg', 'CheXpert-v1.0/train/patient44163/study1/view1_frontal.jpg',
                                                   'CheXpert-v1.0/train/patient40255/study2/view1_frontal.jpg'}, axis=1)]

    c_df = c_df[c_df['Frontal/Lateral'] == 'Frontal']
    c_df = c_df.dropna(axis=0, subset=['Frontal/Lateral'])
    c_df = c_df.loc[~((c_df['Pleural Other'] == 1.) | (c_df['Fracture'] == 1.))]
    c_df.drop(columns=drop_labels, inplace=True)
    c_df[c_df[c_labels].isna()] = 0.
    c_df[c_labels] = c_df[c_labels].astype(float)
    c_df[c_df[c_labels] == -1.] = 0.5
    c_df[c_path_col] = c_df.apply(lambda r: Chexpert_csv_to_path(r), axis=1)
    return c_df


if __name__ == '__main__':
    df = create_Chexpert_df()
    df_len = len(df)
    labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
              'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
              'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Support Devices']

    no_finding_paths = list(df.loc[(df['No Finding'] == 1.) & (df['Support Devices'] == 0.)]['Path'])
    no_finding_len = len(no_finding_paths)

    im_size = (768, 768)
    resize_512 = Resize((512, 512))
    resize_768 = Resize((768, 768))
    deform_resize = v2.Resize((512 // 8, 512 // 8))

    model = DiffusionModule("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/config.yaml")
    model.load_ckpt("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/pl/last-v2.ckpt", ema=True)
    model.eval().to(DEVICE)

    scale = mn.transforms.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=-1, b_max=1, clip=True)
    inv_scale = mn.transforms.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1, clip=True)

    differential_grad = colors.LinearSegmentedColormap.from_list('differential_grad', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:440A57-12.5:5628A5-25:1256D4-37.5:119AB9-49:07C8C3-50:FFFFFF-51:00A64E-62.5:3CC647-75:60F132-87.5:A8FF3E-100:ECF800
        (0.000, (0.267, 0.039, 0.341)),
        (0.125, (0.337, 0.157, 0.647)),
        (0.250, (0.071, 0.337, 0.831)),
        (0.375, (0.067, 0.604, 0.725)),
        (0.490, (0.027, 0.784, 0.765)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.510, (0.000, 0.651, 0.306)),
        (0.625, (0.235, 0.776, 0.278)),
        (0.750, (0.376, 0.945, 0.196)),
        (0.875, (0.659, 1.000, 0.243)),
        (1.000, (0.925, 0.973, 0.000))
    ))

    label_th_arr_intra_pul = torch.tensor([2., 0.8, 0.75, 0.15, 0.4, 0.4, 0.4, 0.4, 0.4, 0.75, 0.4, 0.85])
    label_th_arr_lower = torch.tensor([2., 0.9, 0.75, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.85, 0.02, 0.85])

    pairs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/synthetic_train_pairs/inpainted_healthy_abnormal_disordered'

    pairs_to_create = 14500
    cur_pair_id = 10500
    offset = 0

    err_num = 0

    for _ in tqdm(range(pairs_to_create)):
        if _ == 0:
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        # if cur_pair_id >= pairs_to_create:
        #     break
        # im_idx = random.randint(0, df_len)
        # row = df.iloc[im_idx]
        # path = row['Path']
        # original_labels = list(row[labels])

        out_path = f'{pairs_dir}/pair{cur_pair_id + offset}'
        # if os.path.exists(out_path):
        #     print(f"Path: {out_path} exists!!!")
        #     cur_pair_id += 1
        #     continue
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=False)
        try:
            # create_inpainted_healthy_disordered_abnormal_pair(out_path)
            create_inpainted_healthy_abnormal_pair(out_path)
            cur_pair_id += 1
        except Exception as e:
            err_num += 1
            print(f"Exception: {e}")
            print(f"So far encountered {err_num} errors")
    exit()
    print(f"##########\nCurrent pair id = {cur_pair_id + offset}\n#############")

    while cur_pair_id < pairs_to_create:
        out_path = f'{pairs_dir}/pair{cur_pair_id + offset}'
        os.makedirs(out_path, exist_ok=True)
        try:
            create_inpainted_disordered_abnormal_pair(out_path)
            cur_pair_id += 1
        except Exception as e:
            err_num += 1
            print(f"Exception: {e}")
            print(f"So far encountered {err_num} errors")


