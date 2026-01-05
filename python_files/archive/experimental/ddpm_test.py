import random

import torch
import matplotlib.pyplot as plt
import yaml
from mediffusion import DiffusionModule
import nibabel as nib
import numpy as np
from torchvision.transforms.v2 import Resize, CenterCrop
import monai as mn
import bkh_pytorch_utils as bpu
from matplotlib import colors
from utils import get_sep_lung_masks
from augmentations import get_bounding_rect


def load_img_data(img_path, lungs_path, heart_path=None):
    im = torch.tensor(nib.load(img_path).get_fdata()).float().squeeze()[None, ...]
    im = resize(im).unsqueeze(0)

    lungs = torch.tensor(nib.load(lungs_path).get_fdata()).squeeze()[None, ...]
    lungs = resize(lungs).unsqueeze(0)

    if heart_path is None:
        return im, lungs

    heart = torch.tensor(nib.load(heart_path).get_fdata()).squeeze()[None, ...]
    heart = resize(heart).unsqueeze(0)

    return im, lungs, heart


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val
    alphas_map = alphas_map.squeeze()

    return alphas_map


def scale_and_suppress_non_max(cur_img, new_upper_bound=0.3, scale_fac: float = 1.):
    abs_img = cur_img.abs()
    abs_max = torch.max(abs_img).item()
    new_max = min(abs_max, new_upper_bound)
    scaled_img = torch.exp(scale_fac * abs_img) - 1
    scaled_img = scaled_img / (torch.max(scaled_img) / new_max)
    cur_img = scaled_img * torch.sign(cur_img)
    return cur_img


# def scale_and_suppress_non_max(cur_img, new_max, scale_fac: float = 5.):
#     scaled_img = torch.exp(scale_fac * cur_img.abs()) - 1
#     scaled_img = scaled_img / (torch.max(scaled_img) / new_max)
#     cur_img = scaled_img * torch.sign(cur_img)
#     return cur_img


if __name__ == '__main__':
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
        (1.000, (0.925, 0.973, 0.000))))

    im_path = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_new_new/patient40254/study1/view1_frontal.nii.gz'
    mask_p = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_segs_lungs/patient40254/study1/view1_frontal_seg.nii.gz'
    # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_segs_heart/patient40255/study1/view1_frontal_seg.nii.gz'

    # im_path2 = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_new_new/patient40255/study2/view1_frontal.nii.gz'
    # mask_p2 = '/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_segs_lungs/patient40255/study2/view1_frontal_seg.nii.gz'
    # heart_p2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/train/images_segs_heart/patient40255/study2/view1_frontal_seg.nii.gz'

    c_id = 1

    model = DiffusionModule("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/config.yaml")
    model.load_ckpt("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/pl/last-v2.ckpt", ema=True)
    model.eval().cuda()

    resize = Resize((512, 512))

    # seed = 0
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # noise = model.diffusion.q_sample(im, t=torch.LongTensor([999])).unsqueeze(0)
    # print(noise.shape)
    noise = torch.randn(1, 1, 512, 512)
    # noise = torch.randn(1, 1, 512, 512)
    cls_label = torch.zeros(12, dtype=torch.float)
    labels_to_inpaint = [3,5,7,10]
    labels_str = []
    label_dict = {
        0: 'No Finding',
        1: 'Enlarged Cardiomediastinum',
        2: 'Cardiomegaly',
        3: 'Lung Opacity',
        4: 'Lung Lesion',
        5: 'Edema',
        6: 'Consolidation',
        7: 'Pneumonia',
        8: 'Atelectasis',
        9: 'Pneumothorax',
        10: 'Pleural Effusion',
        11: 'Support Devices',
    }
    for label in labels_to_inpaint:
        cls_label[label] = 1.
        labels_str.append(label_dict[label])
    labels_str = ' & '.join(labels_str)

    # model_kwargs = {"cls": torch.nn.functional.one_hot(cls_label, num_classes=3)}
    # model_kwargs = {"cls": torch.stack([cls_label, cls_label, cls_label, cls_label])}
    model_kwargs = {"cls": cls_label.unsqueeze(0)}

    im1, lungs1 = load_img_data(im_path, mask_p)
    orig_lungs = lungs1.clone()
    # im2, lungs2 = load_img_data(im_path2, mask_p2)

    # lungs1 = torch.nn.functional.max_pool2d(lungs1, kernel_size=5, stride=1, padding=2)
    # lungs1 = -torch.nn.functional.max_pool2d(-lungs1, kernel_size=15, stride=1, padding=7)
    # lungs2 = torch.nn.functional.max_pool2d(lungs2, kernel_size=3, stride=1, padding=1)

    #################
    import torchvision.transforms.v2 as v2
    import gryds
    deform_resize = v2.Resize((512 // 8, 512 // 8))

    x_freq = random.random() * 45 + 15
    y_freq = random.random() * 45 + 15
    x_deform_intensity = random.random() * 0.12 + 0.1
    y_deform_intensity = random.random() * 0.12 + 0.1
    grid_x = deform_resize((torch.rand(1, int(lungs1.shape[-2] / x_freq), int(lungs1.shape[-1] / x_freq)) - 0.5) * x_deform_intensity).squeeze()
    grid_y = deform_resize((torch.rand(1, int(lungs1.shape[-2] / y_freq), int(lungs1.shape[-1] / y_freq)) - 0.5) * y_deform_intensity).squeeze()
    bspline = gryds.BSplineTransformation([grid_y, grid_x], order=1)

    lungs1 = lungs1.T
    lung1, lung2 = get_sep_lung_masks(lungs1.numpy().squeeze(), ret_right_then_left=True)
    lungs1 = torch.tensor(lung1)

    orig_lungs1 = lungs1.clone()

    # mask = lungs[1]
    # lungs1 = torch.flip(lungs1, dims=(1,))
    (y_min, x_min), (y_max, x_max) = get_bounding_rect(lungs1)
    center = ((y_min + y_max) // 2, (x_min + x_max) // 2)
    lungs1 = lungs1.float()
    lungs_outer_side = torch.argmax(lungs1, dim=1, keepdim=True)


    def random_effusion_height_limit():
        init_val = random.random()
        return 0.01365 + 2.47792 * init_val - 4.51568 * (init_val ** 2) + 3.01045 * (init_val ** 3)


    height_limit_perc = random_effusion_height_limit()
    lungs1[:int(y_min + (y_max - y_min) * height_limit_perc), :] = 0.
    # lungs1[:center[0] * 3 // 4, :] = 0.
    # center = ((y_min + y_max) // 2, (x_min + x_max) // 2)
    index_map = torch.arange(0, lungs1.shape[-1]).repeat(lungs1.shape[-1], 1)
    update_map = torch.logical_and(index_map >= lungs_outer_side, index_map < x_max)
    mask_rows = torch.any(lungs1 > 0, dim=1)
    lungs1[mask_rows] = update_map[mask_rows].float()

    interpolator = gryds.Interpolator(lungs1)
    lungs1 = torch.tensor(interpolator.transform(bspline))
    lungs1 = torch.round(lungs1)

    lungs1[index_map < lungs_outer_side] = 0.

    # lungs1 = torch.flip(lungs1, dims=(1,))
    lungs1 = lungs1.T
    lungs1 = lungs1[None, None, ...]
    ###################

    frequency_scale = random.random() * 35 + 25
    th = random.random() * 0.35 + 0.45
    overlay = torch.rand(1, 1, int(lungs1.shape[-2] / frequency_scale), int(lungs1.shape[-1] / frequency_scale))
    overlay = resize(overlay)
    overlay[overlay > th] = 1.
    overlay[overlay <= th] = 0.
    lungs1 = ((orig_lungs1.T * overlay) + lungs1).clamp_max(1.)

    ################

    # mask = mask.unsqueeze(0)

    # mask = torch.logical_or(mask.bool(), heart.bool()).float()
    # mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    # mask = mask.unsqueeze(0)
    #
    # mask[0,0,:,: center[0]] = 0.
    # mask[0,0,:x_min,:] = 0.
    # mask[0,0,x_min:x_max,center[0]:] = 1.
    #
    # mask = torch.roll(mask, dims=(-1,-2), shifts=(15,0))

    # mask[0,0,150:,:] = 0.

    # mask = torch.roll(mask, dims=(-1,-2), shifts=(10,-11))

    frequency_scale = random.random() * 40 + 20
    # frequency_scale = 40
    # frequency_scale = 150
    overlay = torch.rand(1, 1, int(im1.shape[-2] / frequency_scale), int(im1.shape[-1] / frequency_scale))
    overlay = resize(overlay)
    th = 0.4
    overlay[overlay > th] = 1.
    overlay[overlay <= th] = 0.
    # lungs1 = lungs1 * overlay

    # plt.imshow(mask.T.squeeze().cpu().numpy(), cmap='gray')
    # plt.savefig("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/tests/diffusion_img_mask.png")
    # plt.imshow(mask.squeeze().numpy(),cmap='gray')
    # plt.savefig("mask_diff_test.png")
    # exit()

    # mask = torch.zeros_like(noise)
    # mask[0,0,40:110, 70:180]=1
    # mask[0,0,160:225, 70:200]=1

    transforms = mn.transforms.Compose([
        # mn.transforms.LoadImageD(keys="img"),
        # bpu.EnsureGrayscaleD(keys="img"),
        # mn.transforms.ResizeD(keys='img', size_mode="longest", mode="bilinear", spatial_size=256, align_corners=False),
        mn.transforms.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=-1, b_max=1, clip=True),
        # mn.transforms.SpatialPadD(keys='img', spatial_size=(256, 256), mode="constant", constant_values=-1),
        # mn.transforms.ToTensorD(keys=["cls"], dtype=torch.float),
        # mn.transforms.AsDiscreteD(keys=["cls"], to_onehot=[3]),
        # mn.transforms.SelectItemsD(keys=["img", "cls"]),
        # mn.transforms.ToTensorD(keys=["img", "cls"], dtype=torch.float, track_meta=False),
    ])

    # im = torch.cat([im1, im2], dim=0)
    # lungs = torch.cat([lungs1, lungs2], dim=0)
    # im = torch.cat([im, im], dim=0)
    # lungs = torch.cat([lungs, lungs], dim=0)

    im = im1
    im = transforms(im)
    lungs = lungs1

    # im = model.predict(
    #     noise,
    #     model_kwargs=model_kwargs,
    #     classifier_cond_scale=4.,
    #     inference_protocol="DDIM100",
    #     original_image=im,
    #     mask=lungs
    #     # start_denoise_step=200
    # )[0].unsqueeze(0)
    # cls_label[9] = 1.
    # cls_label[0] = 0.
    # model_kwargs = {"cls": cls_label.unsqueeze(0)}
    # noise = torch.randn(1, 1, 512, 512)
    img = model.predict(
        noise,
        model_kwargs=model_kwargs,
        classifier_cond_scale=12.,
        inference_protocol="DDIM100",
        original_image=im,
        mask=lungs
        # start_denoise_step=200
    )


    # plt.imshow(img[0].T.squeeze().cpu().numpy(), cmap="gray")
    # plt.savefig("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/tests/diffusion_img_new.png")
    # plt.clf()
    # plt.close()

    transform_back = mn.transforms.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1, clip=True)
    im = transform_back(im)
    img[0] = transform_back(img[0]).unsqueeze(0)
    # img[1] = transform_back(img[1])

    # lungs = lungs.bool()
    # im[lungs] = (im[lungs] - torch.min(im[lungs])) / (torch.max(im[lungs]) - torch.min(im[lungs]))
    # img[0][lungs] = (img[0][lungs] - torch.min(img[0][lungs])) / (torch.max(img[0][lungs]) - torch.min(img[0][lungs]))
    lungs = lungs.float()

    # dial_mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    # erode_mask = -torch.nn.functional.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    # diff_mask = (dial_mask.bool() ^ erode_mask.bool()).squeeze(0)
    # blur_img = torch.nn.functional.avg_pool2d(img[0], kernel_size=3, stride=1, padding=1)
    # img[0][diff_mask == 1] = blur_img[diff_mask == 1]

    diff_im = (img[0] - im[0])
    # diff_im[diff_im > 0.02] = 0.
    # diff_im2 = (img[1] - im[1].squeeze(0))

    diff_im = scale_and_suppress_non_max(diff_im, 0.3, scale_fac=1.5)
    # diff_im2 = scale_and_suppress_non_max(diff_im2, 0.2, scale_fac=7.5)

    # diff_im *= lungs1.squeeze()
    # diff_im2 *= lungs2.squeeze()

    from augmentations import RandomBsplineAndSimilarityWithMaskTransform, RandomIntensityTransform, RescaleValuesTransform
    bsp_trans = RandomBsplineAndSimilarityWithMaskTransform()
    int_trans = RandomIntensityTransform()
    rescale_vals = RescaleValuesTransform()

    cat_im1 = torch.cat([img[0], diff_im, orig_lungs]).squeeze()
    cat_im2 = torch.cat([im, orig_lungs]).squeeze()

    new_img, msk1 = bsp_trans(cat_im1)
    new_diff_im = new_img[1]
    new_img = new_img[0].unsqueeze(0)
    new_im, msk2 = bsp_trans(cat_im2)
    new_img = rescale_vals(new_img)
    new_im = rescale_vals(new_im)
    new_img = int_trans(new_img)
    new_im = int_trans(new_im)

    new_diff_im[new_diff_im.abs() < 0.02] = 0.

    divnorm = colors.TwoSlopeNorm(vmin=torch.min(new_diff_im).item(), vcenter=0., vmax=torch.max(new_diff_im).item())
    # divnorm2 = colors.TwoSlopeNorm(vmin=torch.min(diff_im2).item(), vcenter=0., vmax=torch.max(diff_im2).item())

    # x_abs = diff_im.abs()
    # max_val = torch.max(x_abs).item()
    # alphas_map = (x_abs / max_val).squeeze()

    # plt.imshow(diff_im.squeeze().cpu().numpy(), cmap=differential_grad, norm=divnorm)
    # # plt.imshow(diff_im.T.squeeze().cpu().numpy(), cmap='gray', alpha=alphas_map)
    # cbar1 = plt.colorbar(fraction=0.05, pad=0.04)
    # plt.savefig("/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/tests/diffusion_difference_img.png")

    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0, 0].imshow(new_im.T.squeeze().cpu(), cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(new_img.T.squeeze().cpu(), cmap='gray')
    ax[0, 1].set_title(f'Inpainted: {labels_str}')
    ax[1, 0].imshow(lungs[0].T.squeeze().cpu(), cmap='gray')
    ax[1, 0].set_title('Inpainting Mask')
    ax[1, 1].imshow(new_img.T.squeeze().cpu(), cmap='gray')
    mean_alphas = generate_alpha_map(diff_im.T)
    imm = ax[1, 1].imshow(new_diff_im.T.squeeze().cpu(), alpha=mean_alphas, cmap=differential_grad, norm=divnorm)
    ax[1, 1].set_title('Difference map')
    fig.colorbar(imm, fraction=0.05, pad=0.04, ax=ax[1, 1], shrink=0.7)
    # ax[2, 0].imshow(im[1].T.squeeze().cpu(), cmap='gray')
    # ax[2, 1].imshow(img[1].T.squeeze().cpu(), cmap='gray')
    # ax[3, 0].imshow(lungs[1].T.squeeze().cpu(), cmap='gray')
    # imm2 = ax[3, 1].imshow(diff_im2.T.squeeze().cpu(), cmap=differential_grad, norm=divnorm2)
    # fig.colorbar(imm2, fraction=0.05, pad=0.04, ax=ax[3, 1], shrink=0.7)
    ax[0, 0].set_axis_off()
    ax[0, 1].set_axis_off()
    ax[1, 0].set_axis_off()
    ax[1, 1].set_axis_off()
    # ax[2, 0].set_axis_off()
    # ax[2, 1].set_axis_off()
    # ax[3, 0].set_axis_off()
    # ax[3, 1].set_axis_off()
    fig.tight_layout()
    plt.savefig(f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Chexpert/outputs/tests/diffusion_output{c_id}.png')
    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()
    plt.cla()
    plt.clf()
    plt.close()
