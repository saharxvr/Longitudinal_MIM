import os

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torchvision.transforms.v2 as v2
import torch
import random
from kornia.geometry import rotate3d
from constants import DEVICE
from archive.data_preparation.case_filtering import AFFINE_DCM
from tqdm import tqdm
import kornia
from skimage.morphology import ball
import gryds
from typing import Tuple
from time import time
import gc
from scipy.ndimage import label


def load_scan_and_seg(file_path, seg_path):
    """Load a .nii.gz medical image and return a NumPy array."""
    scan_nif = nib.load(file_path)

    global affine
    affine = scan_nif.affine

    scan_data = scan_nif.get_fdata()
    scan_data = torch.tensor(np.transpose(scan_data, (2, 0, 1))).to(torch.float32)
    if 'ABD' not in file_path:
        scan_data = torch.flip(scan_data, dims=[0])

    # (1, 2, 0)

    seg_nif = nib.load(seg_path)
    seg_data = seg_nif.get_fdata()
    seg_data = torch.tensor(np.transpose(seg_data, (2, 0, 1))).to(torch.float32)
    if 'ABD' not in file_path:
        seg_data = torch.flip(seg_data, dims=[0])

    return scan_data, seg_data


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val

    return alphas_map


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def plot_diff_on_current(diff_map, current, out_p):
    diff_map = torch.tensor(diff_map).T
    current = torch.tensor(current).T

    alphas = generate_alpha_map(diff_map)
    divnorm = colors.TwoSlopeNorm(vmin=min(torch.min(diff_map).item(), -0.01), vcenter=0., vmax=max(torch.max(diff_map).item(), 0.01))
    fig, ax = plt.subplots()
    ax.imshow(current.squeeze().cpu(), cmap='gray')
    imm1 = ax.imshow(diff_map.squeeze().cpu(), alpha=alphas, cmap=differential_grad, norm=divnorm)
    cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()

    plt.savefig(out_p)

    # ax[0].clear()
    plt.cla()
    plt.clf()
    plt.close()


# def random_deform_3d(volume):
#     deform_resize = torch.nn.Upsample((volume.shape[0] // 24, volume.shape[1] // 24, volume.shape[2] // 24))
#
#     x_freq = random.random() * 75 + 75
#     y_freq = random.random() * 75 + 75
#     z_freq = random.random() * 75 + 75
#     x_deform_intensity = random.random() * 0.03
#     y_deform_intensity = random.random() * 0.03
#     z_deform_intensity = random.random() * 0.03
#     grid_x = deform_resize(((torch.rand(int(volume.shape[0] / x_freq), int(volume.shape[1] / x_freq), int(volume.shape[2] / x_freq)) - 0.5) * x_deform_intensity)[None, None, ...]).squeeze()
#     grid_y = deform_resize(((torch.rand(int(volume.shape[0] / y_freq), int(volume.shape[1] / y_freq), int(volume.shape[2] / y_freq)) - 0.5) * y_deform_intensity)[None, None, ...]).squeeze()
#     grid_z = deform_resize(((torch.rand(int(volume.shape[0] / z_freq), int(volume.shape[1] / z_freq), int(volume.shape[2] / z_freq)) - 0.5) * z_deform_intensity)[None, None, ...]).squeeze()
#     bspline = gryds.BSplineTransformation([grid_y, grid_x, grid_z], order=1)
#     interpolator = gryds.Interpolator(volume)
#     deformed_volume = interpolator.transform(bspline)
#     return deformed_volume


def get_seg_bbox(seg: torch.tensor):
    coord_bounds_to_keep = torch.argwhere(seg != 0.)
    min_c = torch.amin(coord_bounds_to_keep, dim=0)
    max_c = torch.amax(coord_bounds_to_keep, dim=0)
    return min_c, max_c


def crop_according_to_seg(ct_scan, ct_seg):
    seg_bbox_min, seg_bbox_max = get_seg_bbox(ct_seg.squeeze())
    return ct_scan[
                ...,
                max(seg_bbox_min[0].item() - 15, 0): min(seg_bbox_max[0].item() + 15, ct_scan.shape[-3]),
                max(seg_bbox_min[1].item() - 15, 0): min(seg_bbox_max[1].item() + 15, ct_scan.shape[-2]),
                max(seg_bbox_min[2].item() - 15, 0): min(seg_bbox_max[2].item() + 15, ct_scan.shape[-1])
        ], ct_seg[
                ...,
                max(seg_bbox_min[0].item() - 15, 0): min(seg_bbox_max[0].item() + 15, ct_scan.shape[-3]),
                max(seg_bbox_min[1].item() - 15, 0): min(seg_bbox_max[1].item() + 15, ct_scan.shape[-2]),
                max(seg_bbox_min[2].item() - 15, 0): min(seg_bbox_max[2].item() + 15, ct_scan.shape[-1])
        ]


def resize3d(data, size: int):
    ...


def deform_entity(entity: torch.Tensor, deform_resolution, frequencies, intensities):
    shape = entity.shape
    upsample_shape = min(shape[-3], shape[-2], shape[-1])
    deform_resize = torch.nn.Upsample(size=(shape[-3] // deform_resolution, shape[-2] // deform_resolution, shape[-1] // deform_resolution), mode='trilinear')

    x_freq = random.random() * frequencies[0][0] + frequencies[0][1]
    y_freq = random.random() * frequencies[1][0] + frequencies[1][1]
    z_freq = random.random() * frequencies[2][0] + frequencies[2][1]
    x_deform_intensity = random.random() * intensities[0][0] + intensities[0][1]
    y_deform_intensity = random.random() * intensities[1][0] + intensities[1][1]
    z_deform_intensity = random.random() * intensities[2][0] + intensities[2][1]
    grid_x = deform_resize((torch.rand((1, 1, int(upsample_shape // x_freq), int(upsample_shape // x_freq), int(upsample_shape // x_freq))) - 0.5) * x_deform_intensity).squeeze()
    grid_y = deform_resize((torch.rand((1, 1, int(upsample_shape // y_freq), int(upsample_shape // y_freq), int(upsample_shape // y_freq))) - 0.5) * y_deform_intensity).squeeze()
    grid_z = deform_resize((torch.rand((1, 1, int(upsample_shape // z_freq), int(upsample_shape // z_freq), int(upsample_shape // z_freq))) - 0.5) * z_deform_intensity).squeeze()
    bspline = gryds.BSplineTransformationCuda([grid_z, grid_y, grid_x], order=1)
    interpolator = gryds.BSplineInterpolatorCuda(entity.squeeze())

    torch.cuda.empty_cache()
    gc.collect()

    deformed_entity = torch.tensor(interpolator.transform(bspline)).to(DEVICE)
    # blr_fac = random.randint(0, 5) * 2 + 11
    # opac = torch.nn.functional.avg_pool2d(opac, kernel_size=blr_fac, stride=1, padding=blr_fac // 2)
    return deformed_entity


def get_masses(lungs_seg: torch.Tensor, num_masses: int, radius_range: Tuple[int, int]):
    lungs_seg = torch.nn.functional.pad(lungs_seg, (radius_range[1], radius_range[1], radius_range[1], radius_range[1], radius_range[1], radius_range[1]))
    masses = torch.zeros_like(lungs_seg, dtype=torch.float32).to(DEVICE)

    if num_masses == 0:
        masses = masses[radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1]]
        return masses

    lungs_coords = lungs_seg.nonzero()

    for m in range(num_masses):
        radius = random.randint(radius_range[0], radius_range[1])
        coord = lungs_coords[random.randint(0, lungs_coords.shape[0] - 1)]
        mass = torch.tensor(ball(radius, dtype=float), dtype=torch.float32).to(DEVICE)
        masses[coord[0] - radius: coord[0] + radius + 1, coord[1] - radius: coord[1] + radius + 1, coord[2] - radius: coord[2] + radius + 1] = mass

    masses = masses[radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1], radius_range[1]: -radius_range[1]]

    return masses


def get_masses_intensity_map(lungs_seg):
    upsample_shape = min(lungs_seg.shape[-3], lungs_seg.shape[-2], lungs_seg.shape[-1])
    upsample = torch.nn.Upsample(size=(upsample_shape, upsample_shape, upsample_shape), mode='trilinear')
    int_map_inv_freq = random.randint(1, 3)
    if int_map_inv_freq > 1:
        intensity_map = (torch.rand((1, 1, upsample_shape // int_map_inv_freq, upsample_shape // int_map_inv_freq, upsample_shape // int_map_inv_freq)) * 180. + 5.).to(torch.float32).to(DEVICE)
        intensity_map = upsample(intensity_map).squeeze()
    else:
        intensity_map = (torch.rand((upsample_shape, upsample_shape, upsample_shape)) * 180. + 5.).to(torch.float32).to(DEVICE)
    intensity_map = torch.nn.functional.pad(intensity_map, (0, lungs_seg.shape[-1] - intensity_map.shape[-1], 0, lungs_seg.shape[-2] - intensity_map.shape[-2], 0, lungs_seg.shape[-3] - intensity_map.shape[-3]))
    return intensity_map


def add_masses(ct_scan, masses, intensity_map):
    masses_coords = masses != 0.
    ct_scan[masses_coords] = intensity_map[masses_coords]
    ct_scan_avg = torch.nn.functional.avg_pool3d(ct_scan[None, None, ...], kernel_size=3, stride=1, padding=1).squeeze()
    ct_scan[masses_coords] = ct_scan_avg[masses_coords]
    return ct_scan


def create_masses_pair(ct_scan, lungs_seg):
    num_nodules_prior = max(random.randint(-3, 7), 0)
    num_nodules_current = max(random.randint(-3, 7), 0)
    num_nodules_both = max(random.randint(-3, 7), 0)

    radius_range = (10, 25)

    prior_masses = get_masses(lungs_seg, num_masses=num_nodules_prior, radius_range=radius_range)
    current_masses = get_masses(lungs_seg, num_masses=num_nodules_current, radius_range=radius_range)
    both_masses = get_masses(lungs_seg, num_masses=num_nodules_both, radius_range=radius_range)

    if torch.any(prior_masses != 0) or torch.any(current_masses != 0) or torch.any(both_masses != 0):
        combined_masses = prior_masses.clone()
        combined_masses[current_masses != 0.] = 2.
        combined_masses[both_masses != 0.] = 3.

        combined_masses = deform_entity(combined_masses, deform_resolution=8, frequencies=((15, 15), (15, 15), (15, 15)), intensities=((0.05, 0.025), (0.05, 0.025), (0.05, 0.025))).squeeze()

        combined_masses = torch.round(combined_masses)

        prior_masses = torch.zeros_like(combined_masses)
        current_masses = torch.zeros_like(combined_masses)
        both_masses = torch.zeros_like(combined_masses)
        prior_masses[combined_masses == 1.] = 1.
        current_masses[combined_masses == 2.] = 1.
        both_masses[combined_masses == 3.] = 1.

        del combined_masses
        combined_masses = None

        torch.cuda.empty_cache()
        gc.collect()

        prior_masses *= lungs_seg
        current_masses *= lungs_seg
        both_masses *= lungs_seg

    prior_masses = torch.maximum(prior_masses, both_masses)
    current_masses = torch.maximum(current_masses, both_masses)

    del both_masses
    both_masses = None

    torch.cuda.empty_cache()
    gc.collect()

    intensity_map = get_masses_intensity_map(lungs_seg)

    prior = ct_scan.clone()
    current = ct_scan.clone()
    prior = add_masses(prior, prior_masses, intensity_map)
    current = add_masses(current, current_masses, intensity_map)

    return prior, current


def enforce_ndim_4(t: torch.Tensor):
    c_dim = t.ndim
    while c_dim < 4:
        t = t.unsqueeze(0)
        c_dim = t.ndim
    while c_dim > 4:
        t = t.squeeze(0)
        c_dim = t.ndim
    return t


def sign(x):
    if x == 0:
        return 0
    return x / abs(x)


def remove_small_ccs(im, min_count=16):
    ccs, ccs_num = label(im.cpu().squeeze(), struct)
    ccs = torch.from_numpy(ccs).to(DEVICE)
    unique_vals, unique_counts = torch.unique(ccs, return_counts=True)
    unique_vals = unique_vals[unique_counts > min_count]
    ccs = ccs[None, None, ...]
    ccs_indic = torch.isin(ccs, unique_vals)
    new_im = im * ccs_indic
    new_ccs = ccs * ccs_indic
    ccs_num = torch.numel(unique_vals) - 1
    return new_im, new_ccs, ccs_num


# def random_rotate_ct_and_crop(ct_scan: torch.Tensor, ct_seg=None, return_ct_seg=False, rot_ranges_max=(10, 20, 20), rot_ranges_min=(0, 0, 0)):
# def random_rotate_ct_and_crop(ct_scan: torch.Tensor, ct_seg=None, return_ct_seg=False, rot_ranges=(10, 20, 20), min_rot_angles=(0, 0, 0), prev_rot_ranges=None, return_rot_angles=False):
def random_rotate_ct_and_crop(ct_scan: torch.Tensor, ct_seg=None, return_ct_seg=False, rot_ranges=(20, 20, 20), max_angles_sum=40, prev_rot_ranges=None, return_rot_angles=False):
    # rand1 = random.random() * 2 - 1
    # rand2 = random.random() * 2 - 1
    # rand3 = random.random() * 2 - 1
    # rotate_angle1 = torch.tensor((rand1 * (rot_ranges_max[0] - rot_ranges_min[0])) + (rot_ranges_min[0] * sign(rand1)), dtype=torch.float32).to(DEVICE)
    # rotate_angle2 = torch.tensor((rand2 * (rot_ranges_max[1] - rot_ranges_min[1])) + (rot_ranges_min[1] * sign(rand2)), dtype=torch.float32).to(DEVICE)
    # rotate_angle3 = torch.tensor((rand3 * (rot_ranges_max[2] - rot_ranges_min[2])) + (rot_ranges_min[2] * sign(rand3)), dtype=torch.float32).to(DEVICE)
    if rot_ranges[0] > 0:
        rotate_angle1 = torch.tensor(random.random() * (rot_ranges[0] * 2) - rot_ranges[0], dtype=torch.float32).to(DEVICE)
        rotate_angle2 = torch.tensor(random.random() * (rot_ranges[1] * 2) - rot_ranges[1], dtype=torch.float32).to(DEVICE)
        rotate_angle3 = torch.tensor(random.random() * (rot_ranges[2] * 2) - rot_ranges[2], dtype=torch.float32).to(DEVICE)

        if max_angles_sum is not None:
            while rotate_angle1.item() + rotate_angle2.item() + rotate_angle3.item() > max_angles_sum:
                rotate_angle1 = torch.tensor(random.random() * (rot_ranges[0] * 2) - rot_ranges[0], dtype=torch.float32).to(DEVICE)
                rotate_angle2 = torch.tensor(random.random() * (rot_ranges[1] * 2) - rot_ranges[1], dtype=torch.float32).to(DEVICE)
                rotate_angle3 = torch.tensor(random.random() * (rot_ranges[2] * 2) - rot_ranges[2], dtype=torch.float32).to(DEVICE)

        # if prev_rot_ranges is not None:
        #     # while abs(rotate_angle1 - prev_rot_ranges[0]) <= rot_ranges[0] and abs(rotate_angle2 - prev_rot_ranges[1]) <= rot_ranges[1] and abs(rotate_angle3 - prev_rot_ranges[2]) <= rot_ranges[2]:
        #     #     rotate_angle1 = torch.tensor(random.random() * (rot_ranges[0] * 2) - rot_ranges[0], dtype=torch.float32).to(DEVICE)
        #     #     rotate_angle2 = torch.tensor(random.random() * (rot_ranges[1] * 2) - rot_ranges[1], dtype=torch.float32).to(DEVICE)
        #     #     rotate_angle3 = torch.tensor(random.random() * (rot_ranges[2] * 2) - rot_ranges[2], dtype=torch.float32).to(DEVICE)
        #     while abs(rotate_angle1 - prev_rot_ranges[0]) <= min_rot_angles[0]:
        #         rotate_angle1 = torch.tensor(random.random() * (rot_ranges[0] * 2) - rot_ranges[0], dtype=torch.float32).to(DEVICE)
        #     while abs(rotate_angle2 - prev_rot_ranges[1]) <= min_rot_angles[1]:
        #         rotate_angle2 = torch.tensor(random.random() * (rot_ranges[1] * 2) - rot_ranges[1], dtype=torch.float32).to(DEVICE)
        #     while abs(rotate_angle3 - prev_rot_ranges[2]) <= min_rot_angles[2]:
        #         rotate_angle3 = torch.tensor(random.random() * (rot_ranges[2] * 2) - rot_ranges[2], dtype=torch.float32).to(DEVICE)
    else:
        # rotate_angle1 = torch.tensor(0, dtype=torch.float32).to(DEVICE)
        # rotate_angle2 = torch.tensor(0, dtype=torch.float32).to(DEVICE)
        # rotate_angle3 = torch.tensor(0, dtype=torch.float32).to(DEVICE)
        rotate_angle1 = 0
        rotate_angle2 = 0
        rotate_angle3 = 0

    ct_scan = enforce_ndim_4(ct_scan)
    if ct_seg is not None:
        ct_seg = enforce_ndim_4(ct_seg)
        ct_scan = torch.cat([ct_scan, ct_seg], dim=0)

        del ct_seg
        ct_seg = 1

        torch.cuda.empty_cache()
        gc.collect()

    if rot_ranges[0] > 0:
        ct_scan = ct_scan.unsqueeze(0)
        ct_scan = rotate3d(ct_scan, rotate_angle1, rotate_angle2, rotate_angle3)
        ct_scan = ct_scan.squeeze(0)

    if ct_seg is not None:
        ct_scan, ct_seg = crop_according_to_seg(ct_scan[:-1], torch.round(ct_scan[-1:]))

    to_ret = [ct_scan]
    if return_ct_seg:
        to_ret.append(ct_seg)
    if return_rot_angles:
        to_ret.append((rotate_angle1, rotate_angle2, rotate_angle3))
    return to_ret


def project_ct(rotated_ct_scan, dim=2):
    xray_image = torch.mean(rotated_ct_scan, dim=dim)
    xray_image = resize(xray_image[None, ...]).squeeze()
    xray_image = xray_image.cpu().numpy()

    return xray_image


def normalize_xray(xray_image):
    xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image))
    # xray_image, cdf = image_histogram_equalization(xray_image)
    xray_image = kornia.enhance.equalize_clahe(torch.tensor(xray_image), clip_limit=1.2, grid_size=(16, 16)).numpy()
    xray_image = xray_image.astype(np.float32).T

    return xray_image


def generate_randomly_rotated_DDRs_and_save(ct_scan, ct_seg, out_img_dir, num=1390):
    # orig_ct_scan = crop_according_to_seg(ct_scan, ct_seg)[None, None, ...]
    ct_scan = ct_scan[None, ...]
    ct_seg = ct_seg[None, ...].to(torch.float32)

    for i in tqdm(range(num)):
        # rotate_angle1 = torch.tensor(random.random() * 15 - 7.5, dtype=torch.float32).to(DEVICE)
        # rotate_angle2 = torch.tensor(random.random() * 50 - 25, dtype=torch.float32).to(DEVICE)
        # rotate_angle3 = torch.tensor(random.random() * 50 - 25, dtype=torch.float32).to(DEVICE)
        # # rotate_angle1 = torch.tensor(0., dtype=torch.float32).to(DEVICE)
        # # rotate_angle2 = torch.tensor(0., dtype=torch.float32).to(DEVICE)
        # # rotate_angle3 = torch.tensor(0., dtype=torch.float32).to(DEVICE)
        # rotated_ct_scan = rotate3d(orig_ct_scan, rotate_angle1, rotate_angle2, rotate_angle3)
        # rotated_ct_seg = rotate3d(ct_seg, rotate_angle1, rotate_angle2, rotate_angle3)
        # rotated_ct_scan = crop_according_to_seg(rotated_ct_scan, rotated_ct_seg)[0]

        rotated_ct_scan = random_rotate_ct_and_crop(ct_scan, ct_seg)

        # rotated_ct_scan[rotated_ct_scan == 0] = torch.min(rotated_ct_scan)
        rotated_ct_scan = rotated_ct_scan.squeeze()

        xray_image = normalize_xray(project_ct(rotated_ct_scan))

        save_xray_image(xray_image, f'{out_img_dir}/var{i}.nii.gz')

        if i == 5:
            exit()

    # return xray_image


def save_xray_image(xray_image, output_path):
    xray_nif = nib.Nifti1Image(xray_image, affine)
    nib.save(xray_nif, output_path)

    # plt.imsave(output_path, xray_image, cmap='gray')


def generate_random_masses_pairs(scans_dir, segs_dir, out_dir_test, num_pairs):
    # scans_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/test_scans'
    # segs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/test_scans_segs'
    # # orig_out_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test/new'
    # # out_dir_train = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_train'
    # out_dir_test = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs_test'

    # num_pairs = 39

    print(f'Working on dirs:\nscans_dir: {scans_dir}\nsegs_dir: {segs_dir}\nout_dir: {out_dir_test}\nnum_pairs: {num_pairs}\n')

    # num_pairs = 560
    # rot_ranges_lst = [(0, 0, 0), (3, 6, 6), (6, 12, 12), (10, 20, 20), (12, 24, 24), (15, 30, 30), (18, 36, 36), (22, 44, 44)]
    # rot_ranges_lst = [((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (5, 10, 10)), ((5, 10, 10), (15, 25, 25)), ((15, 25, 25), (25, 40, 40))]
    # rot_ranges_lst = [((0, 0, 0), 0), ((10, 10, 10), 20), ((20, 20, 20), 40)]
    rot_ranges_lst = [((15, 15, 15), 15)]
    # rot_ranges_lst = [((0, 0, 0), (15, 25, 25))]
    scan_names = os.listdir(scans_dir)

    # train_names = set([n + '.nii.gz' for n in os.listdir(out_dir_train)])
    # train_names = set(random.sample(scan_names, 90))
    # print(train_names)
    # to_flip = set([f'volume-{i}' for i in [13, 14, 15, 16, 17, 18, 26, 27, 4, 48, 49, 50, 51, 52, 68]])
    to_flip = {'case2', 'case6', 'case7_12_10_2012', 'case9', 'case11'}

    # print('Working on third 3 of cases')

    flag = False

    # for i in range(0, len(scan_names) // 3):
    # for i in range(len(scan_names) // 3, 2 * len(scan_names) // 3):
    # for rot_angles_min, rot_ranges in rot_ranges_lst:
    for rot_ranges, max_sum in rot_ranges_lst:
        print(f'Working on rot_ranges_max {rot_ranges}')
        # out_dir = f'{out_dir_test}/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}'
        out_dir = out_dir_test
        for i in range(len(scan_names)):
            scan_name = scan_names[i]
            
            if scan_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.146429221666426688999739595820.nii.gz':
                flag = True

            if not flag:
                continue

            # if scan_name in train_names:
            #     continue

            # out_dir = out_dir_test

            print(f'Working on scan {scan_name}, index = {i}')

            scan_path = f'{scans_dir}/{scan_name}'
            seg_name = f'{scan_name[:-7]}_seg.nii.gz'
            seg_path = f'{segs_dir}/{seg_name}'
            out_img_dir = f'{out_dir}/{scan_name[:-7]}'
            os.makedirs(out_img_dir, exist_ok=True)

            ct_scan, lungs_seg = load_scan_and_seg(scan_path, seg_path)

            for pair_num in tqdm(range(0, num_pairs)):
                pair_dir = f'{out_img_dir}/pair{pair_num}'
                os.makedirs(pair_dir, exist_ok=True)

                # ct_scan_aug = add_masses(ct_scan, ct_seg)
                prior, current = create_masses_pair(ct_scan.to(DEVICE), lungs_seg.to(DEVICE))

                torch.cuda.empty_cache()
                gc.collect()

                ct_cat = torch.stack([prior, current], dim=0)

                del prior
                del current
                prior = None
                current = None

                torch.cuda.empty_cache()
                gc.collect()

                # rotated_ct_cat, current_rot_angles = random_rotate_ct_and_crop(ct_cat, lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, return_rot_angles=True)
                rotated_ct_cat = random_rotate_ct_and_crop(ct_cat, lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, return_rot_angles=False)[0]

                current_ct = rotated_ct_cat[1]
                registrated_prior_ct = rotated_ct_cat[0]

                current = project_ct(current_ct)
                registrated_prior = project_ct(registrated_prior_ct)

                # save_xray_image(np.flip(np.transpose(current_ct.cpu().numpy().astype(np.int32), (1, 2, 0)), axis=2), f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs/volume-127/current_ct.nii.gz')

                del current_ct
                del registrated_prior_ct
                del rotated_ct_cat
                current_ct = None
                registrated_prior_ct = None
                rotated_ct_cat = None

                torch.cuda.empty_cache()
                gc.collect()

                # rotated_ct_scan = random_rotate_ct_and_crop(ct_cat[0], lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, prev_rot_ranges=current_rot_angles)[0]
                # rotated_ct_scan = random_rotate_ct_and_crop(ct_cat[0], lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, min_rot_angles=(0, 0, 0), prev_rot_ranges=None)[0]
                # rotated_ct_scan = random_rotate_ct_and_crop(ct_cat[0], lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, min_rot_angles=rot_angles_min, prev_rot_ranges=current_rot_angles)[0]
                rotated_ct_scan = random_rotate_ct_and_crop(ct_cat[0], lungs_seg.to(DEVICE), return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, return_rot_angles=False)[0]

                prior_ct = rotated_ct_scan[0]

                prior = project_ct(prior_ct)

                # save_xray_image(np.flip(np.transpose(prior_ct.cpu().numpy().astype(np.int32), (1, 2, 0)), axis=2), f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/synthetic_pairs/volume-127/prior_ct.nii.gz')

                del prior_ct
                prior_ct = None

                del ct_cat
                ct_cat = None

                del rotated_ct_scan
                rotated_ct_scan = None

                torch.cuda.empty_cache()
                gc.collect()

                # projected_seg = project_ct(rotated_ct_seg, use_max=True)

                raw_diff = (current - registrated_prior).T
                current = (current - np.min(current)) / (np.max(current) - np.min(current))
                registrated_prior = (registrated_prior - np.min(registrated_prior)) / (np.max(registrated_prior) - np.min(registrated_prior))
                diff_map = (current - registrated_prior).T
                diff_map[np.abs(raw_diff) < 0.025] = 0.
                diff_map[np.abs(diff_map) < 0.008] = 0.
                # diff_map[~(projected_seg.T != 0.)] = 0.

                diff_map_th = diff_map != 0.
                diff_map_th, _, __ = remove_small_ccs(torch.tensor(diff_map_th).to(DEVICE), min_count=80)
                diff_map_th = diff_map_th.cpu().squeeze().numpy()
                diff_map[diff_map_th == 0.] = 0.

                current = normalize_xray(current)
                prior = normalize_xray(prior)

                torch.cuda.empty_cache()
                gc.collect()

                if scan_name[:-7] in to_flip:
                    current = np.flip(current, axis=1)
                    prior = np.flip(prior, axis=1)
                    diff_map = np.flip(diff_map, axis=1)

                save_xray_image(current, f'{pair_dir}/current.nii.gz')
                save_xray_image(prior, f'{pair_dir}/prior.nii.gz')
                save_xray_image(diff_map, f'{pair_dir}/difference_map.nii.gz')
                # save_xray_image(projected_seg.T, f'{out_img_dir}/seg.nii.gz')

                # plot_diff_on_current(diff_map, current, f'{pair_dir}/diff_map.png')


def main():
    scans_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'
    segs_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs'
    out_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/DRRs2'

    num_scans = len(os.listdir(scans_dir))

    for j, scan_name in enumerate(os.listdir(scans_dir)):
        print(f'Starting to work on CT scan: {scan_name}')
        print(f'Number {j} / {num_scans}')

        scan_path = f'{scans_dir}/{scan_name}'
        seg_name = f'{scan_name.split(".")[0]}_seg.nii.gz'
        seg_path = f'{segs_dir}/{seg_name}'
        out_img_dir = f'{out_dir}/{scan_name.split(".")[0]}'
        os.makedirs(out_img_dir, exist_ok=True)

        # Load the CT scan
        ct_scan, ct_seg = load_scan_and_seg(scan_path, seg_path)

        # Convert to X-ray-like image
        generate_randomly_rotated_DDRs_and_save(ct_scan, ct_seg, out_img_dir)

        # # Save the result
        # save_xray_image(xray_image, output_file)
        # print(f"Chest X-ray image saved as {output_file}")


def check_memory():
    device = 0
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    print(mem_used_MB)


resize = v2.Resize((512, 512))
if __name__ == "__main__":
    from math import ceil

    struct = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))
    # main()
    scan_dirs = [
                # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/scans/GE MEDICAL SYSTEMS',
                #  '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/scans/Philips',
                 '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/scans/SIEMENS',
                 # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_models/scans/Brilliance 16P',
                 # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_models/scans/LightSpeed Pro 16',
                 # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_models/scans/LightSpeed16',
                 # '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_models/scans/Sensation 16'
                 ]
    seg_dirs = [d.replace('/scans/', '/segs/') for d in scan_dirs]
    out_dirs = [d.replace('/scans/', '/synthetic_pairs_train/') for d in scan_dirs]
    # nums_pairs = [ceil(500 / len(os.listdir(d))) for d in scan_dirs]
    # nums_pairs = [ceil(500 / len(os.listdir(d))) for d in scan_dirs]
    nums_pairs = [ceil(5556 / len(os.listdir(d))) for d in scan_dirs]

    for sc_d, sg_d, out_d, n_p in zip(scan_dirs, seg_dirs, out_dirs, nums_pairs):
        generate_random_masses_pairs(sc_d, sg_d, out_d, n_p)

    # export LD_LIBRARY_PATH=/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/venv_new/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
