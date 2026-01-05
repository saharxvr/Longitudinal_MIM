import math

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import nibabel as nib
from skimage.transform import rotate
from skimage.exposure import equalize_adapthist
from scipy.ndimage import center_of_mass
import scipy.ndimage as ndi
import cv2
from imreg import translation


AFFINE_DCM = np.array([
    [-0.139, 0, 0, 0],
    [0, -0.139, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    # im1 = np.asarray(im1).astype(bool)
    # im2 = np.asarray(im2).astype(bool)

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


# resize one x-ray to another
# ->
# find centroids/some other robust center point
# ->
# align centroids/center points
# ->
# move one seg in the direction that maximizes dice/jaccard/other measure, until local max
# ->
# rotate in direction that maximizes measure until local max

# src = dcmread(src_path)

# dst = dcmread(dst_path)


# def move_mask(mask: np.ndarray, dirc: tuple):
#     mask_y, mask_x = np.where(mask)
#     mask_y += np.round(dirc[0]).astype('int64')
#     mask_x += np.round(dirc[1]).astype('int64')
#     mask = np.zeros(mask.shape, dtype=np.bool)
#     mask[(mask_y, mask_x)] = True
#     return mask


def get_min_rects(seg):
    seg = (seg * 255).astype(np.uint8)
    c, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (y1, x1), (h1, w1), angle1 = cv2.minAreaRect(c[0])
    # (y2, x2), (h2, w2), angle2 = cv2.minAreaRect(c[1])
    # (y3, x3), (h3, w3), angle3 = cv2.minAreaRect(np.vstack([c[0], c[1]]))
    # return ((y1, x1), (h1, w1), angle1), ((y2, x2), (h2, w2), angle2), ((y3, x3), (h3, w3), angle3)
    return ((y1, x1), (h1, w1), angle1)#, ((y2, x2), (h2, w2), angle2)


def get_rect_angle(seg: np.ndarray):
    # (a, b, angle1), (c, d, angle2), (e, f, angle3) = get_min_rects(seg)
    # (a, b, angle1), (c, d, angle2) = get_min_rects(seg)
    (a, b, angle1) = get_min_rects(seg)
    # print(f'(x1, y1)={(x1, y1)}, angle1={angle1}, (x2, y2)={(x2, y2)} angle2={angle2}')
    # return angle1, angle2, angle3
    return angle1


# def align_pair(src_dcm_path: str, dst_dcm_path: str, are_segs_corrected=False, do_cropping=True, save_final=False, return_dice=False, separate_regions=False):
#     src_name = ".".join(src_dcm_path.split('.')[:-1])
#     dst_name = ".".join(dst_dcm_path.split('.')[:-1])
#
#     if are_segs_corrected:
#         src_seg_path = f'{src_name}_seg_corrected.nii.gz'
#         dst_seg_path = f'{dst_name}_seg_corrected.nii.gz'
#     else:
#         src_seg_path = f'{src_name}_seg.nii.gz'
#         dst_seg_path = f'{dst_name}_seg.nii.gz'
#
#     output_src_seg_path = f'{src_name}_seg_aligned.nii.gz'
#     output_src_im_path = f'{src_name}_aligned.nii.gz'
#
#     output_dst_seg_path = f'{dst_name}_seg_aligned.nii.gz'
#     output_dst_im_path = f'{dst_name}_aligned.nii.gz'
#
#     src_im = pydicom.dcmread(src_dcm_path)
#     src_im = src_im.pixel_array.T.squeeze()
#     dst_im = pydicom.dcmread(dst_dcm_path)
#     dst_im = dst_im.pixel_array.T.squeeze()
#
#     src = nib.load(src_seg_path)
#     src_seg = src.get_fdata().astype(bool).squeeze()
#     dst = nib.load(dst_seg_path)
#     dst_seg = dst.get_fdata().astype(bool).squeeze()
#
#     if src_seg.shape != dst_seg.shape:
#         print("padding")
#         max_cols = max(src_seg.shape[0], dst_seg.shape[0])
#         max_rows = max(src_seg.shape[1], dst_seg.shape[1])
#
#         src_seg = np.pad(src_seg, ((0, max_cols - src_seg.shape[0]), (0, max_rows - src_seg.shape[1])), constant_values=False)
#         src_im = np.pad(src_im, ((0, max_cols - src_im.shape[0]), (0, max_rows - src_im.shape[1])), constant_values=0)
#         dst_seg = np.pad(dst_seg, ((0, max_cols - dst_seg.shape[0]), (0, max_rows - dst_seg.shape[1])), constant_values=False)
#         dst_im = np.pad(dst_im, ((0, max_cols - dst_im.shape[0]), (0, max_rows - dst_im.shape[1])), constant_values=0)
#
#     # src_angle1, src_angle2 = get_rect_angle(src_seg)
#     dst_angle = get_rect_angle(dst_seg)
#     # angle1 = src_angle1 - dst_angle1
#     # angle2 = src_angle2 - dst_angle2
#     # angle3 = src_angle3 - dst_angle3
#
#     src_centroid = center_of_mass(src_seg)
#     dst_centroid = center_of_mass(dst_seg)
#
#     cur_dice = -1
#
#     src_seg_label, _ = ndi.label(dst_seg)
#     print(np.argwhere(src_seg_label == 1)[0])
#     print(np.argwhere(src_seg_label == 2)[0])
#
#     dst_seg_label, _ = ndi.label(dst_seg)
#     print(np.argwhere(dst_seg_label == 1)[0])
#     print(np.argwhere(dst_seg_label == 2)[0])
#
#     for idx in 1, 2:
#         cur_src_seg_cc = src_seg_label == idx
#         cur_dst_seg_cc = dst_seg_label == idx
#         src_angle = get_rect_angle(cur_src_seg_cc)
#         dst_angle = get_rect_angle(cur_dst_seg_cc)
#         angle = src_angle - dst_angle
#         cur_src_seg_cc_rotated = rotate(cur_src_seg_cc, angle, center=src_centroid)
#         # nift_mid = nib.Nifti1Image(cur_src_seg_rotated.astype(int), AFFINE_DCM)
#         # nib.save(nift_mid, f'{src_name}_seg_test_angle_{angle}.nii.gz')
#
#         # dst_seg_rotated = rotate(dst_seg, -(90 - dst_angle), center=dst_centroid)
#         # dst_im_rotated = rotate(dst_im, -(90 - dst_angle), center=dst_centroid)
#
#         # src_seg_rotated_label, __ = ndi.label(cur_src_seg_rotated)
#
#         cur_transl = np.array(translation(cur_dst_seg_cc, cur_src_seg_cc_rotated))
#         print(f'translation: {cur_transl}')
#
#         new_src = rotate(src_seg, angle, center=src_centroid)
#         new_src = np.roll(new_src, cur_transl, axis=(0, 1))
#
#         angle_dice = dice(new_src, dst_seg)
#         print(angle_dice)
#
#         if angle_dice > cur_dice:
#             print('won')
#             cur_dice = angle_dice
#             transl = cur_transl
#             src_seg_aligned = new_src
#             src_im_rotated = rotate(src_im, angle, center=src_centroid)
#
#     src_centroid = np.array(center_of_mass(src_seg_aligned))
#     c_diff = dst_centroid - src_centroid
#     c_diff_abs = np.abs(c_diff)
#     dirc = np.divide(c_diff, c_diff_abs, out=np.zeros_like(dst_centroid), where=c_diff_abs!=0).astype(int)
#     print(f'dirc={dirc}')
#
#     # print(f"Starting dice: {cur_dice}")
#     shifted_transl = transl.copy()
#     print(f'shifted_transl={shifted_transl}')
#     src_shifted = np.roll(src_seg_aligned, dirc, axis=(0, 1))
#     shifted_dice = dice(src_shifted, dst_seg)
#
#     i = 0
#     while shifted_dice > cur_dice:
#         print(f'Iteration {i}. Dice improved to: {shifted_dice}')
#         src_seg_aligned = src_shifted
#         cur_dice = shifted_dice
#         shifted_transl += dirc
#         src_shifted = np.roll(src_shifted, dirc, axis=(0, 1))
#         shifted_dice = dice(src_shifted, dst_seg)
#         # print(f'New shifted dice: {shifted_dice}')
#         i += 1
#
#     print(f'final shift: {shifted_transl}')
#     src_im_aligned = np.roll(src_im_rotated, shifted_transl, axis=(0, 1))
#
#     final_src_seg = src_seg_aligned
#     final_src_im = src_im_aligned
#
#     to_ret = [final_src_seg, final_src_im]
#
#     if do_cropping:
#         x_min, y_min, x_max, y_max = get_lung_region(src_seg_aligned, dst_seg)
#         final_src_seg = src_seg_aligned[x_min: x_max, y_min: y_max]
#         final_src_im = src_im_aligned[x_min: x_max, y_min: y_max]
#         final_dst_seg = dst_seg[x_min: x_max, y_min: y_max]
#         final_dst_im = dst_im[x_min: x_max, y_min: y_max]
#
#         to_ret.append(final_dst_seg)
#         to_ret.append(final_dst_im)
#
#     if save_final:
#         if separate_regions:
#             final_src_seg = ndi.label(final_src_seg)[0]
#         src_seg_nift = nib.Nifti1Image(final_src_seg.astype(int), AFFINE_DCM)
#         src_im_nift = nib.Nifti1Image(final_src_im, AFFINE_DCM)
#         nib.save(src_seg_nift, output_src_seg_path)
#         nib.save(src_im_nift, output_src_im_path)
#
#         if do_cropping:
#             if separate_regions:
#                 final_dst_seg = ndi.label(final_dst_seg)[0]
#             dst_seg_nift = nib.Nifti1Image(final_dst_seg.astype(int), AFFINE_DCM)
#             dst_im_nift = nib.Nifti1Image(final_dst_im, AFFINE_DCM)
#             nib.save(dst_seg_nift, output_dst_seg_path)
#             nib.save(dst_im_nift, output_dst_im_path)
#
#     if return_dice:
#         to_ret.append(cur_dice)
#
#     return tuple(to_ret)

STRUCT = ndi.generate_binary_structure(2, 2)


def get_angle(seg, ret_sep_lung_masks=False):
    label, _ = ndi.label(seg, structure=STRUCT)
    ccs, counts = np.unique(label, return_counts=True)
    num_ccs = len(ccs)

    if num_ccs == 2:
        print("Warning: Only one lung found.")
        if ret_sep_lung_masks:
            lung1 = label == ccs[1]
            lung1_coords = np.argwhere(lung1)
            top1 = lung1_coords[0]
            top2 = (top1[0], seg.shape[-1] - top1[1])
            label[top2[0]: top2[0] + 3, top2[1]] = 2
            lung2 = label == 2
            return 0., lung1, lung2
        return 0.
    if num_ccs == 1:
        print("Warning: No lungs could be found.")
        if ret_sep_lung_masks:
            y = seg.shape[-2] // 8
            lung1_x = (seg.shape[-1] * 2) // 5
            lung2_x = (seg.shape[-1] * 3) // 5
            label[y: y + 3, lung1_x] = 1
            label[y: y + 3, lung2_x] = 2
            lung1 = label == 1
            lung2 = label == 2
            return 0., lung1, lung2
        return 0.
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
    # lung1_coords = np.argwhere(lung1.T)
    # lung2_coords = np.argwhere(lung2.T)
    lung1_coords = np.argwhere(lung1)
    lung2_coords = np.argwhere(lung2)
    top1 = lung1_coords[0]
    top2 = lung2_coords[0]
    diff = top2 - top1
    m = diff[0] / diff[1]
    angle = math.degrees(math.atan(m))
    if ret_sep_lung_masks:
        return angle, lung1, lung2
    return angle


def get_bounding_rect(seg):
    seg = (seg * 255).astype(np.uint8)
    c, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = np.vstack([c[i] for i in range(len(c))])
    # c = np.vstack([c[0], c[1]])
    y, x, h, w = cv2.boundingRect(c)
    return (y, x), (h, w)


def get_lung_region(src_seg: np.ndarray, dst_seg: np.ndarray):
    (src_y, src_x), (src_h, src_w) = get_bounding_rect(src_seg)
    (dst_y, dst_x), (dst_h, dst_w) = get_bounding_rect(dst_seg)
    print((src_y, src_x), (src_h, src_w))
    print((dst_y, dst_x), (dst_h, dst_w))
    x_min = max(min(src_x, dst_x) - 20, 0)
    y_min = max(min(src_y, dst_y) - 40, 0)
    x_max = min(max(src_x + src_w, dst_x + dst_w) + 20, src_seg.shape[0])
    # TODO: This might need changing once segmentations become better
    y_max = min(max(src_y + src_h, dst_y + dst_h) + 40, src_seg.shape[1])
    return x_min, y_min, x_max, y_max


def get_lung_region_single(seg: np.ndarray):
    (y, x), (h, w) = get_bounding_rect(seg)
    # x_min = max(x - 30, 0)
    # y_min = max(y - 30, 0)
    # x_max = min(x + w + 30, seg.shape[0])
    # y_max = min(y + h + 75, seg.shape[1])
    x_min = max(x - 120, 0)
    y_min = max(y - 120, 0)
    x_max = min(x + w + 120, seg.shape[0])
    y_max = min(y + h + 120, seg.shape[1])
    return x_min, y_min, x_max, y_max


def renormalize_scan(scan, new_mean, new_std):
    new_scan = ((scan - np.mean(scan)) / np.std(scan)) * new_std + new_mean
    return new_scan


def renormalize_lungs(scan: np.ndarray, seg: np.ndarray, new_mean, new_std):
    lungs = scan[seg]
    scan[seg] = ((lungs - np.mean(lungs)) / np.std(lungs)) * new_std + new_mean


def align_pair(src_dcm_path: str, dst_dcm_path: str, are_segs_corrected=False, save_final=False, return_dice=False, separate_regions=True):
    src_name = ".".join(src_dcm_path.split('.')[:-1])
    dst_name = ".".join(dst_dcm_path.split('.')[:-1])

    if are_segs_corrected:
        src_seg_path = f'{src_name}_seg_corrected.nii.gz'
        dst_seg_path = f'{dst_name}_seg_corrected.nii.gz'

        output_src_seg_path = f'{src_name}_seg_corrected_aligned.nii.gz'
        output_src_im_path = f'{src_name}_corrected_aligned.nii.gz'

        output_dst_seg_path = f'{dst_name}_seg_corrected_aligned.nii.gz'
        output_dst_im_path = f'{dst_name}_corrected_aligned.nii.gz'
    else:
        src_seg_path = f'{src_name}_seg.nii.gz'
        dst_seg_path = f'{dst_name}_seg.nii.gz'

        output_src_seg_path = f'{src_name}_seg_aligned.nii.gz'
        output_src_im_path = f'{src_name}_aligned.nii.gz'

        output_dst_seg_path = f'{dst_name}_seg_aligned.nii.gz'
        output_dst_im_path = f'{dst_name}_aligned.nii.gz'

    src_im = pydicom.dcmread(src_dcm_path)
    src_im = src_im.pixel_array.T.squeeze()
    dst_im = pydicom.dcmread(dst_dcm_path)
    dst_im = dst_im.pixel_array.T.squeeze()

    src_im_mean = np.mean(src_im)
    dst_im_mean = np.mean(dst_im)
    src_im_std = np.std(src_im)
    dst_im_std = np.std(dst_im)

    src = nib.load(src_seg_path)
    src_seg = src.get_fdata().astype(bool).squeeze()
    dst = nib.load(dst_seg_path)
    dst_seg = dst.get_fdata().astype(bool).squeeze()

    src_lungs = src_im[src_seg]
    src_lungs_mean = np.mean(src_lungs)
    src_lungs_std = np.std(src_lungs)
    print(src_lungs_mean)

    dst_lungs = dst_im[dst_seg]
    dst_lungs_mean = np.mean(dst_lungs)
    dst_lungs_std = np.std(dst_lungs)
    print(dst_lungs_mean)

    if src_seg.shape != dst_seg.shape:
        print("padding")
        max_cols = max(src_seg.shape[0], dst_seg.shape[0])
        max_rows = max(src_seg.shape[1], dst_seg.shape[1])

        src_seg = np.pad(src_seg, ((0, max_cols - src_seg.shape[0]), (0, max_rows - src_seg.shape[1])), constant_values=False)
        src_im = np.pad(src_im, ((0, max_cols - src_im.shape[0]), (0, max_rows - src_im.shape[1])), constant_values=0)
        dst_seg = np.pad(dst_seg, ((0, max_cols - dst_seg.shape[0]), (0, max_rows - dst_seg.shape[1])), constant_values=False)
        dst_im = np.pad(dst_im, ((0, max_cols - dst_im.shape[0]), (0, max_rows - dst_im.shape[1])), constant_values=0)

    src_centroid = center_of_mass(src_seg)
    dst_centroid = np.array(center_of_mass(dst_seg))

    src_angle = get_angle(src_seg)
    dst_angle = get_angle(dst_seg)

    rotated_src_seg = rotate(src_seg, -src_angle, center=src_centroid, preserve_range=True)
    rotated_src_im = rotate(src_im, -src_angle, center=src_centroid, preserve_range=True)
    rotated_dst_seg = rotate(dst_seg, -dst_angle, center=dst_centroid, preserve_range=True)
    rotated_dst_im = rotate(dst_im, -dst_angle, center=dst_centroid, preserve_range=True)

    transl = np.array(translation(rotated_dst_seg, rotated_src_seg))
    shifted_src_seg = np.roll(rotated_src_seg, transl, axis=(0, 1))
    # shifted_src_im = np.roll(rotated_src_im, transl, axis=(0, 1))

    src_centroid = center_of_mass(shifted_src_seg)

    c_diff = dst_centroid - src_centroid
    c_diff_abs = np.abs(c_diff)
    dirc = 5 * np.divide(c_diff, c_diff_abs, out=np.zeros_like(dst_centroid), where=c_diff_abs!=0).astype(int)

    cur_dice = dice(shifted_src_seg, rotated_dst_seg)
    print(f"cur_dice={cur_dice}")

    src_shifted = np.roll(shifted_src_seg, dirc, axis=(0, 1))
    shifted_dice = dice(src_shifted, rotated_dst_seg)

    t_transl = transl

    i = 0
    while shifted_dice > cur_dice:
        print(f'Iteration {i}. Dice improved to: {shifted_dice}')
        shifted_src_seg = src_shifted
        cur_dice = shifted_dice
        t_transl += dirc
        src_shifted = np.roll(src_shifted, dirc, axis=(0, 1))
        shifted_dice = dice(src_shifted, rotated_dst_seg)
        # print(f'New shifted dice: {shifted_dice}')
        i += 1

    print(f'final shift: {t_transl}')
    shifted_src_im = np.roll(rotated_src_im, t_transl, axis=(0, 1))

    # cen_diff = (dst_centroid - src_centroid).astype(int)
    # shifted_src_seg = np.roll(rotated_src_seg, cen_diff, axis=(0, 1))
    # shifted_src_im = np.roll(rotated_src_im, cen_diff, axis=(0, 1))

    x_min, y_min, x_max, y_max = get_lung_region(shifted_src_seg, rotated_dst_seg)
    final_src_seg = shifted_src_seg[x_min: x_max, y_min: y_max]
    final_src_im = shifted_src_im[x_min: x_max, y_min: y_max]
    final_dst_seg = rotated_dst_seg[x_min: x_max, y_min: y_max]
    final_dst_im = rotated_dst_im[x_min: x_max, y_min: y_max]

    final_src_im = renormalize_scan(final_src_im, src_im_mean, src_im_std)
    final_dst_im = renormalize_scan(final_dst_im, dst_im_mean, dst_im_std)

    # renormalize_lungs(final_src_im, final_src_seg, src_lungs_mean, src_lungs_std)
    # renormalize_lungs(final_dst_im, final_dst_seg, dst_lungs_mean, dst_lungs_std)

    # final_src_im = equalize_adapthist(final_src_im / np.max(final_src_im))
    # final_dst_im = equalize_adapthist(final_dst_im / np.max(final_dst_im))

    # x_max = min(max(src_x + src_w, dst_x + dst_w) + 20, src_seg.shape[0])
    # # TODO: This might need changing once segmentations become better
    # reg_y_max = max(src_y + src_h, dst_y + dst_h)
    # y_max = min(reg_y_max + round(0.3 * (reg_y_max - y_min)), src_seg.shape[1])
    if separate_regions:
        final_src_seg = ndi.label(final_src_seg)[0]
        final_dst_seg = ndi.label(final_dst_seg)[0]

    src_seg_nift = nib.Nifti1Image(final_src_seg.astype(int), AFFINE_DCM)
    src_im_nift = nib.Nifti1Image(final_src_im, AFFINE_DCM)
    nib.save(src_seg_nift, output_src_seg_path)
    nib.save(src_im_nift, output_src_im_path)
    dst_seg_nift = nib.Nifti1Image(final_dst_seg.astype(int), AFFINE_DCM)
    dst_im_nift = nib.Nifti1Image(final_dst_im, AFFINE_DCM)
    nib.save(dst_seg_nift, output_dst_seg_path)
    nib.save(dst_im_nift, output_dst_im_path)

