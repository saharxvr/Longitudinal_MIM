import random
from kornia.geometry import rotate3d
import torch
from typing import Optional
from DRR_utils import get_random_sign, enforce_ndim_4, crop_according_to_seg
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import DEVICE


def get_random_rotation_angles(absolute_value_range_per_axis=(15, 15, 15), max_angles_sum=30, min_angles_sum=7.5, exponent=4.):
    if absolute_value_range_per_axis != (0, 0, 0):
        angles_sum = -1
        while angles_sum > max_angles_sum or angles_sum < min_angles_sum:
            rotate_angle1 = torch.tensor((random.random() ** exponent) * get_random_sign() * absolute_value_range_per_axis[0]).to(DEVICE)
            rotate_angle2 = torch.tensor((random.random() ** exponent) * get_random_sign() * absolute_value_range_per_axis[1]).to(DEVICE)
            rotate_angle3 = torch.tensor((random.random() ** exponent) * get_random_sign() * absolute_value_range_per_axis[2]).to(DEVICE)

            angles_sum = (rotate_angle1.abs() + rotate_angle2.abs() + rotate_angle3.abs()).item()
    else:
        rotate_angle1 = 0
        rotate_angle2 = 0
        rotate_angle3 = 0

    return rotate_angle1, rotate_angle2, rotate_angle3


def rotate_ct_and_crop_according_to_seg(
        ct_scan: torch.Tensor,
    ct_seg: Optional[torch.Tensor] = None,
        *,
        rotate_angle1=0,
        rotate_angle2=0,
        rotate_angle3=0,
        return_ct_seg: bool = False,
):
    """Rotate a CT volume (and optionally its segmentation) by provided angles.

    This is the deterministic counterpart of `random_rotate_ct_and_crop_according_to_seg`.
    Angles are in degrees and can be Python numbers or torch scalars.
    """
    ct_scan = ct_scan.to(DEVICE)
    if ct_seg is not None:
        ct_seg = ct_seg.to(DEVICE)

    ct_scan = enforce_ndim_4(ct_scan)

    # Convert angles to tensors on the correct device
    if not torch.is_tensor(rotate_angle1):
        rotate_angle1 = torch.tensor(float(rotate_angle1), device=DEVICE)
    else:
        rotate_angle1 = rotate_angle1.to(DEVICE)
    if not torch.is_tensor(rotate_angle2):
        rotate_angle2 = torch.tensor(float(rotate_angle2), device=DEVICE)
    else:
        rotate_angle2 = rotate_angle2.to(DEVICE)
    if not torch.is_tensor(rotate_angle3):
        rotate_angle3 = torch.tensor(float(rotate_angle3), device=DEVICE)
    else:
        rotate_angle3 = rotate_angle3.to(DEVICE)

    # Rotate
    if float(rotate_angle1) != 0.0 or float(rotate_angle2) != 0.0 or float(rotate_angle3) != 0.0:
        ct_scan = ct_scan.unsqueeze(0)
        min_val = ct_scan.min()
        ct_scan = rotate3d(ct_scan - min_val, rotate_angle1, rotate_angle2, rotate_angle3, padding_mode='reflection')
        ct_scan += min_val
        ct_scan = ct_scan.squeeze(0)

        if ct_seg is not None:
            ct_seg = enforce_ndim_4(ct_seg).unsqueeze(0)
            ct_seg = rotate3d(ct_seg, rotate_angle1, rotate_angle2, rotate_angle3, padding_mode='zeros')
            ct_seg = torch.round(ct_seg).squeeze(0)

    # Crop according to seg if provided
    if ct_seg is not None:
        ct_seg = enforce_ndim_4(ct_seg)
        ct_scan, ct_seg, _ = crop_according_to_seg(ct_scan, ct_seg, {'seg': ct_seg}, tight_y=False, ext=7)
        ct_seg = ct_seg['seg']

    to_ret = [ct_scan]
    if return_ct_seg:
        to_ret.append(ct_seg)
    return to_ret


def random_rotate_ct_and_crop_according_to_seg(
        ct_scan: torch.Tensor,
        ct_seg=None,
        return_ct_seg: bool = False,
        rot_ranges=(15, 15, 15),
        max_angles_sum=30,
        min_angles_sum=7.5,
        exponent=1.,
        return_angles: bool = False,
):
    rotate_angle1, rotate_angle2, rotate_angle3 = get_random_rotation_angles(rot_ranges, max_angles_sum, min_angles_sum, exponent)
    out = rotate_ct_and_crop_according_to_seg(
        ct_scan,
        ct_seg,
        rotate_angle1=rotate_angle1,
        rotate_angle2=rotate_angle2,
        rotate_angle3=rotate_angle3,
        return_ct_seg=return_ct_seg,
    )
    if return_angles:
        out.append((rotate_angle1, rotate_angle2, rotate_angle3))
    return out