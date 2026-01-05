import torch
import random

def enforce_ndim_4(t: torch.Tensor):
    c_dim = t.ndim
    while c_dim < 4:
        t = t.unsqueeze(0)
        c_dim = t.ndim
    while c_dim > 4:
        t = t.squeeze(0)
        c_dim = t.ndim
    return t


def softmax_and_rescale(t: torch.Tensor, mult=1.):
    orig_max = t.max()
    orig_mask = t > 0.01
    exp_t = torch.exp(t * mult)
    t = exp_t / (torch.sum(exp_t))
    t /= t.max()
    t *= orig_max
    t *= orig_mask
    return t

def get_random_sign():
    return random.randint(0, 1) * 2 - 1

def crop_according_to_seg(ct_scan, cropping_seg, all_segs_dict, tight_y=False, ext=15, h_low_ext=15):
    seg_bbox_min, seg_bbox_max = get_seg_bbox(cropping_seg.squeeze())

    slice_1, slice_2, slice_3 = get_cropping_slices(seg_bbox_min, seg_bbox_max, ct_scan, tight_y, ext=ext, h_low_ext=h_low_ext)

    cropped_ct_scan = ct_scan[..., slice_1, slice_2, slice_3]
    cropped_segs_dict = {}
    for organ_name, seg in all_segs_dict.items():
        cropped_segs_dict[organ_name] = seg[..., slice_1, slice_2, slice_3]

    return cropped_ct_scan, cropped_segs_dict, (slice_1, slice_2, slice_3)

def get_seg_bbox(seg: torch.tensor):
    coord_bounds_to_keep = torch.argwhere(seg == 1)
    min_c = torch.amin(coord_bounds_to_keep, dim=0)
    max_c = torch.amax(coord_bounds_to_keep, dim=0)
    return min_c, max_c

def get_cropping_slices(seg_bbox_min, seg_bbox_max, ct_scan, tight_y, ext=15, h_low_ext=15):
    slice_1 = slice(max(seg_bbox_min[-3].item() - ext, 0), min(seg_bbox_max[-3].item() + h_low_ext, ct_scan.shape[-3]))
    slice_2 = slice(max(seg_bbox_min[-2].item() - ext, 0), min(seg_bbox_max[-2].item() + ext, ct_scan.shape[-2]))

    if tight_y:
        slice_3 = slice(max(seg_bbox_min[-1].item() - ext, 0), min(seg_bbox_max[-1].item() + ext, ct_scan.shape[-1]))
    else:
        slice_3 = slice(None, None)

    return slice_1, slice_2, slice_3

def add_back_cropped(ct_scan, orig_ct_scan, c_slices):
    dev = ct_scan.device
    ct_scan = ct_scan.cpu()
    orig_ct_scan = orig_ct_scan.cpu()

    new_ct_scan = orig_ct_scan.clone()

    new_ct_scan[..., c_slices[0], c_slices[1], c_slices[2]] = ct_scan
    new_ct_scan = new_ct_scan.to(dev)

    return new_ct_scan