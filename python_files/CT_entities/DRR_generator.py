"""
Synthetic DRR Pair Generator.

Generates pairs of synthetic Digitally Reconstructed Radiographs (DRRs) from CT scans
with inserted 3D pathological entities for training longitudinal change detection models.

Usage:
    python DRR_generator.py -n 1000 -o /output/path -CO 0.3 -PL 0.2 -PN 0.1

Arguments:
    -n, --number_pairs: Number of synthetic pairs to generate (required)
    -i, --input: Input CT directories (default: predefined paths)
    -o, --output: Output directory for generated pairs
    -CO, --Consolidation: Probability of consolidation (0-1)
    -PL, --PleuralEffusion: Probability of pleural effusion (0-1)
    -PN, --Pneumothorax: Probability of pneumothorax (0-1)
    -CA, --Cardiomegaly: Probability of cardiomegaly (0-1)
    -FO, --FluidOverload: Probability of fluid overload (0-1)
    -EX, --ExternalDevices: Probability of external devices (0-1)

Output Structure:
    output_dir/
    └── case_name/
        └── pair_N/
            ├── prior.nii.gz        # Baseline DRR
            ├── current.nii.gz      # Followup DRR with changes
            └── diff_map.nii.gz     # Ground truth difference map

Supported Entities:
    - Consolidation: Lung consolidation regions
    - PleuralEffusion: Fluid in pleural space
    - Pneumothorax: Collapsed lung regions
    - Cardiomegaly: Enlarged heart silhouette
    - FluidOverload: General fluid accumulation
    - ExternalDevices: Tubes, lines, etc.
"""

import math
import random

import nibabel as nib
import torch
import numpy as np
import torchvision.transforms.v2 as v2
import kornia
from torchvision.transforms.v2.functional import adjust_sharpness
from skimage.morphology import isotropic_closing
from scipy.ndimage import label
from DRR_utils import *
from CT_Rotations import random_rotate_ct_and_crop_according_to_seg
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import sys
import gc
import json
import argparse
import shutil
import psutil

from skimage.morphology import ball

# ==============================================================================
# Memory Management Utilities
# ==============================================================================

def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def log_memory(label=""):
    """Log current memory usage."""
    mem_gb = get_memory_usage_gb()
    print(f"[Memory] {label}: {mem_gb:.2f} GB")
    return mem_gb


def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def check_memory_and_cleanup(threshold_gb=25.0, label=""):
    """
    Check memory usage and perform aggressive cleanup if above threshold.
    
    Args:
        threshold_gb: Memory threshold in GB to trigger aggressive cleanup.
        label: Label for logging.
        
    Returns:
        bool: True if memory was above threshold and cleanup was performed.
    """
    mem_gb = get_memory_usage_gb()
    if mem_gb > threshold_gb:
        print(f"[Memory Warning] {label}: {mem_gb:.2f} GB > {threshold_gb} GB threshold")
        print("[Memory] Performing aggressive cleanup...")
        
        # Force garbage collection multiple times
        for _ in range(5):
            gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Log memory after cleanup
        new_mem_gb = get_memory_usage_gb()
        print(f"[Memory] After cleanup: {new_mem_gb:.2f} GB (freed {mem_gb - new_mem_gb:.2f} GB)")
        
        # If still above threshold, try more aggressive measures
        if new_mem_gb > threshold_gb * 0.9:
            print("[Memory] Still high, performing deep cleanup...")
            import ctypes
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except:
                pass
            gc.collect()
        
        return True
    return False


def force_cleanup_tensors(*tensors):
    """Force cleanup of specific tensors."""
    for t in tensors:
        if t is not None:
            try:
                del t
            except:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from constants import DEVICE

from Cardiomegaly import Cardiomegaly
from Consolidation import Consolidation
from Pleural_Effusion import PleuralEffusion
from Pneumothorax import Pneumothorax
from Fluid_Overload import FluidOverload
from External_Devices import ExternalDevices


def parse_args():
    """
    Parses command-line arguments for the program.

    Returns:
        argparse.Namespace: An object containing all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="A program for generating pairs of synthetic DRRs with inserted 3D entities and applied 3D rotations.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-n', '--number_pairs',
        type=int,
        required=True,
        help="The total number of synthetic pairs to create (Required!)",
        metavar='INT'
    )

    parser.add_argument(
        '-i', '--input',
        nargs='+',
        type=str,
        default=['/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans', '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'],
        help="A space-separated list of directory paths that contain CTs.",
        metavar='PATH'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/train',
        help="The directory path where the synthetic pairs will be saved.",
        metavar='PATH'
    )

    parser.add_argument(
        '-CO', '-co', '--Consolidation',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-PL', '-pl', '--PleuralEffusion',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-PN', '-pn', '--Pneumothorax',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-FL', '-fl', '--FluidOverload',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-CA', '-ca', '--Cardiomegaly',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-EX', '-ex', '--ExternalDevices',
        type=float,
        default=0.,
        help="The probability at which the entity may appear in a synthetic pair.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '--default_entities',
        action='store_true',
        help="Sets all entity probabilities to the baseline values of training data generation."
    )

    parser.add_argument(
        '-d', '--decay_prob_on_add',
        type=float,
        default=1.,
        help="An exponential decay factor on the probability to add an entity that's applied any time an entity is added. Does not apply to ExternalDevices.",
        metavar='FLOAT'
    )

    parser.add_argument(
        '-r', '--rotation_params',
        nargs='+',
        type=float,
        default=[17.5, 37.5, 0., 1.75],
        help="""\
    A space-separated list of 4 floats determining the rotation parameters.
    1. The maximal absolute value of rotation (in degrees) in all axes.
    2. The maximal possible sum of rotation angles in all axes.
    3. The minimal possible sum of rotation angles in all axes.
    4. An exponent coefficient to bias the distribution of rotation angles -- Higher value = Rotation angles closer to 0.
    """,
        metavar='FLOAT_LIST'
    )

    parser.add_argument(
        '-s', '--slices_for_CTs_list',
        nargs='+',
        type=float,
        default=[0., 1.],
        help="""\
    A space-separated list of 2 floats [a, b] in the range [0, 1], used to slice the list of CTs to go over.
    The program will go over paths[int(len(paths) * a): int(len(paths) * b + 1)]
    Useful as a simple method for splitting the work over multiple processes.
    """,
        metavar='FLOAT_LIST'
    )

    parser.add_argument(
        '-m', '--memory_threshold',
        type=float,
        default=15.0,
        help="Memory threshold in GB. If RAM usage exceeds this value, aggressive cleanup will be triggered. Default: 15.0",
        metavar='FLOAT'
    )

    # Parse and return the arguments
    return parser.parse_args()


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


def load_scan_and_seg(file_path: str, segs_paths_dict: dict):
    """Load a .nii.gz medical image and return a NumPy array.
    
    Uses np.asarray(dataobj) instead of get_fdata() for memory efficiency.
    This avoids creating an extra copy of the data in memory.
    """
    scan_nif = nib.load(file_path)

    global affine
    affine = scan_nif.affine

    # Use dataobj for memory-efficient loading (avoids extra copy)
    scan_data = np.asarray(scan_nif.dataobj, dtype=np.float32)
    scan_data = torch.from_numpy(np.transpose(scan_data, (2, 0, 1)).copy())
    # Explicitly free the nifti object
    scan_nif.uncache()
    del scan_nif
    
    scan_data = torch.flip(scan_data, dims=[0])

    segs_dict = {}
    for organ_name, seg_path in segs_paths_dict.items():
        seg_nif = nib.load(seg_path)
        # Use dataobj for memory-efficient loading
        seg_data = np.asarray(seg_nif.dataobj, dtype=np.float32)
        seg_data = torch.from_numpy(np.transpose(seg_data, (2, 0, 1)).copy())
        # Explicitly free the nifti object
        seg_nif.uncache()
        del seg_nif
        
        seg_data = torch.flip(seg_data, dims=[0])
        segs_dict[organ_name] = seg_data

    return scan_data, segs_dict


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


resize = v2.Resize((512, 512))
def project_ct(rotated_ct_scan, dim=2, is_seg=False):
    if is_seg:
        xray_image = torch.amax(rotated_ct_scan, dim=dim)
    else:
        rotated_ct_scan = torch.clamp_min(rotated_ct_scan, -1000)
        m = torch.min(rotated_ct_scan)
        rotated_ct_scan -= m
        xray_image = torch.sum(rotated_ct_scan, dim=dim)

        # xray_image = torch.mean(rotated_ct_scan, dim=dim)
    xray_image = resize(xray_image[None, ...]).squeeze()
    xray_image = xray_image.cpu().numpy()

    return xray_image


def normalize_xray(xray_image):
    # kernel = torch.tensor([
    #     [0, -1, 0],
    #     [-1, 5, -1],
    #     [0, -1, 0],
    # ], dtype=torch.float32)[None, None, ...]

    xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image))

    xray_image = -np.exp(-xray_image) # TODO: Probably better!
    xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image))

    # xray_image, cdf = image_histogram_equalization(xray_image)


    # xray_image = F.conv2d(torch.tensor(xray_image)[None, None, ...], weight=kernel).squeeze().numpy() #TODO: PADDING
    # xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image))

    # xray_image = kornia.enhance.equalize_clahe(torch.tensor(xray_image), clip_limit=1.25, grid_size=(8, 8)).numpy()
    # xray_image = adjust_sharpness(torch.tensor(xray_image).unsqueeze(0), sharpness_factor=4.).numpy().squeeze()

    xray_image = adjust_sharpness(torch.tensor(xray_image).unsqueeze(0), sharpness_factor=7.).numpy().squeeze()

    # xray_image = unsharp_mask(xray_image, radius=1.0, amount=2.0)

    # xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image))
    xray_image = np.clip(xray_image, a_min=0., a_max=1.)
    return xray_image


def save_arr_as_nifti(arr, output_path):
    if type(arr) == torch.Tensor:
        arr = np.array(arr.float().cpu())
    if arr.ndim == 3:
        arr = np.flip(arr, axis=0)
        arr = np.transpose(arr, (1, 2, 0))
    xray_nif = nib.Nifti1Image(arr, affine)
    nib.save(xray_nif, output_path)


def smooth_segmentation(mask: torch.Tensor, radius=7):
    def binary_dilation(to_dilate: torch.Tensor, rad):
        d_struct = torch.tensor(ball(rad))[None, None, ...].float().to(to_dilate.device)

        dilated = torch.nn.functional.conv3d(to_dilate.squeeze()[None, None, ...], weight=d_struct, padding='same').squeeze()
        dilated[dilated > 0] = 1

        return dilated

    def binary_erosion(to_dilate: torch.Tensor, rad):
        eroded_nif = 1. - binary_dilation(1. - to_dilate, rad)

        return eroded_nif

    closed_mask = binary_dilation(mask, radius)
    closed_mask = binary_erosion(closed_mask, radius)

    return closed_mask


def calculate_diff_map(current_scan, registrated_prior_scan, boundary_seg=None):
    if boundary_seg is not None:
        current_scan = current_scan * boundary_seg
        registrated_prior_scan = registrated_prior_scan * boundary_seg
    raw_diff = current_scan - registrated_prior_scan
    gte_diff = raw_diff >= 0
    lte_diff = raw_diff <= 0
    current_scan = (current_scan - np.min(current_scan)) / (np.max(current_scan) - np.min(current_scan))
    registrated_prior_scan = (registrated_prior_scan - np.min(registrated_prior_scan)) / (np.max(registrated_prior_scan) - np.min(registrated_prior_scan))
    c_diff_map = current_scan - registrated_prior_scan

    # c_diff_map[np.abs(raw_diff) < 0.025] = 0.
    # c_diff_map[np.abs(c_diff_map) < 0.025] = 0.

    c_diff_map[np.logical_and(c_diff_map > 0, lte_diff)] = 0
    c_diff_map[np.logical_and(c_diff_map < 0, gte_diff)] = 0

    # TODO: MAKE SURE OK. CHANGED CUZ VESSEL DILATION CHANGES WERE NOT BEING CAUGHT
    c_diff_map[np.abs(raw_diff) < 0.015] = 0.
    c_diff_map[np.abs(c_diff_map) < 0.015] = 0.

    diff_map_th = c_diff_map != 0.
    diff_map_th, _, __ = remove_small_ccs(torch.tensor(diff_map_th).to(DEVICE), min_count=50)
    diff_map_th = diff_map_th.cpu().squeeze().numpy()
    c_diff_map[diff_map_th == 0.] = 0.

    return c_diff_map


def generate_alpha_map(x: torch.Tensor):
    x_abs = x.abs()
    max_val = max(torch.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val

    return alphas_map


def plot_diff_on_current(c_diff_map, c_current, out_p):
    """Plot difference map overlay on current image and save to file.
    
    This function properly manages matplotlib memory by closing all figures.
    """
    c_diff_map = torch.tensor(c_diff_map)
    c_current = torch.tensor(c_current)

    alphas = generate_alpha_map(c_diff_map)
    divnorm = colors.TwoSlopeNorm(vmin=min(torch.min(c_diff_map).item(), -0.01), vcenter=0., vmax=max(torch.max(c_diff_map).item(), 0.01))
    fig, ax = plt.subplots()
    ax.imshow(c_current.squeeze().cpu(), cmap='gray')
    imm1 = ax.imshow(c_diff_map.squeeze().cpu(), alpha=alphas, cmap=differential_grad, norm=divnorm)
    cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()

    plt.savefig(out_p)

    # Properly close figure to free memory
    plt.cla()
    plt.clf()
    plt.close(fig)
    plt.close('all')  # Ensure all figures are closed
    
    # Free the tensors
    del c_diff_map, c_current, alphas


def log_params(c_params, out_p):
    with open(out_p, "w") as f:
        json.dump(c_params, f, indent=4)


def temp_create_consolidation():
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117_one_lung_eff_scan_1.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\scans_segs\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'

    cts_dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'
    cts_dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans'
    ct_paths1 = [cts_dir1 + f'/{n}' for n in os.listdir(cts_dir1)]
    ct_paths2 = [cts_dir2 + f'/{n}' for n in os.listdir(cts_dir2)]
    ct_paths = sorted(ct_paths1 + ct_paths2)  # len = 387
    pairs_per_ct = 2

    flag = False

    rot_ranges = (0, 0, 0)
    max_sum = 0
    min_sum = 0
    rot_exp = 1

    print(f'Rotation ranges = {rot_ranges}\nMax angles sum = {max_sum}\nMin angles sum = {min_sum}\nAngle distribution exponent = {rot_exp}')

    for k, ct_p in enumerate(ct_paths):
        case_name = ct_p.split('/')[-1][:-7]
        print(f'Working on case {k}: {case_name}')

        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375':
        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.110678335949765929063942738609':
        # if case_name == 'volume-105':
        #     flag = True
        # if not flag:
        #     continue

        scan_p = ct_p
        seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
        middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
        upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
        lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
        upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
        lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
        # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_segs/heart.nii.gz'
        bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
        lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

        segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p, 'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p,
        #                'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p}
        smoothing_radius_dict = {'lungs': 4, 'bronchi': 2, 'lung_vessels': 4, 'middle_right_lobe': 4, 'upper_right_lobe': 4, 'lower_right_lobe': 4, 'upper_left_lobe': 4, 'lower_left_lobe': 4}

        scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
        orig_scan = scan.clone()

        out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/consolidation_test/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}/{case_name}'

        # seg = smooth_segmentation(seg)

        # save_arr_as_nifti(smooth_seg, r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_entities\smooth_seg.nii.gz')
        # exit()

        # if (not os.path.exists(seg_p)) or torch.sum(segs_dict['lungs']) < 20:
        #     print(f'Correcting case {case_name} !!!\n')
        #     # os.remove(seg_p)
        #
        #     combined_lobes = torch.zeros_like(scan)
        #     for lobe_seg in [segs_dict['middle_right_lobe'], segs_dict['upper_right_lobe'], segs_dict['lower_right_lobe'], segs_dict['upper_left_lobe'], segs_dict['lower_left_lobe']]:
        #         combined_lobes = torch.maximum(combined_lobes, lobe_seg)
        #     combined_lobes = smooth_segmentation(combined_lobes.to(DEVICE), radius=3)
        #     assert torch.sum(combined_lobes) > 20, 'Failed to correct'
        #     save_arr_as_nifti(combined_lobes, seg_p)
        # continue

        # cropped_scan, cropped_seg, cropped_heart, slices = crop_according_to_seg(scan, seg, heart_seg=heart, tight=True, return_slices=True)
        cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)

        to_remove = []
        for organ_name, c_seg in cropped_segs_dict.items():
            c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
            # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
            cropped_segs_dict[organ_name] = c_seg
            if torch.sum(c_seg) < 20:
                print(f"Removing abnormal seg: {organ_name}")
                to_remove.append(organ_name)
        for organ_name in to_remove:
            del cropped_segs_dict[organ_name]
            del segs_dict[organ_name]

        # save_arr_as_nifti(segs_dict['lung_vessels'], '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/vessels.nii.gz')

        # cropped_heart = cropped_heart.to(DEVICE)
        #
        # cropped_heart = torch.nn.functional.max_pool3d(cropped_heart[None, None, ...], kernel_size=51, stride=1, padding=25)
        # cropped_heart = -torch.nn.functional.max_pool3d(-cropped_heart, kernel_size=51, stride=1, padding=25)
        # cropped_heart = -torch.nn.functional.max_pool3d(-cropped_heart, kernel_size=17, stride=1, padding=8)
        # cropped_heart = torch.nn.functional.max_pool3d(cropped_heart, kernel_size=17, stride=1, padding=8).squeeze()
        #
        # # print(cropped_heart.shape)
        # # print(slices)
        # # exit()
        #
        # cropped_seg = torch.nn.functional.max_pool3d(cropped_seg[None, None, ...], kernel_size=7, stride=1, padding=3)
        # cropped_seg = -torch.nn.functional.max_pool3d(-cropped_seg, kernel_size=7, stride=1, padding=3).squeeze()
        #
        # o_prior = cropped_scan.to(DEVICE)
        # o_current = cropped_scan.to(DEVICE)
        orig_cropped_scan = cropped_scan.clone().to(DEVICE)

        prior_cropped_segs_dict = cropped_segs_dict
        current_cropped_segs_dict = {k: v.clone() for k, v in prior_cropped_segs_dict.items()}

        # seg_dicts = [{'lungs': cropped_seg, 'heart': cropped_heart}, {'lungs': cropped_seg, 'heart': cropped_heart}]
        for i in range(pairs_per_ct):
            print(f'Working on pair {i}')

            c_out_dir = f'{out_path}/pair{i}'
            os.makedirs(c_out_dir, exist_ok=True)

            # TODO: ADD orig_scan TO CONSOLIDATION
            # TODO: MAKE SURE cropped_segs_dict contains 2 DIFFERENT copies of the segs dict

            ret_dict = Consolidation.add_to_CT_pair([cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE)], [prior_cropped_segs_dict, current_cropped_segs_dict], orig_scan=orig_cropped_scan, log_params=True)
            prior, current = ret_dict['pair_scans']
            cropped_segs_dict = ret_dict['pair_segs']
            params = ret_dict['params']

            # [prior, current], _, params
            torch.cuda.empty_cache()
            gc.collect()

            prior = add_back_cropped(prior, orig_scan, cropping_slices)
            current = add_back_cropped(current, orig_scan, cropping_slices)

            # prior = add_back_cropped(prior, orig_scan, segs_dict['lungs'])
            # current = add_back_cropped(current, orig_scan, segs_dict['lungs'])

            ct_cat = torch.stack([prior, current], dim=0)

            del prior
            del current
            prior = None
            current = None

            torch.cuda.empty_cache()
            gc.collect()

            rotated_ct_cat = random_rotate_ct_and_crop_according_to_seg(ct_cat, segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)[0]

            current_ct = rotated_ct_cat[1]
            registrated_prior_ct = rotated_ct_cat[0]

            current = project_ct(current_ct)
            registrated_prior = project_ct(registrated_prior_ct)

            del current_ct
            del registrated_prior_ct
            del rotated_ct_cat
            current_ct = None
            registrated_prior_ct = None
            rotated_ct_cat = None

            torch.cuda.empty_cache()
            gc.collect()

            rotated_ct_scan = random_rotate_ct_and_crop_according_to_seg(ct_cat[0], segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)[0]

            prior_ct = rotated_ct_scan[0]
            prior = project_ct(prior_ct)

            del prior_ct
            del ct_cat
            del rotated_ct_scan
            prior_ct = None
            ct_cat = None
            rotated_ct_scan = None

            torch.cuda.empty_cache()
            gc.collect()

            diff_map = calculate_diff_map(current, registrated_prior)

            # current = normalize_xray(current)
            # prior = normalize_xray(prior)

            torch.cuda.empty_cache()
            gc.collect()

            # prior_xray = project_ct(prior)
            # prior_xray = normalize_xray(prior_xray)
            #
            # prior_xray = prior_xray.astype(np.float32).T
            #
            # current_xray = project_ct(current)
            # current_xray = normalize_xray(current_xray)
            #
            # current_xray = current_xray.astype(np.float32).T

            # idx = 4
            # adj = "VesselsCephalizationAddHUAndAvg"
            # plt.imsave(f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_DRR_exp{idx}_{adj}.png', prior_xray.T, cmap='gray')
            # save_arr_as_nifti(prior_xray, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_DRR_exp{idx}_{adj}.nii.gz')
            # save_arr_as_nifti(prior.cpu(), f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_CT_exp{idx}_{adj}.nii.gz')

            # plt.imsave(r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\new\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_DRR_t.png', prior_xray.T, cmap='gray')
            # save_arr_as_nifti(prior_xray, r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\new\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_DRR_t.nii.gz')

            save_arr_as_nifti(prior.T, f'{c_out_dir}/prior.nii.gz')
            save_arr_as_nifti(current.T, f'{c_out_dir}/current.nii.gz')
            save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

            plt.imsave(f'{c_out_dir}/prior.png', prior, cmap='gray')
            plt.imsave(f'{c_out_dir}/current.png', current, cmap='gray')

            plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')

            log_params(params, f'{c_out_dir}/params.json')

def temp_create_pleural_effusion():
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117_one_lung_eff_scan_1.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\scans_segs\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'

    cts_dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'
    cts_dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans'
    ct_paths1 = [cts_dir1 + f'/{n}' for n in os.listdir(cts_dir1)]
    ct_paths2 = [cts_dir2 + f'/{n}' for n in os.listdir(cts_dir2)]
    ct_paths = sorted(ct_paths1 + ct_paths2)  # len = 387
    pairs_per_ct = 2

    flag = False

    rot_range = 0
    rot_ranges = (rot_range, rot_range, rot_range)
    max_sum = rot_range * 2
    min_sum = rot_range / 2
    rot_exp = 1

    print(f'Rotation ranges = {rot_ranges}\nMax angles sum = {max_sum}\nMin angles sum = {min_sum}\nAngle distribution exponent = {rot_exp}')

    for k, ct_p in enumerate(ct_paths[221:]):
        case_name = ct_p.split('/')[-1][:-7]
        print(f'Working on case {k}: {case_name}')

        if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.346115813056769250958550383763' or case_name == 'ABD_LYMPH_005':
            print("Bad case. Skipping")
            continue

        out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pleural_effusion_test/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}/{case_name}'

        if os.path.exists(f'{out_path}/pair1/params.json'):
            print("Exists. Skipping.")
            continue

        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375':
        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.110678335949765929063942738609':
        # if case_name == 'volume-105':
        #     flag = True
        # if not flag:
        #     continue

        scan_p = ct_p
        seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
        middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
        upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
        lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
        upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
        lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
        # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_segs/heart.nii.gz'
        bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
        lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

        segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p, 'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p,
        #                'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p}
        smoothing_radius_dict = {'lungs': 4, 'bronchi': 2, 'lung_vessels': 4, 'middle_right_lobe': 4, 'upper_right_lobe': 4, 'lower_right_lobe': 4, 'upper_left_lobe': 4, 'lower_left_lobe': 4}

        scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
        orig_scan = scan.clone()

        # out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pleural_effusion_training/{case_name}'

        cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)

        to_remove = []
        for organ_name, c_seg in cropped_segs_dict.items():
            c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
            # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
            cropped_segs_dict[organ_name] = c_seg
            if torch.sum(c_seg) < 20:
                print(f"Removing abnormal seg: {organ_name}")
                to_remove.append(organ_name)
        for organ_name in to_remove:
            del cropped_segs_dict[organ_name]
            del segs_dict[organ_name]

        orig_cropped_scan = cropped_scan.clone().to(DEVICE)

        # seg_dicts = [{'lungs': cropped_seg, 'heart': cropped_heart}, {'lungs': cropped_seg, 'heart': cropped_heart}]
        for i in range(pairs_per_ct):
            print(f'Working on pair {i}')

            prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
            current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

            c_out_dir = f'{out_path}/pair{i}'

            if os.path.exists(f'{c_out_dir}/params.json'):
                print("Exists. Skipping.")
                continue

            registrated_prior = cropped_scan.clone().to(DEVICE)
            ret_dict = PleuralEffusion.add_to_CT_pair([cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE), registrated_prior], [prior_cropped_segs_dict, current_cropped_segs_dict], orig_scan=orig_cropped_scan, log_params=True)

            prior, current = ret_dict['pair_scans']
            prior_cropped_segs_dict, current_cropped_segs_dict = ret_dict['pair_segs']
            registrated_prior = ret_dict['registrated_prior']
            params = ret_dict['params']
            effusion_msk_in = ret_dict['msk_in']

            torch.cuda.empty_cache()
            gc.collect()

            prior = add_back_cropped(prior, orig_scan, cropping_slices)
            current = add_back_cropped(current, orig_scan, cropping_slices)
            registrated_prior = add_back_cropped(registrated_prior, orig_scan, cropping_slices)
            effusion_msk_in = add_back_cropped(effusion_msk_in, torch.zeros_like(orig_scan), cropping_slices)

            ct_cat = torch.stack([registrated_prior, current], dim=0)

            # del prior
            # del current
            # prior = None
            # current = None

            torch.cuda.empty_cache()
            gc.collect()

            seg_for_rotation = segs_dict['lungs'].clone()
            seg_for_rotation[effusion_msk_in == 1] = 2
            seg_for_rotation[effusion_msk_in == -1] = 3

            rotated_ct_cat, seg_for_rotation = random_rotate_ct_and_crop_according_to_seg(ct_cat, seg_for_rotation, return_ct_seg=True, rot_ranges=rot_ranges, max_angles_sum=max_sum, exponent=5.)

            current_ct = rotated_ct_cat[1]
            registrated_prior_ct = rotated_ct_cat[0]

            current = project_ct(current_ct)
            registrated_prior = project_ct(registrated_prior_ct)

            # TODO: Remember that devices need to be added to CTs AFTER projecting, then projecting again
            del ct_cat
            del current_ct
            del registrated_prior_ct
            del rotated_ct_cat
            ct_cat = None
            current_ct = None
            registrated_prior_ct = None
            rotated_ct_cat = None

            torch.cuda.empty_cache()
            gc.collect()

            seg_for_rotation = seg_for_rotation.squeeze()
            rotated_seg = seg_for_rotation != 0
            msk_in = seg_for_rotation
            msk_in[seg_for_rotation == 1] = 0
            msk_in[seg_for_rotation == 2] = 1
            msk_in[seg_for_rotation == 3] = -1

            # save_arr_as_nifti(rotated_seg, f'{c_out_dir}/rotated_seg_ct_{exp}.nii.gz')
            # save_arr_as_nifti(msk_in, f'{c_out_dir}/msk_in_ct_{exp}.nii.gz')

            # effusion_diff = torch.nan_to_num(torch.sum(msk_in, dim=2) / torch.sum(rotated_seg, dim=2), nan=0, posinf=0, neginf=0).cpu().squeeze()
            effusion_max_diff_val = 0.35
            effusion_diff = (torch.sum(msk_in, dim=2) / torch.max(torch.sum(rotated_seg, dim=2))).cpu().squeeze()
            effusion_diff = torch.nan_to_num(effusion_diff, 0.)
            effusion_diff = resize(effusion_diff[None, ...])
            effusion_diff = torch.nn.functional.avg_pool2d(effusion_diff, kernel_size=7, stride=1, padding=3).squeeze().numpy()
            effusion_diff *= effusion_max_diff_val

            del rotated_seg
            del msk_in
            rotated_seg = None
            msk_in = None

            torch.cuda.empty_cache()
            gc.collect()

            prior_ct = random_rotate_ct_and_crop_according_to_seg(prior, segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, exponent=5.)[0]
            prior_ct = prior_ct.squeeze()

            prior = project_ct(prior_ct)

            del prior_ct
            prior_ct = None

            torch.cuda.empty_cache()
            gc.collect()

            # diff_map = calculate_diff_map(current, registrated_prior)

            current = normalize_xray(current)
            prior = normalize_xray(prior)

            # [prior, current], _, params

            # save_arr_as_nifti(prior, f'{c_out_dir}/prior_ct_{exp}.nii.gz')

            # prior = project_ct(prior)
            # prior = normalize_xray(prior)

            # current = project_ct(current)
            # current = normalize_xray(current)

            # diff_map += effusion_diff
            diff_map = effusion_diff

            os.makedirs(c_out_dir, exist_ok=True)

            save_arr_as_nifti(prior.T, f'{c_out_dir}/prior.nii.gz')
            save_arr_as_nifti(current.T, f'{c_out_dir}/current.nii.gz')
            save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

            plt.imsave(f'{c_out_dir}/prior.png', prior, cmap='gray')
            plt.imsave(f'{c_out_dir}/current.png', current, cmap='gray')

            plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')

            log_params(params, f'{c_out_dir}/params.json')


def temp_create_pneumothorax():
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117_one_lung_eff_scan_1.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987.nii.gz'
    # scan_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\volume-117.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\CT_scans\scans_segs\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'
    # seg_p = r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_seg.nii.gz'

    cts_dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'
    cts_dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans'
    ct_paths1 = [cts_dir1 + f'/{n}' for n in os.listdir(cts_dir1)]
    ct_paths2 = [cts_dir2 + f'/{n}' for n in os.listdir(cts_dir2)]
    ct_paths = sorted(ct_paths1 + ct_paths2)  # len = 387
    pairs_per_ct = 2

    flag = False

    # TODO: CHANGE
    import platform
    computer_name = platform.node()
    comp_angle_dict = {'cas703': 25, 'cas702': 15, 'cas701': 8, 'cas605': 0}
    if computer_name in comp_angle_dict:
        rot_range = comp_angle_dict[computer_name]
    else:
        print("Computer name not in angles dict. Exiting.")
        exit()

    # rot_range = 15
    rot_ranges = (rot_range, rot_range, rot_range)
    max_sum = rot_range * 2
    min_sum = rot_range / 2
    # min_sum = 0
    rot_exp = 1

    print(f'Rotation ranges = {rot_ranges}\nMax angles sum = {max_sum}\nMin angles sum = {min_sum}\nAngle distribution exponent = {rot_exp}')

    first = None
    last = None
    ps = ct_paths[first: last]
    len_ps = len(ps)
    for k, ct_p in enumerate(ps):
        case_name = ct_p.split('/')[-1][:-7]
        print(f'Working on case {k}/{len_ps}: {case_name}')
        print(f'First = {first}, Last = {last}')

        if case_name in {'1.3.6.1.4.1.14519.5.2.1.6279.6001.336894364358709782463716339027', 'volume-105', 'ABD_LYMPH_005'}:
            print(f'Bad case. Skipping.')
            continue

        out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pneumothorax_test/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}/{case_name}'
        # out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pneumothorax_training/{case_name}'

        if os.path.exists(f'{out_path}/pair{pairs_per_ct - 1}/params.json'):
            print('Case completed. Skipping')
            continue

        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375':
        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.110678335949765929063942738609':
        # if case_name == 'volume-105':
        #     flag = True
        # if not flag:
        #     continue

        scan_p = ct_p
        seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
        middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
        upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
        lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
        upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
        lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
        # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_segs/heart.nii.gz'
        bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
        lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

        segs_p_dict = {'lungs': seg_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p,
        #                'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p}
        smoothing_radius_dict = {'lungs': 4, 'bronchi': 2, 'lung_vessels': 4, 'middle_right_lobe': 4, 'upper_right_lobe': 4, 'lower_right_lobe': 4, 'upper_left_lobe': 4, 'lower_left_lobe': 4}

        scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
        orig_scan = scan.clone()

        # out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pleural_effusion_training/{case_name}'

        cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)

        to_remove = []
        for organ_name, c_seg in cropped_segs_dict.items():
            c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
            # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
            cropped_segs_dict[organ_name] = c_seg
            if torch.sum(c_seg) < 20:
                print(f"Removing abnormal seg: {organ_name}")
                to_remove.append(organ_name)
        for organ_name in to_remove:
            del cropped_segs_dict[organ_name]
            del segs_dict[organ_name]

        orig_cropped_scan = cropped_scan.clone().to(DEVICE)

        # seg_dicts = [{'lungs': cropped_seg, 'heart': cropped_heart}, {'lungs': cropped_seg, 'heart': cropped_heart}]
        for i in range(pairs_per_ct):
            torch.cuda.empty_cache()
            gc.collect()

            print(f'Working on pair {i}')
            c_out_dir = f'{out_path}/pair{i}'

            if os.path.exists(f'{c_out_dir}/params.json'):
                print("Pair exists. Skipping.")
                continue

            os.makedirs(c_out_dir, exist_ok=True)

            prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
            current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

            registrated_prior = cropped_scan.clone().to(DEVICE)

            patient_mode = random.choices(['supine', 'erect'], weights=[0.75, 0.25])[0]

            ret_dict = Pneumothorax.add_to_CT_pair([cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE), registrated_prior], [prior_cropped_segs_dict, current_cropped_segs_dict], orig_scan=orig_cropped_scan, patient_mode=patient_mode,
                                                   log_params=True)

            prior, current = ret_dict['pair_scans']
            prior_cropped_segs_dict, current_cropped_segs_dict = ret_dict['pair_segs']
            registrated_prior = ret_dict['registrated_prior']
            params = ret_dict['params']
            pneumothorax_msk_in = ret_dict['msk_in']
            pneumothorax_sulcus_seg = ret_dict['sulcus_seg']

            torch.cuda.empty_cache()
            gc.collect()

            # save_arr_as_nifti(prior, f'{c_out_dir}/prior_ct_{exp}.nii.gz')
            # save_arr_as_nifti(current, f'{c_out_dir}/current_ct_{exp}.nii.gz')

            prior = add_back_cropped(prior, orig_scan, cropping_slices)
            current = add_back_cropped(current, orig_scan, cropping_slices)
            registrated_prior = add_back_cropped(registrated_prior, orig_scan, cropping_slices)
            pneumothorax_msk_in = add_back_cropped(pneumothorax_msk_in, torch.zeros_like(orig_scan), cropping_slices)
            pneumothorax_sulcus_seg = add_back_cropped(pneumothorax_sulcus_seg, torch.zeros_like(orig_scan), cropping_slices)

            ct_cat = torch.stack([registrated_prior, current], dim=0)

            # del prior
            # del current
            # prior = None
            # current = None

            torch.cuda.empty_cache()
            gc.collect()

            seg_for_rotation = segs_dict['lungs'].clone()
            seg_for_rotation[pneumothorax_msk_in == 1] = 4
            seg_for_rotation[pneumothorax_msk_in == -1] = 5
            seg_for_rotation[pneumothorax_sulcus_seg == 1] = 6
            seg_for_rotation[pneumothorax_sulcus_seg == -1] = 7

            rotated_ct_cat, seg_for_rotation = random_rotate_ct_and_crop_according_to_seg(ct_cat, seg_for_rotation, return_ct_seg=True, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)
            # TODO: REMEMBER THAT ROTATION WITH REFLECTION CAN CAUSE MORE SEGMENTATION AREA IN msk_in and seg_for_rotation to appear. NEED TO TAKE CARE OF.
            # TODO: DECIDE PATIENT POSITION HERE AND NOT IN ANY SPECIFIC ENTITY SO IT IS CONSISTENT BETWEEN THEM
            # TODO: ADD IN PLEURAL AND PNEUMOTHORAX AN OPTION TO USE THE EXACT SAME PARAMETERS IN A SPECIFIC LUNG
            # TODO: REMEMBER TO DIFFERENCIATE MASK_IN FROM PLEURAL AND PNEUMO

            current_ct = rotated_ct_cat[1]
            registrated_prior_ct = rotated_ct_cat[0]

            current = project_ct(current_ct)
            registrated_prior = project_ct(registrated_prior_ct)

            # TODO: Remember that devices need to be added to CTs AFTER projecting, then projecting again
            del ct_cat
            del current_ct
            del registrated_prior_ct
            del rotated_ct_cat
            ct_cat = None
            current_ct = None
            registrated_prior_ct = None
            rotated_ct_cat = None

            torch.cuda.empty_cache()
            gc.collect()

            seg_for_rotation = seg_for_rotation.squeeze()
            rotated_seg = seg_for_rotation != 0
            max_rotated_seg_sum = torch.max(torch.sum(rotated_seg, dim=2))

            msk_in = seg_for_rotation.clone()
            msk_in[~torch.isin(msk_in, torch.tensor([4, 5], device=msk_in.device))] = 0
            msk_in[msk_in == 4] = 1
            msk_in[msk_in == 5] = -1

            sulcus_msk_in = seg_for_rotation.clone()
            sulcus_msk_in[~torch.isin(sulcus_msk_in, torch.tensor([6, 7], device=sulcus_msk_in.device))] = 0
            sulcus_msk_in[sulcus_msk_in == 6] = 1
            sulcus_msk_in[sulcus_msk_in == 7] = -1
            # TODO: ADD SULCUS HANDLING

            # save_arr_as_nifti(rotated_seg, f'{c_out_dir}/rotated_seg_ct_{exp}.nii.gz')
            # save_arr_as_nifti(msk_in, f'{c_out_dir}/msk_in_ct_{exp}.nii.gz')

            # effusion_diff = torch.nan_to_num(torch.sum(msk_in, dim=2) / torch.sum(rotated_seg, dim=2), nan=0, posinf=0, neginf=0).cpu().squeeze()
            pneumothorax_max_diff_val = 0.15
            pneumothorax_diff = torch.nan_to_num(torch.sum(msk_in, dim=2) / max_rotated_seg_sum, nan=0.).cpu().squeeze()
            pneumothorax_diff = resize(pneumothorax_diff[None, ...])
            pneumothorax_diff = torch.nn.functional.avg_pool2d(pneumothorax_diff, kernel_size=7, stride=1, padding=3).squeeze().numpy()
            pneumothorax_diff *= pneumothorax_max_diff_val
            # pneumothorax_diff = softmax_and_rescale(pneumothorax_diff, mult=30.).numpy()
            # TODO: POSSIBLE MULT BY SOME MAX VAL

            sulcus_max_diff_val = 0.25
            sulcus_diff = torch.nan_to_num(torch.sum(sulcus_msk_in, dim=2) / (max_rotated_seg_sum / 2), nan=0.).cpu().squeeze()
            sulcus_diff = resize(sulcus_diff[None, ...])
            # sulcus_diff = torch.nn.functional.avg_pool2d(sulcus_diff, kernel_size=7, stride=1, padding=3).squeeze().numpy()
            sulcus_diff = sulcus_diff.squeeze().numpy()
            sulcus_diff *= sulcus_max_diff_val

            del rotated_seg
            del msk_in
            rotated_seg = None
            msk_in = None

            torch.cuda.empty_cache()
            gc.collect()

            prior_ct = random_rotate_ct_and_crop_according_to_seg(prior, segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)[0]
            prior_ct = prior_ct.squeeze()

            prior = project_ct(prior_ct)

            del prior_ct
            prior_ct = None

            torch.cuda.empty_cache()
            gc.collect()

            # TODO: MULT DIFF MAP WITH LUNG SEG SO NO DIFFS OUTSIDE OF LUNGS CUZ OF REFLECTION ROTATION
            # diff_map = calculate_diff_map(current, registrated_prior)
            diff_map = np.zeros_like(current)

            current = normalize_xray(current)
            prior = normalize_xray(prior)

            # [prior, current], _, params

            # save_arr_as_nifti(prior, f'{c_out_dir}/prior_ct_{exp}.nii.gz')

            # prior = project_ct(prior)
            # prior = normalize_xray(prior)

            # current = project_ct(current)
            # current = normalize_xray(current)

            diff_map += pneumothorax_diff
            diff_map += sulcus_diff

            plt.imsave(f'{c_out_dir}/effusion_diff_map.png', pneumothorax_diff, cmap='gray')

            save_arr_as_nifti(prior.T, f'{c_out_dir}/prior.nii.gz')
            save_arr_as_nifti(current.T, f'{c_out_dir}/current.nii.gz')
            save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

            plt.imsave(f'{c_out_dir}/prior.png', prior, cmap='gray')
            plt.imsave(f'{c_out_dir}/current.png', current, cmap='gray')
            plt.imsave(f'{c_out_dir}/diff_map.png', diff_map, cmap='gray')

            plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')

            log_params(params, f'{c_out_dir}/params.json')


def temp_create_fluid_overload():
    # cts_dir1 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans'
    cts_dir2 = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans'
    # ct_paths1 = [cts_dir1 + f'/{n}' for n in os.listdir(cts_dir1)]
    ct_paths2 = [cts_dir2 + f'/{n}' for n in os.listdir(cts_dir2)]
    # ct_paths = sorted(ct_paths1 + ct_paths2)  # len = 387
    ct_paths = sorted(ct_paths2)  # len = 387
    n_paths = len(ct_paths)
    n_total = 25000
    print(f"Number of CTs in dir: {n_paths}")
    pairs_per_ct = math.ceil(n_total / n_paths)
    print(f'Pairs to generate per CT: {pairs_per_ct}')

    flag = False

    rot_ranges = (15, 15, 15)
    max_sum = 30
    min_sum = 0
    rot_exp = 2.5

    print(f'Rotation ranges = {rot_ranges}\nMax angles sum = {max_sum}\nMin angles sum = {min_sum}\nAngle distribution exponent = {rot_exp}')

    import platform
    computer_name = platform.node()
    slicing_dict = {'cas703': (None, n_paths // 3), 'cas702': (2 * n_paths // 3, None), 'cas605': (n_paths // 3, 2 * n_paths // 3)}
    c_data_slicing = slicing_dict[computer_name]

    for k, ct_p in enumerate(ct_paths[c_data_slicing[0]: c_data_slicing[1]]):
        case_name = ct_p.split('/')[-1][:-7]
        print(f'Working on case {k}: {case_name}')

        out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/fluid_overload_training/{case_name}'

        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.106719103982792863757268101375':
        # if case_name == '1.3.6.1.4.1.14519.5.2.1.6279.6001.110678335949765929063942738609':
        # if case_name == 'volume-105':
        #     flag = True
        # if not flag:
        #     continue

        if os.path.exists(f'{out_path}/pair{pairs_per_ct - 1}/params.json'):
            print('Case completed. Skipping')
            continue

        scan_p = ct_p
        seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
        middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
        upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
        lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
        upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
        lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
        # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_segs/heart.nii.gz'
        bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
        lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p, 'lower_left_lobe': lower_left_lobe_p}
        segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p,
                       'lower_left_lobe': lower_left_lobe_p}
        # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p}
        smoothing_radius_dict = {'lungs': 4, 'bronchi': 2, 'lung_vessels': 4, 'middle_right_lobe': 4, 'upper_right_lobe': 4, 'lower_right_lobe': 4, 'upper_left_lobe': 4, 'lower_left_lobe': 4}

        scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
        orig_scan = scan.clone()

        if segs_dict['lungs'].sum() < 20:
            print(f'Found empty seg for case {case_name}. Continuing')
            continue

        # out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/fluid_overload_test/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}/{case_name}'

        # cropped_scan, cropped_seg, cropped_heart, slices = crop_according_to_seg(scan, seg, heart_seg=heart, tight=True, return_slices=True)
        cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)

        bad_seg_flag = False

        for organ_name, c_seg in cropped_segs_dict.items():
            c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
            # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
            cropped_segs_dict[organ_name] = c_seg
            if torch.sum(c_seg) < 20:
                print(f'Found empty {organ_name} for case {case_name}. Continuing')
                bad_seg_flag = True
                break

        if bad_seg_flag:
            continue

        cropped_segs_dict['lung_right'] = torch.maximum(cropped_segs_dict['middle_right_lobe'], torch.maximum(cropped_segs_dict['upper_right_lobe'], cropped_segs_dict['lower_right_lobe']))
        cropped_segs_dict['lung_left'] = torch.maximum(cropped_segs_dict['upper_left_lobe'], cropped_segs_dict['lower_left_lobe'])

        orig_cropped_scan = cropped_scan.clone().to(DEVICE)

        prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
        current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

        # seg_dicts = [{'lungs': cropped_seg, 'heart': cropped_heart}, {'lungs': cropped_seg, 'heart': cropped_heart}]
        for i in range(pairs_per_ct):
            print(f'Working on pair {i}')

            c_out_dir = f'{out_path}/pair{i}'
            os.makedirs(c_out_dir, exist_ok=True)

            if os.path.exists(f'{c_out_dir}/params.json'):
                print("Pair exists. Skipping.")
                continue

            # TODO: ADD orig_scan TO CONSOLIDATION
            # TODO: MAKE SURE cropped_segs_dict contains 2 DIFFERENT copies of the segs dict

            patient_mode = random.choices(['supine', 'erect'], weights=[0.85, 0.15], k=1)[0]

            if random.random() < 0.825 and (not bad_seg_flag):
                ret_dict = FluidOverload.add_to_CT_pair([cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE)], [prior_cropped_segs_dict, current_cropped_segs_dict], patient_mode=patient_mode, orig_scan=orig_cropped_scan, log_params=True)
            else:
                ret_dict = {'pair_scans': [cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE)], 'pair_segs': [prior_cropped_segs_dict, current_cropped_segs_dict], 'params': {}}

            prior, current = ret_dict['pair_scans']
            cropped_segs_dict = ret_dict['pair_segs']
            params = ret_dict['params']

            # [prior, current], _, params
            torch.cuda.empty_cache()
            gc.collect()

            prior = add_back_cropped(prior, orig_scan, cropping_slices)
            current = add_back_cropped(current, orig_scan, cropping_slices)

            # prior = add_back_cropped(prior, orig_scan, segs_dict['lungs'])
            # current = add_back_cropped(current, orig_scan, segs_dict['lungs'])

            ct_cat = torch.stack([prior, current], dim=0)

            del prior
            del current
            prior = None
            current = None

            torch.cuda.empty_cache()
            gc.collect()

            rotated_ct_cat = random_rotate_ct_and_crop_according_to_seg(ct_cat, segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)[0]

            current_ct = rotated_ct_cat[1]
            registrated_prior_ct = rotated_ct_cat[0]

            current = project_ct(current_ct)
            registrated_prior = project_ct(registrated_prior_ct)

            del current_ct
            del registrated_prior_ct
            del rotated_ct_cat
            current_ct = None
            registrated_prior_ct = None
            rotated_ct_cat = None

            torch.cuda.empty_cache()
            gc.collect()

            rotated_ct_scan = random_rotate_ct_and_crop_according_to_seg(ct_cat[0], segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)[0]

            prior_ct = rotated_ct_scan[0]
            prior = project_ct(prior_ct)

            del prior_ct
            del ct_cat
            del rotated_ct_scan
            prior_ct = None
            ct_cat = None
            rotated_ct_scan = None

            torch.cuda.empty_cache()
            gc.collect()

            diff_map = calculate_diff_map(current, registrated_prior)

            current = normalize_xray(current)
            prior = normalize_xray(prior)

            torch.cuda.empty_cache()
            gc.collect()

            # prior_xray = project_ct(prior)
            # prior_xray = normalize_xray(prior_xray)
            #
            # prior_xray = prior_xray.astype(np.float32).T
            #
            # current_xray = project_ct(current)
            # current_xray = normalize_xray(current_xray)
            #
            # current_xray = current_xray.astype(np.float32).T

            # idx = 4
            # adj = "VesselsCephalizationAddHUAndAvg"
            # plt.imsave(f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_DRR_exp{idx}_{adj}.png', prior_xray.T, cmap='gray')
            # save_arr_as_nifti(prior_xray, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_DRR_exp{idx}_{adj}.nii.gz')
            # save_arr_as_nifti(prior.cpu(), f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Entity_Experiments/{case_name}_CT_exp{idx}_{adj}.nii.gz')

            # plt.imsave(r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\new\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_DRR_t.png', prior_xray.T, cmap='gray')
            # save_arr_as_nifti(prior_xray, r'C:\Users\sharp\PycharmProjects\LongitudinalCXRAnalysis\pneumothorax\new\1.3.6.1.4.1.14519.5.2.1.6279.6001.111172165674661221381920536987_DRR_t.nii.gz')

            save_arr_as_nifti(prior.T, f'{c_out_dir}/prior.nii.gz')
            save_arr_as_nifti(current.T, f'{c_out_dir}/current.nii.gz')
            save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

            plt.imsave(f'{c_out_dir}/prior.png', prior, cmap='gray')
            plt.imsave(f'{c_out_dir}/current.png', current, cmap='gray')

            plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')

            log_params(params, f'{c_out_dir}/params.json')

# def temp_create_pleural_effusion():
#     struct = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
#     differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
#         # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
#         (0.000, (0.235, 1.000, 0.239)),
#         (0.400, (0.000, 1.000, 0.702)),
#         (0.500, (1.000, 0.988, 0.988)),
#         (0.600, (1.000, 0.604, 0.000)),
#         (1.000, (0.682, 0.000, 0.000))))
#
#     # temp_main_create_entity_pairs()
#
#     # exit()
#
#     case_name = '1.3.6.1.4.1.14519.5.2.1.6279.6001.244447966386688625240438849169'
#     scan_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/scans/Philips/{case_name}.nii.gz'
#     seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_manufacturers/segs/Philips/{case_name}_seg.nii.gz'
#     middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
#     upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
#     lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
#     upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
#     lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
#     # heart_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_segs/heart.nii.gz'
#     bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
#     lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'
#
#     # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'middle_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p, 'lower_left_lobe': lower_left_lobe_p}
#     # segs_p_dict = {'lungs': seg_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p,
#     #                'lower_left_lobe': lower_left_lobe_p}
#     segs_p_dict = {'lungs': seg_p}
#     smoothing_radius_dict = {'lungs': 7, 'bronchi': 3, 'lung_vessels': 3, 'middle_right_lobe': 7, 'upper_right_lobe': 7, 'lower_right_lobe': 7, 'upper_left_lobe': 7, 'lower_left_lobe': 7}
#
#     scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
#     orig_scan = scan.clone()
#
#     cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)
#     orig_cropped_scan = cropped_scan.clone().to(DEVICE)
#
#     for organ_name, c_seg in cropped_segs_dict.items():
#         c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
#         # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
#         cropped_segs_dict[organ_name] = c_seg
#
#     # TODO: After done with intra-pulmonary. Clone prior and pass as registrated_prior for extra-pulmonary.
#
#     rot_ranges = (15, 15, 15)
#     max_sum = 30
#
#     prior_cropped_segs_dict = cropped_segs_dict
#     current_cropped_segs_dict = {k: v.clone() for k, v in prior_cropped_segs_dict.items()}
#
#     c_out_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/pleural_effusion'
#     exp = '1'
#
#     registrated_prior = cropped_scan.clone().to(DEVICE)
#     ret_dict = PleuralEffusion.add_to_CT_pair([cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE), registrated_prior], [prior_cropped_segs_dict, current_cropped_segs_dict], orig_scan=orig_cropped_scan, log_params=True)
#
#     prior, current = ret_dict['pair_scans']
#     prior_cropped_segs_dict, current_cropped_segs_dict = ret_dict['pair_segs']
#     registrated_prior = ret_dict['registrated_prior']
#     params = ret_dict['params']
#     effusion_msk_in = ret_dict['msk_in']
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     prior = add_back_cropped(prior, orig_scan, cropping_slices)
#     current = add_back_cropped(current, orig_scan, cropping_slices)
#     registrated_prior = add_back_cropped(registrated_prior, orig_scan, cropping_slices)
#     effusion_msk_in = add_back_cropped(effusion_msk_in, torch.zeros_like(orig_scan), cropping_slices)
#
#     ct_cat = torch.stack([registrated_prior, current], dim=0)
#
#     # del prior
#     # del current
#     # prior = None
#     # current = None
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     seg_for_rotation = segs_dict['lungs'].clone()
#     seg_for_rotation[effusion_msk_in == 1] = 2
#     seg_for_rotation[effusion_msk_in == -1] = 3
#
#     rotated_ct_cat, seg_for_rotation = random_rotate_ct_and_crop(ct_cat, seg_for_rotation, return_ct_seg=True, rot_ranges=rot_ranges, max_angles_sum=max_sum, exponent=5.)
#     # TODO: REMEMBER THAT ROTATION WITH REFLECTION CAN CAUSE MORE SEGMENTATION AREA IN msk_in and seg_for_rotation to appear. NEED TO TAKE CARE OF.
#
#     current_ct = rotated_ct_cat[1]
#     registrated_prior_ct = rotated_ct_cat[0]
#
#     current = project_ct(current_ct)
#     registrated_prior = project_ct(registrated_prior_ct)
#
#     # TODO: Remember that devices need to be added to CTs AFTER projecting, then projecting again
#     del ct_cat
#     del current_ct
#     del registrated_prior_ct
#     del rotated_ct_cat
#     ct_cat = None
#     current_ct = None
#     registrated_prior_ct = None
#     rotated_ct_cat = None
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     seg_for_rotation = seg_for_rotation.squeeze()
#     rotated_seg = seg_for_rotation != 0
#     msk_in = seg_for_rotation
#     msk_in[seg_for_rotation == 1] = 0
#     msk_in[seg_for_rotation == 2] = 1
#     msk_in[seg_for_rotation == 3] = -1
#
#     # save_arr_as_nifti(rotated_seg, f'{c_out_dir}/rotated_seg_ct_{exp}.nii.gz')
#     # save_arr_as_nifti(msk_in, f'{c_out_dir}/msk_in_ct_{exp}.nii.gz')
#
#     # effusion_diff = torch.nan_to_num(torch.sum(msk_in, dim=2) / torch.sum(rotated_seg, dim=2), nan=0, posinf=0, neginf=0).cpu().squeeze()
#     effusion_max_diff_val = 0.45
#     effusion_diff = (torch.sum(msk_in, dim=2) / torch.max(torch.sum(rotated_seg, dim=2))).cpu().squeeze()
#     effusion_diff = resize(effusion_diff[None, ...])
#     effusion_diff = torch.nn.functional.avg_pool2d(effusion_diff, kernel_size=7, stride=1, padding=3).squeeze().numpy()
#     effusion_diff *= effusion_max_diff_val
#     # TODO: POSSIBLE MULT BY SOME MAX VAL
#
#     del rotated_seg
#     del msk_in
#     rotated_seg = None
#     msk_in = None
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     prior_ct = random_rotate_ct_and_crop_according_to_seg(prior, segs_dict['lungs'], return_ct_seg=False, rot_ranges=rot_ranges, max_angles_sum=max_sum, exponent=5.)[0]
#     prior_ct = prior_ct.squeeze()
#
#     prior = project_ct(prior_ct)
#
#     del prior_ct
#     prior_ct = None
#
#     torch.cuda.empty_cache()
#     gc.collect()
#
#     diff_map = calculate_diff_map(current, registrated_prior)
#
#     current = normalize_xray(current)
#     prior = normalize_xray(prior)
#
#     # [prior, current], _, params
#
#     # save_arr_as_nifti(prior, f'{c_out_dir}/prior_ct_{exp}.nii.gz')
#
#     # prior = project_ct(prior)
#     # prior = normalize_xray(prior)
#
#     # current = project_ct(current)
#     # current = normalize_xray(current)
#     diff_map += effusion_diff
#
#     plt.imsave(f'{c_out_dir}/effusion_diff_map_{exp}.png', effusion_diff, cmap='gray')
#
#     save_arr_as_nifti(prior.T, f'{c_out_dir}/prior_{exp}.nii.gz')
#     save_arr_as_nifti(current.T, f'{c_out_dir}/current_{exp}.nii.gz')
#     save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map_{exp}.nii.gz')
#
#     plt.imsave(f'{c_out_dir}/prior_{exp}.png', prior, cmap='gray')
#     plt.imsave(f'{c_out_dir}/current_{exp}.png', current, cmap='gray')
#     plt.imsave(f'{c_out_dir}/diff_map_{exp}.png', diff_map, cmap='gray')
#
#     plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')
#
#     log_params(params, f'{c_out_dir}/params_{exp}.json')


def create_seg_diff_map_from_rotated(c_rotated_seg, pos_num, neg_num, max_diff_val, max_seg_depth, avg_k):
    seg_for_diff = c_rotated_seg.clone()
    seg_for_diff[~torch.isin(seg_for_diff, torch.tensor([pos_num, neg_num], device=seg_for_diff.device))] = 0
    seg_for_diff[seg_for_diff == pos_num] = 1
    seg_for_diff[seg_for_diff == neg_num] = -1

    diff_map_for_seg = torch.nan_to_num(torch.sum(seg_for_diff, dim=2) / max_seg_depth, nan=0.).cpu().squeeze()
    diff_map_for_seg = resize(diff_map_for_seg[None, ...])
    if avg_k is not None:
        diff_map_for_seg = torch.nn.functional.avg_pool2d(diff_map_for_seg, kernel_size=avg_k, stride=1, padding=avg_k // 2).squeeze()
    diff_map_for_seg *= max_diff_val

    return diff_map_for_seg.numpy()


def save_arr_as_nifti_temp(arr, output_path):
    arr = np.flip(arr, axis=0)
    xray_nif = nib.Nifti1Image(arr, affine)
    nib.save(xray_nif, output_path)


def load_scan_temp(file_path: str):
    """Load a .nii.gz scan file with memory-efficient loading."""
    scan_nif = nib.load(file_path)

    global affine
    affine = scan_nif.affine

    # Use dataobj for memory-efficient loading (avoids extra copy)
    scan_data = np.asarray(scan_nif.dataobj, dtype=np.float32)
    scan_data = torch.from_numpy(np.transpose(scan_data, (2, 0, 1)).copy())
    # Explicitly free the nifti object
    scan_nif.uncache()
    del scan_nif
    
    scan_data = torch.flip(scan_data, dims=(0,))

    return scan_data


def deform_to_AP(scan, lungs_seg, def_fac):
    import gryds

    lungs_coords = lungs_seg.nonzero().T
    lungs_min_d, lungs_max_d = torch.min(lungs_coords[2]), torch.max(lungs_coords[2])
    lungs_depth = int(lungs_max_d - lungs_min_d)

    x = torch.arange(scan.shape[0])
    y = torch.arange(scan.shape[1])
    z = torch.arange(scan.shape[2])

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)

    scan_cent = (scan.shape[0] // 2, scan.shape[1] // 2, 0)

    grid_x = (scan_cent[0] - grid_x) / scan.shape[0]
    grid_y = (scan_cent[1] - grid_y) / scan.shape[1]

    grid_z_norm = torch.zeros_like(grid_x)
    grid_z_norm[:, :, :lungs_min_d] = def_fac
    grid_z_norm[:, :, lungs_max_d:] = 0
    grid_z_range = torch.linspace(0., def_fac, steps=lungs_depth)[None, None, ...]
    grid_z_norm[:, :, lungs_min_d: lungs_max_d] = grid_z_range

    # grid_z_norm = (0.5 * torch.max(grid_z) - grid_z) / torch.max(grid_z)
    # grid_z_norm *= def_fac
    # grid_z_norm = grid_z_norm ** 2

    grid_x *= grid_z_norm
    grid_y *= grid_z_norm
    grid_z = torch.zeros_like(grid_x)

    min_v = torch.min(scan)
    scan = scan - min_v

    bspline = gryds.BSplineTransformationCuda([grid_x, grid_y, grid_z], order=1)
    interpolator = gryds.BSplineInterpolatorCuda(scan.squeeze().to(DEVICE))
    deformed_scan = torch.tensor(interpolator.transform(bspline)).to(DEVICE)

    deformed_scan = deformed_scan + min_v

    return deformed_scan


def create_drr():
    case_name = 'train_10000_a_1'
    # case_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE-DS/{case_name}.nii.gz'
    case_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans/{case_name}.nii.gz'
    # case_p = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/Case3_CT.nii.gz'
    # scan = load_scan_temp(case_p)
    scan, segs_dict = load_scan_and_seg(case_p, {'lungs': f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'})
    lungs_seg = segs_dict['lungs']
    # scan -= 8100

    c_out_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE-drr'

    # scan = scan[:int(scan.shape[0] * 0.85)]

    # scan = deform_to_AP(scan, lungs_seg, 0.15)
    # save_arr_as_nifti(scan, f'{c_out_dir}/AP_def.nii.gz')

    drr = project_ct(scan)
    drr = normalize_xray(drr)

    save_arr_as_nifti_temp(drr.T, f'{c_out_dir}/{case_name}_exponent_test.nii.gz')
    plt.imsave(f'{c_out_dir}/{case_name}_exponent_test.png', drr, cmap='gray')


def add_entities_to_pair(scans, segs, orig_scan, orig_lungs, intra_pulmonary_entities, extra_pulmonary_entities, devices_entity, entity_prob_decay):
    patient_mode = random.choices(['supine', 'erect'], weights=[0.8, 0.2], k=1)[0]
    pleural_effusion_patient_mode = 'erect' if patient_mode == 'erect' else ('supine' if random.random() < 0.5 else 'semi-supine')

    c_dict = {'scans': scans, 'segs': segs, 'orig_scan': orig_scan, 'orig_lungs': orig_lungs, 'patient_mode': patient_mode, 'pleural_effusion_patient_mode': pleural_effusion_patient_mode, 'added_entity_names': [], 'log_params': True}
    c_entity_prob_mult = 1.

    if len(intra_pulmonary_entities) > 0:
        for entity, prob in intra_pulmonary_entities:
            if random.random() < prob * c_entity_prob_mult:
                c_dict.update(entity.add_to_CT_pair(**c_dict))
                c_dict['added_entity_names'].append(entity.__name__)
                c_entity_prob_mult *= entity_prob_decay
                torch.cuda.empty_cache()
                gc.collect()

    c_dict['registrated_prior'] = c_dict['scans'][0].clone()
    if len(extra_pulmonary_entities) > 0:
        for entity, prob in extra_pulmonary_entities:
            if random.random() < prob * c_entity_prob_mult:
                c_dict.update(entity.add_to_CT_pair(**c_dict))
                c_dict['added_entity_names'].append(entity.__name__)
                c_entity_prob_mult *= entity_prob_decay
                torch.cuda.empty_cache()
                gc.collect()

    if len(devices_entity) > 0:
        for entity, prob in devices_entity:
            if random.random() < prob:
                c_dict.update(entity.add_to_CT_pair(**c_dict))
                c_dict['added_entity_names'].append(entity.__name__)
                torch.cuda.empty_cache()
                gc.collect()

    return c_dict


def main():
    args = parse_args()

    total_pairs_num = args.number_pairs

    # General entities
    all_intra_pulmonary_entities = {'Consolidation', 'FluidOverload'}
    all_extra_pulmonary_entities = {'PleuralEffusion', 'Pneumothorax', 'Cardiomegaly'}
    devices_entity = {'ExternalDevices'}
    entity_name_to_class_dict = {
        'Consolidation': Consolidation,
        'FluidOverload': FluidOverload,
        'PleuralEffusion': PleuralEffusion,
        'Pneumothorax': Pneumothorax,
        'Cardiomegaly': Cardiomegaly,
        'ExternalDevices': ExternalDevices
    }

    # Current entities
    if args.default_entities:
        cur_entities_and_base_probs = {'Consolidation': 0.15, 'FluidOverload': 0.15, 'PleuralEffusion': 0.15, 'Pneumothorax': 0.15, 'Cardiomegaly': 0.15, 'ExternalDevices': 0.5}
    else:
        cur_entities_and_base_probs = {'Consolidation': args.Consolidation, 'FluidOverload': args.FluidOverload, 'PleuralEffusion': args.PleuralEffusion, 'Pneumothorax': args.Pneumothorax, 'Cardiomegaly': args.Cardiomegaly, 'ExternalDevices': args.ExternalDevices}
    entity_prob_decay_on_addition = args.decay_prob_on_add

    print(f'Entity probabilities: {cur_entities_and_base_probs}')

    cur_intra_pulmonary_entities = []
    cur_extra_pulmonary_entities = []
    cur_devices_entity = []

    for c_entity, prob in cur_entities_and_base_probs.items():
        if c_entity in all_intra_pulmonary_entities:
            cur_intra_pulmonary_entities.append((entity_name_to_class_dict[c_entity], prob))
        elif c_entity in all_extra_pulmonary_entities:
            cur_extra_pulmonary_entities.append((entity_name_to_class_dict[c_entity], prob))
        elif c_entity in devices_entity:
            cur_devices_entity.append((entity_name_to_class_dict[c_entity], prob))
        else:
            raise Exception(f"Exception: Unknown entity name given: {c_entity}.\nImplemented entities are:\n{', '.join(entity_name_to_class_dict.keys())}")

    rotation_params = args.rotation_params
    assert len(rotation_params) == 4, f"Four rotation parameters should be given. Instead, {rotation_params} was given."
    rotation_range = rotation_params[0]
    rot_ranges = (rotation_range, rotation_range, rotation_range)
    max_sum = rotation_params[1]
    min_sum = rotation_params[2]
    rot_exp = rotation_params[3]

    out_base_dir = args.output

    ct_paths = []
    for ct_dir in args.input:
        cur_ct_paths = [ct_dir + f'/{n}' for n in os.listdir(ct_dir)]
        ct_paths += cur_ct_paths
    ct_paths = sorted(ct_paths)
    num_paths = len(ct_paths)
    pairs_per_ct = math.ceil(total_pairs_num / num_paths)

    # import platform
    # computer_name = platform.node()

    # comp_angle_dict = {'cas703': 25, 'cas702': 15, 'cas701': 8, 'cas605': 0}
    # if computer_name in comp_angle_dict:
    #     rot_range = comp_angle_dict[computer_name]
    # else:
    #     print("Computer name not in angles dict. Exiting.")
    #     exit()

    # slicing_dict = {'cas703': (None, num_paths // 3), 'cas702': (2 * num_paths // 3, None), 'cas605': (num_paths // 3, 2 * num_paths // 3)}
    # c_data_slicing = slicing_dict[computer_name]

    slices = args.slices_for_CTs_list
    assert len(slices) == 2, f"Two slicing values should be given. Instead, {slices} was given."
    first = int(slices[0] * num_paths)
    last = int(slices[1] * num_paths + 1)

    print(f'Input CT dirs: {args.input}\nOutput dir for synthetic pairs: {args.output}')
    print(f'Total synthetic pairs to be created: {total_pairs_num}\nTotal number of available CTs: {num_paths}\nNumber of synthetic pairs to be created per CT: {pairs_per_ct}')
    print(f'Entities to use and their probabilities: {cur_entities_and_base_probs}\nProbability decay factor on addition: {entity_prob_decay_on_addition}')
    print(f'Rotation ranges = {rot_ranges}\nMax angles sum = {max_sum}\nMin angles sum = {min_sum}\nAngle distribution exponent = {rot_exp}')
    print(f'CT list slicing: {slices}')

    ps = ct_paths[first: last]
    pairs_created = 0
    len_ps = len(ps)
    
    # Memory threshold for aggressive cleanup (in GB) - from command line argument
    MEMORY_THRESHOLD_GB = args.memory_threshold
    print(f'Memory threshold for cleanup: {MEMORY_THRESHOLD_GB} GB')
    
    log_memory("Before starting main loop")
    
    for k, ct_p in enumerate(ps):
        case_name = ct_p.split('/')[-1][:-7]
        print(f'\n{"="*60}')
        print(f'Working on case {k}/{len_ps}: {case_name}')
        print(f'First = {first}, Last = {last}')
        log_memory(f"Start of case {k}")
        
        # Initialize variables to None for safe cleanup in case of error
        scan = orig_scan = segs_dict = None
        cropped_scan = cropped_segs_dict = cropping_slices = None
        orig_cropped_scan = orig_cropped_lungs = None
        prior_cropped_segs_dict = current_cropped_segs_dict = None

        if case_name in {'train_11701_a_2', 'train_1128_a_2', 'train_11716_a_1', 'train_11431_a_2', 'train_10466_a_2', '1.3.6.1.4.1.14519.5.2.1.6279.6001.146429221666426688999739595820', '1.3.6.1.4.1.14519.5.2.1.6279.6001.221945191226273284587353530424'}:
            continue

        # if case_name in {'1.3.6.1.4.1.14519.5.2.1.6279.6001.336894364358709782463716339027', 'volume-105', 'ABD_LYMPH_005'}:
        #     print(f'Bad case. Skipping.')
        #     continue

        out_path = f'{out_base_dir}/{case_name}'

        if os.path.exists(f'{out_path}/pair{pairs_per_ct - 1}/params.json'):
            print('Case completed. Skipping')
            continue
        
        try:
            # scan_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans/{case_name}.nii.gz'
            scan_p = ct_p
            seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
            middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
            upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
            lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
            upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
            lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
            heart_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_heart_segs/{case_name}_heart_seg.nii.gz'
            # heart_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/heart_test/{case_name}_heart_seg.nii.gz'
            bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
            lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

            segs_p_dict = {'lungs': seg_p, 'heart': heart_p, 'bronchi': bronchi_p, 'lung_vessels': lung_vessels_p, 'middle_right_lobe': middle_lobe_p, 'upper_right_lobe': upper_right_lobe_p, 'lower_right_lobe': lower_right_lobe_p, 'upper_left_lobe': upper_left_lobe_p, 'lower_left_lobe': lower_left_lobe_p}
            smoothing_radius_dict = {'lungs': 7, 'heart': 7, 'bronchi': 3, 'lung_vessels': 3, 'middle_right_lobe': 7, 'upper_right_lobe': 7, 'lower_right_lobe': 7, 'upper_left_lobe': 7, 'lower_left_lobe': 7}

            scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
            orig_scan = scan.clone()
            
            # Check memory after loading (CT loading is memory-intensive)
            check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After loading CT {case_name}")

            segs_dict['lungs'] = torch.clamp_max(segs_dict['middle_right_lobe'] + segs_dict['upper_right_lobe'] + segs_dict['lower_right_lobe'] + segs_dict['upper_left_lobe'] + segs_dict['lower_left_lobe'], 1.)

            if segs_dict['lungs'].sum() < 20:
                print(f'Found empty seg for case {case_name}. Continuing')
                continue

            lung_seg_coords = segs_dict['lungs'].nonzero().T
            lungs_h = torch.max(lung_seg_coords[0]) - torch.min(lung_seg_coords[0])
            h_low_ext = (lungs_h // 5).item()

            # out_path = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/fluid_overload_test/angles_{rot_ranges[0]}_{rot_ranges[1]}_{rot_ranges[2]}/{case_name}'

            # cropped_scan, cropped_seg, cropped_heart, slices = crop_according_to_seg(scan, seg, heart_seg=heart, tight=True, return_slices=True)
            cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False, ext=15, h_low_ext=h_low_ext)

            bad_seg_flag = False

            for organ_name, c_seg in cropped_segs_dict.items():
                c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=smoothing_radius_dict[organ_name]).cpu()
                # c_seg = torch.tensor(c_seg, dtype=torch.float32).to(DEVICE)
                cropped_segs_dict[organ_name] = c_seg
                if torch.sum(c_seg) < 20:
                    print(f'Found empty {organ_name} for case {case_name}. Continuing')
                    bad_seg_flag = True
                    break

            if bad_seg_flag:
                continue

            cropped_segs_dict['lung_right'] = torch.maximum(cropped_segs_dict['middle_right_lobe'], torch.maximum(cropped_segs_dict['upper_right_lobe'], cropped_segs_dict['lower_right_lobe']))
            cropped_segs_dict['lung_left'] = torch.maximum(cropped_segs_dict['upper_left_lobe'], cropped_segs_dict['lower_left_lobe'])

            orig_cropped_scan = cropped_scan.clone().cpu()
            orig_cropped_lungs = cropped_segs_dict['lungs'].clone()

            prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
            current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

            for i in range(pairs_per_ct):
                print(f'Working on pair {i}')

                c_out_dir = f'{out_path}/pair{i}'
                os.makedirs(c_out_dir, exist_ok=True)

                if os.path.exists(f'{c_out_dir}/params.json'):
                    print("Pair exists. Skipping.")
                    pairs_created += 1
                    continue

                prior_cropped_segs_dict['lung_right'] = cropped_segs_dict['lung_right'].clone()
                prior_cropped_segs_dict['lung_left'] = cropped_segs_dict['lung_left'].clone()
                prior_cropped_segs_dict['lungs'] = cropped_segs_dict['lungs'].clone()

                current_cropped_segs_dict['lung_right'] = cropped_segs_dict['lung_right'].clone()
                current_cropped_segs_dict['lung_left'] = cropped_segs_dict['lung_left'].clone()
                current_cropped_segs_dict['lungs'] = cropped_segs_dict['lungs'].clone()

                scans = [cropped_scan.clone().to(DEVICE), cropped_scan.clone().to(DEVICE)]
                segs = [prior_cropped_segs_dict, current_cropped_segs_dict]

                ret_dict = add_entities_to_pair(scans, segs, orig_cropped_scan, orig_cropped_lungs, cur_intra_pulmonary_entities, cur_extra_pulmonary_entities, cur_devices_entity, entity_prob_decay_on_addition)

                torch.cuda.empty_cache()
                gc.collect()

                prior, current = ret_dict['scans']
                prior_cropped_segs_dict, current_cropped_segs_dict = ret_dict['segs']
                registrated_prior = ret_dict['registrated_prior']
                added_entity_names = ret_dict['added_entity_names']
                params = {'added_entities': added_entity_names}
                # params = ret_dict['params']

                prior = add_back_cropped(prior, orig_scan, cropping_slices)
                current = add_back_cropped(current, orig_scan, cropping_slices)
                registrated_prior = add_back_cropped(registrated_prior, orig_scan, cropping_slices)

                seg_for_rotation = segs_dict['lungs'].clone()

                if 'PleuralEffusion' in added_entity_names:
                    pleural_effusion_seg = ret_dict['pleural_effusion_seg']
                    pleural_effusion_seg = add_back_cropped(pleural_effusion_seg, torch.zeros_like(orig_scan), cropping_slices)
                    seg_for_rotation[pleural_effusion_seg == 1] = 2
                    seg_for_rotation[pleural_effusion_seg == -1] = 3
                    del pleural_effusion_seg

                if 'Pneumothorax' in added_entity_names:
                    pneumothorax_seg = ret_dict['pneumothorax_seg']
                    pneumothorax_sulcus_seg = ret_dict['pneumothorax_sulcus_seg']
                    pneumothorax_seg = add_back_cropped(pneumothorax_seg, torch.zeros_like(orig_scan), cropping_slices)
                    pneumothorax_sulcus_seg = add_back_cropped(pneumothorax_sulcus_seg, torch.zeros_like(orig_scan), cropping_slices)
                    seg_for_rotation[pneumothorax_seg == 1] = 4
                    seg_for_rotation[pneumothorax_seg == -1] = 5
                    seg_for_rotation[pneumothorax_sulcus_seg == 1] = 6
                    seg_for_rotation[pneumothorax_sulcus_seg == -1] = 7
                    del pneumothorax_seg
                    del pneumothorax_sulcus_seg

                if 'Cardiomegaly' in added_entity_names:
                    cardiomegaly_seg = ret_dict['cardiomegaly_seg']
                    cardiomegaly_progress = ret_dict['cardiomegaly_progress']
                    cardiomegaly_seg = add_back_cropped(cardiomegaly_seg, torch.zeros_like(orig_scan), cropping_slices)
                    seg_for_rotation[cardiomegaly_seg == 1] = 8
                    seg_for_rotation[cardiomegaly_seg == -1] = 9
                    del cardiomegaly_seg

                ct_cat = torch.stack([registrated_prior, current], dim=0)

                del current
                del registrated_prior
                torch.cuda.empty_cache()
                gc.collect()

                rotated_ct_cat, seg_for_rotation = random_rotate_ct_and_crop_according_to_seg(ct_cat, seg_for_rotation, return_ct_seg=True, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)

                current_ct = rotated_ct_cat[1]
                registrated_prior_ct = rotated_ct_cat[0]

                current = project_ct(current_ct)
                registrated_prior = project_ct(registrated_prior_ct)

                del ct_cat
                del current_ct
                del registrated_prior_ct
                del rotated_ct_cat

                torch.cuda.empty_cache()
                gc.collect()

                seg_for_rotation = torch.round(seg_for_rotation).squeeze()
                rotated_seg = seg_for_rotation != 0
                max_rotated_seg_sum = torch.max(torch.sum(rotated_seg, dim=2))

                projected_seg = project_ct(rotated_seg, is_seg=True)

                extra_pulmonary_diff_maps = []

                if 'PleuralEffusion' in added_entity_names:
                    pleural_effusion_diff = create_seg_diff_map_from_rotated(seg_for_rotation, 2, 3, 0.35, max_rotated_seg_sum, 9)
                    extra_pulmonary_diff_maps.append(pleural_effusion_diff)

                if 'Pneumothorax' in added_entity_names:
                    pneumothorax_diff = create_seg_diff_map_from_rotated(seg_for_rotation, 4, 5, 0.25, max_rotated_seg_sum, 9)
                    pneumothorax_sulcus_diff = create_seg_diff_map_from_rotated(seg_for_rotation, 6, 7, 0.25, max_rotated_seg_sum, 9)
                    extra_pulmonary_diff_maps.append(pneumothorax_diff)
                    extra_pulmonary_diff_maps.append(pneumothorax_sulcus_diff)

                if 'Cardiomegaly' in added_entity_names:
                    if cardiomegaly_progress in {1, -1}:
                        cardiomegaly_diff = create_seg_diff_map_from_rotated(seg_for_rotation, 8, 9, 0.35, max_rotated_seg_sum, 9)

                        if cardiomegaly_progress == 1:
                            cardiomegaly_diff = np.clip(cardiomegaly_diff, a_min=0, a_max=None)
                        else:
                            cardiomegaly_diff = np.clip(cardiomegaly_diff, a_min=None, a_max=0)
                    else:
                        cardiomegaly_diff = np.zeros_like(current)
                    extra_pulmonary_diff_maps.append(cardiomegaly_diff)

                del seg_for_rotation
                del rotated_seg
                torch.cuda.empty_cache()
                gc.collect()

                prior_ct, prior_rotated_seg = random_rotate_ct_and_crop_according_to_seg(prior, segs_dict['lungs'], return_ct_seg=True, rot_ranges=rot_ranges, max_angles_sum=max_sum, min_angles_sum=min_sum, exponent=rot_exp)
                prior_ct = prior_ct.squeeze()

                prior = project_ct(prior_ct)

                del prior_ct
                del prior_rotated_seg
                prior_ct = None
                prior_rotated_seg = None
                torch.cuda.empty_cache()
                gc.collect()

                diff_map = calculate_diff_map(current, registrated_prior, projected_seg)

                for extra_pulmonary_diff_map in extra_pulmonary_diff_maps:
                    diff_map += extra_pulmonary_diff_map

                # TODO: REMOVE
                # prior = normalize_xray(prior)
                # current = normalize_xray(current)

                save_arr_as_nifti(prior.T, f'{c_out_dir}/prior.nii.gz')
                save_arr_as_nifti(current.T, f'{c_out_dir}/current.nii.gz')
                save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

                plt.imsave(f'{c_out_dir}/prior.png', prior, cmap='gray')
                plt.imsave(f'{c_out_dir}/current.png', current, cmap='gray')

                plot_diff_on_current(diff_map, current, f'{c_out_dir}/current_with_differences.png')

                log_params(params, f'{c_out_dir}/params.json')

                pairs_created += 1

                # Cleanup after each pair
                del prior, current, diff_map, projected_seg
                del ret_dict, extra_pulmonary_diff_maps
                cleanup_memory()
                
                # Check memory and perform aggressive cleanup if needed
                check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After pair {i}")

                if pairs_created == total_pairs_num:
                    print("Done generating synthetic pairs. Exiting.")
                    exit()
                
        except MemoryError as e:
            print(f"[ERROR] MemoryError on case {case_name}: {e}")
            print("[ERROR] Attempting to recover by skipping this case...")
            # Aggressive cleanup on memory error
            for _ in range(5):
                gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[ERROR] Exception on case {case_name}: {e}")
            import traceback
            traceback.print_exc()
            print("[ERROR] Skipping to next case...")
        
        finally:
            # ==================================================================
            # CRITICAL: Clean up ALL case-level variables after processing
            # This runs whether the case succeeded or failed
            # ==================================================================
            print(f"[Cleanup] Cleaning up case {case_name}...")
            
            # Explicitly delete and set to None (locals() doesn't work in finally)
            scan = None
            orig_scan = None
            segs_dict = None
            cropped_scan = None
            cropped_segs_dict = None
            cropping_slices = None
            orig_cropped_scan = None
            orig_cropped_lungs = None
            prior_cropped_segs_dict = None
            current_cropped_segs_dict = None
            
            # Force garbage collection multiple times
            for _ in range(5):
                gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Additional gc pass
            gc.collect()
            
            # Check memory and log
            check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After case {case_name}")
            log_memory(f"End of case {k}")


if __name__ == '__main__':
    struct = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))
    # create_drr()
    main()