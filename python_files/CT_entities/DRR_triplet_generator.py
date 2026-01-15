"""Synthetic DRR Triplet Generator.

Generates triplets of DRRs per CT:
- prior.nii.gz: clean DRR at "view0" (typically no rotation)
- current.nii.gz: clean DRR at a different random angle ("view1")
- current_entity.nii.gz: same "view1" angle, but with pathology/entity injected with probability
- diff_map.nii.gz: change map between current_entity and current (2D)

Output structure mirrors CT_entities/DRR_generator.py:
    output_dir/
    └── case_name/
        └── pairN/
            ├── prior.nii.gz
            ├── current.nii.gz
            ├── current_entity.nii.gz
            ├── diff_map.nii.gz
            └── params.json

This script is designed to run on the university machine (no local installs).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from typing import Dict, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import label
from skimage.morphology import ball

from constants import DEVICE

from CT_entities.DRR_utils import crop_according_to_seg
from CT_entities.CT_Rotations import get_random_rotation_angles, rotate_ct_and_crop_according_to_seg
from CT_entities.drr_triplet_pipeline import (
    maybe_inject_entity_into_current,
    project_ct_to_drr,
)


# Global affine saved alongside NIfTI outputs (mirrors DRR_generator pattern)
_affine = np.eye(4)


def smooth_segmentation(mask: torch.Tensor, radius: int = 7) -> torch.Tensor:
    """Morphological closing (dilate then erode) for segmentations.

    Matches the helper in DRR_generator but avoids importing that large module.
    """

    def binary_dilation(to_dilate: torch.Tensor, rad: int) -> torch.Tensor:
        d_struct = torch.tensor(ball(rad), dtype=torch.float32, device=to_dilate.device)[None, None, ...]
        dilated = torch.nn.functional.conv3d(to_dilate.squeeze()[None, None, ...], weight=d_struct, padding='same').squeeze()
        dilated[dilated > 0] = 1
        return dilated

    def binary_erosion(to_erode: torch.Tensor, rad: int) -> torch.Tensor:
        return 1.0 - binary_dilation(1.0 - to_erode, rad)

    closed_mask = binary_dilation(mask, radius)
    closed_mask = binary_erosion(closed_mask, radius)
    return closed_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic DRR triplets (view0 clean, view1 clean, view1 entity) with DRR_generator-like controls.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Match DRR_generator main controls
    p.add_argument('-n', '--number_pairs', type=int, required=True, help='Total number of triplets to create (Required).')
    p.add_argument(
        '-i', '--input', nargs='+', type=str,
        default=[
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans',
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans',
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans',
        ],
        help='Space-separated CT directories (same as DRR_generator defaults).'
    )
    p.add_argument(
        '-o', '--output', type=str,
        default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/train_triplets',
        help='Output directory.'
    )

    # Entity probabilities (same names)
    p.add_argument('-CO', '-co', '--Consolidation', type=float, default=0.0)
    p.add_argument('-PL', '-pl', '--PleuralEffusion', type=float, default=0.0)
    p.add_argument('-PN', '-pn', '--Pneumothorax', type=float, default=0.0)
    p.add_argument('-FL', '-fl', '--FluidOverload', type=float, default=0.0)
    p.add_argument('-CA', '-ca', '--Cardiomegaly', type=float, default=0.0)
    p.add_argument('-EX', '-ex', '--ExternalDevices', type=float, default=0.0)

    p.add_argument('--default_entities', action='store_true', help='Use DRR_generator baseline entity probs.')
    p.add_argument('-d', '--decay_prob_on_add', type=float, default=1.0, help='Exponential decay on entity addition (same as DRR_generator).')

    # Rotation params for view1 (same format)
    p.add_argument(
        '-r', '--rotation_params', nargs='+', type=float,
        default=[17.5, 37.5, 0.0, 1.75],
        help=(
            'Rotation params for view1 (4 floats):\n'
            '1) max abs rotation per axis (deg)\n'
            '2) max sum of abs angles\n'
            '3) min sum of abs angles\n'
            '4) exponent bias (higher => closer to 0)\n'
        ),
    )

    # Optional rotation params for view0 (default: no rotation)
    p.add_argument(
        '--rotation_params_view0', nargs='+', type=float,
        default=[0.0, 0.0, 0.0, 1.0],
        help='Rotation params for view0 (same 4-float format). Default is no rotation.'
    )

    # CT slicing (same semantics as DRR_generator)
    p.add_argument(
        '-s', '--slices_for_CTs_list', nargs='+', type=float,
        default=[0.0, 1.0],
        help='Two floats in [0,1] selecting CT list fraction: start end'
    )

    # Triplet-specific knob: probability to inject any entity into the 3rd image
    p.add_argument(
        '--p_entity', type=float, default=0.35,
        help='Probability that current_entity differs from current (gates entity injection).'
    )

    # Seg paths (default to same university structure DRR_generator assumes)
    p.add_argument('--segs_root', type=str, default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs')
    p.add_argument('--lobes_root', type=str, default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs')
    p.add_argument('--heart_root', type=str, default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_heart_segs')
    p.add_argument('--bronchi_root', type=str, default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs')
    p.add_argument('--vessels_root', type=str, default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs')

    # Diff-map thresholds (kept simple but controllable)
    p.add_argument('--diff_abs_th', type=float, default=0.015, help='Absolute difference threshold for diff_map.')
    p.add_argument('--diff_min_cc', type=int, default=50, help='Min connected-component size to keep in diff_map.')

    # Optional outputs
    p.add_argument('--save_png', action='store_true', help='Also save PNGs (requires matplotlib installed).')

    # Memory threshold (same default as DRR_generator)
    p.add_argument('-m', '--memory_threshold', type=float, default=15.0)

    return p.parse_args()


def _load_nifti_as_torch(path: str) -> Tuple[torch.Tensor, np.ndarray]:
    nif = nib.load(path)
    data = np.asarray(nif.dataobj, dtype=np.float32)
    aff = nif.affine
    nif.uncache()
    del nif

    t = torch.from_numpy(np.transpose(data, (2, 0, 1)).copy())
    t = torch.flip(t, dims=[0])
    return t, aff


def load_scan_and_seg(case_path: str, segs_p_dict: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    global _affine
    scan, aff = _load_nifti_as_torch(case_path)
    _affine = aff

    segs: Dict[str, torch.Tensor] = {}
    for organ, seg_path in segs_p_dict.items():
        seg_t, _ = _load_nifti_as_torch(seg_path)
        segs[organ] = seg_t
    return scan, segs


def save_arr_as_nifti(arr: Union[np.ndarray, torch.Tensor], out_path: str) -> None:
    if isinstance(arr, torch.Tensor):
        arr = np.array(arr.detach().cpu().float())
    # DRR outputs are 2D; store as (H,W,1) with affine
    if arr.ndim == 2:
        arr = arr[..., None]
    nif = nib.Nifti1Image(arr, _affine)
    nib.save(nif, out_path)


def _remove_small_ccs_2d(mask: np.ndarray, min_count: int) -> np.ndarray:
    lbl, _ = label(mask.astype(np.uint8))
    if lbl.max() == 0:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for lab in range(1, lbl.max() + 1):
        cc = (lbl == lab)
        if int(cc.sum()) >= int(min_count):
            keep |= cc
    return keep


def calculate_diff_map_2d(current: torch.Tensor, baseline: torch.Tensor, *, abs_th: float, min_cc: int) -> torch.Tensor:
    """2D diff-map between two DRRs (both tensors shaped (1,H,W) or (H,W))."""
    if current.ndim == 3:
        current2 = current.squeeze(0)
    else:
        current2 = current
    if baseline.ndim == 3:
        base2 = baseline.squeeze(0)
    else:
        base2 = baseline

    diff = (current2 - base2).to(torch.float32)
    keep = (diff.abs() >= float(abs_th)).detach().cpu().numpy()
    keep = _remove_small_ccs_2d(keep, min_count=min_cc)

    diff_np = diff.detach().cpu().numpy()
    diff_np[~keep] = 0.0
    return torch.from_numpy(diff_np).to(torch.float32)


def _entity_probs_from_args(args: argparse.Namespace) -> Dict[str, float]:
    if args.default_entities:
        return {
            'Consolidation': 0.15,
            'FluidOverload': 0.15,
            'PleuralEffusion': 0.15,
            'Pneumothorax': 0.15,
            'Cardiomegaly': 0.15,
            'ExternalDevices': 0.5,
        }
    return {
        'Consolidation': float(args.Consolidation),
        'FluidOverload': float(args.FluidOverload),
        'PleuralEffusion': float(args.PleuralEffusion),
        'Pneumothorax': float(args.Pneumothorax),
        'Cardiomegaly': float(args.Cardiomegaly),
        'ExternalDevices': float(args.ExternalDevices),
    }


def _angles_from_rotation_params(rotation_params: list[float]) -> Tuple[Tuple[float, float, float], float, float, float]:
    assert len(rotation_params) == 4
    rot_range = float(rotation_params[0])
    rot_ranges = (rot_range, rot_range, rot_range)
    max_sum = float(rotation_params[1])
    min_sum = float(rotation_params[2])
    exp = float(rotation_params[3])
    return rot_ranges, max_sum, min_sum, exp


def main() -> None:
    args = parse_args()

    entity_probs = _entity_probs_from_args(args)
    rot_ranges1, max_sum1, min_sum1, exp1 = _angles_from_rotation_params(args.rotation_params)
    rot_ranges0, max_sum0, min_sum0, exp0 = _angles_from_rotation_params(args.rotation_params_view0)

    # Collect CT paths
    ct_paths: list[str] = []
    for ct_dir in args.input:
        ct_paths.extend([os.path.join(ct_dir, n) for n in os.listdir(ct_dir)])
    ct_paths = sorted(ct_paths)

    num_paths = len(ct_paths)
    if num_paths == 0:
        raise RuntimeError('No CT files found in input directories.')

    slices = args.slices_for_CTs_list
    assert len(slices) == 2
    first = int(float(slices[0]) * num_paths)
    last = int(float(slices[1]) * num_paths + 1)
    ps = ct_paths[first:last]

    pairs_per_ct = math.ceil(int(args.number_pairs) / len(ps))

    os.makedirs(args.output, exist_ok=True)

    print(f'Output: {args.output}')
    print(f'Total requested triplets: {args.number_pairs}')
    print(f'CT list slice: {slices} -> {len(ps)} CTs')
    print(f'Triplets per CT: {pairs_per_ct}')
    print(f'Entity probs: {entity_probs}, p_entity gate={args.p_entity}, decay={args.decay_prob_on_add}')
    print(f'View0 rotation params: ranges={rot_ranges0}, max_sum={max_sum0}, min_sum={min_sum0}, exp={exp0}')
    print(f'View1 rotation params: ranges={rot_ranges1}, max_sum={max_sum1}, min_sum={min_sum1}, exp={exp1}')

    made = 0

    for k, ct_p in enumerate(ps):
        case_name = os.path.basename(ct_p)
        if case_name.endswith('.nii.gz'):
            case_name = case_name[:-7]

        out_case_dir = os.path.join(args.output, case_name)
        os.makedirs(out_case_dir, exist_ok=True)

        # Skip if last pair exists
        if os.path.exists(os.path.join(out_case_dir, f'pair{pairs_per_ct - 1}', 'params.json')):
            continue

        try:
            # Build seg paths like DRR_generator
            seg_p = os.path.join(args.segs_root, f'{case_name}_seg.nii.gz')
            heart_p = os.path.join(args.heart_root, f'{case_name}_heart_seg.nii.gz')
            bronchi_p = os.path.join(args.bronchi_root, f'{case_name}_bronchia_seg.nii.gz')
            lung_vessels_p = os.path.join(args.vessels_root, f'{case_name}_vessels_seg.nii.gz')

            middle_lobe_p = os.path.join(args.lobes_root, f'{case_name}_lung_middle_lobe_right.nii.gz')
            upper_right_lobe_p = os.path.join(args.lobes_root, f'{case_name}_lung_upper_lobe_right.nii.gz')
            lower_right_lobe_p = os.path.join(args.lobes_root, f'{case_name}_lung_lower_lobe_right.nii.gz')
            upper_left_lobe_p = os.path.join(args.lobes_root, f'{case_name}_lung_upper_lobe_left.nii.gz')
            lower_left_lobe_p = os.path.join(args.lobes_root, f'{case_name}_lung_lower_lobe_left.nii.gz')

            segs_p_dict = {
                'lungs': seg_p,
                'heart': heart_p,
                'bronchi': bronchi_p,
                'lung_vessels': lung_vessels_p,
                'middle_right_lobe': middle_lobe_p,
                'upper_right_lobe': upper_right_lobe_p,
                'lower_right_lobe': lower_right_lobe_p,
                'upper_left_lobe': upper_left_lobe_p,
                'lower_left_lobe': lower_left_lobe_p,
            }

            # Skip if lungs seg missing
            if not os.path.exists(seg_p):
                print(f'[Skip] Missing lungs seg for {case_name}')
                continue

            scan, segs_dict = load_scan_and_seg(ct_p, segs_p_dict)

            # If lobe segs exist, rebuild lungs like DRR_generator does
            if all(os.path.exists(p) for p in [middle_lobe_p, upper_right_lobe_p, lower_right_lobe_p, upper_left_lobe_p, lower_left_lobe_p]):
                segs_dict['lungs'] = torch.clamp_max(
                    segs_dict['middle_right_lobe'] + segs_dict['upper_right_lobe'] + segs_dict['lower_right_lobe'] + segs_dict['upper_left_lobe'] + segs_dict['lower_left_lobe'],
                    1.0,
                )

            if segs_dict['lungs'].sum() < 20:
                print(f'[Skip] Empty lungs seg for {case_name}')
                continue

            # Crop according to lungs
            cropped_scan, cropped_segs_dict, _ = crop_according_to_seg(scan, segs_dict['lungs'], all_segs_dict=segs_dict, tight_y=False)

            # Smooth segs (mirrors DRR_generator defaults loosely)
            smoothing_radius = {
                'lungs': 7,
                'heart': 7,
                'bronchi': 3,
                'lung_vessels': 3,
                'middle_right_lobe': 7,
                'upper_right_lobe': 7,
                'lower_right_lobe': 7,
                'upper_left_lobe': 7,
                'lower_left_lobe': 7,
            }

            bad = False
            for organ, seg in list(cropped_segs_dict.items()):
                if organ in smoothing_radius:
                    seg_sm = smooth_segmentation(seg.to(DEVICE), radius=smoothing_radius[organ]).cpu()
                    cropped_segs_dict[organ] = seg_sm
                    if seg_sm.sum() < 20:
                        bad = True
                        break
            if bad:
                print(f'[Skip] Bad segs for {case_name}')
                continue

            # Generate pairs
            for i in range(pairs_per_ct):
                out_pair_dir = os.path.join(out_case_dir, f'pair{i}')
                os.makedirs(out_pair_dir, exist_ok=True)
                if os.path.exists(os.path.join(out_pair_dir, 'params.json')):
                    made += 1
                    if made >= int(args.number_pairs):
                        return
                    continue

                # Sample angles for view0 and view1
                a0 = get_random_rotation_angles(rot_ranges0, max_sum0, min_sum0, exp0)
                a1 = get_random_rotation_angles(rot_ranges1, max_sum1, min_sum1, exp1)

                # Build view0
                ct0 = rotate_ct_and_crop_according_to_seg(
                    cropped_scan, cropped_segs_dict['lungs'],
                    rotate_angle1=a0[0], rotate_angle2=a0[1], rotate_angle3=a0[2],
                    return_ct_seg=False,
                )[0]
                drr0 = project_ct_to_drr(ct0)

                # Build view1 clean
                ct1 = rotate_ct_and_crop_according_to_seg(
                    cropped_scan, cropped_segs_dict['lungs'],
                    rotate_angle1=a1[0], rotate_angle2=a1[1], rotate_angle3=a1[2],
                    return_ct_seg=False,
                )[0]
                drr1 = project_ct_to_drr(ct1)

                # Build view1 entity (maybe equals view1 clean)
                scan_with_entity, ent_meta = maybe_inject_entity_into_current(
                    cropped_scan.clone().to(DEVICE),
                    {k: v.clone().to(DEVICE) for k, v in cropped_segs_dict.items()},
                    p_inject=float(args.p_entity),
                    entity_probs=entity_probs,
                    entity_prob_decay=float(args.decay_prob_on_add),
                )

                ct2 = rotate_ct_and_crop_according_to_seg(
                    scan_with_entity.to(cropped_scan.device),
                    cropped_segs_dict['lungs'],
                    rotate_angle1=a1[0], rotate_angle2=a1[1], rotate_angle3=a1[2],
                    return_ct_seg=False,
                )[0]
                drr2 = project_ct_to_drr(ct2)

                diff_map = calculate_diff_map_2d(drr2, drr1, abs_th=float(args.diff_abs_th), min_cc=int(args.diff_min_cc))

                # Save NIfTIs (2D)
                save_arr_as_nifti(drr0.squeeze(0), os.path.join(out_pair_dir, 'prior.nii.gz'))
                save_arr_as_nifti(drr1.squeeze(0), os.path.join(out_pair_dir, 'current.nii.gz'))
                save_arr_as_nifti(drr2.squeeze(0), os.path.join(out_pair_dir, 'current_entity.nii.gz'))
                save_arr_as_nifti(diff_map, os.path.join(out_pair_dir, 'diff_map.nii.gz'))

                # Save params
                params = {
                    'angles_view0_deg': (float(a0[0]), float(a0[1]), float(a0[2])),
                    'angles_view1_deg': (float(a1[0]), float(a1[1]), float(a1[2])),
                    'entity_applied': bool(ent_meta.get('entity_applied', False)),
                    'added_entities': ent_meta.get('entity_names', []),
                    'entity_probs': entity_probs,
                    'p_entity_gate': float(args.p_entity),
                    'entity_prob_decay': float(args.decay_prob_on_add),
                    'rotation_params_view0': args.rotation_params_view0,
                    'rotation_params_view1': args.rotation_params,
                    'diff_abs_th': float(args.diff_abs_th),
                    'diff_min_cc': int(args.diff_min_cc),
                }
                with open(os.path.join(out_pair_dir, 'params.json'), 'w') as f:
                    json.dump(params, f, indent=4)

                # Optional PNGs
                if args.save_png:
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(out_pair_dir, 'prior.png'), drr0.squeeze(0).cpu().numpy(), cmap='gray')
                    plt.imsave(os.path.join(out_pair_dir, 'current.png'), drr1.squeeze(0).cpu().numpy(), cmap='gray')
                    plt.imsave(os.path.join(out_pair_dir, 'current_entity.png'), drr2.squeeze(0).cpu().numpy(), cmap='gray')

                made += 1

                # Cleanup per-sample
                del ct0, ct1, ct2, drr0, drr1, drr2, diff_map
                torch.cuda.empty_cache()
                gc.collect()

                if made >= int(args.number_pairs):
                    return

        except Exception as e:
            print(f'[ERROR] Case {case_name} failed: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
