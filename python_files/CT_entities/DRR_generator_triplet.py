"""CT → 3-DRR generator (prior / angle+devices / angle+devices+entity).

This script is derived from [CT_entities/DRR_generator_clean.py](CT_entities/DRR_generator_clean.py)
with a different output contract per pair:

Per pair we generate *three* DRRs (names match the original prior/current semantics):
1) prior: baseline CT projected at a chosen (fixed) prior angle.
2) intermediate: baseline CT at angle X, with "non-detection" modifications (external devices).
3) current: intermediate plus an optional pathology entity (progression/regression), then projected.

Additionally, we save a `diff_map` like the original generator:
- `diff_map = current - prior` (2D), computed via `base.calculate_diff_map`.

Key logic blocks (search for these headers in code):
- ENTITY INSERTION (3D): where pathology entities are added.
- DEVICES (3D): where external devices are added (intended as nuisance / non-target changes).
- ANGLES / ROTATION: where angles are applied deterministically.
- DRR PROJECTION: where we actually take the DRR.

Notes on angles:
- DRR1 uses `--prior_angles` (default: 0 0 0)
- DRR2 and DRR3 use `--angle_x` (default: 10 0 0)

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from constants import DEVICE

from CT_Rotations import rotate_ct_and_crop_according_to_seg

from Cardiomegaly import Cardiomegaly
from Consolidation import Consolidation
from Pleural_Effusion import PleuralEffusion
from Pneumothorax import Pneumothorax
from Fluid_Overload import FluidOverload
from External_Devices import ExternalDevices

# Reuse well-tested utilities from the cleaned generator.
import CT_entities.DRR_generator_clean as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate triplets of DRRs: (prior), (angle X + devices), (angle X + devices + entity)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('-n', '--number_pairs', type=int, required=True)
    parser.add_argument(
        '-i',
        '--input',
        nargs='+',
        type=str,
        default=[
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans',
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans',
            '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans',
        ],
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/train_triplets',
    )

    # Pathology entities (optional for DRR3)
    parser.add_argument('-CO', '--Consolidation', type=float, default=0.0)
    parser.add_argument('-PL', '--PleuralEffusion', type=float, default=0.0)
    parser.add_argument('-PN', '--Pneumothorax', type=float, default=0.0)
    parser.add_argument('-FL', '--FluidOverload', type=float, default=0.0)
    parser.add_argument('-CA', '--Cardiomegaly', type=float, default=0.0)
    parser.add_argument('--default_entities', action='store_true')
    parser.add_argument(
        '-d',
        '--decay_prob_on_add',
        type=float,
        default=1.0,
        help='Probability decay after each added pathology entity.',
    )

    # Devices are treated as nuisance changes and are always attempted for DRR2/3.
    parser.add_argument(
        '--devices_prob',
        type=float,
        default=1.0,
        help='Probability to add devices nuisance changes to DRR2/DRR3. Default: 1.0',
    )

    # Deterministic angles
    parser.add_argument(
        '--prior_angles',
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=('YAW', 'PITCH', 'ROLL'),
        help='Angles (deg) used for DRR1 (prior).',
    )
    parser.add_argument(
        '--angle_x',
        nargs=3,
        type=float,
        default=[10.0, 0.0, 0.0],
        metavar=('YAW', 'PITCH', 'ROLL'),
        help='Angles (deg) used for DRR2/DRR3 (angle X).',
    )

    parser.add_argument(
        '-s',
        '--slices_for_CTs_list',
        nargs=2,
        type=float,
        default=[0.0, 1.0],
    )
    parser.add_argument('-m', '--memory_threshold', type=float, default=15.0)

    return parser.parse_args()


def log_params(params: Dict[str, Any], out_p: str) -> None:
    with open(out_p, 'w') as f:
        json.dump(params, f, indent=4)


# -----------------------------------------------------------------------------
# DEVICES (3D) – nuisance changes
# -----------------------------------------------------------------------------

def _devices_dir() -> str:
    return '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/MedicalDevices'


def apply_devices_to_scan_and_registered_prior(
    *,
    scan: torch.Tensor,
    registrated_prior: torch.Tensor,
    lungs_seg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Add external devices to `scan` and also to `registrated_prior`.

    This is intended to create nuisance changes that should NOT appear as pathology
    differences between DRR2 and DRR3.

    We mimic the "current+registered_prior" branch of `ExternalDevices.add_to_CT_pair`,
    but we force the result to affect both tensors.
    """
    params: Dict[str, Any] = {
        'devices_added': False,
        'devices_components': [],
    }

    devices_p = _devices_dir()
    if not os.path.isdir(devices_p):
        params['devices_warning'] = f"Devices directory not found: {devices_p}"
        return scan, registrated_prior, params

    # Choose at least one device category to apply.
    add_wires = add_stickers = add_devices = False
    while not (add_wires or add_stickers or add_devices):
        add_wires = random.random() < 0.4
        add_stickers = random.random() < 0.4
        add_devices = random.random() < 0.3

    if add_wires:
        num_cables = int((random.random() ** 2.5) * 8 + 1)
        scan, registrated_prior, _ = ExternalDevices.add_cables(
            scan, lungs_seg, p1s=None, num_cables=num_cables, r_prior=registrated_prior
        )
        params['devices_components'].append({'type': 'cables', 'count': num_cables})

    if add_stickers:
        num_stickers = int((random.random() ** 1.5) * 6 + 1)
        scan, registrated_prior, p1s = ExternalDevices.add_stickers(
            scan, lungs_seg, n=num_stickers, r_prior=registrated_prior
        )
        params['devices_components'].append({'type': 'stickers', 'count': num_stickers})
        if random.random() < 0.85:
            scan, registrated_prior, _ = ExternalDevices.add_cables(
                scan, lungs_seg, p1s=p1s, num_cables=None, r_prior=registrated_prior
            )
            params['devices_components'].append({'type': 'cables_from_stickers'})

    if add_devices:
        num_devices = int(math.ceil((random.random() ** 3) * 3))
        list_devs = os.listdir(devices_p)
        random.shuffle(list_devs)
        list_devs = list_devs[:num_devices]

        used = []
        for dev_name in list_devs:
            dev_path = f'{devices_p}/{dev_name}'
            scan, dev_p1, registrated_prior = ExternalDevices.add_device(
                scan, lungs_seg, dev_path, registrated_prior=registrated_prior
            )
            used.append(dev_name)
            if random.random() < 0.6:
                scan, registrated_prior, _ = ExternalDevices.add_cables(
                    scan, lungs_seg, p1s=dev_p1, num_cables=None, r_prior=registrated_prior
                )

        params['devices_components'].append({'type': 'device_volumes', 'names': used})

    params['devices_added'] = True
    return scan, registrated_prior, params


# -----------------------------------------------------------------------------
# ANGLES / ROTATION (deterministic)
# -----------------------------------------------------------------------------

def rotate_and_project(
    ct: torch.Tensor,
    seg: torch.Tensor,
    angles_deg: Tuple[float, float, float],
) -> np.ndarray:
    """Rotate+crop deterministically, then project to DRR.

    DRR PROJECTION happens inside `base.project_ct`.
    """
    yaw, pitch, roll = angles_deg
    rotated_ct, _ = rotate_ct_and_crop_according_to_seg(
        ct,
        seg,
        rotate_angle1=yaw,
        rotate_angle2=pitch,
        rotate_angle3=roll,
        return_ct_seg=True,
    )
    rotated_ct = rotated_ct.squeeze()
    return base.project_ct(rotated_ct)


def rotate_stack_and_project(
    ct_stack: torch.Tensor,
    seg: torch.Tensor,
    angles_deg: Tuple[float, float, float],
) -> List[np.ndarray]:
    """Rotate+crop a (C,D,H,W) stack with the same angles and project each channel."""
    yaw, pitch, roll = angles_deg
    rotated_ct_stack, _ = rotate_ct_and_crop_according_to_seg(
        ct_stack,
        seg,
        rotate_angle1=yaw,
        rotate_angle2=pitch,
        rotate_angle3=roll,
        return_ct_seg=True,
    )
    rotated_ct_stack = rotated_ct_stack  # shape (C,D,H,W)
    return [base.project_ct(rotated_ct_stack[i]) for i in range(rotated_ct_stack.shape[0])]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    total_pairs_num = int(args.number_pairs)

    # Pathology entities for DRR3
    intra_names = {'Consolidation', 'FluidOverload'}
    extra_names = {'PleuralEffusion', 'Pneumothorax', 'Cardiomegaly'}

    entity_name_to_class = {
        'Consolidation': Consolidation,
        'FluidOverload': FluidOverload,
        'PleuralEffusion': PleuralEffusion,
        'Pneumothorax': Pneumothorax,
        'Cardiomegaly': Cardiomegaly,
    }

    if args.default_entities:
        probs = {
            'Consolidation': 0.15,
            'FluidOverload': 0.15,
            'PleuralEffusion': 0.15,
            'Pneumothorax': 0.15,
            'Cardiomegaly': 0.15,
        }
    else:
        probs = {
            'Consolidation': float(args.Consolidation),
            'FluidOverload': float(args.FluidOverload),
            'PleuralEffusion': float(args.PleuralEffusion),
            'Pneumothorax': float(args.Pneumothorax),
            'Cardiomegaly': float(args.Cardiomegaly),
        }

    cur_intra: List[Tuple[type, float]] = []
    cur_extra: List[Tuple[type, float]] = []

    for name, prob in probs.items():
        if name in intra_names:
            cur_intra.append((entity_name_to_class[name], prob))
        elif name in extra_names:
            cur_extra.append((entity_name_to_class[name], prob))
        else:
            raise ValueError(f"Unknown entity name: {name}")

    entity_prob_decay_on_addition = float(args.decay_prob_on_add)

    prior_angles = tuple(float(x) for x in args.prior_angles)
    angle_x = tuple(float(x) for x in args.angle_x)

    out_base_dir = args.output

    ct_paths: List[str] = []
    for ct_dir in args.input:
        ct_paths.extend([ct_dir + f'/{n}' for n in os.listdir(ct_dir)])
    ct_paths = sorted(ct_paths)

    num_paths = len(ct_paths)
    pairs_per_ct = math.ceil(total_pairs_num / max(num_paths, 1))

    a, b = (float(args.slices_for_CTs_list[0]), float(args.slices_for_CTs_list[1]))
    first = int(a * num_paths)
    last = int(b * num_paths + 1)
    ps = ct_paths[first:last]

    MEMORY_THRESHOLD_GB = float(args.memory_threshold)

    pairs_created = 0

    print(f"Total triplets to create: {total_pairs_num} | CTs: {num_paths} | triplets/CT: {pairs_per_ct}")
    print(f"Angles: prior={prior_angles} | angle_x={angle_x}")
    print(f"Devices prob: {args.devices_prob}")

    for k, ct_p in enumerate(ps):
        case_name = ct_p.split('/')[-1][:-7]
        out_path = f'{out_base_dir}/{case_name}'

        if os.path.exists(f'{out_path}/pair{pairs_per_ct - 1}/params.json'):
            print(f"Case {case_name} done. Skipping")
            continue

        try:
            scan_p = ct_p
            seg_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case_name}_seg.nii.gz'
            middle_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_middle_lobe_right.nii.gz'
            upper_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_right.nii.gz'
            lower_right_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_right.nii.gz'
            upper_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_upper_lobe_left.nii.gz'
            lower_left_lobe_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_lobes_segs/{case_name}_lung_lower_lobe_left.nii.gz'
            heart_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_heart_segs/{case_name}_heart_seg.nii.gz'
            bronchi_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_bronchi_segs/{case_name}_bronchia_seg.nii.gz'
            lung_vessels_p = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_vessels_segs/{case_name}_vessels_seg.nii.gz'

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
            smoothing_radius_dict = {
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

            scan, segs_dict = base.load_scan_and_seg(scan_p, segs_p_dict)
            orig_scan = scan.clone()

            # Rebuild lungs from lobes
            segs_dict['lungs'] = torch.clamp_max(
                segs_dict['middle_right_lobe']
                + segs_dict['upper_right_lobe']
                + segs_dict['lower_right_lobe']
                + segs_dict['upper_left_lobe']
                + segs_dict['lower_left_lobe'],
                1.0,
            )

            if segs_dict['lungs'].sum() < 20:
                print(f"Empty lungs seg for {case_name}. Skipping")
                continue

            # Crop around lungs for faster entity logic
            lung_coords = segs_dict['lungs'].nonzero().T
            lungs_h = torch.max(lung_coords[0]) - torch.min(lung_coords[0])
            h_low_ext = int((lungs_h // 5).item())

            cropped_scan, cropped_segs_dict, cropping_slices = base.crop_according_to_seg(
                scan,
                segs_dict['lungs'],
                all_segs_dict=segs_dict,
                tight_y=False,
                ext=15,
                h_low_ext=h_low_ext,
            )

            for organ_name, c_seg in list(cropped_segs_dict.items()):
                c_seg = base.smooth_segmentation(c_seg.to(DEVICE), radius=int(smoothing_radius_dict[organ_name])).cpu()
                cropped_segs_dict[organ_name] = c_seg
                if torch.sum(c_seg) < 20:
                    raise RuntimeError(f"Bad segmentation: {organ_name}")

            cropped_segs_dict['lung_right'] = torch.maximum(
                cropped_segs_dict['middle_right_lobe'],
                torch.maximum(cropped_segs_dict['upper_right_lobe'], cropped_segs_dict['lower_right_lobe']),
            )
            cropped_segs_dict['lung_left'] = torch.maximum(
                cropped_segs_dict['upper_left_lobe'],
                cropped_segs_dict['lower_left_lobe'],
            )

            orig_cropped_scan = cropped_scan.clone().cpu()
            orig_cropped_lungs = cropped_segs_dict['lungs'].clone()

            # Per-pair template seg dicts
            prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
            current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

            for i in range(pairs_per_ct):
                c_out_dir = f'{out_path}/pair{i}'
                os.makedirs(c_out_dir, exist_ok=True)

                if os.path.exists(f'{c_out_dir}/params.json'):
                    pairs_created += 1
                    continue

                params: Dict[str, Any] = {
                    'prior_angles_deg': prior_angles,
                    'angle_x_deg': angle_x,
                }

                # -----------------------------------------------------------------
                # prior: baseline CT at prior_angles
                # -----------------------------------------------------------------
                prior_full = orig_scan.clone().to(DEVICE)
                lungs_full = segs_dict['lungs'].clone().to(DEVICE)
                drr1 = rotate_and_project(prior_full, lungs_full, prior_angles)

                # -----------------------------------------------------------------
                # intermediate/current base: start from baseline CT crop, add DEVICES (nuisance)
                # -----------------------------------------------------------------
                # Work in cropped space for devices so the device placement is consistent,
                # then restore to full-size before rotation.
                base_scan2 = cropped_scan.clone().to(DEVICE)
                base_reg_prior2 = base_scan2.clone()
                lungs_crop = cropped_segs_dict['lungs'].clone().to(DEVICE)

                if random.random() < float(args.devices_prob):
                    base_scan2, base_reg_prior2, dev_params = apply_devices_to_scan_and_registered_prior(
                        scan=base_scan2,
                        registrated_prior=base_reg_prior2,
                        lungs_seg=lungs_crop,
                    )
                    params.update(dev_params)
                else:
                    params['devices_added'] = False

                # current starts as intermediate base (same devices, same angle)
                base_scan3 = base_scan2.clone()

                # -----------------------------------------------------------------
                # ENTITY INSERTION (3D): optional pathology, affects ONLY current
                # -----------------------------------------------------------------
                # IMPORTANT: pass a copy of scan2 as "prior" so scan2 stays unchanged.
                segs2 = [
                    {k: v.clone() for k, v in prior_cropped_segs_dict.items()},
                    {k: v.clone() for k, v in current_cropped_segs_dict.items()},
                ]
                ret_ent = base.add_entities_to_pair(
                    scans=[base_scan2.clone(), base_scan3],
                    segs=segs2,
                    orig_scan=orig_cropped_scan,
                    orig_lungs=orig_cropped_lungs,
                    intra_pulmonary_entities=cur_intra,
                    extra_pulmonary_entities=cur_extra,
                    devices_entity=[],
                    entity_prob_decay=entity_prob_decay_on_addition,
                )
                params['added_entities'] = ret_ent['added_entity_names']
                base_scan3 = ret_ent['scans'][1]

                # Restore to full-size for deterministic rotation
                scan2_full = base.add_back_cropped(base_scan2, orig_scan, cropping_slices)
                scan3_full = base.add_back_cropped(base_scan3, orig_scan, cropping_slices)

                # Rotate+project intermediate/current with the same angles.
                # We rotate both volumes together to guarantee identical cropping.
                ct_stack = torch.stack([scan2_full, scan3_full], dim=0)
                drr2, drr3 = rotate_stack_and_project(ct_stack, lungs_full, angle_x)

                # -----------------------------------------------------------------
                # Save
                # -----------------------------------------------------------------
                # Diff map like the original generator: from prior -> final current
                diff_map = base.calculate_diff_map(drr3, drr1, boundary_seg=None)

                base.save_arr_as_nifti(drr1.T, f'{c_out_dir}/prior.nii.gz')
                base.save_arr_as_nifti(drr2.T, f'{c_out_dir}/intermediate.nii.gz')
                base.save_arr_as_nifti(drr3.T, f'{c_out_dir}/current.nii.gz')
                base.save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

                plt.imsave(f'{c_out_dir}/prior.png', drr1, cmap='gray')
                plt.imsave(f'{c_out_dir}/intermediate.png', drr2, cmap='gray')
                plt.imsave(f'{c_out_dir}/current.png', drr3, cmap='gray')
                plt.imsave(f'{c_out_dir}/diff_map.png', diff_map, cmap='gray')
                base.plot_diff_on_current(diff_map, drr3, f'{c_out_dir}/current_with_differences.png')

                log_params(params, f'{c_out_dir}/params.json')

                pairs_created += 1

                # Cleanup
                del base_scan2, base_scan3, base_reg_prior2
                del scan2_full, scan3_full, ct_stack
                del diff_map
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pairs_created >= total_pairs_num:
                    print('Done generating triplets. Exiting.')
                    return

            # memory guard
            if base.check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f'After case {case_name}'):
                pass

        except Exception as e:
            print(f"[ERROR] Exception on case {case_name}: {e}")
            import traceback

            traceback.print_exc()
            print('[ERROR] Skipping to next case...')

        finally:
            for _ in range(2):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
