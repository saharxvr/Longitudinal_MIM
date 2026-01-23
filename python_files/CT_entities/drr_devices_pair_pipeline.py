"""Paired DRR generator: clean vs devices from identical angles.

Goal
----
For each CT case, pre-sample N rotation angles and generate *paired* DRRs:
- `prior.nii.gz`   : clean DRR (no devices)
- `current.nii.gz` : same-angle DRR with injected external devices
- `diff_map.nii.gz`: (current - prior)

Each pair is saved in an organized per-case directory with the angles encoded
in the folder name and also stored in `params.json`.

This script intentionally reuses the existing project primitives:
- Fixed-angle rotation+crop: `CT_entities/CT_Rotations.py::rotate_ct_and_crop_according_to_seg`
- Angle sampling constraints: `CT_entities/CT_Rotations.py::get_random_rotation_angles`
- Device insertion ops: `CT_entities/External_Devices.py::ExternalDevices` (low-level helpers)
- DRR projection: identical to `CT_entities/DRR_generator_clean.py::project_ct`

Typical use
-----------
python CT_entities/drr_devices_pair_pipeline.py \
  --ct_dirs /path/to/CTs \
  --lungs_seg_pattern /path/to/segs/{case}_seg.nii.gz \
  --devices_dir /path/to/MedicalDevices \
  --output /path/to/out \
	--device_variants_per_ct 4 \
	--num_angles_per_variant 5 \
  --rotation_params 17.5 37.5 0.0 1.75 \
  --seed 123

Notes
-----
- This is a *data generation* script; it does not modify training code.
- Rotation angles are in degrees.
- `prior/current` naming is kept to be compatible with existing dataset loaders.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.v2 as v2

from constants import DEVICE
from CT_Rotations import get_random_rotation_angles, rotate_ct_and_crop_according_to_seg
from External_Devices import ExternalDevices


# -----------------------------------------------------------------------------
# Small globals
# -----------------------------------------------------------------------------

_resize = v2.Resize((512, 512))


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

affine: Optional[np.ndarray] = None


def _case_name_from_path(p: str) -> str:
	base = os.path.basename(p)
	if base.endswith('.nii.gz'):
		return base[:-7]
	if base.endswith('.nii'):
		return base[:-4]
	return os.path.splitext(base)[0]


def load_ct_and_lungs_seg(ct_path: str, lungs_seg_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Load CT + lungs segmentation as (D,H,W) torch tensors.

	Matches `DRR_generator_clean.load_scan_and_seg` conventions:
	- load NIfTI to numpy
	- transpose to (D,H,W)
	- flip depth axis
	"""
	global affine

	ct_nif = nib.load(ct_path)
	affine = ct_nif.affine
	ct = np.asarray(ct_nif.dataobj, dtype=np.float32)
	ct = torch.from_numpy(np.transpose(ct, (2, 0, 1)).copy())
	ct_nif.uncache()
	del ct_nif
	ct = torch.flip(ct, dims=[0])

	seg_nif = nib.load(lungs_seg_path)
	seg = np.asarray(seg_nif.dataobj, dtype=np.float32)
	seg = torch.from_numpy(np.transpose(seg, (2, 0, 1)).copy())
	seg_nif.uncache()
	del seg_nif
	seg = torch.flip(seg, dims=[0])

	# Binarize defensively
	seg = (seg > 0.5).float()
	return ct, seg


def save_arr_as_nifti(arr: np.ndarray | torch.Tensor, out_path: str) -> None:
	"""Save array as NIfTI using the affine captured when loading the CT."""
	if isinstance(arr, torch.Tensor):
		arr = np.array(arr.detach().float().cpu())

	if affine is None:
		raise RuntimeError('Affine not initialized. Call load_ct_and_lungs_seg first.')

	nif = nib.Nifti1Image(arr, affine)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	nib.save(nif, out_path)


def log_params(d: Dict[str, Any], out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, 'w', encoding='utf-8') as f:
		json.dump(d, f, indent=2)


# -----------------------------------------------------------------------------
# DRR projection (same logic as DRR_generator_clean.project_ct)
# -----------------------------------------------------------------------------


def project_ct(rotated_ct_scan: torch.Tensor, dim: int = 2, *, is_seg: bool = False) -> np.ndarray:
	"""Project a (rotated) CT volume to a 2D DRR-like image."""
	if is_seg:
		xray_image = torch.amax(rotated_ct_scan, dim=dim)
	else:
		rotated_ct_scan = torch.clamp_min(rotated_ct_scan, -1000)
		m = torch.min(rotated_ct_scan)
		rotated_ct_scan = rotated_ct_scan - m
		xray_image = torch.sum(rotated_ct_scan, dim=dim)

	xray_image = _resize(xray_image[None, ...]).squeeze()
	return xray_image.detach().cpu().numpy()


def normalize_01(x: np.ndarray) -> np.ndarray:
	mn = float(np.min(x))
	mx = float(np.max(x))
	return (x - mn) / (mx - mn + 1e-8)


# -----------------------------------------------------------------------------
# Devices injection (single CT)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DevicesConfig:
	devices_dir: str
	p_wires: float = 0.4
	p_stickers: float = 0.4
	p_devices: float = 0.3
	max_cables: int = 9
	max_stickers: int = 7
	max_devices: int = 3
	p_cables_after_stickers: float = 0.85
	p_cables_after_device: float = 0.6
	force_any: bool = True


def _add_external_devices_to_single_ct(
	scan: torch.Tensor,
	lungs_seg: torch.Tensor,
	cfg: DevicesConfig,
	*,
	rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
	"""Inject external devices into a single CT volume.

	Uses `ExternalDevices.add_cables/add_stickers/add_device` directly so we
	don't accidentally modify the 'clean' CT.
	"""
	if rng is None:
		rng = random

	if not os.path.isdir(cfg.devices_dir):
		raise FileNotFoundError(f"devices_dir not found: {cfg.devices_dir}")

	added: Dict[str, Any] = {
		'added_wires': False,
		'added_stickers': False,
		'added_devices': False,
		'num_cables': 0,
		'num_stickers': 0,
		'device_files': [],
	}

	# Decide what to add; ensure at least one if requested
	add_wires = rng.random() < cfg.p_wires
	add_stickers = rng.random() < cfg.p_stickers
	add_devices = rng.random() < cfg.p_devices

	if cfg.force_any and (not add_wires and not add_stickers and not add_devices):
		# Pick one category at random
		choice = rng.choice(['wires', 'stickers', 'devices'])
		add_wires = choice == 'wires'
		add_stickers = choice == 'stickers'
		add_devices = choice == 'devices'

	# Apply in a stable order
	p1s = None
	if add_wires:
		num_cables = int((rng.random() ** 2.5) * cfg.max_cables + 1)
		scan, _ = ExternalDevices.add_cables(scan, lungs_seg, p1s=None, num_cables=num_cables, r_prior=None)
		added['added_wires'] = True
		added['num_cables'] = num_cables

	if add_stickers:
		num_stickers = int((rng.random() ** 1.5) * cfg.max_stickers + 1)
		scan, p1s = ExternalDevices.add_stickers(scan, lungs_seg, n=num_stickers, r_prior=None)
		added['added_stickers'] = True
		added['num_stickers'] = num_stickers
		if rng.random() < cfg.p_cables_after_stickers:
			scan, _ = ExternalDevices.add_cables(scan, lungs_seg, p1s, num_cables=None, r_prior=None)

	if add_devices:
		list_devs = os.listdir(cfg.devices_dir)
		list_devs = [n for n in list_devs if n.endswith('.nii') or n.endswith('.nii.gz')]
		if len(list_devs) == 0:
			raise RuntimeError(f"No NIfTI devices found in: {cfg.devices_dir}")
		rng.shuffle(list_devs)

		num_devices = int(math.ceil((rng.random() ** 3) * cfg.max_devices))
		num_devices = max(1, min(num_devices, len(list_devs)))
		for i in range(num_devices):
			dev_name = list_devs[i]
			dev_p = os.path.join(cfg.devices_dir, dev_name)
			scan, dev_p1 = ExternalDevices.add_device(scan, lungs_seg, dev_p, registrated_prior=None)
			added['device_files'].append(dev_name)
			if rng.random() < cfg.p_cables_after_device:
				scan, _ = ExternalDevices.add_cables(scan, lungs_seg, p1s=dev_p1, num_cables=None, r_prior=None)

		added['added_devices'] = True

	return scan, added


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description='Generate paired DRRs (clean vs devices) from fixed angles per CT.',
		formatter_class=argparse.RawTextHelpFormatter,
	)

	p.add_argument(
		'--ct_dirs',
		nargs='+',
		required=True,
		help='Space-separated list of directories containing CT NIfTI files.',
	)
	p.add_argument(
		'--lungs_seg_pattern',
		required=True,
		help='Format string with {case} placeholder pointing to lungs seg NIfTI.\nExample: /path/segs/{case}_seg.nii.gz',
	)
	p.add_argument('--output', '-o', required=True, help='Output base directory.')
	p.add_argument(
		'--device_variants_per_ct',
		type=int,
		default=1,
		help='How many different device injections to generate per CT (each variant gets its own device placement).',
	)
	p.add_argument(
		'--num_angles_per_variant',
		type=int,
		required=True,
		help='How many angles/pairs to generate per (CT, device-variant).',
	)

	p.add_argument(
		'--rotation_params',
		nargs=4,
		type=float,
		default=[17.5, 37.5, 0.0, 1.75],
		help='Four floats: [max_abs_per_axis_deg, max_sum_deg, min_sum_deg, exponent]',
	)

	p.add_argument('--devices_dir', required=True, help='Directory containing device NIfTI volumes (MedicalDevices).')
	p.add_argument('--p_wires', type=float, default=0.4)
	p.add_argument('--p_stickers', type=float, default=0.4)
	p.add_argument('--p_devices', type=float, default=0.3)
	p.add_argument('--no_force_any', action='store_true', help='If set, allow generating a "devices" sample with no devices added.')

	p.add_argument('--max_cables', type=int, default=9)
	p.add_argument('--max_stickers', type=int, default=7)
	p.add_argument('--max_devices', type=int, default=3)

	p.add_argument('--seed', type=int, default=0, help='Global RNG seed for reproducibility.')
	p.add_argument(
		'--slices_for_CTs_list',
		nargs=2,
		type=float,
		default=[0.0, 1.0],
		help='Two floats [a,b] in [0,1] to process ct_paths[int(a*N):int(b*N+1)].',
	)

	p.add_argument('--save_png', action='store_true', help='Also save PNG previews (normalized to [0,1]).')
	p.add_argument('--overwrite', action='store_true', help='Overwrite existing pair directories.')
	p.add_argument('--memory_cleanup_every', type=int, default=5, help='Run gc/cuda cache cleanup every K pairs.')

	return p.parse_args()


def _sanitize_angle_tag(a: float) -> str:
	# file-system friendly tag: keep sign, replace '.' with 'p'
	s = f"{a:.2f}"
	return s.replace('.', 'p')


def main() -> None:
	args = parse_args()

	seed = int(args.seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	rot_range = float(args.rotation_params[0])
	rot_ranges = (rot_range, rot_range, rot_range)
	max_sum = float(args.rotation_params[1])
	min_sum = float(args.rotation_params[2])
	rot_exp = float(args.rotation_params[3])

	devices_cfg = DevicesConfig(
		devices_dir=str(args.devices_dir),
		p_wires=float(args.p_wires),
		p_stickers=float(args.p_stickers),
		p_devices=float(args.p_devices),
		max_cables=int(args.max_cables),
		max_stickers=int(args.max_stickers),
		max_devices=int(args.max_devices),
		force_any=(not bool(args.no_force_any)),
	)

	# Collect CT paths
	ct_paths: List[str] = []
	for d in args.ct_dirs:
		for n in os.listdir(d):
			p = os.path.join(d, n)
			if os.path.isfile(p) and (n.endswith('.nii') or n.endswith('.nii.gz')):
				ct_paths.append(p)
	ct_paths = sorted(ct_paths)

	if len(ct_paths) == 0:
		raise RuntimeError('No CT NIfTI files found under ct_dirs.')

	a, b = [float(x) for x in args.slices_for_CTs_list]
	first = int(a * len(ct_paths))
	last = int(b * len(ct_paths) + 1)
	ct_paths = ct_paths[first:last]

	variants_per_ct = int(args.device_variants_per_ct)
	if variants_per_ct < 1:
		raise ValueError('--device_variants_per_ct must be >= 1')

	angles_per_variant = int(args.num_angles_per_variant)
	if angles_per_variant < 1:
		raise ValueError('--num_angles_per_variant must be >= 1')

	pairs_done = 0
	for ct_idx, ct_path in enumerate(ct_paths):
		case = _case_name_from_path(ct_path)
		lungs_seg_path = str(args.lungs_seg_pattern).format(case=case)

		print(f"\n{'='*60}\nCT {ct_idx+1}/{len(ct_paths)} | case={case}")
		print(f"CT: {ct_path}")
		print(f"Lungs seg: {lungs_seg_path}")

		if not os.path.isfile(lungs_seg_path):
			print('[WARN] Missing lungs seg. Skipping case.')
			continue

		ct_cpu, lungs_cpu = load_ct_and_lungs_seg(ct_path, lungs_seg_path)
		if float(lungs_cpu.sum().item()) < 20:
			print('[WARN] Empty lungs seg. Skipping case.')
			continue

		ct = ct_cpu.to(DEVICE)
		lungs = lungs_cpu.to(DEVICE)

		for variant_idx in range(variants_per_ct):
			# Create a devices-augmented CT per (case, variant)
			variant_seed = seed + 1000003 * (ct_idx + 1) + 9176 * (variant_idx + 1)
			rng = random.Random(variant_seed)
			dev_ct, devices_meta = _add_external_devices_to_single_ct(ct.clone(), lungs, devices_cfg, rng=rng)

			# Pre-sample angles per variant
			angles: List[Tuple[float, float, float]] = []
			for _ in range(angles_per_variant):
				a1, a2, a3 = get_random_rotation_angles(rot_ranges, max_sum, min_sum, rot_exp)
				angles.append((float(a1), float(a2), float(a3)))

			for i, (a1, a2, a3) in enumerate(angles):
				pair_dir = os.path.join(
					args.output,
					case,
					f"variant{variant_idx:03d}",
					f"pair{i:05d}_ax{_sanitize_angle_tag(a1)}_ay{_sanitize_angle_tag(a2)}_az{_sanitize_angle_tag(a3)}",
				)

				params_path = os.path.join(pair_dir, 'params.json')
				if os.path.exists(params_path) and (not args.overwrite):
					print(f"[Skip] Exists: {pair_dir}")
					continue

				os.makedirs(pair_dir, exist_ok=True)

				# Rotate+crop both volumes together to ensure identical crop/framing.
				ct_cat = torch.stack([ct, dev_ct], dim=0)
				rotated_ct_cat, _ = rotate_ct_and_crop_according_to_seg(
					ct_cat,
					lungs,
					rotate_angle1=a1,
					rotate_angle2=a2,
					rotate_angle3=a3,
					return_ct_seg=True,
				)

				clean_rot = rotated_ct_cat[0]
				dev_rot = rotated_ct_cat[1]

				clean_drr = project_ct(clean_rot)
				dev_drr = project_ct(dev_rot)
				diff_map = dev_drr - clean_drr

				save_arr_as_nifti(clean_drr.T, os.path.join(pair_dir, 'prior.nii.gz'))
				save_arr_as_nifti(dev_drr.T, os.path.join(pair_dir, 'current.nii.gz'))
				save_arr_as_nifti(diff_map.T, os.path.join(pair_dir, 'diff_map.nii.gz'))

				params: Dict[str, Any] = {
					'case': case,
					'ct_path': ct_path,
					'lungs_seg_path': lungs_seg_path,
					'device_variant_idx': variant_idx,
					'rotation_angles_deg': (a1, a2, a3),
					'rotation_params': {
						'rot_ranges': rot_ranges,
						'max_sum': max_sum,
						'min_sum': min_sum,
						'exponent': rot_exp,
					},
					'devices_config': {
						'devices_dir': devices_cfg.devices_dir,
						'p_wires': devices_cfg.p_wires,
						'p_stickers': devices_cfg.p_stickers,
						'p_devices': devices_cfg.p_devices,
						'max_cables': devices_cfg.max_cables,
						'max_stickers': devices_cfg.max_stickers,
						'max_devices': devices_cfg.max_devices,
						'force_any': devices_cfg.force_any,
					},
					'devices_added': devices_meta,
					'seed': seed,
					'variant_rng_seed': variant_seed,
				}
				log_params(params, params_path)

				if args.save_png:
					# Local import (matplotlib is slow to import).
					import matplotlib.pyplot as plt

					plt.imsave(os.path.join(pair_dir, 'prior.png'), normalize_01(clean_drr), cmap='gray')
					plt.imsave(os.path.join(pair_dir, 'current.png'), normalize_01(dev_drr), cmap='gray')
					plt.imsave(os.path.join(pair_dir, 'diff_map.png'), normalize_01(np.abs(diff_map)), cmap='magma')

				pairs_done += 1
				if args.memory_cleanup_every > 0 and (pairs_done % int(args.memory_cleanup_every) == 0):
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()

				print(f"[OK] {pair_dir}")

			# Free per-variant tensors
			del dev_ct
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		# Free per-case tensors aggressively
		del ct_cpu, lungs_cpu, ct, lungs
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	print(f"\nDone. Generated {pairs_done} paired samples.")


if __name__ == '__main__':
	main()
