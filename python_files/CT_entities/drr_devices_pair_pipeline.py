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
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.v2 as v2

# Ensure project root is importable when running as a script from any CWD.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
	sys.path.insert(0, _project_root)

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

	if add_devices and (not os.path.isdir(cfg.devices_dir)):
		raise FileNotFoundError(
			"devices_dir not found but p_devices>0. "
			f"Got: {cfg.devices_dir}. "
			"Either pass a valid --devices_dir or set --p_devices 0."
		)

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

	# Keep defaults aligned with CT_entities/DRR_generator.py so you can run without
	# providing paths on the lab servers.
	default_ct_dirs = [
		'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/CT-RATE_scans',
		'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/LUNA_scans',
		'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans',
	]

	p.add_argument(
		'--ct_dirs',
		'-i',
		'--input',
		nargs='+',
		default=default_ct_dirs,
		help='Space-separated list of directories containing CT NIfTI files.',
	)
	# In DRR_generator the lungs seg is usually stored under CT_scans/scans_segs
	# with the pattern: {case}_seg.nii.gz
	p.add_argument(
		'--lungs_seg_pattern',
		default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/scans_segs/{case}_seg.nii.gz',
		help='Format string with {case} placeholder pointing to lungs seg NIfTI.',
	)
	p.add_argument(
		'--output',
		'-o',
		default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/devices_pairs',
		help='Output base directory.',
	)
	p.add_argument(
		'--device_variants_per_ct',
		type=int,
		default=4,
		help='How many different device injections to generate per CT (each variant gets its own device placement).',
	)
	p.add_argument(
		'--num_angles_per_variant',
		type=int,
		default=5,
		help='How many angles/pairs to generate per (CT, device-variant).',
	)

	p.add_argument(
		'--rotation_params',
		nargs=4,
		type=float,
		default=[17.5, 37.5, 0.0, 1.75],
		help='Four floats: [max_abs_per_axis_deg, max_sum_deg, min_sum_deg, exponent]',
	)

	p.add_argument(
		'--devices_dir',
		default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/MedicalDevices',
		help=(
			'Directory containing 3D device NIfTI volumes (MedicalDevices).\n'
			'Matches the default used in External_Devices.py.\n'
			'Only required if --p_devices > 0.'
		),
	)
	p.add_argument('--p_wires', type=float, default=0.4)
	p.add_argument('--p_stickers', type=float, default=0.4)
	p.add_argument('--p_devices', type=float, default=0.3)
	p.add_argument('--no_force_any', action='store_true', help='If set, allow generating a "devices" sample with no devices added.')

	p.add_argument('--max_cables', type=int, default=9)
	p.add_argument('--max_stickers', type=int, default=7)
	p.add_argument('--max_devices', type=int, default=3)

	p.add_argument('--seed', type=int, default=0, help='Global RNG seed for reproducibility.')
	p.add_argument(
		'--device',
		default=str(DEVICE),
		help='Torch device to run generation on (e.g. cpu, cuda, cuda:0). Use cpu to avoid CUDA OOM.',
	)
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
	p.add_argument(
		'--max_errors',
		type=int,
		default=200,
		help='Maximum number of caught errors before aborting (prevents infinite error loops).',
	)
	p.add_argument(
		'--errors_log',
		default='errors.jsonl',
		help='Filename (under --output) to append JSONL error records to.',
	)
	p.add_argument(
		'--done_marker',
		default='_DONE.json',
		help='Filename created under each variant dir when the variant finished successfully. Used to skip work on resume.',
	)
	p.add_argument(
		'--bootstrap_done_markers',
		action='store_true',
		help=(
			'One-time fast resume setup: scan existing output folders and write missing per-variant done markers\n'
			'when all expected pairs (0..num_angles_per_variant-1) already exist. Does not generate any data.'
		),
	)

	return p.parse_args()


def _pair_idx_from_dirname(name: str) -> Optional[int]:
	"""Extract pair index from a folder name like 'pair00012_ax...'."""
	m = re.match(r'^pair(\d+)_', name)
	if not m:
		return None
	try:
		return int(m.group(1))
	except Exception:
		return None


def _bootstrap_done_markers_for_output(
	output_dir: str,
	*,
	case_names: Sequence[str],
	variants_per_ct: int,
	angles_per_variant: int,
	done_marker_name: str,
	overwrite: bool,
) -> None:
	"""Scan output tree and write missing done markers for fully completed variants.

	This is meant to be a one-time setup to enable fast resume for runs created
	before the done-marker optimization existed.
	"""
	expected = set(range(int(angles_per_variant)))
	bootstrapped = 0
	considered = 0
	for case in case_names:
		case_dir = os.path.join(output_dir, case)
		if not os.path.isdir(case_dir):
			continue
		for variant_idx in range(int(variants_per_ct)):
			variant_dir = os.path.join(case_dir, f"variant{variant_idx:03d}")
			if not os.path.isdir(variant_dir):
				continue
			done_path = os.path.join(variant_dir, done_marker_name)
			if os.path.exists(done_path) and (not overwrite):
				continue

			considered += 1
			found_ok: set[int] = set()
			try:
				for entry in os.listdir(variant_dir):
					pair_idx = _pair_idx_from_dirname(entry)
					if pair_idx is None:
						continue
					if pair_idx not in expected:
						continue
					pair_dir = os.path.join(variant_dir, entry)
					if not os.path.isdir(pair_dir):
						continue
					if os.path.exists(os.path.join(pair_dir, 'params.json')):
						found_ok.add(pair_idx)
			except Exception:
				# If a directory is transient/broken, don't mark it.
				continue

			if found_ok == expected:
				os.makedirs(variant_dir, exist_ok=True)
				with open(done_path, 'w', encoding='utf-8') as f:
					json.dump(
						{
							'case': case,
							'variant_idx': variant_idx,
							'angles_per_variant': int(angles_per_variant),
							'timestamp': time.time(),
							'bootstrapped': True,
							'criterion': 'all params.json exist for pair indices 0..num_angles_per_variant-1',
						},
						f,
						indent=2,
					)
				bootstrapped += 1
				print(f"[OK] Bootstrapped done marker: {done_path}")

	print(f"\nBootstrap finished. Wrote {bootstrapped} done markers (checked {considered} variants).")


def _sanitize_angle_tag(a: float) -> str:
	# file-system friendly tag: keep sign, replace '.' with 'p'
	s = f"{a:.2f}"
	return s.replace('.', 'p')


def _append_error_record(out_dir: str, filename: str, record: Dict[str, Any]) -> None:
	"""Append an error record as a JSON line under the output directory."""
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, filename)
	with open(path, 'a', encoding='utf-8') as f:
		f.write(json.dumps(record, ensure_ascii=False) + '\n')


def _write_done_marker(done_path: str, payload: Dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(done_path), exist_ok=True)
	with open(done_path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)


def _variant_complete_on_disk(variant_dir: str, angles_per_variant: int) -> bool:
	"""Return True if all expected pair params exist under variant_dir.

	Checks for params.json existence for pair indices 0..angles_per_variant-1.
	This is used only on shutdown paths (Ctrl+C) to avoid falsely marking a
	partial variant as done.
	"""
	for i in range(int(angles_per_variant)):
		# Angle tag varies, so we need to search by prefix.
		prefix = f"pair{i:05d}_"
		found = False
		try:
			for entry in os.listdir(variant_dir):
				if not entry.startswith(prefix):
					continue
				pair_dir = os.path.join(variant_dir, entry)
				if os.path.isdir(pair_dir) and os.path.exists(os.path.join(pair_dir, 'params.json')):
					found = True
					break
		except Exception:
			return False
		if not found:
			return False
	return True


def main() -> None:
	args = parse_args()
	device = torch.device(str(args.device))

	shutdown_requested = {'flag': False, 'signal': None}

	def _on_signal(sig: int, _frame: Any) -> None:
		shutdown_requested['flag'] = True
		shutdown_requested['signal'] = sig
		# Convert signals to KeyboardInterrupt so we can run a single graceful path.
		raise KeyboardInterrupt

	# Graceful Ctrl+C handling (and SIGTERM on Linux clusters).
	try:
		signal.signal(signal.SIGINT, _on_signal)
	except Exception:
		pass
	try:
		signal.signal(signal.SIGTERM, _on_signal)
	except Exception:
		pass

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

	# One-time setup: generate missing variant done markers from already-generated outputs.
	# This enables fast resume without re-checking every pair directory on subsequent runs.
	if bool(args.bootstrap_done_markers):
		case_names = [_case_name_from_path(p) for p in ct_paths]
		_bootstrap_done_markers_for_output(
			str(args.output),
			case_names=case_names,
			variants_per_ct=variants_per_ct,
			angles_per_variant=angles_per_variant,
			done_marker_name=str(args.done_marker),
			overwrite=bool(args.overwrite),
		)
		return

	pairs_done = 0
	errors = 0
	errors_out_dir = str(args.output)
	errors_log_name = str(args.errors_log)
	max_errors = int(args.max_errors)
	current_case: Optional[str] = None
	current_variant_idx: Optional[int] = None
	current_pair_idx: Optional[int] = None
	current_pair_dir: Optional[str] = None
	current_variant_dir: Optional[str] = None
	current_done_path: Optional[str] = None
	variant_ok = 0
	for ct_idx, ct_path in enumerate(ct_paths):
		case = _case_name_from_path(ct_path)
		current_case = case
		lungs_seg_path = str(args.lungs_seg_pattern).format(case=case)

		print(f"\n{'='*60}\nCT {ct_idx+1}/{len(ct_paths)} | case={case}")
		print(f"CT: {ct_path}")
		print(f"Lungs seg: {lungs_seg_path}")

		if not os.path.isfile(lungs_seg_path):
			print('[WARN] Missing lungs seg. Skipping case.')
			continue

		try:
			ct_cpu, lungs_cpu = load_ct_and_lungs_seg(ct_path, lungs_seg_path)
			if float(lungs_cpu.sum().item()) < 20:
				print('[WARN] Empty lungs seg. Skipping case.')
				continue
			# Keep the big volumes on CPU; stream them to GPU per variant.
			# This drastically lowers peak/steady CUDA memory.
		except KeyboardInterrupt:
			_append_error_record(
				errors_out_dir,
				errors_log_name,
				{
					'timestamp': time.time(),
					'phase': 'interrupt',
					'case': current_case,
					'variant_idx': current_variant_idx,
					'pair_idx': current_pair_idx,
					'pair_dir': current_pair_dir,
					'signal': shutdown_requested.get('signal'),
					'error_type': 'KeyboardInterrupt',
					'error': 'Interrupted by user',
				},
			)
			print('[INFO] Interrupted. Exiting gracefully.')
			return
		except Exception as e:
			errors += 1
			print(f"[ERR] Failed loading case={case}: {type(e).__name__}: {e}")
			_append_error_record(
				errors_out_dir,
				errors_log_name,
				{
					'timestamp': time.time(),
					'phase': 'load_case',
					'case': case,
					'ct_path': ct_path,
					'lungs_seg_path': lungs_seg_path,
					'error_type': type(e).__name__,
					'error': str(e),
					'traceback': traceback.format_exc(),
				},
			)
			if errors >= max_errors:
				raise RuntimeError(f"Too many errors ({errors}); aborting.")
			continue

		for variant_idx in range(variants_per_ct):
			current_variant_idx = variant_idx
			variant_dir = os.path.join(args.output, case, f"variant{variant_idx:03d}")
			current_variant_dir = variant_dir
			done_path = os.path.join(variant_dir, str(args.done_marker))
			current_done_path = done_path
			if os.path.exists(done_path) and (not args.overwrite):
				print(f"[Skip] Variant done: {done_path}")
				continue

			try:
				ct = ct_cpu.to(device, non_blocking=True)
				lungs = lungs_cpu.to(device, non_blocking=True)
				# Create a devices-augmented CT per (case, variant)
				variant_seed = seed + 1000003 * (ct_idx + 1) + 9176 * (variant_idx + 1)
				rng = random.Random(variant_seed)
				dev_ct, devices_meta = _add_external_devices_to_single_ct(ct.clone(), lungs, devices_cfg, rng=rng)
			except KeyboardInterrupt:
				# Graceful shutdown on Ctrl+C/SIGINT.
				_append_error_record(
					errors_out_dir,
					errors_log_name,
					{
						'timestamp': time.time(),
						'phase': 'interrupt',
						'case': current_case,
						'variant_idx': current_variant_idx,
						'pair_idx': current_pair_idx,
						'pair_dir': current_pair_dir,
						'signal': shutdown_requested.get('signal'),
						'error_type': 'KeyboardInterrupt',
						'error': 'Interrupted by user',
					},
				)
				print('[INFO] Interrupted. Exiting gracefully.')
				return
			except Exception as e:
				errors += 1
				print(f"[ERR] Failed devices injection case={case} variant={variant_idx}: {type(e).__name__}: {e}")
				_append_error_record(
					errors_out_dir,
					errors_log_name,
					{
						'timestamp': time.time(),
						'phase': 'inject_devices',
						'case': case,
						'variant_idx': variant_idx,
						'ct_path': ct_path,
						'lungs_seg_path': lungs_seg_path,
						'variant_seed': variant_seed,
						'error_type': type(e).__name__,
						'error': str(e),
						'traceback': traceback.format_exc(),
					},
				)
				if errors >= max_errors:
					raise RuntimeError(f"Too many errors ({errors}); aborting.")
				# Best-effort cleanup
				try:
					del ct
				except Exception:
					pass
				try:
					del lungs
				except Exception:
					pass
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
				continue

			# Pre-sample angles per variant
			angles: List[Tuple[float, float, float]] = []
			for _ in range(angles_per_variant):
				a1, a2, a3 = get_random_rotation_angles(rot_ranges, max_sum, min_sum, rot_exp)
				angles.append((float(a1), float(a2), float(a3)))

			variant_ok = 0

			for i, (a1, a2, a3) in enumerate(angles):
				current_pair_idx = i
				pair_dir = os.path.join(
					args.output,
					case,
					f"variant{variant_idx:03d}",
					f"pair{i:05d}_ax{_sanitize_angle_tag(a1)}_ay{_sanitize_angle_tag(a2)}_az{_sanitize_angle_tag(a3)}",
				)
				current_pair_dir = pair_dir

				params_path = os.path.join(pair_dir, 'params.json')
				if os.path.exists(params_path) and (not args.overwrite):
					print(f"[Skip] Exists: {pair_dir}")
					variant_ok += 1
					continue

				os.makedirs(pair_dir, exist_ok=True)

				try:
					# Rotate+crop both volumes together to ensure identical crop/framing.
					with torch.inference_mode():
						ct_cat = torch.stack([ct, dev_ct], dim=0)
						rotated_ct_cat, _ = rotate_ct_and_crop_according_to_seg(
							ct_cat,
							lungs,
							rotate_angle1=a1,
							rotate_angle2=a2,
							rotate_angle3=a3,
							return_ct_seg=True,
							device=device,
						)

					clean_rot = rotated_ct_cat[0]
					dev_rot = rotated_ct_cat[1]

					with torch.inference_mode():
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
					# Free GPU tensors for this pair ASAP
					del ct_cat, rotated_ct_cat, clean_rot, dev_rot
					variant_ok += 1
				except KeyboardInterrupt:
					_append_error_record(
						errors_out_dir,
						errors_log_name,
						{
							'timestamp': time.time(),
							'phase': 'interrupt',
							'case': current_case,
							'variant_idx': current_variant_idx,
							'pair_idx': current_pair_idx,
							'pair_dir': current_pair_dir,
							'signal': shutdown_requested.get('signal'),
							'error_type': 'KeyboardInterrupt',
							'error': 'Interrupted by user',
						},
					)
					# If the current variant already finished (or was fully present), write done marker.
					try:
						if (
							current_variant_dir
							and current_done_path
							and (not bool(args.overwrite))
							and (
								variant_ok == angles_per_variant
								or _variant_complete_on_disk(current_variant_dir, angles_per_variant)
							)
						):
							_write_done_marker(
								current_done_path,
								{
									'case': current_case,
									'variant_idx': current_variant_idx,
									'angles_per_variant': int(angles_per_variant),
									'timestamp': time.time(),
									'interrupted_after_completion': True,
								},
							)
							print(f"[OK] Wrote done marker: {current_done_path}")
					except Exception:
						pass
					print('[INFO] Interrupted. Exiting gracefully.')
					return
				except RuntimeError as e:
					# Attempt to recover from CUDA OOM by freeing cache and skipping.
					msg = str(e).lower()
					if 'out of memory' in msg and torch.cuda.is_available():
						try:
							gc.collect()
							torch.cuda.empty_cache()
							torch.cuda.ipc_collect()
						except Exception:
							pass
					errors += 1
					print(
						f"[ERR] Failed pair case={case} variant={variant_idx} i={i} "
						f"angles=({a1:.2f},{a2:.2f},{a3:.2f}): {type(e).__name__}: {e}"
					)
					_append_error_record(
						errors_out_dir,
						errors_log_name,
						{
							'timestamp': time.time(),
							'phase': 'generate_pair',
							'case': case,
							'variant_idx': variant_idx,
							'pair_idx': i,
							'pair_dir': pair_dir,
							'rotation_angles_deg': (a1, a2, a3),
							'variant_seed': variant_seed,
							'error_type': type(e).__name__,
							'error': str(e),
							'traceback': traceback.format_exc(),
						},
					)
					if errors >= max_errors:
						raise RuntimeError(f"Too many errors ({errors}); aborting.")
					# Aggressive cleanup after failure
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						torch.cuda.ipc_collect()
					continue
				except Exception as e:
					errors += 1
					print(
						f"[ERR] Failed pair case={case} variant={variant_idx} i={i} "
						f"angles=({a1:.2f},{a2:.2f},{a3:.2f}): {type(e).__name__}: {e}"
					)
					_append_error_record(
						errors_out_dir,
						errors_log_name,
						{
							'timestamp': time.time(),
							'phase': 'generate_pair',
							'case': case,
							'variant_idx': variant_idx,
							'pair_idx': i,
							'pair_dir': pair_dir,
							'rotation_angles_deg': (a1, a2, a3),
							'variant_seed': variant_seed,
							'error_type': type(e).__name__,
							'error': str(e),
							'traceback': traceback.format_exc(),
						},
					)
					if errors >= max_errors:
						raise RuntimeError(f"Too many errors ({errors}); aborting.")
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						torch.cuda.ipc_collect()
					continue

			# Mark variant done if all pairs are present (created or skipped)
			if (variant_ok == angles_per_variant) and (not args.overwrite):
				payload = {
					'case': case,
					'variant_idx': variant_idx,
					'angles_per_variant': angles_per_variant,
					'variant_seed': seed + 1000003 * (ct_idx + 1) + 9176 * (variant_idx + 1),
					'timestamp': time.time(),
				}
				try:
					_write_done_marker(done_path, payload)
					print(f"[OK] Wrote done marker: {done_path}")
				except KeyboardInterrupt:
					# If Ctrl+C hits exactly while writing the marker, finish the write anyway.
					try:
						_write_done_marker(done_path, payload)
						print(f"[OK] Wrote done marker: {done_path}")
					except Exception:
						pass
					_append_error_record(
						errors_out_dir,
						errors_log_name,
						{
							'timestamp': time.time(),
							'phase': 'interrupt',
							'case': current_case,
							'variant_idx': current_variant_idx,
							'pair_idx': current_pair_idx,
							'pair_dir': current_pair_dir,
							'signal': shutdown_requested.get('signal'),
							'error_type': 'KeyboardInterrupt',
							'error': 'Interrupted by user',
						},
					)
					print('[INFO] Interrupted. Exiting gracefully.')
					return

			# Free per-variant tensors
			del dev_ct, ct, lungs
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()

		# Free per-case tensors aggressively
		del ct_cpu, lungs_cpu
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()

	print(f"\nDone. Generated {pairs_done} paired samples.")


if __name__ == '__main__':
	main()
