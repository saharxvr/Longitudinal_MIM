"""CT → DRR synthetic pair generator (refactored copy).

This module is a cleaned, documented copy of `CT_entities/DRR_generator.py`.

Goals of this refactor:
- Keep the original generator intact.
- Remove one-off / experimental helper functions ("temp_*", ad-hoc tests).
- Make the pipeline easy to follow by clearly marking the three key logic blocks:
  1) Where entities are inserted into the CT volume.
  2) Where rotation angles are sampled/applied (pose changes).
  3) Where DRRs are produced (projection) and saved.

High-level pipeline (per CT case, per pair):
1. Load CT + organ segmentations.
2. Crop around lungs (speeds up entity insertion) and smooth masks.
3. ENTITY INSERTION: probabilistically modify the cropped prior/current CTs.
4. Restore to original CT size.
5. ANGLE / ROTATION: rotate+crop a (registered_prior, current) stack and a prior-only CT.
6. DRR GENERATION: project rotated CTs to 2D DRRs.
7. Build a difference map, including extra maps for some extra-pulmonary entities.
8. Save `prior.nii.gz`, `current.nii.gz`, `diff_map.nii.gz` and PNG previews.

Note: the heavy-lifting for volumetric rotation/cropping is in
`CT_entities/CT_Rotations.py::random_rotate_ct_and_crop_according_to_seg`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import psutil
import torch
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import label
from skimage.morphology import ball
from torchvision.transforms.v2.functional import adjust_sharpness

from CT_Rotations import random_rotate_ct_and_crop_according_to_seg
from DRR_utils import add_back_cropped, crop_according_to_seg

from constants import DEVICE

from Cardiomegaly import Cardiomegaly
from Consolidation import Consolidation
from Pleural_Effusion import PleuralEffusion
from Pneumothorax import Pneumothorax
from Fluid_Overload import FluidOverload
from External_Devices import ExternalDevices


# -----------------------------------------------------------------------------
# Globals (small constants)
# -----------------------------------------------------------------------------

resize = v2.Resize((512, 512))

_CC_STRUCT = np.ones((3, 3), dtype=np.int8)  # for connected-component filtering

differential_grad = colors.LinearSegmentedColormap.from_list(
	'my_gradient',
	(
		(0.000, (0.235, 1.000, 0.239)),
		(0.400, (0.000, 1.000, 0.702)),
		(0.500, (1.000, 0.988, 0.988)),
		(0.600, (1.000, 0.604, 0.000)),
		(1.000, (0.682, 0.000, 0.000)),
	),
)


# -----------------------------------------------------------------------------
# Memory management utilities
# -----------------------------------------------------------------------------

def get_memory_usage_gb() -> float:
	"""Return current process RSS memory usage in GB."""
	process = psutil.Process()
	return process.memory_info().rss / (1024 ** 3)


def log_memory(label: str = "") -> float:
	"""Print and return current process memory usage in GB."""
	mem_gb = get_memory_usage_gb()
	print(f"[Memory] {label}: {mem_gb:.2f} GB")
	return mem_gb


def cleanup_memory() -> None:
	"""Force garbage collection and clear CUDA cache."""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def check_memory_and_cleanup(threshold_gb: float, label: str = "") -> bool:
	"""If RSS exceeds threshold, run aggressive cleanup and return True."""
	mem_gb = get_memory_usage_gb()
	if mem_gb <= threshold_gb:
		return False

	print(f"[Memory Warning] {label}: {mem_gb:.2f} GB > {threshold_gb} GB")
	print("[Memory] Performing aggressive cleanup...")

	for _ in range(5):
		gc.collect()

	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

	new_mem_gb = get_memory_usage_gb()
	print(f"[Memory] After cleanup: {new_mem_gb:.2f} GB (freed {mem_gb - new_mem_gb:.2f} GB)")
	return True


# -----------------------------------------------------------------------------
# IO + preprocessing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for synthetic pair generation."""
	parser = argparse.ArgumentParser(
		description=(
			"Generate pairs of synthetic DRRs from CT scans with inserted 3D entities "
			"and applied random 3D rotations."
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)

	parser.add_argument('-n', '--number_pairs', type=int, required=True, help='Total number of pairs to create')
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
		help='Space-separated list of CT directories',
	)
	parser.add_argument(
		'-o',
		'--output',
		type=str,
		default='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/train',
		help='Output directory',
	)

	parser.add_argument('-CO', '--Consolidation', type=float, default=0.0)
	parser.add_argument('-PL', '--PleuralEffusion', type=float, default=0.0)
	parser.add_argument('-PN', '--Pneumothorax', type=float, default=0.0)
	parser.add_argument('-FL', '--FluidOverload', type=float, default=0.0)
	parser.add_argument('-CA', '--Cardiomegaly', type=float, default=0.0)
	parser.add_argument('-EX', '--ExternalDevices', type=float, default=0.0)

	parser.add_argument(
		'--default_entities',
		action='store_true',
		help='Use baseline probabilities used in training data generation.',
	)

	parser.add_argument(
		'-d',
		'--decay_prob_on_add',
		type=float,
		default=1.0,
		help='Exponential probability decay applied after each added entity (not for devices).',
	)

	parser.add_argument(
		'-r',
		'--rotation_params',
		nargs='+',
		type=float,
		default=[17.5, 37.5, 0.0, 1.75],
		help=(
			"Four floats: [max_abs_per_axis_deg, max_sum_deg, min_sum_deg, exponent].\n"
			"Higher exponent biases angles towards 0."
		),
	)

	parser.add_argument(
		'-s',
		'--slices_for_CTs_list',
		nargs='+',
		type=float,
		default=[0.0, 1.0],
		help='Two floats [a,b] in [0,1] to process ct_paths[int(a*N):int(b*N+1)].',
	)

	parser.add_argument(
		'-m',
		'--memory_threshold',
		type=float,
		default=15.0,
		help='RSS memory threshold in GB that triggers aggressive cleanup.',
	)

	return parser.parse_args()


def load_scan_and_seg(file_path: str, segs_paths_dict: Dict[str, str]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
	"""Load a CT scan and its segmentations (nii.gz) as torch tensors.

	Returns tensors in (D,H,W) with a flip on depth to match the original script.
	Uses `np.asarray(nifti.dataobj)` to reduce peak memory vs `get_fdata()`.
	"""
	scan_nif = nib.load(file_path)

	global affine
	affine = scan_nif.affine

	scan_data = np.asarray(scan_nif.dataobj, dtype=np.float32)
	scan_data = torch.from_numpy(np.transpose(scan_data, (2, 0, 1)).copy())
	scan_nif.uncache()
	del scan_nif

	scan_data = torch.flip(scan_data, dims=[0])

	segs_dict: Dict[str, torch.Tensor] = {}
	for organ_name, seg_path in segs_paths_dict.items():
		seg_nif = nib.load(seg_path)
		seg_data = np.asarray(seg_nif.dataobj, dtype=np.float32)
		seg_data = torch.from_numpy(np.transpose(seg_data, (2, 0, 1)).copy())
		seg_nif.uncache()
		del seg_nif
		seg_data = torch.flip(seg_data, dims=[0])
		segs_dict[organ_name] = seg_data

	return scan_data, segs_dict


def save_arr_as_nifti(arr: np.ndarray | torch.Tensor, output_path: str) -> None:
	"""Save an array as NIfTI using the globally-cached affine from `load_scan_and_seg`."""
	if isinstance(arr, torch.Tensor):
		arr = np.array(arr.float().cpu())

	if arr.ndim == 3:
		arr = np.flip(arr, axis=0)
		arr = np.transpose(arr, (1, 2, 0))

	xray_nif = nib.Nifti1Image(arr, affine)
	nib.save(xray_nif, output_path)


# -----------------------------------------------------------------------------
# DRR generation (projection + post-processing)
# -----------------------------------------------------------------------------

def project_ct(rotated_ct_scan: torch.Tensor, dim: int = 2, *, is_seg: bool = False) -> np.ndarray:
	"""Project a (rotated) CT volume to a 2D DRR-like image.

	**THIS IS WHERE THE DRR IS TAKEN**

	- For intensities: clamp at -1000 HU, shift to be non-negative, sum along `dim`.
	- For segmentations: max-projection along `dim`.
	- Always resized to 512x512.

	Returns a numpy float array.
	"""
	if is_seg:
		xray_image = torch.amax(rotated_ct_scan, dim=dim)
	else:
		rotated_ct_scan = torch.clamp_min(rotated_ct_scan, -1000)
		m = torch.min(rotated_ct_scan)
		rotated_ct_scan = rotated_ct_scan - m
		xray_image = torch.sum(rotated_ct_scan, dim=dim)

	xray_image = resize(xray_image[None, ...]).squeeze()
	return xray_image.cpu().numpy()


def normalize_xray(xray_image: np.ndarray) -> np.ndarray:
	"""Contrast-normalize a projected DRR for visualization / learning."""
	xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image) + 1e-8)
	xray_image = -np.exp(-xray_image)
	xray_image = (xray_image - np.min(xray_image)) / (np.max(xray_image) - np.min(xray_image) + 1e-8)
	xray_image = adjust_sharpness(torch.tensor(xray_image).unsqueeze(0), sharpness_factor=7.0).numpy().squeeze()
	return np.clip(xray_image, a_min=0.0, a_max=1.0)


# -----------------------------------------------------------------------------
# Masks / diff maps
# -----------------------------------------------------------------------------

def remove_small_ccs(im: torch.Tensor, min_count: int = 16) -> Tuple[torch.Tensor, torch.Tensor, int]:
	"""Remove connected components smaller than `min_count` from a binary 3D mask."""
	ccs, _ = label(im.cpu().squeeze().numpy(), _CC_STRUCT)
	ccs = torch.from_numpy(ccs).to(DEVICE)

	unique_vals, unique_counts = torch.unique(ccs, return_counts=True)
	unique_vals = unique_vals[unique_counts > min_count]

	ccs = ccs[None, None, ...]
	ccs_indic = torch.isin(ccs, unique_vals)

	new_im = im * ccs_indic
	new_ccs = ccs * ccs_indic
	ccs_num = int(torch.numel(unique_vals) - 1)

	return new_im, new_ccs, ccs_num


def smooth_segmentation(mask: torch.Tensor, radius: int = 7) -> torch.Tensor:
	"""Morphologically smooth a 3D binary mask using (dilate→erode) with a ball struct."""

	def binary_dilation(to_dilate: torch.Tensor, rad: int) -> torch.Tensor:
		d_struct = torch.tensor(ball(rad))[None, None, ...].float().to(to_dilate.device)
		dilated = torch.nn.functional.conv3d(
			to_dilate.squeeze()[None, None, ...], weight=d_struct, padding='same'
		).squeeze()
		dilated[dilated > 0] = 1
		return dilated

	def binary_erosion(to_erode: torch.Tensor, rad: int) -> torch.Tensor:
		return 1.0 - binary_dilation(1.0 - to_erode, rad)

	closed = binary_dilation(mask, radius)
	closed = binary_erosion(closed, radius)
	return closed


def calculate_diff_map(
	current_scan: np.ndarray,
	registrated_prior_scan: np.ndarray,
	boundary_seg: Optional[np.ndarray] = None,
) -> np.ndarray:
	"""Compute a signed difference map (current - registered_prior) with filtering."""
	if boundary_seg is not None:
		current_scan = current_scan * boundary_seg
		registrated_prior_scan = registrated_prior_scan * boundary_seg

	raw_diff = current_scan - registrated_prior_scan
	gte_diff = raw_diff >= 0
	lte_diff = raw_diff <= 0

	current_n = (current_scan - np.min(current_scan)) / (np.max(current_scan) - np.min(current_scan) + 1e-8)
	prior_n = (registrated_prior_scan - np.min(registrated_prior_scan)) / (np.max(registrated_prior_scan) - np.min(registrated_prior_scan) + 1e-8)
	c_diff_map = current_n - prior_n

	c_diff_map[np.logical_and(c_diff_map > 0, lte_diff)] = 0
	c_diff_map[np.logical_and(c_diff_map < 0, gte_diff)] = 0

	c_diff_map[np.abs(raw_diff) < 0.015] = 0.0
	c_diff_map[np.abs(c_diff_map) < 0.015] = 0.0

	diff_map_th = c_diff_map != 0.0
	diff_map_th, _, __ = remove_small_ccs(torch.tensor(diff_map_th).to(DEVICE), min_count=50)
	diff_map_th = diff_map_th.cpu().squeeze().numpy()
	c_diff_map[diff_map_th == 0.0] = 0.0

	return c_diff_map


def create_seg_diff_map_from_rotated(
	c_rotated_seg: torch.Tensor,
	pos_num: int,
	neg_num: int,
	max_diff_val: float,
	max_seg_depth: torch.Tensor,
	avg_k: Optional[int],
) -> np.ndarray:
	"""Create a 2D diff map from a rotated multi-label 3D seg (effusion/pnx/cardiomegaly)."""
	seg_for_diff = c_rotated_seg.clone()
	seg_for_diff[~torch.isin(seg_for_diff, torch.tensor([pos_num, neg_num], device=seg_for_diff.device))] = 0
	seg_for_diff[seg_for_diff == pos_num] = 1
	seg_for_diff[seg_for_diff == neg_num] = -1

	diff_map_for_seg = torch.nan_to_num(torch.sum(seg_for_diff, dim=2) / max_seg_depth, nan=0.0).cpu().squeeze()
	diff_map_for_seg = resize(diff_map_for_seg[None, ...])
	if avg_k is not None:
		diff_map_for_seg = torch.nn.functional.avg_pool2d(
			diff_map_for_seg, kernel_size=avg_k, stride=1, padding=avg_k // 2
		).squeeze()
	diff_map_for_seg = diff_map_for_seg * max_diff_val

	return diff_map_for_seg.numpy()


def generate_alpha_map(x: torch.Tensor) -> torch.Tensor:
	x_abs = x.abs()
	max_val = max(torch.max(x_abs).item(), 0.07)
	return x_abs / max_val


def plot_diff_on_current(c_diff_map: np.ndarray, c_current: np.ndarray, out_p: str) -> None:
	"""Overlay difference map on current DRR and save; closes figures to avoid leaks."""
	c_diff_map_t = torch.tensor(c_diff_map)
	c_current_t = torch.tensor(c_current)

	alphas = generate_alpha_map(c_diff_map_t)
	divnorm = colors.TwoSlopeNorm(
		vmin=min(torch.min(c_diff_map_t).item(), -0.01),
		vcenter=0.0,
		vmax=max(torch.max(c_diff_map_t).item(), 0.01),
	)

	fig, ax = plt.subplots()
	ax.imshow(c_current_t.squeeze().cpu(), cmap='gray')
	imm1 = ax.imshow(c_diff_map_t.squeeze().cpu(), alpha=alphas, cmap=differential_grad, norm=divnorm)
	plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
	ax.set_axis_off()
	fig.tight_layout()
	plt.savefig(out_p)

	plt.close(fig)
	plt.close('all')


def log_params(params: Dict[str, Any], out_p: str) -> None:
	with open(out_p, 'w') as f:
		json.dump(params, f, indent=4)


# -----------------------------------------------------------------------------
# ENTITY INSERTION (3D)
# -----------------------------------------------------------------------------

def add_entities_to_pair(
	scans: List[torch.Tensor],
	segs: List[Dict[str, torch.Tensor]],
	orig_scan: torch.Tensor,
	orig_lungs: torch.Tensor,
	intra_pulmonary_entities: Sequence[Tuple[type, float]],
	extra_pulmonary_entities: Sequence[Tuple[type, float]],
	devices_entity: Sequence[Tuple[type, float]],
	entity_prob_decay: float,
) -> Dict[str, Any]:
	"""Apply entity insertion to (prior, current) CT volumes.

	**THIS IS WHERE ENTITIES ARE ADDED**
	"""
	patient_mode = random.choices(['supine', 'erect'], weights=[0.8, 0.2], k=1)[0]
	pleural_effusion_patient_mode = 'erect' if patient_mode == 'erect' else ('supine' if random.random() < 0.5 else 'semi-supine')

	ctx: Dict[str, Any] = {
		'scans': scans,
		'segs': segs,
		'orig_scan': orig_scan,
		'orig_lungs': orig_lungs,
		'patient_mode': patient_mode,
		'pleural_effusion_patient_mode': pleural_effusion_patient_mode,
		'added_entity_names': [],
		'log_params': True,
	}

	prob_mult = 1.0

	for entity, prob in intra_pulmonary_entities:
		if random.random() < prob * prob_mult:
			ctx.update(entity.add_to_CT_pair(**ctx))
			ctx['added_entity_names'].append(entity.__name__)
			prob_mult *= entity_prob_decay
			cleanup_memory()

	ctx['registrated_prior'] = ctx['scans'][0].clone()

	for entity, prob in extra_pulmonary_entities:
		if random.random() < prob * prob_mult:
			ctx.update(entity.add_to_CT_pair(**ctx))
			ctx['added_entity_names'].append(entity.__name__)
			prob_mult *= entity_prob_decay
			cleanup_memory()

	for entity, prob in devices_entity:
		if random.random() < prob:
			ctx.update(entity.add_to_CT_pair(**ctx))
			ctx['added_entity_names'].append(entity.__name__)
			cleanup_memory()

	return ctx


# -----------------------------------------------------------------------------
# ANGLES / ROTATION + DRR projection helper
# -----------------------------------------------------------------------------

def rotate_pair_and_project(
	registrated_prior: torch.Tensor,
	current: torch.Tensor,
	seg_for_rotation: torch.Tensor,
	*,
	rot_ranges: Tuple[float, float, float],
	max_sum: float,
	min_sum: float,
	rot_exp: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, Tuple[float, float, float]]:
	"""Rotate+crop the (registered_prior, current) stack and project both to DRRs.

	**THIS IS WHERE ANGLES ARE SAMPLED / APPLIED** (for the pair)
	"""
	ct_cat = torch.stack([registrated_prior, current], dim=0)

	rotated_ct_cat, rotated_seg, angles = random_rotate_ct_and_crop_according_to_seg(
		ct_cat,
		seg_for_rotation,
		return_ct_seg=True,
		rot_ranges=rot_ranges,
		max_angles_sum=max_sum,
		min_angles_sum=min_sum,
		exponent=rot_exp,
		return_angles=True,
	)

	current_ct = rotated_ct_cat[1]
	registrated_prior_ct = rotated_ct_cat[0]

	current_drr = project_ct(current_ct)
	registrated_prior_drr = project_ct(registrated_prior_ct)

	rotated_seg = torch.round(rotated_seg).squeeze()
	rotated_boundary = (rotated_seg != 0)
	projected_boundary = project_ct(rotated_boundary, is_seg=True)

	a1, a2, a3 = angles
	return current_drr, registrated_prior_drr, projected_boundary, rotated_seg, (float(a1), float(a2), float(a3))


def rotate_prior_and_project(
	prior: torch.Tensor,
	lungs_seg: torch.Tensor,
	*,
	rot_ranges: Tuple[float, float, float],
	max_sum: float,
	min_sum: float,
	rot_exp: float,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
	"""Rotate+crop the prior CT alone and project to a DRR."""
	rotated_prior, _, angles = random_rotate_ct_and_crop_according_to_seg(
		prior,
		lungs_seg,
		return_ct_seg=True,
		rot_ranges=rot_ranges,
		max_angles_sum=max_sum,
		min_angles_sum=min_sum,
		exponent=rot_exp,
		return_angles=True,
	)
	rotated_prior = rotated_prior.squeeze()
	prior_drr = project_ct(rotated_prior)

	a1, a2, a3 = angles
	return prior_drr, (float(a1), float(a2), float(a3))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
	args = parse_args()

	total_pairs_num = int(args.number_pairs)

	intra_names = {'Consolidation', 'FluidOverload'}
	extra_names = {'PleuralEffusion', 'Pneumothorax', 'Cardiomegaly'}
	device_names = {'ExternalDevices'}

	entity_name_to_class = {
		'Consolidation': Consolidation,
		'FluidOverload': FluidOverload,
		'PleuralEffusion': PleuralEffusion,
		'Pneumothorax': Pneumothorax,
		'Cardiomegaly': Cardiomegaly,
		'ExternalDevices': ExternalDevices,
	}

	if args.default_entities:
		probs = {
			'Consolidation': 0.15,
			'FluidOverload': 0.15,
			'PleuralEffusion': 0.15,
			'Pneumothorax': 0.15,
			'Cardiomegaly': 0.15,
			'ExternalDevices': 0.5,
		}
	else:
		probs = {
			'Consolidation': float(args.Consolidation),
			'FluidOverload': float(args.FluidOverload),
			'PleuralEffusion': float(args.PleuralEffusion),
			'Pneumothorax': float(args.Pneumothorax),
			'Cardiomegaly': float(args.Cardiomegaly),
			'ExternalDevices': float(args.ExternalDevices),
		}

	entity_prob_decay_on_addition = float(args.decay_prob_on_add)

	cur_intra: List[Tuple[type, float]] = []
	cur_extra: List[Tuple[type, float]] = []
	cur_devices: List[Tuple[type, float]] = []

	for name, prob in probs.items():
		if name in intra_names:
			cur_intra.append((entity_name_to_class[name], prob))
		elif name in extra_names:
			cur_extra.append((entity_name_to_class[name], prob))
		elif name in device_names:
			cur_devices.append((entity_name_to_class[name], prob))
		else:
			raise ValueError(f"Unknown entity name: {name}")

	rotation_params = list(args.rotation_params)
	if len(rotation_params) != 4:
		raise ValueError(f"Expected 4 rotation params, got {rotation_params}")

	rotation_range = float(rotation_params[0])
	rot_ranges = (rotation_range, rotation_range, rotation_range)
	max_sum = float(rotation_params[1])
	min_sum = float(rotation_params[2])
	rot_exp = float(rotation_params[3])

	out_base_dir = args.output

	ct_paths: List[str] = []
	for ct_dir in args.input:
		ct_paths.extend([ct_dir + f'/{n}' for n in os.listdir(ct_dir)])
	ct_paths = sorted(ct_paths)

	num_paths = len(ct_paths)
	pairs_per_ct = math.ceil(total_pairs_num / max(num_paths, 1))

	slices = list(args.slices_for_CTs_list)
	if len(slices) != 2:
		raise ValueError(f"Expected 2 slicing values, got {slices}")

	first = int(float(slices[0]) * num_paths)
	last = int(float(slices[1]) * num_paths + 1)

	print(f"Input CT dirs: {args.input}\nOutput dir: {args.output}")
	print(f"Total pairs: {total_pairs_num} | CTs: {num_paths} | pairs/CT: {pairs_per_ct}")
	print(f"Entity probabilities: {probs} | decay on add: {entity_prob_decay_on_addition}")
	print(f"Rotation: ranges={rot_ranges} max_sum={max_sum} min_sum={min_sum} exp={rot_exp}")
	print(f"CT list slicing: {slices}")

	MEMORY_THRESHOLD_GB = float(args.memory_threshold)
	print(f"Memory threshold: {MEMORY_THRESHOLD_GB} GB")

	ps = ct_paths[first:last]
	pairs_created = 0

	log_memory("Before starting")

	for k, ct_p in enumerate(ps):
		case_name = ct_p.split('/')[-1][:-7]
		print(f"\n{'='*60}")
		print(f"Working on case {k}/{len(ps)}: {case_name}")
		log_memory(f"Start case {k}")

		out_path = f'{out_base_dir}/{case_name}'
		if os.path.exists(f'{out_path}/pair{pairs_per_ct - 1}/params.json'):
			print('Case completed. Skipping')
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

			scan, segs_dict = load_scan_and_seg(scan_p, segs_p_dict)
			orig_scan = scan.clone()

			check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After loading {case_name}")

			segs_dict['lungs'] = torch.clamp_max(
				segs_dict['middle_right_lobe']
				+ segs_dict['upper_right_lobe']
				+ segs_dict['lower_right_lobe']
				+ segs_dict['upper_left_lobe']
				+ segs_dict['lower_left_lobe'],
				1.0,
			)

			if segs_dict['lungs'].sum() < 20:
				print('Empty lungs seg. Skipping.')
				continue

			lung_seg_coords = segs_dict['lungs'].nonzero().T
			lungs_h = torch.max(lung_seg_coords[0]) - torch.min(lung_seg_coords[0])
			h_low_ext = int((lungs_h // 5).item())

			cropped_scan, cropped_segs_dict, cropping_slices = crop_according_to_seg(
				scan,
				segs_dict['lungs'],
				all_segs_dict=segs_dict,
				tight_y=False,
				ext=15,
				h_low_ext=h_low_ext,
			)

			for organ_name, c_seg in list(cropped_segs_dict.items()):
				c_seg = smooth_segmentation(c_seg.to(DEVICE), radius=int(smoothing_radius_dict[organ_name])).cpu()
				cropped_segs_dict[organ_name] = c_seg
				if torch.sum(c_seg) < 20:
					print(f'Empty {organ_name}. Skipping case.')
					raise RuntimeError(f'Bad segmentation: {organ_name}')

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

			prior_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}
			current_cropped_segs_dict = {k: v.clone() for k, v in cropped_segs_dict.items()}

			for i in range(pairs_per_ct):
				print(f'Working on pair {i}')

				c_out_dir = f'{out_path}/pair{i}'
				os.makedirs(c_out_dir, exist_ok=True)

				if os.path.exists(f'{c_out_dir}/params.json'):
					print('Pair exists. Skipping.')
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

				# ENTITY INSERTION
				ret = add_entities_to_pair(
					scans,
					segs,
					orig_cropped_scan,
					orig_cropped_lungs,
					cur_intra,
					cur_extra,
					cur_devices,
					entity_prob_decay_on_addition,
				)

				prior, current = ret['scans']
				registrated_prior = ret['registrated_prior']
				added_entity_names = ret['added_entity_names']

				params: Dict[str, Any] = {'added_entities': added_entity_names}

				prior = add_back_cropped(prior, orig_scan, cropping_slices)
				current = add_back_cropped(current, orig_scan, cropping_slices)
				registrated_prior = add_back_cropped(registrated_prior, orig_scan, cropping_slices)

				seg_for_rotation = segs_dict['lungs'].clone()

				if 'PleuralEffusion' in added_entity_names:
					pe_seg = add_back_cropped(ret['pleural_effusion_seg'], torch.zeros_like(orig_scan), cropping_slices)
					seg_for_rotation[pe_seg == 1] = 2
					seg_for_rotation[pe_seg == -1] = 3

				if 'Pneumothorax' in added_entity_names:
					pnx = add_back_cropped(ret['pneumothorax_seg'], torch.zeros_like(orig_scan), cropping_slices)
					pnx_s = add_back_cropped(ret['pneumothorax_sulcus_seg'], torch.zeros_like(orig_scan), cropping_slices)
					seg_for_rotation[pnx == 1] = 4
					seg_for_rotation[pnx == -1] = 5
					seg_for_rotation[pnx_s == 1] = 6
					seg_for_rotation[pnx_s == -1] = 7

				cardiomegaly_progress = None
				if 'Cardiomegaly' in added_entity_names:
					cardiomegaly_progress = ret['cardiomegaly_progress']
					cm_seg = add_back_cropped(ret['cardiomegaly_seg'], torch.zeros_like(orig_scan), cropping_slices)
					seg_for_rotation[cm_seg == 1] = 8
					seg_for_rotation[cm_seg == -1] = 9

				# ANGLES / ROTATION + DRR GENERATION
				current_drr, reg_prior_drr, projected_boundary, rotated_seg, angles_pair = rotate_pair_and_project(
					registrated_prior,
					current,
					seg_for_rotation,
					rot_ranges=rot_ranges,
					max_sum=max_sum,
					min_sum=min_sum,
					rot_exp=rot_exp,
				)
				params['rotation_angles_pair_deg'] = angles_pair

				prior_drr, angles_prior = rotate_prior_and_project(
					prior,
					segs_dict['lungs'],
					rot_ranges=rot_ranges,
					max_sum=max_sum,
					min_sum=min_sum,
					rot_exp=rot_exp,
				)
				params['rotation_angles_prior_deg'] = angles_prior

				diff_map = calculate_diff_map(current_drr, reg_prior_drr, projected_boundary)

				# Extra maps: derived from the *same* rotated multi-label seg used during projection
				rotated_boundary = rotated_seg != 0
				max_depth = torch.max(torch.sum(rotated_boundary, dim=2))

				if 'PleuralEffusion' in added_entity_names:
					diff_map += create_seg_diff_map_from_rotated(rotated_seg, 2, 3, 0.35, max_depth, 9)

				if 'Pneumothorax' in added_entity_names:
					diff_map += create_seg_diff_map_from_rotated(rotated_seg, 4, 5, 0.25, max_depth, 9)
					diff_map += create_seg_diff_map_from_rotated(rotated_seg, 6, 7, 0.25, max_depth, 9)

				if 'Cardiomegaly' in added_entity_names:
					if cardiomegaly_progress in {1, -1}:
						cm = create_seg_diff_map_from_rotated(rotated_seg, 8, 9, 0.35, max_depth, 9)
						if cardiomegaly_progress == 1:
							cm = np.clip(cm, a_min=0, a_max=None)
						else:
							cm = np.clip(cm, a_min=None, a_max=0)
					else:
						cm = np.zeros_like(current_drr)
					diff_map += cm

				save_arr_as_nifti(prior_drr.T, f'{c_out_dir}/prior.nii.gz')
				save_arr_as_nifti(current_drr.T, f'{c_out_dir}/current.nii.gz')
				save_arr_as_nifti(diff_map.T, f'{c_out_dir}/diff_map.nii.gz')

				plt.imsave(f'{c_out_dir}/prior.png', prior_drr, cmap='gray')
				plt.imsave(f'{c_out_dir}/current.png', current_drr, cmap='gray')
				plot_diff_on_current(diff_map, current_drr, f'{c_out_dir}/current_with_differences.png')
				log_params(params, f'{c_out_dir}/params.json')

				pairs_created += 1
				cleanup_memory()
				check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After pair {i}")

				if pairs_created >= total_pairs_num:
					print('Done generating synthetic pairs. Exiting.')
					return

		except MemoryError as e:
			print(f"[ERROR] MemoryError on case {case_name}: {e}")
			print('[ERROR] Skipping this case after cleanup...')
			for _ in range(5):
				gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		except Exception as e:
			print(f"[ERROR] Exception on case {case_name}: {e}")
			import traceback

			traceback.print_exc()
			print('[ERROR] Skipping to next case...')

		finally:
			print(f"[Cleanup] Finished case {case_name}")
			for _ in range(3):
				gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
			check_memory_and_cleanup(MEMORY_THRESHOLD_GB, f"After case {case_name}")
			log_memory(f"End case {k}")


if __name__ == '__main__':
	main()