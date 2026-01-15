"""Triplet DRR pipeline.

Produces DRR triplets:
1) DRR of the clean CT ("original/current state")
2) DRR of the same CT at a different (random) angle
3) DRR at the same different angle, with a sampled entity/pathology injected with a given probability

This is designed for training deformation/decoder models where views 2 and 3 share geometry.

Notes:
- Uses simple line-integral style projection (sum along an axis), consistent with `CT_entities/DRR_generator.py`.
- Uses existing 3D entity injection code (e.g., Consolidation/PleuralEffusion/etc.).
"""

from __future__ import annotations

import gc
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path when running as a script
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CUR_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from constants import DEVICE
from CT_entities.DRR_utils import crop_according_to_seg
from CT_entities.CT_Rotations import get_random_rotation_angles, rotate_ct_and_crop_according_to_seg

from CT_entities.Cardiomegaly import Cardiomegaly
from CT_entities.Consolidation import Consolidation
from CT_entities.Pleural_Effusion import PleuralEffusion
from CT_entities.Pneumothorax import Pneumothorax
from CT_entities.Fluid_Overload import FluidOverload
from CT_entities.External_Devices import ExternalDevices


def _resize_2d_to_512(x: torch.Tensor) -> torch.Tensor:
    # x: (H, W)
    x4 = x[None, None, ...]
    y4 = F.interpolate(x4, size=(512, 512), mode='bilinear', align_corners=False)
    return y4.squeeze(0).squeeze(0)


@dataclass(frozen=True)
class RotationParams:
    rot_ranges: tuple[float, float, float] = (17.5, 17.5, 17.5)
    max_angles_sum: float = 37.5
    min_angles_sum: float = 0.0
    exponent: float = 1.75


def _load_nifti_as_torch(path: str) -> torch.Tensor:
    nif = nib.load(path)
    data = np.asarray(nif.dataobj, dtype=np.float32)
    nif.uncache()
    del nif

    # Match DRR_generator convention: transpose (z, x, y) then flip z
    t = torch.from_numpy(np.transpose(data, (2, 0, 1)).copy())
    t = torch.flip(t, dims=[0])
    return t


def load_scan_and_segs(ct_path: str, segs_paths: dict[str, str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ct = _load_nifti_as_torch(ct_path)
    segs: dict[str, torch.Tensor] = {}
    for name, seg_path in segs_paths.items():
        segs[name] = _load_nifti_as_torch(seg_path)
    return ct, segs


def project_ct_to_drr(ct_scan: torch.Tensor, *, dim: int = 2) -> torch.Tensor:
    """Project a (D,H,W) CT into a (1,512,512) DRR tensor in [0,1]."""
    ct_scan = ct_scan.to(torch.float32)
    ct_scan = torch.clamp_min(ct_scan, -1000)
    min_val = ct_scan.min()
    ct_scan = ct_scan - min_val

    xray = torch.sum(ct_scan, dim=dim)
    xray = _resize_2d_to_512(xray)

    # Normalize (and mimic the exp/sharpness from DRR_generator)
    xray_np = xray.detach().cpu().numpy()
    xray_np = (xray_np - np.min(xray_np)) / (np.max(xray_np) - np.min(xray_np) + 1e-8)
    xray_np = -np.exp(-xray_np)
    xray_np = (xray_np - np.min(xray_np)) / (np.max(xray_np) - np.min(xray_np) + 1e-8)
    xray_np = np.clip(xray_np, 0.0, 1.0)

    return torch.from_numpy(xray_np).to(torch.float32).unsqueeze(0)


def default_segs_path_builder(
    ct_path: str,
    *,
    lungs_dir: str,
    bronchi_dir: Optional[str] = None,
    lobes_dir: Optional[str] = None,
    vessels_dir: Optional[str] = None,
) -> dict[str, str]:
    """Build segmentation paths using the naming convention used by DRR_generator.

    Required:
    - lungs_dir: contains `{case_name}_seg.nii.gz`

    Optional (needed by some entities):
    - bronchi_dir: `{case_name}_bronchia_seg.nii.gz`
    - lobes_dir: lobe segmentations (if present in your setup)
    - vessels_dir: `{case_name}_vessels_seg.nii.gz` (currently not always used)
    """
    case_name = os.path.basename(ct_path)
    if case_name.endswith('.nii.gz'):
        case_name = case_name[:-7]

    segs: dict[str, str] = {
        'lungs': os.path.join(lungs_dir, f'{case_name}_seg.nii.gz'),
    }

    if bronchi_dir is not None:
        segs['bronchi'] = os.path.join(bronchi_dir, f'{case_name}_bronchia_seg.nii.gz')

    if vessels_dir is not None:
        segs['lung_vessels'] = os.path.join(vessels_dir, f'{case_name}_vessels_seg.nii.gz')

    # Lobe segs are optional and depend on how you generated them.
    if lobes_dir is not None:
        lobe_names = [
            'middle_right_lobe',
            'upper_right_lobe',
            'lower_right_lobe',
            'upper_left_lobe',
            'lower_left_lobe',
        ]
        for ln in lobe_names:
            # Common convention in this repo appears to be `{case_name}_{lobe}.nii.gz` but varies.
            # If you use a different convention, pass a custom `segs_path_builder`.
            candidate = os.path.join(lobes_dir, f'{case_name}_{ln}.nii.gz')
            if os.path.exists(candidate):
                segs[ln] = candidate

    return segs


def add_entities_to_pair(
    *,
    scans: list[torch.Tensor],
    segs: list[dict[str, torch.Tensor]],
    orig_scan: torch.Tensor,
    orig_lungs: torch.Tensor,
    intra_pulmonary_entities: list[tuple[type, float]],
    extra_pulmonary_entities: list[tuple[type, float]],
    devices_entity: list[tuple[type, float]],
    entity_prob_decay: float,
) -> dict[str, Any]:
    """Copied/trimmed from DRR_generator.py to avoid importing that huge script."""

    patient_mode = random.choices(['supine', 'erect'], weights=[0.8, 0.2], k=1)[0]
    pleural_effusion_patient_mode = 'erect' if patient_mode == 'erect' else ('supine' if random.random() < 0.5 else 'semi-supine')

    c_dict: dict[str, Any] = {
        'scans': scans,
        'segs': segs,
        'orig_scan': orig_scan,
        'orig_lungs': orig_lungs,
        'patient_mode': patient_mode,
        'pleural_effusion_patient_mode': pleural_effusion_patient_mode,
        'added_entity_names': [],
        'log_params': True,
    }
    c_entity_prob_mult = 1.0

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


_ENTITY_NAME_TO_CLASS: dict[str, type] = {
    'Consolidation': Consolidation,
    'FluidOverload': FluidOverload,
    'PleuralEffusion': PleuralEffusion,
    'Pneumothorax': Pneumothorax,
    'Cardiomegaly': Cardiomegaly,
    'ExternalDevices': ExternalDevices,
}

_INTRA = {'Consolidation', 'FluidOverload'}
_EXTRA = {'PleuralEffusion', 'Pneumothorax', 'Cardiomegaly'}
_DEVICES = {'ExternalDevices'}


def _split_entities(entity_probs: dict[str, float]) -> tuple[list[tuple[type, float]], list[tuple[type, float]], list[tuple[type, float]]]:
    intra: list[tuple[type, float]] = []
    extra: list[tuple[type, float]] = []
    devices: list[tuple[type, float]] = []

    for name, prob in entity_probs.items():
        if prob <= 0:
            continue
        if name not in _ENTITY_NAME_TO_CLASS:
            raise ValueError(f'Unknown entity name: {name}. Known: {sorted(_ENTITY_NAME_TO_CLASS.keys())}')
        cls = _ENTITY_NAME_TO_CLASS[name]
        if name in _INTRA:
            intra.append((cls, float(prob)))
        elif name in _EXTRA:
            extra.append((cls, float(prob)))
        elif name in _DEVICES:
            devices.append((cls, float(prob)))

    return intra, extra, devices


def maybe_inject_entity_into_current(
    clean_scan: torch.Tensor,
    segs: dict[str, torch.Tensor],
    *,
    p_inject: float,
    entity_probs: dict[str, float],
    entity_prob_decay: float = 1.0,
    max_tries: int = 5,
    change_eps: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return a scan with entity injected into *current* with prob p_inject.

    Because entity code is pair-based and can sometimes apply changes to the "prior" only,
    we retry until we detect that the "current" differs from the clean scan.
    """

    meta: dict[str, Any] = {
        'entity_applied': False,
        'entity_names': [],
    }

    if random.random() >= p_inject:
        return clean_scan, meta

    intra, extra, devices = _split_entities(entity_probs)
    if not (intra or extra or devices):
        return clean_scan, meta

    if 'lungs' not in segs:
        raise ValueError('Segmentation dict must contain "lungs" for cropping/injection.')

    for _ in range(max_tries):
        scans = [clean_scan.clone().to(DEVICE), clean_scan.clone().to(DEVICE)]
        seg_list = [segs, segs]

        out = add_entities_to_pair(
            scans=scans,
            segs=seg_list,
            orig_scan=clean_scan.clone().to(DEVICE),
            orig_lungs=segs['lungs'].clone().to(DEVICE),
            intra_pulmonary_entities=intra,
            extra_pulmonary_entities=extra,
            devices_entity=devices,
            entity_prob_decay=entity_prob_decay,
        )

        current = out['scans'][1].detach()
        delta = torch.mean(torch.abs(current - clean_scan.to(current.device))).item()
        if delta > change_eps:
            meta['entity_applied'] = True
            meta['entity_names'] = out.get('added_entity_names', [])
            return current.to(clean_scan.device), meta

    return clean_scan, meta


class DRRTripletFromCTDataset(torch.utils.data.Dataset):
    """PyTorch dataset generating triplet DRRs from CT + segmentations on-the-fly."""

    def __init__(
        self,
        ct_paths: list[str],
        segs_path_builder: Callable[[str], dict[str, str]],
        *,
        rotation: RotationParams = RotationParams(),
        p_entity: float = 0.35,
        entity_probs: Optional[Dict[str, float]] = None,
        entity_prob_decay: float = 1.0,
        seed: Optional[int] = None,
        projection_dim: int = 2,
        return_meta: bool = True,
    ):
        self.ct_paths = list(ct_paths)
        self.segs_path_builder = segs_path_builder
        self.rotation = rotation
        self.p_entity = float(p_entity)
        self.entity_probs = entity_probs or {
            'Consolidation': 0.15,
            'FluidOverload': 0.15,
            'PleuralEffusion': 0.15,
            'Pneumothorax': 0.15,
            'Cardiomegaly': 0.15,
            'ExternalDevices': 0.5,
        }
        self.entity_prob_decay = float(entity_prob_decay)
        self.projection_dim = int(projection_dim)
        self.return_meta = bool(return_meta)

        self._rng = random.Random(seed) if seed is not None else None

    def __len__(self) -> int:
        return len(self.ct_paths)

    def __getitem__(self, idx: int):
        # Optional per-item deterministic RNG
        if self._rng is not None:
            state = random.getstate()
            random.setstate(self._rng.getstate())

        ct_path = self.ct_paths[idx]
        seg_paths = self.segs_path_builder(ct_path)
        ct, segs = load_scan_and_segs(ct_path, seg_paths)

        # Crop first (speed + keeps consistent framing)
        if 'lungs' not in segs:
            raise ValueError('Segmentation dict must include "lungs".')
        ct_c, segs_c, _ = crop_according_to_seg(ct, segs['lungs'], all_segs_dict=segs, tight_y=False)

        # Sample a target view rotation (angles for views 2 and 3)
        a1, a2, a3 = get_random_rotation_angles(
            self.rotation.rot_ranges,
            self.rotation.max_angles_sum,
            self.rotation.min_angles_sum,
            self.rotation.exponent,
        )

        # Maybe inject entity into a copy (this will be used only for view 3)
        # Use Python random for the internal entity code (consistent with existing entities)
        scan_path, ent_meta = maybe_inject_entity_into_current(
            ct_c.to(DEVICE),
            {k: v.to(DEVICE) for k, v in segs_c.items()},
            p_inject=self.p_entity,
            entity_probs=self.entity_probs,
            entity_prob_decay=self.entity_prob_decay,
        )

        # View 1: clean, base angle
        drr0 = project_ct_to_drr(ct_c, dim=self.projection_dim)

        # View 2: clean, rotated
        ct_rot = rotate_ct_and_crop_according_to_seg(
            ct_c, segs_c['lungs'], rotate_angle1=a1, rotate_angle2=a2, rotate_angle3=a3, return_ct_seg=False
        )[0]
        drr1 = project_ct_to_drr(ct_rot, dim=self.projection_dim)

        # View 3: pathological (or clean), same rotation
        ct_path_rot = rotate_ct_and_crop_according_to_seg(
            scan_path.to(ct_c.device), segs_c['lungs'], rotate_angle1=a1, rotate_angle2=a2, rotate_angle3=a3, return_ct_seg=False
        )[0]
        drr2 = project_ct_to_drr(ct_path_rot, dim=self.projection_dim)

        if self._rng is not None:
            # Update deterministic RNG state based on global random
            self._rng.setstate(random.getstate())
            random.setstate(state)

        if not self.return_meta:
            return drr0, drr1, drr2

        meta = {
            'ct_path': ct_path,
            'angles_deg': (float(a1), float(a2), float(a3)),
            **ent_meta,
        }
        return drr0, drr1, drr2, meta
