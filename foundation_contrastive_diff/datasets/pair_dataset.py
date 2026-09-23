"""
Longitudinal pair dataset with change-type supervision (RQ1/RQ2/RQ3).

Reads the manifest produced by data_generation/build_manifest.py (+ split_dataset.py)
and serves (prior, current) CXR pairs, the ground-truth signed difference map, and the
labels emitted by the synthetic DRR pipeline:

    sample = {
        'img_prior':    [1, H, W]  float in [0, 1],
        'img_curr':     [1, H, W]  float in [0, 1],
        'gt_diff':      [1, H, W]  signed change map, clipped to [-1, +1],
        'anomaly_type': int        index into constants.ANOMALY_TYPES (effective),
        'direction':    int        index into constants.DIRECTION_TYPES,
        'is_pathology': int        0 = nuisance / no realized change, 1 = pathology change,
        'change_group_id': str     shared across variants of the same change (RQ2 positives),
        'pair_id':      str,
    }

Uses the enriched-manifest records (schema >= 3) so labels reflect the *realized*
change (effective_anomaly_type / realized_change), not merely the applied entity.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Make the study's constants importable regardless of CWD.
_FCD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FCD_ROOT not in sys.path:
    sys.path.insert(0, _FCD_ROOT)
import constants as C  # noqa: E402


def _read_manifest(root: str) -> List[Dict[str, Any]]:
    """Prefer the split manifest (has a 'split' field); fall back to the plain one."""
    for name in ("manifest_split.jsonl", "manifest.jsonl"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]
            if recs:
                return recs
    raise FileNotFoundError(
        f"No manifest found in {root}. Run build_manifest.py (and split_dataset.py) first."
    )


def _load_nii_2d(path: str) -> np.ndarray:
    import nibabel as nib  # local import: heavy, only needed at load time
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    return np.squeeze(arr)


def _to_chw(arr: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    if t.ndim != 2:
        t = t.reshape(t.shape[-2], t.shape[-1])
    t = t[None, None, ...]  # [1,1,H,W]
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0]  # [1,H,W]


def _norm01(t: torch.Tensor) -> torch.Tensor:
    mn, mx = t.amin(), t.amax()
    return (t - mn) / (mx - mn + 1e-8)


class LongitudinalPairDataset(Dataset):
    """Synthetic longitudinal pairs with per-pair change-type labels, from the manifest."""

    def __init__(
        self,
        dataset_root: str,
        split: Optional[str] = None,
        img_size: int = C.IMG_SIZE,
        clip_gt: float = 1.0,
        only_realized: bool = False,
    ):
        """
        dataset_root: folder containing manifest[_split].jsonl and the pair tree.
        split: 'train' | 'val' | 'test' | None (None = all records).
        only_realized: keep only pairs with a realized change (drops progress=0 no-change).
        """
        super().__init__()
        self.root = dataset_root
        self.img_size = int(img_size)
        self.clip_gt = float(clip_gt)

        records = _read_manifest(dataset_root)
        if split is not None:
            records = [r for r in records if r.get("split") == split]
            if not records:
                raise ValueError(f"No records for split={split!r}. Did you run split_dataset.py?")
        if only_realized:
            records = [r for r in records if r.get("realized_change", True)]

        self.records = records
        self.anomaly_types = list(C.ANOMALY_TYPES)
        self.direction_types = list(C.DIRECTION_TYPES)

    def __len__(self) -> int:
        return len(self.records)

    def _anomaly_index(self, rec: Dict[str, Any]) -> int:
        a = rec.get("effective_anomaly_type", rec.get("anomaly_type", "none"))
        return self.anomaly_types.index(a) if a in self.anomaly_types else 0

    def _direction_index(self, rec: Dict[str, Any]) -> int:
        d = rec.get("direction", "none")
        # 'mixed' isn't in DIRECTION_TYPES -> treat as 'none' for the categorical label.
        return self.direction_types.index(d) if d in self.direction_types else 0

    def _abspath(self, rec: Dict[str, Any], key: str, fallback: str) -> str:
        p = rec.get(key)
        if p and os.path.isabs(p) and os.path.isfile(p):
            return p
        # Reconstruct from pair_id if the stored abs path isn't valid on this machine.
        return os.path.join(self.root, rec["pair_id"].replace("/", os.sep), fallback)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        size = self.img_size

        prior = _norm01(_to_chw(_load_nii_2d(self._abspath(rec, "prior_path", "prior.nii.gz")), size))
        curr = _norm01(_to_chw(_load_nii_2d(self._abspath(rec, "current_path", "current.nii.gz")), size))
        gt = _to_chw(_load_nii_2d(self._abspath(rec, "diff_map_path", "diff_map.nii.gz")), size)
        gt = gt.clamp(-self.clip_gt, self.clip_gt)

        return {
            "img_prior": prior,
            "img_curr": curr,
            "gt_diff": gt,
            "anomaly_type": self._anomaly_index(rec),
            "direction": self._direction_index(rec),
            "is_pathology": int(bool(rec.get("realized_change", rec.get("effective_anomaly_type", "none") != "none"))),
            "change_group_id": rec.get("change_group_id", ""),
            "pair_id": rec.get("pair_id", ""),
        }


class CachedPairDataset(Dataset):
    """Reads precomputed backbone features (training/cache_features.py) for head training.

    Serves the same keys as LongitudinalPairDataset plus the cached tokens, so the head
    trains without the backbone or NIfTIs:
        p_prior/p_curr [N, D], cls_prior/cls_curr [D], gt_diff [1,H,W], + labels.
    """

    def __init__(self, dataset_root: str, cache_dir: str, split: Optional[str] = None):
        super().__init__()
        recs = _read_manifest(dataset_root)
        if split is not None:
            recs = [r for r in recs if r.get("split") == split]
        types = list(C.ANOMALY_TYPES)
        self.paths = []
        self.labels = []  # effective anomaly index per sample (for class-balanced sampling)
        for r in recs:
            pid = r.get("pair_id", "")
            p = os.path.join(cache_dir, pid.replace("/", os.sep), "feat.pt")
            if pid and os.path.isfile(p):
                self.paths.append(p)
                a = r.get("effective_anomaly_type", r.get("anomaly_type", "none"))
                self.labels.append(types.index(a) if a in types else 0)
        if not self.paths:
            raise FileNotFoundError(
                f"No cached feat.pt found under {cache_dir} for split={split!r}. "
                f"Run training/cache_features.py first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = torch.load(self.paths[idx], map_location="cpu")
        return {
            "p_prior": d["p_prior"].float(),
            "cls_prior": d["cls_prior"].float(),
            "p_curr": d["p_curr"].float(),
            "cls_curr": d["cls_curr"].float(),
            "gt_diff": d["gt_diff"].float(),
            "anomaly_type": d["anomaly_type"],
            "direction": d["direction"],
            "is_pathology": d["is_pathology"],
            "change_group_id": d["change_group_id"],
            "pair_id": d["pair_id"],
        }
