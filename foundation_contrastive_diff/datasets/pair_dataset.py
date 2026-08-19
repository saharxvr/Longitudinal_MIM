"""
Longitudinal pair dataset with change-type supervision.

Loads (prior, current) CXR pairs plus the ground-truth signed difference map and the
change-type labels emitted by the synthetic DRR pipeline:

    sample = {
        'img_prior':    [1, H, W],
        'img_curr':     [1, H, W],
        'gt_diff':      [1, H, W]   signed change map in [-1, +1],
        'anomaly_type': int         index into constants.ANOMALY_TYPES,
        'direction':    int         index into constants.DIRECTION_TYPES,
        'is_pathology': int         0 = nuisance, 1 = pathology,
    }

Reuses NIfTI loading + normalization conventions from python_files/datasets.py
(LongitudinalMIMDataset). The DRR generator must be extended to log which entity was
added/removed and the change direction (see RESEARCH_PLAN.md section 4).
"""

import os
from glob import glob

import torch
from torch.utils.data import Dataset

# NOTE: when wired up, import shared helpers from the parent project, e.g.
#   import sys; sys.path.append('../python_files')
#   from datasets import LongitudinalMIMDataset  # for reference / reuse


class LongitudinalPairDataset(Dataset):
    """Synthetic longitudinal pairs with per-pair change-type labels."""

    def __init__(self, pairs_dir: str, img_size: int = 512, with_labels: bool = True):
        super().__init__()
        self.pairs_dir = pairs_dir
        self.img_size = img_size
        self.with_labels = with_labels

        # TODO(Phase 1): index pair folders/files produced by DRR_generator.py.
        self.samples = sorted(glob(os.path.join(pairs_dir, "**", "*.nii.gz"), recursive=True))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        # TODO(Phase 1): load prior/current NIfTI, GT diff map, and the JSON/CSV
        # change-type metadata emitted alongside each synthetic pair.
        raise NotImplementedError(
            "Wire up loading of synthetic pairs + change-type labels — see RESEARCH_PLAN.md."
        )
