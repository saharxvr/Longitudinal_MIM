"""Longitudinal v2 modules (staged training + optional deformation).

This package intentionally does NOT modify the legacy training scripts.
It provides a new model/trainer that preserves the existing call contract:

    pred = model(bls, fus)  # -> [B, 1, 512, 512]

while enabling staged training with triplets (prior/intermediate/current).
"""

from .model import LongitudinalMIMDeformV2
from .trainer import StageConfig, TrainerV2
from .datasets import TripletDRRDataset, DRRImageDataset

__all__ = [
    "LongitudinalMIMDeformV2",
    "StageConfig",
    "TrainerV2",
    "TripletDRRDataset",
    "DRRImageDataset",
]
