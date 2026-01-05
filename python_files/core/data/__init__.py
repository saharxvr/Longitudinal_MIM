"""
Core Data Package
=================

Dataset classes for training and evaluation.

Modules:
--------
- base: Base dataset classes and shared transforms
- contrastive: Contrastive learning datasets
- classification: Normal/abnormal classification datasets  
- longitudinal: Longitudinal (paired) image datasets
- patch_reconstruction: Patch-based reconstruction datasets

Usage:
------
    from core.data import LongitudinalMIMDataset, ContrastiveLearningDataset
    from core.data.base import BaseTransformDataset
"""

from core.data.base import BaseTransformDataset
from core.data.contrastive import (
    ContrastiveLearningDataset,
    ContrastiveLearningSmallDataset,
    ContrastiveCXRSampler,
    GeneralContrastiveLearningDataset
)
from core.data.classification import (
    NoFindingDataset,
    SmallNoFindingDataset,
    ExpertNoFindingDataset,
    NormalAbnormalDataset
)
from core.data.longitudinal import LongitudinalMIMDataset
from core.data.patch_reconstruction import PatchReconstructionDataset
