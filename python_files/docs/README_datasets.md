# Datasets

## Overview

`datasets.py` provides PyTorch Dataset classes for loading medical imaging data for training and evaluation. The main dataset for longitudinal change detection is `LongitudinalMIMDataset`.

## Dataset Classes

| Class | Purpose | Primary Use |
|-------|---------|-------------|
| `LongitudinalMIMDataset` | BL/FU pairs with diff maps | Main training dataset |
| `ContrastiveLearningDataset` | Multi-label CXR classification | Contrastive pretraining |
| `MaskedReconstructionDataset` | Single CXRs for MIM | Masked reconstruction |
| `DetectionDataset` | Labeled CXRs for detection | Multi-label detection |
| `NoFindingDataset` | Normal CXRs only | Baseline sampling |

## Main Dataset: LongitudinalMIMDataset

### Purpose
Loads baseline/followup chest X-ray pairs with corresponding ground truth difference maps for training the longitudinal change detection model.

### Data Sources
The dataset supports multiple data sources:

1. **Entity Directories**: CXR images with segmentation masks
2. **Inpaint Directories**: Inpainted abnormality pairs
3. **DRR Single Directories**: Single CT DRR variations
4. **DRR Pair Directories**: Synthetic BL/FU pairs from CT scans (primary source)

### Expected Directory Structure

```
# DRR Pair Directories (main source)
DRR_pair_dir/
├── CT_case_001/
│   ├── pair_0/
│   │   ├── prior.nii.gz        # Baseline DRR
│   │   ├── current.nii.gz      # Followup DRR
│   │   └── diff_map.nii.gz     # Ground truth difference
│   ├── pair_1/
│   │   └── ...
│   └── ...
├── CT_case_002/
│   └── ...

# Entity Directories
entity_dir/
├── case_001.nii.gz
├── case_002.nii.gz
entity_dir_segs/  # Matching segmentation directory
├── case_001_seg.nii.gz
├── case_002_seg.nii.gz
```

### Usage

```python
from datasets import LongitudinalMIMDataset

# Initialize dataset
dataset = LongitudinalMIMDataset(
    entity_dirs=['/path/to/entity_data'],
    inpaint_dirs=['/path/to/inpaint_data'],
    DRR_single_dirs=['/path/to/drr_single'],
    DRR_pair_dirs=['/path/to/drr_pairs'],  # Primary source
    abnor_both_p=0.5,       # Prob of abnormality in both images
    invariance=None,        # 'abnormality', 'devices', or None
    overlay_diff_p=0.9      # Prob of overlaying difference
)

# Create DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate
for baseline, followup, gt_diff, fu_mask in dataloader:
    # baseline: [B, 1, 512, 512] - Baseline CXR
    # followup: [B, 1, 512, 512] - Followup CXR
    # gt_diff:  [B, 1, 512, 512] - Ground truth difference
    # fu_mask:  [B, 1, 512, 512] - Followup lung mask
    pass
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_dirs` | list[str] | - | Directories with CXR + segmentation pairs |
| `inpaint_dirs` | list[str] | - | Directories with inpainted pairs |
| `DRR_single_dirs` | list[str] | - | Directories with DRR variations |
| `DRR_pair_dirs` | list[str] | - | Directories with synthetic DRR pairs |
| `abnor_both_p` | float | 0.5 | Probability of adding abnormality to both BL and FU |
| `invariance` | str | None | Type of invariance ('abnormality', 'devices') |
| `overlay_diff_p` | float | 0.9 | Probability of overlaying difference |

### Output Format

```python
# Returns tuple: (baseline, followup, gt_diff, fu_mask)
baseline: torch.Tensor  # [1, 512, 512], range [0, 1]
followup: torch.Tensor  # [1, 512, 512], range [0, 1]
gt_diff:  torch.Tensor  # [1, 512, 512], signed difference
fu_mask:  torch.Tensor  # [1, 512, 512], binary lung mask
```

### Built-in Augmentations

The dataset automatically applies augmentations:

| Transform | Description | Applied To |
|-----------|-------------|------------|
| `RandomAffineWithMaskTransform` | Random affine (scale, translate) | BL/FU |
| `RandomBsplineAndSimilarityWithMaskTransform` | B-spline deformation | BL/FU |
| `CropResizeWithMaskTransform` | Crop to lungs + resize | BL |
| `RandomIntensityTransform` | CLAHE, color jitter | BL/FU |
| `RandomFlipBLWithFU` | Swap BL↔FU | Pair |
| `RandomAbnormalizationTransform` | Synthetic abnormality overlay | Optional |

## Other Datasets

### ContrastiveLearningDataset

Multi-label classification dataset for contrastive pretraining.

```python
from datasets import ContrastiveLearningDataset

dataset = ContrastiveLearningDataset(
    data_folder='/path/to/data',
    labels_path='/path/to/labels.csv',
    ds_labels=['Atelectasis', 'Consolidation', 'Effusion'],
    groups=None,             # Optional label grouping
    get_weights=[1, 1, 1],   # Class weights
    label_no_finding=True,   # Include "No Finding" class
    train=True               # Enable shuffling
)
```

### MaskedReconstructionDataset

Single CXR images for masked image modeling pretraining.

```python
from datasets import MaskedReconstructionDataset

dataset = MaskedReconstructionDataset(
    data_folder='/path/to/cxrs',
    train_paths=[...],       # List of training paths
    rot_chance=0.075,        # Rotation probability
    hor_flip_chance=0.03,    # Horizontal flip probability
    clahe_p=0.4              # CLAHE probability
)
```

### DetectionDataset

Labeled CXRs for multi-label abnormality detection.

```python
from datasets import DetectionDataset

dataset = DetectionDataset(
    data_folder='/path/to/data',
    labels_path='/path/to/labels.csv',
    ds_labels=['Atelectasis', 'Cardiomegaly', 'Effusion'],
    train=True
)
```

## Data Requirements

### File Format
- **Image Format**: NIfTI (.nii.gz)
- **Resolution**: Any (resized to 512×512 internally)
- **Value Range**: 0-255 (normalized to 0-1)
- **Orientation**: RAS orientation expected

### Labels CSV Format
For classification datasets:
```csv
id,labels
image_001.nii.gz,[True False False True ...]
image_002.nii.gz,[False True True False ...]
```

## Label Constants

```python
ALL_LABELS = [
    'No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly',
    'Lung_Opacity', 'Lung_Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion',
    'Pleural_Other', 'Fracture', 'Support_Devices'
]
ALL_LABELS_NUM = 14
```

## Custom Samplers

### ContrastiveCXRSampler
Balanced sampling for contrastive learning with class reweighting.

```python
from datasets import ContrastiveCXRSampler, ContrastiveLearningDataset

dataset = ContrastiveLearningDataset(...)
sampler = ContrastiveCXRSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

## Performance Tips

### Memory Optimization
```python
# Use num_workers for parallel data loading
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

### Caching
For repeated access to the same data:
```python
# Pre-load data in memory (if RAM allows)
from functools import lru_cache

@lru_cache(maxsize=1000)
def load_cached(path):
    return nib.load(path).get_fdata()
```

## Dependencies

```python
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import nibabel as nib
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as v2
```

## Integration with Training

```python
from datasets import LongitudinalMIMDataset
from models import LongitudinalMIMModelBig
from torch.utils.data import DataLoader, random_split

# Create dataset
dataset = LongitudinalMIMDataset(
    entity_dirs=[],
    inpaint_dirs=[],
    DRR_single_dirs=[],
    DRR_pair_dirs=['/data/synthetic_pairs/train']
)

# Train/val split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training loop
model = LongitudinalMIMModelBig()
for bl, fu, gt_diff, mask in train_loader:
    pred_diff = model(bl, fu)
    loss = F.l1_loss(pred_diff, gt_diff)
    # ...
```
