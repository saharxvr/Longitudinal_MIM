# CT_entities Module

## Overview

The `CT_entities/` module contains classes for generating synthetic 3D pathological entities that can be inserted into CT scans to create realistic training data for the longitudinal change detection model.

## Module Structure

```
CT_entities/
├── Entity3D.py           # Abstract base class for all entities
├── Consolidation.py      # Lung consolidation generator
├── Pleural_Effusion.py   # Pleural effusion generator
├── Pneumothorax.py       # Pneumothorax generator
├── Cardiomegaly.py       # Cardiac enlargement generator
├── Fluid_Overload.py     # Pulmonary edema generator
├── External_Devices.py   # Medical device insertion
├── DRR_generator.py      # Main generator script
├── DRR_utils.py          # DRR utility functions
├── CT_Rotations.py       # 3D rotation utilities
└── CXR_from_CT.py        # CT to X-ray projection
```

## Base Class: Entity3D

All pathological entities inherit from the abstract `Entity3D` class.

### Key Methods

| Method | Description |
|--------|-------------|
| `add_to_CT_pair()` | Insert entity into CT scan pair |
| `calc_lungs_height()` | Calculate lung height for scaling |
| `get_balls()` | Generate spherical structures |
| `fill_borders_gap()` | Fill gaps at entity boundaries |
| `get_intensity_map()` | Generate random intensity variations |
| `binary_erosion/dilation()` | Morphological operations |

### Usage Pattern

```python
from Consolidation import Consolidation

# Input: prior CT, current CT, segmentations
result = Consolidation.add_to_CT_pair(
    scans=[prior_ct, current_ct],
    segs=[prior_segs, current_segs],
    log_params=True  # Return generation parameters
)

# Result contains modified scans and metadata
modified_prior = result['prior']
modified_current = result['current']
diff_mask = result['diff_mask']
```

## Entity Classes

### 1. Consolidation

Generates lung consolidation regions (pneumonia, infiltrates).

**Characteristics:**
- Random number of consolidation balls
- Variable size based on lung height
- B-spline deformation for realistic shapes
- Intensity variation with depth

**Parameters:**
- `num_nodules`: Range of consolidation regions (1-400)
- `radius_range`: Size relative to lung height (2-40%)
- `intensity`: HU values (-400 to 30)

### 2. Pleural Effusion

Generates fluid accumulation in pleural space.

**Characteristics:**
- Gravity-dependent distribution
- Unilateral or bilateral
- Variable fluid levels
- Smooth intensity gradients

**Parameters:**
- `side`: Left, right, or bilateral
- `level`: Fluid height (0-100% of lung)
- `intensity`: HU values (0 to 30)

### 3. Pneumothorax

Generates collapsed lung regions with air in pleural space.

**Characteristics:**
- Peripheral air collection
- Variable collapse degree
- Sharp lung edge visualization

**Parameters:**
- `side`: Affected side
- `collapse_percent`: Degree of collapse (10-90%)

### 4. Cardiomegaly

Generates enlarged cardiac silhouette.

**Characteristics:**
- Symmetric or asymmetric enlargement
- Preserves heart shape
- Affects cardiothoracic ratio

**Parameters:**
- `enlargement_factor`: Scale factor (1.1-1.5)
- `direction`: Predominant enlargement direction

### 5. Fluid Overload

Generates pulmonary edema patterns.

**Characteristics:**
- Perihilar distribution
- Bilateral involvement
- Kerley lines simulation
- Bat-wing pattern

**Parameters:**
- `severity`: Mild, moderate, severe
- `distribution`: Perihilar, diffuse

### 6. External Devices

Inserts medical devices (tubes, lines, catheters).

**Characteristics:**
- Anatomically placed
- Various device types
- Realistic trajectories

**Supported Devices:**
- Endotracheal tube (ETT)
- Nasogastric tube (NGT)
- Central venous catheters
- Chest tubes
- Pacemaker leads

## DRR Generation Pipeline

### DRR_utils.py Functions

| Function | Description |
|----------|-------------|
| `enforce_ndim_4()` | Ensure 4D tensor format |
| `softmax_and_rescale()` | Softmax with rescaling |
| `crop_according_to_seg()` | Crop CT to segmentation bounds |
| `get_seg_bbox()` | Get segmentation bounding box |
| `add_back_cropped()` | Insert cropped region back |

### CT_Rotations.py Functions

| Function | Description |
|----------|-------------|
| `random_rotate_ct_and_crop_according_to_seg()` | Apply 3D rotation with segmentation handling |
| `get_rotation_matrix()` | Generate 3D rotation matrix |
| `apply_rotation()` | Apply rotation to volume |

## Workflow

```
1. Load CT Scan + Segmentations
   └── Load .nii.gz files (CT, lungs, heart, etc.)

2. Initialize Entity
   └── e.g., Consolidation()

3. Generate Entity Parameters
   ├── Random region selection
   ├── Number and size of features
   └── Intensity distribution

4. Insert Entity into CT
   ├── Generate 3D entity shape
   ├── Apply deformation
   ├── Set intensity values
   └── Blend with original CT

5. Generate DRR Projection
   └── Ray-sum projection to 2D X-ray

6. Create Difference Map
   └── |Modified DRR - Original DRR|
```

## Segmentation Requirements

Each entity requires specific segmentations:

| Entity | Required Segmentations |
|--------|----------------------|
| Consolidation | lungs, bronchi |
| Pleural Effusion | lungs, ribs |
| Pneumothorax | lungs |
| Cardiomegaly | heart, lungs |
| Fluid Overload | lungs, heart |
| External Devices | various (device-specific) |

### Segmentation Format

```python
segs = {
    'lungs': torch.Tensor,      # [D, H, W] binary mask
    'heart': torch.Tensor,       # [D, H, W] binary mask
    'bronchi': torch.Tensor,     # [D, H, W] binary mask
    'ribs': torch.Tensor,        # [D, H, W] binary mask
    # ...
}
```

## Example: Adding Consolidation

```python
import torch
import nibabel as nib
from Consolidation import Consolidation

# Load CT and segmentations
ct_nif = nib.load('ct_scan.nii.gz')
ct_data = torch.tensor(ct_nif.get_fdata())

lungs_seg = torch.tensor(nib.load('lungs_seg.nii.gz').get_fdata())
bronchi_seg = torch.tensor(nib.load('bronchi_seg.nii.gz').get_fdata())

segs = {
    'lungs': lungs_seg,
    'bronchi': bronchi_seg
}

# Create prior (original) and current (to be modified)
prior = ct_data.clone()
current = ct_data.clone()

# Add consolidation to current
result = Consolidation.add_to_CT_pair(
    scans=[prior, current],
    segs=[segs, segs],
    log_params=True
)

# Extract results
modified_current = result['current']
diff_mask = result['diff_mask']
params = result['params']  # Generation parameters
```

## Memory Considerations

Entity generation can be memory-intensive due to 3D operations:

```python
import gc
import torch

# After processing each CT
gc.collect()
torch.cuda.empty_cache()

# Use lower resolution for testing
ct_data = torch.nn.functional.interpolate(
    ct_data[None, None, ...], 
    scale_factor=0.5
).squeeze()
```

## Dependencies

```python
import torch
import numpy as np
import nibabel as nib
import gryds  # For B-spline deformations
from scipy import ndimage
from skimage.morphology import ball, erosion, dilation
```

## Configuration Constants

Key constants from `constants.py`:

```python
DEVICE = 'cuda'  # or 'cpu'

# Entity-specific (in respective files)
DEFAULT_HU_AIR = -1000
DEFAULT_HU_WATER = 0
DEFAULT_HU_SOFT_TISSUE = 40
```

## Adding New Entities

To add a new pathological entity:

1. Create new file `NewEntity.py`
2. Inherit from `Entity3D`
3. Implement `add_to_CT_pair()` method
4. Add import to `DRR_generator.py`
5. Add command-line argument for probability

```python
from Entity3D import Entity3D

class NewEntity(Entity3D):
    @staticmethod
    def add_to_CT_pair(scans, segs, *args, **kwargs):
        # Implementation
        prior = scans[0]
        current = scans[1]
        
        # Modify current with entity
        # ...
        
        return {
            'prior': prior,
            'current': modified_current,
            'diff_mask': diff_mask,
            'params': {...}  # if log_params
        }
```
