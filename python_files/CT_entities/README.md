# CT Entities - Synthetic DRR Generation

This module generates synthetic baseline/followup CXR pairs from CT scans with inserted 3D pathological entities.

## Overview

The DRR (Digitally Reconstructed Radiograph) generator creates training data for the longitudinal change detection model by:
1. Loading CT volumes with lung segmentations
2. Optionally rotating the CT in 3D
3. Inserting synthetic 3D pathological entities
4. Projecting to 2D (DRR generation)
5. Computing ground truth difference maps

## Files

| File | Description |
|------|-------------|
| `DRR_generator.py` | Main generation script with CLI |
| `DRR_utils.py` | DRR projection and utility functions |
| `CT_Rotations.py` | 3D CT rotation utilities |
| `CXR_from_CT.py` | CT to CXR projection functions |
| `Entity3D.py` | Base class for 3D entities |
| `Consolidation.py` | Lung consolidation entity |
| `Pleural_Effusion.py` | Pleural effusion entity |
| `Pneumothorax.py` | Pneumothorax entity |
| `Cardiomegaly.py` | Cardiomegaly entity |
| `Fluid_Overload.py` | Fluid overload entity |
| `External_Devices.py` | External devices (tubes, lines) |

## Usage

### Command Line

```bash
cd CT_entities
python DRR_generator.py \
    -n 1000 \
    -i /path/to/CT/scans \
    -o /path/to/output \
    -CO 0.3 \
    -PL 0.2 \
    -PN 0.1 \
    -CA 0.15 \
    -FO 0.1 \
    -EX 0.05
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-n, --number_pairs` | Number of pairs to generate | Required |
| `-i, --input` | Input CT directories | Predefined paths |
| `-o, --output` | Output directory | Predefined path |
| `-CO, --Consolidation` | Consolidation probability | 0.0 |
| `-PL, --PleuralEffusion` | Pleural effusion probability | 0.0 |
| `-PN, --Pneumothorax` | Pneumothorax probability | 0.0 |
| `-CA, --Cardiomegaly` | Cardiomegaly probability | 0.0 |
| `-FO, --FluidOverload` | Fluid overload probability | 0.0 |
| `-EX, --ExternalDevices` | External devices probability | 0.0 |

## Output Structure

```
output_dir/
└── ct_case_name/
    └── pair_0/
        ├── prior.nii.gz        # Baseline DRR (512x512)
        ├── current.nii.gz      # Followup DRR with changes
        └── diff_map.nii.gz     # Ground truth difference map
```

## Entity Classes

All entities inherit from `Entity3D` and implement:
- `generate()`: Create the 3D entity in CT space
- `get_mask()`: Return binary mask of affected region

### Example: Adding a New Entity

```python
from Entity3D import Entity3D

class NewEntity(Entity3D):
    def __init__(self, ct_volume, lung_seg):
        super().__init__(ct_volume, lung_seg)
        
    def generate(self):
        # Modify self.ct_volume
        # Return affected region mask
        pass
```

## Requirements

- nibabel (NIfTI I/O)
- torch (GPU acceleration)
- scipy, skimage (morphology operations)
- kornia (image processing)

## Data Format

**Input:**
- CT volumes: NIfTI format (.nii.gz)
- Lung segmentations: Binary NIfTI masks

**Output:**
- DRR images: 512x512 NIfTI, normalized to [0, 255]
- Difference maps: Signed values for change detection
