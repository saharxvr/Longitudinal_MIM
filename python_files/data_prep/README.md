# Data Preprocessing Module (`data_prep/`)

Utilities for preprocessing CXR and CT datasets. Contains scripts for format conversion, CSV generation, and pair creation.

## Module Structure

```
data_prep/
├── __init__.py          # Module exports
├── io_operations.py     # Format conversion (PNG, DICOM, NIfTI)
├── dataset_csv.py       # CSV creation and filtering
├── pair_creation.py     # Longitudinal pair generation
└── image_processing.py  # Image-level preprocessing
```

## Quick Start

### Converting Images

```python
from data_prep import convert_png_to_nifti, convert_dicom_to_nifti

# Single image
convert_png_to_nifti("chest_xray.png", target_size=(512, 512))

# DICOM to NIfTI (handles MONOCHROME1 inversion)
convert_dicom_to_nifti("scan.dcm", invert_if_monochrome1=True)

# Batch conversion
from data_prep import convert_cases_to_nib
convert_cases_to_nib("input_folder/", "output_folder/")
```

### Creating Dataset CSVs

```python
from data_prep import create_no_finding_csv, create_specific_abnormalities_csv

# No Finding subset for contrastive learning
create_no_finding_csv(
    dataset='mimic',
    output_path='no_finding_AP.csv',
    view_position='AP'
)

# Specific abnormalities for supervised training
create_specific_abnormalities_csv(
    dataset='padchest',
    output_path='abnormalities.csv',
    abnormalities={'pneumothorax', 'consolidation', 'cardiomegaly'}
)
```

### Creating Longitudinal Pairs

```python
from data_prep import create_longitudinal_pairs, validate_pairs

# Create BL-FU pairs from patient directories
count = create_longitudinal_pairs(
    patient_dirs_pattern="/data/mimic/p*/p*",
    max_days_apart=14,
    view_position='AP'
)
print(f"Created {count} pairs")

# Validate existing pairs
valid, invalid = validate_pairs(
    pairs_pattern="/data/mimic/p*/p*/pair_*",
    max_days_apart=14,
    delete_invalid=True
)
```

### Image Preprocessing

```python
from data_prep import crop_edges, normalize_intensity, invert_photometric

# Remove black borders
cropped = crop_edges(image, threshold_ratio=0.1)

# Normalize to 0-255
normalized = normalize_intensity(image, clip_percentile=(1, 99))

# Invert MONOCHROME1
inverted = invert_photometric(image)
```

## Supported Datasets

| Dataset | No Finding | Abnormalities | Pair Creation |
|---------|------------|---------------|---------------|
| MIMIC-CXR | ✓ | - | ✓ |
| CXR-14 (NIH) | ✓ | ✓ | - |
| PadChest | ✓ | ✓ | - |
| VinDr-CXR | ✓ | - | - |
| CheXpert | - | - | - |

## File Formats

### Input Formats
- PNG (8-bit grayscale)
- DICOM (.dcm)
- JPEG

### Output Format
- NIfTI (.nii.gz) - Compressed NIfTI-1 format

### Standard Affine Matrix
All converted images use:
```python
AFFINE_DCM = np.array([
    [-0.139, 0, 0, 0],
    [0, -0.139, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float64)
```

## Pair Directory Structure

After running `create_longitudinal_pairs`:

```
patient_dir/
├── s12345678/           # Original study 1
│   └── image.dcm
├── s23456789/           # Original study 2
│   └── image.dcm
└── pair_s12345678_s23456789/
    ├── BL_s12345678/    # Symlinks to baseline
    │   └── image.dcm -> ../../s12345678/image.dcm
    └── FU_s23456789/    # Symlinks to followup
        └── image.dcm -> ../../s23456789/image.dcm
```

## Notes

- Most functions use hardcoded paths from `config/paths.py`
- Modify paths in config before running preprocessing scripts
- Large datasets should be processed on cluster nodes
- MONOCHROME1 images are automatically inverted during DICOM conversion
