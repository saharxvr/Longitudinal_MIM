# DRR Generator

## Overview

`CT_entities/DRR_generator.py` generates synthetic pairs of Digitally Reconstructed Radiographs (DRRs) from 3D CT scans. These synthetic X-ray pairs simulate temporal changes with ground truth difference maps for training the longitudinal change detection model.

## Features

- **Synthetic DRR Generation**: Creates X-ray-like projections from CT volumes
- **3D Pathology Insertion**: Adds realistic 3D pathological entities
- **Rotation Simulation**: Applies 3D rotations to simulate patient repositioning
- **Ground Truth Generation**: Produces pixel-perfect difference maps
- **Memory Management**: Built-in memory monitoring and cleanup to prevent OOM crashes

## Usage

### Basic Usage
```bash
python CT_entities/DRR_generator.py -n 1000 -o /output/path
```

### Full Example
```bash
python CT_entities/DRR_generator.py \
    -n 5000 \
    -i /path/to/CT_scans \
    -o /output/synthetic_pairs \
    -CO 0.3 \
    -PL 0.2 \
    -PN 0.1 \
    -CA 0.15 \
    -FO 0.1 \
    -EX 0.25 \
    -d 0.7 \
    -m 20
```

## Command-Line Arguments

### Required Arguments

| Argument | Short | Type | Description |
|----------|-------|------|-------------|
| `--number_pairs` | `-n` | int | Total number of synthetic pairs to generate |

### Input/Output

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--input` | `-i` | str[] | Predefined | Directories containing CT scans |
| `--output` | `-o` | str | Predefined | Output directory for generated pairs |

### Entity Probabilities

Each entity has an independent probability of appearing in a synthetic pair (0.0 to 1.0):

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--Consolidation` | `-CO` | 0.0 | Lung consolidation probability |
| `--PleuralEffusion` | `-PL` | 0.0 | Pleural effusion probability |
| `--Pneumothorax` | `-PN` | 0.0 | Pneumothorax probability |
| `--Cardiomegaly` | `-CA` | 0.0 | Enlarged heart probability |
| `--FluidOverload` | `-FL` | 0.0 | Fluid overload probability |
| `--ExternalDevices` | `-EX` | 0.0 | Medical devices probability |
| `--default_entities` | - | False | Use default training probabilities |

### Advanced Options

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--decay_prob_on_add` | `-d` | float | 1.0 | Probability decay after adding entity |
| `--rotation_params` | `-r` | float[] | [17.5, 37.5, 0., 1.75] | Rotation distribution parameters |
| `--slices_for_CTs_list` | `-s` | float[] | [0., 1.] | CT list slice for parallel processing |
| `--memory_threshold` | `-m` | float | 25.0 | RAM threshold (GB) for cleanup |

### Rotation Parameters Format
```
-r max_angle max_sum min_sum exponent
```
- `max_angle`: Maximum rotation per axis (degrees)
- `max_sum`: Maximum total rotation across all axes
- `min_sum`: Minimum total rotation across all axes
- `exponent`: Distribution bias (higher = closer to 0°)

## Output Structure

```
output_dir/
├── CT_case_001/
│   ├── pair_0/
│   │   ├── prior.nii.gz        # Baseline DRR (X-ray projection)
│   │   ├── current.nii.gz      # Followup DRR with changes
│   │   └── diff_map.nii.gz     # Ground truth difference map
│   ├── pair_1/
│   │   └── ...
│   └── ...
├── CT_case_002/
│   └── ...
└── generation_log.json         # Metadata for all generated pairs
```

## Supported Pathological Entities

### 1. Consolidation (`-CO`)
Lung consolidation regions simulating pneumonia or other infiltrates.
- Inserted as 3D volumetric regions in lung parenchyma
- Variable size, shape, and intensity

### 2. Pleural Effusion (`-PL`)
Fluid accumulation in the pleural space.
- Inserted in dependent lung regions
- Simulates various severity levels

### 3. Pneumothorax (`-PN`)
Collapsed lung regions with air in pleural space.
- Variable size and location
- Includes subtle and obvious cases

### 4. Cardiomegaly (`-CA`)
Enlarged cardiac silhouette.
- Simulates various degrees of enlargement
- Affects mediastinal contours

### 5. Fluid Overload (`-FO`)
General pulmonary edema pattern.
- Bilateral distribution
- Perihilar predominance

### 6. External Devices (`-EX`)
Medical devices and lines.
- Tubes (ETT, NGT, chest tubes)
- Central lines and catheters
- Various positions and configurations

## Memory Management

The generator includes built-in memory monitoring to prevent OOM crashes during long runs:

### Memory Utilities
- `get_memory_usage_gb()`: Returns current RAM usage in GB
- `log_memory(label)`: Logs memory with context label
- `cleanup_memory()`: Forces garbage collection and CUDA cache clear
- `check_memory_and_cleanup(threshold_gb)`: Conditional cleanup if above threshold

### Memory Threshold Flag
```bash
# Set memory threshold to 20GB (default: 25GB)
python DRR_generator.py -n 1000 -m 20
```

When RAM exceeds the threshold, the generator:
1. Logs a warning
2. Forces multiple garbage collection passes
3. Clears CUDA cache
4. Reports freed memory

## Parallel Processing

For large-scale generation, split work across multiple processes:

```bash
# Process 1: First half of CT scans
python DRR_generator.py -n 2500 -s 0.0 0.5 -o /output/part1

# Process 2: Second half of CT scans
python DRR_generator.py -n 2500 -s 0.5 1.0 -o /output/part2
```

## Pipeline

```
1. Load CT Scan
   └── Load .nii.gz volume + segmentation masks

2. For each pair to generate:
   ├── Apply 3D Rotation
   │   └── Random rotation with configurable distribution
   │
   ├── Insert Pathological Entities
   │   ├── Sample entities based on probabilities
   │   ├── Insert 3D volumes into CT
   │   └── Track inserted regions for diff_map
   │
   ├── Generate DRRs
   │   ├── Baseline: Original or slightly modified CT
   │   ├── Followup: CT with inserted pathology
   │   └── Project 3D → 2D using ray-sum
   │
   ├── Compute Difference Map
   │   └── Ground truth: |followup - baseline|
   │
   └── Save Outputs
       └── prior.nii.gz, current.nii.gz, diff_map.nii.gz

3. Memory Checkpoint
   └── Check and cleanup if needed
```

## CT Input Requirements

- **Format**: NIfTI (.nii.gz)
- **Modality**: Chest CT with lung segmentation
- **Required Segmentations**:
  - Lung masks (left/right)
  - Optional: Heart, ribs, mediastinum

### Expected CT Structure
```
CT_scans/
├── case_001/
│   ├── ct.nii.gz
│   └── segmentations/
│       ├── lung_left.nii.gz
│       ├── lung_right.nii.gz
│       └── heart.nii.gz
├── case_002/
│   └── ...
```

## Entity Classes

The generator uses modular entity classes defined in `CT_entities/`:

| File | Class | Description |
|------|-------|-------------|
| `Consolidation.py` | `Consolidation` | Lung consolidation generator |
| `Pleural_Effusion.py` | `PleuralEffusion` | Pleural fluid generator |
| `Pneumothorax.py` | `Pneumothorax` | Pneumothorax generator |
| `Cardiomegaly.py` | `Cardiomegaly` | Cardiac enlargement |
| `Fluid_Overload.py` | `FluidOverload` | Pulmonary edema |
| `External_Devices.py` | `ExternalDevices` | Medical device insertion |

## Dependencies

```
torch
nibabel
numpy
kornia
scipy
skimage
psutil (for memory monitoring)
matplotlib (optional, for visualization)
```

## Common Issues

### Out of Memory
```bash
# Lower memory threshold for earlier cleanup
python DRR_generator.py -n 1000 -m 15
```

### Slow Generation
- Use parallel processing with `-s` flag
- Reduce number of entities per pair with `-d` decay

### Missing Segmentations
- Ensure lung segmentation masks are available
- Check segmentation file paths in `segs_paths_dict`
