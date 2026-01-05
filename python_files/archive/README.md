# Archive

This folder contains code that is **not used** by the main training workflow (`longitudinal_MIM_training.py`) or DRR generation (`CT_entities/DRR_generator.py`).

## Contents

### `refactored_modules/`
Previously refactored modular code structure. Not imported by main scripts.
- `config/` - Centralized configuration module
- `core/` - Split models and datasets
- `data_prep/` - Data preprocessing utilities
- `utils/` - Shared utility functions

### `test_scripts/`
Test and evaluation scripts for DRR quality:
- `DRRs_test_Magnitude.py`
- `DRRs_test_Manufacturers.py`  
- `DRRs_test_Rotation.py`

### `data_preparation/`
Data preprocessing and filtering utilities:
- `case_filtering.py` - CSV creation, pair filtering
- `png_to_nifti_recursive.py` - Image format conversion

### `evaluation/`
Post-training evaluation scripts:
- `Prediction.py` - Inference on real CXR pairs
- `Observer_Variability.py` - Observer agreement analysis

### `experimental/`
Experimental and legacy code from `extra/`:
- DDPM training/testing
- Masked reconstruction experiments
- PCA visualization
- Various test scripts

## Restoring Files

If you need any of these files, simply move them back:

```bash
# Example: Restore evaluation scripts
mv archive/evaluation/* ./Evaluation/

# Example: Restore a specific test script
mv archive/test_scripts/DRRs_test_Rotation.py ./
```
