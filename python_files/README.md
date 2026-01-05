# Longitudinal CXR Analysis

Deep learning pipeline for detecting changes between longitudinal chest X-ray pairs using synthetic DRR training data.

## Project Structure

```
python_files/
├── longitudinal_MIM_training.py   # 🎯 Main training script
├── models.py                      # Neural network architectures
├── datasets.py                    # Dataset loaders
├── constants.py                   # Configuration & hyperparameters
├── utils.py                       # Utility functions
├── augmentations.py               # Data augmentation transforms
│
├── CT_entities/                   # 🎯 Synthetic DRR generation
│   ├── DRR_generator.py           # Main DRR pair generator
│   ├── DRR_utils.py               # DRR helper functions
│   ├── CT_Rotations.py            # 3D rotation utilities
│   ├── Entity3D.py                # Base class for 3D entities
│   ├── CXR_from_CT.py             # CT to CXR projection
│   ├── Consolidation.py           # Lung consolidation entity
│   ├── Pleural_Effusion.py        # Pleural effusion entity
│   ├── Pneumothorax.py            # Pneumothorax entity
│   ├── Cardiomegaly.py            # Cardiomegaly entity
│   ├── Fluid_Overload.py          # Fluid overload entity
│   └── External_Devices.py        # External devices entity
│
├── losses/                        # Custom loss functions
│   └── vgg_losses.py              # VGG perceptual loss
│
└── archive/                       # Archived/unused code
    ├── refactored_modules/        # Previously refactored code
    ├── test_scripts/              # Test scripts (DRRs_test_*.py)
    ├── data_preparation/          # Data prep utilities
    ├── evaluation/                # Evaluation scripts
    └── experimental/              # Experimental code
```

## Quick Start

### Training

```bash
python longitudinal_MIM_training.py
```

Key configuration in `constants.py`:
- `BATCH_SIZE`, `MAX_LR` - Training hyperparameters
- `USE_L1`, `USE_SSIM`, `USE_PERC_STYLE` - Loss function flags
- `TRAIN_CSV`, `VAL_CSV` - Dataset paths

### Generating Synthetic DRR Pairs

```bash
cd CT_entities
python DRR_generator.py -n 1000 -o /output/path \
    -CO 0.3 -PL 0.2 -PN 0.1 -CA 0.15 -FO 0.1
```

Arguments:
- `-n`: Number of pairs to generate
- `-o`: Output directory
- `-CO`: Consolidation probability
- `-PL`: Pleural effusion probability  
- `-PN`: Pneumothorax probability
- `-CA`: Cardiomegaly probability
- `-FO`: Fluid overload probability

## Model Architecture

```
Baseline CXR ─┐
              ├─→ Shared EfficientNet Encoder ─→ Bottleneck ─→ Decoder ─→ Change Map
Followup CXR ─┘                                  (ViT+Conv)              ([-1, +1])
```

- **Encoder**: EfficientNet-B7 (first 4 blocks)
- **Bottleneck**: Dual-branch (Transformer + CNN)  
- **Decoder**: 6-stage upsampling with Tanh output
- **Output**: Signed change map (positive = new findings, negative = resolved)

## Key Files

| File | Purpose |
|------|---------|
| `longitudinal_MIM_training.py` | Training loop with L1, SSIM, perceptual losses |
| `models.py` | `LongitudinalMIMModel` and variants |
| `datasets.py` | `LongitudinalMIMDataset` for BL/FU pairs |
| `CT_entities/DRR_generator.py` | Synthetic pair generation with 3D entities |

## Archive

The `archive/` folder contains code that is not part of the main workflow:

- **refactored_modules/**: Previous attempt at modular reorganization
- **test_scripts/**: DRR testing scripts for different conditions
- **data_preparation/**: CSV creation, image conversion utilities
- **evaluation/**: Inference and observer variability analysis
- **experimental/**: DDPM, masked reconstruction, and other experiments
