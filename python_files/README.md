# Longitudinal CXR Analysis

Deep learning pipeline for detecting changes between longitudinal chest X-ray pairs using synthetic DRR training data.

## Project Structure

```
python_files/
├── longitudinal_MIM_training.py   # 🎯 Main training script
├── models.py                      # Neural network architectures (documented)
├── datasets.py                    # Dataset loaders (documented)
├── constants.py                   # Configuration & hyperparameters (documented)
├── utils.py                       # Utility functions (documented)
├── augmentations.py               # Data augmentation transforms (documented)
│
├── CT_entities/                   # 🎯 Synthetic DRR generation
│   ├── README.md                  # Module documentation
│   ├── DRR_generator.py           # Main DRR pair generator (documented)
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
├── docs/                          # 📚 Detailed documentation
│   ├── README_training.md         # Training script documentation
│   ├── README_DRR_generator.md    # DRR generation documentation
│   ├── README_models.md           # Model architecture documentation
│   ├── README_datasets.md         # Dataset usage documentation
│   └── README_CT_entities.md      # CT entities module documentation
│
├── losses/                        # Custom loss functions
│   └── vgg_losses.py              # VGG perceptual loss
│
└── archive/                       # Archived/unused code
    └── README.md                  # Archive documentation
```

## Documentation

### Quick Reference

| Component | README | Description |
|-----------|--------|-------------|
| **Training** | [docs/README_training.md](docs/README_training.md) | Training script usage, parameters, outputs |
| **DRR Generator** | [docs/README_DRR_generator.md](docs/README_DRR_generator.md) | Synthetic data generation pipeline |
| **Models** | [docs/README_models.md](docs/README_models.md) | Neural network architectures |
| **Datasets** | [docs/README_datasets.md](docs/README_datasets.md) | Dataset classes and data loading |
| **CT Entities** | [docs/README_CT_entities.md](docs/README_CT_entities.md) | 3D pathological entity generation |

### Inline Documentation

All core files include comprehensive docstrings:

| File | Documentation |
|------|---------------|
| `constants.py` | Organized sections with parameter descriptions |
| `utils.py` | Full docstrings for all functions/classes |
| `augmentations.py` | Class and module-level documentation |
| `datasets.py` | Dataset class documentation with usage examples |
| `models.py` | Architecture documentation for all models |
| `longitudinal_MIM_training.py` | Usage and configuration documentation |
| `CT_entities/` | Separate README.md with usage guide |

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
