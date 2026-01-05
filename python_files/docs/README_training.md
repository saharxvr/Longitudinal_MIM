# Longitudinal MIM Training

## Overview

`longitudinal_MIM_training.py` is the main training script for the Longitudinal Masked Image Modeling (MIM) model. This script trains a model to predict pixel-wise difference maps between baseline and followup chest X-rays.

## Usage

### Basic Training
```bash
python longitudinal_MIM_training.py
```

### Command-Line Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--checkpoint` | `-c` | str | None | Path to checkpoint file to resume training |
| `--save_name` | `-n` | str | Auto | Custom name for model save directory |

### Example
```bash
# Start fresh training with custom save name
python longitudinal_MIM_training.py -n my_experiment_v1

# Resume from checkpoint
python longitudinal_MIM_training.py -c ./checkpoints/longitudinal_mim_epoch_5.pt
```

## Configuration

### Key Constants (constants.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 4 | Samples per batch |
| `UPDATE_EVERY_BATCHES` | 12 | Gradient accumulation steps |
| `LONGITUDINAL_MIM_EPOCHS` | 10 | Total training epochs |
| `MAX_LR` | 6e-4 | Maximum learning rate |
| `WEIGHT_DECAY` | 1e-2 | AdamW weight decay |
| `IMG_SIZE` | 512 | Input image size |

### Loss Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_L1` | True | L1 reconstruction loss |
| `USE_L2` | True | L2 reconstruction loss |
| `USE_SSIM` | False | SSIM loss |
| `USE_PERC_STYLE` | False | Perceptual/Style loss |
| `USE_FOURIER` | False | Fourier domain loss |
| `USE_GAN` | False | GAN adversarial loss |

### Loss Weights

| Weight | Default | Description |
|--------|---------|-------------|
| `LAMBDA_L1_ALL` | 1.0 | L1 loss on all pixels |
| `LAMBDA_L1_MASKED` | 1.0 | L1 loss on masked regions |
| `LAMBDA_L2` | 1.0 | L2 loss weight |
| `LAMBDA_SSIM` | 1.5 | SSIM loss weight |
| `LAMBDA_P` | 2.0 | Perceptual loss weight |
| `LAMBDA_S` | 2.0 | Style loss weight |

## Output Structure

```
output_dir/
├── longitudinal_mim_<timestamp>/
│   ├── longitudinal_mim_best.pt      # Best validation loss checkpoint
│   ├── longitudinal_mim_epoch_N.pt   # Checkpoints per epoch
│   ├── training_params.json          # Saved hyperparameters
│   ├── loss_curves.png               # Training/validation loss plot
│   └── detailed_loss_curves.png      # Per-component loss breakdown
```

## Saved Outputs

### 1. Model Checkpoints
Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state
- `epoch`: Epoch number
- `train_loss` / `val_loss`: Loss values

### 2. Training Parameters (training_params.json)
Automatically saves ~40+ hyperparameters including:
- Learning rate, batch size, epochs
- Loss function configuration
- Model architecture settings
- Data augmentation settings
- Dataset paths and sizes

### 3. Loss Curve Plots
- **loss_curves.png**: Combined train/val loss over epochs
- **detailed_loss_curves.png**: Per-component losses (L1, L2, SSIM, etc.)

## Training Flow

```
1. Load Dataset
   └── LongitudinalMIMDataset (baseline, followup, diff_map pairs)

2. Initialize Model
   └── LongitudinalMIMModelBig (EfficientNet + ViT encoder, U-Net decoder)

3. Training Loop (per epoch)
   ├── Training Phase
   │   ├── Forward pass: diff_pred = model(baseline, followup)
   │   ├── Compute losses (L1, L2, SSIM, etc.)
   │   ├── Gradient accumulation
   │   └── Optimizer step
   │
   ├── Validation Phase
   │   └── Evaluate on held-out set
   │
   └── Save Checkpoint & Plots

4. Final Output
   └── Best model, loss curves, parameters JSON
```

## Data Requirements

The training script expects data in the following format:
- **Input**: Pairs of chest X-rays (baseline + followup)
- **Target**: Ground truth difference maps
- **Format**: NIfTI (.nii.gz) files
- **Size**: 512x512 pixels

### Dataset Structure
```
data/
├── case_001/
│   ├── prior.nii.gz        # Baseline image
│   ├── current.nii.gz      # Followup image
│   └── diff_map.nii.gz     # Ground truth difference
├── case_002/
│   └── ...
```

## Model Architecture

The training uses `LongitudinalMIMModelBig` which consists of:

1. **Dual Encoder**: Processes baseline and followup separately
   - EfficientNet-B7 feature extractor
   - ViT-Base transformer encoder

2. **Difference Attention**: Cross-attention between temporal features

3. **U-Net Decoder**: Generates difference map prediction
   - Skip connections from encoder
   - Progressive upsampling

## Dependencies

```
torch >= 1.12
transformers (HuggingFace)
nibabel
numpy
matplotlib
kornia
```

## Monitoring Training

### TensorBoard (if enabled)
```bash
tensorboard --logdir=./runs
```

### Console Output
Training prints per-batch and per-epoch statistics:
```
Epoch 1/10
[100/500] Train Loss: 0.0234 | L1: 0.012 | L2: 0.008 | SSIM: 0.003
...
Epoch 1 Complete: Train Loss: 0.0198 | Val Loss: 0.0187
Saved checkpoint: longitudinal_mim_epoch_1.pt
```

## Common Issues

### Out of Memory
- Reduce `BATCH_SIZE` in constants.py
- Increase `UPDATE_EVERY_BATCHES` to maintain effective batch size

### Slow Convergence
- Increase `MAX_LR` slightly
- Ensure `USE_L1` and `USE_L2` are both enabled

### Validation Loss Not Decreasing
- Check dataset quality
- Try enabling `USE_SSIM` for better perceptual quality
