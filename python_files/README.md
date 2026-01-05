# Longitudinal CXR Analysis

Deep learning pipeline for detecting and visualizing changes between longitudinal chest X-ray (CXR) pairs.

## Project Overview

This project implements a self-supervised learning approach for longitudinal CXR analysis, designed to:
1. Detect clinically significant changes between baseline (BL) and followup (FU) CXRs
2. Generate difference maps highlighting new findings, resolved findings, and progression
3. Support synthetic data generation using CT-derived DRRs (Digitally Reconstructed Radiographs)

## Project Structure

```
python_files/
├── config/                 # Centralized configuration
│   ├── device.py          # GPU selection
│   ├── paths.py           # Dataset paths
│   ├── model_config.py    # Architecture parameters
│   ├── training_config.py # Training hyperparameters
│   └── data_config.py     # Label definitions
│
├── utils/                  # Shared utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── visualization.py   # Plotting functions
│   ├── io_utils.py        # File I/O
│   ├── schedulers.py      # LR schedulers
│   ├── image_processing.py # Image transforms
│   └── losses.py          # Custom losses
│
├── core/                   # Core components
│   ├── data/              # Dataset implementations
│   │   ├── base.py        # BaseTransformDataset
│   │   ├── contrastive.py # Contrastive datasets
│   │   ├── classification.py
│   │   ├── longitudinal.py # BL-FU pair datasets
│   │   └── patch_reconstruction.py
│   │
│   └── models/            # Neural networks
│       ├── encoders.py    # EfficientNet encoder
│       ├── bottleneck.py  # Transformer + CNN bottleneck
│       ├── decoders.py    # Reconstruction decoders
│       ├── detection.py   # Classification heads
│       └── longitudinal.py # Full longitudinal models
│
├── data_prep/             # Data preprocessing
│   ├── io_operations.py   # Format conversion
│   ├── dataset_csv.py     # CSV generation
│   ├── pair_creation.py   # Longitudinal pair creation
│   └── image_processing.py
│
├── CT_entities/           # Synthetic abnormality generation
│   ├── DRR_generator.py   # DRR rendering from CT
│   ├── Consolidation.py   # Lung consolidation
│   ├── Pleural_Effusion.py
│   ├── Pneumothorax.py
│   └── ...
│
├── Evaluation/            # Evaluation scripts
│   ├── Prediction.py
│   └── Observer_Variability.py
│
└── extra/                 # Experimental scripts
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd python_files

# Create environment
conda create -n cxr_analysis python=3.10
conda activate cxr_analysis

# Install dependencies
pip install torch torchvision
pip install transformers nibabel pydicom
pip install matplotlib pandas tqdm scikit-learn
```

## Quick Start

### Training

```python
from core.models import LongitudinalMIMModel
from core.data import LongitudinalDataset
from config import DEVICE, BATCH_SIZE, MAX_LR

# Create model
model = LongitudinalMIMModel(dec=6).to(DEVICE)

# Create dataset
dataset = LongitudinalDataset(pairs_csv="train_pairs.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
for epoch in range(100):
    for batch in loader:
        bl = batch['baseline'].to(DEVICE)
        fu = batch['followup'].to(DEVICE)
        gt = batch['diff_map'].to(DEVICE)
        
        pred = model(bl, fu)
        loss = F.l1_loss(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference

```python
import torch
import nibabel as nib
from core.models import LongitudinalMIMModel
from utils import load_checkpoint

# Load model
model = LongitudinalMIMModel(dec=6)
model = load_checkpoint("checkpoint.pt", model)
model.eval()

# Load images
bl = torch.tensor(nib.load("baseline.nii.gz").get_fdata())[None, None]
fu = torch.tensor(nib.load("followup.nii.gz").get_fdata())[None, None]

# Predict
with torch.no_grad():
    change_map = model(bl / 255., fu / 255.)
    
# change_map values in [-1, 1]:
#   Positive: New findings in followup
#   Negative: Resolved findings
#   Zero: No change
```

## Model Architecture

```
Baseline Image ─┐
                ├─→ SharedEncoder ─→ Bottleneck ─→ FeatureDiff ─→ Decoder ─→ ChangeMap
Followup Image ─┘   (EfficientNet)   (ViT+Conv)    (FU - BL)     (Decoder6)
```

**Key Components:**
- **EfficientNetMiniEncoder**: Pretrained EfficientNet-B7 backbone (first 4 blocks)
- **BottleneckBlock**: Dual-branch (Transformer + CNN) for global and local features
- **Decoder6**: 6-stage upsampling decoder with Tanh output

## Datasets

Supported datasets:
- **MIMIC-CXR**: Primary longitudinal dataset
- **CXR-14 (NIH ChestX-ray14)**: Classification pretraining
- **PadChest**: Additional classification data
- **VinDr-CXR**: Validation dataset
- **Synthetic DRRs**: CT-derived training data

## Configuration

All settings are in `config/`:

```python
# config/training_config.py
BATCH_SIZE = 8
MAX_LR = 3e-4
USE_MASKED_LOSS = True
PERCEPTUAL_LOSS_WEIGHT = 0.1
```

## Evaluation

```python
from utils import dice_coefficient, calculate_detection_metrics

# Segmentation metrics
dice = dice_coefficient(prediction, ground_truth)

# Detection metrics
metrics = calculate_detection_metrics(pred, gt, threshold=0.5)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
```

## Citation

If you use this code, please cite:
```
@article{longitudinal_cxr_2024,
  title={Longitudinal Chest X-Ray Analysis for Change Detection},
  author={...},
  year={2024}
}
```

## License

This project is for research purposes only.
