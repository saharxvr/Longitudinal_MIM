# Core Module (`core/`)

Core components of the Longitudinal CXR Analysis pipeline. Contains data loading and neural network architectures.

## Module Structure

```
core/
├── data/                 # Dataset implementations
│   ├── __init__.py
│   ├── base.py          # BaseTransformDataset
│   ├── contrastive.py   # Contrastive learning datasets
│   ├── classification.py # Classification datasets
│   ├── longitudinal.py  # Longitudinal pair datasets
│   └── patch_reconstruction.py  # Masked reconstruction datasets
│
└── models/              # Neural network architectures
    ├── __init__.py
    ├── utils.py         # Model utilities
    ├── blocks.py        # Building blocks
    ├── encoders.py      # Encoder architectures
    ├── decoders.py      # Decoder architectures
    ├── bottleneck.py    # Bottleneck modules
    ├── detection.py     # Classification models
    └── longitudinal.py  # Full longitudinal models
```

## Data Module (`core/data/`)

### Base Classes

```python
from core.data import BaseTransformDataset

class MyDataset(BaseTransformDataset):
    def __getitem__(self, idx):
        # Load data...
        image = self.apply_transforms(image)  # Inherited method
        return image
```

### Available Datasets

| Dataset | Purpose | Input Format |
|---------|---------|--------------|
| `CXRContrastiveDataset` | Self-supervised pretraining | Single CXR |
| `CXRClassificationDataset` | Multi-label classification | CXR + labels |
| `LongitudinalDataset` | Change detection training | BL/FU pair + diff map |
| `PatchReconstructionDataset` | Masked image modeling | CXR with masking |

### Usage Example

```python
from core.data import LongitudinalDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = LongitudinalDataset(
    pairs_csv="train_pairs.csv",
    transform=train_transforms,
    return_segmentation=True
)

# Create dataloader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    baseline = batch['baseline']
    followup = batch['followup']
    diff_map = batch['diff_map']
```

## Models Module (`core/models/`)

### Architecture Overview

```
Input Image
    │
    ▼
┌─────────────────────┐
│  EfficientNetMini   │  Pretrained EfficientNet-B7 backbone
│     Encoder         │  Output: (B, 640, 32, 32)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Bottleneck        │  CNN + Transformer branches
│   (ViT + Conv)      │  Global + Local features
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Decoder           │  Multi-scale upsampling
│   (Decoder6)        │  Output: (B, 1, 512, 512)
└─────────────────────┘
```

### Key Components

#### Encoders
```python
from core.models import EfficientNetMiniEncoder

encoder = EfficientNetMiniEncoder(
    inp=1,           # Input channels (1 for grayscale)
    dropout=True,    # Enable spatial dropout
    block_idxs=(0, 1, 2, 3)  # Which EfficientNet blocks
)

features = encoder(image)  # (B, 640, 32, 32)
```

#### Bottleneck
```python
from core.models import BottleneckEncoder

bottleneck = BottleneckEncoder(in_channels=1)
features, encoded = bottleneck(image, get_encoded=True)
```

#### Decoders
```python
from core.models import Decoder6

decoder = Decoder6()
output = decoder(features)  # (B, 1, 512, 512), range [-1, 1]
```

#### Full Models
```python
from core.models import LongitudinalMIMModel

# Main model for change detection
model = LongitudinalMIMModel(
    dec=6,              # Use Decoder6
    use_pos_embed=True, # Add positional embeddings
    init_weights=True   # Custom weight initialization
)

# Forward pass
baseline = torch.randn(2, 1, 512, 512)
followup = torch.randn(2, 1, 512, 512)
change_map = model(baseline, followup)  # (2, 1, 512, 512)
```

### Model Variants

| Model | Purpose | Notes |
|-------|---------|-------|
| `LongitudinalMIMModel` | Primary model | "MODEL THAT WORKED FOR id9" |
| `LongitudinalMIMModelBig` | Extended model | Dual bottleneck paths |
| `LongitudinalMIMModelBigTransformer` | Transformer variant | Added transformer branches |
| `LongitudinalMIMModelTest` | Lightweight test | No pretrained weights |
| `MaskedReconstructionModel` | Self-supervised pretraining | MAE-style reconstruction |
| `DetectionContrastiveModel` | Classification | Multi-label prediction |

### Training Stages

For `DetectionContrastiveModel`:

```python
model = DetectionContrastiveModel(in_channels=1)

# Stage 1: Detection only (freeze encoder)
model.detection_stage()

# Stage 2: Contrastive + Detection
model.contrastive_detection_stage()
```

## Utilities

### Model Utils
```python
from core.models import count_parameters, freeze_and_unfreeze

# Count parameters
total = count_parameters(model, trainable_only=False)
trainable = count_parameters(model, trainable_only=True)

# Selective freezing for transfer learning
freeze_and_unfreeze(
    to_freeze=[model.encoder],
    to_unfreeze=[model.decoder]
)
```
