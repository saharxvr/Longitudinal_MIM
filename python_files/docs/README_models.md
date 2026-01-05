# Models Architecture

## Overview

`models.py` defines the neural network architectures for longitudinal chest X-ray change detection. The main model (`LongitudinalMIMModelBig`) processes paired baseline and followup images to predict pixel-wise difference maps.

## Model Hierarchy

```
models.py
├── Building Blocks
│   ├── ChannelExpansionLayer/Block
│   ├── EfficientNetMiniEncoder
│   ├── FeatureEmbeddings
│   ├── TransformerBranch
│   ├── BottleneckBlock
│   └── DecoderBlock
│
├── Encoders
│   ├── BottleneckEncoder
│   └── ContrastiveEncoder
│
├── Main Models
│   ├── LongitudinalMIMModel
│   └── LongitudinalMIMModelBig (recommended)
│
└── Auxiliary Models
    ├── MaskedReconstructionModel
    ├── ClassificationModel
    └── Discriminator (for GAN training)
```

## Main Model: LongitudinalMIMModelBig

### Architecture Overview

```
Input: (baseline, followup) → both [B, 1, 512, 512]
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 │
┌─────────────┐  ┌─────────────┐        │
│   Encoder   │  │   Encoder   │  (shared weights)
│  Baseline   │  │  Followup   │        │
└──────┬──────┘  └──────┬──────┘        │
       │                │               │
       ▼                ▼               │
  [B, 640, 32, 32]  [B, 640, 32, 32]   │
       │                │               │
       └───────┬────────┘               │
               ▼                        │
    ┌─────────────────────┐             │
    │  Difference Fusion  │             │
    │  (concat + compress)│             │
    └──────────┬──────────┘             │
               ▼                        │
         [B, 640, 32, 32]               │
               │                        │
               ▼                        │
    ┌─────────────────────┐             │
    │      Decoder        │◄────────────┘
    │   (U-Net style)     │   skip connections
    └──────────┬──────────┘
               ▼
    Output: diff_map [B, 1, 512, 512]
```

### Components

#### 1. Dual Encoder (Shared Weights)
- **EfficientNet-B7** feature extractor (select blocks)
- **ViT-Base** transformer encoder
- Processes both images through identical network

#### 2. Difference Fusion
- Concatenates baseline and followup features
- Multiple compression paths:
  - Depthwise convolution (1x1)
  - Standard convolution (3x3)
  - Depthwise separable spatial (7x7)

#### 3. U-Net Decoder
- Progressive upsampling (32 → 64 → 128 → 256 → 512)
- Skip connections from encoder
- Final sigmoid activation for [0, 1] output

### Usage

```python
from models import LongitudinalMIMModelBig
from constants import DEVICE

# Initialize model
model = LongitudinalMIMModelBig().to(DEVICE)

# Forward pass
baseline = torch.randn(4, 1, 512, 512).to(DEVICE)  # [B, C, H, W]
followup = torch.randn(4, 1, 512, 512).to(DEVICE)

diff_pred = model(baseline, followup)  # [4, 1, 512, 512]
```

## Building Blocks

### ChannelExpansionLayer
Convolutional layer that expands channel dimensions with normalization.

```python
# Expands from in_channels to in_channels * out_multiplier
layer = ChannelExpansionLayer(in_channels=64, out_multiplier=2)
# Output: 128 channels
```

### EfficientNetMiniEncoder
Lightweight EfficientNet-B7 encoder using selected blocks.

```python
encoder = EfficientNetMiniEncoder(
    block_idxs=(0, 4, 11, 18, 28, 38, 51, 52),  # Selected blocks
    inp=1,           # Input channels
    exp=8.,          # First expansion ratio
    exp2=8.,         # Second expansion ratio
    dropout=False
)
```

### TransformerBranch
ViT-based transformer branch with down/up sampling.

```python
# Input: [B, 640, 32, 32] → Output: [B, 640, 32, 32]
transformer_branch = TransformerBranch(vit_encoder)
```

### BottleneckBlock
Combines transformer and convolutional branches.

```python
# Parallel processing + feature compression
bottleneck = BottleneckBlock(transformer_encoder)
```

### DecoderBlock
Single upsampling block with deconvolution.

```python
decoder_block = DecoderBlock(
    in_channels=640,
    out_channels=320,
    deconv_kernel_size=2,
    deconv_stride=2,
    sigmoid=False  # True for final block
)
```

## Configuration Constants

| Constant | Default | Description |
|----------|---------|-------------|
| `IMG_SIZE` | 512 | Input image size |
| `FEATURE_SIZE` | 32 | Encoded feature map size |
| `FEATURE_CHANNELS` | 640 | Encoder output channels |
| `HIDDEN_CHANNELS` | 768 | Transformer hidden size |
| `INTER_CHANNELS` | 704 | Intermediate channels |
| `PATCHES_NUM` | 256 | Number of ViT patches (16×16) |
| `GROUP_NORM_GROUPS` | 32 | Groups for GroupNorm |
| `INIT_STD` | 0.02 | Weight init std deviation |

## Alternative Models

### LongitudinalMIMModel (Smaller)
Simpler architecture without full U-Net decoder. Faster but less accurate.

```python
from models import LongitudinalMIMModel
model = LongitudinalMIMModel()
```

### MaskedReconstructionModel
Baseline model for masked image modeling pretraining.

```python
from models import MaskedReconstructionModel
model = MaskedReconstructionModel(in_channels=1)
```

### ClassificationModel
Multi-label classification head on top of encoder.

```python
from models import ClassificationModel
model = ClassificationModel(in_channels=1, labels_num=14)
```

### Discriminator
PatchGAN discriminator for adversarial training.

```python
from models import Discriminator
disc = Discriminator()
```

## Weight Initialization

All models support custom weight initialization:

```python
# Initialization is controlled by INIT_WEIGHTS constant
# Default: True (Kaiming/truncated normal init)

# Key initialization strategies:
# - Conv layers: Kaiming normal (fan_in, relu)
# - Norm layers: weight=1, bias=0
# - Embeddings: Truncated normal (std=INIT_STD)
```

## Memory Footprint

| Model | Parameters | GPU Memory (BS=4) |
|-------|------------|-------------------|
| LongitudinalMIMModelBig | ~180M | ~12 GB |
| LongitudinalMIMModel | ~120M | ~8 GB |
| MaskedReconstructionModel | ~100M | ~6 GB |

## Dependencies

```python
import torch
import torch.nn as nn
from transformers import (
    EfficientNetModel, 
    EfficientNetConfig,
    ViTEncoder, 
    ViTConfig
)
```

## Model Checkpointing

### Save Model
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')
```

### Load Model
```python
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Tips for Training

1. **Gradient Accumulation**: Use `UPDATE_EVERY_BATCHES` to simulate larger batch sizes
2. **Mixed Precision**: Model supports FP16 training with `torch.cuda.amp`
3. **Encoder Freezing**: Can freeze encoder weights for fine-tuning:
   ```python
   for param in model.encoder.parameters():
       param.requires_grad = False
   ```

## Architecture Diagrams

### Encoder Detail
```
Input [1, 512, 512]
    │
    ▼
ChannelExpansionBlock (1 → 64 → 512)
    │
    ▼
EfficientNet Blocks (select 8 blocks)
    │
    ▼
SiLU Activation
    │
    ▼
Output [640, 32, 32]
```

### Decoder Detail
```
Input [640, 32, 32]
    │
    ▼
DecoderBlock [640 → 320, 64×64]
    │ + skip
    ▼
DecoderBlock [320 → 160, 128×128]
    │ + skip
    ▼
DecoderBlock [160 → 80, 256×256]
    │ + skip
    ▼
DecoderBlock [80 → 40, 512×512]
    │
    ▼
Conv1x1 [40 → 1]
    │
    ▼
Sigmoid
    │
    ▼
Output [1, 512, 512]
```
