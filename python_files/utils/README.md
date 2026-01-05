# Utilities Module (`utils/`)

Shared utility functions used across the project. These functions eliminate code duplication and provide consistent implementations for common operations.

## Module Structure

```
utils/
├── __init__.py           # Re-exports all utilities
├── metrics.py            # Evaluation metrics (Dice, detection)
├── visualization.py      # Plotting and overlay functions
├── io_utils.py           # File I/O operations
├── schedulers.py         # Learning rate schedulers
├── image_processing.py   # Image manipulation functions
└── losses.py             # Custom loss functions
```

## Files Overview

### `metrics.py`
Evaluation metrics for segmentation and detection.

```python
from utils import dice_coefficient, calculate_detection_metrics

# Segmentation overlap
dice = dice_coefficient(prediction, ground_truth)

# Detection metrics
metrics = calculate_detection_metrics(pred_map, gt_map, threshold=0.5)
# Returns: precision, recall, F1, etc.
```

### `visualization.py`
Plotting functions for model outputs and comparisons.

```python
from utils import create_alpha_map, plot_difference_overlay

# Create opacity mask for overlays
alpha = create_alpha_map(difference_map, threshold=0.1)

# Plot baseline vs followup with difference overlay
plot_difference_overlay(baseline, followup, prediction, ground_truth)
```

### `io_utils.py`
File reading/writing utilities.

```python
from utils import (
    load_nifti, 
    save_nifti, 
    load_checkpoint,
    save_checkpoint
)

# NIfTI operations
image = load_nifti("scan.nii.gz")
save_nifti(image, "output.nii.gz")

# Checkpoint management
model, optimizer, epoch = load_checkpoint("model.pt", model, optimizer)
save_checkpoint(model, optimizer, epoch, "checkpoint.pt")
```

### `schedulers.py`
Learning rate scheduling implementations.

```python
from utils import get_linear_warmup_cosine_scheduler

scheduler = get_linear_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=100,
    min_lr=1e-6
)
```

### `image_processing.py`
Image manipulation and augmentation helpers.

```python
from utils import (
    crop_to_content,
    pad_to_square,
    apply_window_level
)

# Remove black borders
cropped = crop_to_content(cxr_image)

# Ensure square aspect ratio
squared = pad_to_square(cropped)

# Apply CT windowing (if applicable)
windowed = apply_window_level(ct_slice, window=400, level=50)
```

### `losses.py`
Custom loss functions for training.

```python
from utils import (
    DiceLoss,
    FocalLoss,
    PerceptualLoss,
    CombinedLoss
)

# Dice loss for segmentation
dice_loss = DiceLoss()

# Focal loss for imbalanced classification
focal_loss = FocalLoss(gamma=2.0)

# Perceptual/feature loss
perceptual_loss = PerceptualLoss(feature_extractor)
```

## Common Usage Patterns

### Training Loop
```python
from utils import (
    dice_coefficient,
    save_checkpoint,
    get_linear_warmup_cosine_scheduler
)

scheduler = get_linear_warmup_cosine_scheduler(optimizer, 5, 100)

for epoch in range(epochs):
    for batch in dataloader:
        # Training step...
        
    # Validation
    dice = dice_coefficient(predictions, targets)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, epoch, f"epoch_{epoch}.pt")
    scheduler.step()
```

### Visualization
```python
from utils import create_alpha_map, plot_difference_overlay

# After inference
pred = model(baseline, followup)

# Visualize results
alpha = create_alpha_map(pred, threshold=0.1)
plot_difference_overlay(baseline, followup, pred, ground_truth, alpha)
```

## Previously Duplicated Code

These utilities were extracted from multiple files to eliminate duplication:

| Function | Previously in | Occurrences |
|----------|--------------|-------------|
| `dice_coefficient` | utils.py, Observer_Variability.py, Prediction.py, extra/eval_masked_reconstruction.py | 4 |
| `create_alpha_map` | 11 different files | 11 |
| `crop_to_content` | utils.py, case_filtering.py, preprocessing.py | 3 |
