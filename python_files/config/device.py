"""
Device Configuration
====================

Automatic GPU/CPU device selection for PyTorch operations.

This module automatically selects the best available compute device:
1. If no GPU available: uses CPU
2. If single GPU: uses that GPU
3. If multiple GPUs: selects GPU with most free memory

Exports:
--------
DEVICE : str
    The selected device string (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')

Usage:
------
    from config.device import DEVICE
    
    model = MyModel().to(DEVICE)
    tensor = torch.zeros(10).to(DEVICE)
"""

import numpy as np
import torch

# Automatic device selection based on GPU availability and memory
cuda_dev_count = torch.cuda.device_count()

if not torch.cuda.is_available():
    # No GPU available - use CPU
    DEVICE = 'cpu'
elif cuda_dev_count == 1:
    # Single GPU available
    DEVICE = 'cuda'
else:
    # Multiple GPUs - select the one with most free memory
    free_memory = [torch.cuda.mem_get_info(k)[0] for k in range(cuda_dev_count)]
    best_gpu = np.argmax(free_memory).item()
    DEVICE = f'cuda:{best_gpu}'
    torch.cuda.set_device(DEVICE)

print(f'Device = {DEVICE}')
