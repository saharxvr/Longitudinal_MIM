"""
Model Utility Functions.

Provides helper functions for:
- Parameter counting and model inspection
- Selective weight freezing for transfer learning
- Weight verification and debugging

These utilities support the training workflow by enabling
fine-grained control over which model components are trainable.
"""

from typing import Iterable, Union
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model to analyze
        trainable_only: If True, count only trainable parameters.
                       If False, count all parameters.
    
    Returns:
        Total number of parameters (trainable or all)
    
    Example:
        >>> model = EfficientNetMiniEncoder()
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> total = count_parameters(model, trainable_only=False)
        >>> print(f"Trainable: {trainable:,}, Total: {total:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_and_unfreeze(
    to_freeze: Iterable[Union[nn.Module, nn.Parameter]],
    to_unfreeze: Iterable[Union[nn.Module, nn.Parameter]]
) -> None:
    """
    Selectively freeze and unfreeze model components.
    
    Used for transfer learning and staged training where different
    parts of the model should be trainable at different phases.
    
    Args:
        to_freeze: Modules or parameters to set requires_grad=False
        to_unfreeze: Modules or parameters to set requires_grad=True
    
    Example:
        >>> # During fine-tuning, freeze encoder and train decoder
        >>> freeze_and_unfreeze(
        ...     to_freeze=[model.encoder],
        ...     to_unfreeze=[model.decoder, model.bottleneck]
        ... )
        
        >>> # For contrastive pre-training
        >>> model.detection_stage()  # Uses this internally
    
    Note:
        Handles both nn.Module (iterates parameters) and nn.Parameter objects
    """
    for module_or_param in to_freeze:
        if isinstance(module_or_param, nn.Module):
            for param in module_or_param.parameters():
                param.requires_grad = False
        elif isinstance(module_or_param, nn.Parameter):
            module_or_param.requires_grad = False
            
    for module_or_param in to_unfreeze:
        if isinstance(module_or_param, nn.Module):
            for param in module_or_param.parameters():
                param.requires_grad = True
        elif isinstance(module_or_param, nn.Parameter):
            module_or_param.requires_grad = True


def weights_check(model: nn.Module, verbose: bool = False) -> dict:
    """
    Check and summarize model weights for debugging.
    
    Useful for verifying that weights are properly initialized,
    loaded from checkpoints, or updated during training.
    
    Args:
        model: Model to inspect
        verbose: If True, print detailed statistics for each layer
    
    Returns:
        Dictionary with weight statistics:
        - 'total_params': Total parameter count
        - 'trainable_params': Trainable parameter count
        - 'frozen_params': Frozen parameter count
        - 'layers': List of layer info dicts
    
    Example:
        >>> stats = weights_check(model, verbose=True)
        >>> print(f"Frozen: {stats['frozen_params']:,}")
    """
    stats = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'layers': []
    }
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        stats['total_params'] += num_params
        
        layer_info = {
            'name': name,
            'shape': list(param.shape),
            'params': num_params,
            'trainable': param.requires_grad,
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
        }
        stats['layers'].append(layer_info)
        
        if param.requires_grad:
            stats['trainable_params'] += num_params
        else:
            stats['frozen_params'] += num_params
            
        if verbose:
            status = "trainable" if param.requires_grad else "frozen"
            print(f"{name}: {list(param.shape)}, {num_params:,} params ({status})")
            print(f"  mean={layer_info['mean']:.6f}, std={layer_info['std']:.6f}")
    
    return stats
