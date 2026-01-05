"""
Base Dataset Classes
====================

Shared functionality for all dataset classes including
common transforms and utility functions.

Usage:
------
    from core.data.base import BaseTransformDataset, get_proj_mat
    
    class MyDataset(BaseTransformDataset):
        def __getitem__(self, idx):
            item = self.load_item(idx)
            return self.apply_transforms(item)
"""

import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
from typing import Optional


class BaseTransformDataset(Dataset):
    """
    Base dataset class with common transform functionality.
    
    Provides rotation, horizontal flip, and vertical flip transforms
    that are shared across most dataset classes in the project.
    
    Parameters
    ----------
    rot_chance : float, default=0.075
        Probability of applying 90/180/270 degree rotation.
    hor_flip_chance : float, default=0.03
        Probability of horizontal flip.
    ver_flip_chance : float, default=0.03
        Probability of vertical flip.
        
    Subclasses should implement:
    - __len__(): Return dataset length
    - __getitem__(idx): Load and return item (should call apply_transforms)
    """
    
    def __init__(
        self,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__()
        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_chance)
    
    def apply_transforms(self, item: torch.Tensor) -> torch.Tensor:
        """
        Apply random geometric transforms to image.
        
        Parameters
        ----------
        item : torch.Tensor or array-like
            Input image, will be converted to tensor if needed.
            
        Returns
        -------
        torch.Tensor
            Transformed image.
        """
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item)
        
        # Random 90-degree rotation
        if torch.rand(1).item() < self.rot_chance:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        
        # Random flips
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        
        return item
    
    def __len__(self) -> int:
        raise NotImplementedError("Subclass must implement __len__")
    
    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclass must implement __getitem__")


def get_standard_vector(size: int, idx: int) -> torch.Tensor:
    """
    Create a one-hot boolean vector.
    
    Parameters
    ----------
    size : int
        Vector length.
    idx : int
        Index to set to True.
        
    Returns
    -------
    torch.Tensor
        Boolean tensor with True at idx position.
    """
    vec = torch.zeros(size, dtype=torch.bool)
    vec[idx] = True
    return vec


def get_proj_mat(size: int, ds_labels: list, all_labels: list) -> torch.Tensor:
    """
    Create projection matrix for mapping dataset labels to common labels.
    
    Used to project from a common label set (e.g., CXR-14's 14 labels)
    to a dataset-specific subset.
    
    Parameters
    ----------
    size : int
        Size of the common label space.
    ds_labels : list
        Dataset-specific labels.
    all_labels : list
        Full list of all possible labels.
        
    Returns
    -------
    torch.Tensor
        Boolean projection matrix, shape (len(ds_labels), size).
    """
    if ds_labels == all_labels:
        return torch.eye(size, dtype=torch.bool)
    
    mat = []
    for label in ds_labels:
        idx = all_labels.index(label)
        mat.append(get_standard_vector(size, idx))
    
    return torch.stack(mat)
