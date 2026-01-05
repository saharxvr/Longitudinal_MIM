"""
Patch Reconstruction Dataset
============================

Dataset for masked patch reconstruction pretraining.
"""

import random
import torch
import nibabel as nib
from glob import glob
from typing import List

from core.data.base import BaseTransformDataset


class PatchReconstructionDataset(BaseTransformDataset):
    """
    Dataset for self-supervised patch reconstruction.
    
    Loads images from multiple folders for masked image modeling
    pretraining where random patches are masked and reconstructed.
    
    Parameters
    ----------
    data_folders : list[str]
        List of folder paths containing NIfTI images.
        Will recursively search for *.nii.gz files.
    rot_chance : float
        Probability of 90/180/270 degree rotation.
    hor_flip_chance : float
        Probability of horizontal flip.
    ver_flip_chance : float
        Probability of vertical flip.
    """
    
    def __init__(
        self,
        data_folders: List[str],
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        self.file_paths = []
        for folder in data_folders:
            self.file_paths.extend(
                glob(f'{folder}/**/*.nii.gz', recursive=True)
            )
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and transform a single image.
        
        Returns
        -------
        torch.Tensor
            Image tensor, shape (1, H, W).
        """
        item = nib.load(self.file_paths[idx]).get_fdata()[None, ...]
        item = self.apply_transforms(item)
        return item
