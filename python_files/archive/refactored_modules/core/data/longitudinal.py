"""
Longitudinal MIM Dataset
========================

Dataset for longitudinal (paired baseline/follow-up) image analysis
with difference map prediction.
"""

import os
import random
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
from random import shuffle
from typing import Optional, List, Tuple
import torchvision.transforms.v2 as v2
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation

from config import IMG_SIZE


class LongitudinalMIMDataset(Dataset):
    """
    Dataset for longitudinal chest X-ray analysis.
    
    Loads paired baseline/follow-up images with difference maps
    for training models to detect changes between sequential CXRs.
    
    Supports multiple data sources:
    - Entity-based synthetic pairs (CT_entities generated)
    - Inpainted pairs (diffusion-based)
    - DRR single pairs (same CT, different projections)
    - DRR pairs (synthetic pathology pairs)
    
    Parameters
    ----------
    entity_dirs : list[str]
        Directories with entity-based synthetic images.
    inpaint_dirs : list[str]
        Directories with inpainted image pairs.
    DRR_single_dirs : list[str]
        Directories with DRR variations from single CT.
    DRR_pair_dirs : list[str]
        Directories with synthetic pathology DRR pairs.
    abnor_both_p : float
        Probability to add abnormalities to both images.
    invariance : str, optional
        Type of invariance ('abnormality' or 'devices').
    overlay_diff_p : float
        Probability to overlay difference in output.
    """
    
    def __init__(
        self,
        entity_dirs: List[str],
        inpaint_dirs: List[str],
        DRR_single_dirs: List[str],
        DRR_pair_dirs: List[str],
        abnor_both_p: float = 0.5,
        invariance: Optional[str] = None,
        overlay_diff_p: float = 0.9
    ):
        self.paths = []
        self.abnor_both_p = abnor_both_p
        self.overlay_diff_p = overlay_diff_p
        
        # Collect entity pairs
        for entity_dir in entity_dirs:
            seg_dir = f"{entity_dir}_segs"
            self.paths.extend([
                (f'{entity_dir}/{n}', f'{seg_dir}/{n.split(".")[0]}_seg.nii.gz')
                for n in os.listdir(entity_dir)
            ])
        
        # Collect inpaint pairs
        for inpaint_dir in inpaint_dirs:
            for p in os.listdir(inpaint_dir):
                self.paths.append((
                    f'{inpaint_dir}/{p}/im1.nii.gz',
                    f'{inpaint_dir}/{p}/im2.nii.gz',
                    f'{inpaint_dir}/{p}/seg1.nii.gz',
                    f'{inpaint_dir}/{p}/seg2.nii.gz',
                    f'{inpaint_dir}/{p}/difference_map.nii.gz'
                ))
        
        # Collect DRR single pairs
        for DRR_dir in DRR_single_dirs:
            for case_dir in os.listdir(DRR_dir):
                dir_path = f'{DRR_dir}/{case_dir}'
                self.paths.extend([
                    (f'{dir_path}/var{i}.nii.gz', f'{dir_path}/var{i+1}.nii.gz')
                    for i in range(600)
                ])
        
        # Collect DRR pairs with difference maps
        for DRR_dir in DRR_pair_dirs:
            for case_dir in os.listdir(DRR_dir):
                case_dir_abs = f'{DRR_dir}/{case_dir}'
                if not os.path.isdir(case_dir_abs):
                    continue
                for pair_dir in os.listdir(case_dir_abs):
                    dir_path = f'{case_dir_abs}/{pair_dir}'
                    
                    # Check for difference map
                    diff_path = None
                    for diff_name in ['diff_map.nii.gz', 'difference_map.nii.gz']:
                        if os.path.exists(f'{dir_path}/{diff_name}'):
                            diff_path = f'{dir_path}/{diff_name}'
                            break
                    
                    if diff_path:
                        self.paths.append((
                            f'{dir_path}/prior.nii.gz',
                            f'{dir_path}/current.nii.gz',
                            diff_path
                        ))
        
        self.t_len = len(self.paths)
        
        # Setup transforms (lazy imports to avoid circular dependencies)
        self._setup_transforms(invariance)
        
        self.resize_dict = {
            512: v2.Resize((512, 512)),
            768: v2.Resize((768, 768))
        }
        
        self.return_label = False
        self.target_ornt = axcodes2ornt(('R', 'A', 'S'))
    
    def _setup_transforms(self, invariance: Optional[str]):
        """Setup augmentation transforms."""
        # Import here to avoid circular imports
        try:
            from augmentations import (
                RandomAbnormalizationTransform,
                RandomAffineWithMaskTransform,
                RandomBsplineAndSimilarityWithMaskTransform,
                CropResizeWithMaskTransform,
                RescaleValuesTransform,
                RandomIntensityTransform,
                RandomFlipBLWithFU,
                RandomChannelsFlip
            )
            
            if invariance is None:
                self.random_abnormalization_tf = RandomAbnormalizationTransform(
                    lung_abnormalities=True, devices=True, size=768,
                    none_chance_to_update=0.13
                )
                self.has_inv = False
            elif invariance == 'abnormality':
                self.random_abnormalization_tf = RandomAbnormalizationTransform(
                    lung_abnormalities=False, devices=True, size=768,
                    none_chance_to_update=0.75
                )
                self.random_abnormalization_tf_inv = RandomAbnormalizationTransform(
                    lung_abnormalities=True, devices=False, size=512,
                    none_chance_to_update=0.13
                )
                self.has_inv = True
            elif invariance == 'devices':
                self.random_abnormalization_tf = RandomAbnormalizationTransform(
                    lung_abnormalities=True, devices=False, size=768,
                    none_chance_to_update=0.13
                )
                self.random_abnormalization_tf_inv = RandomAbnormalizationTransform(
                    lung_abnormalities=False, devices=True, size=512,
                    none_chance_to_update=0.75
                )
                self.has_inv = True
            else:
                self.has_inv = False
            
            self.random_affine_tf = RandomAffineWithMaskTransform()
            self.random_bspline_tf = RandomBsplineAndSimilarityWithMaskTransform()
            self.random_affine_tf_inpaint = RandomAffineWithMaskTransform(
                scale_x_p=0.25, scale_y_p=0.25, trans_x_p=0.25, trans_y_p=0.25
            )
            self.random_bspline_tf_inpaint = RandomBsplineAndSimilarityWithMaskTransform(
                rot_p=0.3, scale_y_p=0.175, scale_x_p=0.175,
                trans_y_p=0.175, trans_x_p=0.175
            )
            self.crop_resize_with_mask_tf = CropResizeWithMaskTransform()
            self.rescale_values_tf = RescaleValuesTransform()
            self.random_intensity_tf = RandomIntensityTransform(
                clahe_p=0.25, clahe_clip_limit=(0.75, 2.5),
                blur_p=0., jitter_p=0.35
            )
            self.fu_tf = v2.RandomChoice(
                [self.random_affine_tf, self.random_bspline_tf],
                p=[0.2, 0.8]
            )
            self.bl_tf = v2.RandomChoice(
                [self.crop_resize_with_mask_tf, self.random_bspline_tf, self.random_affine_tf],
                p=[0.85, 0.1, 0.05]
            )
            self.tf_inpaint = v2.RandomChoice(
                [self.crop_resize_with_mask_tf, self.random_bspline_tf_inpaint, self.random_affine_tf_inpaint],
                p=[0.3, 0.15, 0.55]
            )
            self.random_bl_fu_flip_tf = RandomFlipBLWithFU(p=0.5)
            self.random_channels_flip_tf = RandomChannelsFlip(p=0.5)
            self._transforms_loaded = True
            
        except ImportError:
            self._transforms_loaded = False
    
    def __len__(self) -> int:
        return self.t_len
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load NIfTI image as tensor."""
        return torch.tensor(nib.load(path).get_fdata().T[None, ...])
    
    def get_DRR_pair(self, paths: Tuple[str, ...]) -> Tuple[torch.Tensor, ...]:
        """Load DRR image pair with difference map."""
        im1_path, im2_path, diff_path = paths
        
        im1 = self._load_image(im1_path)
        im2 = self._load_image(im2_path)
        seg2 = torch.ones_like(im2, dtype=torch.bool)
        diff_map = self._load_image(diff_path)
        
        return im1, im2, diff_map, seg2
    
    def get_DRR_single_pair(self, paths: Tuple[str, str]) -> Tuple[torch.Tensor, ...]:
        """Load DRR pair without difference (same CT, different variations)."""
        im1_path, im2_path = paths
        
        im1 = self._load_image(im1_path)
        im2 = self._load_image(im2_path)
        seg2 = torch.ones_like(im2, dtype=torch.bool)
        diff_map = torch.zeros_like(im1, dtype=torch.float)
        
        return im1, im2, diff_map, seg2
    
    def __getitem__(self, idx: int):
        """
        Load and return a training sample.
        
        Returns
        -------
        bl : torch.Tensor
            Baseline (prior) image.
        fu : torch.Tensor
            Follow-up (current) image.
        gt : torch.Tensor
            Ground truth difference map.
        fu_mask : torch.Tensor
            Mask for follow-up image.
        """
        paths = self.paths[idx]
        
        if len(paths) == 3:
            bl, fu, gt, fu_mask = self.get_DRR_pair(paths)
        elif len(paths) == 2:
            bl, fu, gt, fu_mask = self.get_DRR_single_pair(paths)
        else:
            # Default to DRR pair handling
            bl, fu, gt, fu_mask = self.get_DRR_pair(paths[:3])
        
        if self.return_label:
            return bl, fu, gt, 0, 0
        
        return bl, fu, gt, fu_mask
    
    def shuffle(self):
        """Shuffle paths for new epoch."""
        shuffle(self.paths)
    
    def reorient_to_standard(self, nifti_img) -> np.ndarray:
        """
        Reorient NIfTI image to standard RAS orientation.
        
        Parameters
        ----------
        nifti_img : nib.Nifti1Image
            Input NIfTI image.
            
        Returns
        -------
        np.ndarray
            Reoriented image data.
        """
        data = nifti_img.get_fdata()
        affine = nifti_img.affine
        
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        current_ornt = nib.orientations.io_orientation(affine)
        transform = ornt_transform(current_ornt, self.target_ornt)
        reoriented_data = apply_orientation(data, transform)
        
        return np.squeeze(reoriented_data)
