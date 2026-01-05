"""
Classification Datasets
=======================

Datasets for normal/abnormal and no-finding classification tasks.

Classes:
--------
- NoFindingDataset: Multi-source 'No Finding' images
- SmallNoFindingDataset: Single-folder normal images
- ExpertNoFindingDataset: Expert-annotated normal images
- NormalAbnormalDataset: Binary classification from multiple sources
"""

import os
import pandas as pd
import torch
import nibabel as nib
from random import shuffle, Random
from typing import Optional, Tuple, List

from core.data.base import BaseTransformDataset
from config import (
    MIMIC_OTHER_AP, CXR14_FOLDER, PNEUMONIA_FOLDER, PNEUMONIA_DS_NORMAL,
    PADCHEST_FOLDER, VINDR_FOLDER
)


class NoFindingDataset(BaseTransformDataset):
    """
    Dataset combining 'No Finding' images from multiple sources.
    
    Combines images from MIMIC-CXR, CXR-14, and Pneumonia datasets
    that are labeled as having no pathological findings.
    
    Parameters
    ----------
    mimic_path : str
        Path to MIMIC 'No Finding' CSV.
    cxr14_path : str
        Path to CXR-14 'No Finding' CSV.
    pneumonia_path : str
        Path to folder with normal pneumonia dataset images.
    """
    
    def __init__(
        self,
        mimic_path: str,
        cxr14_path: str,
        pneumonia_path: str,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        mimic_csv = pd.read_csv(mimic_path)
        cxr14_csv = pd.read_csv(cxr14_path)
        
        # Start with pneumonia normal images
        self.files_paths = [
            os.path.join(pneumonia_path, f) for f in os.listdir(pneumonia_path)
        ]
        
        # Add MIMIC images
        for i in range(len(mimic_csv)):
            subject_id = 'p' + str(mimic_csv.iloc[i]['subject_id'])
            study_id = 's' + str(mimic_csv.iloc[i]['study_id'])
            dicom_id = str(mimic_csv.iloc[i]['dicom_id'])
            sub_folder = subject_id[:3]
            dicom_path = os.path.join(MIMIC_OTHER_AP, sub_folder, subject_id, study_id, dicom_id)
            
            if os.path.exists(dicom_path):
                self.files_paths.append(dicom_path)
        
        # Add CXR-14 images
        for i in range(len(cxr14_csv)):
            c_id = cxr14_csv.iloc[i]['id']
            c_path = f"{CXR14_FOLDER}/AP_images/{c_id}"
            self.files_paths.append(c_path)
        
        self.t_len = len(self.files_paths)
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, path


class SmallNoFindingDataset(BaseTransformDataset):
    """
    Simple dataset from single directory of normal images.
    
    Parameters
    ----------
    dir_path : str
        Path to directory containing NIfTI images.
    """
    
    def __init__(
        self,
        dir_path: str = PNEUMONIA_DS_NORMAL,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        self.files_paths = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
        ]
        self.t_len = len(self.files_paths)
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, path


class ExpertNoFindingDataset(BaseTransformDataset):
    """
    Dataset with expert-annotated 'No Finding' images.
    
    Combines VinDr, PadChest, and Pneumonia dataset normal images
    that have been verified by radiologists.
    
    Parameters
    ----------
    vindr_train_csv_path : str
        Path to VinDr training set 'No Finding' CSV.
    vindr_test_csv_path : str
        Path to VinDr test set 'No Finding' CSV.
    padchest_csv_path : str
        Path to PadChest 'No Finding' CSV.
    pneumonia_dir_path : str
        Path to normal images directory.
    """
    
    def __init__(
        self,
        vindr_train_csv_path: str,
        vindr_test_csv_path: str,
        padchest_csv_path: str,
        pneumonia_dir_path: str = PNEUMONIA_DS_NORMAL,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        # Start with pneumonia normal images
        self.file_paths = [
            os.path.join(pneumonia_dir_path, f) 
            for f in os.listdir(pneumonia_dir_path)
        ]
        
        # Add PadChest images
        pc_df = pd.read_csv(padchest_csv_path)
        pc_names = [
            f"{PADCHEST_FOLDER}/images/{n.split('.')[0]}.nii.gz"
            for n in pc_df["ImageID"]
        ]
        self.file_paths.extend(pc_names)
        
        # Add VinDr train images
        vin_train_df = pd.read_csv(vindr_train_csv_path)
        vin_train_names = [
            f"{VINDR_FOLDER}/train/{n}.nii.gz"
            for n in vin_train_df['image_id']
        ]
        self.file_paths.extend(vin_train_names)
        
        # Add VinDr test images
        vin_test_df = pd.read_csv(vindr_test_csv_path)
        vin_test_names = [
            f"{VINDR_FOLDER}/test/{n}.nii.gz"
            for n in vin_test_df['image_id']
        ]
        self.file_paths.extend(vin_test_names)
        
        self.t_len = len(self.file_paths)
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.file_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, path


class NormalAbnormalDataset(BaseTransformDataset):
    """
    Binary normal/abnormal classification dataset from multiple sources.
    
    Combines VinDr, PadChest, Pneumonia, and optionally CXR-14 and MIMIC
    datasets for binary classification.
    
    Parameters
    ----------
    pneumonia_dir_path : str
        Path to pneumonia dataset directory.
    padchest_dir_path : str
        Path to PadChest directory.
    vindr_dir_path : str
        Path to VinDr directory.
    mode : str
        'train', 'test', or '' for full dataset.
    use_non_expert : bool
        Whether to include non-expert annotated datasets (CXR-14, MIMIC).
    """
    
    def __init__(
        self,
        pneumonia_dir_path: str = PNEUMONIA_FOLDER,
        padchest_dir_path: str = PADCHEST_FOLDER,
        vindr_dir_path: str = VINDR_FOLDER,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03,
        mode: str = '',
        use_non_expert: bool = False
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        assert mode in {'', 'train', 'test'}
        
        # Labels: [Abnormal, Normal]
        abnormal_label = torch.tensor([1, 0], dtype=torch.bool)
        normal_label = torch.tensor([0, 1], dtype=torch.bool)
        
        # Pneumonia dataset
        pneumonia_healthy_path = f"{pneumonia_dir_path}/normal_nibs"
        pneumonia_sick_path = f"{pneumonia_dir_path}/pneumonia_nibs"
        pp_healthy = [
            (f"{pneumonia_healthy_path}/{n}", normal_label)
            for n in os.listdir(pneumonia_healthy_path)
        ]
        pp_sick = [
            (f"{pneumonia_sick_path}/{n}", abnormal_label)
            for n in os.listdir(pneumonia_sick_path)
        ]
        
        # PadChest dataset
        pc_csv_healthy_path = f"{padchest_dir_path}/no_finding_only_normal.csv"
        pc_healthy_df = pd.read_csv(pc_csv_healthy_path)
        pc_healthy_ids = list(pc_healthy_df['ImageID'])
        pc_healthy = [
            (f"{padchest_dir_path}/images/{n.split('.')[0]}.nii.gz", normal_label)
            for n in pc_healthy_ids
        ]
        
        pc_csv_sick_path = f"{padchest_dir_path}/specific_abnormalities.csv"
        if os.path.exists(pc_csv_sick_path):
            pc_sick_df = pd.read_csv(pc_csv_sick_path)
            pc_sick_ids = list(pc_sick_df['ImageID'])
            pc_sick = [
                (f"{padchest_dir_path}/images/{n.split('.')[0]}.nii.gz", abnormal_label)
                for n in pc_sick_ids
            ]
        else:
            pc_sick = []
        
        # VinDr dataset
        vindr_healthy_train = self._load_vindr_subset(
            vindr_dir_path, 'train', normal_label, healthy=True
        )
        vindr_sick_train = self._load_vindr_subset(
            vindr_dir_path, 'train', abnormal_label, healthy=False
        )
        vindr_healthy_test = self._load_vindr_subset(
            vindr_dir_path, 'test', normal_label, healthy=True
        )
        vindr_sick_test = self._load_vindr_subset(
            vindr_dir_path, 'test', abnormal_label, healthy=False
        )
        
        # Organize arrays
        self.arrs = [
            vindr_healthy_train, vindr_sick_train,
            vindr_healthy_test, vindr_sick_test,
            pc_healthy, pc_sick, pp_healthy, pp_sick
        ]
        self.labels_dict = {
            0: 'VinDr_train_Normal', 1: 'VinDr_train_Abnormal',
            2: 'VinDr_test_Normal', 3: 'VinDr_test_Abnormal',
            4: 'PadChest_Normal', 5: 'PadChest_Abnormal',
            6: 'PediPneumonia_Normal', 7: 'PediPneumonia_Abnormal'
        }
        
        # Apply train/test split
        if mode == 'train':
            for i in range(len(self.arrs)):
                Random(42).shuffle(self.arrs[i])
                self.arrs[i] = self.arrs[i][:int(len(self.arrs[i]) * 0.75)]
        elif mode == 'test':
            for i in range(len(self.arrs)):
                Random(42).shuffle(self.arrs[i])
                self.arrs[i] = self.arrs[i][int(len(self.arrs[i]) * 0.75):]
        
        # Flatten all arrays
        self.files_labels = [t for arr in self.arrs for t in arr]
        self.t_len = len(self.files_labels)
    
    def _load_vindr_subset(
        self,
        vindr_dir: str,
        split: str,
        label: torch.Tensor,
        healthy: bool
    ) -> List[Tuple[str, torch.Tensor]]:
        """Load VinDr subset (healthy or sick)."""
        csv_healthy = f"{vindr_dir}/no_finding_{split}.csv"
        csv_all = f"{vindr_dir}/labels_{split}.csv"
        
        if not os.path.exists(csv_healthy) or not os.path.exists(csv_all):
            return []
        
        healthy_df = pd.read_csv(csv_healthy)
        all_df = pd.read_csv(csv_all)
        
        healthy_ids = set(healthy_df['image_id'])
        all_ids = set(all_df['image_id'])
        
        if healthy:
            ids = healthy_ids
        else:
            ids = all_ids - healthy_ids
        
        return [
            (f"{vindr_dir}/{split}/{n}.nii.gz", label)
            for n in sorted(ids)
        ]
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.files_labels[idx]
        data = nib.load(path).get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, label
    
    def get_items_by_label_idx(
        self,
        label_idx: int,
        count: int,
        get_names: bool = False
    ):
        """Get specific items by label index for visualization."""
        items_shuffled = self.arrs[label_idx].copy()
        shuffle(items_shuffled)
        
        items = []
        names = []
        for name, _ in items_shuffled[:count]:
            data = nib.load(name).get_fdata()[None, ...]
            items.append(torch.tensor(data))
            names.append(name)
        
        if items:
            items = torch.stack(items)
        else:
            items = torch.tensor([])
        
        if get_names:
            return items, names
        return items
    
    def get_labels(self) -> str:
        return 'Abnormal|Normal'
