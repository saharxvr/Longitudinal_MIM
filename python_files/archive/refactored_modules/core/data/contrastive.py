"""
Contrastive Learning Datasets
=============================

Datasets for contrastive learning and self-supervised pretraining.

Classes:
--------
- ContrastiveLearningDataset: Multi-label CXR-14 based dataset
- ContrastiveLearningSmallDataset: Binary pneumonia/normal dataset
- GeneralContrastiveLearningDataset: Flexible multi-source dataset
- ContrastiveCXRSampler: Custom sampler for balanced label sampling
"""

import os
import math
import random
import pandas as pd
import torch
import nibabel as nib
from torch.utils.data import Dataset, RandomSampler
from random import shuffle, Random
from typing import Iterator, Optional, List
from functools import reduce
from operator import and_

from core.data.base import BaseTransformDataset, get_proj_mat
from config import (
    ALL_LABELS, ALL_LABELS_NUM, LABEL_NO_FINDING, NO_FINDING_PROB_FACTOR,
    CUR_LABELS_NUM, DEVICE, CSVS_TO_LABEL_MAPPING, CSVS_TO_IM_PATH_GETTERS
)


class ContrastiveLearningDataset(BaseTransformDataset):
    """
    Dataset for contrastive learning with balanced label sampling.
    
    Samples images with weighted probability based on label frequency
    to ensure balanced representation of all classes.
    
    Parameters
    ----------
    data_folder : str
        Root folder containing images.
    labels_path : str
        Path to CSV with labels.
    ds_labels : list[str]
        Labels to use from dataset.
    groups : list[list[int]], optional
        Label groupings for hierarchical classification.
    get_weights : list, optional
        Manual weights for each label class.
    label_no_finding : bool
        Whether to include 'No Finding' as explicit class.
    do_shuffle : bool
        Whether to shuffle data each epoch.
    train : bool
        Whether this is training set (enables shuffling).
    """
    
    def __init__(
        self,
        data_folder: str,
        labels_path: str,
        ds_labels: List[str],
        groups: Optional[List[List[int]]] = None,
        get_weights: Optional[List[float]] = None,
        label_no_finding: bool = LABEL_NO_FINDING,
        do_shuffle: bool = True,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03,
        train: bool = True
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        self.data_folder = data_folder
        self.do_shuffle = do_shuffle
        self.labels = ds_labels
        self.groups = groups
        
        # Setup label grouping
        if groups:
            labels_num = len(groups) + 1
            group_func = self._group_labels
        else:
            labels_num = len(ds_labels) + 1
            group_func = lambda x: x
        
        proj_mat = get_proj_mat(ALL_LABELS_NUM, ds_labels, ALL_LABELS)
        
        # Parse CSV and organize by label
        data = pd.read_csv(labels_path)
        self.img_to_label_arrs = [[] for _ in range(labels_num)]
        self.indices = [0] * labels_num
        
        for i in range(len(data)):
            c_name = data.iloc[i]['id']
            c_labels = data.iloc[i]['labels'][1:-1].split()
            c_labels = torch.tensor([c_l == "True" for c_l in c_labels], dtype=torch.bool)
            c_labels = torch.sum(c_labels * proj_mat, dim=1)
            c_labels = group_func(c_labels)
            
            if torch.sum(c_labels) > 0:
                chosen_arr = torch.multinomial(c_labels.float(), 1).item()
                if label_no_finding:
                    c_labels = torch.cat([c_labels, torch.zeros(1, dtype=torch.bool)])
                self.img_to_label_arrs[chosen_arr].append((c_name, c_labels))
            else:
                if label_no_finding:
                    c_labels = torch.cat([c_labels, torch.ones(1, dtype=torch.bool)])
                self.img_to_label_arrs[-1].append((c_name, c_labels))
        
        # Calculate weights
        if get_weights:
            self.pos_weights = ((sum(get_weights) - torch.tensor(get_weights)) / torch.tensor(get_weights))
        else:
            self.pos_weights = torch.ones(labels_num)
        
        self.arr_lens = [len(arr) for arr in self.img_to_label_arrs]
        
        # Sampling weights
        if not get_weights:
            self.get_weights = torch.ones(len(self.img_to_label_arrs)).float()
            self.get_weights[-1] = NO_FINDING_PROB_FACTOR
        else:
            self.get_weights = torch.tensor(get_weights).float()
        
        # Calculate effective length
        self.t_len = sum(len(arr) for arr in self.img_to_label_arrs[:-1])
        self.t_len = int(self.t_len * (1 / (1 - NO_FINDING_PROB_FACTOR / torch.sum(self.get_weights))))
        self.t_len = math.ceil(self.t_len)
        
        if train:
            self.shuffle()
    
    def _group_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Group labels according to self.groups."""
        new_labels = [torch.max(labels[group]).item() for group in self.groups]
        return torch.tensor(new_labels).bool()
    
    def get_pos_weights(self) -> torch.Tensor:
        """Return positive class weights for loss weighting."""
        return self.pos_weights
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int):
        chosen_arr = torch.multinomial(self.get_weights, 1).item()
        idx_in_arr = self.indices[chosen_arr] % self.arr_lens[chosen_arr]
        name, labels = self.img_to_label_arrs[chosen_arr][idx_in_arr]
        
        img_data = nib.load(f"{self.data_folder}/{name}").get_fdata()[None, ...]
        img_data = self.apply_transforms(img_data)
        self.indices[chosen_arr] += 1
        
        return img_data, labels
    
    def get_items_by_label_idx(self, label_idx, count: int) -> torch.Tensor:
        """Get specific items by label index for visualization."""
        items = []
        if isinstance(label_idx, int):
            for i in range(min(count, len(self.img_to_label_arrs[label_idx]))):
                name, _ = self.img_to_label_arrs[label_idx][i]
                img_data = nib.load(f"{self.data_folder}/{name}").get_fdata()[None, ...]
                items.append(torch.tensor(img_data))
        
        if items:
            return torch.stack(items)
        return torch.tensor([])
    
    def get_labels(self) -> str:
        """Return string representation of active labels."""
        if self.labels == ALL_LABELS and not self.groups:
            return "All_labels"
        elif not self.groups:
            return '|'.join(self.labels)
        
        labels_str = ''
        for group in self.groups:
            group_labels = [self.labels[i] for i in group]
            labels_str += '[' + '|'.join(group_labels) + ']'
        return labels_str
    
    def shuffle(self):
        """Shuffle data for new epoch."""
        if self.do_shuffle:
            for i in range(len(self.img_to_label_arrs)):
                shuffle(self.img_to_label_arrs[i])
                self.indices[i] = 0


class ContrastiveLearningSmallDataset(BaseTransformDataset):
    """
    Small binary classification dataset (Pneumonia vs Normal).
    
    Parameters
    ----------
    data_folder : str
        Root folder containing normal_nibs/ and pneumonia_nibs/ subfolders.
    """
    
    def __init__(
        self,
        data_folder: str = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal',
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.03,
        ver_flip_chance: float = 0.03
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        self.data_folder = data_folder
        normal_folder = f"{data_folder}/normal_nibs"
        pneumonia_folder = f"{data_folder}/pneumonia_nibs"
        
        self.files_list = []
        self.files_list.extend([
            (f'/normal_nibs/{f}', torch.tensor([0, 1], dtype=torch.bool))
            for f in os.listdir(normal_folder)
        ])
        self.files_list.extend([
            (f'/pneumonia_nibs/{f}', torch.tensor([1, 0], dtype=torch.bool))
            for f in os.listdir(pneumonia_folder)
        ])
    
    def __len__(self) -> int:
        return len(self.files_list)
    
    def __getitem__(self, idx: int):
        name, labels = self.files_list[idx]
        data = nib.load(f"{self.data_folder}/{name}").get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, labels
    
    def get_labels(self) -> str:
        return "Pneumonia|No_Finding"


class GeneralContrastiveLearningDataset(BaseTransformDataset):
    """
    Flexible multi-source contrastive learning dataset.
    
    Combines multiple CSV sources with configurable label mappings.
    Supports train/test splitting and unlabeled data inclusion.
    
    Parameters
    ----------
    csvs : list[str]
        List of CSV file paths.
    label_names : list[str]
        Labels to use.
    unlabeled_perc : float
        Percentage of unlabeled data to include.
    mode : str
        'train', 'test', or '' for full dataset.
    """
    
    def __init__(
        self,
        csvs: List[str],
        label_names: List[str],
        unlabeled_perc: float = 0.,
        rot_chance: float = 0.075,
        hor_flip_chance: float = 0.05,
        ver_flip_chance: float = 0.05,
        mode: str = ''
    ):
        super().__init__(rot_chance, hor_flip_chance, ver_flip_chance)
        
        assert mode in {'', 'train', 'test'}
        
        self.label_names = label_names
        self.labels_dict = {i: label for i, label in enumerate(label_names)}
        
        csvs_to_labels = {
            csv: {label: CSVS_TO_LABEL_MAPPING[csv][label] for label in label_names}
            for csv in csvs
        }
        include_unlabeled = unlabeled_perc > 0.
        
        self.files_labels = []
        self.ds_labels_columns_list = [f'Label_{n}' for n in label_names]
        self.all_df = pd.DataFrame(columns=['id'] + self.ds_labels_columns_list)
        
        for csv, labels in csvs_to_labels.items():
            c_df = pd.read_csv(csv)
            column_names = set(c_df.columns.values.tolist())
            c_path_getter = CSVS_TO_IM_PATH_GETTERS[csv][0]
            c_id_column = CSVS_TO_IM_PATH_GETTERS[csv][1]
            
            # Initialize label columns
            for ds_label in label_names:
                c_df[f'Label_{ds_label}'] = 0
            
            # Map labels
            for label_name, label_list in labels.items():
                c_df[f'Label_{label_name}'] = c_df.apply(
                    lambda row: int(any([row[col] == 1 for col in label_list if col in row])),
                    axis=1
                )
            
            # Get paths
            c_df[c_id_column] = c_df.apply(c_path_getter, axis=1)
            c_df = c_df.rename(columns={c_id_column: 'id'})
            self.all_df = pd.concat([self.all_df, c_df[['id'] + self.ds_labels_columns_list]])
            
            # Collect samples
            cur_arr = []
            cur_unlabeled = []
            
            for i in range(len(c_df)):
                c_path = c_df.iloc[i]['id']
                c_labels = torch.tensor(
                    c_df.iloc[i][self.ds_labels_columns_list].tolist(),
                    dtype=torch.bool
                )
                if torch.sum(c_labels).item() > 0:
                    cur_arr.append((c_path, c_labels))
                elif include_unlabeled:
                    cur_unlabeled.append((c_path, c_labels))
            
            # Train/test split
            if mode == 'train':
                Random(42).shuffle(cur_arr)
                cur_arr = cur_arr[:int(len(cur_arr) * 0.75)]
            elif mode == 'test':
                Random(42).shuffle(cur_arr)
                cur_arr = cur_arr[int(len(cur_arr) * 0.75):]
            
            self.files_labels.extend(cur_arr)
            
            if include_unlabeled:
                cur_unlabeled = cur_unlabeled[:int(unlabeled_perc * len(cur_arr))]
                self.files_labels.extend(cur_unlabeled)
        
        self.t_len = len(self.files_labels)
        
        # Calculate positive weights
        self.pos_weight = []
        for label_col in self.ds_labels_columns_list:
            num_oc = len(self.all_df[self.all_df[label_col] == 1])
            self.pos_weight.append((self.t_len - num_oc) / max(num_oc, 1))
        self.pos_weight = torch.tensor(self.pos_weight).to(DEVICE).float()
    
    def get_pos_weight(self) -> torch.Tensor:
        return self.pos_weight
    
    def __len__(self) -> int:
        return self.t_len
    
    def __getitem__(self, idx: int):
        path, label = self.files_labels[idx]
        data = nib.load(path).get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, label
    
    def get_labels(self) -> str:
        return '|'.join(self.label_names)


class ContrastiveCXRSampler(RandomSampler):
    """
    Custom sampler that shuffles dataset after each epoch.
    
    Ensures ContrastiveLearningDataset reshuffles its internal
    arrays after each complete iteration.
    """
    
    def __init__(self, data_source: ContrastiveLearningDataset):
        super().__init__(data_source)
        self._num_samples = len(data_source)
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        
        if self.replacement:
            for _ in range(self._num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self._num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self._num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self._num_samples % n]
        
        # Shuffle dataset for next epoch
        self.data_source.shuffle()
