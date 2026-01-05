"""
Dataset classes for Longitudinal CXR Analysis.

This module provides PyTorch Dataset implementations for various training tasks:

Main Classes (used in training):
--------------------------------
- LongitudinalMIMDataset: Main dataset for longitudinal change detection training
  Loads baseline/followup pairs with ground truth difference maps from:
  - Synthetic DRR pairs (CT_entities/DRR_generator.py output)
  - Inpainted pairs
  - Entity overlay pairs

Supporting Classes (for other tasks):
-------------------------------------
- PatchReconstructionDataset: Self-supervised masked reconstruction
- ContrastiveLearningDataset: Contrastive learning with labels
- NoFindingDataset: Normal CXR images only
- NormalAbnormalDataset: Binary classification dataset
- GeneralContrastiveLearningDataset: Multi-source contrastive learning

Data Format:
------------
All datasets expect NIfTI (.nii.gz) format images.
Images are loaded and normalized to [0, 1] range.
"""

import math
import os.path
import random

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, RandomSampler
import torch
import pandas as pd
from typing import Iterator
import nibabel as nib
from glob import glob
from constants import *
from random import shuffle, Random
from typing import Optional
import torchvision.transforms as tf
import torchvision.transforms.v2 as v2
from utils import get_mimic_path, check_existence_of_mimic_path, scale_and_suppress_non_max, get_max_inpaint_diff_val
from numpy.random import permutation
import kornia
from functools import reduce
from operator import or_, and_
from augmentations import *
from CT_entities.CXR_from_CT import image_histogram_equalization
from nibabel.orientations import axcodes2ornt, ornt_transform, aff2axcodes, apply_orientation
from nibabel import as_closest_canonical


# =============================================================================
# MASKED RECONSTRUCTION DATASET
# =============================================================================

class PatchReconstructionDataset(Dataset):
    """
    Dataset for self-supervised patch reconstruction (MAE-style).
    
    Loads single CXR images for masked autoencoder pretraining.
    
    Args:
        data_folders: List of directories containing .nii.gz files
        rot_chance: Probability of random 90° rotation
        hor_flip_chance: Probability of horizontal flip
        ver_flip_change: Probability of vertical flip
    """
    
    def __init__(self, data_folders, rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03):
        super().__init__()

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.file_paths = []
        for folder in data_folders:
            self.file_paths.extend(glob(f'{folder}/**/*.nii.gz', recursive=True))

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        item = nib.load(self.file_paths[idx]).get_fdata()[None, ...]
        item = self.apply_transforms(item)
        return item


def get_standard_vector(size, idx):
    vec = torch.zeros(size, dtype=torch.bool)
    vec[idx] = True
    return vec


def get_proj_mat(size, ds_labels):
    if ds_labels == ALL_LABELS:
        return torch.eye(size, dtype=torch.bool)
    mat = []
    for label in ds_labels:
        idx = ALL_LABELS.index(label)
        mat.append(get_standard_vector(size, idx))
    mat = torch.stack(mat)
    return mat


class ContrastiveLearningDataset(Dataset):
    def __init__(self, data_folder: str, labels_path: str, ds_labels: list[str], groups: Optional[list[list[int]]] = None, get_weights=None, label_no_finding=LABEL_NO_FINDING, do_shuffle: bool=True,
                 rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03, train=True):
        super().__init__()

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.data_folder = data_folder
        self.do_shuffle = do_shuffle
        self.labels = ds_labels
        self.groups = groups

        if groups:
            labels_num = len(groups) + 1
            group_func = self.group_labels
        else:
            labels_num = len(ds_labels) + 1
            group_func = lambda x: x

        proj_mat = get_proj_mat(ALL_LABELS_NUM, ds_labels)

        data = pd.read_csv(labels_path)
        # num_samples = float(len(data))
        self.img_to_label_arrs = [[] for _ in range(labels_num)]
        self.indices = [0 for _ in range(labels_num)]
        # self.pos_weights = [0 for _ in range(labels_num)]
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

            # for j in range(labels_num):
            #     self.pos_weights[j] += c_labels[j]

        # self.pos_weights = torch.tensor(self.pos_weights, dtype=torch.half)
        # neg_counts = num_samples - self.pos_weights
        # self.pos_weights = (neg_counts / self.pos_weights).half()
        self.pos_weights = ((sum(get_weights) - torch.tensor(get_weights)) / torch.tensor(get_weights))

        self.arr_lens = [len(self.img_to_label_arrs[i]) for i in range(len(self.img_to_label_arrs))]

        if not get_weights:
            self.get_weights = torch.ones(len(self.img_to_label_arrs)).float()
            self.get_weights[-1] = NO_FINDING_PROB_FACTOR
        else:
            self.get_weights = torch.tensor(get_weights).float()

        self.t_len = 0
        for i in range(len(self.img_to_label_arrs[:-1])):
            self.t_len += len(self.img_to_label_arrs[i])
        self.t_len *= 1 / (1 - NO_FINDING_PROB_FACTOR / torch.sum(self.get_weights))
        self.t_len = math.ceil(self.t_len)

        if train:
            self.shuffle()

    def group_labels(self, labels):
        new_labels = []
        for group in self.groups:
            new_labels.append(torch.max(labels[group]).item())
        new_labels = torch.tensor(new_labels).bool()
        return new_labels

    def get_pos_weights(self):
        return self.pos_weights

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return self.t_len

    def __getitem__(self, idx):
        chosen_arr = torch.multinomial(self.get_weights, 1).item()
        idx_in_arr = self.indices[chosen_arr] % self.arr_lens[chosen_arr]
        name, labels = self.img_to_label_arrs[chosen_arr][idx_in_arr]
        img_data = nib.load(self.data_folder + f'/{name}').get_fdata()[None, ...]
        img_data = self.apply_transforms(img_data)
        self.indices[chosen_arr] += 1
        return img_data, labels

    def get_items_by_label_idx(self, label_idx, count):
        items = []
        if type(label_idx) == int:
            for i in range(count):
                name, _ = self.img_to_label_arrs[label_idx][i]
                img_data = nib.load(self.data_folder + f'/{name}').get_fdata()[None, ...]
                items.append(torch.tensor(img_data))
            items = torch.stack(items)
            return items
        else:
            assert len(label_idx) == CUR_LABELS_NUM
            arrs = [self.img_to_label_arrs[i] for i in range(len(label_idx)) if label_idx[i] == 1]
            assert len(arrs) == sum(label_idx)
            label_idx = torch.tensor(label_idx, dtype=torch.bool)
            for arr in arrs:
                for i in range(len(arr)-1, -1, -1):
                    name, label = arr[i]
                    if torch.equal(label, label_idx):
                        img_data = nib.load(self.data_folder + f'/{name}').get_fdata()[None, ...]
                        items.append(torch.tensor(img_data))
                    if len(items) == count:
                        items = torch.stack(items)
                        return items
            items_len = len(items)
            items = torch.stack(items)
            if items_len < count:
                print(f"Warning. Only found {items_len} items with the labels {label_idx}")
            return items

    def get_labels(self):
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
        if self.do_shuffle:
            print("Shuffling")
            for i in range(len(self.img_to_label_arrs)):
                shuffle(self.img_to_label_arrs[i])
                self.indices[i] = 0


class ContrastiveLearningSmallDataset(Dataset):
    def __init__(self, data_folder='/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal',
                 rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03):
        super().__init__()

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.data_folder = data_folder
        normal_folder = data_folder + '/normal_nibs'
        pneumonia_folder = data_folder + '/pneumonia_nibs'

        self.files_list = []
        self.files_list.extend([('/normal_nibs/' + f, torch.tensor([0, 1], dtype=torch.bool)) for f in os.listdir(normal_folder)])
        self.files_list.extend([('/pneumonia_nibs/' + f, torch.tensor([1, 0], dtype=torch.bool)) for f in os.listdir(pneumonia_folder)])

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        name, labels = self.files_list[idx]
        data = nib.load(self.data_folder + f'/{name}').get_fdata()[None, ...]
        data = self.apply_transforms(data)
        return data, labels

    def get_labels(self):
        return "Pneumonia|No_Finding"

    def get_items_by_label_idx(self, label_idx, count):
        if label_idx == 0:
            folder = '/normal_nibs/'
        elif label_idx == 1:
            folder = '/pneumonia_nibs/'
        else:
            raise "Invalid label_idx"

        items = []
        for item in self.files_list:
            if len(items) == count:
                break
            name, label = item
            if not name.startswith(folder):
                continue
            data = nib.load(self.data_folder + f'/{name}').get_fdata()[None, ...]
            items.append(torch.tensor(data))

        items_len = len(items)
        items = torch.stack(items)
        if items_len < count:
            print(f"Warning. Only found {items_len} items with the labels {label_idx}")
        return items


class ContrastiveCXRSampler(RandomSampler):
    def __init__(self, data_source: ContrastiveLearningDataset):
        super().__init__(data_source)

        self._num_samples = len(data_source)

    def __iter__(self) -> Iterator[int]:
        self.data_source: ContrastiveLearningDataset
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
        self.data_source.shuffle()


class NoFindingDataset(Dataset):
    def __init__(self, mimic_path, cxr14_path, pneumonia_path, rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03):
        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        mimic_csv = pd.read_csv(mimic_path)
        cxr14_csv = pd.read_csv(cxr14_path)

        self.files_paths = [os.path.join(pneumonia_path, f) for f in os.listdir(pneumonia_path)]

        for i in range(len(mimic_csv)):
            subject_id = 'p' + str(mimic_csv.iloc[i]['subject_id'])
            study_id = 's' + str(mimic_csv.iloc[i]['study_id'])
            dicom_id = str(mimic_csv.iloc[i]['dicom_id'])
            sub_folder = subject_id[:3]
            dicom_path = os.path.join(MIMIC_OTHER_AP, sub_folder, subject_id, study_id, dicom_id)

            if not os.path.exists(dicom_path):
                continue

            self.files_paths.append(dicom_path)

        for i in range(len(cxr14_csv)):
            c_id = cxr14_csv.iloc[i]['id']
            c_path = CXR14_FOLDER + f'/AP_images/{c_id}'

            self.files_paths.append(c_path)

        self.t_len = len(self.files_paths)

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return self.t_len

    def __getitem__(self, idx):
        path = self.files_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        tf_data = self.apply_transforms(data)
        return tf_data, path


class SmallNoFindingDataset(Dataset):
    def __init__(self, dir_path=PNEUMONIA_DS_NORMAL, rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03):
        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.files_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

        self.t_len = len(self.files_paths)

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return self.t_len

    def __getitem__(self, idx):
        path = self.files_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        tf_data = self.apply_transforms(data)
        return tf_data, path


class ExpertNoFindingDataset(Dataset):
    def __init__(self, vindr_train_csv_path, vindr_test_csv_path, padchest_csv_path, pneumonia_dir_path=PNEUMONIA_DS_NORMAL, rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03):
        super(ExpertNoFindingDataset).__init__()

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.file_paths = [os.path.join(pneumonia_dir_path, f) for f in os.listdir(pneumonia_dir_path)]

        pc_df = pd.read_csv(padchest_csv_path)
        pc_names = list(pc_df["ImageID"])
        pc_names = [PADCHEST_FOLDER + f"/images/{n.split('.')[0]}.nii.gz" for n in pc_names]

        self.file_paths.extend(pc_names)

        vin_train_df = pd.read_csv(vindr_train_csv_path)
        vin_train_names = list(vin_train_df['image_id'])
        vin_train_names = [f'{VINDR_FOLDER}/train/{n}.nii.gz' for n in vin_train_names]

        vin_test_df = pd.read_csv(vindr_test_csv_path)
        vin_test_names = list(vin_test_df['image_id'])
        vin_test_names = [f'{VINDR_FOLDER}/test/{n}.nii.gz' for n in vin_test_names]

        self.file_paths.extend(vin_train_names)
        self.file_paths.extend(vin_test_names)

        pedipneumonia_dir = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/pneumonia_normal/normal_nibs'
        self.file_paths.extend([f'{pneumonia_dir_path}/{n}' for n in os.listdir(pneumonia_dir_path)])

        self.t_len = len(self.file_paths)

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __len__(self):
        return self.t_len

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        data = nib.load(path).get_fdata()[None, ...]
        tf_data = self.apply_transforms(data)
        return tf_data, path


class NormalAbnormalDataset(Dataset):
    def __init__(self, pneumonia_dir_path=PNEUMONIA_FOLDER, padchest_dir_path=PADCHEST_FOLDER, vindr_dir_path=VINDR_FOLDER,
                 rot_chance=0.075, hor_flip_chance=0.03, ver_flip_change=0.03, mode='', use_non_expert=False):
        super(NormalAbnormalDataset).__init__()

        assert mode in {'', 'train', 'test'}

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        pneumonia_healthy_path = pneumonia_dir_path + '/normal_nibs'
        pneumonia_sick_path = pneumonia_dir_path + '/pneumonia_nibs'
        pp_healthy = [(pneumonia_healthy_path + f'/{n}', torch.tensor([0, 1], dtype=torch.bool)) for n in os.listdir(pneumonia_healthy_path)]
        pp_sick = [(pneumonia_sick_path + f'/{n}', torch.tensor([1, 0], dtype=torch.bool)) for n in os.listdir(pneumonia_sick_path)]

        # pc_csv_healthy_path = padchest_dir_path + '/no_finding.csv'
        pc_csv_healthy_path = padchest_dir_path + '/no_finding_only_normal.csv'
        # pc_csv_all_path = padchest_dir_path + '/relevant_images.csv'
        pc_healthy_df = pd.read_csv(pc_csv_healthy_path)
        # pc_all_df = pd.read_csv(pc_csv_all_path)
        pc_healthy_ids = list(pc_healthy_df['ImageID'])
        # pc_all_ids = list(pc_all_df['ImageID'])
        # pc_sick_ids = sorted(list(set(pc_all_ids).difference(set(pc_healthy_ids))))
        pc_healthy = [(padchest_dir_path + f'/images/{n.split(".")[0]}.nii.gz', torch.tensor([0, 1], dtype=torch.bool)) for n in pc_healthy_ids]
        # self.files_labels.extend([(padchest_dir_path + f'/images/{n.split(".")[0]}.nii.gz', torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.bool)) for n in pc_sick_ids])

        pc_csv_sick_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PadChest/specific_abnormalities.csv'
        pc_sick_df = pd.read_csv(pc_csv_sick_path)
        pc_sick_ids = list(pc_sick_df['ImageID'])
        pc_sick = [(padchest_dir_path + f'/images/{n.split(".")[0]}.nii.gz', torch.tensor([1, 0], dtype=torch.bool)) for n in pc_sick_ids]

        vindr_csv_healthy_train_path = vindr_dir_path + '/no_finding_train.csv'
        vindr_csv_healthy_test_path = vindr_dir_path + '/no_finding_test.csv'
        vindr_csv_all_train_path = vindr_dir_path + '/labels_train.csv'
        vindr_csv_all_test_path = vindr_dir_path + '/labels_test.csv'
        vindr_healthy_train_df = pd.read_csv(vindr_csv_healthy_train_path)
        vindr_healthy_test_df = pd.read_csv(vindr_csv_healthy_test_path)
        vindr_all_train_df = pd.read_csv(vindr_csv_all_train_path)
        vindr_all_test_df = pd.read_csv(vindr_csv_all_test_path)
        vindr_healthy_train_ids = list(vindr_healthy_train_df['image_id'])
        vindr_healthy_test_ids = list(vindr_healthy_test_df['image_id'])
        vindr_all_train_ids = list(vindr_all_train_df['image_id'])
        vindr_all_test_ids = list(vindr_all_test_df['image_id'])
        vindr_sick_train_ids = sorted(list(set(vindr_all_train_ids).difference(set(vindr_healthy_train_ids))))
        vindr_sick_test_ids = sorted(list(set(vindr_all_test_ids).difference(set(vindr_healthy_test_ids))))
        vindr_train_healthy = [(vindr_dir_path + f'/train/{n}.nii.gz', torch.tensor([0, 1], dtype=torch.bool)) for n in vindr_healthy_train_ids]
        vindr_train_sick = [(vindr_dir_path + f'/train/{n}.nii.gz', torch.tensor([1, 0], dtype=torch.bool)) for n in vindr_sick_train_ids]
        vindr_test_healthy = [(vindr_dir_path + f'/test/{n}.nii.gz', torch.tensor([0, 1], dtype=torch.bool)) for n in vindr_healthy_test_ids]
        vindr_test_sick = [(vindr_dir_path + f'/test/{n}.nii.gz', torch.tensor([1, 0], dtype=torch.bool)) for n in vindr_sick_test_ids]

        self.arrs = [vindr_train_healthy, vindr_train_sick, vindr_test_healthy, vindr_test_sick, pc_healthy, pc_sick, pp_healthy, pp_sick]
        self.labels_dict = {0: 'VinDr_train_Normal', 1: 'VinDr_train_Abnormal', 2: 'VinDr_test_Normal', 3: 'VinDr_test_Abnormal',
                            4: 'PadChest_Normal', 5: 'PadChest_Abnormal', 6: 'PediPneumonia_Normal', 7: 'PediPneumonia_Abnormal'}

        if use_non_expert:
            cxr14_all_PA_path = CXR14_FOLDER + '/All_PA.csv'
            cxr14_healthy_PA_path = CXR14_FOLDER + '/No_Finding_PA.csv'
            cxr14_all_PA_df = pd.read_csv(cxr14_all_PA_path)
            cxr14_healthy_PA_df = pd.read_csv(cxr14_healthy_PA_path)
            cxr14_all_PA_ids = list(cxr14_all_PA_df['id'])
            cxr14_healthy_PA_ids = list(cxr14_healthy_PA_df['id'])
            cxr14_sick_PA_ids = sorted(list(set(cxr14_all_PA_ids).difference(set(cxr14_healthy_PA_ids))))
            cxr_pa_healthy = [(CXR14_FOLDER + f'/PA_images/{n}', torch.tensor([0, 1], dtype=torch.bool)) for n in cxr14_healthy_PA_ids]
            cxr_pa_sick = [(CXR14_FOLDER + f'/PA_images/{n}', torch.tensor([1, 0], dtype=torch.bool)) for n in cxr14_sick_PA_ids]

            cxr14_all_AP_path = CXR14_FOLDER + '/All_AP.csv'
            cxr14_healthy_AP_path = CXR14_FOLDER + '/No_Finding_AP.csv'
            cxr14_all_AP_df = pd.read_csv(cxr14_all_AP_path)
            cxr14_healthy_AP_df = pd.read_csv(cxr14_healthy_AP_path)
            cxr14_all_AP_ids = list(cxr14_all_AP_df['id'])
            cxr14_healthy_AP_ids = list(cxr14_healthy_AP_df['id'])
            cxr14_sick_AP_ids = sorted(list(set(cxr14_all_AP_ids).difference(set(cxr14_healthy_AP_ids))))
            cxr_ap_healthy = [(CXR14_FOLDER + f'/AP_images/{n}', torch.tensor([0, 1], dtype=torch.bool)) for n in cxr14_healthy_AP_ids]
            cxr_ap_sick = [(CXR14_FOLDER + f'/AP_images/{n}', torch.tensor([1, 0], dtype=torch.bool)) for n in cxr14_sick_AP_ids]

            mimic_healthy_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/No_Finding.csv'
            mimic_healthy_df = pd.read_csv(mimic_healthy_path)
            mimic_healthy_subjects = list(mimic_healthy_df['subject_id'])[:500]
            mimic_healthy_studies = list(mimic_healthy_df['study_id'])[:500]
            mimic_healthy_dicoms = list(mimic_healthy_df['dicom_id'])
            mimic_all_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/physionet.org/mimic-cxr-2.0.0-metadata.csv.gz'
            mimic_all_df = pd.read_csv(mimic_all_path)
            mimic_all_df = mimic_all_df[mimic_all_df['ViewPosition'] == 'AP']
            mimic_all_df = pd.merge(mimic_all_df, mimic_healthy_df, on=['subject_id', 'study_id', 'dicom_id'], how='outer', indicator=True)
            mimic_sick_df = mimic_all_df[mimic_all_df['_merge'] == 'left_only']
            mimic_sick_subjects = list(mimic_sick_df['subject_id'])[:500]
            mimic_sick_studies = list(mimic_sick_df['study_id'])[:500]
            mimic_sick_dicoms = list(mimic_sick_df['dicom_id'])[:500]
            mimic_healthy_dicoms = mimic_healthy_dicoms[:500]
            mimic_ap_healthy = [(get_mimic_path(sub, stu, dic), torch.tensor([0, 1], dtype=torch.bool)) for sub, stu, dic in zip(mimic_healthy_subjects, mimic_healthy_studies, mimic_healthy_dicoms) if check_existence_of_mimic_path(get_mimic_path(sub, stu, dic))]
            mimic_ap_sick = [(get_mimic_path(sub, stu, dic, True), torch.tensor([1, 0], dtype=torch.bool)) for sub, stu, dic in zip(mimic_sick_subjects, mimic_sick_studies, mimic_sick_dicoms) if check_existence_of_mimic_path(get_mimic_path(sub, stu, dic, True))]

            self.arrs.extend([cxr_pa_healthy, cxr_pa_sick, cxr_ap_healthy, cxr_ap_sick, mimic_ap_healthy, mimic_ap_sick])
            self.labels_dict.update({8: 'CXR14_PA_Normal', 9: 'CXR14_PA_Abnormal', 10: 'CXR14_AP_Normal', 11: 'CXR14_AP_Abnormal',
                                     12: 'MIMIC_AP_Normal', 13: 'MIMIC_AP_Abnormal'})

        if mode == 'train':
            for i in range(len(self.arrs)):
                Random(42).shuffle(self.arrs[i])
                self.arrs[i] = self.arrs[i][: int(len(self.arrs[i]) * 0.75)]
        elif mode == 'test':
            for i in range(len(self.arrs)):
                Random(42).shuffle(self.arrs[i])
                self.arrs[i] = self.arrs[i][int(len(self.arrs[i]) * 0.75):]

        self.files_labels = [t for arr in self.arrs for t in arr]

        self.t_len = len(self.files_labels)

    def __len__(self):
        return self.t_len

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __getitem__(self, idx):
        path, label = self.files_labels[idx]
        data = nib.load(path).get_fdata()[None, ...]
        tf_data = self.apply_transforms(data)
        return tf_data, label

    def get_items_by_label_idx(self, label_idx, count, get_names=False):
        items_shuffled = self.arrs[label_idx].copy()
        shuffle(items_shuffled)

        items = []
        names = []
        for item in items_shuffled:
            if len(items) == count:
                break
            name, label = item
            data = nib.load(name).get_fdata()[None, ...]
            items.append(torch.tensor(data))
            names.append(name)

        items_len = len(items)
        items = torch.stack(items)
        if items_len < count:
            print(f"Warning. Only found {items_len} items with the labels {label_idx}")
        if get_names:
            return items, names
        return items

    def get_labels(self):
        return 'Abnormal|Normal'


class GeneralContrastiveLearningDataset(Dataset):
    def __init__(self, csvs: list[str], label_names: list[str], unlabeled_perc=0.,
                 rot_chance=0.075, hor_flip_chance=0.05, ver_flip_change=0.05, mode='') -> None:
        super(GeneralContrastiveLearningDataset).__init__()

        assert mode in {'', 'train', 'test'}

        csvs_to_labels = {csv: {label: CSVS_TO_LABEL_MAPPING[csv][label] for label in label_names} for csv in csvs}
        include_unlabeled = unlabeled_perc > 0.

        self.labels_dict = {i: label for i, label in enumerate(label_names)}

        self.rot_chance = rot_chance
        self.hor_flip = tf.RandomHorizontalFlip(p=hor_flip_chance)
        self.ver_flip = tf.RandomVerticalFlip(p=ver_flip_change)

        self.label_names = label_names

        self.arrs_mapping = {csv: [] for csv in csvs}

        found_labels = set()
        assert len(csvs_to_labels) > 0
        for k, v in csvs_to_labels.items():
            assert os.path.exists(k)
            for l, p in v.items():
                assert len(p) > 0
                found_labels.add(l)
        assert found_labels == set(label_names)

        self.files_labels = []
        self.ds_labels_columns_list = [f'Label_{n}' for n in label_names]
        self.all_df = pd.DataFrame(columns=['id'] + self.ds_labels_columns_list)

        for csv, labels in csvs_to_labels.items():
            c_df = pd.read_csv(csv)

            column_names = set(c_df.columns.values.tolist())
            c_path_getter = CSVS_TO_IM_PATH_GETTERS[csv][0]
            c_id_column = CSVS_TO_IM_PATH_GETTERS[csv][1]

            for ds_label in label_names:
                c_df[f'Label_{ds_label}'] = 0

            for label_name, label_list in labels.items():
                for column in label_list:
                    assert column in column_names

                c_df[f'Label_{label_name}'] = c_df.apply(lambda c_row: int(any([c_row[col] == 1 for col in label_list])), axis=1)

            c_df[c_id_column] = c_df.apply(c_path_getter, axis=1)
            c_df = c_df.rename(columns={c_id_column: 'id'})
            self.all_df = pd.concat([self.all_df, c_df[['id'] + self.ds_labels_columns_list]])

            cur_arr = []
            cur_unlabeled = []

            for i in range(len(c_df)):
                c_path = c_df.iloc[i]['id']
                c_labels = torch.tensor(c_df.iloc[i][self.ds_labels_columns_list].tolist(), dtype=torch.bool)
                if torch.sum(c_labels).item() > 0:
                    cur_arr.append((c_path, c_labels))
                elif include_unlabeled:
                    cur_unlabeled.append((c_path, c_labels))

            if mode == 'train':
                Random(42).shuffle(cur_arr)
                Random(42).shuffle(cur_unlabeled)
                cur_arr = cur_arr[: int(len(cur_arr) * 0.75)]
            elif mode == 'test':
                Random(42).shuffle(cur_arr)
                Random(42).shuffle(cur_unlabeled)
                cur_arr = cur_arr[int(len(cur_arr) * 0.75):]

            self.files_labels.extend(cur_arr)

            if include_unlabeled:
                cur_unlabeled = cur_unlabeled[: int(unlabeled_perc * len(cur_arr))]
                self.files_labels.extend(cur_unlabeled)

        self.all_df[self.ds_labels_columns_list] = self.all_df[self.ds_labels_columns_list].astype(int)
        all_ids = set([p[0] for p in self.files_labels])
        self.all_df = self.all_df[self.all_df['id'].isin(all_ids)]

        self.t_len = len(self.files_labels)
        print(f'Len of dataset: {self.t_len}')

        self.pos_weight = []
        for label_col in self.ds_labels_columns_list:
            num_oc = len(self.all_df[self.all_df[label_col] == 1])
            print(f'Number of cases for {label_col} is {num_oc}')
            self.pos_weight.append((self.t_len - num_oc) / num_oc)
        # self.pos_weight[2] = self.pos_weight[2] / 3.
        # self.pos_weight[3] = self.pos_weight[3] / 1.5
        self.pos_weight = torch.tensor(self.pos_weight).to(DEVICE).float()

    def get_pos_weight(self):
        return self.pos_weight

    def __len__(self):
        return self.t_len

    def apply_transforms(self, item):
        item = torch.tensor(item)
        apply_rot = torch.rand(1).item() < self.rot_chance
        if apply_rot:
            rot_times = random.choice([1, 2, 3])
            item = torch.rot90(item, k=rot_times, dims=[-2, -1])
        item = self.hor_flip(item)
        item = self.ver_flip(item)
        return item

    def __getitem__(self, idx):
        path, label = self.files_labels[idx]
        data = nib.load(path).get_fdata()[None, ...]
        tf_data = self.apply_transforms(data)
        return tf_data, label

    def get_labels(self):
        return '|'.join(self.label_names)

    def get_items_by_label_idx(self, label_idx, count, get_names=False):
        if type(label_idx) in {list, tuple}:
            assert len(label_idx) == len(self.ds_labels_columns_list)
            columns = {self.ds_labels_columns_list[i] for i in range(len(self.ds_labels_columns_list)) if label_idx[i] == 1}
        elif type(label_idx) == int:
            assert 0 <= label_idx < len(self.ds_labels_columns_list)
            columns = {self.ds_labels_columns_list[label_idx]}
        else:
            raise 'Invalid label type given'

        neg_columns = set(self.ds_labels_columns_list).difference(columns)

        all_label_ids = list(self.all_df[reduce(and_, [self.all_df[col].eq(1) for col in columns], self.all_df['id'].notna()) & reduce(and_, [self.all_df[n_col].eq(0) for n_col in neg_columns], self.all_df['id'].notna())]['id'])
        shuffle(all_label_ids)

        items = []
        names = []
        for name in all_label_ids:
            if len(items) == count:
                break
            data = nib.load(name).get_fdata()[None, ...]
            items.append(torch.tensor(data))
            names.append(name)

        items_len = len(items)
        items = torch.stack(items)
        if items_len < count:
            print(f"Warning. Only found {items_len} items with the labels {label_idx}")
        if get_names:
            return items, names
        return items


# =============================================================================
# LONGITUDINAL MIM DATASET (MAIN TRAINING DATASET)
# =============================================================================

class LongitudinalMIMDataset(Dataset):
    """
    Main dataset for Longitudinal Masked Image Modeling training.
    
    Loads baseline/followup CXR pairs with ground truth difference maps.
    Supports multiple data sources:
    - Entity directories: CXR + segmentation mask pairs
    - Inpaint directories: Inpainted abnormality pairs
    - DRR single directories: Single CT DRR variations
    - DRR pair directories: Synthetic BL/FU pairs from CT (main source)
    
    Args:
        entity_dirs: List of directories with CXR + segmentation pairs
        inpaint_dirs: List of directories with inpainted pairs
        DRR_single_dirs: List of directories with DRR variations
        DRR_pair_dirs: List of directories with synthetic DRR pairs
            Expected structure: DRR_dir/case/pair/{prior.nii.gz, current.nii.gz, diff_map.nii.gz}
        abnor_both_p: Probability of adding abnormality to both BL and FU
        invariance: Type of invariance ('abnormality', 'devices', or None)
        overlay_diff_p: Probability of overlaying difference
    
    Returns:
        Tuple of (baseline, followup, ground_truth_diff, followup_mask)
        All tensors are [1, 512, 512] in range [0, 1]
    """
    
    def __init__(self, entity_dirs, inpaint_dirs, DRR_single_dirs, DRR_pair_dirs, abnor_both_p=0.5, invariance=None, overlay_diff_p=0.9):
        self.paths = []
        self.abnor_both_p = abnor_both_p
        self.overlay_diff_p = overlay_diff_p

        for entity_dir in entity_dirs:
            seg_dir = entity_dir + '_segs'
            self.paths.extend([(f'{entity_dir}/{n}', f'{seg_dir}/{n.split(".")[0]}_seg.nii.gz') for n in os.listdir(entity_dir)])

        for inpaint_dir in inpaint_dirs:
            self.paths.extend([(f'{inpaint_dir}/{p}/im1.nii.gz', f'{inpaint_dir}/{p}/im2.nii.gz',
                                f'{inpaint_dir}/{p}/seg1.nii.gz', f'{inpaint_dir}/{p}/seg2.nii.gz',
                                f'{inpaint_dir}/{p}/difference_map.nii.gz') for p in os.listdir(inpaint_dir)])

        for DRR_dir in DRR_single_dirs:
            for case_dir in os.listdir(DRR_dir):
                dir_path = f'{DRR_dir}/{case_dir}'
                # self.paths.extend([(f'{dir_path}/var{i}.nii.gz', f'{dir_path}/var{i+1}.nii.gz') for i in range(1389)])
                self.paths.extend([(f'{dir_path}/var{i}.nii.gz', f'{dir_path}/var{i+1}.nii.gz') for i in range(600)])

        for DRR_dir in DRR_pair_dirs:
            for case_dir in os.listdir(DRR_dir):
                case_dir_abs = f'{DRR_dir}/{case_dir}'
                print(case_dir_abs)
                for pair_dir in os.listdir(case_dir_abs):
                    dir_path = f'{DRR_dir}/{case_dir}/{pair_dir}'
                    if os.path.exists(f'{dir_path}/diff_map.nii.gz'):
                        self.paths.append((f'{dir_path}/prior.nii.gz', f'{dir_path}/current.nii.gz', f'{dir_path}/diff_map.nii.gz'))
                    elif os.path.exists(f'{dir_path}/difference_map.nii.gz'):
                        self.paths.append((f'{dir_path}/prior.nii.gz', f'{dir_path}/current.nii.gz', f'{dir_path}/difference_map.nii.gz'))
                    elif os.path.exists(f'{dir_path}/difference_map.nii.gz'):
                        self.paths.append((f'{dir_path}/prior.nii.gz', f'{dir_path}/current.nii.gz', f'{dir_path}/difference_map.nii.gz'))
                    else:
                        print(f'{dir_path}/diff_map.nii.gz')
                        continue
                    # current_with_differences
        self.t_len = len(self.paths)

        if invariance is None:
            self.random_abnormalization_tf = RandomAbnormalizationTransform(lung_abnormalities=True, devices=True, size=768, none_chance_to_update=0.13)
            # self.random_abnormalization_tf = RandomAbnormalizationTransform(lung_abnormalities=True, devices=False, size=768, none_chance_to_update=0.2)
            self.has_inv = False
        elif invariance == 'abnormality':
            self.random_abnormalization_tf = RandomAbnormalizationTransform(lung_abnormalities=False, devices=True, size=768, none_chance_to_update=0.75)
            self.random_abnormalization_tf_inv = RandomAbnormalizationTransform(lung_abnormalities=True, devices=False, size=512, none_chance_to_update=0.13)
            self.has_inv = True
        elif invariance == 'devices':
            self.random_abnormalization_tf = RandomAbnormalizationTransform(lung_abnormalities=True, devices=False, size=768, none_chance_to_update=0.13)
            self.random_abnormalization_tf_inv = RandomAbnormalizationTransform(lung_abnormalities=False, devices=True, size=512, none_chance_to_update=0.75)
            self.has_inv = True
        self.random_affine_tf = RandomAffineWithMaskTransform()
        self.random_bspline_tf = RandomBsplineAndSimilarityWithMaskTransform()
        self.random_affine_tf_inpaint = RandomAffineWithMaskTransform(scale_x_p=0.25, scale_y_p=0.25, trans_x_p=0.25, trans_y_p=0.25)
        self.random_bspline_tf_inpaint = RandomBsplineAndSimilarityWithMaskTransform(rot_p=0.3, scale_y_p=0.175, scale_x_p=0.175, trans_y_p=0.175, trans_x_p=0.175)
        self.crop_resize_with_mask_tf = CropResizeWithMaskTransform()
        self.rescale_values_tf = RescaleValuesTransform()
        self.random_intensity_tf = RandomIntensityTransform(clahe_p=0.25, clahe_clip_limit=(0.75, 2.5), blur_p=0., jitter_p=0.35)
        # self.random_pairwise_intensity_tf = PairwiseRandomIntensityTransform(clahe_p=0.5, clahe_range=(0.75, 2.5), clahe_var=0.15)
        self.fu_tf = v2.RandomChoice([self.random_affine_tf, self.random_bspline_tf], p=[0.2, 0.8])
        self.bl_tf = v2.RandomChoice([self.crop_resize_with_mask_tf, self.random_bspline_tf, self.random_affine_tf], p=[0.85, 0.1, 0.05])
        self.tf_inpaint = v2.RandomChoice([self.crop_resize_with_mask_tf, self.random_bspline_tf_inpaint, self.random_affine_tf_inpaint], p=[0.3, 0.15, 0.55])
        # self.end_tf = v2.Compose([self.rescale_values_tf, self.random_photometric_tf])
        self.random_bl_fu_flip_tf = RandomFlipBLWithFU(p=0.5)
        self.random_channels_flip_tf = RandomChannelsFlip(p=0.5)

        self.resize_dict = {512: v2.Resize((512, 512)), 768: v2.Resize((768, 768))}

        self.return_label = False

        self.target_ornt = axcodes2ornt(('R', 'A', 'S'))

    def __len__(self):
        return self.t_len

    def get_entity_pair(self, paths):
        print(paths)
        img_path, seg_path = paths

        img = torch.tensor(nib.load(img_path).get_fdata().T[None, ...])
        mask = torch.tensor(nib.load(seg_path).get_fdata().T[None, ...])
        # mask = torch.tensor(nib.load(seg_path).get_fdata().T)

        abnor = self.random_abnormalization_tf(img.clone(), mask)

        if random.random() < self.abnor_both_p:
            img = self.random_abnormalization_tf(img, mask)

        # overlaid1, overlaid2 = self.random_abnormalization_tf(img, mask)
        # cat_img = torch.cat([overlaid1, overlaid2, mask], dim=0)

        cat_img = torch.cat([img, abnor, mask], dim=0)

        bl, bl_mask = self.bl_tf(cat_img)
        fu, fu_mask = self.fu_tf(cat_img)

        bl = self.rescale_values_tf(bl)
        fu = self.rescale_values_tf(fu)

        # bl, fu = self.random_pairwise_intensity_tf(bl, fu)

        bl, fu, bl_mask, fu_mask = self.random_bl_fu_flip_tf(bl, fu, bl_mask, fu_mask)
        bl, fu = self.random_channels_flip_tf(bl, fu)

        # if random.random() < 0.84:
        if random.random() < self.overlay_diff_p:
            bl = bl[0: 1]
            fu_gt = fu[0: 1]
            fu = fu[1: 2]
        else:
            bl = bl[1: 2]
            fu_gt = fu[1: 2]
            fu = fu[1: 2]
        gt = fu - fu_gt

        if self.has_inv and random.random() < 0.5:
            bl = self.rescale_values_tf(self.random_abnormalization_tf_inv(bl * 255., bl_mask))
            fu = self.rescale_values_tf(self.random_abnormalization_tf_inv(fu * 255., fu_mask))

        bl = self.random_intensity_tf(bl)
        fu = self.random_intensity_tf(fu)

        return bl, fu, gt, fu_mask

    def get_inpaint_pair(self, paths):
        im1_path, im2_path, seg1_path, seg2_path, diff_im_path = paths

        #TODO: Check if .T is needed
        im1 = torch.tensor(nib.load(im1_path).get_fdata().T[None, ...])
        im2 = torch.tensor(nib.load(im2_path).get_fdata().T[None, ...])
        seg1 = torch.tensor(nib.load(seg1_path).get_fdata().T[None, ...])
        seg2 = torch.tensor(nib.load(seg2_path).get_fdata().T[None, ...])

        if diff_im_path.endswith('.pt'):
            diff_im = torch.zeros((1, im2.shape[-2], im2.shape[-1]), dtype=torch.float16)
        else:
            diff_im = torch.tensor(nib.load(diff_im_path).get_fdata().T[None, ...])

        #TODO: Add low prob option to add entities

        if im1.shape[-1] != seg1.shape[-1]:
            seg1 = self.resize_dict[im1.shape[-1]](seg1)
        if im2.shape[-1] != seg2.shape[-1]:
            seg2 = self.resize_dict[im2.shape[-1]](seg2)
        if im2.shape[-1] != diff_im.shape[-1]:
            diff_im = self.resize_dict[im2.shape[-1]](diff_im)

        cat_im1 = torch.cat([im1, seg1])
        cat_im2 = torch.cat([im2, diff_im, seg2])

        im1, seg1 = self.tf_inpaint(cat_im1)
        cat_im2, seg2 = self.tf_inpaint(cat_im2)
        diff_im = cat_im2[1].unsqueeze(0)
        im2 = cat_im2[0].unsqueeze(0)

        im1 = self.rescale_values_tf(im1)
        im2 = self.rescale_values_tf(im2)

        im1 = self.random_intensity_tf(im1)
        im2 = self.random_intensity_tf(im2)

        max_abs_diff_val = torch.max(diff_im.abs()).item()
        if max_abs_diff_val > 0.25:
            new_max_val = get_max_inpaint_diff_val(max_abs_diff_val)
            scale_fac = random.random() * 0.7 + 0.75
            diff_im = scale_and_suppress_non_max(diff_im, new_upper_bound=new_max_val, scale_fac=scale_fac)
        elif max_abs_diff_val > 0.05:
            # new_max_val = max_abs_diff_val * 0.652
            new_max_val = max_abs_diff_val * 0.788
            scale_fac = random.random() * 0.7 + 0.75
            diff_im = scale_and_suppress_non_max(diff_im, new_upper_bound=new_max_val, scale_fac=scale_fac)

        if self.return_label:
            if 'inpainted_abnormal_disordered/' in im1_path:
                return im1, im2, diff_im, 0
            elif 'inpainted_abnormal_lower/' in im1_path:
                return im1, im2, diff_im, 1
            elif 'inpainted_healthy/' in im1_path:
                return im1, im2, diff_im, 2
            elif 'inpainted_healthy_abnormal_disordered/' in im1_path:
                return im1, im2, diff_im, 3
            elif 'inpainted_healthy_abnormal_lower/' in im1_path:
                return im1, im2, diff_im, 4
            else:
                raise Exception()

        return im1, im2, diff_im, seg2

    def get_DRR_single_pair(self, paths):
        im1_path, im2_path = paths

        im1 = torch.tensor(nib.load(im1_path).get_fdata().T[None, ...])
        im2 = torch.tensor(nib.load(im2_path).get_fdata().T[None, ...])
        seg2 = torch.ones_like(im2, dtype=torch.bool)
        diff_map = torch.zeros_like(im1, dtype=torch.float)

        # p = random.random()
        #
        # if p < 0.425:
        #     im1 = self.rescale_values_tf(torch.tensor(image_histogram_equalization(im1.numpy())[0]))
        #     im2 = self.rescale_values_tf(torch.tensor(image_histogram_equalization(im2.numpy())[0]))
        # elif p < 0.65:
        #     im1 = self.random_intensity_tf(self.rescale_values_tf(im1))
        #     im2 = self.random_intensity_tf(self.rescale_values_tf(im2))
        # else:
        #     im1 = self.rescale_values_tf(im1)
        #     im2 = self.rescale_values_tf(im2)

        return im1, im2, diff_map, seg2

    def get_DRR_pair(self, paths):
        im1_path, im2_path, diff_path = paths

        im1 = torch.tensor(nib.load(im1_path).get_fdata().T[None, ...])
        im2 = torch.tensor(nib.load(im2_path).get_fdata().T[None, ...])
        seg2 = torch.ones_like(im2, dtype=torch.bool)
        diff_map = torch.tensor(nib.load(diff_path).get_fdata().T[None, ...])

        # nif_prior = nib.load(im1_path)
        # nif_current = nib.load(im2_path)
        # nif_diff = nib.load(diff_path)

        # nif_prior = nib.Nifti1Image(nif_prior.get_fdata()[:, :, np.newaxis], nif_prior.affine)
        # nif_current = nib.Nifti1Image(nif_current.get_fdata()[:, :, np.newaxis], nif_current.affine)
        # nif_diff = nib.Nifti1Image(nif_diff.get_fdata()[:, :, np.newaxis], nif_diff.affine)
        #
        # nif_prior = as_closest_canonical(nif_prior)
        # nif_current = as_closest_canonical(nif_current)
        # nif_diff = as_closest_canonical(nif_diff)

        # im1 = torch.tensor(np.flip(np.squeeze(nif_prior.get_fdata()), axis=[0, 1]).copy()).T[None, ...]
        # im2 = torch.tensor(np.flip(np.squeeze(nif_current.get_fdata()), axis=[0, 1]).copy()).T[None, ...]
        # diff_map = torch.tensor(np.flip(np.squeeze(nif_diff.get_fdata()), axis=[0, 1]).copy()).T[None, ...]
        # seg2 = torch.ones_like(im2, dtype=torch.bool)

        return im1, im2, diff_map, seg2
        # return im1, im2, diff_map, name

    # def get_DRR_pair(self, paths):
    #     im1_path, im2_path, diff_path = paths
    #
    #     nif1 = nib.load(im1_path)
    #     nif2 = nib.load(im2_path)
    #     im1 = nif1.get_fdata()
    #     im2 = nif2.get_fdata()
    #     to_flip = []
    #     if nif1.affine[0][0] > 0:
    #         to_flip.append(1)
    #     if nif1.affine[2][2] > 0:
    #         to_flip.append(0)
    #     if to_flip:
    #         im1 = np.flip(im1, axis=1)
    #         im2 = np.flip(im2, axis=1)
    #     im1 = torch.tensor(im1.copy()).T[None, ...]
    #     im2 = torch.tensor(im2.copy()).T[None, ...]
    #     seg2 = torch.ones_like(im2, dtype=torch.bool)
    #     diff_map = torch.tensor(nib.load(diff_path).get_fdata().T[None, ...])
    #
    #     return im1, im2, diff_map, seg2

    def __getitem__(self, idx):
        paths = self.paths[idx]
        if self.return_label:
            bl, fu, gt, diff_label = self.get_DRR_pair(paths)
            label = 3
        else:
            bl, fu, gt, fu_mask = self.get_DRR_pair(paths)
        if self.return_label:
            return bl, fu, gt, label, diff_label

        return bl, fu, gt, fu_mask

        if 'Chexpert' in paths[0]:
            if self.return_label:
                bl, fu, gt, diff_label = self.get_inpaint_pair(paths)
                label = 1
            else:
                bl, fu, gt, fu_mask = self.get_inpaint_pair(paths)
        elif 'DRRs' in paths[0]:
            if self.return_label:
                bl, fu, gt, diff_label = self.get_DRR_single_pair(paths)
                label = 2
            else:
                bl, fu, gt, fu_mask = self.get_DRR_single_pair(paths)
        elif 'final' in paths[0] or 'synthetic_pairs' in paths[0] or 'consolidation' in paths[0] or 'pleural' in paths[0] or 'pneumo' in paths[0] or 'fluid' in paths[0]:
            if self.return_label:
                bl, fu, gt, diff_label = self.get_DRR_pair(paths)
                label = 3
            else:
                bl, fu, gt, fu_mask = self.get_DRR_pair(paths)
        else:
            if self.return_label:
                bl, fu, gt, diff_label = self.get_entity_pair(paths)
                label = 0
            else:
                bl, fu, gt, fu_mask = self.get_entity_pair(paths)

        if self.return_label:
            return bl, fu, gt, label, diff_label

        return bl, fu, gt, fu_mask

        # return bl, fu, gt, fu_mask, img_path.split('/')[-1], bl_mask

    def shuffle(self):
        shuffle(self.paths)

    def reorient_to_standard(self, nifti_img):
        data = nifti_img.get_fdata()
        affine = nifti_img.affine

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        # Get current orientation
        current_ornt = nib.orientations.io_orientation(affine)

        # Compute the transform to the target orientation
        transform = ornt_transform(current_ornt, self.target_ornt)

        # Apply the transform to the data array
        reoriented_data = apply_orientation(data, transform)

        return np.squeeze(reoriented_data)


# if __name__ == '__main__':
#     a = LongitudinalMIMDataset(['images'], abnor_both_p=1.)
#     c_bl, c_fu, c_gt, __, n, ___ = a[0]
#     plt.imshow(c_bl.squeeze(), cmap='gray')
#     plt.show()
#     plt.imshow(c_fu.squeeze(), cmap='gray')
#     plt.show()
#     imm = plt.imshow(c_gt.squeeze())
#     cbar = plt.colorbar(imm)
#     plt.show()
