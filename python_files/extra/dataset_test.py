import os

from datasets import ContrastiveLearningDataset, ContrastiveCXRSampler, NormalAbnormalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from constants import *
from collections import Counter

if __name__ == '__main__':
    # train_dataset = ContrastiveLearningDataset(CXR14_FOLDER + '/AP_images', CXR14_FOLDER + '/train_labels.csv', CUR_LABELS,
    #                                            groups=CUR_LABEL_GROUPS, do_shuffle=True, get_weights=GET_WEIGHTS, label_no_finding=LABEL_NO_FINDING)
    # train_sampler = ContrastiveCXRSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, sampler=train_sampler)
    # print(len(train_dataset))
    #
    # samples_dict = Counter()
    # names_dict = {}
    #
    # for i, batch in tqdm(enumerate(train_dataloader)):
    #     inputs, labels, name = batch
    #     k = str(labels.tolist())
    #     samples_dict[k] += 1
    #     if k not in names_dict:
    #         names_dict[k] = [name]
    #     elif len(names_dict[k]) < 3:
    #         names_dict[k].append(name)
    #     else:
    #         continue
    #
    # print(samples_dict)
    # print("###################")
    # print(names_dict)
    img_pixels = 512 * 512
    ds = NormalAbnormalDataset()
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    outliers = []
    for item in tqdm(dl):
        data, label, path = item
        path = path[0]
        min_v = torch.min(data)
        perc_min_v = torch.sum(data == min_v).item() / img_pixels
        if 'VinDrCXR' not in path:
            continue
        if 'pneumonia_normal' in path and perc_min_v > 0.3:
            print(f"Adding outlier {path} due to perc_min_v {perc_min_v}")
            outliers.append(path)
            continue
        elif 'pneumonia_normal' not in path and perc_min_v > 0.25:
            print(f"Adding outlier {path} due to perc_min_v {perc_min_v}")
            outliers.append(path)
            continue
        data_std = torch.std(data).item()
        if data_std < 5:
            print(f"Adding outlier {path} due to std {data_std}")
            outliers.append(path)

    print(f'found {len(outliers)} outliers')

    for out_path in tqdm(outliers):
        out_name = out_path.split('/')[-1]
        if 'pneumonia_normal' in out_path:
            continue
            # os.symlink(out_path, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/to_crop/pneumonia/{out_name}')
        elif 'PadChest' in out_path:
            continue
            # os.symlink(out_path, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/to_crop/padchest/{out_name}')
        elif 'VinDrCXR' in out_path and 'train' in out_path:
            os.symlink(out_path, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/to_crop/vindr_train/{out_name}')
        elif 'VinDrCXR' in out_path and 'test' in out_path:
            os.symlink(out_path, f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/to_crop/vindr_test/{out_name}')
        else:
            print(f'Found weird outlier path: {out_path}')
