import os
import nibabel as nib
import torch
from scipy.ndimage import label, binary_closing
from constants import DEVICE, USE_MASK_TOKEN, USE_PATCH_DEC, USE_POS_EMBED
import numpy as np
import numpy.random as np_rd
import random
from datasets import LongitudinalMIMDataset
from models import LongitudinalMIMModelBig
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from tqdm import tqdm
from torchvision.transforms.v2.functional import adjust_sharpness
import kornia
import json
from skimage.morphology import disk


def create_violin_plot(pos_arrs, neg_arrs, title, y_label, save_path):
    ticks = [m + 1 for m in range(len(pos_arrs))]
    maxes, mins = [], []

    pos_parts = plt.violinplot(pos_arrs, showmeans=True, showextrema=True, showmedians=False, side='low')
    neg_parts = plt.violinplot(neg_arrs, showmeans=True, showextrema=True, showmedians=False, side='high')

    print(f'Creating violin plot for {title} for {save_path}')

    for parts, arrs, col, std_offset in [(pos_parts, pos_arrs, 'red', -0.12), (neg_parts, neg_arrs, 'green', 0.12)]:
        for k, v in parts.items():
            if k == 'bodies':
                for i, pc in enumerate(v):
                    pc.set_facecolor(col)
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.4)

                    c_mean = np.mean(arrs[i])
                    c_std = np.std(arrs[i])
                    print(f'Cur mean: {c_mean}')
                    print(f'Cur std: {c_std}')
                    print()
                    maxes.append(c_mean + c_std + 0.05)
                    mins.append(c_mean - c_std - 0.05)
                    plt.hlines(c_mean - c_std, i + 1 + std_offset, i + 1, color='blue', linewidth=1.6, linestyles='dashed')
                    plt.hlines(c_mean + c_std, i + 1 + std_offset, i + 1, color='blue', linewidth=1.6, linestyles='dashed')
            else:
                v.set_color('black')

    max_val = max(max(maxes), 1.)
    min_val = min(min(mins), 0.)
    plt.ylim(min_val, max_val)

    plt.title(title)
    plt.xticks(ticks=ticks, labels=['Low', 'Medium', 'High'], rotation=12, fontsize=10)
    plt.xlabel('Change magnitude', fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    legend_patches = [
        Patch(facecolor='red', edgecolor='black', label='Positive'),
        Patch(facecolor='green', edgecolor='black', label='Negative')
    ]
    plt.legend(handles=legend_patches, loc='best')

    pos_means = [sum(a) / len(a) for a in pos_arrs]
    neg_means = [sum(a) / len(a) for a in neg_arrs]
    plt.plot(ticks, pos_means, linestyle=(2.5, (5, 10)), linewidth=1.2, alpha=0.85, color='red')
    plt.plot(ticks, neg_means, linestyle=(7.5, (5, 10)), linewidth=1.2, alpha=0.85, color='green')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close()


def consolidation_magnitude(gt):
    max_val = np.amax(gt)
    area_pos = np.sum(gt > 0)
    if max_val == 0:
        max_val, area_pos = None, None

    min_val = np.abs(np.amin(gt))
    area_neg = np.sum(gt < 0)
    if min_val == 0:
        min_val, area_neg = None, None

    return (max_val, area_pos), (min_val, area_neg)

def pleural_effusion_magnitude(gt):
    area_pos = np.sum(gt > 0)
    if area_pos == 0:
        area_pos = None

    area_neg = np.sum(gt < 0)
    if area_neg == 0:
        area_neg = None

    return area_pos, area_neg


def pneumothorax_magnitude(gt):
    area_pos = np.sum(gt > 0)
    if area_pos == 0:
        area_pos = None

    area_neg = np.sum(gt < 0)
    if area_neg == 0:
        area_neg = None

    return area_pos, area_neg


def fluid_overload_magnitude(gt):
    max_val = np.amax(gt)
    if max_val == 0:
        max_val = None

    min_val = np.abs(np.amin(gt))
    if min_val == 0:
        min_val = None

    return max_val, min_val


def main():
    entity_name = 'consolidation'
    case_dirs = [
        f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/{entity_name}_test/results/{entity_name}_rotation_invariance_angles_15_15_15',
        # f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/{entity_name}_test/results/{entity_name}_rotation_invariance_angles_8_8_8'
    ]
    save_dir = f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/{entity_name}_test/results/magnitude_effects'

    mag_func_dict = {'consolidation': consolidation_magnitude, 'pleural_effusion': pleural_effusion_magnitude, 'pneumothorax': pneumothorax_magnitude, 'fluid_overload': fluid_overload_magnitude}
    double_mag_entities = {'consolidation'}

    mag_dict_pos = {}
    mag_dict_neg = {}
    if entity_name in double_mag_entities:
        mag_dict2_pos = {}
        mag_dict2_neg = {}
    magnitude_func = mag_func_dict[entity_name]

    for case_dir in case_dirs:
        for pair_name in tqdm(os.listdir(case_dir)):
            pair_p = f'{case_dir}/{pair_name}'
            gt_p = f'{pair_p}/gt.nii.gz'
            gt = nib.load(gt_p).get_fdata()

            mag = magnitude_func(gt)

            if entity_name in double_mag_entities:
                mag_pos, mag_pos2 = mag[0]
                mag_neg, mag_neg2 = mag[1]
                if mag_pos is not None:
                    mag_dict_pos[pair_p], mag_dict2_pos[pair_p] = mag_pos, mag_pos2
                if mag_neg is not None:
                    mag_dict_neg[pair_p], mag_dict2_neg[pair_p] = mag_neg, mag_neg2
            else:
                mag_pos, mag_neg = mag
                if mag_pos is not None:
                    mag_dict_pos[pair_p] = mag_pos
                if mag_neg is not None:
                    mag_dict_neg[pair_p] = mag_neg

    low_th_pos, med_th_pos = np.percentile(np.array(list(mag_dict_pos.values())), [33, 66])
    low_th_neg, med_th_neg = np.percentile(np.array(list(mag_dict_neg.values())), [33, 66])
    if entity_name in double_mag_entities:
        low_th2_pos, med_th2_pos = np.percentile(np.array(list(mag_dict2_pos.values())), [33, 66])
        low_th2_neg, med_th2_neg = np.percentile(np.array(list(mag_dict2_neg.values())), [33, 66])
    recalls_pos = [[], [], []]
    recalls_neg = [[], [], []]
    precisions_pos = [[], [], []]
    precisions_neg = [[], [], []]

    if entity_name in double_mag_entities:
        for pair_p in tqdm(set(mag_dict_pos).union(set(mag_dict_neg))):
            add_pos = True
            add_neg = True

            if pair_p in mag_dict_pos:
                mag_pos = mag_dict_pos[pair_p]
                mag2_pos = mag_dict2_pos[pair_p]
            else:
                add_pos = False
                mag_pos = float('inf')
                mag2_pos = float('inf')

            if pair_p in mag_dict_neg:
                mag_neg = mag_dict_neg[pair_p]
                mag2_neg = mag_dict2_neg[pair_p]
            else:
                add_neg = False
                mag_neg = float('inf')
                mag2_neg = float('inf')
            
            if mag_pos < low_th_pos and mag2_pos < low_th2_pos:
                c_idx_pos = 0
            elif mag_pos > med_th_pos and mag2_pos > med_th2_pos:
                c_idx_pos = 2
            elif low_th_pos <= mag_pos <= med_th_pos and low_th2_pos <= mag2_pos <= med_th2_pos:
                c_idx_pos = 1
            else:
                add_pos = False
                
            if mag_neg < low_th_neg and mag2_neg < low_th2_neg:
                c_idx_neg = 0
            elif mag_neg > med_th_neg and mag2_neg > med_th2_neg:
                c_idx_neg = 2
            elif low_th_neg <= mag_neg <= med_th_neg and low_th2_neg <= mag2_neg <= med_th2_neg:
                c_idx_neg = 1
            else:
                add_neg = False

            json_path = f'{pair_p}/params.json'
            with open(json_path) as f:
                stats_dict = json.load(f)

                recall_pos = stats_dict['Recall (pos)']
                recall_neg = stats_dict['Recall (neg)']
                precision_pos = stats_dict['Precision (pos)']
                precision_neg = stats_dict['Precision (neg)']
                
                if add_pos:
                    recalls_pos[c_idx_pos].append(recall_pos)
                    precisions_pos[c_idx_pos].append(precision_pos)
                if add_neg:
                    recalls_neg[c_idx_neg].append(recall_neg)
                    precisions_neg[c_idx_neg].append(precision_neg)
    else:
        for pair_p in tqdm(set(mag_dict_pos).union(set(mag_dict_neg))):
            add_pos = True
            add_neg = True

            if pair_p in mag_dict_pos:
                mag_pos = mag_dict_pos[pair_p]
            else:
                add_pos = False
                mag_pos = float('inf')

            if pair_p in mag_dict_neg:
                mag_neg = mag_dict_neg[pair_p]
            else:
                add_neg = False
                mag_neg = float('inf')
            
            if mag_pos < low_th_pos:
                c_idx_pos = 0
            elif mag_pos > med_th_pos:
                c_idx_pos = 2
            else:
                c_idx_pos = 1
                
            if mag_neg < low_th_neg:
                c_idx_neg = 0
            elif mag_neg > med_th_neg:
                c_idx_neg = 2
            else:
                c_idx_neg = 1

            json_path = f'{pair_p}/params.json'
            with open(json_path) as f:
                stats_dict = json.load(f)

                recall_pos = stats_dict['Recall (pos)']
                recall_neg = stats_dict['Recall (neg)']
                precision_pos = stats_dict['Precision (pos)']
                precision_neg = stats_dict['Precision (neg)']

                if add_pos:
                    recalls_pos[c_idx_pos].append(recall_pos)
                    precisions_pos[c_idx_pos].append(precision_pos)
                if add_neg:
                    recalls_neg[c_idx_neg].append(recall_neg)
                    precisions_neg[c_idx_neg].append(precision_neg)

    print(len(recalls_pos[0]), len(recalls_pos[1]), len(recalls_pos[2]))
    print(len(recalls_neg[0]), len(recalls_neg[1]), len(recalls_neg[2]))
    print(len(precisions_pos[0]), len(precisions_pos[1]), len(precisions_pos[2]))
    print(len(precisions_neg[0]), len(precisions_neg[1]), len(precisions_neg[2]))

    os.makedirs(save_dir, exist_ok=True)
    create_violin_plot(recalls_pos, recalls_neg, title='Recall (per image)', y_label='Recall', save_path=f'{save_dir}/violin_recall.png')
    create_violin_plot(precisions_pos, precisions_neg, title='Precision (per image)', y_label='Precision', save_path=f'{save_dir}/violin_precision.png')


if __name__ == '__main__':
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))

    main()

