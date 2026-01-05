import numpy as np
import json
import nibabel as nib
from skimage.draw import ellipse
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.ndimage import label
from itertools import combinations
import torchvision.transforms.v2 as v2


# One time use function
def fix_annotation_paths(base_dir):
    for physician_n in os.listdir(base_dir):
        c_phys_p = f'{base_dir}/{physician_n}'
        for pair_n in os.listdir(c_phys_p):
            c_pair_p = f'{c_phys_p}/{pair_n}'

            pair_num = int(''.join([c for c in pair_n if c.isdigit()]))
            new_pair_p = f'{c_phys_p}/{pair_num}.json'

            print(f'Mapping {c_pair_p} -> {new_pair_p}')
            os.rename(c_pair_p, new_pair_p)


def load_xray(file_path: str):
    xray_nif = nib.load(file_path)

    global affine
    affine = xray_nif.affine
    xray_data = xray_nif.get_fdata()

    return xray_data


def save_arr_as_nifti(arr, output_path):
    xray_nif = nib.Nifti1Image(arr, affine)
    nib.save(xray_nif, output_path)


def generate_alpha_map(x):
    x_abs = np.abs(x)
    max_val = max(np.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val

    return alphas_map


def plot_diff_on_current(c_diff_map, c_current, out_p):
    alphas = generate_alpha_map(c_diff_map)
    divnorm = colors.TwoSlopeNorm(vmin=min(np.min(c_diff_map).item(), -0.01), vcenter=0., vmax=max(np.max(c_diff_map).item(), 0.01))
    fig, ax = plt.subplots()
    ax.imshow(c_current.squeeze(), cmap='gray')
    imm1 = ax.imshow(c_diff_map.squeeze(), alpha=alphas, cmap=differential_grad, norm=divnorm)
    cbar1 = plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()

    plt.savefig(out_p)

    # ax[0].clear()
    plt.cla()
    plt.clf()
    plt.close()


def plot_matrix(matrix_df: pd.DataFrame, output_path: str, title='Correlation Matrix'):
    """
    Plots an N x N  matrix as a heatmap with clear row/column labels
    and saves the figure as a PNG file.

    Args:
        matrix_df (pd.DataFrame): The N x N matrix, where
                                        the index and columns contain the names.
        output_path (str): The full path and filename (e.g., 'path/to/matrix.png')
                           to save the plot.
    """
    N = matrix_df.shape[0]
    figsize = max(8, int(N * 0.6))
    plt.figure(figsize=(figsize, figsize))

    # 2. Create the Heatmap using Seaborn
    # sns.heatmap is ideal for this task.
    # annot=True shows the correlation values on the plot.
    # fmt=".2f" formats the values to two decimal places.
    # cmap="vlag" is a good diverging color map for correlations (from -1 to 1).
    # linewidths=0.5 adds borders for clarity.
    # cbar_kws fine-tunes the color bar appearance.
    ax = sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=0,
        vmax=1,
        center=0.25,
        linewidths=0.,
        linecolor='black',
        cbar_kws={'shrink': 1., 'aspect': 20, 'label': 'PAI'},
        annot_kws={"fontsize": 14, 'fontweight': 'bold'}
    )

    # 2. Add Visual Separators for the Last Row/Column and sub titles
    # These lines visually "detach" the last variable from the rest
    separator_index = N - 1

    # Add vertical line to separate the last column
    ax.axvline(x=separator_index, ymin=1 / N, color='black', linewidth=1.25, linestyle='--')

    # Add horizontal line to separate the last row
    ax.axhline(y=separator_index, xmax=1 - 1 / N, color='black', linewidth=1.25, linestyle='--')

    group1_x = (N - 1) / 2
    group2_x = (N - 1) + 0.5
    text_y_pos = -0.04

    # Plot Group 1 Label
    ax.text(
        group1_x, text_y_pos,
        'H - H',
        ha='center', va='bottom',
        fontsize=14, fontweight='bold', fontstyle='italic', color='#333333'
    )

    # Plot Group 2 Label
    ax.text(
        group2_x, text_y_pos,
        'M - H',
        ha='center', va='bottom',
        fontsize=14, fontweight='bold', fontstyle='italic', color='#333333'  # Matching the target column color
    )

    # 3. Apply the Row/Column Names
    # The names are automatically taken from the DataFrame's index/columns.
    # Adjusting tick label properties for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')  # Rotate x-axis labels
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')  # Keep y-axis labels horizontal

    # target_color = "#2800FF"
    #
    # for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
    #     for i, c_label in enumerate(labels):
    #         if i == N - 1:
    #             c_label.set_weight('bold')
    #             c_label.set_color(target_color)

    # 4. Set Title and Layout
    plt.title(f'{title}', fontsize=16, fontweight='bold', pad=30)
    # tight_layout prevents labels from being cut off
    plt.tight_layout()

    # 5. Save the plot to the specified path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # 6. Close the figure to free up memory
    plt.close()

    print(f"Matrix plot saved successfully to: **{output_path}**")


def plot_curves(pos_arr, neg_arr, n, out_p):
    x_values = np.arange(1, len(pos_arr) + 1)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 5)  # Standard width for documents
    })

    fig, ax = plt.subplots()

    ax.plot(x_values, pos_arr,
            label=f'Positive changes',
            color='#003366',  # Formal Navy
            linestyle='-',
            linewidth=2.0,
            marker='o',  # Markers help distinguish specific data points
            markersize=6)

    ax.plot(x_values, neg_arr,
            label=f'Negative changes',
            color='#800000',      # Formal Maroon
            linestyle='--',
            linewidth=2.0,
            marker='s',
            markersize=6)

    ax.set_ylim(0, 1)
    ax.set_xticks(x_values)
    ax.set_xlim(0.5, len(pos_arr) + 0.5)

    ax.set_xlabel('Consensus Level')
    ax.set_ylabel('Sensitivity')
    ax.set_title(f'Sensitivity at Consensus Levels ({n})', pad=15)

    ax.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')

    for x, y in zip(x_values, pos_arr):
        ax.annotate(f'{y:.2f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8), # (0, -14) prev
                    ha='center',
                    fontsize=9,
                    color='#003366')

    # Annotate Curve 2 (Below points to minimize overlap)
    for x, y in zip(x_values, neg_arr):
        ax.annotate(f'{y:.2f}',
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, -14),
                    ha='center',
                    fontsize=9,
                    color='#800000')

    plt.tight_layout()

    plt.savefig(out_p)

    plt.clf()
    plt.cla()
    plt.close()


def plot_curves_multiple(pos_arrs_d: dict, neg_arrs_d: dict, arr_len, out_p):
    x_values = np.arange(1, arr_len + 1)

    colors_pos = ['#0b5394', '#990000', '#38761d', '#b45f06', '#741b47']
    colors_neg = ['#3d85c6', '#cc0000', '#6aa84f', '#e69138', '#a64d79']

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 5)  # Standard width for documents
    })

    fig, ax = plt.subplots()
    
    for k, n in enumerate(pos_arrs_d.keys()):
        pos_arr = pos_arrs_d[n]
        neg_arr = neg_arrs_d[n]
        
        ax.plot(x_values, pos_arr,
                label=f'Positive changes {n}',
                color=colors_pos[k],  # Formal Navy
                linestyle='-',
                linewidth=2.0,
                marker='o',  # Markers help distinguish specific data points
                markersize=6)
    
        ax.plot(x_values, neg_arr,
                label=f'Negative changes {n}',
                color=colors_neg[k],      # Formal Maroon
                linestyle='--',
                linewidth=2.0,
                marker='s',
                markersize=6)

    ax.set_ylim(0, 1)
    ax.set_xticks(x_values)
    ax.set_xlim(0.5, arr_len + 0.5)

    ax.set_xlabel('Consensus Level')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity at Consensus Levels (Clinicians)', pad=15)

    ax.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')

    # for x, y in zip(x_values, pos_arr):
    #     ax.annotate(f'{y:.2f}',
    #                 (x, y),
    #                 textcoords="offset points",
    #                 xytext=(0, -14), # (0, 8) prev
    #                 ha='center',
    #                 fontsize=9,
    #                 color='#003366')
    # 
    # # Annotate Curve 2 (Below points to minimize overlap)
    # for x, y in zip(x_values, neg_arr):
    #     ax.annotate(f'{y:.2f}',
    #                 (x, y),
    #                 textcoords="offset points",
    #                 xytext=(0, -14),
    #                 ha='center',
    #                 fontsize=9,
    #                 color='#800000')

    plt.tight_layout()

    plt.savefig(out_p)

    plt.clf()
    plt.cla()
    plt.close()


def load_labels_map(json_path, shape, label_mapping_dict=None):
    if label_mapping_dict is None:
        label_mapping_dict = {
            ('Appearance', None, None): 3,
            ('Disappearance', None, None): -3,
            ('Persistence', 'Increase', 'Increase'): 2,
            ('Persistence', 'Decrease', 'Decrease'): -2,
            ('Persistence', 'Increase', 'None'): 1,
            ('Persistence', 'Decrease', 'None'): -1,
            ('Persistence', 'None', 'Increase'): 1,
            ('Persistence', 'None', 'Decrease'): -1,
            ('Persistence', 'None', 'None'): 0,
            ('Persistence', 'Increase', 'Decrease'): (1, -1),
            ('Persistence', 'Decrease', 'Increase'): (1, -1),
        }

    labels_map_pos = np.zeros(shape)
    labels_map_neg = np.zeros(shape)
    with open(json_path) as f:
        json_labels = json.load(f)
        for l in json_labels[1:]:
            rr, cc = ellipse(l['cx'], l['cy'], l['rx'], l['ry'], shape=shape, rotation=np.deg2rad(l['angle']))
            label_type = l['label']
            persistence_size_change = l['size_change'] if label_type == 'Persistence' else None
            persistence_intensity_change = l['intensity_change'] if label_type == 'Persistence' else None

            c_label = label_mapping_dict[(label_type, persistence_size_change, persistence_intensity_change)]
            if c_label == 0:
                continue

            if type(c_label) == int:
                labels_map = labels_map_pos if c_label > 0 else labels_map_neg
                labels_map[rr, cc] = c_label
            elif type(c_label) == tuple:
                labels_map_pos[rr, cc] = c_label[0]
                labels_map_neg[rr, cc] = c_label[1]
            else:
                raise Exception()

    return labels_map_pos, labels_map_neg


def load_model_labels_map(nif_path):
    labels = nib.load(nif_path).get_fdata()
    labels = resize(torch.tensor(labels)[None, None, ...]).squeeze().numpy()

    return (labels > 0).astype(int), (labels < 0).astype(int)


def update_matches_dict(matches_dict, label_to_physician_dict, ccs_avi, ccs_benny, ccs_sigal, ccs_smadar, t=1):
    physicians_names = ['Avi', 'Benny', 'Sigal', 'Smadar']
    ccs_arr = [ccs_avi, ccs_benny, ccs_sigal, ccs_smadar]
    for i in range(len(ccs_arr)):
        # print(f'Current physician name = {physicians_names[i]}')
        other_ccs_arr = ccs_arr[i + 1:]
        for j, other_ccs in enumerate(other_ccs_arr):
            c_other_phy_name = physicians_names[j + i + 1]
            # print(f'Other physician name = {c_other_phy_name}')

            vals_other, counts_other = np.unique(other_ccs, return_counts=True)
            # print(f'Other physician vals_other = {vals_other}')
            # print(f'Other physician counts_other = {counts_other}')
            # exit()
            for v in vals_other:
                if v == 0:
                    continue

                # print(f'Current other physician val = {v}')

                v_ccs = ccs_arr[i] * (other_ccs == v)
                intersecting_vals, intersection_counts = np.unique(v_ccs, return_counts=True)

                # print(f'Intersection vals: {intersecting_vals}')
                # print(f'Intersection counts: {intersection_counts}')

                if 0 in intersecting_vals:
                    list_inter_vals = intersecting_vals.tolist()
                    z_idx = list_inter_vals.index(0)
                    intersecting_vals = np.delete(intersecting_vals, z_idx)
                    intersection_counts = np.delete(intersection_counts, z_idx)

                if len(intersecting_vals) == 0:
                    continue

                # print(f'Intersection vals: {intersecting_vals}')
                # print(f'Intersection counts: {intersection_counts}')

                best_val = intersecting_vals[np.argsort(intersection_counts)[::-1][0]]

                # print(f'Best val = {best_val}')

                another_from_other_in = False
                for v_other in vals_other:
                    if v_other in matches_dict[best_val]:
                        another_from_other_in = True

                if another_from_other_in:
                    continue

                other_ccs[other_ccs == v] = best_val

                matches_dict[best_val].update(matches_dict[v])
                label_to_physician_dict[best_val].update(label_to_physician_dict[v])

                if v != best_val:
                    del matches_dict[v]
                    del label_to_physician_dict[v]

    return label_to_physician_dict


def get_detections(label_map_avi, label_map_benny, label_map_sigal, label_map_smadar, t=1):
    ccs_avi, ccs_num_avi = label(label_map_avi, struct)
    ccs_benny, ccs_num_benny = label(label_map_benny, struct)
    ccs_sigal, ccs_num_sigal = label(label_map_sigal, struct)
    ccs_smadar, ccs_num_smadar = label(label_map_smadar, struct)

    ccs_benny[ccs_benny > 0] += ccs_num_avi
    ccs_sigal[ccs_sigal > 0] += ccs_num_avi + ccs_num_benny
    ccs_smadar[ccs_smadar > 0] += ccs_num_avi + ccs_num_benny + ccs_num_sigal

    matches_dict = {i + 1: {i + 1} for i in range(ccs_num_avi + ccs_num_benny + ccs_num_sigal + ccs_num_smadar)}
    label_to_physician_dict = {}
    label_to_physician_dict.update({i + 1: {'Avi'} for i in range(ccs_num_avi)})
    label_to_physician_dict.update({i + 1: {'Benny'} for i in range(ccs_num_avi, ccs_num_avi + ccs_num_benny)})
    label_to_physician_dict.update({i + 1: {'Sigal'} for i in range(ccs_num_avi + ccs_num_benny, ccs_num_avi + ccs_num_benny + ccs_num_sigal)})
    label_to_physician_dict.update({i + 1: {'Smadar'} for i in range(ccs_num_avi + ccs_num_benny + ccs_num_sigal, ccs_num_avi + ccs_num_benny + ccs_num_sigal + ccs_num_smadar)})

    new_label_to_physician_dict = update_matches_dict(matches_dict, label_to_physician_dict, ccs_avi, ccs_benny, ccs_sigal, ccs_smadar, t=1)
    return new_label_to_physician_dict


def get_pairwise_detections(label_map1, label_map2):
    ccs1, ccs_num1 = label(label_map1 != 0, struct)
    ccs2, ccs_num2 = label(label_map2 != 0, struct)

    agreements = 0
    disagreements = 0

    vals1, counts1 = np.unique(ccs1, return_counts=True)

    for v in vals1:
        if v == 0:
            continue

        inter_in_2 = ccs2 * (ccs1 == v)

        vals2_inter, counts2_inter = np.unique(inter_in_2, return_counts=True)

        if 0 in vals2_inter:
            list_inter_vals = vals2_inter.tolist()
            z_idx = list_inter_vals.index(0)
            vals2_inter = np.delete(vals2_inter, z_idx)
            counts2_inter = np.delete(counts2_inter, z_idx)

        if len(vals2_inter) == 0:
            disagreements += 1
            continue

        inter_val_2 = vals2_inter[np.argsort(counts2_inter)[::-1][0]]
        ccs2[ccs2 == inter_val_2] = 0

        agreements += 1

    num_no_inter_in_2 = len(np.unique(ccs2)) - 1
    disagreements += num_no_inter_in_2

    return agreements, disagreements


def get_HMDR_and_UDPP_counts(model_map, human_maps):
    # UDPP = Unmatched Detections Per Pair
    # HMDR = Human-Matched Detection Rate

    ccs_model, total_preds = label(model_map, struct)

    human_union_map = np.zeros_like(human_maps[0])
    for human_map in human_maps:
        human_union_map[human_map != 0] = 1

    inter_ccs_model = ccs_model * (human_union_map == 1)
    overlapping = len(np.unique(inter_ccs_model)) - 1
    not_overlapping = total_preds - overlapping

    return overlapping, not_overlapping, total_preds


def get_sensitivity_at_consensus_levels(model_map, human_maps):
    num_humans = len(human_maps)
    human_maps = [(human_map != 0).astype(int)[None, ...] for human_map in human_maps]
    model_map = model_map != 0

    sum_map = np.sum(np.concatenate(human_maps, axis=0), axis=0).squeeze()
    human_maps = [label(human_map.squeeze())[0] for human_map in human_maps]
    # GT_Ks = [np.zeros_like(sum_map) for _ in range(num_humans)]
    sensitivities = []
    consensus_map = np.zeros_like(sum_map)

    for j, human_map in enumerate(human_maps):
        c_consensus_map = np.zeros_like(human_map)

        vals = np.unique(human_map)
        for val in vals:
            if val == 0:
                continue

            inter_val = sum_map * (human_map == val)
            val_consensus_level = np.amax(inter_val)

            c_consensus_map[human_map == val] = val_consensus_level

        consensus_map = np.maximum(consensus_map, c_consensus_map)

    for i in range(num_humans):
        consensus_level = i + 1
        consensus_level_i_map = consensus_map >= consensus_level

        consensus_i_ccs, consensus_i_ccs_num = label(consensus_level_i_map)
        model_inter_map = consensus_i_ccs * model_map

        model_detections_num = len(np.unique(model_inter_map)) - 1

        sensitivities.append((model_detections_num, consensus_i_ccs_num))

    # for i in range(num_humans):
    #     consensus_size = i + 1
    #
    #     for c_human_maps in combinations(human_maps, consensus_size):
    #         c_consensus_map = np.ones_like(c_human_maps[0])
    #
    #         for c_human_map in c_human_maps:

    return sensitivities


# def plot_detection_table_graphs(detection_table, category_name):
#     print(f'Category name: {category_name}')
#     detection_table = np.array(detection_table)
#     print(detection_table.shape)
#     det_mean = np.mean(detection_table, axis=1)
#     det_std = np.std(detection_table, axis=1)
#
#     print(np.mean(det_mean))
#     print(np.mean(det_std))
#     print(np.corrcoef(detection_table.T))


def main():
    model_outputs_base_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ICU_cases/test_predictions/all_entities_model'
    annotations_base_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/PhysicianAnnotations'
    pairs_base_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/Pairs_ICU'
    out_path = '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability'

    num_humans = 5
    num_observers = 6
    detections_table_pos = []
    detections_table_neg = []
    detections_table_all = []
    physician_to_idx_dict = {'Avi': 0, 'Benny': 1, 'Sigal': 2, 'Smadar': 3, 'Nitzan': 4, 'Model': 5}

    num_pairs = 30
    bias = 0

    sensitivities_pos = [[0, 0] for _ in range(num_humans)]
    sensitivities_neg = [[0, 0] for _ in range(num_humans)]
    
    sensitivities_pos_avi = [[0, 0] for _ in range(num_humans - 1)]
    sensitivities_neg_avi = [[0, 0] for _ in range(num_humans - 1)]
    
    sensitivities_pos_benny = [[0, 0] for _ in range(num_humans - 1)]
    sensitivities_neg_benny = [[0, 0] for _ in range(num_humans - 1)]
    
    sensitivities_pos_sigal = [[0, 0] for _ in range(num_humans - 1)]
    sensitivities_neg_sigal = [[0, 0] for _ in range(num_humans - 1)]
    
    sensitivities_pos_smadar = [[0, 0] for _ in range(num_humans - 1)]
    sensitivities_neg_smadar = [[0, 0] for _ in range(num_humans - 1)]
    
    sensitivities_pos_nitzan = [[0, 0] for _ in range(num_humans - 1)]
    sensitivities_neg_nitzan = [[0, 0] for _ in range(num_humans - 1)]

    pairwise_agreement_mat_pos = np.eye(num_observers)
    pairwise_agreement_mat_neg = np.eye(num_observers)

    pairwise_disagreement_mat_pos = np.zeros((num_observers, num_observers))
    pairwise_disagreement_mat_neg = np.zeros((num_observers, num_observers))

    pairwise_agreement_per_pair_mat_pos = np.eye(num_observers) * num_pairs
    pairwise_agreement_per_pair_mat_neg = np.eye(num_observers) * num_pairs

    agreements_num_list_pos = [[[] for _ in range(num_observers)] for __ in range(num_observers)]
    agreements_num_list_neg = [[[] for _ in range(num_observers)] for __ in range(num_observers)]
    disagreements_num_list_pos = [[[] for _ in range(num_observers)] for __ in range(num_observers)]
    disagreements_num_list_neg = [[[] for _ in range(num_observers)] for __ in range(num_observers)]

    total_labels_pos = {phy: [] for phy in physician_to_idx_dict.keys()}
    total_labels_neg = {phy: [] for phy in physician_to_idx_dict.keys()}

    total_model_preds_pos = 0
    total_model_preds_neg = 0
    total_overlapping_model_preds_pos = 0
    total_overlapping_model_preds_neg = 0
    not_overlapping_model_preds_pos = []
    not_overlapping_model_preds_neg = []
    
    total_avi_preds_pos = 0
    total_avi_preds_neg = 0
    total_overlapping_avi_preds_pos = 0
    total_overlapping_avi_preds_neg = 0
    not_overlapping_avi_preds_pos = []
    not_overlapping_avi_preds_neg = []
    
    total_benny_preds_pos = 0
    total_benny_preds_neg = 0
    total_overlapping_benny_preds_pos = 0
    total_overlapping_benny_preds_neg = 0
    not_overlapping_benny_preds_pos = []
    not_overlapping_benny_preds_neg = []
    
    total_sigal_preds_pos = 0
    total_sigal_preds_neg = 0
    total_overlapping_sigal_preds_pos = 0
    total_overlapping_sigal_preds_neg = 0
    not_overlapping_sigal_preds_pos = []
    not_overlapping_sigal_preds_neg = []
    
    total_smadar_preds_pos = 0
    total_smadar_preds_neg = 0
    total_overlapping_smadar_preds_pos = 0
    total_overlapping_smadar_preds_neg = 0
    not_overlapping_smadar_preds_pos = []
    not_overlapping_smadar_preds_neg = []
    
    total_nitzan_preds_pos = 0
    total_nitzan_preds_neg = 0
    total_overlapping_nitzan_preds_pos = 0
    total_overlapping_nitzan_preds_neg = 0
    not_overlapping_nitzan_preds_pos = []
    not_overlapping_nitzan_preds_neg = []

    for i in range(bias, num_pairs + bias):
        print(f'Pair {i + 1}')
        pair_path = f'{pairs_base_path}/pair{i+1}'
        current_path = f'{pair_path}/{sorted([p for p in os.listdir(pair_path) if p.endswith(".nii.gz")])[1]}'
        annotation_path_avi = f'{annotations_base_path}/Avi/{i+1}.json'
        annotation_path_benny = f'{annotations_base_path}/Benny/{i+1}.json'
        annotation_path_sigal = f'{annotations_base_path}/Sigal/{i+1}.json'
        annotation_path_smadar = f'{annotations_base_path}/Smadar/{i+1}.json'
        annotation_path_nitzan = f'{annotations_base_path}/Nitzan/{i+1}.json'
        annotation_path_model = f'{model_outputs_base_path}/pair{i+1}/output.nii.gz'

        current = load_xray(current_path)
        label_map_pos_avi, label_map_neg_avi = load_labels_map(annotation_path_avi, current.shape)
        label_map_pos_benny, label_map_neg_benny = load_labels_map(annotation_path_benny, current.shape)
        label_map_pos_sigal, label_map_neg_sigal = load_labels_map(annotation_path_sigal, current.shape)
        label_map_pos_smadar, label_map_neg_smadar = load_labels_map(annotation_path_smadar, current.shape)
        label_map_pos_nitzan, label_map_neg_nitzan = load_labels_map(annotation_path_nitzan, current.shape)
        label_map_pos_model, label_map_neg_model = load_model_labels_map(annotation_path_model)

        # if i == 3:
        #     label_map_pos_avi = (label_map_pos_avi != 0).astype(int)
        #     label_map_neg_avi = -((label_map_neg_avi != 0).astype(int))
        #     plot_diff_on_current(label_map_pos_avi.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Avi4Pos.png')
        #     plot_diff_on_current(label_map_neg_avi.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Avi4Neg.png')
        #
        #     label_map_pos_benny = (label_map_pos_benny != 0).astype(int)
        #     label_map_neg_benny = -((label_map_neg_benny != 0).astype(int))
        #     plot_diff_on_current(label_map_pos_benny.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Benny4Pos.png')
        #     plot_diff_on_current(label_map_neg_benny.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Benny4Neg.png')
        #
        #     label_map_pos_sigal = (label_map_pos_sigal != 0).astype(int)
        #     label_map_neg_sigal = -((label_map_neg_sigal != 0).astype(int))
        #     plot_diff_on_current(label_map_pos_sigal.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Sigal4Pos.png')
        #     plot_diff_on_current(label_map_neg_sigal.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Sigal4Neg.png')
        #
        #     label_map_pos_smadar = (label_map_pos_smadar != 0).astype(int)
        #     label_map_neg_smadar = -((label_map_neg_smadar != 0).astype(int))
        #     plot_diff_on_current(label_map_pos_smadar.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Smadar4Pos.png')
        #     plot_diff_on_current(label_map_neg_smadar.T, current.T, '/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/Smadar4Neg.png')
        #     exit()

        # detections_pos = {'Avi': [], 'Benny': [], 'Sigal': [], 'Smadar': []}
        # detections_neg = {'Avi': [], 'Benny': [], 'Sigal': [], 'Smadar': []}

        # TODO: Sensitivity for consensus levels measures
        
        # Model
        
        c_sensitivities_pos = get_sensitivity_at_consensus_levels(label_map_pos_model, (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg = get_sensitivity_at_consensus_levels(label_map_neg_model, (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))

        for k, (detections, changes) in enumerate(c_sensitivities_pos):
            sensitivities_pos[k][0] += detections
            sensitivities_pos[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg):
            sensitivities_neg[k][0] += detections
            sensitivities_neg[k][1] += changes
        
        # Humans

        c_sensitivities_pos_avi = get_sensitivity_at_consensus_levels(label_map_pos_avi, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_avi = get_sensitivity_at_consensus_levels(label_map_neg_avi, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        
        for k, (detections, changes) in enumerate(c_sensitivities_pos_avi):
            sensitivities_pos_avi[k][0] += detections
            sensitivities_pos_avi[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg_avi):
            sensitivities_neg_avi[k][0] += detections
            sensitivities_neg_avi[k][1] += changes
        
        c_sensitivities_pos_benny = get_sensitivity_at_consensus_levels(label_map_pos_benny, (label_map_pos_avi, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_benny = get_sensitivity_at_consensus_levels(label_map_neg_benny, (label_map_neg_avi, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        
        for k, (detections, changes) in enumerate(c_sensitivities_pos_benny):
            sensitivities_pos_benny[k][0] += detections
            sensitivities_pos_benny[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg_benny):
            sensitivities_neg_benny[k][0] += detections
            sensitivities_neg_benny[k][1] += changes
        
        c_sensitivities_pos_sigal = get_sensitivity_at_consensus_levels(label_map_pos_sigal, (label_map_pos_benny, label_map_pos_avi, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_sigal = get_sensitivity_at_consensus_levels(label_map_neg_sigal, (label_map_neg_benny, label_map_neg_avi, label_map_neg_smadar, label_map_neg_nitzan))
        
        for k, (detections, changes) in enumerate(c_sensitivities_pos_sigal):
            sensitivities_pos_sigal[k][0] += detections
            sensitivities_pos_sigal[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg_sigal):
            sensitivities_neg_sigal[k][0] += detections
            sensitivities_neg_sigal[k][1] += changes
        
        c_sensitivities_pos_smadar = get_sensitivity_at_consensus_levels(label_map_pos_smadar, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_avi, label_map_pos_nitzan))
        c_sensitivities_neg_smadar = get_sensitivity_at_consensus_levels(label_map_neg_smadar, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_avi, label_map_neg_nitzan))
        
        for k, (detections, changes) in enumerate(c_sensitivities_pos_smadar):
            sensitivities_pos_smadar[k][0] += detections
            sensitivities_pos_smadar[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg_smadar):
            sensitivities_neg_smadar[k][0] += detections
            sensitivities_neg_smadar[k][1] += changes
        
        c_sensitivities_pos_nitzan = get_sensitivity_at_consensus_levels(label_map_pos_nitzan, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_avi))
        c_sensitivities_neg_nitzan = get_sensitivity_at_consensus_levels(label_map_neg_nitzan, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_avi))
        
        for k, (detections, changes) in enumerate(c_sensitivities_pos_nitzan):
            sensitivities_pos_nitzan[k][0] += detections
            sensitivities_pos_nitzan[k][1] += changes

        for k, (detections, changes) in enumerate(c_sensitivities_neg_nitzan):
            sensitivities_neg_nitzan[k][0] += detections
            sensitivities_neg_nitzan[k][1] += changes
        
        # TODO: HMDR/UDPP measures
        
        # Model
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_model, (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_model, (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        
        total_model_preds_pos += total_preds_pos
        total_model_preds_neg += total_preds_neg
        total_overlapping_model_preds_pos += overlapping_pos
        total_overlapping_model_preds_neg += overlapping_neg
        not_overlapping_model_preds_pos.append(not_overlapping_pos)
        not_overlapping_model_preds_neg.append(not_overlapping_neg)
        
        # Avi
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_avi, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_avi, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))

        total_avi_preds_pos += total_preds_pos
        total_avi_preds_neg += total_preds_neg
        total_overlapping_avi_preds_pos += overlapping_pos
        total_overlapping_avi_preds_neg += overlapping_neg
        not_overlapping_avi_preds_pos.append(not_overlapping_pos)
        not_overlapping_avi_preds_neg.append(not_overlapping_neg)
        
        # Benny
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_benny, (label_map_pos_avi, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_benny, (label_map_neg_avi, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))

        total_benny_preds_pos += total_preds_pos
        total_benny_preds_neg += total_preds_neg
        total_overlapping_benny_preds_pos += overlapping_pos
        total_overlapping_benny_preds_neg += overlapping_neg
        not_overlapping_benny_preds_pos.append(not_overlapping_pos)
        not_overlapping_benny_preds_neg.append(not_overlapping_neg)
        
        # Sigal
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_sigal, (label_map_pos_benny, label_map_pos_avi, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_sigal, (label_map_neg_benny, label_map_neg_avi, label_map_neg_smadar, label_map_neg_nitzan))

        total_sigal_preds_pos += total_preds_pos
        total_sigal_preds_neg += total_preds_neg
        total_overlapping_sigal_preds_pos += overlapping_pos
        total_overlapping_sigal_preds_neg += overlapping_neg
        not_overlapping_sigal_preds_pos.append(not_overlapping_pos)
        not_overlapping_sigal_preds_neg.append(not_overlapping_neg)
        
        # Smadar
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_smadar, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_avi, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_smadar, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_avi, label_map_neg_nitzan))

        total_smadar_preds_pos += total_preds_pos
        total_smadar_preds_neg += total_preds_neg
        total_overlapping_smadar_preds_pos += overlapping_pos
        total_overlapping_smadar_preds_neg += overlapping_neg
        not_overlapping_smadar_preds_pos.append(not_overlapping_pos)
        not_overlapping_smadar_preds_neg.append(not_overlapping_neg)
        
        # Nitzan
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_nitzan, (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_nitzan, (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar))

        total_nitzan_preds_pos += total_preds_pos
        total_nitzan_preds_neg += total_preds_neg
        total_overlapping_nitzan_preds_pos += overlapping_pos
        total_overlapping_nitzan_preds_neg += overlapping_neg
        not_overlapping_nitzan_preds_pos.append(not_overlapping_pos)
        not_overlapping_nitzan_preds_neg.append(not_overlapping_neg)

        # TODO: PAI
        inf_list_pos = ((label_map_pos_avi, 'Avi'), (label_map_pos_benny, 'Benny'), (label_map_pos_sigal, 'Sigal'), (label_map_pos_smadar, 'Smadar'), (label_map_pos_nitzan, 'Nitzan'), (label_map_pos_model, 'Model'))
        inf_list_neg = ((label_map_neg_avi, 'Avi'), (label_map_neg_benny, 'Benny'), (label_map_neg_sigal, 'Sigal'), (label_map_neg_smadar, 'Smadar'), (label_map_neg_nitzan, 'Nitzan'), (label_map_neg_model, 'Model'))

        for inf_list, agreement_mat, disagreement_mat, agreement_per_pair_mat, total_labels_lst, lists_for_per_pair_all in zip([inf_list_pos, inf_list_neg], [pairwise_agreement_mat_pos, pairwise_agreement_mat_neg], [pairwise_disagreement_mat_pos, pairwise_disagreement_mat_neg], [pairwise_agreement_per_pair_mat_pos, pairwise_agreement_per_pair_mat_neg], [total_labels_pos, total_labels_neg], [(agreements_num_list_pos, disagreements_num_list_pos), (agreements_num_list_neg, disagreements_num_list_neg)]):
            for label_map1_inf, label_map2_inf in combinations(inf_list, 2):
                name1 = label_map1_inf[1]
                name2 = label_map2_inf[1]

                label_map1 = label_map1_inf[0]
                label_map2 = label_map2_inf[0]

                agreements, disagreements = get_pairwise_detections(label_map1, label_map2)

                agreement_mat[physician_to_idx_dict[name1], physician_to_idx_dict[name2]] += 2 * agreements
                agreement_mat[physician_to_idx_dict[name2], physician_to_idx_dict[name1]] += 2 * agreements

                disagreement_mat[physician_to_idx_dict[name1], physician_to_idx_dict[name2]] += disagreements
                disagreement_mat[physician_to_idx_dict[name2], physician_to_idx_dict[name1]] += disagreements

                agreement_per_pair_mat[physician_to_idx_dict[name1], physician_to_idx_dict[name2]] += np.nan_to_num(2 * agreements / np.array(2 * agreements + disagreements), nan=1., posinf=1., neginf=1.)
                agreement_per_pair_mat[physician_to_idx_dict[name2], physician_to_idx_dict[name1]] += np.nan_to_num(2 * agreements / np.array(2 * agreements + disagreements), nan=1., posinf=1., neginf=1.)

                ag_list = lists_for_per_pair_all[0]
                ag_list[physician_to_idx_dict[name1]][physician_to_idx_dict[name2]].append(2 * agreements)

                disag_list = lists_for_per_pair_all[1]
                disag_list[physician_to_idx_dict[name1]][physician_to_idx_dict[name2]].append(disagreements)

            for label_map in inf_list:
                name = label_map[1]
                label_map = label_map[0]

                labels, ccs_num = label(label_map, struct)
                # ccs_num = len(np.unique(labels)) - 1

                total_labels_lst[name].append(ccs_num)

        continue

        # TODO: Combined analysis (not relevant right now)
        label_map_all_avi = label_map_pos_avi.copy()
        label_map_all_benny = label_map_pos_benny.copy()
        label_map_all_sigal = label_map_pos_sigal.copy()
        label_map_all_smadar = label_map_pos_smadar.copy()

        label_map_all_avi[label_map_neg_avi != 0] = label_map_neg_avi[label_map_neg_avi != 0]
        label_map_all_benny[label_map_neg_benny != 0] = label_map_neg_benny[label_map_neg_benny != 0]
        label_map_all_sigal[label_map_neg_sigal != 0] = label_map_neg_sigal[label_map_neg_sigal != 0]
        label_map_all_smadar[label_map_neg_smadar != 0] = label_map_neg_smadar[label_map_neg_smadar != 0]

        labels_map_categories = [
            (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar),
            (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar),
            (label_map_all_avi, label_map_all_benny, label_map_all_sigal, label_map_all_smadar)
        ]

        detections_table_pos, detections_table_neg, detections_table_all = [], [], []

        for (label_map_avi, label_map_benny, label_map_sigal, label_map_smadar), detection_table in zip(labels_map_categories, [detections_table_pos, detections_table_neg, detections_table_all]):
            new_label_to_physician_dict = get_detections(label_map_avi, label_map_benny, label_map_sigal, label_map_smadar)

            for physicians in new_label_to_physician_dict.values():
                c_detections = [0 for _ in range(num_observers)]
                for p in physicians:
                    c_detections[physician_to_idx_dict[p]] = 1
                detection_table.append(c_detections)

    # TODO: Sensitivity at consensus levels

    final_sensitivities = {f'Sensitivity Consensus Level {i + 1} (Positive)': s[0] / s[1] for i, s in enumerate(sensitivities_pos)}
    final_sensitivities.update({f'Sensitivity Consensus Level {i + 1} (Negative)': s[0] / s[1] for i, s in enumerate(sensitivities_neg)})

    final_sensitivities.update({f'Total detections & changes at Consensus Level {i + 1} (Positive)': (s[0], s[1]) for i, s in enumerate(sensitivities_pos)})
    final_sensitivities.update({f'Total detections & changes at Consensus Level {i + 1} (Negative)': (s[0], s[1]) for i, s in enumerate(sensitivities_neg)})

    plot_curves([s[0] / s[1] for s in sensitivities_pos], [s[0] / s[1] for s in sensitivities_neg], '$M_{ICU}$', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels.png')

    plot_curves([s[0] / s[1] for s in sensitivities_pos_avi], [s[0] / s[1] for s in sensitivities_neg_avi], 'A', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_avi.png')
    plot_curves([s[0] / s[1] for s in sensitivities_pos_benny], [s[0] / s[1] for s in sensitivities_neg_benny], 'B', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_benny.png')
    plot_curves([s[0] / s[1] for s in sensitivities_pos_sigal], [s[0] / s[1] for s in sensitivities_neg_sigal], 'C', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_sigal.png')
    plot_curves([s[0] / s[1] for s in sensitivities_pos_smadar], [s[0] / s[1] for s in sensitivities_neg_smadar], 'D', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_smadar.png')
    plot_curves([s[0] / s[1] for s in sensitivities_pos_nitzan], [s[0] / s[1] for s in sensitivities_neg_nitzan], 'E', f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_nitzan.png')


    # plot_curves_multiple(
    #     {'Avi': [s[0] / s[1] for s in sensitivities_pos_avi], 'Benny': [s[0] / s[1] for s in sensitivities_pos_benny], 'Sigal': [s[0] / s[1] for s in sensitivities_pos_sigal], 'Smadar': [s[0] / s[1] for s in sensitivities_pos_smadar], 'Nitzan': [s[0] / s[1] for s in sensitivities_pos_nitzan]},
    #     {'Avi': [s[0] / s[1] for s in sensitivities_neg_avi], 'Benny': [s[0] / s[1] for s in sensitivities_neg_benny], 'Sigal': [s[0] / s[1] for s in sensitivities_neg_sigal], 'Smadar': [s[0] / s[1] for s in sensitivities_neg_smadar], 'Nitzan': [s[0] / s[1] for s in sensitivities_neg_nitzan]},
    #     num_humans - 1,
    #     f'/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/ObserverVariability/sensitivity_consensus_levels_all_humans.png'
    # )
    
    with open(f'{out_path}/sensitivity_measures.json', 'w') as f:
        json.dump(final_sensitivities, f, indent=4)

    # TODO: HMDR/UDPP

    # Model
    hmdr_model_pos = total_overlapping_model_preds_pos / total_model_preds_pos
    hmdr_model_neg = total_overlapping_model_preds_neg / total_model_preds_neg
    udpp_model_pos = sum(not_overlapping_model_preds_pos) / num_pairs
    udpp_model_neg = sum(not_overlapping_model_preds_neg) / num_pairs
    udpp_model_pos_std = np.std(not_overlapping_model_preds_pos)
    udpp_model_neg_std = np.std(not_overlapping_model_preds_neg)

    # Avi
    hmdr_avi_pos = total_overlapping_avi_preds_pos / total_avi_preds_pos
    hmdr_avi_neg = total_overlapping_avi_preds_neg / total_avi_preds_neg
    udpp_avi_pos = sum(not_overlapping_avi_preds_pos) / num_pairs
    udpp_avi_neg = sum(not_overlapping_avi_preds_neg) / num_pairs
    udpp_avi_pos_std = np.std(not_overlapping_avi_preds_pos)
    udpp_avi_neg_std = np.std(not_overlapping_avi_preds_neg)
    
    # Benny
    hmdr_benny_pos = total_overlapping_benny_preds_pos / total_benny_preds_pos
    hmdr_benny_neg = total_overlapping_benny_preds_neg / total_benny_preds_neg
    udpp_benny_pos = sum(not_overlapping_benny_preds_pos) / num_pairs
    udpp_benny_neg = sum(not_overlapping_benny_preds_neg) / num_pairs
    udpp_benny_pos_std = np.std(not_overlapping_benny_preds_pos)
    udpp_benny_neg_std = np.std(not_overlapping_benny_preds_neg)
    
    # Sigal
    hmdr_sigal_pos = total_overlapping_sigal_preds_pos / total_sigal_preds_pos
    hmdr_sigal_neg = total_overlapping_sigal_preds_neg / total_sigal_preds_neg
    udpp_sigal_pos = sum(not_overlapping_sigal_preds_pos) / num_pairs
    udpp_sigal_neg = sum(not_overlapping_sigal_preds_neg) / num_pairs
    udpp_sigal_pos_std = np.std(not_overlapping_sigal_preds_pos)
    udpp_sigal_neg_std = np.std(not_overlapping_sigal_preds_neg)
    
    # Smadar
    hmdr_smadar_pos = total_overlapping_smadar_preds_pos / total_smadar_preds_pos
    hmdr_smadar_neg = total_overlapping_smadar_preds_neg / total_smadar_preds_neg
    udpp_smadar_pos = sum(not_overlapping_smadar_preds_pos) / num_pairs
    udpp_smadar_neg = sum(not_overlapping_smadar_preds_neg) / num_pairs
    udpp_smadar_pos_std = np.std(not_overlapping_smadar_preds_pos)
    udpp_smadar_neg_std = np.std(not_overlapping_smadar_preds_neg)
    
    # Nitzan
    hmdr_nitzan_pos = total_overlapping_nitzan_preds_pos / total_nitzan_preds_pos
    hmdr_nitzan_neg = total_overlapping_nitzan_preds_neg / total_nitzan_preds_neg
    udpp_nitzan_pos = sum(not_overlapping_nitzan_preds_pos) / num_pairs
    udpp_nitzan_neg = sum(not_overlapping_nitzan_preds_neg) / num_pairs
    udpp_nitzan_pos_std = np.std(not_overlapping_nitzan_preds_pos)
    udpp_nitzan_neg_std = np.std(not_overlapping_nitzan_preds_neg)

    # print(f'Precision pos = {union_precision_model_pos}')
    # print(f'Precision neg = {union_precision_model_neg}')
    # print(f'FPPI pos = {fppi_model_pos}')
    # print(f'FPPI neg = {fppi_model_neg}')

    precision_measures_dict = {'Model HMDR (Positive)': hmdr_model_pos, 'Model HMDR (Negative)': hmdr_model_neg,
                               'Avi HMDR (Positive)': hmdr_avi_pos, 'Avi HMDR (Negative)': hmdr_avi_neg,
                               'Benny HMDR (Positive)': hmdr_benny_pos, 'Benny HMDR (Negative)': hmdr_benny_neg,
                               'Sigal HMDR (Positive)': hmdr_sigal_pos, 'Sigal HMDR (Negative)': hmdr_sigal_neg,
                               'Smadar HMDR (Positive)': hmdr_smadar_pos, 'Smadar HMDR (Negative)': hmdr_smadar_neg,
                               'Nitzan HMDR (Positive)': hmdr_nitzan_pos, 'Nitzan HMDR (Negative)': hmdr_nitzan_neg,
                               'UDPP Model (Positive)': udpp_model_pos, 'UDPP Model (Negative)': udpp_model_neg,
                               'UDPP Avi (Positive)': udpp_avi_pos, 'UDPP Avi (Negative)': udpp_avi_neg,
                               'UDPP Benny (Positive)': udpp_benny_pos, 'UDPP Benny (Negative)': udpp_benny_neg,
                               'UDPP Sigal (Positive)': udpp_sigal_pos, 'UDPP Sigal (Negative)': udpp_sigal_neg,
                               'UDPP Smadar (Positive)': udpp_smadar_pos, 'UDPP Smadar (Negative)': udpp_smadar_neg,
                               'UDPP Nitzan (Positive)': udpp_nitzan_pos, 'UDPP Nitzan (Negative)': udpp_nitzan_neg,
                               'UDPP STD Model (Positive)': udpp_model_pos_std, 'UDPP STD Model (Negative)': udpp_model_neg_std,
                               'UDPP STD Avi (Positive)': udpp_avi_pos_std, 'UDPP STD Avi (Negative)': udpp_avi_neg_std,
                               'UDPP STD Benny (Positive)': udpp_benny_pos_std, 'UDPP STD Benny (Negative)': udpp_benny_neg_std,
                               'UDPP STD Sigal (Positive)': udpp_sigal_pos_std, 'UDPP STD Sigal (Negative)': udpp_sigal_neg_std,
                               'UDPP STD Smadar (Positive)': udpp_smadar_pos_std, 'UDPP STD Smadar (Negative)': udpp_smadar_neg_std,
                               'UDPP STD Nitzan (Positive)': udpp_nitzan_pos_std, 'UDPP STD Nitzan (Negative)': udpp_nitzan_neg_std,
                               }
    with open(f'{out_path}/precision_measures.json', 'w') as f:
        json.dump(precision_measures_dict, f, indent=4)

    # TODO: PAI

    # mat_per_annot = (pairwise_agreement_mat_pos + pairwise_agreement_mat_neg) / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos + pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    # print(mat_per_annot)
    # print(np.mean(mat[mat_per_annot < 1.]))

    mat_per_label_pos = pairwise_agreement_mat_pos / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos)
    mat_per_label_neg = pairwise_agreement_mat_neg / (pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_label_all = (pairwise_agreement_mat_pos + pairwise_agreement_mat_neg) / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos + pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_pair_pos = pairwise_agreement_per_pair_mat_pos / num_pairs
    mat_per_pair_neg = pairwise_agreement_per_pair_mat_neg / num_pairs

    mat_per_pair_all = np.eye(num_observers)
    for n1, n2 in combinations(physician_to_idx_dict.keys(), 2):
        pai_per_pair = 0
        for k in range(num_pairs):
            c_ag_pos = agreements_num_list_pos[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_ag_neg = agreements_num_list_neg[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_ag = c_ag_pos + c_ag_neg

            c_disag_pos = disagreements_num_list_pos[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_disag_neg = disagreements_num_list_neg[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_disag = c_disag_pos + c_disag_neg

            c_pai = 0
            c_pai += np.nan_to_num(c_ag / np.array(c_ag + c_disag), nan=1., posinf=1., neginf=1.)
            if (c_ag_pos + c_disag_pos == 0 and c_ag_neg + c_disag_neg > 0) or (c_ag_pos + c_disag_pos > 0 and c_ag_neg + c_disag_neg == 0):
                c_pai = c_pai * 0.5 + 0.5
            pai_per_pair += c_pai
        mat_per_pair_all[physician_to_idx_dict[n1], physician_to_idx_dict[n2]] = pai_per_pair / num_pairs
        mat_per_pair_all[physician_to_idx_dict[n2], physician_to_idx_dict[n1]] = pai_per_pair / num_pairs

    per_label_df_pos = {phy_n: mat_per_label_pos[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
    per_label_df_neg = {phy_n: mat_per_label_neg[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
    per_label_df_all = {phy_n: mat_per_label_all[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
    per_pair_df_pos = {phy_n: mat_per_pair_pos[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
    per_pair_df_neg = {phy_n: mat_per_pair_neg[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
    per_pair_df_all = {phy_n: mat_per_pair_all[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}

    index_names = list(physician_to_idx_dict.keys())
    per_label_df_pos = pd.DataFrame(per_label_df_pos, index=index_names)
    per_label_df_neg = pd.DataFrame(per_label_df_neg, index=index_names)
    per_label_df_all = pd.DataFrame(per_label_df_all, index=index_names)
    per_pair_df_pos = pd.DataFrame(per_pair_df_pos, index=index_names)
    per_pair_df_neg = pd.DataFrame(per_pair_df_neg, index=index_names)
    per_pair_df_all = pd.DataFrame(per_pair_df_all, index=index_names)

    c_out_path_label_pos = f'{out_path}/per_label_agreement_pos.png'
    c_out_path_label_neg = f'{out_path}/per_label_agreement_neg.png'
    c_out_path_label_all = f'{out_path}/per_label_agreement_all.png'
    c_out_path_pair_pos = f'{out_path}/per_pair_agreement_pos.png'
    c_out_path_pair_neg = f'{out_path}/per_pair_agreement_neg.png'
    c_out_path_pair_all = f'{out_path}/per_pair_agreement_all.png'

    plot_matrix(per_label_df_pos, c_out_path_label_pos, title='Pairwise Agreement Index Per Detection (positive)')
    plot_matrix(per_label_df_neg, c_out_path_label_neg, title='Pairwise Agreement Index Per Detection (negative)')
    plot_matrix(per_label_df_all, c_out_path_label_all, title='Pairwise Agreement Index Per Detection (all)')
    plot_matrix(per_pair_df_pos, c_out_path_pair_pos, title='Pairwise Agreement Index Per Pair (positive)')
    plot_matrix(per_pair_df_neg, c_out_path_pair_neg, title='Pairwise Agreement Index Per Pair (negative)')
    plot_matrix(per_pair_df_all, c_out_path_pair_all, title='Pairwise Agreement Index Per Pair (all)')

    print(f'Per label mat pos:\n{mat_per_label_pos}')
    print(f'Per label mat neg:\n{mat_per_label_neg}')
    print(f'Per label mat all:\n{mat_per_label_all}')
    print(f'Per pair mat pos:\n{mat_per_pair_pos}')
    print(f'Per pair mat neg:\n{mat_per_pair_neg}')
    print(f'Per pair mat all:\n{mat_per_pair_all}')
    print(f'Total labels num pos:{total_labels_pos}')
    print(f'Total labels num neg:{total_labels_neg}')

    total_labels_data_pos = {}
    total_labels_data_neg = {}
    total_labels_data_all = {}
    for phy, l in total_labels_pos.items():
        arr = np.array(l)
        t_labels = np.sum(arr)
        mean_labels = np.mean(arr)
        std_labels = np.std(arr)
        max_labels = np.max(arr)
        min_labels = np.min(arr)
        total_labels_data_pos[phy] = (int(t_labels), mean_labels, std_labels, int(max_labels), int(min_labels))
    for phy, l in total_labels_neg.items():
        arr = np.array(l)
        t_labels = np.sum(arr)
        mean_labels = np.mean(arr)
        std_labels = np.std(arr)
        max_labels = np.max(arr)
        min_labels = np.min(arr)
        total_labels_data_neg[phy] = (int(t_labels), mean_labels, std_labels, int(max_labels), int(min_labels))
    for phy in physician_to_idx_dict.keys():
        pos_labels = total_labels_pos[phy]
        neg_labels = total_labels_neg[phy]
        all_labels = np.array([pos_labels[i] + neg_labels[i] for i in range(len(pos_labels))])
        t_labels = np.sum(all_labels)
        mean_labels = np.mean(all_labels)
        std_labels = np.std(all_labels)
        max_labels = np.max(all_labels)
        min_labels = np.min(all_labels)
        total_labels_data_all[phy] = (int(t_labels), mean_labels, std_labels, int(max_labels), int(min_labels))

    print(total_labels_data_pos)
    print(total_labels_data_neg)
    print(total_labels_data_all)

    with open(f'{out_path}/total_labels_marked_pos.json', 'w') as f:
        json.dump(total_labels_data_pos, f, indent=4)

    with open(f'{out_path}/total_labels_marked_neg.json', 'w') as f:
        json.dump(total_labels_data_neg, f, indent=4)

    with open(f'{out_path}/total_labels_marked_all.json', 'w') as f:
        json.dump(total_labels_data_all, f, indent=4)

    exit()

    # Combined outputs
    for detection_table, category_name in zip([detections_table_pos, detections_table_neg, detections_table_all], ['Positive', 'Negative', 'All']):
        detection_table = np.array(detection_table)
        det_mean = np.mean(detection_table, axis=1)
        det_std = np.std(detection_table, axis=1)

        print(category_name)
        print(np.mean(det_mean))
        print(np.mean(det_std))
        corr_matrix = np.corrcoef(detection_table.T)
        print(corr_matrix)
        corr_df = {phy_n: corr_matrix[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}
        index_names = list(physician_to_idx_dict.keys())
        corr_df = pd.DataFrame(corr_df, index=index_names)
        c_out_path = f'{out_path}/{category_name}_corr_mat_.png'

        plot_correlation_matrix(corr_df, category_name, c_out_path)


if __name__ == '__main__':
    resize = v2.Resize((768, 768))
    struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:3CFF3D-40:00FFB3-50:FFFCFC-60:FF9A00-100:AE0000
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))

    AVI_POS = 0
    BENNY_POS = 1
    SIGAL_POS = 2
    SMADAR_POS = 3

    main()

