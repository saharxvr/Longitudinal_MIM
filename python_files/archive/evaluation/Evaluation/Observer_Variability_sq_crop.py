"""
Observer Variability analysis for square-crop predictions.

The model output is a 512x512 image produced by square-cropping the *current*
scan (centered on the seg bounding-box) and then resizing to 512x512.
A crop_info.json saved alongside the output records the crop coordinates.
This script uses crop_info.json to inverse-map the model output back into the
original full-size coordinate space before comparing with doctor annotations.

Also generates per-pair overlay plots of model output + annotations on the
original scan for visual verification.
"""

import numpy as np
import json
import nibabel as nib
from skimage.draw import ellipse
from skimage.transform import resize as sk_resize
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import colors
from scipy.ndimage import label
from itertools import combinations
import argparse
import re

STRUCT_3x3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


# ---------------------------------------------------------------------------
# Helpers carried over from the seg_corrected version
# ---------------------------------------------------------------------------

def resolve_annotation_path(ann_base, physician, pair_num):
    physician_dir = os.path.join(ann_base, physician)
    candidates = []
    for root, _, files in os.walk(physician_dir):
        for fname in files:
            if not fname.lower().endswith('.json'):
                continue
            nums = re.findall(r'\d+', fname)
            if nums and int(nums[0]) == pair_num:
                candidates.append(os.path.join(root, fname))
    if not candidates:
        raise FileNotFoundError(f"No annotation for {physician} pair {pair_num}")
    if len(candidates) == 1:
        return candidates[0]
    # Prefer shallowest path (fewest directory levels = directly in pairs folder)
    return min(candidates, key=lambda p: (p.count(os.sep), len(p)))


def resolve_pair_path(pairs_roots, pair_num):
    for root in pairs_roots:
        for prefix in ('pair', 'Pair'):
            p = os.path.join(root, f'{prefix}{pair_num}')
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"Pair folder not found for pair {pair_num}")


def resolve_model_output(preds_base, pair_num):
    for prefix in ('pair', 'Pair'):
        p = os.path.join(preds_base, f'{prefix}{pair_num}', 'output.nii.gz')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Model output not found for pair {pair_num}")


def resolve_crop_info(preds_base, pair_num):
    for prefix in ('pair', 'Pair'):
        p = os.path.join(preds_base, f'{prefix}{pair_num}', 'crop_info.json')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"crop_info.json not found for pair {pair_num}")


def load_xray(file_path: str):
    xray_nif = nib.load(file_path)
    global affine
    affine = xray_nif.affine
    xray_data = xray_nif.get_fdata()
    return xray_data


def generate_alpha_map(x):
    x_abs = np.abs(x)
    max_val = max(np.max(x_abs).item(), 0.07)
    alphas_map = x_abs / max_val
    return alphas_map


def plot_diff_on_current(c_diff_map, c_current, out_p):
    alphas = generate_alpha_map(c_diff_map)
    divnorm = colors.TwoSlopeNorm(
        vmin=min(np.min(c_diff_map).item(), -0.01), vcenter=0.,
        vmax=max(np.max(c_diff_map).item(), 0.01))
    fig, ax = plt.subplots()
    ax.imshow(c_current.squeeze(), cmap='gray')
    imm1 = ax.imshow(c_diff_map.squeeze(), alpha=alphas, cmap=differential_grad, norm=divnorm)
    plt.colorbar(imm1, fraction=0.05, pad=0.04, ax=ax)
    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches='tight')
    plt.cla(); plt.clf(); plt.close()


# ---------------------------------------------------------------------------
# Plotting helpers (unchanged from original)
# ---------------------------------------------------------------------------

def plot_matrix(matrix_df: pd.DataFrame, output_path: str, title='Correlation Matrix'):
    N = matrix_df.shape[0]
    figsize = max(8, int(N * 0.6))
    plt.figure(figsize=(figsize, figsize))
    ax = sns.heatmap(
        matrix_df, annot=True, fmt=".2f", cmap="vlag",
        vmin=0, vmax=1, center=0.25,
        linewidths=0., linecolor='black',
        cbar_kws={'shrink': 1., 'aspect': 20, 'label': 'PAI'},
        annot_kws={"fontsize": 14, 'fontweight': 'bold'})

    separator_index = N - 1
    ax.axvline(x=separator_index, ymin=1 / N, color='black', linewidth=1.25, linestyle='--')
    ax.axhline(y=separator_index, xmax=1 - 1 / N, color='black', linewidth=1.25, linestyle='--')

    group1_x = (N - 1) / 2
    group2_x = (N - 1) + 0.5
    text_y_pos = -0.04
    ax.text(group1_x, text_y_pos, 'H - H', ha='center', va='bottom',
            fontsize=14, fontweight='bold', fontstyle='italic', color='#333333')
    ax.text(group2_x, text_y_pos, 'M - H', ha='center', va='bottom',
            fontsize=14, fontweight='bold', fontstyle='italic', color='#333333')

    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.yticks(rotation=0, fontsize=14, fontweight='bold')
    plt.title(f'{title}', fontsize=16, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrix plot saved successfully to: **{output_path}**")


def plot_curves(pos_arr, neg_arr, n, out_p):
    x_values = np.arange(1, len(pos_arr) + 1)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 11, 'figure.figsize': (8, 5)})

    fig, ax = plt.subplots()
    ax.plot(x_values, pos_arr, label='Positive changes', color='#003366',
            linestyle='-', linewidth=2.0, marker='o', markersize=6)
    ax.plot(x_values, neg_arr, label='Negative changes', color='#800000',
            linestyle='--', linewidth=2.0, marker='s', markersize=6)

    ax.set_ylim(0, 1); ax.set_xticks(x_values); ax.set_xlim(0.5, len(pos_arr) + 0.5)
    ax.set_xlabel('Consensus Level'); ax.set_ylabel('Sensitivity')
    ax.set_title(f'Sensitivity at Consensus Levels ({n})', pad=15)
    ax.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')

    for x, y in zip(x_values, pos_arr):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9, color='#003366')
    for x, y in zip(x_values, neg_arr):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                    xytext=(0, -14), ha='center', fontsize=9, color='#800000')

    plt.tight_layout(); plt.savefig(out_p)
    plt.clf(); plt.cla(); plt.close()


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

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
            rr, cc = ellipse(l['cx'], l['cy'], l['rx'], l['ry'],
                             shape=shape, rotation=np.deg2rad(l['angle']))
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


def load_model_labels_map(nif_path, crop_info_path, current_shape,
                          min_cc_size=50, min_cc_intensity=0.03):
    """
    Load the 512x512 model output and inverse-map it back to the original
    coordinate space using the crop_info.json saved during prediction.

    The crop_info records coordinates in the *transposed* (torch) space
    (Prediction.py loads images with .T). Since nifti files and annotations
    live in the *non-transposed* nifti space, we transpose into torch space,
    place the output, and transpose back.

    Steps:
      1. Read crop_info.json → get sq_x_min, sq_y_min, square_size, orig_h, orig_w
      2. Load 512x512 model output, transpose to torch space
      3. Resize to (square_size, square_size)
      4. Place into zero canvas of (orig_h, orig_w) at [sq_x_min:, sq_y_min:]
      5. Transpose back to nifti space → should match current_shape
    """
    with open(crop_info_path) as f:
        crop_info = json.load(f)

    ci = crop_info['current']
    sq_x_min = ci['sq_x_min']
    sq_y_min = ci['sq_y_min']
    square_size = ci['square_size']
    orig_h = ci['orig_h']
    orig_w = ci['orig_w']

    model_output = nib.load(nif_path).get_fdata()
    if model_output.ndim == 3:
        model_output = model_output[:, :, 0]

    # Transpose to torch space (same space as crop_info coordinates)
    model_output = model_output.T

    # Resize from 512x512 to the original square crop size
    model_resized = sk_resize(model_output, (square_size, square_size),
                              order=1, preserve_range=True, anti_aliasing=False)

    # Place into full-size canvas in torch space (orig_h x orig_w)
    canvas = np.zeros((orig_h, orig_w), dtype=model_resized.dtype)
    canvas[sq_x_min:sq_x_min + square_size, sq_y_min:sq_y_min + square_size] = model_resized

    # Transpose back to nifti space
    full_output = canvas.T

    assert full_output.shape == current_shape, \
        f"Shape mismatch: inverse-mapped {full_output.shape} vs current {current_shape}"

    print(f"  Inverse mapping: sq=({sq_x_min}, {sq_y_min}), size={square_size}, "
          f"orig=({orig_h}x{orig_w}), target={current_shape}")

    # Filter: remove CCs that are too small or not intense enough
    pos_map = (full_output > 0).astype(int)
    neg_map = (full_output < 0).astype(int)

    for binary_map, sign in [(pos_map, 1), (neg_map, -1)]:
        ccs, num_ccs = label(binary_map, STRUCT_3x3)
        for cc_val in range(1, num_ccs + 1):
            cc_mask = ccs == cc_val
            cc_size = np.sum(cc_mask)
            cc_mean_intensity = np.abs(full_output[cc_mask]).mean()
            if cc_size < min_cc_size or cc_mean_intensity < min_cc_intensity:
                binary_map[cc_mask] = 0

    return pos_map, neg_map, full_output


# ---------------------------------------------------------------------------
# Per-pair overlay visualization
# ---------------------------------------------------------------------------

def plot_pair_overlay(current, full_model_output, human_pos, human_neg,
                      pair_name, plots_dir):
    """
    Save a 3-panel figure:
      1. Model output overlaid on original scan
      2. Annotation (pos/neg) overlaid on original scan
      3. Both overlaid for comparison
    """
    pair_plots_dir = os.path.join(plots_dir, pair_name)
    os.makedirs(pair_plots_dir, exist_ok=True)

    current_t = current.T  # transpose to (H, W) for imshow

    # --- Panel 1: Model output on scan ---
    model_t = full_model_output.T
    plot_diff_on_current(model_t, current_t,
                         os.path.join(pair_plots_dir, 'model_on_original.png'))

    # --- Panel 2: Annotation on scan ---
    combined_annot = human_pos.copy()
    combined_annot[human_neg != 0] = human_neg[human_neg != 0]
    annot_t = combined_annot.T
    plot_diff_on_current(annot_t, current_t,
                         os.path.join(pair_plots_dir, 'annotation_on_original.png'))

    # --- Panel 3: Side-by-side comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Original scan
    axes[0].imshow(current_t, cmap='gray')
    axes[0].set_title('Original Scan', fontsize=14, fontweight='bold')
    axes[0].set_axis_off()

    # Model output
    axes[1].imshow(current_t, cmap='gray')
    alphas_m = generate_alpha_map(model_t)
    vmin_m = min(np.min(model_t).item(), -0.01)
    vmax_m = max(np.max(model_t).item(), 0.01)
    divnorm_m = colors.TwoSlopeNorm(vmin=vmin_m, vcenter=0., vmax=vmax_m)
    axes[1].imshow(model_t, alpha=alphas_m, cmap=differential_grad, norm=divnorm_m)
    axes[1].set_title('Model Output', fontsize=14, fontweight='bold')
    axes[1].set_axis_off()

    # Annotation
    axes[2].imshow(current_t, cmap='gray')
    alphas_a = generate_alpha_map(annot_t)
    vmin_a = min(np.min(annot_t).item(), -0.01)
    vmax_a = max(np.max(annot_t).item(), 0.01)
    divnorm_a = colors.TwoSlopeNorm(vmin=vmin_a, vcenter=0., vmax=vmax_a)
    axes[2].imshow(annot_t, alpha=alphas_a, cmap=differential_grad, norm=divnorm_a)
    axes[2].set_title('Annotation', fontsize=14, fontweight='bold')
    axes[2].set_axis_off()

    fig.suptitle(pair_name, fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(pair_plots_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Detection / agreement metrics (unchanged from original)
# ---------------------------------------------------------------------------

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
            idx = vals2_inter.tolist().index(0)
            vals2_inter = np.delete(vals2_inter, idx)
            counts2_inter = np.delete(counts2_inter, idx)
        if len(vals2_inter) == 0:
            disagreements += 1
            continue
        inter_val_2 = vals2_inter[np.argsort(counts2_inter)[::-1][0]]
        ccs2[ccs2 == inter_val_2] = 0
        agreements += 1

    disagreements += max(0, len(np.unique(ccs2)) - 1)
    return agreements, disagreements


def get_HMDR_and_UDPP_counts(model_map, human_maps):
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

    return sensitivities


def plot_correlation_matrix(corr_df, category_name, c_out_path):
    plot_matrix(corr_df, c_out_path, title=f'{category_name} Correlation Matrix')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_outputs_base_path, annotations_base_path, pairs_roots,
         out_path, plots_dir, num_pairs=60, bias=0,
         min_cc_size=50, min_cc_intensity=0.03):
    os.makedirs(out_path, exist_ok=True)
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

    num_humans = 5
    num_observers = 6
    physician_to_idx_dict = {'Avi': 0, 'Benny': 1, 'Sigal': 2, 'Smadar': 3, 'Nitzan': 4, 'Model': 5}

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

    total_model_preds_pos = 0; total_model_preds_neg = 0
    total_overlapping_model_preds_pos = 0; total_overlapping_model_preds_neg = 0
    not_overlapping_model_preds_pos = []; not_overlapping_model_preds_neg = []

    total_avi_preds_pos = 0; total_avi_preds_neg = 0
    total_overlapping_avi_preds_pos = 0; total_overlapping_avi_preds_neg = 0
    not_overlapping_avi_preds_pos = []; not_overlapping_avi_preds_neg = []

    total_benny_preds_pos = 0; total_benny_preds_neg = 0
    total_overlapping_benny_preds_pos = 0; total_overlapping_benny_preds_neg = 0
    not_overlapping_benny_preds_pos = []; not_overlapping_benny_preds_neg = []

    total_sigal_preds_pos = 0; total_sigal_preds_neg = 0
    total_overlapping_sigal_preds_pos = 0; total_overlapping_sigal_preds_neg = 0
    not_overlapping_sigal_preds_pos = []; not_overlapping_sigal_preds_neg = []

    total_smadar_preds_pos = 0; total_smadar_preds_neg = 0
    total_overlapping_smadar_preds_pos = 0; total_overlapping_smadar_preds_neg = 0
    not_overlapping_smadar_preds_pos = []; not_overlapping_smadar_preds_neg = []

    total_nitzan_preds_pos = 0; total_nitzan_preds_neg = 0
    total_overlapping_nitzan_preds_pos = 0; total_overlapping_nitzan_preds_neg = 0
    not_overlapping_nitzan_preds_pos = []; not_overlapping_nitzan_preds_neg = []

    pairs_processed = 0

    for i in range(bias, num_pairs + bias):
        print(f'Pair {i + 1}')
        try:
            pair_path = resolve_pair_path(pairs_roots, i + 1)
        except FileNotFoundError as e:
            print(f'SKIP pair {i+1}: {e}')
            continue

        # --- Load current scan ---
        scans = sorted([p for p in os.listdir(pair_path)
                        if p.endswith('.nii.gz') and '_seg' not in p])
        if len(scans) < 2:
            print(f'SKIP pair {i+1}: fewer than 2 scans in {pair_path}')
            continue
        current_path = f'{pair_path}/{scans[1]}'
        current = load_xray(current_path)

        # --- Load annotations ---
        try:
            annotation_path_avi = resolve_annotation_path(annotations_base_path, 'Avi', i + 1)
            annotation_path_benny = resolve_annotation_path(annotations_base_path, 'Benny', i + 1)
            annotation_path_sigal = resolve_annotation_path(annotations_base_path, 'Sigal', i + 1)
            annotation_path_smadar = resolve_annotation_path(annotations_base_path, 'Smadar', i + 1)
            annotation_path_nitzan = resolve_annotation_path(annotations_base_path, 'Nitzan', i + 1)
            annotation_path_model = resolve_model_output(model_outputs_base_path, i + 1)
            crop_info_path = resolve_crop_info(model_outputs_base_path, i + 1)
        except FileNotFoundError as e:
            print(f'SKIP pair {i+1}: {e}')
            continue

        # Doctor annotations rasterized at current.shape (full image coords)
        label_map_pos_avi, label_map_neg_avi = load_labels_map(annotation_path_avi, current.shape)
        label_map_pos_benny, label_map_neg_benny = load_labels_map(annotation_path_benny, current.shape)
        label_map_pos_sigal, label_map_neg_sigal = load_labels_map(annotation_path_sigal, current.shape)
        label_map_pos_smadar, label_map_neg_smadar = load_labels_map(annotation_path_smadar, current.shape)
        label_map_pos_nitzan, label_map_neg_nitzan = load_labels_map(annotation_path_nitzan, current.shape)

        # Model output: inverse-mapped back to current.shape using crop_info.json
        try:
            label_map_pos_model, label_map_neg_model, full_model_output = load_model_labels_map(
                annotation_path_model, crop_info_path, current.shape,
                min_cc_size=min_cc_size, min_cc_intensity=min_cc_intensity)
        except Exception as e:
            print(f'SKIP pair {i+1}: model inverse mapping failed: {e}')
            continue

        # --- Per-pair overlay plot ---
        if plots_dir:
            # Combine all human annotations for visualization (use first available)
            human_pos_union = label_map_pos_avi.copy()
            human_neg_union = label_map_neg_avi.copy()
            plot_pair_overlay(current, full_model_output, human_pos_union, human_neg_union,
                              f'pair{i+1}', plots_dir)

        # --- Sensitivity at consensus levels ---

        # Model
        c_sensitivities_pos = get_sensitivity_at_consensus_levels(
            label_map_pos_model,
            (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg = get_sensitivity_at_consensus_levels(
            label_map_neg_model,
            (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))

        for k, (detections, changes) in enumerate(c_sensitivities_pos):
            sensitivities_pos[k][0] += detections
            sensitivities_pos[k][1] += changes
        for k, (detections, changes) in enumerate(c_sensitivities_neg):
            sensitivities_neg[k][0] += detections
            sensitivities_neg[k][1] += changes

        # Humans
        c_sensitivities_pos_avi = get_sensitivity_at_consensus_levels(label_map_pos_avi, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_avi = get_sensitivity_at_consensus_levels(label_map_neg_avi, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        for k, (d, c) in enumerate(c_sensitivities_pos_avi): sensitivities_pos_avi[k][0] += d; sensitivities_pos_avi[k][1] += c
        for k, (d, c) in enumerate(c_sensitivities_neg_avi): sensitivities_neg_avi[k][0] += d; sensitivities_neg_avi[k][1] += c

        c_sensitivities_pos_benny = get_sensitivity_at_consensus_levels(label_map_pos_benny, (label_map_pos_avi, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_benny = get_sensitivity_at_consensus_levels(label_map_neg_benny, (label_map_neg_avi, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        for k, (d, c) in enumerate(c_sensitivities_pos_benny): sensitivities_pos_benny[k][0] += d; sensitivities_pos_benny[k][1] += c
        for k, (d, c) in enumerate(c_sensitivities_neg_benny): sensitivities_neg_benny[k][0] += d; sensitivities_neg_benny[k][1] += c

        c_sensitivities_pos_sigal = get_sensitivity_at_consensus_levels(label_map_pos_sigal, (label_map_pos_benny, label_map_pos_avi, label_map_pos_smadar, label_map_pos_nitzan))
        c_sensitivities_neg_sigal = get_sensitivity_at_consensus_levels(label_map_neg_sigal, (label_map_neg_benny, label_map_neg_avi, label_map_neg_smadar, label_map_neg_nitzan))
        for k, (d, c) in enumerate(c_sensitivities_pos_sigal): sensitivities_pos_sigal[k][0] += d; sensitivities_pos_sigal[k][1] += c
        for k, (d, c) in enumerate(c_sensitivities_neg_sigal): sensitivities_neg_sigal[k][0] += d; sensitivities_neg_sigal[k][1] += c

        c_sensitivities_pos_smadar = get_sensitivity_at_consensus_levels(label_map_pos_smadar, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_avi, label_map_pos_nitzan))
        c_sensitivities_neg_smadar = get_sensitivity_at_consensus_levels(label_map_neg_smadar, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_avi, label_map_neg_nitzan))
        for k, (d, c) in enumerate(c_sensitivities_pos_smadar): sensitivities_pos_smadar[k][0] += d; sensitivities_pos_smadar[k][1] += c
        for k, (d, c) in enumerate(c_sensitivities_neg_smadar): sensitivities_neg_smadar[k][0] += d; sensitivities_neg_smadar[k][1] += c

        c_sensitivities_pos_nitzan = get_sensitivity_at_consensus_levels(label_map_pos_nitzan, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_avi))
        c_sensitivities_neg_nitzan = get_sensitivity_at_consensus_levels(label_map_neg_nitzan, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_avi))
        for k, (d, c) in enumerate(c_sensitivities_pos_nitzan): sensitivities_pos_nitzan[k][0] += d; sensitivities_pos_nitzan[k][1] += c
        for k, (d, c) in enumerate(c_sensitivities_neg_nitzan): sensitivities_neg_nitzan[k][0] += d; sensitivities_neg_nitzan[k][1] += c

        # --- HMDR / UDPP ---

        # Model
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_model, (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_model, (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        total_model_preds_pos += total_preds_pos; total_model_preds_neg += total_preds_neg
        total_overlapping_model_preds_pos += overlapping_pos; total_overlapping_model_preds_neg += overlapping_neg
        not_overlapping_model_preds_pos.append(not_overlapping_pos); not_overlapping_model_preds_neg.append(not_overlapping_neg)

        # Avi
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_avi, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_avi, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        total_avi_preds_pos += total_preds_pos; total_avi_preds_neg += total_preds_neg
        total_overlapping_avi_preds_pos += overlapping_pos; total_overlapping_avi_preds_neg += overlapping_neg
        not_overlapping_avi_preds_pos.append(not_overlapping_pos); not_overlapping_avi_preds_neg.append(not_overlapping_neg)

        # Benny
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_benny, (label_map_pos_avi, label_map_pos_sigal, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_benny, (label_map_neg_avi, label_map_neg_sigal, label_map_neg_smadar, label_map_neg_nitzan))
        total_benny_preds_pos += total_preds_pos; total_benny_preds_neg += total_preds_neg
        total_overlapping_benny_preds_pos += overlapping_pos; total_overlapping_benny_preds_neg += overlapping_neg
        not_overlapping_benny_preds_pos.append(not_overlapping_pos); not_overlapping_benny_preds_neg.append(not_overlapping_neg)

        # Sigal
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_sigal, (label_map_pos_benny, label_map_pos_avi, label_map_pos_smadar, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_sigal, (label_map_neg_benny, label_map_neg_avi, label_map_neg_smadar, label_map_neg_nitzan))
        total_sigal_preds_pos += total_preds_pos; total_sigal_preds_neg += total_preds_neg
        total_overlapping_sigal_preds_pos += overlapping_pos; total_overlapping_sigal_preds_neg += overlapping_neg
        not_overlapping_sigal_preds_pos.append(not_overlapping_pos); not_overlapping_sigal_preds_neg.append(not_overlapping_neg)

        # Smadar
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_smadar, (label_map_pos_benny, label_map_pos_sigal, label_map_pos_avi, label_map_pos_nitzan))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_smadar, (label_map_neg_benny, label_map_neg_sigal, label_map_neg_avi, label_map_neg_nitzan))
        total_smadar_preds_pos += total_preds_pos; total_smadar_preds_neg += total_preds_neg
        total_overlapping_smadar_preds_pos += overlapping_pos; total_overlapping_smadar_preds_neg += overlapping_neg
        not_overlapping_smadar_preds_pos.append(not_overlapping_pos); not_overlapping_smadar_preds_neg.append(not_overlapping_neg)

        # Nitzan
        overlapping_pos, not_overlapping_pos, total_preds_pos = get_HMDR_and_UDPP_counts(label_map_pos_nitzan, (label_map_pos_avi, label_map_pos_benny, label_map_pos_sigal, label_map_pos_smadar))
        overlapping_neg, not_overlapping_neg, total_preds_neg = get_HMDR_and_UDPP_counts(label_map_neg_nitzan, (label_map_neg_avi, label_map_neg_benny, label_map_neg_sigal, label_map_neg_smadar))
        total_nitzan_preds_pos += total_preds_pos; total_nitzan_preds_neg += total_preds_neg
        total_overlapping_nitzan_preds_pos += overlapping_pos; total_overlapping_nitzan_preds_neg += overlapping_neg
        not_overlapping_nitzan_preds_pos.append(not_overlapping_pos); not_overlapping_nitzan_preds_neg.append(not_overlapping_neg)

        # --- PAI ---
        inf_list_pos = ((label_map_pos_avi, 'Avi'), (label_map_pos_benny, 'Benny'), (label_map_pos_sigal, 'Sigal'), (label_map_pos_smadar, 'Smadar'), (label_map_pos_nitzan, 'Nitzan'), (label_map_pos_model, 'Model'))
        inf_list_neg = ((label_map_neg_avi, 'Avi'), (label_map_neg_benny, 'Benny'), (label_map_neg_sigal, 'Sigal'), (label_map_neg_smadar, 'Smadar'), (label_map_neg_nitzan, 'Nitzan'), (label_map_neg_model, 'Model'))

        for inf_list, agreement_mat, disagreement_mat, agreement_per_pair_mat, total_labels_lst, lists_for_per_pair_all in zip(
                [inf_list_pos, inf_list_neg],
                [pairwise_agreement_mat_pos, pairwise_agreement_mat_neg],
                [pairwise_disagreement_mat_pos, pairwise_disagreement_mat_neg],
                [pairwise_agreement_per_pair_mat_pos, pairwise_agreement_per_pair_mat_neg],
                [total_labels_pos, total_labels_neg],
                [(agreements_num_list_pos, disagreements_num_list_pos), (agreements_num_list_neg, disagreements_num_list_neg)]):

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

            for label_map_entry in inf_list:
                name = label_map_entry[1]
                lm = label_map_entry[0]
                labels_arr, ccs_num = label(lm, struct)
                total_labels_lst[name].append(ccs_num)

        pairs_processed += 1

    if pairs_processed == 0:
        print("No pairs were processed!")
        return

    # --- Sensitivity at consensus levels ---
    final_sensitivities = {f'Sensitivity Consensus Level {i + 1} (Positive)': s[0] / s[1] if s[1] > 0 else 0
                           for i, s in enumerate(sensitivities_pos)}
    final_sensitivities.update({f'Sensitivity Consensus Level {i + 1} (Negative)': s[0] / s[1] if s[1] > 0 else 0
                                for i, s in enumerate(sensitivities_neg)})
    final_sensitivities.update({f'Total detections & changes at Consensus Level {i + 1} (Positive)': (s[0], s[1])
                                for i, s in enumerate(sensitivities_pos)})
    final_sensitivities.update({f'Total detections & changes at Consensus Level {i + 1} (Negative)': (s[0], s[1])
                                for i, s in enumerate(sensitivities_neg)})

    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg],
                '$M_{ICU}$', f'{out_path}/sensitivity_consensus_levels.png')

    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos_avi],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg_avi],
                'A', f'{out_path}/sensitivity_consensus_levels_avi.png')
    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos_benny],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg_benny],
                'B', f'{out_path}/sensitivity_consensus_levels_benny.png')
    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos_sigal],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg_sigal],
                'C', f'{out_path}/sensitivity_consensus_levels_sigal.png')
    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos_smadar],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg_smadar],
                'D', f'{out_path}/sensitivity_consensus_levels_smadar.png')
    plot_curves([s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_pos_nitzan],
                [s[0] / s[1] if s[1] > 0 else 0 for s in sensitivities_neg_nitzan],
                'E', f'{out_path}/sensitivity_consensus_levels_nitzan.png')

    with open(f'{out_path}/sensitivity_measures.json', 'w') as f:
        json.dump(final_sensitivities, f, indent=4)

    # --- HMDR / UDPP ---
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    hmdr_model_pos = safe_div(total_overlapping_model_preds_pos, total_model_preds_pos)
    hmdr_model_neg = safe_div(total_overlapping_model_preds_neg, total_model_preds_neg)
    udpp_model_pos = sum(not_overlapping_model_preds_pos) / pairs_processed
    udpp_model_neg = sum(not_overlapping_model_preds_neg) / pairs_processed
    udpp_model_pos_std = float(np.std(not_overlapping_model_preds_pos))
    udpp_model_neg_std = float(np.std(not_overlapping_model_preds_neg))

    hmdr_avi_pos = safe_div(total_overlapping_avi_preds_pos, total_avi_preds_pos)
    hmdr_avi_neg = safe_div(total_overlapping_avi_preds_neg, total_avi_preds_neg)
    udpp_avi_pos = sum(not_overlapping_avi_preds_pos) / pairs_processed
    udpp_avi_neg = sum(not_overlapping_avi_preds_neg) / pairs_processed
    udpp_avi_pos_std = float(np.std(not_overlapping_avi_preds_pos))
    udpp_avi_neg_std = float(np.std(not_overlapping_avi_preds_neg))

    hmdr_benny_pos = safe_div(total_overlapping_benny_preds_pos, total_benny_preds_pos)
    hmdr_benny_neg = safe_div(total_overlapping_benny_preds_neg, total_benny_preds_neg)
    udpp_benny_pos = sum(not_overlapping_benny_preds_pos) / pairs_processed
    udpp_benny_neg = sum(not_overlapping_benny_preds_neg) / pairs_processed
    udpp_benny_pos_std = float(np.std(not_overlapping_benny_preds_pos))
    udpp_benny_neg_std = float(np.std(not_overlapping_benny_preds_neg))

    hmdr_sigal_pos = safe_div(total_overlapping_sigal_preds_pos, total_sigal_preds_pos)
    hmdr_sigal_neg = safe_div(total_overlapping_sigal_preds_neg, total_sigal_preds_neg)
    udpp_sigal_pos = sum(not_overlapping_sigal_preds_pos) / pairs_processed
    udpp_sigal_neg = sum(not_overlapping_sigal_preds_neg) / pairs_processed
    udpp_sigal_pos_std = float(np.std(not_overlapping_sigal_preds_pos))
    udpp_sigal_neg_std = float(np.std(not_overlapping_sigal_preds_neg))

    hmdr_smadar_pos = safe_div(total_overlapping_smadar_preds_pos, total_smadar_preds_pos)
    hmdr_smadar_neg = safe_div(total_overlapping_smadar_preds_neg, total_smadar_preds_neg)
    udpp_smadar_pos = sum(not_overlapping_smadar_preds_pos) / pairs_processed
    udpp_smadar_neg = sum(not_overlapping_smadar_preds_neg) / pairs_processed
    udpp_smadar_pos_std = float(np.std(not_overlapping_smadar_preds_pos))
    udpp_smadar_neg_std = float(np.std(not_overlapping_smadar_preds_neg))

    hmdr_nitzan_pos = safe_div(total_overlapping_nitzan_preds_pos, total_nitzan_preds_pos)
    hmdr_nitzan_neg = safe_div(total_overlapping_nitzan_preds_neg, total_nitzan_preds_neg)
    udpp_nitzan_pos = sum(not_overlapping_nitzan_preds_pos) / pairs_processed
    udpp_nitzan_neg = sum(not_overlapping_nitzan_preds_neg) / pairs_processed
    udpp_nitzan_pos_std = float(np.std(not_overlapping_nitzan_preds_pos))
    udpp_nitzan_neg_std = float(np.std(not_overlapping_nitzan_preds_neg))

    precision_measures_dict = {
        'Model HMDR (Positive)': hmdr_model_pos, 'Model HMDR (Negative)': hmdr_model_neg,
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

    # --- PAI ---
    mat_per_label_pos = pairwise_agreement_mat_pos / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos)
    mat_per_label_neg = pairwise_agreement_mat_neg / (pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_label_all = (pairwise_agreement_mat_pos + pairwise_agreement_mat_neg) / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos + pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_pair_pos = pairwise_agreement_per_pair_mat_pos / pairs_processed
    mat_per_pair_neg = pairwise_agreement_per_pair_mat_neg / pairs_processed

    mat_per_pair_all = np.eye(num_observers)
    print('\n=== PAI per-pair diagnostic ===')
    for n1, n2 in combinations(physician_to_idx_dict.keys(), 2):
        pai_per_pair = 0
        per_pair_pos_vals = []
        per_pair_neg_vals = []
        per_pair_all_vals = []
        boost_count = 0
        both_empty_count = 0
        n_pairs_for_this = len(agreements_num_list_pos[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]])
        for k in range(n_pairs_for_this):
            c_ag_pos = agreements_num_list_pos[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_ag_neg = agreements_num_list_neg[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_ag = c_ag_pos + c_ag_neg
            c_disag_pos = disagreements_num_list_pos[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_disag_neg = disagreements_num_list_neg[physician_to_idx_dict[n1]][physician_to_idx_dict[n2]][k]
            c_disag = c_disag_pos + c_disag_neg

            # per-pair pos / neg PAI (matches mat_per_pair_pos / mat_per_pair_neg logic)
            pos_pai_k = np.nan_to_num(c_ag_pos / np.array(c_ag_pos + c_disag_pos), nan=1., posinf=1., neginf=1.)
            neg_pai_k = np.nan_to_num(c_ag_neg / np.array(c_ag_neg + c_disag_neg), nan=1., posinf=1., neginf=1.)
            per_pair_pos_vals.append(float(pos_pai_k))
            per_pair_neg_vals.append(float(neg_pai_k))

            c_pai = np.nan_to_num(c_ag / np.array(c_ag + c_disag), nan=1., posinf=1., neginf=1.)
            one_sided = (c_ag_pos + c_disag_pos == 0 and c_ag_neg + c_disag_neg > 0) or (c_ag_pos + c_disag_pos > 0 and c_ag_neg + c_disag_neg == 0)
            if one_sided:
                boost_count += 1
                c_pai = c_pai * 0.5 + 0.5
            if c_ag_pos + c_disag_pos == 0 and c_ag_neg + c_disag_neg == 0:
                both_empty_count += 1
            per_pair_all_vals.append(float(c_pai))
            pai_per_pair += c_pai
        mat_per_pair_all[physician_to_idx_dict[n1], physician_to_idx_dict[n2]] = pai_per_pair / pairs_processed
        mat_per_pair_all[physician_to_idx_dict[n2], physician_to_idx_dict[n1]] = pai_per_pair / pairs_processed

        pos_mean = float(np.mean(per_pair_pos_vals)) if per_pair_pos_vals else float('nan')
        neg_mean = float(np.mean(per_pair_neg_vals)) if per_pair_neg_vals else float('nan')
        all_mean = float(np.mean(per_pair_all_vals)) if per_pair_all_vals else float('nan')
        print(f'\n[{n1} vs {n2}] pairs={n_pairs_for_this} both_empty={both_empty_count} one_sided_boosted={boost_count}')
        print(f'  pos per-pair vals: {[round(v, 3) for v in per_pair_pos_vals]}')
        print(f'  neg per-pair vals: {[round(v, 3) for v in per_pair_neg_vals]}')
        print(f'  all per-pair vals: {[round(v, 3) for v in per_pair_all_vals]}')
        print(f'  means -> pos={pos_mean:.4f}  neg={neg_mean:.4f}  all={all_mean:.4f}'
              f'  (mean(pos,neg)={(pos_mean+neg_mean)/2:.4f})')

    index_names = list(physician_to_idx_dict.keys())

    per_label_df_pos = pd.DataFrame({phy_n: mat_per_label_pos[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)
    per_label_df_neg = pd.DataFrame({phy_n: mat_per_label_neg[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)
    per_label_df_all = pd.DataFrame({phy_n: mat_per_label_all[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)
    per_pair_df_pos = pd.DataFrame({phy_n: mat_per_pair_pos[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)
    per_pair_df_neg = pd.DataFrame({phy_n: mat_per_pair_neg[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)
    per_pair_df_all = pd.DataFrame({phy_n: mat_per_pair_all[i].tolist() for i, phy_n in enumerate(physician_to_idx_dict.keys())}, index=index_names)

    plot_matrix(per_label_df_pos, f'{out_path}/per_label_agreement_pos.png', title='Pairwise Agreement Index Per Detection (positive)')
    plot_matrix(per_label_df_neg, f'{out_path}/per_label_agreement_neg.png', title='Pairwise Agreement Index Per Detection (negative)')
    plot_matrix(per_label_df_all, f'{out_path}/per_label_agreement_all.png', title='Pairwise Agreement Index Per Detection (all)')
    plot_matrix(per_pair_df_pos, f'{out_path}/per_pair_agreement_pos.png', title='Pairwise Agreement Index Per Pair (positive)')
    plot_matrix(per_pair_df_neg, f'{out_path}/per_pair_agreement_neg.png', title='Pairwise Agreement Index Per Pair (negative)')
    plot_matrix(per_pair_df_all, f'{out_path}/per_pair_agreement_all.png', title='Pairwise Agreement Index Per Pair (all)')

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
        total_labels_data_pos[phy] = (int(np.sum(arr)), float(np.mean(arr)), float(np.std(arr)), int(np.max(arr)), int(np.min(arr)))
    for phy, l in total_labels_neg.items():
        arr = np.array(l)
        total_labels_data_neg[phy] = (int(np.sum(arr)), float(np.mean(arr)), float(np.std(arr)), int(np.max(arr)), int(np.min(arr)))
    for phy in physician_to_idx_dict.keys():
        pos_labels = total_labels_pos[phy]
        neg_labels = total_labels_neg[phy]
        all_labels = np.array([pos_labels[idx] + neg_labels[idx] for idx in range(len(pos_labels))])
        total_labels_data_all[phy] = (int(np.sum(all_labels)), float(np.mean(all_labels)), float(np.std(all_labels)), int(np.max(all_labels)), int(np.min(all_labels)))

    with open(f'{out_path}/total_labels_marked_pos.json', 'w') as f:
        json.dump(total_labels_data_pos, f, indent=4)
    with open(f'{out_path}/total_labels_marked_neg.json', 'w') as f:
        json.dump(total_labels_data_neg, f, indent=4)
    with open(f'{out_path}/total_labels_marked_all.json', 'w') as f:
        json.dump(total_labels_data_all, f, indent=4)

    print(f'\nProcessed {pairs_processed} pairs. Results saved to {out_path}')
    if plots_dir:
        print(f'Per-pair overlay plots saved to {plots_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Observer Variability Analysis (square-crop with crop_info.json)')
    parser.add_argument('--model-preds-dir', required=True,
                        help='Path to model predictions directory (with crop_info.json per pair)')
    parser.add_argument('--annotations-dir', required=True,
                        help='Path to annotations base directory')
    parser.add_argument('--pairs-roots', nargs='+', required=True,
                        help='Pair root directories (Pairs1 .. Pairs8)')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for OV results')
    parser.add_argument('--plots-dir', default=None,
                        help='Directory for per-pair overlay plots (model + annotation on original scan). '
                             'If omitted, no per-pair plots are generated.')
    parser.add_argument('--num-pairs', type=int, default=60)
    parser.add_argument('--bias', type=int, default=0)
    parser.add_argument('--min-cc-size', type=int, default=50,
                        help='Minimum connected component size (pixels) to keep in model output.')
    parser.add_argument('--min-cc-intensity', type=float, default=0.03,
                        help='Minimum mean absolute intensity of a CC to keep.')
    args = parser.parse_args()

    struct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))

    main(args.model_preds_dir, args.annotations_dir, args.pairs_roots,
         args.out_dir, args.plots_dir, args.num_pairs, args.bias,
         args.min_cc_size, args.min_cc_intensity)
