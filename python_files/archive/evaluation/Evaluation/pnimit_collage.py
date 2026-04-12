"""
PNIMIT collage: for each pair show (2 rows x 3 columns):
  Row 1: Prior scan | Current scan | Doctor annotations on current
  Row 2: Full-thorax model (seg-corrected) | Lungs-med5 model (seg-corrected) | Annotations + Full Thorax overlay

Doctor annotation labels:
  Positive (red, worsening):  Appearance, Persistence+Increase
  Negative (green, improving): Disappearance, Persistence+Decrease
  Neutral (yellow): Persistence+None
"""

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize as sk_resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Ellipse


def compute_seg_bbox(seg_data, crop_pad_val=15):
    seg_coords = np.nonzero(seg_data)
    x_min = max(int(np.min(seg_coords[0])) - crop_pad_val, 0)
    x_max = min(int(np.max(seg_coords[0])) + crop_pad_val, seg_data.shape[0] - 1)
    y_min = max(int(np.min(seg_coords[1])) - crop_pad_val, 0)
    y_max = min(int(np.max(seg_coords[1])) + crop_pad_val, seg_data.shape[1] - 1)
    return x_min, x_max, y_min, y_max


def inverse_map_output(output_nii_path, seg_path, full_shape):
    """Load 512x512 model output and map back to full image coords via seg bbox."""
    seg_data = nib.load(seg_path).get_fdata()
    if seg_data.ndim == 3:
        seg_data = seg_data[:, :, 0]
    x_min, x_max, y_min, y_max = compute_seg_bbox(seg_data)
    crop_h, crop_w = x_max - x_min, y_max - y_min

    model_out = nib.load(output_nii_path).get_fdata()
    if model_out.ndim == 3:
        model_out = model_out[:, :, 0]

    model_cropped = sk_resize(model_out, (crop_h, crop_w), order=1,
                              preserve_range=True, anti_aliasing=False)
    canvas = np.zeros(full_shape, dtype=model_cropped.dtype)
    canvas[x_min:x_max, y_min:y_max] = model_cropped
    return canvas


def generate_alpha_map(x, min_alpha=0.0):
    x_abs = np.abs(x)
    max_val = max(np.max(x_abs).item(), 0.07)
    return np.clip(x_abs / max_val, min_alpha, 1.0)


def classify_annotation(ann):
    """Return 'pos', 'neg', or 'neutral' based on annotation label."""
    label = ann.get('label', '')
    if label == 'Appearance':
        return 'pos'
    elif label == 'Disappearance':
        return 'neg'
    elif label == 'Persistence':
        size_ch = ann.get('size_change', 'None')
        int_ch = ann.get('intensity_change', 'None')
        has_increase = (size_ch == 'Increase' or int_ch == 'Increase')
        has_decrease = (size_ch == 'Decrease' or int_ch == 'Decrease')
        if has_increase and not has_decrease:
            return 'pos'
        elif has_decrease and not has_increase:
            return 'neg'
        elif has_increase and has_decrease:
            return 'mixed'
        else:
            return 'neutral'
    return 'neutral'


def short_label(ann):
    """Short text for the annotation."""
    label = ann.get('label', '')
    tag = ann.get('tag', '')
    cl = classify_annotation(ann)
    symbol = {'pos': '▲', 'neg': '▼', 'neutral': '—', 'mixed': '▲▼'}.get(cl, '?')
    parts = []
    if label == 'Appearance':
        parts.append('App')
    elif label == 'Disappearance':
        parts.append('Disapp')
    elif label == 'Persistence':
        sc = ann.get('size_change', 'None')
        ic = ann.get('intensity_change', 'None')
        changes = []
        if sc != 'None':
            changes.append(f'S:{sc[:3]}')
        if ic != 'None':
            changes.append(f'I:{ic[:3]}')
        parts.append(f'Pers({",".join(changes) if changes else "stable"})')
    if tag and tag != 'Other':
        parts.append(tag[:5])
    return f'{symbol} {" ".join(parts)}'


COLOR_MAP = {
    'pos': (1.0, 0.2, 0.2),      # red - worsening
    'neg': (0.2, 0.9, 0.2),      # green - improving
    'neutral': (1.0, 1.0, 0.3),  # yellow - stable
    'mixed': (1.0, 0.6, 0.0),    # orange - mixed
}


def draw_annotations(ax, annotations, img_shape):
    """Draw ellipse outlines with labels on an axis.

    Annotation coords are in 792x792 canvas space (annotation tool resizes .T to 792x792).
    Display is imshow(current.T) at original resolution, so we scale from 792 to actual dims.
    img_shape is (H, W) of the original (non-transposed) data.
    After .T the imshow axes are: x-axis = H cols, y-axis = W rows.
    """
    CANVAS_SIZE = 792
    scale_x = img_shape[0] / CANVAS_SIZE
    scale_y = img_shape[1] / CANVAS_SIZE

    for ann in annotations:
        cl = classify_annotation(ann)
        color = COLOR_MAP.get(cl, (1, 1, 1))
        cx = ann['cx'] * scale_x
        cy = ann['cy'] * scale_y
        rx = ann['rx'] * scale_x
        ry = ann['ry'] * scale_y
        angle = ann.get('angle', 0)

        ell = Ellipse((cx, cy), 2 * rx, 2 * ry, angle=angle,
                      edgecolor=color, facecolor='none', linewidth=2, linestyle='-')
        ax.add_patch(ell)

        txt = short_label(ann)
        ax.text(cx, cy - ry - 5, txt, color=color, fontsize=7,
                fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))


def find_annotation_json(ann_root, pair_name):
    """Find annotation JSON in the annotations root for a given pair name."""
    pair_dir = os.path.join(ann_root, pair_name)
    if os.path.isdir(pair_dir):
        for f in os.listdir(pair_dir):
            if f.lower().endswith('.json'):
                return os.path.join(pair_dir, f)
    for f in os.listdir(ann_root):
        if f.lower().endswith('.json') and pair_name.lower() in f.lower():
            return os.path.join(ann_root, f)
    return None


def find_seg_for_current(seg_root, pair_name, current_name):
    """Find seg mask file for the current scan."""
    pair_seg_dir = os.path.join(seg_root, pair_name)
    if os.path.isdir(pair_seg_dir):
        cands = [
            os.path.join(pair_seg_dir, f'{current_name}_seg.nii.gz'),
            os.path.join(pair_seg_dir, f'{current_name}_lung_seg.nii.gz'),
        ]
        for c in cands:
            if os.path.exists(c):
                return c
    cands = [
        os.path.join(seg_root, f'{current_name}_seg.nii.gz'),
        os.path.join(seg_root, f'{current_name}_lung_seg.nii.gz'),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def process_pair(pair_name, pairs_root, preds_ft_root, preds_med5_root,
                 segs_ft_root, segs_med5_root, ann_root, out_dir):
    """Generate collage for one PNIMIT pair (2 rows x 3 columns)."""
    pair_dir = os.path.join(pairs_root, pair_name)
    if not os.path.isdir(pair_dir):
        print(f'SKIP {pair_name}: pair dir not found')
        return

    scans = sorted([f for f in os.listdir(pair_dir)
                    if f.endswith('.nii.gz') and '_seg' not in f])
    if len(scans) < 2:
        print(f'SKIP {pair_name}: not enough scans')
        return

    prior_path = os.path.join(pair_dir, scans[0])
    current_path = os.path.join(pair_dir, scans[1])
    prior_name = scans[0].replace('.nii.gz', '')
    current_name = scans[1].replace('.nii.gz', '')

    prior = nib.load(prior_path).get_fdata()
    current = nib.load(current_path).get_fdata()
    if prior.ndim == 3:
        prior = prior[:, :, 0]
    if current.ndim == 3:
        current = current[:, :, 0]

    full_shape = current.shape
    print(f'{pair_name}: prior={scans[0]}, current={scans[1]}, shape={full_shape}')

    # --- Load model outputs with inverse mapping ---
    pred_ft_path = os.path.join(preds_ft_root, pair_name, 'output.nii.gz')
    pred_med5_path = os.path.join(preds_med5_root, pair_name, 'output.nii.gz')

    seg_ft_path = find_seg_for_current(segs_ft_root, pair_name, current_name)
    seg_med5_path = find_seg_for_current(segs_med5_root, pair_name, current_name)

    has_ft = os.path.exists(pred_ft_path) and seg_ft_path
    has_med5 = os.path.exists(pred_med5_path) and seg_med5_path

    model_ft = inverse_map_output(pred_ft_path, seg_ft_path, full_shape) if has_ft else None
    model_med5 = inverse_map_output(pred_med5_path, seg_med5_path, full_shape) if has_med5 else None

    # --- Load annotations ---
    ann_json = find_annotation_json(ann_root, pair_name)
    annotations = []
    if ann_json:
        with open(ann_json) as f:
            data = json.load(f)
        annotations = [e for e in data[1:] if isinstance(e, dict)]

    # --- Colormap ---
    differential_grad = colors.LinearSegmentedColormap.from_list('my_gradient', (
        (0.000, (0.235, 1.000, 0.239)),
        (0.400, (0.000, 1.000, 0.702)),
        (0.500, (1.000, 0.988, 0.988)),
        (0.600, (1.000, 0.604, 0.000)),
        (1.000, (0.682, 0.000, 0.000))))

    # --- Build figure: 2 rows x 3 cols ---
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # Row 1, Col 0: Prior
    axes[0, 0].imshow(prior.T, cmap='gray')
    axes[0, 0].set_title(f'Prior: {prior_name}', fontsize=11, fontweight='bold')
    axes[0, 0].set_axis_off()

    # Row 1, Col 1: Current
    axes[0, 1].imshow(current.T, cmap='gray')
    axes[0, 1].set_title(f'Current: {current_name}', fontsize=11, fontweight='bold')
    axes[0, 1].set_axis_off()

    # Row 1, Col 2: Doctor annotations on current
    axes[0, 2].imshow(current.T, cmap='gray')
    if annotations:
        draw_annotations(axes[0, 2], annotations, full_shape)
        n_pos = sum(1 for a in annotations if classify_annotation(a) == 'pos')
        n_neg = sum(1 for a in annotations if classify_annotation(a) == 'neg')
        n_other = len(annotations) - n_pos - n_neg
        axes[0, 2].set_title(
            f'Annotations ({len(annotations)}): '
            f'Red={n_pos} Green={n_neg} Other={n_other}',
            fontsize=11, fontweight='bold')
    else:
        axes[0, 2].set_title('No Annotations', fontsize=11, fontweight='bold', color='gray')
    axes[0, 2].set_axis_off()

    # --- Helper for model output ---
    def plot_model_output(ax, current_img, model_out, title):
        ax.imshow(current_img.T, cmap='gray')
        if model_out is not None:
            vmin = min(np.min(model_out).item(), -0.01)
            vmax = max(np.max(model_out).item(), 0.01)
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
            alphas = generate_alpha_map(model_out)
            ax.imshow(model_out.T, alpha=alphas.T, cmap=differential_grad, norm=divnorm)
            ax.set_title(title, fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'{title}\n(not available)', fontsize=11, fontweight='bold', color='gray')
        ax.set_axis_off()

    # Row 2, Col 0: Full Thorax model
    plot_model_output(axes[1, 0], current, model_ft, 'Model: Full Thorax (seg-corrected)')

    # Row 2, Col 1: Lungs Med-Margin5 model
    plot_model_output(axes[1, 1], current, model_med5, 'Model: Lungs Med-Margin5 (seg-corrected)')

    # Row 2, Col 2: Annotations + Full Thorax model overlay
    axes[1, 2].imshow(current.T, cmap='gray')
    if model_ft is not None:
        vmin = min(np.min(model_ft).item(), -0.01)
        vmax = max(np.max(model_ft).item(), 0.01)
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        alphas = generate_alpha_map(model_ft)
        axes[1, 2].imshow(model_ft.T, alpha=alphas.T, cmap=differential_grad, norm=divnorm)
    if annotations:
        draw_annotations(axes[1, 2], annotations, full_shape)
    axes[1, 2].set_title('Annotations + Full Thorax Model', fontsize=11, fontweight='bold')
    axes[1, 2].set_axis_off()

    fig.suptitle(f'{pair_name}', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{pair_name}_collage.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='PNIMIT collage: model predictions + doctor annotations')
    parser.add_argument('--pairs-root', required=True, help='Pairs_PNIMIT_1_pairs dir')
    parser.add_argument('--preds-ft', required=True, help='predictions_pnimit_full_thorax dir')
    parser.add_argument('--preds-med5', required=True, help='predictions_pnimit_lungs_med5 dir')
    parser.add_argument('--segs-ft', required=True, help='roi_masks_pnimit/full_thorax dir')
    parser.add_argument('--segs-med5', required=True, help='roi_masks_pnimit/lungs_med_margin5 dir')
    parser.add_argument('--annotations', required=True, help='PNIMIT annotations root (pair_A4_1_2 style)')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    pair_names = sorted([d for d in os.listdir(args.preds_ft) if os.path.isdir(os.path.join(args.preds_ft, d))])
    print(f'Found {len(pair_names)} PNIMIT pairs')

    for pn in pair_names:
        process_pair(pn, args.pairs_root, args.preds_ft, args.preds_med5,
                     args.segs_ft, args.segs_med5, args.annotations, args.out_dir)


if __name__ == '__main__':
    main()
