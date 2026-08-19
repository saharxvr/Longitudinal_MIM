"""
Regenerate the 6 individual PAI heatmap plots with anonymized labels
(A, B, C, D, E, M_ICU) from no_cc_itamar_segs_30 data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Anonymisation mapping: Avi→A, Benny→B, Sigal→C, Smadar→D, Nitzan→E, Model→M_ICU ──
anon = ['A', 'B', 'C', 'D', 'E', r'$M_{ICU}$']

# ── Data ──
det_all = np.array([
    [1.00, 0.47, 0.62, 0.51, 0.62, 0.54],
    [0.47, 1.00, 0.61, 0.55, 0.56, 0.47],
    [0.62, 0.61, 1.00, 0.63, 0.68, 0.56],
    [0.51, 0.55, 0.63, 1.00, 0.72, 0.51],
    [0.62, 0.56, 0.68, 0.72, 1.00, 0.56],
    [0.54, 0.47, 0.56, 0.51, 0.56, 1.00],
])
pair_all = np.array([
    [1.00, 0.50, 0.65, 0.54, 0.66, 0.60],
    [0.50, 1.00, 0.64, 0.60, 0.56, 0.54],
    [0.65, 0.64, 1.00, 0.69, 0.70, 0.59],
    [0.54, 0.60, 0.69, 1.00, 0.74, 0.58],
    [0.66, 0.56, 0.70, 0.74, 1.00, 0.61],
    [0.60, 0.54, 0.59, 0.58, 0.61, 1.00],
])
det_pos = np.array([
    [1.00, 0.50, 0.69, 0.57, 0.65, 0.56],
    [0.50, 1.00, 0.75, 0.58, 0.60, 0.55],
    [0.69, 0.75, 1.00, 0.67, 0.75, 0.63],
    [0.57, 0.58, 0.67, 1.00, 0.71, 0.55],
    [0.65, 0.60, 0.75, 0.71, 1.00, 0.60],
    [0.56, 0.55, 0.63, 0.55, 0.60, 1.00],
])
pair_pos = np.array([
    [1.00, 0.41, 0.59, 0.55, 0.66, 0.59],
    [0.41, 1.00, 0.70, 0.52, 0.48, 0.48],
    [0.59, 0.70, 1.00, 0.64, 0.64, 0.60],
    [0.55, 0.52, 0.64, 1.00, 0.70, 0.61],
    [0.66, 0.48, 0.64, 0.70, 1.00, 0.62],
    [0.59, 0.48, 0.60, 0.61, 0.62, 1.00],
])
det_neg = np.array([
    [1.00, 0.43, 0.55, 0.44, 0.59, 0.52],
    [0.43, 1.00, 0.45, 0.50, 0.50, 0.38],
    [0.55, 0.45, 1.00, 0.59, 0.62, 0.49],
    [0.44, 0.50, 0.59, 1.00, 0.72, 0.47],
    [0.59, 0.50, 0.62, 0.72, 1.00, 0.53],
    [0.52, 0.38, 0.49, 0.47, 0.53, 1.00],
])
pair_neg = np.array([
    [1.00, 0.49, 0.58, 0.49, 0.57, 0.56],
    [0.49, 1.00, 0.48, 0.57, 0.48, 0.47],
    [0.58, 0.48, 1.00, 0.62, 0.62, 0.50],
    [0.49, 0.57, 0.62, 1.00, 0.69, 0.52],
    [0.57, 0.48, 0.62, 0.69, 1.00, 0.53],
    [0.56, 0.47, 0.50, 0.52, 0.53, 1.00],
])


def plot_matrix(mat, output_path, title):
    """Replicate the exact style of the OV script's plot_matrix."""
    N = mat.shape[0]
    df = pd.DataFrame(mat, index=anon, columns=anon)

    figsize = max(8, int(N * 0.6))
    plt.figure(figsize=(figsize, figsize))
    ax = sns.heatmap(
        df, annot=True, fmt=".2f", cmap="vlag",
        vmin=0, vmax=1, center=0.25,
        linewidths=0., linecolor='black',
        cbar=False,
        annot_kws={"fontsize": 22, 'fontweight': 'bold'})

    separator_index = N - 1
    ax.axvline(x=separator_index, ymin=1 / N, color='black',
               linewidth=1.25, linestyle='--')
    ax.axhline(y=separator_index, xmax=1 - 1 / N, color='black',
               linewidth=1.25, linestyle='--')

    group1_x = (N - 1) / 2
    group2_x = (N - 1) + 0.5
    text_y_pos = -0.04
    ax.text(group1_x, text_y_pos, 'H - H', ha='center', va='bottom',
            fontsize=22, fontweight='bold', fontstyle='italic', color='#333333')
    ax.text(group2_x, text_y_pos, 'M - H', ha='center', va='bottom',
            fontsize=22, fontweight='bold', fontstyle='italic', color='#333333')

    plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
    plt.yticks(rotation=0, fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


out_dir = os.path.join(os.path.dirname(__file__), 'files', 'ov_results_sq',
                       'no_cc_itamar_segs_30')

plots = [
    (det_pos,  'per_label_agreement_pos_anon.png',
     'Pairwise Agreement Index Per Detection (positive)'),
    (det_neg,  'per_label_agreement_neg_anon.png',
     'Pairwise Agreement Index Per Detection (negative)'),
    (det_all,  'per_label_agreement_all_anon.png',
     'Pairwise Agreement Index Per Detection (all)'),
    (pair_pos, 'per_pair_agreement_pos_anon.png',
     'Pairwise Agreement Index Per Pair (positive)'),
    (pair_neg, 'per_pair_agreement_neg_anon.png',
     'Pairwise Agreement Index Per Pair (negative)'),
    (pair_all, 'per_pair_agreement_all_anon.png',
     'Pairwise Agreement Index Per Pair (all)'),
]

for mat, fname, title in plots:
    plot_matrix(mat, os.path.join(out_dir, fname), title)

# ── Combined 3x2 figure by compositing the saved individual PNGs ──
from PIL import Image

# Layout: rows = All/Positive/Negative, cols = Per-detection/Per-Pair
grid = [
    ('per_label_agreement_all_anon.png', 'per_pair_agreement_all_anon.png'),
    ('per_label_agreement_pos_anon.png', 'per_pair_agreement_pos_anon.png'),
    ('per_label_agreement_neg_anon.png', 'per_pair_agreement_neg_anon.png'),
]
row_labels = ['All\nchanges', 'Positive\nchanges', 'Negative\nchanges']

# Load all images to get dimensions
imgs = []
for row in grid:
    row_imgs = []
    for fname in row:
        img = Image.open(os.path.join(out_dir, fname))
        row_imgs.append(img)
    imgs.append(row_imgs)

# All individual plots have the same size
cell_w, cell_h = imgs[0][0].size

# Spacing
label_w = 700   # left margin for row labels
header_h = 150  # top margin for column headers
gap_x = 40      # horizontal gap between columns
gap_y = 40      # vertical gap between rows

total_w = label_w + cell_w + gap_x + cell_w + 20
total_h = header_h + cell_h * 3 + gap_y * 2 + 20

canvas = Image.new('RGB', (total_w, total_h), 'white')

# Add column headers
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

col_labels = ['Per Detection', 'Per Pair']
for c_idx, col_label in enumerate(col_labels):
    fig_hdr = Figure(figsize=(cell_w / 300, header_h / 300), dpi=300)
    canvas_hdr = FigureCanvasAgg(fig_hdr)
    fig_hdr.text(0.5, 0.5, col_label, ha='center', va='center',
                 fontsize=30, fontweight='bold')
    canvas_hdr.draw()
    buf_hdr = canvas_hdr.buffer_rgba()
    hdr_img = Image.frombuffer('RGBA',
                               (int(fig_hdr.get_figwidth() * 300),
                                int(fig_hdr.get_figheight() * 300)),
                               buf_hdr, 'raw', 'RGBA', 0, 1).convert('RGB')
    x = label_w + c_idx * (cell_w + gap_x)
    canvas.paste(hdr_img, (x, 0))

for r in range(3):
    for c in range(2):
        x = label_w + c * (cell_w + gap_x)
        y = header_h + r * (cell_h + gap_y)
        canvas.paste(imgs[r][c], (x, y))

# Add row labels
for r, label in enumerate(row_labels):
    fig_lbl = Figure(figsize=(label_w / 300, cell_h / 300), dpi=300)
    canvas_lbl = FigureCanvasAgg(fig_lbl)
    fig_lbl.text(0.5, 0.5, label, ha='center', va='center',
                 fontsize=30, fontweight='bold', rotation=0)
    canvas_lbl.draw()
    buf = canvas_lbl.buffer_rgba()
    lbl_img = Image.frombuffer('RGBA',
                               (int(fig_lbl.get_figwidth() * 300),
                                int(fig_lbl.get_figheight() * 300)),
                               buf, 'raw', 'RGBA', 0, 1).convert('RGB')
    y = header_h + r * (cell_h + gap_y)
    canvas.paste(lbl_img, (0, y))

combined_path = os.path.join(out_dir, 'combined_pai_figure.png')
canvas.save(combined_path, dpi=(300, 300))
print(f'Saved combined: {combined_path}')
