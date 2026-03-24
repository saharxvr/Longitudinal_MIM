from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

pair = Path('python_files/annotation tool/Pairs_PNIMIT_1_pairs/pair_A2_1_2')
current_p = pair / 'SynapseExport (6).nii.gz'
old_seg_p = pair / 'SynapseExport (6)_lung_seg.nii.gz'
new_seg_p = pair / 'SynapseExport (6)_lung_seg_resized.nii.gz'
out_p = Path('python_files/Sahar_work/files/predictions_pnimit/pair_A2_1_2/seg_method_compare.png')

cur = nib.load(str(current_p)).get_fdata().T.astype(np.float32)
old_seg = nib.load(str(old_seg_p)).get_fdata().T.astype(np.float32) > 0
new_seg = nib.load(str(new_seg_p)).get_fdata().T.astype(np.float32) > 0

vmin, vmax = float(np.min(cur)), float(np.max(cur))
if vmax > vmin:
    cur = (cur - vmin) / (vmax - vmin)
else:
    cur = np.zeros_like(cur)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(cur, cmap='gray', origin='upper')
axes[0].imshow(old_seg.astype(np.float32), alpha=0.35, cmap='Reds', origin='upper')
axes[0].set_title('Original seg (square-pad pipeline)')
axes[0].axis('off')

axes[1].imshow(cur, cmap='gray', origin='upper')
axes[1].imshow(new_seg.astype(np.float32), alpha=0.35, cmap='Blues', origin='upper')
axes[1].set_title('New seg (resize 512->back)')
axes[1].axis('off')

# show disagreement areas
only_old = np.logical_and(old_seg, ~new_seg)
only_new = np.logical_and(new_seg, ~old_seg)
axes[2].imshow(cur, cmap='gray', origin='upper')
axes[2].imshow(only_old.astype(np.float32), alpha=0.45, cmap='Reds', origin='upper')
axes[2].imshow(only_new.astype(np.float32), alpha=0.45, cmap='Blues', origin='upper')
axes[2].set_title('Disagreement (red=old only, blue=new only)')
axes[2].axis('off')

fig.tight_layout()
out_p.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_p, dpi=150)
plt.close(fig)

# simple centroid report
oy, ox = np.argwhere(old_seg).mean(axis=0)
ny, nx = np.argwhere(new_seg).mean(axis=0)
print('old_center_yx', round(float(oy), 2), round(float(ox), 2))
print('new_center_yx', round(float(ny), 2), round(float(nx), 2))
print('delta_yx(new-old)', round(float(ny-oy), 2), round(float(nx-ox), 2))
print('saved', out_p)
