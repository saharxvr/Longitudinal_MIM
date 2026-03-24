import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _safe_stem_nii_gz(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith('.nii.gz') else path.stem


def _build_colormap():
    return colors.LinearSegmentedColormap.from_list(
        'my_gradient',
        (
            (0.000, (0.235, 1.000, 0.239)),
            (0.400, (0.000, 1.000, 0.702)),
            (0.500, (1.000, 0.988, 0.988)),
            (0.600, (1.000, 0.604, 0.000)),
            (1.000, (0.682, 0.000, 0.000)),
        ),
    )


def _resize_nearest(arr: np.ndarray, target_hw):
    t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
    t = F.interpolate(t, size=target_hw, mode='nearest')
    return t.squeeze().numpy()


def _resize_bilinear(arr: np.ndarray, target_hw):
    t = torch.from_numpy(arr.astype(np.float32))[None, None, ...]
    t = F.interpolate(t, size=target_hw, mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def _load_current_and_seg(pair_dir: Path):
    nii_files = sorted([p for p in pair_dir.glob('*.nii.gz') if not p.name.endswith('_lung_seg.nii.gz')])
    if len(nii_files) < 2:
        raise FileNotFoundError(f'Expected at least 2 non-seg NIfTI files in {pair_dir}, got {len(nii_files)}')

    current_path = nii_files[1]
    current_seg_path = pair_dir / f'{_safe_stem_nii_gz(current_path)}_lung_seg.nii.gz'
    if not current_seg_path.exists():
        raise FileNotFoundError(f'Missing current seg file: {current_seg_path}')

    current = nib.load(str(current_path)).get_fdata().T.astype(np.float32)
    seg = nib.load(str(current_seg_path)).get_fdata().T.astype(np.float32)
    return current, seg


def _normalize_gray(img: np.ndarray):
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def _overlay_output(ax, current_img, output_map, cmap, title):
    output_abs = np.abs(output_map)
    max_val = max(float(np.max(output_abs)), 0.01)
    alpha = output_abs / max_val

    divnorm = colors.TwoSlopeNorm(
        vmin=min(float(np.min(output_map)), -0.01),
        vcenter=0.0,
        vmax=max(float(np.max(output_map)), 0.01),
    )

    ax.imshow(current_img, cmap='gray')
    im = ax.imshow(output_map, alpha=alpha, cmap=cmap, norm=divnorm)
    ax.set_title(title)
    ax.axis('off')
    return im


def make_collage_for_pair(pair_name: str, predictions_root: Path, pairs_root: Path, out_name: str, cmap):
    pred_pair_dir = predictions_root / pair_name
    pair_dir = pairs_root / pair_name

    output_path = pred_pair_dir / 'output.nii.gz'
    if not output_path.exists():
        raise FileNotFoundError(f'Missing model output for {pair_name}: {output_path}')

    masked_path = pred_pair_dir / 'output_masked_by_seg.nii.gz'

    output = nib.load(str(output_path)).get_fdata().T.astype(np.float32)
    if masked_path.exists():
        output_masked = nib.load(str(masked_path)).get_fdata().T.astype(np.float32)
    else:
        output_masked = None

    target_hw = output.shape

    current, seg = _load_current_and_seg(pair_dir)
    current_rs = _resize_bilinear(current, target_hw)
    current_rs = _normalize_gray(current_rs)
    seg_rs = _resize_nearest(seg, target_hw)
    seg_mask = seg_rs > 0.5

    if output_masked is None:
        output_masked = output * seg_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(current_rs, cmap='gray')
    axes[0, 0].set_title('Current image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(current_rs, cmap='gray')
    axes[0, 1].imshow(seg_mask.astype(np.float32), alpha=0.35, cmap='Blues')
    axes[0, 1].set_title('Segmentation overlay')
    axes[0, 1].axis('off')

    im1 = _overlay_output(axes[1, 0], current_rs, output, cmap, 'Model output')
    im2 = _overlay_output(axes[1, 1], current_rs, output_masked, cmap, 'Model output cropped by seg')

    cbar1 = fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=8)
    cbar2 = fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=8)

    fig.suptitle(pair_name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = pred_pair_dir / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=Path('python_files/Sahar_work/files/predictions_pnimit_in_segmentations'),
    )
    parser.add_argument(
        '--pairs-dir',
        type=Path,
        default=Path('python_files/annotation tool/Pairs_PNIMIT_1_pairs'),
    )
    parser.add_argument('--out-name', type=str, default='collage_seg_model_masked.png')
    args = parser.parse_args()

    predictions_root = args.predictions_dir
    pairs_root = args.pairs_dir

    if not predictions_root.exists():
        raise FileNotFoundError(f'Predictions directory not found: {predictions_root}')
    if not pairs_root.exists():
        raise FileNotFoundError(f'Pairs directory not found: {pairs_root}')

    cmap = _build_colormap()

    processed = 0
    failed = []

    for pred_pair_dir in sorted([p for p in predictions_root.iterdir() if p.is_dir()]):
        pair_name = pred_pair_dir.name
        try:
            make_collage_for_pair(pair_name, predictions_root, pairs_root, args.out_name, cmap)
            processed += 1
        except Exception as e:
            failed.append((pair_name, str(e)))

    print(f'Processed collages: {processed}')
    if failed:
        print(f'Failed pairs: {len(failed)}')
        for pair_name, err in failed:
            print(f'{pair_name}: {err}')


if __name__ == '__main__':
    main()
