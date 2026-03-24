import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
from torchvision.transforms.functional import adjust_sharpness


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


def preprocess_with_seg(img: torch.Tensor, boundary_seg: torch.Tensor, resize, crop_pad_val=15):
    assert img.shape == boundary_seg.shape, (img.shape, boundary_seg.shape)

    boundary_seg_coords = boundary_seg.nonzero().T
    x_min, x_max = torch.min(boundary_seg_coords[-2]), torch.max(boundary_seg_coords[-2])
    y_min, y_max = torch.min(boundary_seg_coords[-1]), torch.max(boundary_seg_coords[-1])

    x_min = max(x_min.item() - crop_pad_val, 0)
    y_min = max(y_min.item() - crop_pad_val, 0)
    x_max = min(x_max.item() + crop_pad_val, img.shape[-2] - 1)
    y_max = min(y_max.item() + crop_pad_val, img.shape[-1] - 1)

    img = img[..., x_min:x_max, y_min:y_max]
    seg = boundary_seg[..., x_min:x_max, y_min:y_max]

    img = resize(img[None, ...])
    seg = resize(seg[None, ...])

    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    img = adjust_sharpness(img, sharpness_factor=4.0)
    img = torch.clamp(img, 0.0, 1.0)

    return img.squeeze().numpy(), seg.squeeze().numpy()


def preprocess_no_seg(img: torch.Tensor, resize):
    img = resize(img[None, ...])
    img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    img = adjust_sharpness(img, sharpness_factor=4.0)
    img = torch.clamp(img, 0.0, 1.0)
    return img.squeeze().numpy()


def resize_mask_nearest(mask: np.ndarray, target_hw):
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
    mask_t = F.interpolate(mask_t, size=target_hw, mode='nearest')
    return mask_t.squeeze().numpy()


def harmonize_seg_to_img(seg: torch.Tensor, img: torch.Tensor):
    if seg.shape == img.shape:
        return seg

    seg_np = seg.numpy().astype(np.float32)
    seg_rs = resize_mask_nearest(seg_np, img.shape)
    return torch.from_numpy(seg_rs)


def _plot_overlay(current_img, output_map, out_path: Path, colormap, title='Model output'):
    output_abs = np.abs(output_map)
    max_val = max(float(np.max(output_abs)), 0.01)
    alpha = output_abs / max_val

    divnorm = colors.TwoSlopeNorm(
        vmin=min(float(np.min(output_map)), -0.01),
        vcenter=0.0,
        vmax=max(float(np.max(output_map)), 0.01),
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(current_img, cmap='gray')
    im = plt.imshow(output_map, alpha=alpha, cmap=colormap, norm=divnorm)
    plt.colorbar(im)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _get_current_and_seg(pair_dir: Path):
    nii_files = sorted([p for p in pair_dir.glob('*.nii.gz') if not p.name.endswith('_lung_seg.nii.gz')])
    if len(nii_files) < 2:
        raise FileNotFoundError(f'Expected 2 non-seg NIfTI files in {pair_dir}, found {len(nii_files)}')

    current_path = nii_files[1]
    current_seg_path = pair_dir / f'{_safe_stem_nii_gz(current_path)}_lung_seg.nii.gz'
    if not current_seg_path.exists():
        raise FileNotFoundError(f'Missing current segmentation file: {current_seg_path}')

    return current_path, current_seg_path


def process_pair(pair_name: str, pred_pair_dir: Path, pairs_root: Path, resize, colormap, mask_mode='full-resize'):
    pair_dir = pairs_root / pair_name
    if not pair_dir.exists():
        raise FileNotFoundError(f'Pair folder not found in pairs root: {pair_dir}')

    output_path = pred_pair_dir / 'output.nii.gz'
    if not output_path.exists():
        raise FileNotFoundError(f'Model output not found: {output_path}')

    current_path, current_seg_path = _get_current_and_seg(pair_dir)

    current = torch.tensor(nib.load(str(current_path)).get_fdata().T, dtype=torch.float32)
    current_seg = torch.tensor(nib.load(str(current_seg_path)).get_fdata().T, dtype=torch.float32)
    current_seg = harmonize_seg_to_img(current_seg, current)

    if torch.count_nonzero(current_seg) == 0:
        raise ValueError(f'Empty segmentation mask for {pair_name}: {current_seg_path}')

    if mask_mode == 'crop-resize':
        current_proc, seg_proc = preprocess_with_seg(current, current_seg, resize, crop_pad_val=15)
    elif mask_mode == 'full-resize':
        current_proc = preprocess_no_seg(current, resize)
        seg_proc = resize_mask_nearest(current_seg.numpy().astype(np.float32), (512, 512))
    else:
        raise ValueError(f'Unknown mask mode: {mask_mode}')

    output = nib.load(str(output_path)).get_fdata().T
    if output.shape != seg_proc.shape:
        raise ValueError(f'Shape mismatch for {pair_name}: output={output.shape}, seg={seg_proc.shape}')

    seg_mask = seg_proc > 0.5
    output_masked = output * seg_mask

    output_bin_masked = ((output > 0).astype(np.float32) - (output < 0).astype(np.float32)) * seg_mask

    _plot_overlay(current_proc, output_masked, pred_pair_dir / 'model_output_masked_by_seg.png', colormap)
    _plot_overlay(current_proc, output_bin_masked, pred_pair_dir / 'model_output_masked_by_seg_bin.png', colormap)

    masked_nif = nib.Nifti1Image(output_masked.T.astype(np.float32), affine=np.eye(4))
    nib.save(masked_nif, str(pred_pair_dir / 'output_masked_by_seg.nii.gz'))


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
    parser.add_argument(
        '--mask-mode',
        type=str,
        choices=['full-resize', 'crop-resize'],
        default='full-resize',
        help='full-resize matches outputs produced without seg-cropping; crop-resize matches seg-cropped preprocessing.',
    )
    args = parser.parse_args()

    predictions_dir = args.predictions_dir
    pairs_dir = args.pairs_dir

    if not predictions_dir.exists():
        raise FileNotFoundError(f'Predictions directory not found: {predictions_dir}')
    if not pairs_dir.exists():
        raise FileNotFoundError(f'Pairs directory not found: {pairs_dir}')

    resize = v2.Resize((512, 512))
    colormap = _build_colormap()

    pair_dirs = sorted([p for p in predictions_dir.iterdir() if p.is_dir()])
    processed = 0

    for pred_pair_dir in pair_dirs:
        pair_name = pred_pair_dir.name
        process_pair(pair_name, pred_pair_dir, pairs_dir, resize, colormap, mask_mode=args.mask_mode)
        processed += 1

    print(f'Processed {processed} pairs in {predictions_dir} (mask_mode={args.mask_mode})')


if __name__ == '__main__':
    main()
