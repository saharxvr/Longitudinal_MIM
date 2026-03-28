from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_2d(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D NIfTI, got shape {arr.shape} for {path}")
    return arr


def normalize_01(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def render_preview(pair_name: str, input_nii: Path, mask_nii: Path, out_dir: Path) -> Path:
    img = normalize_01(load_2d(input_nii))
    mask = (load_2d(mask_nii) > 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.T, cmap="gray")
    axes[0].set_title(f"{pair_name} - CXR")
    axes[0].axis("off")

    axes[1].imshow(mask.T, cmap="gray")
    axes[1].set_title(f"{pair_name} - Thorax Mask")
    axes[1].axis("off")

    axes[2].imshow(img.T, cmap="gray")
    axes[2].imshow(mask.T, cmap="Reds", alpha=0.35)
    axes[2].set_title(f"{pair_name} - Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    out_path = out_dir / f"{pair_name}_preview.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_panel(preview_paths: list[Path], out_dir: Path) -> Path:
    images = [plt.imread(str(p)) for p in preview_paths]
    fig, axes = plt.subplots(len(images), 1, figsize=(16, 5 * len(images)))
    if len(images) == 1:
        axes = [axes]

    for ax, im, p in zip(axes, images, preview_paths):
        ax.imshow(im)
        ax.set_title(p.stem)
        ax.axis("off")

    fig.tight_layout()
    panel_path = out_dir / "pnimit_thorax_preview_panel.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def main() -> None:
    root = Path(".")
    pairs_root = root / "annotation tool" / "Pairs_PNIMIT_1_pairs"
    seg_root = root / "Sahar_work" / "files" / "pnimit_thorax_segmentations"
    out_dir = seg_root / "preview_png"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        "pair_A10_1_2": "SynapseExport (28).nii.gz",
        "pair_A11_1_2": "SynapseExport (31).nii.gz",
        "pair_A12_1_2": "SynapseExport (34).nii.gz",
    }

    preview_paths: list[Path] = []
    for pair_name, scan_name in selected.items():
        input_nii = pairs_root / pair_name / scan_name
        mask_nii = seg_root / pair_name / "thorax_output.nii.gz"
        if not input_nii.exists() or not mask_nii.exists():
            print(f"Skipping {pair_name}: missing input or mask")
            continue
        preview_paths.append(render_preview(pair_name, input_nii, mask_nii, out_dir))

    if preview_paths:
        panel_path = render_panel(preview_paths, out_dir)
        print(f"Saved panel: {panel_path}")
    else:
        print("No previews generated.")


if __name__ == "__main__":
    main()
