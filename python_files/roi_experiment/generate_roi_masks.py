"""Generate 8 ROI masks per CXR pair using ChestX-Det anatomical segmentation.

ROI approaches:
  1. full_image        — no masking (all ones)
  2. lungs             — Left Lung + Right Lung
  3. lungs_heart       — Lungs + Heart
  4. lungs_mediastinum — Lungs + Mediastinum
  5. full_thorax       — Lungs + Heart + Mediastinum + Diaphragm
  6. lungs_margin5     — Lungs dilated by 5% of lung bbox diagonal
  7. lungs_med_margin5 — Lungs+Mediastinum dilated by 5%
  8. lungs_margin20    — Lungs dilated by 20% of lung bbox diagonal
  9. lungs_med_margin20 — Lungs+Mediastinum dilated by 20%
  10. lungs_convex_hull — Convex hull enclosing both lungs

Usage:
    python roi_experiment/generate_roi_masks.py \
        --pairs-roots "annotation tool/Pairs1" "annotation tool/Pairs2" ... \
        --out-dir Sahar_work/files/roi_masks \
        --start-pair 1 --end-pair 100 \
        --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.morphology import convex_hull_image

# ── ROI definitions ─────────────────────────────────────────────────────────

ROI_NAMES = [
    "full_image",
    "lungs",
    "lungs_heart",
    "lungs_mediastinum",
    "full_thorax",
    "lungs_margin5",
    "lungs_med_margin5",
    "lungs_convex_hull",
]

# ChestX-Det channel groups for each ROI
_CHANNEL_GROUPS: dict[str, list[str]] = {
    "lungs":              ["Left Lung", "Right Lung"],
    "lungs_heart":        ["Left Lung", "Right Lung", "Heart"],
    "lungs_mediastinum":  ["Left Lung", "Right Lung", "Mediastinum"],
    "full_thorax":        ["Left Lung", "Right Lung", "Heart", "Mediastinum",
                           "Facies Diaphragmatica"],
}

MARGIN_FRACTION_5 = 0.05
MARGIN_FRACTION_20 = 0.20
THRESHOLD = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_pspnet(device: torch.device):
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    import torchxrayvision as xrv
    model = xrv.baseline_models.chestx_det.PSPNet()
    model = model.to(device)
    model.eval()
    return model, list(model.targets)


def _load_nifti_2d(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata(), dtype=np.float32)
    if data.ndim > 2:
        data = np.squeeze(data)
    return data, nii


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _bbox_diagonal(mask: np.ndarray) -> float:
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def _dilate_mask(mask: np.ndarray, margin_fraction: float) -> np.ndarray:
    diag = _bbox_diagonal(mask)
    radius = max(1, int(diag * margin_fraction))
    # Use distance transform instead of binary_dilation — O(n) regardless of radius
    dist = distance_transform_edt(mask == 0)
    return (dist <= radius).astype(np.uint8)


def _make_convex_hull(mask: np.ndarray) -> np.ndarray:
    if mask.max() == 0:
        return mask.copy()
    return convex_hull_image(mask > 0).astype(np.uint8)


def _save_mask(mask: np.ndarray, ref_nii: nib.Nifti1Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine=ref_nii.affine, header=ref_nii.header)
    nib.save(nii, str(out_path))


# ── Per-image ROI computation ────────────────────────────────────────────────

def compute_channel_masks(
    img_np: np.ndarray,
    model,
    targets: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run PSPNet once and extract binary masks for each channel group."""
    # Prepare input: transpose, normalize, scale to XRV range
    img = img_np.T.copy()
    img = _normalize_01(img)
    img = img * 2048.0 - 1024.0
    inp = torch.from_numpy(img)[None, None, ...].to(device)
    inp = F.interpolate(inp, size=(512, 512), mode="bilinear", align_corners=False)

    with torch.no_grad():
        out = model(inp)                             # [1, C, 512, 512]
        out = F.interpolate(out, size=img.shape, mode="bilinear", align_corners=False)
        probs = torch.sigmoid(out).squeeze(0).cpu().numpy()  # [C, H, W]

    channel_masks: dict[str, np.ndarray] = {}
    for group_name, channel_names in _CHANNEL_GROUPS.items():
        idxs = [targets.index(n) for n in channel_names]
        combined = np.max(probs[idxs], axis=0)
        binary = (combined >= THRESHOLD).astype(np.uint8)
        # Transpose back to original orientation
        channel_masks[group_name] = binary.T

    return channel_masks


def build_all_roi_masks(
    channel_masks: dict[str, np.ndarray],
    img_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Build all 8 ROI masks from the channel group masks."""
    masks: dict[str, np.ndarray] = {}

    # 1. Full image
    masks["full_image"] = np.ones(img_shape[:2], dtype=np.uint8)

    # 2-5. Direct channel groups
    for name in ("lungs", "lungs_heart", "lungs_mediastinum", "full_thorax"):
        masks[name] = channel_masks[name]

    # 6. Lungs + 5% margin
    masks["lungs_margin5"] = _dilate_mask(channel_masks["lungs"], MARGIN_FRACTION_5)

    # 7. Lungs + Mediastinum + 5% margin
    masks["lungs_med_margin5"] = _dilate_mask(channel_masks["lungs_mediastinum"], MARGIN_FRACTION_5)

    # 8. Convex hull of lungs
    masks["lungs_convex_hull"] = _make_convex_hull(channel_masks["lungs"])

    return masks


# ── Pair resolution ──────────────────────────────────────────────────────────

def resolve_pair_path(pairs_roots: list[str], pair_num: int) -> Path:
    for root in pairs_roots:
        for prefix in ("pair", "Pair"):
            p = Path(root) / f"{prefix}{pair_num}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Pair folder not found for pair {pair_num}")


def pick_current_nii(pair_dir: Path) -> Path:
    nii_files = sorted(
        [p for p in pair_dir.iterdir()
         if p.name.endswith(".nii.gz") and "_lung_seg" not in p.name and "_seg" not in p.name],
    )
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected >=2 raw .nii.gz in {pair_dir}, found {len(nii_files)}")
    return nii_files[1]  # "current" image (second chronologically)


def pick_both_nii(pair_dir: Path) -> tuple[Path, Path]:
    """Return (prior_path, current_path) — the two raw .nii.gz files."""
    nii_files = sorted(
        [p for p in pair_dir.iterdir()
         if p.name.endswith(".nii.gz") and "_lung_seg" not in p.name and "_seg" not in p.name],
    )
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Expected >=2 raw .nii.gz in {pair_dir}, found {len(nii_files)}")
    return nii_files[0], nii_files[1]


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ROI masks for ROI experiment.")
    p.add_argument("--pairs-roots", nargs="+", required=True,
                   help="Directories containing pair folders (Pairs1, Pairs2, ...)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output root for ROI masks.")
    p.add_argument("--start-pair", type=int, default=1)
    p.add_argument("--end-pair", type=int, default=100)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, targets = _load_pspnet(device)

    for pair_num in range(args.start_pair, args.end_pair + 1):
        try:
            pair_dir = resolve_pair_path(args.pairs_roots, pair_num)
            prior_path, current_path = pick_both_nii(pair_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] Pair {pair_num}: {e}")
            continue

        # Process both images in the pair
        for nii_path in [prior_path, current_path]:
            # Image stem without .nii.gz  (e.g. "9A")
            img_stem = nii_path.name[:-7]

            img_np, ref_nii = _load_nifti_2d(nii_path)
            channel_masks = compute_channel_masks(img_np, model, targets, device)
            roi_masks = build_all_roi_masks(channel_masks, img_np.shape)

            for roi_name, mask in roi_masks.items():
                # Output:  out_dir/{roi_name}/{img_stem}_seg.nii.gz
                out_path = args.out_dir / roi_name / f"{img_stem}_seg.nii.gz"
                _save_mask(mask, ref_nii, out_path)

            fg_summary = {n: int(m.sum()) for n, m in roi_masks.items()}
            print(f"Pair {pair_num} / {img_stem}: {fg_summary}")

    print("Done.")


if __name__ == "__main__":
    main()
