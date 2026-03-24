from pathlib import Path
import math
import traceback
import sys
import os

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label as cc_label

REPO = Path(r"c:/Users/saharaharon/thesis/Longitudinal_MIM")
PY_FILES = REPO / "python_files"
sys.path.insert(0, str(PY_FILES))

import predict_lung_segmentation as pls

roots = [
    REPO / "python_files/annotation tool/Pairs1",
    REPO / "python_files/annotation tool/Pairs2",
    REPO / "python_files/annotation tool/Pairs3",
    REPO / "python_files/annotation tool/Pairs4",
    REPO / "python_files/annotation tool/Pairs5",
    REPO / "python_files/annotation tool/Pairs6",
    REPO / "python_files/annotation tool/Pairs7",
    REPO / "python_files/annotation tool/Pairs8",
    REPO / "python_files/annotation tool/Pairs_PNIMIT_1_pairs",
]

pair_dirs = []
for root in roots:
    if not root.exists():
        continue
    pair_dirs.extend([p for p in root.iterdir() if p.is_dir()])

pair_dirs = sorted(pair_dirs, key=lambda p: p.name.lower())

# Make TorchXRayVision Windows-safe for progress bar output.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def refine_lung_mask(mask: np.ndarray) -> np.ndarray:
    mask_bin = mask.astype(bool)
    if not np.any(mask_bin):
        return mask.astype(np.uint8)

    labeled, num = cc_label(mask_bin)
    if num > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        # Keep up to the two largest connected components (left/right lungs).
        keep_labels = np.argsort(counts)[-2:]
        keep_labels = [int(v) for v in keep_labels if int(v) > 0 and counts[int(v)] > 0]
        if keep_labels:
            mask_bin = np.isin(labeled, keep_labels)

    mask_bin = binary_fill_holes(mask_bin)
    return mask_bin.astype(np.uint8)


@torch.no_grad()
def infer_single(model, device, nii_path: Path, out_path: Path):
    img_np, src_nii = pls._load_nifti_2d(nii_path, transpose_input=True)
    img_np = pls._normalize_01(img_np)
    img_np = img_np * 2048.0 - 1024.0

    inp = torch.from_numpy(img_np)[None, None, ...].to(device)
    original_hw = tuple(inp.shape[-2:])

    # Keep geometric behavior consistent with predict_lung_segmentation.py by
    # resizing to 512x512 for the model and then resizing logits back.
    inp = F.interpolate(inp, size=(512, 512), mode="bilinear", align_corners=False)

    out = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]

    prob = pls._output_to_prob(out)
    if tuple(prob.shape[-2:]) != original_hw:
        prob = F.interpolate(prob, size=original_hw, mode="bilinear", align_corners=False)

    mask = (prob >= 0.5).to(torch.uint8).squeeze().cpu().numpy().T
    mask = refine_lung_mask(mask)

    out_nii = nib.Nifti1Image(mask.astype(np.uint8), affine=src_nii.affine, header=src_nii.header)
    nib.save(out_nii, str(out_path))


def load_2d_for_preview(nii_path: Path):
    data = np.asarray(nib.load(str(nii_path)).get_fdata())
    if data.ndim == 2:
        img = data.T
    elif data.ndim == 3:
        if 1 in data.shape:
            img = np.squeeze(data).T
        else:
            z = data.shape[2] // 2
            img = data[:, :, z].T
    else:
        return None

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(np.min(img)), float(np.max(img))
    if vmax > vmin:
        norm = (img - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(img, dtype=np.float32)
    return norm


def build_review_images(pair_dir: Path):
    out_dir = pair_dir / "review_png"
    out_dir.mkdir(exist_ok=True)

    nii_files = sorted(pair_dir.glob("*.nii.gz"))
    exported = []

    for nii_path in nii_files:
        arr = load_2d_for_preview(nii_path)
        if arr is None:
            continue
        png_path = out_dir / f"{nii_path.name[:-7]}.png"
        plt.figure(figsize=(5.5, 5.5))
        plt.imshow(arr, cmap="gray", origin="upper")
        plt.title(nii_path.name)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=130)
        plt.close()
        exported.append((nii_path.name, arr))

    if exported:
        n = len(exported)
        cols = 2
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 7.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = axes[:, None]

        for i, (name, arr) in enumerate(exported):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            ax.imshow(arr, cmap="gray", origin="upper")
            ax.set_title(name)
            ax.axis("off")

        for i in range(len(exported), rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "contact_sheet.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pls._XrvChestXDetWrapper().to(device)
    model.eval()

    processed_pairs = 0
    processed_inputs = 0
    seg_written = 0
    failures = []

    print(f"[bulk] device={device}, pair_dirs={len(pair_dirs)}")

    for idx, pair_dir in enumerate(pair_dirs, start=1):
        try:
            nii_inputs = sorted([
                p for p in pair_dir.glob("*.nii.gz")
                if not p.name.endswith("_lung_seg.nii.gz") and not p.name.endswith("_seg.nii.gz")
            ])

            for inp in nii_inputs:
                out = pair_dir / f"{inp.name[:-7]}_lung_seg.nii.gz"
                infer_single(model, device, inp, out)
                seg_written += 1
                processed_inputs += 1

            build_review_images(pair_dir)
            processed_pairs += 1

            if idx % 10 == 0 or idx == len(pair_dirs):
                print(f"[bulk] done {idx}/{len(pair_dirs)} pairs | segs={seg_written}")

        except Exception as e:
            failures.append((str(pair_dir), str(e)))
            print(f"[bulk][error] {pair_dir}: {e}")
            traceback.print_exc()

    summary = REPO / "python_files/annotation tool/bulk_seg_review_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"pair_dirs_total={len(pair_dirs)}\\n")
        f.write(f"pairs_processed={processed_pairs}\\n")
        f.write(f"inputs_processed={processed_inputs}\\n")
        f.write(f"segmentations_written={seg_written}\\n")
        f.write(f"failures={len(failures)}\\n")
        for p, err in failures:
            f.write(f"FAIL\\t{p}\\t{err}\\n")

    print(f"[bulk] complete. summary={summary}")
    if failures:
        sys.exit(1)
