"""
Predict lung segmentation mask from a chest X-ray NIfTI using a trained model.

This script supports three loading modes:
1) TorchScript model (recommended): --model-path model.ts --model-type torchscript
2) Python model + checkpoint: provide --model-module, --model-class, --model-path
3) Open-source pretrained model (no local checkpoint): --model-type xrv-chestx-det
4) Open-source pretrained thorax mask (no local checkpoint): --model-type xrv-chestx-det-thorax

Expected model input:
- Tensor of shape [B, 1, H, W]

Expected model output:
- [B, 1, H, W] logits or probabilities, or
- [B, 2, H, W] class logits/probabilities (foreground is class index 1)

Example (TorchScript):
python predict_lung_segmentation.py \
  --input-nii path/to/cxr.nii.gz \
  --output-nii path/to/cxr_lung_seg.nii.gz \
  --model-path path/to/lung_seg_model.ts \
  --model-type torchscript \
  --device cuda

Example (Python class + checkpoint):
python predict_lung_segmentation.py \
  --input-nii path/to/cxr.nii.gz \
  --output-nii path/to/cxr_lung_seg.nii.gz \
  --model-path path/to/checkpoint.pt \
  --model-type python \
  --model-module my_models \
  --model-class UNet \
  --model-kwargs "{'in_channels': 1, 'out_channels': 1}"

Example (Open-source pretrained ChestX-Det):
python predict_lung_segmentation.py \
    --input-nii path/to/cxr.nii.gz \
    --output-nii path/to/cxr_lung_seg.nii.gz \
    --model-type xrv-chestx-det \
    --device cuda

Example (Open-source pretrained ChestX-Det thorax):
python predict_lung_segmentation.py \
    --input-nii path/to/cxr.nii.gz \
    --output-nii path/to/cxr_thorax_seg.nii.gz \
    --model-type xrv-chestx-det-thorax \
    --device cuda
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict lung segmentation from a chest X-ray NIfTI.")
    parser.add_argument("--input-nii", type=Path, required=True, help="Input chest X-ray NIfTI file (.nii or .nii.gz).")
    parser.add_argument("--output-nii", type=Path, required=True, help="Output lung mask NIfTI file.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to trained model (.ts/.pt/.pth).")
    parser.add_argument(
        "--model-type",
        choices=["torchscript", "python", "xrv-chestx-det", "xrv-chestx-det-thorax"],
        default="xrv-chestx-det",
        help="Model loading mode. Use torchscript when possible.",
    )
    parser.add_argument("--model-module", type=str, default=None, help="Python module containing the model class.")
    parser.add_argument("--model-class", type=str, default=None, help="Model class name in --model-module.")
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="Python dict literal for model constructor kwargs, e.g. \"{'in_channels':1,'out_channels':1}\".",
    )
    parser.add_argument("--state-dict-key", type=str, default="model_dict", help="State-dict key if checkpoint is a dict.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask when output is probabilistic.")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional model input size. If provided, input is resized before inference and mask is resized back.",
    )
    parser.add_argument(
        "--transpose-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transpose HxW input to match this repo's common NIfTI convention before inference.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_nifti_2d(path: Path, transpose_input: bool) -> tuple[np.ndarray, nib.Nifti1Image]:
    nii = nib.load(str(path))
    data = np.asarray(nii.get_fdata())

    if data.ndim == 2:
        img = data
    elif data.ndim == 3 and 1 in data.shape:
        img = np.squeeze(data)
        if img.ndim != 2:
            raise ValueError(f"After squeeze, expected 2D image but got shape {img.shape} from {path}")
    else:
        raise ValueError(
            f"Expected a 2D chest X-ray NIfTI (or 3D with singleton axis), got shape {data.shape} from {path}"
        )

    if transpose_input:
        img = img.T

    return img.astype(np.float32), nii


def _normalize_01(img: np.ndarray) -> np.ndarray:
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def _build_python_model(module_name: str, class_name: str, model_kwargs: dict) -> torch.nn.Module:
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    model = model_cls(**model_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Constructed object {class_name} is not a torch.nn.Module")
    return model


class _XrvChestXDetWrapper(torch.nn.Module):
    """Adapter around TorchXRayVision ChestX-Det model returning a 1-channel target map."""

    def __init__(self, target_mode: str = "lung"):
        super().__init__()
        # TorchXRayVision's download progress bar prints block characters.
        # On some Windows terminals with cp1252 stdout this can crash.
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        import torchxrayvision as xrv

        self.model = xrv.baseline_models.chestx_det.PSPNet()
        targets = list(self.model.targets)
        if target_mode == "lung":
            selected_names = ["Left Lung", "Right Lung"]
        elif target_mode == "thorax":
            # Thorax proxy for CXR: combine lungs + central thoracic structures.
            selected_names = ["Left Lung", "Right Lung", "Heart", "Mediastinum", "Facies Diaphragmatica"]
        else:
            raise ValueError(f"Unsupported ChestX-Det target mode: {target_mode}")

        self.selected_idxs = [targets.index(name) for name in selected_names]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        selected = out[:, self.selected_idxs, ...]
        return torch.max(selected, dim=1, keepdim=True).values


def _load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.model_type == "xrv-chestx-det":
        model = _XrvChestXDetWrapper(target_mode="lung").to(device)
        model.eval()
        return model

    if args.model_type == "xrv-chestx-det-thorax":
        model = _XrvChestXDetWrapper(target_mode="thorax").to(device)
        model.eval()
        return model

    if args.model_path is None:
        raise ValueError("--model-path is required for model types torchscript and python")

    model_path = str(args.model_path)

    if args.model_type == "torchscript":
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model

    if not args.model_module or not args.model_class:
        raise ValueError("For model_type=python, both --model-module and --model-class are required")

    kwargs = ast.literal_eval(args.model_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--model-kwargs must evaluate to a dict")

    model = _build_python_model(args.model_module, args.model_class, kwargs).to(device)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and args.state_dict_key in checkpoint and isinstance(checkpoint[args.state_dict_key], dict):
        state_dict = checkpoint[args.state_dict_key]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys while loading state dict: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys while loading state dict: {len(unexpected)}")

    model.eval()
    return model


def _output_to_prob(out: torch.Tensor) -> torch.Tensor:
    if out.ndim != 4:
        raise ValueError(f"Expected model output with shape [B, C, H, W], got {tuple(out.shape)}")

    if out.shape[1] == 1:
        if torch.max(out).item() <= 1.0 and torch.min(out).item() >= 0.0:
            prob = out
        else:
            prob = torch.sigmoid(out)
        return prob

    if out.shape[1] >= 2:
        probs = torch.softmax(out, dim=1)
        return probs[:, 1:2, ...]

    raise ValueError(f"Unsupported output channel count: {out.shape[1]}")


def run_inference(args: argparse.Namespace) -> None:
    if not args.input_nii.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_nii}")
    if args.model_type in {"torchscript", "python"}:
        if args.model_path is None:
            raise ValueError("--model-path must be provided for model_type=torchscript/python")
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {args.model_path}")

    device = _resolve_device(args.device)
    model = _load_model(args, device)

    img_np, src_nii = _load_nifti_2d(args.input_nii, transpose_input=args.transpose_input)
    img_np = _normalize_01(img_np)

    # TorchXRayVision segmentation expects intensity in roughly [-1024, 1024].
    if args.model_type in {"xrv-chestx-det", "xrv-chestx-det-thorax"}:
        img_np = img_np * 2048.0 - 1024.0

    inp = torch.from_numpy(img_np)[None, None, ...].to(device)
    original_hw = tuple(inp.shape[-2:])

    if args.input_size is not None:
        target_hw = (int(args.input_size[0]), int(args.input_size[1]))
        inp = F.interpolate(inp, size=target_hw, mode="bilinear", align_corners=False)

    with torch.no_grad():
        out = model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]

    prob = _output_to_prob(out)
    if tuple(prob.shape[-2:]) != original_hw:
        prob = F.interpolate(prob, size=original_hw, mode="bilinear", align_corners=False)

    mask = (prob >= float(args.threshold)).to(torch.uint8).squeeze().cpu().numpy()

    if args.transpose_input:
        mask = mask.T

    out_nii = nib.Nifti1Image(mask.astype(np.uint8), affine=src_nii.affine, header=src_nii.header)
    args.output_nii.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(args.output_nii))

    if args.model_type == "xrv-chestx-det-thorax":
        print(f"Saved thorax mask to: {args.output_nii}")
    else:
        print(f"Saved lung mask to: {args.output_nii}")
    print(f"Mask foreground pixels: {int(mask.sum())}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
