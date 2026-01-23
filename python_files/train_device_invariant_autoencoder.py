"""Train a device-invariant autoencoder on paired DRRs (clean vs devices).

This training script is designed for medical imaging workflows:
- Single-channel 512x512 DRRs
- GroupNorm-based decoder (reuses existing `models.Decoder6`)
- AdamW + optional AMP
- Deterministic folder-based dataset

Dataset expectation
-------------------
Input roots should contain sample directories produced by:
- CT_entities/drr_devices_pair_pipeline.py

Each sample directory must contain:
- prior.nii.gz   (clean)
- current.nii.gz (with devices)
Optionally:
- diff_map.nii.gz

Model
-----
Shared encoder E and a single decoder D.

Training objective is *device removal / device invariance*:
- Encode both clean and device-contaminated images.
- Decode both latents to the **clean** target.

Losses (default)
----------------
- Reconstruction-to-clean (both inputs): L1
- Invariance: L1 on normalized encoder feature maps

Optional
--------
- Residual consistency: also match residual (dev-clean) via (pred_from_dev - pred_from_clean)

Run example
-----------
python train_device_invariant_autoencoder.py \
  --train_roots D:/out_pairs \
  --save_dir D:/runs/devinv_001 \
  --epochs 20 --batch_size 4 --lr 2e-4 --lambda_inv 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from constants import DEVICE as DEFAULT_DEVICE, LONGITUDINAL_LOAD_PATH


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def _load_nifti_2d(path: str) -> np.ndarray:
	arr = nib.load(path).get_fdata()
	if arr.ndim != 2:
		raise ValueError(f"Expected 2D nifti at {path}, got shape {arr.shape}")
	return arr.astype(np.float32)


def _is_pair_dir(d: str) -> bool:
	req = ["prior.nii.gz", "current.nii.gz"]
	return all(os.path.isfile(os.path.join(d, r)) for r in req)


def _find_pair_dirs(roots: Sequence[str]) -> List[str]:
	pair_dirs: List[str] = []
	for root in roots:
		if not root:
			continue
		if os.path.isdir(root) and _is_pair_dir(root):
			pair_dirs.append(root)
			continue
		for cur, _dirs, _files in os.walk(root):
			if _is_pair_dir(cur):
				pair_dirs.append(cur)
	pair_dirs.sort()
	return pair_dirs


class PairedDevicesDRRDataset(Dataset):
	"""Loads (clean, devices) DRR pairs from disk.

	Returns tensors in [-1, 1] as [1,H,W].
	"""

	def __init__(
		self,
		roots: Sequence[str],
		*,
		normalize: str = "minmax",
		eps: float = 1e-8,
		return_paths: bool = False,
	):
		self.sample_dirs = _find_pair_dirs(list(roots))
		if not self.sample_dirs:
			raise ValueError(
				"PairedDevicesDRRDataset found 0 samples. Provide roots containing folders with prior.nii.gz + current.nii.gz"
			)
		if normalize not in {"minmax", "zscore", "none"}:
			raise ValueError(f"Unknown normalize mode: {normalize}")
		self.normalize = normalize
		self.eps = float(eps)
		self.return_paths = bool(return_paths)

	def __len__(self) -> int:
		return len(self.sample_dirs)

	def _norm_to_minus1_1(self, x: np.ndarray) -> np.ndarray:
		if self.normalize == "none":
			# Still map to [-1,1] assuming inputs are already [0,1]
			xn = x
		elif self.normalize == "minmax":
			mn = float(np.min(x))
			mx = float(np.max(x))
			xn = (x - mn) / (mx - mn + self.eps)
		elif self.normalize == "zscore":
			mu = float(np.mean(x))
			sd = float(np.std(x))
			xn = (x - mu) / (sd + self.eps)
			# squash to [0,1] robustly (tanh), then map to [-1,1]
			xn = 0.5 * (np.tanh(xn) + 1.0)
		else:
			raise AssertionError("unreachable")
		return (2.0 * xn - 1.0).astype(np.float32)

	def __getitem__(self, idx: int):
		d = self.sample_dirs[idx]
		clean = _load_nifti_2d(os.path.join(d, "prior.nii.gz"))
		dev = _load_nifti_2d(os.path.join(d, "current.nii.gz"))
		clean = self._norm_to_minus1_1(clean)
		dev = self._norm_to_minus1_1(dev)

		clean_t = torch.from_numpy(clean).unsqueeze(0)  # [1,H,W]
		dev_t = torch.from_numpy(dev).unsqueeze(0)

		if self.return_paths:
			return clean_t, dev_t, d
		return clean_t, dev_t


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class DeviceInvariantAutoencoder(nn.Module):
	def __init__(self):
		super().__init__()

		# Reuse existing medical-imaging-friendly blocks from models.py.
		# NOTE: models.py imports transformers; keep import here to avoid heavy import
		# when only using the dataset.
		from models import EfficientNetMiniEncoder, Decoder6

		self.encoder = EfficientNetMiniEncoder(inp=1)
		self.decoder = Decoder6()

	def encode(self, x: torch.Tensor) -> torch.Tensor:
		return self.encoder(x)

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		return self.decoder(z)


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
	if not prefix:
		return dict(state_dict)
	out: Dict[str, torch.Tensor] = {}
	for k, v in state_dict.items():
		if k.startswith(prefix):
			out[k[len(prefix) :]] = v
	return out


def load_encoder_init(model: DeviceInvariantAutoencoder, load_path: str) -> None:
	"""Initialize encoder weights similar to other training entrypoints.

	Supports either:
	- A checkpoint dict with key `model_dict` (loads non-strict into the whole model)
	- A raw state_dict (tries to load into `model.encoder`, with common prefix stripping)

	If the path is empty or does not exist, this is a no-op.
	"""
	if not load_path:
		print('[init] No load_path provided; training from scratch.')
		return
	if not os.path.isfile(load_path):
		print(f"[init] load_path not found: {load_path} (skipping init)")
		return

	print(f"[init] Loading init weights from: {load_path}")
	ckpt = torch.load(load_path, map_location='cpu')

	# Full checkpoint
	if isinstance(ckpt, dict) and 'model_dict' in ckpt and isinstance(ckpt['model_dict'], dict):
		strict = False
		missing, unexpected = model.load_state_dict(ckpt['model_dict'], strict=strict)
		print(f"[init] Loaded checkpoint model_dict (strict={strict}). missing={len(missing)} unexpected={len(unexpected)}")
		return

	# Encoder-only or other raw state_dict
	if not isinstance(ckpt, dict):
		raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

	state = ckpt
	# Common prefixes from other models
	for prefix in ('backbone.encoder.', 'encoder.'):
		stripped = _strip_prefix(state, prefix)
		if stripped:
			state = stripped

	try:
		model.encoder.load_state_dict(state, strict=True)
		print('[init] Loaded encoder state_dict (strict=True).')
		return
	except RuntimeError as e:
		print(f"[init] Encoder strict load failed: {e}")

	# Try non-strict as a fallback (useful if state_dict includes extra keys)
	missing, unexpected = model.encoder.load_state_dict(state, strict=False)
	print(f"[init] Loaded encoder state_dict (strict=False). missing={len(missing)} unexpected={len(unexpected)}")


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------


def feature_invariance_loss(z1: torch.Tensor, z2: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
	"""L1 on feature maps, optionally channel-normalized per spatial location."""
	# z: [B,C,H,W] -> [B,C,HW]
	z1f = z1.flatten(2)
	z2f = z2.flatten(2)
	if normalize:
		z1f = F.normalize(z1f, dim=1)
		z2f = F.normalize(z2f, dim=1)
	return F.l1_loss(z1f, z2f)


# -----------------------------------------------------------------------------
# Train config
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
	train_roots: List[str]
	val_roots: Optional[List[str]]
	save_dir: str
	device: str
	seed: int
	load_path: str

	epochs: int
	batch_size: int
	lr: float
	weight_decay: float

	lambda_rec: float
	lambda_inv: float
	lambda_residual: float

	normalize: str
	num_workers: int
	amp: bool
	grad_clip: float
	log_every: int
	save_every: int


def parse_args() -> TrainConfig:
	p = argparse.ArgumentParser(description="Train device-invariant autoencoder on paired DRRs.")
	p.add_argument("--train_roots", nargs="+", required=True)
	p.add_argument("--val_roots", nargs="+", default=None)
	p.add_argument("--save_dir", required=True)
	p.add_argument("--device", default=str(DEFAULT_DEVICE))
	p.add_argument("--seed", type=int, default=123)
	p.add_argument(
		"--load_path",
		type=str,
		default=str(LONGITUDINAL_LOAD_PATH),
		help='Optional init weights. Can be a checkpoint with model_dict or an encoder-only state_dict.',
	)

	p.add_argument("--epochs", type=int, default=20)
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--lr", type=float, default=2e-4)
	p.add_argument("--weight_decay", type=float, default=1e-2)

	p.add_argument("--lambda_rec", type=float, default=1.0)
	p.add_argument("--lambda_inv", type=float, default=0.2)
	p.add_argument("--lambda_residual", type=float, default=0.0)

	p.add_argument("--normalize", choices=["minmax", "zscore", "none"], default="minmax")
	p.add_argument("--num_workers", type=int, default=0)
	p.add_argument("--amp", action="store_true")
	p.add_argument("--grad_clip", type=float, default=1.0)
	p.add_argument("--log_every", type=int, default=50)
	p.add_argument("--save_every", type=int, default=1)

	args = p.parse_args()

	return TrainConfig(
		train_roots=list(args.train_roots),
		val_roots=list(args.val_roots) if args.val_roots else None,
		save_dir=str(args.save_dir),
		device=str(args.device),
		seed=int(args.seed),
		load_path=str(args.load_path),
		epochs=int(args.epochs),
		batch_size=int(args.batch_size),
		lr=float(args.lr),
		weight_decay=float(args.weight_decay),
		lambda_rec=float(args.lambda_rec),
		lambda_inv=float(args.lambda_inv),
		lambda_residual=float(args.lambda_residual),
		normalize=str(args.normalize),
		num_workers=int(args.num_workers),
		amp=bool(args.amp),
		grad_clip=float(args.grad_clip),
		log_every=int(args.log_every),
		save_every=int(args.save_every),
	)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_eval(
	model: DeviceInvariantAutoencoder,
	dl: DataLoader,
	cfg: TrainConfig,
) -> Dict[str, float]:
	model.eval()
	device = torch.device(cfg.device)

	loss_sum: Dict[str, float] = {"loss": 0.0, "rec": 0.0, "inv": 0.0, "residual": 0.0}
	count = 0

	for batch in dl:
		if len(batch) == 3:
			clean, dev, _path = batch
		else:
			clean, dev = batch

		clean = clean.to(device)
		dev = dev.to(device)

		z_c = model.encode(clean)
		z_d = model.encode(dev)

		pred_from_clean = model.decode(z_c)
		pred_from_dev = model.decode(z_d)

		# Both should reconstruct the clean target
		l_rec = F.l1_loss(pred_from_clean, clean) + F.l1_loss(pred_from_dev, clean)
		l_inv = feature_invariance_loss(z_c, z_d, normalize=True)

		l_res = torch.tensor(0.0, device=device)
		if cfg.lambda_residual > 0:
			gt_res = dev - clean
			pred_res = pred_from_dev - pred_from_clean
			l_res = F.l1_loss(pred_res, gt_res)

		loss = cfg.lambda_rec * l_rec + cfg.lambda_inv * l_inv + cfg.lambda_residual * l_res

		loss_sum["loss"] += float(loss.item())
		loss_sum["rec"] += float(l_rec.item())
		loss_sum["inv"] += float(l_inv.item())
		loss_sum["residual"] += float(l_res.item())
		count += 1

	if count == 0:
		return {k: 0.0 for k in loss_sum}
	return {k: v / count for k, v in loss_sum.items()}


def main() -> None:
	cfg = parse_args()

	os.makedirs(cfg.save_dir, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = os.path.join(cfg.save_dir, stamp)
	os.makedirs(run_dir, exist_ok=True)

	# Repro
	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	device = torch.device(cfg.device)
	use_amp = bool(cfg.amp) and device.type == "cuda"
	scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	with open(os.path.join(run_dir, "train_config.json"), "w", encoding="utf-8") as f:
		json.dump(asdict(cfg), f, indent=2)

	train_ds = PairedDevicesDRRDataset(cfg.train_roots, normalize=cfg.normalize)
	val_ds = PairedDevicesDRRDataset(cfg.val_roots or cfg.train_roots, normalize=cfg.normalize)

	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=(device.type == "cuda"),
		drop_last=True,
	)
	val_dl = DataLoader(
		val_ds,
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=(device.type == "cuda"),
		drop_last=False,
	)

	model = DeviceInvariantAutoencoder().to(device)
	load_encoder_init(model, cfg.load_path)
	opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

	history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_rec": [], "val_rec": [], "train_inv": [], "val_inv": []}

	print(f"Run dir: {run_dir}")
	print(f"Device: {device} | AMP: {use_amp}")
	print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

	global_step = 0
	for epoch in range(1, cfg.epochs + 1):
		model.train()

		running = {"loss": 0.0, "rec": 0.0, "inv": 0.0}
		steps = 0

		for clean, dev in train_dl:
			clean = clean.to(device)
			dev = dev.to(device)

			opt.zero_grad(set_to_none=True)

			with torch.cuda.amp.autocast(enabled=use_amp):
				z_c = model.encode(clean)
				z_d = model.encode(dev)

				pred_from_clean = model.decode(z_c)
				pred_from_dev = model.decode(z_d)

				# Both should reconstruct the clean target
				l_rec = F.l1_loss(pred_from_clean, clean) + F.l1_loss(pred_from_dev, clean)
				l_inv = feature_invariance_loss(z_c, z_d, normalize=True)

				l_res = torch.tensor(0.0, device=device)
				if cfg.lambda_residual > 0:
					gt_res = dev - clean
					pred_res = pred_from_dev - pred_from_clean
					l_res = F.l1_loss(pred_res, gt_res)

				loss = cfg.lambda_rec * l_rec + cfg.lambda_inv * l_inv + cfg.lambda_residual * l_res

			scaler.scale(loss).backward()
			if cfg.grad_clip and cfg.grad_clip > 0:
				scaler.unscale_(opt)
				torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
			scaler.step(opt)
			scaler.update()

			running["loss"] += float(loss.item())
			running["rec"] += float(l_rec.item())
			running["inv"] += float(l_inv.item())
			steps += 1
			global_step += 1

			if cfg.log_every > 0 and (global_step % cfg.log_every == 0):
				print(
					f"epoch {epoch}/{cfg.epochs} step {global_step} | "
					f"loss={running['loss']/steps:.4f} rec={running['rec']/steps:.4f} inv={running['inv']/steps:.4f}"
				)

		train_metrics = {k: v / max(1, steps) for k, v in running.items()}
		val_metrics = run_eval(model, val_dl, cfg)

		history["train_loss"].append(train_metrics["loss"])
		history["train_rec"].append(train_metrics["rec"])
		history["train_inv"].append(train_metrics["inv"])
		history["val_loss"].append(val_metrics["loss"])
		history["val_rec"].append(val_metrics["rec"])
		history["val_inv"].append(val_metrics["inv"])

		print(
			f"[epoch {epoch}] train loss={train_metrics['loss']:.4f} (rec={train_metrics['rec']:.4f}, inv={train_metrics['inv']:.4f}) | "
			f"val loss={val_metrics['loss']:.4f} (rec={val_metrics['rec']:.4f}, inv={val_metrics['inv']:.4f})"
		)

		with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
			json.dump(history, f, indent=2)

		if cfg.save_every > 0 and (epoch % cfg.save_every == 0):
			ckpt = {
				"epoch": epoch,
				"global_step": global_step,
				"model": model.state_dict(),
				"optimizer": opt.state_dict(),
				"config": asdict(cfg),
			}
			torch.save(ckpt, os.path.join(run_dir, f"ckpt_epoch_{epoch:03d}.pt"))

	print("Done.")


if __name__ == "__main__":
	main()
