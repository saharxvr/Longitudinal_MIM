from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from constants import DEVICE

from .datasets import TripletItem
from .model import LongitudinalMIMDeformV2, _device_type


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if str(device).startswith("cuda"):
        return x.to(device=device, dtype=torch.float16)
    return x.to(device=device, dtype=torch.float32)


def _minmax_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B,1,H,W]
    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def flow_smoothness_l2(flow: torch.Tensor) -> torch.Tensor:
    """Simple TV/L2 smoothness on flow field.

    flow: [B,2,H,W]
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())


@dataclass
class StageConfig:
    stage: str  # 'A' | 'B' | 'C'
    epochs: int
    lr: float
    weight_decay: float = 0.0
    lambda_smooth: float = 0.05
    lambda_recon: float = 0.0


class TrainerV2:
    """Staged trainer for LongitudinalMIMDeformV2.

    Stages:
      A) Diff supervision (pair-style): train model(prior, current)->diff_full
      B) Deformation supervision (triplet): train deform head to warp prior->intermediate
      C) Entity-only supervision (triplet): train model(prior, current)->diff_entity

    Notes:
      - Keeps your existing input contract: model(bls, fus)
      - Uses AMP only on CUDA.
    """

    def __init__(self, model: LongitudinalMIMDeformV2, *, device: Optional[str] = None):
        self.model = model
        self.device = torch.device(device or DEVICE)
        self.model.to(self.device)

        self.l1 = nn.L1Loss()

        self.scaler = torch.cuda.amp.GradScaler(enabled=str(self.device).startswith("cuda"))

    def _set_trainable(self, *, encoder: bool, diff_head: bool, deform_head: bool, recon_head: bool) -> None:
        # Backbone structure comes from LongitudinalMIMModelBig
        for p in self.model.backbone.encoder.parameters():
            p.requires_grad = encoder
        # Everything except encoder is considered "diff head" for this trainer.
        for name, module in self.model.backbone.named_children():
            if name == "encoder":
                continue
            for p in module.parameters():
                p.requires_grad = diff_head

        if self.model.deform_head is not None:
            for p in self.model.deform_head.parameters():
                p.requires_grad = deform_head

        if self.model.recon_head is not None:
            for p in self.model.recon_head.parameters():
                p.requires_grad = recon_head

    def _optimizer(self, cfg: StageConfig) -> optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train_stage(self, cfg: StageConfig, dataloader: DataLoader) -> None:
        if cfg.stage not in {"A", "B", "C", "E"}:
            raise ValueError(f"Unknown stage: {cfg.stage}")

        if cfg.stage == "A":
            # train encoder + diff head; deformation head off
            self._set_trainable(encoder=True, diff_head=True, deform_head=False, recon_head=self.model.enable_reconstruction)
        elif cfg.stage == "B":
            # freeze encoder, train deformation head only
            self._set_trainable(encoder=False, diff_head=False, deform_head=True, recon_head=False)
        elif cfg.stage == "E":
            # encoder pretraining via reconstruction only
            self._set_trainable(encoder=True, diff_head=False, deform_head=False, recon_head=True)
        else:
            # Stage C: train encoder+diff head; deformation head optional but typically frozen
            self._set_trainable(encoder=True, diff_head=True, deform_head=False, recon_head=self.model.enable_reconstruction)

        optimizer = self._optimizer(cfg)

        dev_type = _device_type(self.device)

        self.model.train()
        for epoch in range(1, cfg.epochs + 1):
            running = 0.0
            for batch in dataloader:
                # -------------------------
                # Stage E: image-only recon
                # -------------------------
                if cfg.stage == "E":
                    if not self.model.enable_reconstruction:
                        raise RuntimeError("Stage E requires enable_reconstruction=True")

                    if not isinstance(batch, torch.Tensor):
                        # Try common tuple/dict patterns
                        if isinstance(batch, (tuple, list)):
                            img = batch[0]
                        elif isinstance(batch, dict):
                            img = batch.get("image") or next(iter(batch.values()))
                        else:
                            img = batch
                    else:
                        img = batch

                    img = _to_device(img, self.device)
                    img_n = _minmax_norm(img)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=dev_type == "cuda"):
                        recon = self.model.reconstruct(img_n)
                        loss = self.l1(recon, img_n)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()

                    running += float(loss.detach().cpu().item())
                    continue

                # TripletDRRDataset returns a TripletItem; DataLoader will collate dataclasses into a dict-like.
                if isinstance(batch, dict):
                    prior = batch["prior"]
                    intermediate = batch["intermediate"]
                    current = batch["current"]
                    diff_entity = batch["diff_entity"]
                    diff_full = batch["diff_full"]
                    mask = batch["mask"]
                elif isinstance(batch, TripletItem):
                    prior, intermediate, current = batch.prior, batch.intermediate, batch.current
                    diff_entity, diff_full, mask = batch.diff_entity, batch.diff_full, batch.mask
                else:
                    # Fallback: tuple
                    prior, intermediate, current, diff_entity, diff_full, mask = batch

                prior = _to_device(prior, self.device)
                intermediate = _to_device(intermediate, self.device)
                current = _to_device(current, self.device)
                diff_entity = _to_device(diff_entity, self.device)
                diff_full = _to_device(diff_full, self.device)
                mask = _to_device(mask, self.device)

                # Normalize inputs similarly to legacy training
                prior_n = _minmax_norm(prior)
                intermediate_n = _minmax_norm(intermediate)
                current_n = _minmax_norm(current)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=dev_type == "cuda"):
                    if cfg.stage == "A":
                        pred = self.model(prior_n, current_n)
                        loss = self.l1(pred * mask, diff_full * mask)
                        if cfg.lambda_recon > 0:
                            recon_p = self.model.reconstruct(prior_n)
                            recon_c = self.model.reconstruct(current_n)
                            loss = loss + cfg.lambda_recon * (
                                self.l1(recon_p * mask, prior_n * mask) + self.l1(recon_c * mask, current_n * mask)
                            )
                    elif cfg.stage == "B":
                        deform = self.model.predict_deformation(prior_n, intermediate_n)
                        photo = self.l1(deform.prior_warped * mask, intermediate_n * mask)
                        smooth = flow_smoothness_l2(deform.flow_img)
                        loss = photo + cfg.lambda_smooth * smooth
                    else:
                        pred = self.model(prior_n, current_n)
                        loss = self.l1(pred * mask, diff_entity * mask)
                        if cfg.lambda_recon > 0:
                            recon_p = self.model.reconstruct(prior_n)
                            recon_c = self.model.reconstruct(current_n)
                            loss = loss + cfg.lambda_recon * (
                                self.l1(recon_p * mask, prior_n * mask) + self.l1(recon_c * mask, current_n * mask)
                            )

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running += float(loss.detach().cpu().item())

            avg = running / max(1, len(dataloader))
            print(f"[Stage {cfg.stage}] epoch {epoch}/{cfg.epochs} | loss={avg:.6f}")
