"""
Training entry point for foundation-based contrastive difference detection.

Pipeline:
    frozen backbone (shared) -> difference head -> {difference map, embedding}
                                              -> projection heads {z_path, z_nuis}

Loss = lambda_seg * L_seg(diff_map, gt_diff)
     + lambda_contrastive * SupCon(z_path, anomaly_type)
     + lambda_orthogonality * orth(z_path, z_nuis)
     + lambda_direction * L_dir   (optional)

Run:
    python -m training.train_foundation_diff --config configs/default.yaml

This is a Phase-1 scaffold: the training loop structure is in place; backbone, dataset
loading, and segmentation loss need to be completed (see RESEARCH_PLAN.md).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make sibling packages importable when run as a script.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from models import FrozenCXRBackbone, DifferenceHead, DisentangledProjectionHeads
from losses import supervised_contrastive_loss, orthogonality_loss
from datasets import LongitudinalPairDataset


def build_model():
    backbone = FrozenCXRBackbone(
        name=C.BACKBONE,
        checkpoint=C.CHEXFOUND_CHECKPOINT,
        config=C.CHEXFOUND_CONFIG,
        last_n_layers=C.LAST_N_LAYERS,
        freeze=C.FREEZE_BACKBONE,
    ).to(C.DEVICE)
    head = DifferenceHead(
        backbone_dim=C.BACKBONE_DIM,
        d_glori=C.D_GLORI,
        num_change_queries=C.NUM_CHANGE_QUERIES,
        num_heads=C.GLORI_NUM_HEADS,
        grid=C.FEATURE_GRID,
        embed_dim=C.EMBED_DIM,
        fusion_mode=C.FUSION_MODE,
        use_adaptive_temperature=C.USE_ADAPTIVE_TEMPERATURE,
        use_pyramid_patch_merging=C.USE_PYRAMID_PATCH_MERGING,
        integrate_global_cls=C.INTEGRATE_GLOBAL_CLS,
        out_range=C.DECODER_OUT_RANGE,
    ).to(C.DEVICE)
    proj = DisentangledProjectionHeads(
        embed_dim=C.EMBED_DIM,
        proj_dim=C.PROJ_DIM,
        use_disentanglement=C.USE_DISENTANGLEMENT,
    ).to(C.DEVICE)
    return backbone, head, proj


def segmentation_loss(pred_diff: torch.Tensor, gt_diff: torch.Tensor) -> torch.Tensor:
    """L1 on the signed change map. Extend with SSIM / perceptual (reuse parent losses)."""
    return F.l1_loss(pred_diff, gt_diff)


def train_one_epoch(backbone, head, proj, loader, optimizer):
    head.train()
    proj.train()
    running = 0.0

    for step, batch in enumerate(loader):
        img_prior = batch["img_prior"].to(C.DEVICE)
        img_curr = batch["img_curr"].to(C.DEVICE)
        gt_diff = batch["gt_diff"].to(C.DEVICE)
        anomaly_type = batch["anomaly_type"].to(C.DEVICE)

        # Frozen CheXFound -> patch tokens + [CLS]. With USE_FEATURE_CACHE these are
        # precomputed (see RQ1_PLAN.md section 3) and the backbone call is skipped.
        f_prior = backbone(img_prior)
        f_curr = backbone(img_curr)

        pred_diff, z = head(
            f_prior["patch_tokens"], f_curr["patch_tokens"],
            f_prior["cls_token"], f_curr["cls_token"],
        )
        z_path, z_nuis = proj(z)

        loss = C.LAMBDA_SEG * segmentation_loss(pred_diff, gt_diff)
        if C.USE_CONTRASTIVE:
            loss = loss + C.LAMBDA_CONTRASTIVE * supervised_contrastive_loss(
                z_path, anomaly_type, temperature=C.SUPCON_TEMPERATURE
            )
        if C.USE_DISENTANGLEMENT:
            loss = loss + C.LAMBDA_ORTHOGONALITY * orthogonality_loss(z_path, z_nuis)

        loss = loss / C.UPDATE_EVERY_BATCHES
        loss.backward()
        if (step + 1) % C.UPDATE_EVERY_BATCHES == 0:
            optimizer.step()
            optimizer.zero_grad()

        running += loss.item() * C.UPDATE_EVERY_BATCHES

    return running / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    # TODO: load YAML and override constants (kept minimal for the scaffold).

    backbone, head, proj = build_model()

    train_ds = LongitudinalPairDataset(C.TRAIN_PAIRS_DIR, img_size=C.IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True, num_workers=4)

    trainable = list(head.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=C.MAX_LR, weight_decay=C.WEIGHT_DECAY)

    n_trainable = sum(p.numel() for p in trainable if p.requires_grad)
    print(f"Trainable parameters (head + proj): {n_trainable:,}")

    os.makedirs(C.SAVE_FOLDER, exist_ok=True)
    for epoch in range(C.EPOCHS):
        loss = train_one_epoch(backbone, head, proj, train_loader, optimizer)
        print(f"Epoch {epoch + 1}/{C.EPOCHS} - loss: {loss:.4f}")
        torch.save(
            {"head": head.state_dict(), "proj": proj.state_dict(), "epoch": epoch},
            os.path.join(C.SAVE_FOLDER, f"ckpt_epoch{epoch + 1}.pt"),
        )


if __name__ == "__main__":
    main()
