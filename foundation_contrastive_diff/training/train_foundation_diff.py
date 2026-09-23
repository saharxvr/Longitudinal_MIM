"""
RQ1 training — frozen RAD-DINO features -> D-GLoRI head -> signed change map.

Only the D-GLoRI head is trained; the backbone is frozen. By default we train on
**precomputed features** (training/cache_features.py) so each epoch is a cheap head-only
pass; use --no_cache to run the backbone on the fly from NIfTIs instead.

Loss: losses.change_map_loss (weighted-L1 + soft-Dice + sign penalty) on the sparse,
signed GT. Optimizer AdamW + cosine schedule with linear warmup. We checkpoint the best
model by validation Dice and dump a few GT-vs-pred heatmaps per validation.

Run (cached path):
    python -m training.train_foundation_diff --dataset_root /path/to/fcd_train \
        --cache_dir ./feature_cache --save_folder ./checkpoints --epochs 50
"""

import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from models import DifferenceHead
from losses import change_map_loss
from datasets import LongitudinalPairDataset, CachedPairDataset
from evaluation.change_detection_metrics import dice_score, iou_score, directional_sensitivity


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_head(device):
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
        out_size=C.IMG_SIZE,
        out_range=C.DECODER_OUT_RANGE,
    ).to(device)
    return head


def _tokens_from_batch(batch, backbone, device):
    """Return (p_prior, p_curr, cls_prior, cls_curr) from cached feats or the backbone."""
    if backbone is None:  # cached-feature path
        return (
            batch["p_prior"].to(device), batch["p_curr"].to(device),
            batch["cls_prior"].to(device), batch["cls_curr"].to(device),
        )
    fp = backbone(batch["img_prior"].to(device))
    fc = backbone(batch["img_curr"].to(device))
    return fp["patch_tokens"], fc["patch_tokens"], fp["cls_token"], fc["cls_token"]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def cosine_warmup(optimizer, warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_one_epoch(head, backbone, loader, optimizer, scheduler, device, args):
    head.train()
    totals = {"total": 0.0, "l1": 0.0, "dice": 0.0, "sign": 0.0}
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        p_prior, p_curr, cls_prior, cls_curr = _tokens_from_batch(batch, backbone, device)
        gt = batch["gt_diff"].to(device)

        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)
        loss, comp = change_map_loss(
            pred, gt, tau=args.tau, w_l1=args.w_l1, w_dice=args.w_dice,
            w_sign=args.w_sign, pos_weight=args.pos_weight,
        )
        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        for k in totals:
            totals[k] += comp[k]
    n = max(1, len(loader))
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(head, backbone, loader, device, threshold=0.1):
    head.eval()
    dices, ious, spos, sneg = [], [], [], []
    for batch in loader:
        p_prior, p_curr, cls_prior, cls_curr = _tokens_from_batch(batch, backbone, device)
        gt = batch["gt_diff"].to(device)
        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)

        pm = (pred.abs() > threshold).float()
        gm = (gt.abs() > threshold).float()
        for b in range(pred.shape[0]):
            dices.append(dice_score(pm[b], gm[b]))
            ious.append(iou_score(pm[b], gm[b]))
        d = directional_sensitivity(pred, gt, threshold)
        spos.append(d["sensitivity_positive"])
        sneg.append(d["sensitivity_negative"])

    def mean(x):
        return sum(x) / max(1, len(x))
    return {"dice": mean(dices), "iou": mean(ious),
            "sens_pos": mean(spos), "sens_neg": mean(sneg)}


@torch.no_grad()
def save_sample_plots(head, backbone, loader, device, out_path, n=6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    head.eval()
    batch = next(iter(loader))
    p_prior, p_curr, cls_prior, cls_curr = _tokens_from_batch(batch, backbone, device)
    pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)
    gt = batch["gt_diff"]
    n = min(n, pred.shape[0])
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    ax = ax.reshape(2, n)
    for i in range(n):
        ax[0, i].imshow(gt[i, 0].cpu(), cmap="bwr", vmin=-1, vmax=1)
        ax[0, i].set_title("GT"); ax[0, i].axis("off")
        ax[1, i].imshow(pred[i, 0].cpu(), cmap="bwr", vmin=-1, vmax=1)
        ax[1, i].set_title("pred"); ax[1, i].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_loaders(args, device):
    if args.no_cache:
        from models import FrozenCXRBackbone
        backbone = FrozenCXRBackbone(
            name=C.BACKBONE, model_id=C.RAD_DINO_MODEL,
            backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
            last_n_layers=C.LAST_N_LAYERS, freeze=C.FREEZE_BACKBONE,
        ).to(device)
        backbone.eval()
        train_ds = LongitudinalPairDataset(args.dataset_root, split="train", img_size=C.IMG_SIZE)
        val_ds = LongitudinalPairDataset(args.dataset_root, split="val", img_size=C.IMG_SIZE)
        labels = None
    else:
        backbone = None
        train_ds = CachedPairDataset(args.dataset_root, args.cache_dir, split="train")
        val_ds = CachedPairDataset(args.dataset_root, args.cache_dir, split="val")
        labels = train_ds.labels

    sampler, shuffle = None, True
    if args.balanced and labels is not None:
        counts = torch.bincount(torch.tensor(labels), minlength=len(C.ANOMALY_TYPES)).float()
        inv = 1.0 / counts.clamp(min=1)
        weights = inv[torch.tensor(labels)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    return backbone, train_loader, val_loader


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="RQ1: train D-GLoRI change-map head on frozen features")
    p.add_argument("-o", "--dataset_root", required=True, help="folder with manifest[_split].jsonl")
    p.add_argument("--cache_dir", default=C.FEATURE_CACHE_DIR)
    p.add_argument("--save_folder", default=C.SAVE_FOLDER)
    p.add_argument("--plots_folder", default=C.PLOTS_FOLDER)
    p.add_argument("--no_cache", action="store_true", help="run backbone on the fly (no feature cache)")
    p.add_argument("--balanced", action="store_true", help="class-balanced sampler over anomaly type")
    p.add_argument("--epochs", type=int, default=C.EPOCHS)
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--accum", type=int, default=C.UPDATE_EVERY_BATCHES)
    p.add_argument("--lr", type=float, default=C.MAX_LR)
    p.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    p.add_argument("--warmup_epochs", type=float, default=2.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    # loss hyperparameters
    p.add_argument("--tau", type=float, default=0.02, help="|GT| threshold for change pixels")
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_dice", type=float, default=1.0)
    p.add_argument("--w_sign", type=float, default=0.5)
    p.add_argument("--pos_weight", type=float, default=10.0)
    p.add_argument("--eval_threshold", type=float, default=0.1, help="|map| threshold for Dice/IoU")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)

    backbone, train_loader, val_loader = make_loaders(args, device)
    head = build_head(device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[RQ1] trainable head params: {n_trainable/1e6:.2f}M | "
          f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = cosine_warmup(optimizer, warmup_steps, total_steps)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(head, backbone, train_loader, optimizer, scheduler, device, args)
        va = evaluate(head, backbone, val_loader, device, threshold=args.eval_threshold)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d}/{args.epochs} | lr {lr_now:.2e} | "
              f"train total {tr['total']:.4f} (l1 {tr['l1']:.4f} dice {tr['dice']:.4f} sign {tr['sign']:.4f}) | "
              f"val dice {va['dice']:.4f} iou {va['iou']:.4f} "
              f"sens+ {va['sens_pos']:.3f} sens- {va['sens_neg']:.3f}",
              flush=True)

        torch.save({"head": head.state_dict(), "epoch": epoch, "val": va},
                   os.path.join(args.save_folder, "last.pt"))
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save({"head": head.state_dict(), "epoch": epoch, "val": va},
                       os.path.join(args.save_folder, "best.pt"))
            save_sample_plots(head, backbone, val_loader, device,
                              os.path.join(args.plots_folder, "val_best_samples.png"))

    print(f"[RQ1] done. best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
