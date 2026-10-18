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
import csv
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from models import build_change_head
from losses import change_map_loss
from datasets import LongitudinalPairDataset, CachedPairDataset
from evaluation.change_detection_metrics import dice_score, iou_score, directional_sensitivity
from evaluation.training_diagnostics import PredictionDiagnostics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Optional: slice cached features to the last K backbone layers (last K*768 dims).
# The last-4 cache stores layers concatenated oldest->newest, so the final 768 dims
# are the last layer. This lets last-1 / last-2 reuse the last-4 cache (no re-caching).
_FEAT_DIM = None  # set in main() from --feat_last_k


def head_spec(patch_dim=None, head_type="dglori", dpt_channels=96):
    common = dict(
        backbone_dim=C.BACKBONE_DIM,
        patch_dim=patch_dim or (C.BACKBONE_DIM * C.LAST_N_LAYERS),
        cls_dim=C.BACKBONE_DIM,
        grid=C.FEATURE_GRID,
        embed_dim=C.EMBED_DIM,
        out_size=C.IMG_SIZE,
        out_range=C.DECODER_OUT_RANGE,
    )
    if head_type in ("dpt", "dpt_cross_attention"):
        kwargs = dict(common, channels=dpt_channels)
        if head_type == "dpt_cross_attention":
            kwargs.update(num_heads=4, attention_depth=2)
            if common["patch_dim"] == C.BACKBONE_DIM:
                kwargs["num_layers"] = 1
        return {"type": head_type, "kwargs": kwargs}
    return {"type": head_type, "kwargs": dict(common,
        d_glori=C.D_GLORI,
        num_change_queries=C.NUM_CHANGE_QUERIES,
        num_heads=C.GLORI_NUM_HEADS,
        fusion_mode=C.FUSION_MODE,
        use_adaptive_temperature=C.USE_ADAPTIVE_TEMPERATURE,
        use_pyramid_patch_merging=C.USE_PYRAMID_PATCH_MERGING,
        integrate_global_cls=C.INTEGRATE_GLOBAL_CLS,
    )}


def build_head(device, patch_dim=None, head_type="dglori", dpt_channels=96):
    return build_change_head(head_spec(patch_dim, head_type, dpt_channels), device)


def _tokens_from_batch(batch, backbone, device):
    """Return (p_prior, p_curr, cls_prior, cls_curr) from cached feats or the backbone."""
    if backbone is None:  # cached-feature path
        pp, pc = batch["p_prior"].to(device), batch["p_curr"].to(device)
        if _FEAT_DIM is not None:                 # slice to last-K layers
            pp, pc = pp[..., -_FEAT_DIM:], pc[..., -_FEAT_DIM:]
        return pp, pc, batch["cls_prior"].to(device), batch["cls_curr"].to(device)
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
def train_one_epoch(head, backbone, loader, optimizer, scheduler, device, args, w_dice=None,
                    diagnostics=None):
    head.train()
    w_dice = args.w_dice if w_dice is None else w_dice
    totals = {"total": 0.0, "l1": 0.0, "l2": 0.0, "dice": 0.0, "sign": 0.0}
    n_ok, n_bad = 0, 0
    started = time.perf_counter()
    gradient_norms = []
    optimizer_steps, nonfinite_gradients, pending_batches = 0, 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        p_prior, p_curr, cls_prior, cls_curr = _tokens_from_batch(batch, backbone, device)
        gt = batch["gt_diff"].to(device)

        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)
        loss, comp = change_map_loss(
            pred, gt, tau=args.tau, w_l1=args.w_l1, w_dice=w_dice,
            w_sign=args.w_sign, w_l2=args.w_l2, pos_weight=args.pos_weight,
            target_mode=args.target_mode, target_threshold=args.target_threshold,
            background_weight=args.background_weight,
        )
        # NaN guard: skip corrupt/exploding batches instead of poisoning the whole run.
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            n_bad += 1
            pending_batches = 0
            continue
        if diagnostics is not None:
            diagnostics.update(pred, gt, batch)
        (loss / args.accum).backward()
        pending_batches += 1
        if (step + 1) % args.accum == 0:
            if args.grad_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            else:
                norms = [parameter.grad.detach().norm() for parameter in head.parameters()
                         if parameter.grad is not None]
                norm = torch.stack(norms).norm() if norms else torch.tensor(0.)
            if torch.isfinite(norm):
                gradient_norms.append(float(norm))
            else:
                nonfinite_gradients += 1
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            optimizer_steps += 1
            pending_batches = 0

        n_ok += 1
        for k in totals:
            totals[k] += comp[k]
    if n_bad:
        print(f"  [warn] skipped {n_bad} non-finite batches this epoch", flush=True)
    n = max(1, n_ok)
    return {**{k: v / n for k, v in totals.items()},
            "health": dict(successful_batches=n_ok, nonfinite_loss_batches=n_bad,
                           optimizer_steps=optimizer_steps, pending_backward_batches=pending_batches,
                           nonfinite_gradient_steps=nonfinite_gradients,
                           grad_norm_mean=sum(gradient_norms) / len(gradient_norms) if gradient_norms else None,
                           grad_norm_max=max(gradient_norms, default=None),
                           seconds=time.perf_counter() - started)}


@torch.no_grad()
def evaluate(head, backbone, loader, device, threshold=0.1, tau=0.02, pos_weight=10.0,
             gt_threshold=None, target_mode="continuous", target_threshold=0.05,
             background_weight=1.0, diagnostics=None, w_l1=1.0, w_l2=0.0,
             w_dice=1.0, w_sign=0.5):
    head.eval()
    gt_threshold = threshold if gt_threshold is None else gt_threshold
    dices, ious, spos, sneg, nuis_fp = [], [], [], [], []
    l1s, l2s, l1ms, l2ms = [], [], [], []
    loss_components = {key: [] for key in ("total", "dice", "sign")}
    for batch in loader:
        p_prior, p_curr, cls_prior, cls_curr = _tokens_from_batch(batch, backbone, device)
        gt = batch["gt_diff"].to(device)
        is_path = batch["is_pathology"]
        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)
        if diagnostics is not None:
            diagnostics.update(pred, gt, batch)

        pm = (pred.abs() > threshold).float()
        gm = (gt.abs() > gt_threshold).float()
        _, comp = change_map_loss(pred, gt, tau=tau, pos_weight=pos_weight,
                                  target_mode=target_mode, target_threshold=target_threshold,
                                  background_weight=background_weight, w_l1=w_l1, w_l2=w_l2,
                                  w_dice=w_dice, w_sign=w_sign)
        l1s.append(comp["l1"]); l2s.append(comp["l2"])
        for key in loss_components:
            loss_components[key].append(comp[key])
        target = (torch.where(gt.abs() > target_threshold, gt.sign(), torch.zeros_like(gt))
                  if target_mode == "signed_mask" else gt)
        m = target.abs() > tau
        if m.any():
            diff = pred - target
            l1ms.append(float(diff.abs()[m].mean())); l2ms.append(float((diff[m] ** 2).mean()))
        for b in range(pred.shape[0]):
            # Split change vs no-change: empty-GT pairs score Dice=1.0 and would inflate
            # the mean, so Dice/IoU/sens are change-only; no-change -> false-positive area.
            if int(is_path[b]) == 0:
                nuis_fp.append(float(pm[b].mean()))
                continue
            dices.append(dice_score(pm[b], gm[b]))
            ious.append(iou_score(pm[b], gm[b]))
            d = directional_sensitivity(pred[b:b + 1], gt[b:b + 1], threshold, gt_threshold)
            # Only count a direction's recall on pairs that actually contain that direction,
            # else pure-appearance/pure-resolution pairs inject spurious 0s and deflate sens.
            if (gt[b] > gt_threshold).any():
                spos.append(d["sensitivity_positive"])
            if (gt[b] < -gt_threshold).any():
                sneg.append(d["sensitivity_negative"])

    def mean(x):
        return sum(x) / max(1, len(x))
    return {"dice": mean(dices), "iou": mean(ious),
            "sens_pos": mean(spos), "sens_neg": mean(sneg), "nuis_fp": mean(nuis_fp),
            "nuis_fp95": float(torch.quantile(torch.tensor(nuis_fp), 0.95)) if nuis_fp else 0.0,
            "nuis_fp_max": max(nuis_fp, default=0.0),
            "l1": mean(l1s), "l2": mean(l2s),
            "l1_ch": mean(l1ms), "l2_ch": mean(l2ms),
            **{f"loss_{key}": mean(values) for key, values in loss_components.items()}}


def print_prediction_metrics(diagnostics, split, target_threshold=None):
    label = "train online sampled" if split == "train" else "val fixed head"
    if diagnostics is None:
        print(f"[{label}] overlap diagnostics not collected this epoch", flush=True)
        return
    summary = diagnostics.summary()
    sweeps = {(row["group"], row["gt_threshold"], row["sign"]): row
              for row in diagnostics.sweep_rows()
              if row["prediction_threshold"] == diagnostics.primary_prediction}
    cutoffs = [diagnostics.primary_gt]
    if target_threshold is not None and target_threshold not in cutoffs:
        cutoffs.append(target_threshold)

    def display(value):
        return "n/a" if value is None else f"{value:.4f}"

    nuisance = summary["groups"]["nuisance"]
    nuisance_area = nuisance.get("predicted_area", {}).get("mean")
    for cutoff in cutoffs:
        support = sweeps.get(("pathology", cutoff, "support"), {})
        signed = sweeps.get(("pathology", cutoff, "signed"), {})
        print(f"[{label}] pred_abs>{diagnostics.primary_prediction:g} raw_gt_abs>{cutoff:g} | "
              f"pathology_pairs={support.get('pairs', 0):g} "
              f"active_gt_pairs={support.get('active_reference_pairs', 0):g} | "
              f"macro_active_gt support_dice={display(support.get('macro_dice'))} "
              f"support_iou={display(support.get('macro_iou'))} "
              f"signed_dice={display(signed.get('macro_dice'))} | "
              f"micro_signed precision={display(signed.get('micro_precision'))} "
              f"recall={display(signed.get('micro_recall'))} | "
              f"nuisance_pairs={nuisance['pairs']} fp_area_mean={display(nuisance_area)} "
              f"fp_area_p95={display(nuisance.get('predicted_area_p95'))} "
              f"fp_area_max={display(nuisance.get('predicted_area_max'))} "
              f"invalid_pairs={summary['invalid_pairs']}", flush=True)
    if diagnostics.components:
        pathology = summary["groups"]["pathology"]
        print(f"[{label}] component_detection raw_gt_abs>{diagnostics.primary_gt:g} "
              f"pooled_signed_F1={display(pathology.get('component_all_pooled_f1'))} "
              f"invalid_pairs={summary['invalid_pairs']}", flush=True)


@torch.no_grad()
def save_sample_plots(head, backbone, loader, device, out_path, n=6, target_threshold=None):
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
    if target_threshold is not None:
        gt = torch.where(gt.abs() > target_threshold, gt.sign(), torch.zeros_like(gt))
    # Prefer change pairs so the panel isn't dominated by empty (no-change) GT.
    is_path = batch.get("is_pathology")
    if is_path is not None:
        idx = [i for i in range(pred.shape[0]) if int(is_path[i]) == 1]
        idx = (idx + [i for i in range(pred.shape[0]) if int(is_path[i]) == 0])[:n]
    else:
        idx = list(range(min(n, pred.shape[0])))
    n = len(idx)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    ax = ax.reshape(2, n)
    for col, i in enumerate(idx):
        ax[0, col].imshow(gt[i, 0].cpu(), cmap="bwr", vmin=-1, vmax=1)
        ax[0, col].set_title("GT"); ax[0, col].axis("off")
        ax[1, col].imshow(pred[i, 0].cpu(), cmap="bwr", vmin=-1, vmax=1)
        ax[1, col].set_title("pred"); ax[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mid-training tracking
# ---------------------------------------------------------------------------
def log_metrics_row(csv_path, row):
    """Append one epoch's metrics to a CSV (writing the header on first call)."""
    new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def plot_curves(csv_path, out_path):
    """Redraw training curves from metrics.csv (loss + val Dice/IoU vs epoch)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    epochs, tl, vd, vi = [], [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"])); tl.append(float(r["train_total"]))
            vd.append(float(r["val_dice"])); vi.append(float(r["val_iou"]))
    if not epochs:
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, tl, "tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, vd, "tab:blue", label="val Dice")
    ax2.plot(epochs, vi, "tab:cyan", label="val IoU")
    ax2.set_ylabel("val Dice / IoU", color="tab:blue"); ax2.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def make_loaders(args, device):
    if args.no_cache:
        from models import FrozenCXRBackbone
        backbone = FrozenCXRBackbone(
            name=C.BACKBONE, model_id=C.RAD_DINO_MODEL,
            backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
            last_n_layers=args.feat_last_k or C.LAST_N_LAYERS, freeze=C.FREEZE_BACKBONE,
        ).to(device)
        backbone.eval()
        train_ds = LongitudinalPairDataset(args.dataset_root, split="train", img_size=C.IMG_SIZE,
                                                    rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
        val_ds = LongitudinalPairDataset(args.dataset_root, split="val", img_size=C.IMG_SIZE,
                                                rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
        labels = [train_ds._anomaly_index(r) for r in train_ds.records]  # enables --balanced
    else:
        backbone = None
        train_ds = CachedPairDataset(args.dataset_root, args.cache_dir, split="train",
                                                rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
        val_ds = CachedPairDataset(args.dataset_root, args.cache_dir, split="val",
                                            rotation_k=args.rotation_k, flip_horizontal=args.flip_horizontal)
        labels = train_ds.labels

    # Overfit sanity test: train AND eval on the same N change pairs (is_pathology==1).
    # If val(=train) Dice -> ~0.9 the head can represent the task; if it stalls the
    # bottleneck is representational (decoder resolution / features), not data/optim.
    if args.overfit > 0:
        from torch.utils.data import Subset
        idx = [i for i, l in enumerate(labels) if l != 0][:args.overfit] if labels is not None \
            else list(range(args.overfit))
        sub = Subset(train_ds, idx)
        print(f"[overfit] training + evaluating on {len(idx)} change pairs", flush=True)
        train_loader = DataLoader(sub, batch_size=min(args.batch_size, len(idx)), shuffle=True,
                                  num_workers=args.num_workers, drop_last=False)
        val_loader = DataLoader(sub, batch_size=min(args.batch_size, len(idx)), shuffle=False,
                                num_workers=args.num_workers)
        return backbone, train_loader, val_loader

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
    p.add_argument("--config", default="", help="YAML config; its values become defaults, CLI flags override")
    p.add_argument("-o", "--dataset_root", default=None, help="folder with manifest[_split].jsonl")
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
    p.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm (0 = off); prevents NaN blow-ups")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--rotation_k", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--flip_horizontal", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=None)
    # loss hyperparameters
    p.add_argument("--tau", type=float, default=0.02, help="|GT| threshold for change pixels")
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_dice", type=float, default=1.0)
    p.add_argument("--w_sign", type=float, default=0.5)
    p.add_argument("--w_l2", type=float, default=0.0, help="L2/MSE weight (set w_dice=w_sign=0, w_l2=1 for Itamar-style reconstruction)")
    p.add_argument("--pos_weight", type=float, default=10.0)
    p.add_argument("--target_mode", choices=["continuous", "signed_mask"], default="continuous")
    p.add_argument("--target_threshold", type=float, default=0.05)
    p.add_argument("--background_weight", type=float, default=1.0)
    p.add_argument("--dice_warmup_epochs", type=int, default=0,
                   help="ramp w_dice linearly 0 -> w_dice over the first N epochs (0 = off; L1 learns magnitude first)")
    p.add_argument("--eval_threshold", type=float, default=0.1, help="|map| threshold for Dice/IoU")
    p.add_argument("--eval_gt_threshold", type=float, default=None, help="GT support threshold; defaults to eval_threshold")
    p.add_argument("--resume", action="store_true",
                   help="resume from <save_folder>/last.pt (restores head/optimizer/scheduler/epoch)")
    p.add_argument("--resume_from", default="", help="explicit checkpoint path to resume from (overrides --resume)")
    p.add_argument("--overfit", type=int, default=0,
                   help="sanity test: train+eval on the same N change pairs (0 = off). Use with high --epochs, no --balanced.")
    p.add_argument("--feat_last_k", type=int, default=0,
                   help="use only the last K backbone layers from the cache (0 = all cached). last-1 from a last-4 cache = 1.")
    p.add_argument("--head", choices=["dglori", "dpt", "dpt_cross_attention"], default="dglori")
    p.add_argument("--dpt_channels", type=int, default=96)
    p.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--diagnostics_every", type=int, default=1)
    p.add_argument("--diagnostics_pred_thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    p.add_argument("--diagnostics_gt_thresholds", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    p.add_argument("--diagnostics_components", action="store_true",
                   help="also measure CPU connected-component HMDR/UDPP/F1 at primary thresholds")

    # Config file: pre-scan for --config, load it, and make its keys the arg defaults.
    import sys as _sys
    cfg_path = ""
    for i, a in enumerate(_sys.argv):
        if a == "--config" and i + 1 < len(_sys.argv):
            cfg_path = _sys.argv[i + 1]
        elif a.startswith("--config="):
            cfg_path = a.split("=", 1)[1]
    if cfg_path:
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        p.set_defaults(**cfg)       # config = defaults; anything on the CLI still overrides
    args = p.parse_args()
    if args.diagnostics_every < 1:
        p.error("diagnostics_every must be positive")
    if any(not 0 <= value < 1 for value in args.diagnostics_pred_thresholds + args.diagnostics_gt_thresholds):
        p.error("diagnostic thresholds must be finite and in [0,1)")
    return args


def make_diagnostics(args):
    gt_thresholds = list(args.diagnostics_gt_thresholds)
    if args.target_mode == "signed_mask":
        gt_thresholds.append(args.target_threshold)
    return PredictionDiagnostics(
        args.diagnostics_pred_thresholds, gt_thresholds,
        primary_prediction=args.eval_threshold,
        primary_gt=args.eval_threshold if args.eval_gt_threshold is None else args.eval_gt_threshold,
        target_mode=args.target_mode, target_threshold=args.target_threshold,
        tau=args.tau, pos_weight=args.pos_weight, background_weight=args.background_weight,
        components=args.diagnostics_components)


def main():
    args = parse_args()
    device = args.device
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    if not args.dataset_root:
        raise SystemExit("--dataset_root is required (pass -o ... or set dataset_root in --config)")
    if args.head == "dpt" and args.feat_last_k != 4:
        raise SystemExit("DPT requires --feat_last_k 4 and a last-four cache")
    if args.head == "dpt_cross_attention" and args.feat_last_k not in (1, 4):
        raise SystemExit("Cross-attention DPT requires --feat_last_k 1 or 4")
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.plots_folder, exist_ok=True)

    # Record the exact resolved parameters of this run (config + CLI overrides).
    import json as _json
    with open(os.path.join(args.save_folder, "run_config.json"), "w") as f:
        _json.dump({k: v for k, v in vars(args).items()}, f, indent=2)

    # --feat_last_k: slice cached features to the last K layers (reuse a last-4 cache).
    global _FEAT_DIM
    _FEAT_DIM = None
    patch_dim = None
    if args.feat_last_k > 0:
        _FEAT_DIM = args.feat_last_k * C.BACKBONE_DIM
        patch_dim = _FEAT_DIM
        print(f"[RQ1] using last {args.feat_last_k} layer(s) -> patch_dim {patch_dim}", flush=True)

    backbone, train_loader, val_loader = make_loaders(args, device)
    spec = head_spec(patch_dim, args.head, args.dpt_channels)
    head = build_change_head(spec, device)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[RQ1] trainable head params: {n_trainable/1e6:.2f}M | "
          f"train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    scheduler = cosine_warmup(optimizer, warmup_steps, total_steps)

    best_dice = -1.0
    start_epoch = 0
    metrics_csv = os.path.join(args.save_folder, "metrics.csv")

    # Resume: restore head + optimizer + scheduler + epoch + best score so a reboot
    # costs at most the current epoch.
    ckpt_path = args.resume_from or (os.path.join(args.save_folder, "last.pt") if args.resume else "")
    if ckpt_path and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        saved_config = ck.get("run_config", {})
        for key, default in (("rotation_k", 0), ("flip_horizontal", False), ("target_mode", "continuous"),
                             ("target_threshold", 0.05), ("background_weight", 1.0),
                             ("eval_gt_threshold", None)):
            if saved_config.get(key, default) != getattr(args, key):
                raise ValueError(f"Resume checkpoint differs in {key}; use a new run directory")
        if "head_spec" in ck and ck["head_spec"] != spec:
            raise ValueError("Resume checkpoint architecture does not match this run")
        head.load_state_dict(ck["head"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0))
        best_dice = float(ck.get("best_dice", -1.0))
        print(f"[resume] {ckpt_path} -> start at epoch {start_epoch + 1}, best_dice {best_dice:.4f}", flush=True)
    elif ckpt_path:
        print(f"[resume] no checkpoint at {ckpt_path}; starting fresh", flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        track = args.diagnostics and (epoch == 1 or epoch % args.diagnostics_every == 0)
        train_diagnostics = make_diagnostics(args) if track else None
        val_diagnostics = make_diagnostics(args) if track else None
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        # w_dice warm-up: let L1 fix magnitude first, then ramp Dice in for localization.
        if args.dice_warmup_epochs > 0:
            w_dice_eff = args.w_dice * min(1.0, epoch / args.dice_warmup_epochs)
        else:
            w_dice_eff = args.w_dice
        tr = train_one_epoch(head, backbone, train_loader, optimizer, scheduler, device, args,
                             w_dice=w_dice_eff, diagnostics=train_diagnostics)
        va = evaluate(head, backbone, val_loader, device, threshold=args.eval_threshold,
                      tau=args.tau, pos_weight=args.pos_weight, gt_threshold=args.eval_gt_threshold,
                      target_mode=args.target_mode, target_threshold=args.target_threshold,
                      background_weight=args.background_weight, diagnostics=val_diagnostics,
                      w_l1=args.w_l1, w_l2=args.w_l2, w_dice=w_dice_eff, w_sign=args.w_sign)
        if track:
            metadata = dict(run_config=vars(args), effective_w_dice=w_dice_eff,
                            loss_batch_means={
                                "train": {key: tr[key] for key in ("total", "l1", "l2", "dice", "sign")},
                                "val": dict(total=va["loss_total"], l1=va["l1"], l2=va["l2"],
                                            dice=va["loss_dice"], sign=va["loss_sign"])},
                            learning_rate=optimizer.param_groups[0]["lr"],
                            training_health=tr["health"], epoch_seconds=time.perf_counter() - epoch_started,
                            cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                                                       if torch.cuda.is_available() and str(device).startswith("cuda") else None))
            directory = os.path.join(args.save_folder, "diagnostics")
            train_diagnostics.save(directory, epoch, "train", metadata)
            val_diagnostics.save(directory, epoch, "val", metadata)
            print(f"[diagnostics] epoch {epoch}: {len(train_diagnostics.rows)} online train / "
                  f"{len(val_diagnostics.rows)} validation pairs; {directory}", flush=True)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:03d}/{args.epochs} | lr {lr_now:.2e} | "
              f"target={args.target_mode} cutoff={args.target_threshold:g} | "
              f"weights l1={args.w_l1:g} l2={args.w_l2:g} dice={w_dice_eff:g} sign={args.w_sign:g} "
              f"foreground={1 + args.pos_weight:g} background={args.background_weight:g}", flush=True)
        for split, values in (("train online sampled", tr), ("val fixed head", va)):
            print(f"[{split}] loss_batch_mean total={values.get('loss_total', values.get('total')):.6f} "
                f"weighted_l1={values['l1']:.6f} weighted_l2={values['l2']:.6f} "
                f"dice_loss={values.get('loss_dice', values.get('dice')):.6f} "
                f"sign_loss={values.get('loss_sign', values.get('sign')):.6f}", flush=True)
        primary_gt = args.eval_threshold if args.eval_gt_threshold is None else args.eval_gt_threshold
        print(f"[val checkpoint metric] change_only_support_dice={va['dice']:.4f} "
              f"iou={va['iou']:.4f} pred_abs>{args.eval_threshold:g} raw_gt_abs>{primary_gt:g} "
              f"sensitivity_positive={va['sens_pos']:.4f} sensitivity_negative={va['sens_neg']:.4f}", flush=True)
        metric_target = args.target_threshold if args.target_mode == "signed_mask" else None
        print_prediction_metrics(train_diagnostics, "train", metric_target)
        print_prediction_metrics(val_diagnostics, "val", metric_target)

        # Mid-training tracking: append a CSV row + redraw curves every epoch.
        log_metrics_row(metrics_csv, {
            "epoch": epoch, "lr": lr_now,
            "train_total": tr["total"], "train_l1": tr["l1"], "train_l2": tr["l2"],
            "train_dice": tr["dice"], "train_sign": tr["sign"],
            "val_dice": va["dice"], "val_iou": va["iou"],
            "val_l1": va["l1"], "val_l2": va["l2"],
            "val_l1_ch": va["l1_ch"], "val_l2_ch": va["l2_ch"],
            "val_sens_pos": va["sens_pos"], "val_sens_neg": va["sens_neg"],
            "val_nuis_fp": va["nuis_fp"],
        })
        plot_curves(metrics_csv, os.path.join(args.plots_folder, "train_curves.png"))

        is_best = va["dice"] > best_dice
        if is_best:
            best_dice = va["dice"]
        ckpt = {"head": head.state_dict(), "head_spec": spec, "run_config": vars(args),
            "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_dice": best_dice, "val": va}
        torch.save(ckpt, os.path.join(args.save_folder, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_folder, "best.pt"))
            save_sample_plots(head, backbone, val_loader, device,
                              os.path.join(args.plots_folder, "val_best_samples.png"),
                              target_threshold=args.target_threshold if args.target_mode == "signed_mask" else None)

    print(f"[RQ1] done. best val dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
