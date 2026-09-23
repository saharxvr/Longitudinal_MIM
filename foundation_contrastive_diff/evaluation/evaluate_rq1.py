"""
RQ1 evaluation — load a trained D-GLoRI head and report change-detection metrics.

Answers the RQ1 claims quantitatively on a held-out split (default: test):
    * Dice / IoU on the signed change map, broken down PER ANOMALY TYPE.
    * Directional sensitivity (sens+ = appeared, sens- = resolved).
    * Nuisance false-positive area: on nuisance-only pairs (is_pathology == 0, GT ~ 0),
      the fraction of pixels the model wrongly flags as change -> tests the
      "ignores devices / projection angle" claim (lower is better).

Writes a JSON report + a per-type bar chart + a qualitative GT-vs-pred panel.

PAI vs. radiologists is deferred to the real annotated ICU/PNIMIT pairs (no radiologist
GT exists for the synthetic set); see pairwise_agreement_index in change_detection_metrics.

Run:
    python -m evaluation.evaluate_rq1 --dataset_root /path/to/fcd_train \
        --cache_dir ./feature_cache --ckpt ./checkpoints/best.pt --split test
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from models import DifferenceHead
from datasets import CachedPairDataset, LongitudinalPairDataset
from evaluation.change_detection_metrics import dice_score, iou_score, directional_sensitivity


def build_head(device):
    return DifferenceHead(
        backbone_dim=C.BACKBONE_DIM, d_glori=C.D_GLORI,
        num_change_queries=C.NUM_CHANGE_QUERIES, num_heads=C.GLORI_NUM_HEADS,
        grid=C.FEATURE_GRID, embed_dim=C.EMBED_DIM, fusion_mode=C.FUSION_MODE,
        use_adaptive_temperature=C.USE_ADAPTIVE_TEMPERATURE,
        use_pyramid_patch_merging=C.USE_PYRAMID_PATCH_MERGING,
        integrate_global_cls=C.INTEGRATE_GLOBAL_CLS,
        out_size=C.IMG_SIZE, out_range=C.DECODER_OUT_RANGE,
    ).to(device)


def _tokens(batch, backbone, device):
    if backbone is None:
        return (batch["p_prior"].to(device), batch["p_curr"].to(device),
                batch["cls_prior"].to(device), batch["cls_curr"].to(device))
    fp = backbone(batch["img_prior"].to(device)); fc = backbone(batch["img_curr"].to(device))
    return fp["patch_tokens"], fc["patch_tokens"], fp["cls_token"], fc["cls_token"]


def _mean(x):
    return float(sum(x) / max(1, len(x)))


@torch.no_grad()
def run_eval(head, backbone, loader, device, thr):
    head.eval()
    per_type = defaultdict(lambda: {"dice": [], "iou": [], "sens_pos": [], "sens_neg": []})
    nuis_fp = []                        # predicted-change area on nuisance-only pairs
    examples = {}                       # one (gt, pred) per anomaly type for the panel

    for batch in loader:
        p_prior, p_curr, cls_prior, cls_curr = _tokens(batch, backbone, device)
        gt = batch["gt_diff"].to(device)
        pred, _ = head(p_prior, p_curr, cls_prior, cls_curr)

        atype = batch["anomaly_type"]
        is_path = batch["is_pathology"]
        pm = (pred.abs() > thr).float()
        gm = (gt.abs() > thr).float()

        for b in range(pred.shape[0]):
            a = int(atype[b]); path = int(is_path[b])
            name = C.ANOMALY_TYPES[a] if a < len(C.ANOMALY_TYPES) else str(a)
            if path == 0:
                nuis_fp.append(float(pm[b].mean()))       # any flagged pixel is a false positive
                continue
            per_type[name]["dice"].append(dice_score(pm[b], gm[b]))
            per_type[name]["iou"].append(iou_score(pm[b], gm[b]))
            d = directional_sensitivity(pred[b:b + 1], gt[b:b + 1], thr)
            per_type[name]["sens_pos"].append(d["sensitivity_positive"])
            per_type[name]["sens_neg"].append(d["sensitivity_negative"])
            if name not in examples:
                examples[name] = (gt[b, 0].cpu(), pred[b, 0].cpu())

    summary = {}
    all_d, all_i = [], []
    for name, m in per_type.items():
        summary[name] = {"n": len(m["dice"]), "dice": _mean(m["dice"]), "iou": _mean(m["iou"]),
                         "sens_pos": _mean(m["sens_pos"]), "sens_neg": _mean(m["sens_neg"])}
        all_d += m["dice"]; all_i += m["iou"]
    report = {
        "overall": {"n": len(all_d), "dice": _mean(all_d), "iou": _mean(all_i)},
        "per_anomaly_type": summary,
        "nuisance_false_positive_area": {"n": len(nuis_fp), "mean_fp_area": _mean(nuis_fp)},
        "threshold": thr,
    }
    return report, examples


def plot_per_type(report, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    types = list(report["per_anomaly_type"].keys())
    if not types:
        return
    dice = [report["per_anomaly_type"][t]["dice"] for t in types]
    iou = [report["per_anomaly_type"][t]["iou"] for t in types]
    x = range(len(types))
    fig, ax = plt.subplots(figsize=(1.6 * len(types) + 2, 4))
    ax.bar([i - 0.2 for i in x], dice, width=0.4, label="Dice")
    ax.bar([i + 0.2 for i in x], iou, width=0.4, label="IoU")
    ax.set_xticks(list(x)); ax.set_xticklabels(types, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.legend()
    ax.set_title("RQ1: change-map Dice / IoU per anomaly type")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_examples(examples, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(examples.keys())
    if not names:
        return
    n = len(names)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    ax = ax.reshape(2, n)
    for i, name in enumerate(names):
        gt, pred = examples[name]
        ax[0, i].imshow(gt, cmap="bwr", vmin=-1, vmax=1)
        ax[0, i].set_title(f"{name}\nGT"); ax[0, i].axis("off")
        ax[1, i].imshow(pred, cmap="bwr", vmin=-1, vmax=1)
        ax[1, i].set_title("pred"); ax[1, i].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="RQ1 evaluation of the D-GLoRI change-map head")
    p.add_argument("-o", "--dataset_root", required=True)
    p.add_argument("--cache_dir", default=C.FEATURE_CACHE_DIR)
    p.add_argument("--ckpt", default=os.path.join(C.SAVE_FOLDER, "best.pt"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=C.DEVICE)
    p.add_argument("--threshold", type=float, default=0.1, help="|map| threshold for change masks")
    p.add_argument("--out_dir", default=os.path.join(C.PLOTS_FOLDER, "eval"))
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    if args.no_cache:
        from models import FrozenCXRBackbone
        backbone = FrozenCXRBackbone(
            name=C.BACKBONE, model_id=C.RAD_DINO_MODEL,
            backbone_img_size=C.BACKBONE_IMG_SIZE, num_patch_tokens=C.NUM_PATCH_TOKENS,
            last_n_layers=C.LAST_N_LAYERS, freeze=C.FREEZE_BACKBONE,
        ).to(device).eval()
        ds = LongitudinalPairDataset(args.dataset_root, split=args.split, img_size=C.IMG_SIZE)
    else:
        backbone = None
        ds = CachedPairDataset(args.dataset_root, args.cache_dir, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    head = build_head(device)
    state = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(state["head"])
    print(f"[eval] loaded {args.ckpt} (epoch {state.get('epoch', '?')}) | split={args.split} | n={len(ds)}")

    report, examples = run_eval(head, backbone, loader, device, args.threshold)

    json_path = os.path.join(args.out_dir, f"report_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    plot_per_type(report, os.path.join(args.out_dir, f"per_type_{args.split}.png"))
    plot_examples(examples, os.path.join(args.out_dir, f"examples_{args.split}.png"))

    o = report["overall"]; nf = report["nuisance_false_positive_area"]
    print(f"[eval] overall (n={o['n']}): Dice {o['dice']:.4f} | IoU {o['iou']:.4f}")
    for name, m in report["per_anomaly_type"].items():
        print(f"    {name:16s} n={m['n']:5d}  Dice {m['dice']:.3f}  IoU {m['iou']:.3f}  "
              f"sens+ {m['sens_pos']:.3f}  sens- {m['sens_neg']:.3f}")
    print(f"[eval] nuisance-only FP area (n={nf['n']}): {nf['mean_fp_area']:.4f}  (lower is better)")
    print(f"[eval] wrote {json_path} + plots to {args.out_dir}")


if __name__ == "__main__":
    main()
