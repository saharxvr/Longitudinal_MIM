"""
Diagnostic #2 (cheapest): does RAD-DINO's latent space even ENCODE our change signal?

For every val pair we read the cached patch tokens and ask, per 37x37 patch, whether the
prior->current *feature movement* tracks the ground-truth change magnitude:

    change_score(patch) = 1 - cos(prior_token, curr_token)      # unsigned feature movement
    l2_score(patch)     = ||curr_token - prior_token||_2
    gt_mag(patch)       = mean(|gt_diff|) pooled to 37x37

We report, pooled over all patches of all pairs:
    * Pearson + Spearman correlation of (change_score, gt_mag) and (l2_score, gt_mag)
    * AUROC of change_score / l2_score separating change patches (gt_mag > tau) from the rest

Interpretation:
    * High correlation / AUROC (>~0.75)  -> the change signal SURVIVES in RAD-DINO's space;
      the head is the bottleneck -> DPT / better head is the right lever.
    * Near-zero correlation / AUROC ~0.5 -> the frozen backbone is largely BLIND to our
      deltas; no head can recover it -> adapt / replace the backbone.

No training, no backbone forward pass: pure read over the existing feature cache.

Run:
    python -m evaluation.feature_change_correlation -o $DATA --cache_dir $CACHE4 \
        --split val --feat_last_k 1 --tau 0.05 [--limit 400]
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants as C
from datasets import CachedPairDataset


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / (denom + 1e-12))


def _rank(x: torch.Tensor) -> torch.Tensor:
    # average ranks (ties broken by order; good enough for a monotonic-correlation probe)
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rank(x), _rank(y))


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the Mann-Whitney U statistic (rank-sum). labels in {0,1}."""
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rank(scores) + 1.0  # 1-based ranks
    sum_pos = float(ranks[pos].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--dataset_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--feat_last_k", type=int, default=1,
                    help="use the last K backbone layers from a last-4 cache (1=last layer only)")
    ap.add_argument("--tau", type=float, default=0.05, help="|gt| patch threshold for a change patch")
    ap.add_argument("--limit", type=int, default=0, help="cap #pairs (0 = all)")
    args = ap.parse_args()

    ds = CachedPairDataset(args.dataset_root, args.cache_dir, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    grid = C.FEATURE_GRID
    feat_dim = args.feat_last_k * C.BACKBONE_DIM if args.feat_last_k > 0 else None

    cos_all, l2_all, gt_all, lbl_all = [], [], [], []
    per_type = defaultdict(lambda: {"cos": [], "gt": []})
    n = 0
    for batch in loader:
        pp = batch["p_prior"][0]            # [1369, D]
        pc = batch["p_curr"][0]
        if feat_dim is not None:
            pp, pc = pp[..., -feat_dim:], pc[..., -feat_dim:]
        gt = batch["gt_diff"][0]            # [1, H, W]

        cos = 1.0 - F.cosine_similarity(pp, pc, dim=-1)          # [1369]
        l2 = (pc - pp).norm(dim=-1)                              # [1369]
        gt_mag = F.adaptive_avg_pool2d(gt.abs().unsqueeze(0), grid).reshape(-1)  # [1369]

        cos_all.append(cos); l2_all.append(l2); gt_all.append(gt_mag)
        lbl_all.append((gt_mag > args.tau).float())

        a = int(batch["anomaly_type"][0])
        name = C.ANOMALY_TYPES[a] if a < len(C.ANOMALY_TYPES) else str(a)
        per_type[name]["cos"].append(cos); per_type[name]["gt"].append(gt_mag)

        n += 1
        if args.limit and n >= args.limit:
            break

    cos_all = torch.cat(cos_all); l2_all = torch.cat(l2_all)
    gt_all = torch.cat(gt_all); lbl_all = torch.cat(lbl_all)

    print(f"\n[feature-change correlation]  split={args.split}  pairs={n}  "
          f"patches={cos_all.numel()}  feat_last_k={args.feat_last_k}  tau={args.tau}")
    print(f"  change patches: {int(lbl_all.sum())} / {lbl_all.numel()} "
          f"({100*float(lbl_all.mean()):.1f}%)")
    print("  ---- pooled over all patches ----")
    print(f"  cos-distance  : pearson {_pearson(cos_all, gt_all):+.3f}  "
          f"spearman {_spearman(cos_all, gt_all):+.3f}  AUROC {_auroc(cos_all, lbl_all):.3f}")
    print(f"  l2-distance   : pearson {_pearson(l2_all, gt_all):+.3f}  "
          f"spearman {_spearman(l2_all, gt_all):+.3f}  AUROC {_auroc(l2_all, lbl_all):.3f}")

    print("  ---- per anomaly type (cos-distance, Spearman) ----")
    for name, m in sorted(per_type.items()):
        c = torch.cat(m["cos"]); g = torch.cat(m["gt"])
        print(f"    {name:<18} n_pairs {len(m['cos']):>4}  spearman {_spearman(c, g):+.3f}")

    auc = _auroc(cos_all, lbl_all)
    verdict = ("signal SURVIVES -> head is the bottleneck (try DPT)"
               if auc >= 0.75 else
               "weak/absent signal -> backbone likely the bottleneck (adapt/replace)"
               if auc <= 0.6 else "ambiguous -> run the linear probe to decide")
    print(f"\n  verdict (cos AUROC {auc:.3f}): {verdict}\n")


if __name__ == "__main__":
    main()
