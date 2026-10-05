# RQ1 — run5 (ablation C: last-4 features)

Ablation C: feed the D-GLoRI head the **concatenation of the last 4 RAD-DINO layers**
(3072-dim patch tokens) instead of the last layer only (768-dim). Backbone still frozen,
head-only training. Tests whether richer multi-layer frozen features raise the Dice ceiling.

## Config
| | |
|---|---|
| backbone | RAD-DINO, **frozen**, `LAST_N_LAYERS=4` (patch_dim 3072, cls_dim 768) |
| features | cached last-4 (`fcd_cache_last4`, ~360 GB) |
| epochs | 60 |
| batch / accum | 8 / 4 (eff. 32) |
| optimizer | AdamW lr 3e-4, wd 1e-2, cosine + 2-ep warmup, **grad_clip 1.0** |
| sampler | class-balanced |
| loss | w_l1 1.0, w_dice 1.5 (warmup 10 ep), w_sign 0.5, pos_weight 10, tau 0.02 |
| eval threshold | 0.05 |

Launch:
```bash
python -u -m training.train_foundation_diff -o $DATA --cache_dir $CACHE4 \
    --save_folder $RUN5/checkpoints --plots_folder $RUN5/plots \
    --epochs 60 --batch_size 8 --accum 4 --lr 3e-4 --balanced \
    --w_dice 1.5 --pos_weight 10 --eval_threshold 0.05 --dice_warmup_epochs 10 \
    --num_workers 12 --grad_clip 1.0
```

## Results (eval @ threshold 0.05, val split, change-only)
| anomaly | Dice | IoU | sens+ | sens- | n |
|---|---|---|---|---|---|
| pleural_effusion | 0.550 | 0.440 | 0.380 | 0.381 | 294 |
| fluid_overload | 0.468 | 0.374 | 0.319 | 0.383 | 309 |
| pneumothorax | 0.448 | 0.331 | 0.349 | 0.332 | 377 |
| consolidation | 0.401 | 0.293 | 0.396 | 0.217 | 313 |
| **overall** | **0.464** | **0.357** | | | 1293 |

- **Nuisance false-positive area: 0.0202 (2.0%)** — up from last-1's 0.9% (over-firing from pos_weight; tunable down).

## vs last-1 baseline (run1)
| | run1 (last-1) | run5 (last-4) | Δ |
|---|---|---|---|
| overall Dice | 0.433 | **0.464** | **+0.031 (+7%)** |
| consolidation | 0.344 | **0.401** | +0.057 |
| pleural_effusion | 0.511 | 0.550 | +0.039 |
| pneumothorax | 0.420 | 0.448 | +0.028 |
| fluid_overload | 0.465 | 0.468 | ~0 |
| nuisance FP | 0.009 | 0.020 | ↑ 2.2× |

## Notes
- **last-4 consistently beats last-1** on every anomaly type; biggest gain on the hardest (consolidation, +17% relative). Supports "richer frozen features -> better localization".
- Converged **fast** (hit ~0.44 by epoch ~3, annealing climb to ~0.46 by epoch 44 then flat).
- Still **fitting-limited**: train soft-Dice loss plateaued ~0.655 (overlap ~0.35), tiny train-val gap -> not overfitting; the ceiling is representational/data, not generalization. (Overfit test pending to confirm.)
- Infra notes: needed grad_clip (last-4 activations NaN'd at epoch 6 without it); cached 360 GB didn't fit RAM so NFS-bound (~20 min/epoch); OS update broke the shared venv (rebuilt py3.13 venv + NVIDIA LD_LIBRARY_PATH fix).

## Files
- `train_curves.png`, `val_best_samples.png`, `metrics.csv`, `eval/report_val.json`
