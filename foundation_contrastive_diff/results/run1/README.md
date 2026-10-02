# RQ1 — run1 (frozen last-layer baseline)

First successful RQ1 training run. Frozen RAD-DINO (last **1** layer) + D-GLoRI head,
signed change-map loss (weighted-L1 + soft-Dice + sign). This is the baseline all later
runs are compared against.

## Config
| | |
|---|---|
| backbone | RAD-DINO `microsoft/rad-dino`, **frozen**, `LAST_N_LAYERS=1` |
| head | D-GLoRI, 9.46M trainable params |
| epochs | 50 |
| batch / accum | 8 / 4 (eff. 32) |
| optimizer | AdamW, lr 3e-4, wd 1e-2, cosine + 2-epoch warmup |
| sampler | class-balanced (`--balanced`) |
| loss weights | w_l1 1.0, **w_dice 1.0**, w_sign 0.5, **pos_weight 10**, tau 0.02 |
| dice warm-up | none |

Launch:
```bash
python -u -m training.train_foundation_diff -o $DATA --cache_dir $CACHE \
    --save_folder $RUN/checkpoints --plots_folder $RUN/plots \
    --epochs 50 --batch_size 8 --accum 4 --lr 3e-4 --balanced
```

## Results (eval @ threshold 0.05, val split, change-only unless noted)
| anomaly | Dice | IoU | sens+ | sens- | n |
|---|---|---|---|---|---|
| pleural_effusion | 0.511 | 0.413 | 0.303 | 0.306 | 294 |
| fluid_overload | 0.465 | 0.378 | 0.277 | 0.297 | 309 |
| pneumothorax | 0.420 | 0.322 | 0.267 | 0.240 | 377 |
| consolidation | 0.344 | 0.257 | 0.303 | 0.142 | 313 |
| **overall** | **0.433** | **0.340** | | | 1293 |

- **Nuisance false-positive area (no-change pairs): 0.0090 (0.9%)** — the headline RQ1 result (ignores devices / projection angle).
- Best training-log val Dice (mixed metric, thr 0.1): 0.391 (epoch 48).

## Notes
- val Dice sat ~0.20 until epoch ~22, then climbed to 0.39 as the cosine LR annealed to ~0. The climb is **annealing-driven**.
- Per-type ordering is clinically sensible: effusion/fluid (large, basal) easiest; consolidation (diffuse, patchy) hardest.
- Predictions are peaky/blobby vs the diffuse faint GT — the main quality gap.

## Files
- `train_curves.png` — loss + val Dice/IoU vs epoch
- `val_best_samples.png` — GT vs pred heatmaps (best epoch)
- `metrics.csv` — per-epoch metrics
