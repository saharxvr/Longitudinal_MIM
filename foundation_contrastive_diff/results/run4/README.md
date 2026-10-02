# RQ1 — run4 (Dice warm-up + tuned loss)

Second tuned run. Same frozen RAD-DINO (last **1** layer) + D-GLoRI head as run1, but with
a **w_dice warm-up ramp** and stronger localization weights, on a 60-epoch schedule.
Goal: beat run1's ~0.43 Dice. Outcome: **faster convergence, same endpoint** → loss/schedule
tuning is saturated for this architecture.

(runs 2 and 3 were aborted: run2 used a 150-epoch cosine that stretched the annealing out so
it stalled at ~0.26; run3 was superseded by run4.)

## Config
| | |
|---|---|
| backbone | RAD-DINO, **frozen**, `LAST_N_LAYERS=1` |
| head | D-GLoRI, 9.46M trainable params |
| epochs | 60 |
| batch / accum | 8 / 4 (eff. 32) |
| optimizer | AdamW, lr 3e-4, wd 1e-2, cosine + 2-epoch warmup |
| sampler | class-balanced |
| loss weights | w_l1 1.0, **w_dice 1.5**, w_sign 0.5, **pos_weight 15**, tau 0.02 |
| **dice warm-up** | **10 epochs** (w_dice ramps 0 → 1.5) |
| eval threshold | 0.05 |

Launch:
```bash
python -u -m training.train_foundation_diff -o $DATA --cache_dir $CACHE \
    --save_folder $RUN4/checkpoints --plots_folder $RUN4/plots \
    --epochs 60 --batch_size 8 --accum 4 --lr 3e-4 --balanced \
    --w_dice 1.5 --pos_weight 15 --eval_threshold 0.05 --dice_warmup_epochs 10
```

## Results (eval @ threshold 0.05, val split, change-only unless noted)
| anomaly | Dice | IoU | sens+ | sens- | n |
|---|---|---|---|---|---|
| pleural_effusion | 0.517 | 0.420 | 0.309 | 0.306 | 294 |
| fluid_overload | 0.459 | 0.371 | 0.271 | 0.300 | 309 |
| pneumothorax | 0.400 | 0.306 | 0.263 | 0.221 | 377 |
| consolidation | 0.341 | 0.251 | 0.298 | 0.148 | 313 |
| **overall (best.pt, ep60)** | **0.426** | **0.334** | | | 1293 |

- **Nuisance false-positive area: 0.0091 (0.9%)** — same as run1 (claim intact).

## Notes (the instructive part)
- **Dice warm-up killed the early plateau:** val Dice hit 0.30 by epoch 2 vs run1's flat ~0.20 for 20+ epochs. At **epoch 17** the clean eval already matched run1's *final* Dice (0.438) with better consolidation.
- **Transient over-firing:** at epoch 17 nuisance FP was **0.041 (4.1%)** because pos_weight 15 boosted recall too hard. The **annealing phase (epoch 33→60, LR→0) tightened predictions** (sens ↓, Dice ↑) and **FP recovered to 0.9%**.
- **Endpoint ≈ run1** (0.426 vs 0.433). Confirms **~0.43 Dice + 0.9% FP is a reproducible ceiling** for frozen-last-layer features on this diffuse synthetic GT. The warm-up bought speed, not a higher ceiling.
- **Conclusion:** next gains require more representational capacity (ablation C = `LAST_N_LAYERS=4`), not more loss tuning.

## Files
- `train_curves.png`, `val_best_samples.png`, `metrics.csv`
- `eval/report_val.json`, `eval/per_type_val.png`, `eval/examples_val.png`
