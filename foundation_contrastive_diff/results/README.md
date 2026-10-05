# RQ1 results — run index

Signed longitudinal change-map detection: frozen RAD-DINO + D-GLoRI head.
Each `runN/` has its own README with full config, eval table, and notes.

## Comparison (eval @ threshold 0.05, val split, change-only Dice)

| run | backbone | key config | overall Dice | nuisance FP | notes |
|---|---|---|---|---|---|
| [run1](run1/README.md) | RAD-DINO last-1 | 50 ep, pos_weight 10 | 0.433 | **0.9%** | baseline; annealing-driven climb |
| [run4](run4/README.md) | RAD-DINO last-1 | 60 ep, pos_weight 15, dice warm-up 10 | 0.426 | 0.9% | faster convergence, same endpoint |
| [run5](run5/README.md) | RAD-DINO **last-4** | 60 ep, grad_clip, dice warm-up 10 | **0.464** | 2.0% | **ablation C: +0.031 Dice, every type up** |

runs 2 & 3 aborted (150-ep cosine stalled / superseded) — see run4 README.

## Headline RQ1 findings
- **Localization:** last-1 ~0.43 Dice; **last-4 ~0.46 (+7%)**, with consolidation (hardest) improving most. Per-type ordering effusion > fluid > pneumothorax > consolidation (clinically sensible).
- **Direction-aware:** sign loss ≈ 0 (appeared vs resolved correct).
- **Nuisance invariance:** <1% FP (last-1) / 2% (last-4) false-positive area on device/angle-only pairs.
- **Ceiling:** loss/schedule tuning saturated per feature set; **more frozen layers (last-4) raised the ceiling** ~0.43→0.46. Still fitting-limited (flat train Dice) -> next lever is multi-scale/higher-res features (ViT-Adapter/FPN) or a capacity check (overfit test).
