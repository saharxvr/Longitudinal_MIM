# RQ1 results — run index

Signed longitudinal change-map detection: frozen RAD-DINO + D-GLoRI head.
Each `runN/` has its own README with full config, eval table, and notes.

## Comparison (eval @ threshold 0.05, val split, change-only Dice)

| run | backbone | key config | overall Dice | nuisance FP | notes |
|---|---|---|---|---|---|
| [run1](run1/README.md) | RAD-DINO last-1 | 50 ep, pos_weight 10 | **0.433** | **0.9%** | baseline; annealing-driven climb |
| [run4](run4/README.md) | RAD-DINO last-1 | 60 ep, pos_weight 15, dice warm-up 10 | 0.426 | 0.9% | faster convergence, same endpoint |

runs 2 & 3 aborted (150-ep cosine stalled / superseded) — see run4 README.

## Headline RQ1 findings
- **Localization:** ~0.43 Dice; per-type ordering effusion > fluid > pneumothorax > consolidation (clinically sensible).
- **Direction-aware:** sign loss ≈ 0 (appeared vs resolved correct).
- **Nuisance invariance:** **<1% false-positive area** on device/angle-only pairs — reproducible across runs. *The headline.*
- **Ceiling:** loss/schedule tuning saturated at ~0.43 for frozen-last-layer features. Next lever = representational capacity (ablation C: `LAST_N_LAYERS=4`).
