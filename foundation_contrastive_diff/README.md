# Foundation-Based Difference Detection with Latent Contrastive Learning

Follow-up research to the Longitudinal CXR change-detection project. This direction fuses a
**frozen chest X-ray foundation model** with a **lightweight "difference head"** and
**contrastive latent-space training** to achieve efficient and semantically structured
longitudinal CXR change detection.

> Reference: [arXiv:2502.05142](https://arxiv.org/pdf/2502.05142) (CXR-Diff) and
> Itamar Sabban MSc Thesis (parent project).

## Core Idea

The frozen backbone is **CheXFound** (Yang et al., IEEE TMI 2025, [arXiv:2502.05142](https://arxiv.org/abs/2502.05142),
[code](https://github.com/RPIDIAL/CheXFound), MIT) — a ViT-L/16 DINOv2 model pretrained on
~987K CXRs. Its **GLoRI** (Global and Local Representations Integration) head is re-purposed
for change detection. See [RQ1_PLAN.md](RQ1_PLAN.md) for the concrete RQ1 design.

```
Prior CXR  ─┐
            ├─→ [FROZEN Foundation Backbone] ─→ feat_prior ─┐
Current CXR ─┘  (shared weights, no grad)     feat_curr  ─┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │  Difference Head   │  (only trainable part)
                                          │  fuse(feat_prior,  │
                                          │       feat_curr)   │
                                          └───┬───────────┬────┘
                                              │           │
                                   difference map     difference embedding z
                                   (segmentation)     (latent vector)
                                              │           │
                                      seg loss │           │ supervised contrastive /
                                               │           │ triplet + orthogonality
```

Only the small difference head is trained; the backbone stays frozen. This reuses powerful
pre-learned radiological features, drastically cutting GPU cost vs. training a full network
from scratch (as the original CXR-Diff did over 60k synthetic pairs).

## What's Novel vs. the Parent Project

1. **Frozen foundation backbone** instead of a from-scratch EfficientNet encoder — parameter
   efficient, faster to converge, trainable on limited GPUs.
2. **Latent contrastive shaping** of the difference embedding — the model learns *what* the
   change is (pathology vs. nuisance, anomaly type, appearance vs. disappearance), not just a
   change/no-change heatmap.
3. **Disentangled projection heads** (pathology subspace vs. nuisance subspace) with an
   orthogonality constraint.

## Layout

```
foundation_contrastive_diff/
├── README.md                      # this file
├── RESEARCH_PLAN.md               # detailed phased implementation plan
├── RQ1_PLAN.md                    # RQ1 plan: CheXFound + GLoRI difference head
├── constants.py                   # config & hyperparameters for this study
├── configs/
│   └── default.yaml               # experiment config
├── models/
│   ├── backbone.py                # frozen foundation-model wrapper
│   ├── difference_head.py         # lightweight Siamese difference head
│   └── projection_heads.py        # pathology / nuisance projection heads
├── losses/
│   └── contrastive_losses.py      # SupCon, triplet, orthogonality
├── datasets/
│   └── pair_dataset.py            # longitudinal pairs + change-type labels
├── training/
│   └── train_foundation_diff.py   # training entry point
└── evaluation/
    ├── change_detection_metrics.py
    └── latent_space_analysis.py   # t-SNE/PCA, clustering metrics
```

## Quick Start (target API)

```bash
cd foundation_contrastive_diff
python -m training.train_foundation_diff --config configs/default.yaml
```

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for the phased plan, milestones, and validation
experiments.

## Relationship to Existing Code

- **Reuses** the synthetic DRR pair generator (`python_files/CT_entities/DRR_generator.py`) and
  its change-type metadata as supervision for the contrastive objective.
- **Reuses** the pair-loading conventions from `python_files/datasets.py`
  (`LongitudinalMIMDataset`) and NIfTI loading utilities.
- **Replaces** the trainable encoder of `LongitudinalMIMModel` with a frozen foundation backbone.
```
