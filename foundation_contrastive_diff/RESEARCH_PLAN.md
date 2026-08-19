# Research Plan: Foundation-Based Difference Detection with Latent Contrastive Learning

This document defines the implementation plan for the follow-up study. It is organized into
phases with concrete deliverables, the experiments that validate each contribution, and the
hyperparameters / decisions that need to be locked before scaling up.

---

## 1. Research Questions & Hypotheses

| # | Question | Hypothesis |
|---|----------|------------|
| RQ1 | Can a **frozen CXR foundation backbone + lightweight difference head** match or exceed the from-scratch CXR-Diff model on longitudinal change detection? | Yes, with far fewer trainable parameters and faster convergence. |
| RQ2 | Does a **supervised contrastive latent loss** organize the difference embedding into semantically meaningful clusters (pathology type, appearance vs. disappearance)? | Yes — embeddings become linearly separable by change type above chance. |
| RQ3 | Does **disentangling pathology vs. nuisance** subspaces reduce false positives from non-clinical changes (positioning, exposure)? | Yes — fewer FP on background-only change pairs. |

---

## 2. Architecture Specification

### 2.1 Frozen Foundation Backbone (`models/backbone.py`)
- **Candidates** (pick one, ablate later):
  - TorchXRayVision DenseNet-121 (pretrained on ~1M CXRs) — easiest, CNN feature maps.
  - A CXR ViT / BiomedCLIP image encoder — patch-token embeddings.
  - The parent project's EfficientNet-B7 encoder weights (for a controlled comparison).
- **Interface**: `forward(img) -> feature_map [B, C, H, W]` (and/or pooled embedding).
- **Frozen**: `requires_grad = False`, `eval()` mode, no BN stat updates.
- Both prior and current images pass through the **same** backbone (Siamese, shared weights).

### 2.2 Difference Head (`models/difference_head.py`)
- Input: `feat_prior`, `feat_curr` (matched spatial size).
- **Fusion options** (configurable, ablate):
  1. Element-wise difference `feat_curr - feat_prior`.
  2. Concatenation `[feat_prior, feat_curr]` → 1×1 conv.
  3. Lightweight cross-attention between the two feature sets.
- **Two outputs**:
  1. **Difference map**: small decoder → signed change map in `[-1, +1]`
     (positive = new finding, negative = resolved) — consistent with the parent project.
  2. **Difference embedding** `z`: global-pooled vector summarizing the change.
- Keep trainable parameter count low (target: < ~5–10M, vs. full-model ~tens of M).

### 2.3 Projection Heads (`models/projection_heads.py`)
- From `z`, two MLP projection heads:
  - `z_path` — pathology-relevant change subspace.
  - `z_nuis` — nuisance / acquisition change subspace.
- **Orthogonality constraint** between the two subspaces for disentanglement.

---

## 3. Supervision & Losses (`losses/contrastive_losses.py`)

The synthetic DRR pipeline knows exactly **where** and **what** each change is. Use these labels:

| Label axis | Values | Used for |
|------------|--------|----------|
| Pathology vs. nuisance | {pathology, background} | binary contrastive grouping |
| Anomaly type | {effusion, pneumothorax, consolidation, fluid overload, none} | multi-class SupCon |
| Direction | {appearance (+), disappearance (−), none} | second latent axis |

**Loss terms** (weighted sum, weights in `constants.py`):
1. **Segmentation loss** on difference map — L1 / SSIM / perceptual (reuse parent losses).
2. **Supervised contrastive loss** (SupCon) on `z_path` over anomaly-type labels.
3. **Triplet loss** (optional alternative / addition) — anchor/positive/negative trios.
4. **Orthogonality loss** between `z_path` and `z_nuis`.
5. (Optional) **direction loss** — small classifier / contrastive on appearance vs. disappearance.

`L_total = λ_seg·L_seg + λ_con·L_supcon + λ_orth·L_orth (+ λ_dir·L_dir)`

---

## 4. Data (`datasets/pair_dataset.py`)

- Reuse synthetic DRR pairs from `python_files/CT_entities/DRR_generator.py`.
- **Augment metadata**: emit per-pair change-type label(s) alongside the BL/FU images and GT
  difference map (extend the generator to log which entity was added/removed and direction).
- Each sample: `(img_prior, img_curr, gt_diff_map, change_type_label, direction_label)`.
- Reuse NIfTI loading and normalization conventions from `python_files/datasets.py`.
- **Real test set**: the ICU longitudinal pairs / PNIMIT annotated pairs used in the parent
  project for final evaluation (no contrastive labels — segmentation + radiologist agreement).

---

## 5. Phased Implementation

### Phase 0 — Scaffolding (this commit)
- [x] Create directory + module skeletons + plan.

### Phase 1 — Frozen backbone + difference head (RQ1)
- [ ] Implement `backbone.py` wrapper for chosen foundation model; verify frozen.
- [ ] Implement `difference_head.py` (start with element-wise diff + small decoder).
- [ ] Wire `pair_dataset.py` to existing synthetic pairs.
- [ ] Train with **segmentation loss only**; reproduce parent-project quality.
- [ ] Log trainable-param count and GPU memory.

### Phase 2 — Latent contrastive learning (RQ2)
- [ ] Add difference embedding output + projection heads.
- [ ] Implement SupCon / triplet losses; add change-type labels to dataset.
- [ ] Joint training (segmentation + contrastive).
- [ ] Latent-space analysis (t-SNE/PCA, silhouette, linear-probe accuracy).

### Phase 3 — Disentanglement (RQ3)
- [ ] Add pathology/nuisance projection heads + orthogonality loss.
- [ ] Construct background-only nuisance pairs (positioning/exposure shifts, no pathology).
- [ ] Measure false-positive reduction on nuisance pairs.

### Phase 4 — Ablations & write-up
- [ ] Backbone choice ablation (foundation vs. parent encoder vs. ImageNet).
- [ ] Fusion-strategy ablation (diff vs. concat vs. cross-attn).
- [ ] Contrastive on/off ablation.
- [ ] Efficiency report + paper figures.

---

## 6. Validation Experiments

### 6.1 Resource efficiency (RQ1)
- Compare training time, GPU memory, trainable-param count vs. from-scratch CXR-Diff.
- Expected: large reduction in trainable params and compute.

### 6.2 Change-detection performance (RQ1)
- Segmentation accuracy of detected changes on the real ICU/PNIMIT test pairs.
- Metrics: **Pairwise Agreement Index (PAI)** with radiologists, sensitivity for
  positive/negative changes, Dice/IoU on change regions.
- Ablation: frozen backbone vs. fully-trained — expect similar accuracy, faster convergence.

### 6.3 Latent-space analysis (RQ2)
- t-SNE / PCA of difference embeddings colored by change category.
- Quantify with silhouette score, inter-/intra-class centroid separation.
- Linear probe / kNN classifier on embeddings → predict change type above chance.

### 6.4 Contrastive ablation (RQ3)
- Toggle contrastive loss off; expect more nuisance-driven false positives and lower
  cluster separation. With it on: fewer FP, higher separation / retrieval accuracy.

---

## 7. Metrics Summary

| Goal | Metric |
|------|--------|
| Efficiency | trainable params, GPU mem (GB), wall-clock/epoch |
| Segmentation | Dice/IoU, PAI vs. radiologists, pos/neg sensitivity |
| Latent structure | silhouette, centroid separation, linear-probe / kNN accuracy |
| Disentanglement | FP rate on nuisance-only pairs, subspace correlation |

---

## 8. Open Decisions / TODO before scaling

- [ ] Pick the foundation backbone (license + feature-map resolution matter).
- [ ] Decide spatial resolution alignment between backbone features and decoder.
- [ ] Choose SupCon vs. triplet (or both) and embedding dimensionality.
- [ ] Define exact synthetic change-type taxonomy emitted by the DRR generator.
- [ ] Confirm real test set + radiologist annotations available for PAI.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Frozen features too generic for fine change detection | Allow optional shallow adapter / LoRA on last backbone block. |
| Synthetic→real domain gap | Light fine-tune of head on a small real slice; domain augmentation. |
| Contrastive collapse | Temperature tuning, hard-negative mining, normalize embeddings. |
| Backbone feature resolution too low for pixel diff map | Use multi-scale features / skip connections from backbone. |
