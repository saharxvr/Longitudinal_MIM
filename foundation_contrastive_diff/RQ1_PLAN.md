# RQ1 Plan — Foundation-Based Difference Detection with CheXFound + GLoRI

> **RQ1**: *Can a **pretrained chest X-ray foundation model** (CheXFound), adapted with a
> lightweight global/local difference head, **detect and localize the longitudinal changes of a
> chosen set of anomalies** between a prior and a current CXR?*

The core question is **feasibility of foundation-model-based longitudinal change detection**:
the backbone was pretrained on single images (DINOv2 self-distillation), never on *pairs* — can
its frozen representations be turned into an accurate signed **change map** for the chosen
anomaly types (`consolidation`, `pleural_effusion`, `pneumothorax`, `fluid_overload`), while
ignoring nuisance change (devices, projection angle)?

This plan grounds RQ1 in a specific foundation model and a specific global+local integration
mechanism:

- **Backbone**: **CheXFound** — Yang et al., *Chest X-ray Foundation Model with Global and
  Local Representations Integration*, IEEE TMI 2025 ([arXiv:2502.05142](https://arxiv.org/abs/2502.05142),
  code [github.com/RPIDIAL/CheXFound](https://github.com/RPIDIAL/CheXFound), MIT license).
- **Integration mechanism**: **GLoRI** (Global and Local Representations Integration),
  re-purposed from multilabel classification to **difference detection** (a "D-GLoRI" head).

RQ1 is the **segmentation-only** stage (no contrastive loss yet — that is RQ2). The primary
goal is to show the frozen foundation model *can* find the chosen anomaly changes; matching
from-scratch quality with far fewer trainable parameters and faster convergence is a
**secondary** benefit, reported but not the headline.

---

## 0. Has this been done? (related work)

Short answer: **not in this exact form.** A literature scan (arXiv, mid-2026) shows longitudinal
CXR work clusters into categories that are *adjacent but different* from a dense, frozen-foundation
**difference heatmap**:

| Line of work | Representative papers | Output | Gap vs. our RQ1 |
|--------------|-----------------------|--------|-----------------|
| Temporal **report generation** / VLM | BioViL-T; CoCa-CXR (2502.20509); CXRMate-2 (2604.18967); TRACE (2602.02963); "Enhanced Contrastive Learning w/ Multi-view Longitudinal" (CVPR'25) | **text** comparing prior/current | no pixel-level change map |
| Progression **classification / VQA** | CheXRelNet (MICCAI'22); CheXLearner (2505.06903); "Directional Semantic Transitions" (2606.15938, MICCAI'26); CheXTemporal (2605.11304) | per-region/global **labels** ("improved/worse/stable") | region/label-level, not a dense signed heatmap |
| Dense change **localization** (other modalities) | To et al. MS-brain MRI (2106.00919); Lung-lesion-burden CT (2504.06924) | dense change map / burden | **not CXR**, not foundation-model based |
| Disease-progression **modeling** (multimodal) | Spatiotemporal disentanglement (2510.11112, NeurIPS'25); EHR→CXR generation (2410.17918) | future image / fused representation | predicts/encodes, doesn't *segment the change* |

**What is novel here**: producing a **dense, pixel-level signed change heatmap** of *chosen
anomalies* on longitudinal **CXR pairs**, driven by a **frozen CXR foundation model** (CheXFound)
that was pretrained on single images — plus the explicit **nuisance-invariance** (device/angle
changes rendered but ignored). The closest dense-change work (To et al.) is brain MRI and trained
from scratch; the closest CXR work is text/label/region-level. This positions RQ1 as the first
foundation-model-based dense longitudinal-change segmenter for CXR. *(Worth a deeper systematic
review before writing the thesis intro — these are the main threads to cite and contrast.)*

### Outside medicine: this paradigm is mature in remote sensing

The *method* — **frozen foundation backbone + Siamese + difference head → dense change map** — is
well established in **remote-sensing change detection** (satellite before/after imagery). So the
method is **not** the novelty; the **domain transfer to longitudinal CXR** (+ nuisance-invariance
+ synthetic-DRR supervision) is. Key analogues to cite:

| Non-medical work | Relevance |
|------------------|-----------|
| Time Travelling Pixels (2312.16202) | frozen **SAM** bitemporal features → change detection (closest twin to our recipe) |
| Tri-path DINO (2603.01498) | frozen **DINO** features → multi-class change detection (same backbone family as CheXFound) |
| SAM-CD / "Adapting SAM for Change Detection" (2309.01429, TGRS) | foundation-model adapter for dense change maps |
| PeftCD (2509.09572) | frozen vision foundation model + **PEFT/LoRA** → change detection (informs ablation F) |
| FC-Siamese metric nets (1810.09111); SARAS-Net (2212.01287); street-scene CD (2010.09925) | pre-foundation Siamese feature-differencing change heads |
| ComPtr (2307.12349) | general **bi-source dense-prediction transformer** (generic D-GLoRI fusion) |

**Framing for the thesis**: "remote-sensing-style foundation change detection, brought to chest
radiography." Borrow their proven tricks: feature-differencing fusion, pseudo-change false-positive
handling, focal/Tversky losses for rare change (cf. To et al.), and PEFT/LoRA backbone adaptation.

---

## 1. Key facts about CheXFound (drive the design)

| Property | Value |
|----------|-------|
| Architecture | ViT-Large, patch size 16 |
| Input resolution | 512 × 512 |
| Patch grid | 32 × 32 = **1024 patch tokens** + 1 `[CLS]` |
| Embedding dim `D_model` | 1024 |
| Pretraining | DINOv2 self-distillation (MIM loss wt 3, `[CLS]` align wt 1) on **CXR-987K** (~987K CXRs, 12 sources) |
| Downstream input | concat of patch tokens from the **last 4 transformer layers** |
| Released weights | `teacher_checkpoint.pth` (Google Drive) |
| Runtime requirements | PyTorch 2.0 + xFormers 0.0.18 + Linux + CUDA GPU |

> **Environment**: this study runs on the **school Linux PCs with CUDA GPUs** (the same
> machines used to generate the synthetic DRRs). CheXFound's native Linux + xFormers + GPU
> requirements are therefore satisfied directly — no Windows/WSL workaround is needed.

**GLoRI head** (original, for classification): linear-embed patch tokens → `D_GLoRI = 768`,
8-head cross-attention with `M` *disease queries*, an **adaptive-temperature** branch for
fine-grained local features, a **pyramid patch merging** branch (8×8/4×4/2×2 pooling) for
coarse-grained local features, and a **`[CLS]` skip connection** for global features. For
semantic segmentation the paper instead trained a **UPerNet decoder** on the frozen features.

---

## 2. Architecture for RQ1 (Siamese CheXFound + D-GLoRI)

```
                 ┌──────────────── FROZEN CheXFound (ViT-L/16 @512) ───────────────┐
 prior CXR  ───► │  patch tokens P_prior [1024×1024] (32×32 grid) + CLS_prior      │
 current CXR ──► │  patch tokens P_curr  [1024×1024] (32×32 grid) + CLS_curr       │
                 │  (concat last-4-layer tokens; shared weights)                   │
                 └─────────────────────────────────────────────────────────────────┘
                              │ local                              │ global
                  ΔP = P_curr − P_prior  [32×32×D]      ΔCLS = CLS_curr − CLS_prior
                              │                                    │
                 ┌───────────▼──────── D-GLoRI ───────────────┐    │
                 │ linear-embed ΔP → D_GLoRI                   │    │
                 │ ┌─ fine-grained: change-type queries ×     │    │
                 │ │   adaptive-temperature cross-attention    │    │
                 │ └─ coarse-grained: pyramid patch merging    │    │
                 │ fuse(fine, coarse) → enriched ΔP grid       │    │
                 └───────────┬─────────────────────────────────┘    │
                             │ (32×32×C)                             │
                 ┌───────────▼───────────┐              ┌────────────▼──────────┐
                 │  UPerNet-style decoder │              │  global change embed  │
                 │  32×32 → 512×512       │              │  (skip CLS) → z        │  (RQ2)
                 │  Tanh → signed map     │              └────────────────────────┘
                 └────────────────────────┘
                       difference map [−1,+1]
```

**Two output branches**:
1. **Difference map** (RQ1 focus): UPerNet-style decoder upsamples the GLoRI-enriched
   `32×32` patch-difference grid to a `512×512` **signed** change map in `[−1, +1]`
   (positive = new finding, negative = resolved) — same convention as the parent
   `LongitudinalMIMModel`.
2. **Global change embedding `z`** (built now, supervised in RQ2): integrates `ΔCLS` (global)
   with pooled GLoRI local features — the "global + local" integration applied to *change*.

**Local fusion options to ablate**: `ΔP = P_curr − P_prior` (feature differencing) vs. channel
concat `[P_prior, P_curr]` → 1×1 vs. cross-image attention (current queries attend to prior
tokens). Default: feature differencing.

**Output = a single change heatmap (not categorized).** The head emits **one** signed change
map in `[−1, +1]` (positive = new finding, negative = resolved) covering *all* chosen anomaly
types at once — it does **not** classify which anomaly changed. Per-anomaly performance is
recovered at **evaluation** time by stratifying the test pairs by the anomaly that was inserted,
not by any class head in the model.

**Change queries (generic, not per-class)**: GLoRI's `M` disease queries become `M` *generic
latent change queries* (`NUM_CHANGE_QUERIES`, default 8) — an internal cross-attention
mechanism that lets different queries specialize to different spatial/appearance patterns of
change. They are **not** labeled with anomaly classes, so `M` is a free hyperparameter
decoupled from `len(ANOMALY_TYPES)`. Their attention maps are still a free qualitative
by-product, and the pooled query features seed the RQ2 contrastive embedding.

---

## 3. Efficiency strategy (central to RQ1)

Because the backbone is **frozen**, precompute once and never run the 307M ViT again during
head training:

1. **Feature caching**: run CheXFound over every (prior, current) pair **once** on the school
   Linux GPU PCs, save the last-4-layer patch tokens + `[CLS]` to disk (`.pt`/`.npy`). Train
   D-GLoRI on cached tensors.
   - Drastically cuts GPU time/memory: the 307M ViT runs only during the one-time caching
     pass, so head-training iterations are fast and fit comfortably on a single GPU.
2. **Same environment throughout**: extraction, head training, and evaluation all run on the
   Linux + CUDA school PCs (same setup as the DRR pipeline). The cached `.pt` tensors are also
   portable if you later want to iterate on the head off-cluster.
3. Report: trainable params (head only) vs. full CXR-Diff; peak GPU memory; wall-clock to
   convergence; (optional) FLOPs of head vs. full model.

---

## 3a. Data requirements & label conventions

The synthetic DRR pipeline supervises the head. Two design rules:

- **Change-type taxonomy (5 classes)**: `none`, `consolidation`, `pleural_effusion`,
  `pneumothorax`, `fluid_overload` (= `M` change-type queries). **Cardiomegaly is excluded**
  (no cardiomegaly *change* is generated).
- **Nuisance variation present in every pair, but ignored**: each (prior, current) pair is
  rendered with **device insertion/removal and projection-angle (positioning) changes across
  all anomaly types**. These are *nuisance* — they must be **rendered** so the model sees them,
  but **zeroed out of the ground-truth difference map** so the head learns invariance to them.
  - GT change map = signed `[−1,+1]` for *pathology* change only; device pixels and
    angle-induced shifts contribute **0** (no change).
  - This makes device/angle robustness a measurable property (false-positive rate on
    device-only / angle-only pairs) and directly seeds RQ3 (pathology vs. nuisance subspaces).

| Pair condition | What changes | GT change map |
|----------------|--------------|---------------|
| Pathology pair | one of the 5 change types appears/resolves | signed map on the pathology region |
| Device-only pair | a device is added/removed | **all-zero** (ignored) |
| Angle-only pair | projection angle / positioning shifts | **all-zero** (ignored) |
| Mixed pair | pathology + device + angle together | signed map on pathology **only** |

> Data action item: ensure the DRR generator outputs (a) the rendered DRR pair, (b) a signed
> pathology-only GT difference map, and (c) per-pair labels (anomaly type, direction,
> device-present flag, angle-delta) so nuisance robustness can be evaluated.

---

## 4. Implementation steps

| Step | Deliverable | File |
|------|-------------|------|
| 1.1 | Vendor / install CheXFound; load `teacher_checkpoint.pth`; freeze; expose last-4-layer patch tokens + `[CLS]` | `models/backbone.py` |
| 1.2 | Feature-caching script: dump per-pair tokens to disk | `training/cache_features.py` (new) |
| 1.3 | Cached-feature dataset (loads `.pt`, returns ΔP inputs + GT diff map + labels) | `datasets/pair_dataset.py` |
| 1.4 | D-GLoRI head: linear-embed + change-type-query cross-attn + adaptive temperature + pyramid patch merging + CLS skip | `models/difference_head.py` |
| 1.5 | UPerNet-style decoder 32×32 → 512×512, Tanh | `models/difference_head.py` |
| 1.6 | Train with **segmentation loss only** (L1 + SSIM, reuse parent losses) | `training/train_foundation_diff.py` |
| 1.7 | Efficiency + change-detection evaluation | `evaluation/change_detection_metrics.py` |

---

## 5. Experiments & validation

### 5.1 Change-detection performance (primary)
- **Change detection/localization** for the chosen types (`consolidation`, `pleural_effusion`,
  `pneumothorax`, `fluid_overload`): does the frozen foundation model + head produce a correct
  signed change **heatmap**? (single heatmap output, not categorized).
- Evaluate on a held-out synthetic split **and** real ICU / PNIMIT longitudinal pairs with
  radiologist annotations.
- Metrics: Dice / IoU on change regions (overall; **stratified by anomaly type at eval time**),
  **Pairwise Agreement Index (PAI)** vs. radiologists, sensitivity for positive (new) and
  negative (resolved) changes, and **false-positive rate on nuisance-only (device/angle) pairs**.
- **Expected**: the foundation model can find the chosen anomaly changes; per-type Dice/IoU
  well above chance; low FP on nuisance-only pairs.

### 5.2 Resource efficiency (secondary)
- Trainable params (D-GLoRI + decoder) vs. full CXR-Diff-Net.
- Peak GPU memory, wall-clock/epoch, epochs-to-converge (with feature caching vs. without).
- **Expected**: order-of-magnitude fewer trainable params; large compute reduction.

### 5.3 RQ1 ablations
| Ablation | Compares | Tests |
|----------|----------|-------|
| A. Representation integration | global+local (D-GLoRI) vs. local-only vs. global-only (`[CLS]` diff) | value of GLoRI-style integration for *change* |
| B. GLoRI components | +adaptive-temperature, +pyramid-patch-merging (incremental, as in CheXFound Table 11) | which local-feature branch helps change detection |
| C. Feature depth | last-4-layer concat vs. last-layer only | depth of frozen features needed |
| D. Local fusion | feature-diff vs. concat vs. cross-image attention | best way to compare the two feature sets |
| E. Backbone | CheXFound vs. RAD-DINO vs. ImageNet ViT vs. parent EfficientNet | value of the CXR foundation model specifically |
| F. Frozen vs. fine-tuned | frozen backbone vs. unfreezing last block / LoRA | convergence vs. accuracy trade-off |

---

## 6. Milestones

- **M1**: CheXFound loads, frozen, emits expected `[B,1024,1024]` patch tokens + `[CLS]` at 512².
- **M2**: Feature cache built for the synthetic train/val pairs.
- **M3**: D-GLoRI + decoder trains (seg loss only); loss curves sane; sample diff maps plotted.
- **M4**: Frozen foundation model + head **detects the chosen anomaly changes** on a held-out
  synthetic split (per-type Dice/IoU reported); low FP on nuisance-only pairs.
- **M5**: Real-test-set evaluation (Dice/IoU/PAI per anomaly) + ablations A–F + efficiency report.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| CheXFound needs Linux + xFormers + GPU | Already satisfied — run on the school Linux GPU PCs (same as the DRR pipeline). |
| 32×32 frozen grid too coarse for fine change maps | UPerNet-style multi-scale decoder; pyramid patch merging; optionally fuse a shallow high-res stem. |
| Frozen features miss subtle longitudinal change | Ablation F (unfreeze last block / LoRA); MIM-weighted CheXFound already emphasizes local detail. |
| Synthetic→real domain gap | Light head fine-tune on a small real slice; domain augmentation. |
| Checkpoint/license logistics | CheXFound is MIT; weights on Google Drive — vendor under `third_party/chexfound/` and pin the commit. |

---

## 8. Decisions to lock before coding

- [ ] Confirm access to CheXFound `teacher_checkpoint.pth` + config (Google Drive).
- [ ] `D_GLoRI` (start 768, matching paper) and `M` change-type queries (= `len(ANOMALY_TYPES)`).
- [ ] Decoder type: UPerNet vs. simple progressive upsampler (start simple, upgrade if needed).
- [ ] Feature-cache storage budget (1024×1024 tokens × pairs × last-4-layers — estimate disk).
