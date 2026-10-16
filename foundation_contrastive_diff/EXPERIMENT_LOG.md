# RQ1 Experiment Log

Append-only log of experiments, results, and decisions for the foundation-based longitudinal
change-detection study (RQ1). Newest entries at the top of each section. The full chat
transcripts are auto-saved by VS Code under
`…/workspaceStorage/<id>/GitHub.copilot-chat/transcripts/*.jsonl`; this file is the curated,
human-readable record. **Every new experiment or decision must be appended here.**

RQ1: Can a FROZEN RAD-DINO + a lightweight D-GLoRI difference head produce a single signed
change heatmap [-1,+1] localizing longitudinal change (consolidation, pleural_effusion,
pneumothorax, fluid_overload) while ignoring nuisance (devices, projection angle)?

---

## Results ladder (val, change-only Dice unless noted)

| Model (reads the SAME frozen RAD-DINO features) | Val Dice | Notes |
|---|---|---|
| raw cosine of ΔP (unsupervised, no training) | AUROC 0.62 | barely above chance |
| linear probe (769 params, last-1) | **0.27** | learned linear readout |
| D-GLoRI head, last-1, L1+Dice (run1) | 0.433 | |
| D-GLoRI head, last-1, L1+L2 (run_l1) | 0.449 | L1+L2 > Dice |
| D-GLoRI head, last-4, L1+Dice (run5) | 0.464 | |
| D-GLoRI head, last-4, L1+L2 | ~0.46 (in progress) | |
| DPT-style head, last-4, L1+L2 | **0.4647** | best reported epochs 3-12: epoch 11; run ongoing |
| resolution ceiling (perfect 37² predictor) | 0.79 | from data_quality_check |
| overfit-16 (L1+L2) | 0.948 | capacity is NOT the limit |

Per-anomaly (run5, last-4, thr 0.05): consolidation 0.401, pneumothorax 0.448,
pleural_effusion 0.550, fluid_overload 0.468; overall 0.464; nuisance FP 0.020.

---

## Key findings / decisions

- **2026-10-16 — Optional temporal cross-attention DPT ablation implemented.**
  User proposes adding cross-attention after reading Lin et al. Hypothesis: comparing
  separate prior/current features before signed fusion may improve correspondence and
  nuisance rejection over subtract-first DPT. Not a demonstrated cause of the current
  validation plateau, and not evidence that DPT decoding itself is inadequate.
  `DPTCrossAttentionHead` projects each stream to 96 channels per cached layer, applies
  two bidirectional full-grid cross-attention blocks (four heads, direction-shared
  weights, residual paths and FFN), subtracts current-minus-prior, then reuses the DPT
  reassembly and signed decoder. Attention runs before upsampling, at 1369 tokens.
  Frozen RAD-DINO/cache unchanged. This differs from Lin's concatenation-based head.
  The old D-GLoRI implementation already supports an optional temporal cross-attention
  fusion mode; learned change-query attention alone is not the same operation.
  New head: **2,952,545** parameters vs DPT **2,057,825**. Config
  `configs/last4_dpt_cross_attention.yaml` matches DPT loss/optimizer/batch settings and
  uses a separate run directory. `HEAD_CONFIG` selects it in the existing audited launcher.
  Twelve CPU tests pass: both heads' training/resume, evaluation, shape/range,
  backward/checkpoint reload plus temporal-order/identical-stream and attention-gradient
  checks. Full [1,1369,3072] forward/backward yields [1,1,512,512] and [1,256], with
  finite gradients and nonzero gradients in every attention block. Intermediate signed
  features reverse under swapping, but final-output antisymmetry is not enforced.
  GPU batch-8 memory/speed and scientific performance are unverified. No cluster run
  started, no active experiment stopped, and no superiority claim is made.

- **2026-10-16 — Literature discussion: frozen features plus learned change heads.**
  User asks for precedents for training a comparison/difference head on pretrained latent
  features, preferably without changing the encoder. Primary sources checked:
  - Lin et al., *Robust Scene Change Detection Using Visual Foundation Models and
    Cross-Attention Mechanisms* (2024 preprint; v2 February 2025): frozen DINOv2
    ViT-S/14, two directional full-image cross-attention comparators, concatenation,
    3x3/1x1 convolutional prediction head and upsampling. Binary changed/unchanged
    segmentation on street scenes, including viewpoint variation. Closest verified
    pure downstream-head precedent: https://arxiv.org/html/2409.16850v2
  - Ding et al., *Adapting Segment Anything Model for Change Detection in HR Remote
    Sensing Images* (2023 preprint; TGRS 2024): SAM-CD uses frozen FastSAM, external
    feature adaptors, multiscale U-Net-like decoding and a temporal semantic-consistency
    auxiliary branch. Reports 2.49M trainable downstream parameters. Binary remote-sensing
    output, not clinical signed change: https://arxiv.org/html/2309.01429v4
  - Li et al., *A New Learning Paradigm for Foundation Model-based Remote Sensing
    Change Detection* (2023 preprint; TGRS 2024): BAN keeps its foundation model
    (default CLIP ViT-L/14) unchanged but also trains a separate image-processing
    Bi-TAB branch and bridging modules. Not a head receiving cached features alone:
    https://arxiv.org/html/2312.01163v2
  - Cheng and Hsu, *ChangeDINO* (ISPRS Annals 2026): frozen DINOv3 plus a lightweight
    trainable image stream, multiscale absolute-difference priors, differential
    transformer decoding and morphology refinement. Hybrid, not latent-only:
    https://isprs-annals.copernicus.org/articles/XI-3-2026/523/2026/
  - Karwande et al., *CheXRelNet* (MICCAI 2022): pretrained ResNet101-autoencoder
    local/global features plus anatomical graph attention for CXR change-relation
    classification. Relevant clinical precedent, not dense signed-map supervision.
    Paper does not unambiguously establish a strictly frozen encoder; public encoder
    code exposes a configurable `cnn_freeze` flag. Do not claim it is frozen by default:
    https://arxiv.org/html/2208.03873v2
  - Zheng et al., *Segment Any Change* (NeurIPS 2024): AnyChange uses SAM latent
    matching without training. Useful comparator, not a learned-head precedent:
    https://arxiv.org/abs/2402.01188
  AdaDINO (https://arxiv.org/html/2608.07982v1) keeps original weights frozen but
  trains modules inside the backbone; representations therefore change during training
  and cannot reuse our unchanged single-image feature cache. DPT remains a single-image
  dense-decoder reference, not evidence for frozen two-image clinical change detection.
  These sources support the general methodology, not a novelty claim or transfer of
  satellite/street-scene scores to CXR. No architecture or training settings changed.

- **2026-10-16 — Runtime-aware training advice supersedes the 60-epoch advice below.**
  User reports approximately 45 minutes per DPT epoch. Recommendation: continue the
  existing run unchanged to epoch 20, then reassess validation progress and compute cost.
  From epoch 12 this is about 6 hours, versus 36 hours to epoch 60. This is a suggested
  reassessment point, not a configured early-stopping rule or evidence of overfitting.
  Preserve the best validation checkpoint; cluster progress after epoch 12 is unknown.

- **2026-10-16 — DPT training update: user-reported epochs 3-12 of 60.**
  Best in the supplied excerpt: epoch 11, Dice **0.4647**, IoU **0.3668**,
  masked-change L1 **0.0314**, L2 **0.0018**, sens+ **0.501**, sens- **0.448**,
  nuisance FP area **0.0122**. Epoch 12 Dice **0.4536**, FP **0.0063**.
  Training loss falls 0.0664 -> 0.0450 (epochs 3 -> 12); validation Dice improves
  overall but fluctuates. The epoch-12 dip alone does not establish overfitting.
  DPT is currently around the earlier last-four D-GLoRI range (~0.46), not a demonstrated
  improvement. The prior last-four L1+L2 final best is still unavailable (previously
  reported epoch-12 Dice 0.4604). Continue the planned 60 epochs unchanged, retain the
  best validation checkpoint, and leave test evaluation until model selection is complete.
  The printed training `dice` is an unweighted loss component, not a Dice score;
  `w_dice=0` means it contributes nothing to the optimized L1+L2 objective.
  Source: pasted cluster output, not independently read from cluster checkpoints/CSV.

  | Epoch | Train total | Val Dice | Val IoU | L1 change | L2 change | Sens+ | Sens- | Nuisance FP |
  |---|---|---|---|---|---|---|---|---|
  | 3 | 0.0664 | 0.3793 | 0.2890 | 0.0359 | 0.0023 | 0.361 | 0.373 | 0.0139 |
  | 4 | 0.0606 | 0.4127 | 0.3154 | 0.0342 | 0.0021 | 0.482 | 0.393 | 0.0177 |
  | 5 | 0.0576 | 0.4138 | 0.3112 | 0.0350 | 0.0021 | 0.554 | 0.476 | 0.0410 |
  | 6 | 0.0545 | 0.4285 | 0.3361 | 0.0338 | 0.0021 | 0.430 | 0.387 | 0.0098 |
  | 7 | 0.0527 | 0.4326 | 0.3369 | 0.0327 | 0.0020 | 0.434 | 0.436 | 0.0159 |
  | 8 | 0.0506 | 0.4323 | 0.3422 | 0.0335 | 0.0021 | 0.405 | 0.380 | 0.0080 |
  | 9 | 0.0485 | 0.4399 | 0.3426 | 0.0328 | 0.0020 | 0.499 | 0.398 | 0.0161 |
  | 10 | 0.0480 | 0.4482 | 0.3554 | 0.0325 | 0.0019 | 0.467 | 0.399 | 0.0067 |
  | 11 | 0.0464 | 0.4647 | 0.3668 | 0.0314 | 0.0018 | 0.501 | 0.448 | 0.0122 |
  | 12 | 0.0450 | 0.4536 | 0.3610 | 0.0321 | 0.0019 | 0.467 | 0.372 | 0.0063 |

- **2026-10-16 — Last-four cache rebuilt; DPT-style decoder comparison prepared.**
  User reports cache completion after checking 720 GB free. Cluster contents are not yet
  independently verified; the launcher checks manifest split disjointness and full file
  coverage, samples feature shapes, and records the manifest SHA256 before training.
  New `DPTDifferenceHead`: four separate signed token-difference streams (768 channels
  each), learned reassembly to 148/74/37/18 grids, 96-channel residual top-down fusion,
  final CLS-difference conditioning, Tanh signed 512x512 output and 256-d embedding.
  Trainable parameters: **2,057,825**. These are adjacent final four ViT blocks, not the
  widely spaced taps of canonical DPT. Reassembly adds no new image samples.
  `configs/last4_dpt.yaml` matches last4 L1+L2: 60 epochs, batch8/accum4, AdamW lr3e-4,
  balanced sampling, threshold .05, same existing patient splits. Separate `run_last4_dpt`
  outputs; D-GLoRI remains default. Checkpoints include architecture metadata for resume/eval.
  CPU full-size forward/backward and checkpoint reload passed; five persistent tests
  cover decoder gradients, config/default-head compatibility, evaluator slicing, and
  cached trainer save/resume. The resume test exposed and fixed stale `best_dice` in
  `last.pt` (best score is now updated before saving). Bash syntax and synthetic preflight
  acceptance/leak rejection passed; preflight CUDA calls were mocked for that local check.
  Local project venv: torch 2.10.0+cpu (system Python's torch import has a missing DLL).
  GPU memory/runtime and
  validation performance remain unmeasured; no improvement is assumed. Test split remains
  held out from training/model selection (launcher only checks metadata/cache integrity).
  Interpretation correction for older entries below: pooled cosine/probe rankings do not
  establish universal SOTA, clinical blindness, or a causal pretraining-objective effect.
  Synthetic-to-real score differences do not quantify leakage; external pretraining overlap
  is unverified. Patch-averaged-GT reconstruction is a reference, not an information ceiling.

- **2026-10-15 — REAL-PAIRS TEST (clean arbiter, leakage-free): RAD-DINO wins again.**
  135 real radiologist-annotated pairs (ICU + PNIMIT), GT = union of doctors' change ellipses,
  cos AUROC: **rad_dino 0.614** > biomedclip 0.580 > torchxrayvision 0.522 > pubmedclip 0.505 >
  dinov2 0.497 > sam 0.483 > clip 0.466. Robust: RAD-DINO is #1 on BOTH synthetic (0.586) AND
  real (0.614) — and HIGHER on real (it's trained on real XR; DRRs were mildly OOD for it).
  torchxrayvision COLLAPSED 0.647→0.522 (vindicates the grid-7 AUROC-inflation caveat).
  BiomedCLIP stable (0.579≈0.580) = genuine 2nd. General models (dinov2/sam/clip) ≈ chance on real.
- **2026-10-15 — ITAMAR on real pairs (confound quantified, nuanced result): 0.660.**
  Synthetic 0.731 → real **0.660** (−0.071 drop = the leakage+DRR-domain confound, real & measured).
  BUT a GENUINE advantage survives: on leakage-free real CXRs where Itamar is OUT-of-domain
  (DRR-trained), he STILL beats RAD-DINO **0.660 vs 0.614 (+0.046)**. Fair comparison (chg% 17.4
  @grid32 ≈ 16.8 @grid37, not a grid artifact). ⇒ Reconstruction objective has a REAL, transferable
  advantage for longitudinal change over DINO invariance — not just a confounded oracle.
  ⇒ RQ1 ANSWER (off-the-shelf FROZEN foundation models): RAD-DINO is the best AVAILABLE foundation
  model (Itamar isn't one — small, in-house, CT-cohort-specific). But ~0.61 AUROC → motivates the
  adapter head. FUTURE DIRECTION: a reconstruction-pretrained FOUNDATION model (large external CXR)
  could beat RAD-DINO here.
- **2026-10-15 — HYPOTHESIS CONFIRMED: Itamar's reconstruction encoder beats RAD-DINO.**
  On the fair 2.5k subset (grid-comparable), itamar_ae cos AUROC **0.731** vs RAD-DINO **0.586**
  (+0.145), probe **0.281** vs 0.261. The decisive per-class signal: **pneumothorax corr +0.04
  (RAD-DINO, blind) → +0.257 (itamar)**, effusion +0.377. A reconstruction (L1+L2) objective
  preserves the subtle intensity deltas that RAD-DINO's DINO invariance objective discards.
  ** ROLE: Itamar's model is an ORACLE / upper-bound reference, NOT an RQ1 answer** — it was
  trained ON this synthetic DRR family, so it is in-domain, not an off-the-shelf frozen foundation
  model. It only tells us WHAT to look for: prefer reconstruction/intensity-sensitive pretraining
  over DINO-style invariance.
- **2026-10-15 — CONFOUND on the Itamar oracle (two mechanisms), raised by user.**
  (1) **CT-level leakage:** fcd DRRs come from Itamar's CT cohort; if he trained on the CTs behind
  our VAL split (our val is held out from OUR head training, NOT from HIS encoder training), his
  features have seen that anatomy → inflates his change-sensitivity on our val.
  (2) **Domain match:** he trained on the SAME DRR rendering pipeline → in-domain; all sweep
  foundation models trained on REAL CXR → out-of-domain on synthetic DRRs, handicapping them.
  ⇒ Itamar's +0.145 is CONFOUNDED (objective + domain + possible leakage); do NOT use it as
  evidence that "reconstruction > DINO". The CLEAN test is whether EXTERNAL intensity-sensitive
  models (MedSAM/SAM) beat RAD-DINO in the sweep. The foundation-model sweep itself is leakage-free
  (none saw these CTs). TODO: check split_dataset.py provenance (CT-disjoint?) to see if a
  leakage-controlled re-probe of Itamar on unseen CTs is possible.
- **2026-10-15 — GOAL (restated): find a SOTA *frozen* CXR/medical foundation model (NO
  fine-tune, NO retrain) whose features expose change best.** Screen off-the-shelf candidates via
  cos+probe. Candidates: BiomedCLIP, RadImageNet, MedSAM/SAM-Med2D, CheXFound (needs ckpt),
  MedImageInsight (auth), Google CXR-Foundation/ELIXR (gated). Ranked bets: RadImageNet, MedSAM,
  CheXFound (intensity-sensitive / CXR-specific, non-DINO).
- **2026-10-13 — Plateau diagnosed as faint/entangled features, not head capacity or bad data.**
  All four loss×feature cells land 0.43–0.46. Overfit-16 = 0.95 (capacity fine), res_ceiling
  = 0.79 (data learnable). So the limit is how well the frozen features expose the change.
- **2026-10-13 — Feature-change correlation (diagnostic #2):** last-1 cos-AUROC **0.624**,
  last-4 **0.612** (≈ identical). → **Last layer is enough; drop last-4** (use `$CACHE1`, fast).
  Per-type: effusion strongest (+0.147), pneumothorax weakest (+0.049), `none` ≈ 0 (sanity ok).
- **2026-10-13 — Linear probe (diagnostic #1):** last-1 probe Dice **0.27** vs full head 0.449.
  Probe ≪ head → info IS in the features but nonlinearly entangled; head power converts to Dice.
  Interpretation: not a fully blind backbone, but faint features; a better head (DPT) is
  justified AND a better-suited backbone may fit our target better.
- **2026-10-13 — Decision: use the cheap probe+cos as a BACKBONE-FITNESS SCREEN.** Linear-probe
  performance is the standard SSL proxy for representation quality. Screen candidate backbones
  cheaply (cache a ~2–3k-pair subset → cos + probe), rank them, full-train only the winner.
  Calibration point: RAD-DINO probe 0.27 → head 0.449.
- **2026-10-13 — Sens metric fix:** sens+/- now averaged only over pairs that contain that
  direction (pure-appearance/resolution pairs were injecting spurious 0s). Eval-only; rerun eval
  on any best.pt for honest sens.

---

## Backbone-fitness screen (cheap cos + linear probe on frozen features)

Candidate backbones and the hypothesis for each:

| Backbone | Checkpoint / source | Grid×dim | Hypothesis |
|---|---|---|---|
| rad_dino (baseline) | `microsoft/rad-dino` (HF) | 37×768 | cos 0.62, probe 0.27 (measured) |
| itamar_ae | `/cs/casmip/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id32_Epoch8_…L1L2_GN.pt` | 32×640 | reconstruction-trained → preserves intensity → **should probe higher** |
| dinov2_imagenet | torch.hub `dinov2_vitb14` | 37×768 | control: DINO objective in general vs RAD-DINO specifically |
| torchxrayvision | `densenet121-res224-all` (pip xrv) | 7×1024 | supervised CXR classifier, intensity-sensitive |
| biomedclip / chexzero | open_clip / HF (optional) | — | vision-language inductive bias |

itamar_ae API: `LongitudinalMIMDeformV2` → `model.backbone.encoder(x)` → `encoded_bn` →
`[B,640,32,32]`; flatten to tokens `[B,1024,640]`, grid 32, no CLS (use global avg-pool).

Screen results (fill as they land):

| Backbone | cos AUROC | probe Dice | notes |
|---|---|---|---|
| rad_dino (full data) | 0.624 | 0.27 | full-data probe (16,464 train); full-head 0.449 |
| rad_dino (2.5k subset) | 0.586 | 0.261 | fair baseline; subset ≈ full (0.27→0.261) → screen trustworthy. L2 AUROC 0.633 > cos |
| dinov2_imagenet (2.5k) | 0.569 | ~0.20 (ep4) | **worse than RAD-DINO on both**; CXR pretraining helps. Nuisance FP high (0.19–0.47) |
| clip_vitb16 (2.5k, grid 14) | 0.510 | 0.152 | general CLIP ≈ chance; worst |
| pubmedclip (2.5k, grid 7) | 0.554 | 0.188 | biomedical CLIP; below RAD-DINO |
| biomedclip (2.5k, grid 14) | 0.579 | 0.174 | biomedical CLIP; biomedical > general CLIP (cos 0.51→0.58) but < RAD-DINO |
| sam_vitb (2.5k, grid 64) | 0.564 | **0.045** | SAM segmentation features USELESS for change (probe ≈ 0) |
| medsam (2.5k) | FAILED | — | HF `flaviagiammarino/medsam` 404 (repo moved); needs valid source |
| itamar_ae (2.5k, grid 32) | **0.731** (L2 0.653) | **0.281** | ORACLE (confounded, see above). recon encoder; pneumothorax +0.257 (vs rad 0.04!) |
| torchxrayvision (2.5k, grid 7) | 0.647 (L2 0.697) | 0.204 | highest raw AUROC BUT grid-7: AUROC inflated by coarse pooling, Dice crippled by low res. effusion +0.248 (supervised) |

**SYNTHETIC SWEEP VERDICT (2026-10-15):** among REAL frozen foundation models, **RAD-DINO wins**
(probe 0.264, cos 0.586). No general VLM (CLIP 0.152), biomedical VLM (PubMedCLIP 0.188, BiomedCLIP
0.174), or SAM (0.045, useless) beats it. Biomedical pretraining > general (CLIP cos 0.51 →
BiomedCLIP 0.58) but < CXR-specific RAD-DINO. → RAD-DINO is the best off-the-shelf choice on
synthetic. CAVEAT: all are out-of-domain on synthetic DRRs → the REAL-PAIRS test (radiologist
annotations, ~170 pairs) is the clean arbiter and may re-rank.

**Grid caveat (2026-10-15):** cos/L2 AUROC is only cleanly comparable at the SAME grid. xrv
(grid 7) pools ~5x more area per patch → AUROC inflates while Dice is resolution-capped. Clean
within-grid ranking (grid 37): rad_dino > dinov2. Supervised xrv features are change-sensitive
(esp. effusion) but resolution-limited — reinforces that the pretraining OBJECTIVE matters.

**Methodology note (2026-10-14):** probe column must use a FIXED subset size across backbones.
cos AUROC is training-free so comparable as-is. RAD-DINO subset probe 0.261 ≈ full-data 0.27,
so the 2.5k screen predicts full-data well.

**Control finding (2026-10-15):** on equal 2.5k footing RAD-DINO beats generic DINOv2-ImageNet
(cos 0.586 vs 0.569, probe 0.261 vs ~0.20) → RAD-DINO's CXR pretraining genuinely helps; but both
DINO-objective backbones are weak (cos 0.57–0.63), motivating the reconstruction-trained Itamar AE.

---

## Environment (cluster)

See `AGENT_CONTEXT.md` for the full per-session preamble. Key: `$CACHE1` (90 GB, fits RAM,
fast) is now the default; `$CACHE4` (360 GB, NFS-bound) deprecated since last-1 ≈ last-4.

- **2026-10-14 — Freed `fcd_cache_last4` (~360 GB).** Deprecated (last-1 ≈ last-4); deleted to
  make room for the backbone-screen subset caches (`fcd_screen/*`). `fcd_cache_last1` kept.
