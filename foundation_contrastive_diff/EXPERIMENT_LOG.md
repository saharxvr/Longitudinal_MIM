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
| resolution ceiling (perfect 37² predictor) | 0.79 | from data_quality_check |
| overfit-16 (L1+L2) | 0.948 | capacity is NOT the limit |

Per-anomaly (run5, last-4, thr 0.05): consolidation 0.401, pneumothorax 0.448,
pleural_effusion 0.550, fluid_overload 0.468; overall 0.464; nuisance FP 0.020.

---

## Key findings / decisions

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
| itamar_ae (2.5k, grid 32) | **0.731** (L2 0.653) | **0.281** | **WINNER.** recon encoder; pneumothorax +0.257 (vs rad 0.04!), effusion +0.377, fluid +0.257 |
| torchxrayvision (2.5k, grid 7) | 0.647 (L2 0.697) | 0.204 | highest raw AUROC BUT grid-7: AUROC inflated by coarse pooling, Dice crippled by low res. effusion +0.248 (supervised) |

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
