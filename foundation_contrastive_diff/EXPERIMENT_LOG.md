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

| Model / feature setup | Val Dice | Notes |
|---|---|---|
| raw cosine of ΔP (unsupervised, no training) | AUROC 0.62 | barely above chance |
| linear probe (769 params, last-1) | **0.27** | learned linear readout |
| D-GLoRI head, last-1, L1+Dice (run1) | 0.433 | |
| D-GLoRI head, last-1, L1+L2 (run_l1) | 0.449 | L1+L2 > Dice |
| D-GLoRI head, last-4, L1+Dice (run5) | 0.464 | |
| D-GLoRI head, last-4, L1+L2 | ~0.46 (in progress) | |
| DPT-style head, last-4, L1+L2 | **0.4647** | best reported epochs 3-12: epoch 11; run ongoing |
| Cross-attention + DPT, native last-1, L1+L2 | **0.5505** | exported validation report; best checkpoint epoch 39, threshold 0.05 |
| Cross-attention + DPT, native last-1, 90deg CW, signed mask, BG0.75 | **0.6002** | exported validation report; best checkpoint epoch45; prediction threshold0.5, original GT threshold0.05 |
| resolution ceiling (perfect 37² predictor) | 0.79 | from data_quality_check |
| overfit-16 (L1+L2) | 0.948 | capacity is NOT the limit |

Per-anomaly (run5, last-4, thr 0.05): consolidation 0.401, pneumothorax 0.448,
pleural_effusion 0.550, fluid_overload 0.468; overall 0.464; nuisance FP 0.020.

---

## Key findings / decisions

- **2026-10-17 - Clinical100 comparison-head inference prepared, not yet executed.**
  User requested trying the trained head on the100 clinician-annotated ICU pairs for
  qualitative direction. Added evaluation/evaluate_clinical.py and
  training/run_clinical100.sh, pinned to the verified epoch45 checkpoint SHA256 and
  threshold0.5. Strict coverage preflight confirms100 pairs,200 scans and500 annotations
  (Avi/Benny/Sigal/Smadar/Nitzan) locally; ambiguous/missing pair folders or scan counts
  stop the run, rather than silently skipping. No frozen encoder/head weights changed.
  There is NO trained foundation checkpoint in the local study tree; actual predictions
  require the campus best.pt. Campus clinical-data availability is not yet verified.

  Orientation check on real pair1 demonstrated legacy transpose differs from training
  CW rotation by a horizontal flip. Clinical inference therefore reproduces training's
  raw-array resize/minmax/rotation exactly, without adding the legacy transpose/crop or
  sharpening. Predictions are inverse-rotated and resized into current NIfTI coordinates;
  review panels transpose source-coordinate arrays for upright display. The actual
  model-input view is included separately. Affine and source shape are preserved in
  raw-score NIfTI exports; masks are thresholded at original resolution. Scan ordering
  deliberately matches the existing evaluator's sorted filenames, recorded per case.
  This does not independently prove chronological ordering or cross-cohort independence.

  Outputs will include100 ten-panel reviews, raw scores, signed masks, per-clinician
  signed Dice/directional recall, sign-specific3-of-5 consensus, empty-reference FP
  area, per-pair CSV and provenance (checkpoint/config and annotation hashes). Human
  ellipses are coarse regions, not dense pixel GT. No PAI claim: the study PAI function
  is a placeholder. Do not compare these Dice directly to synthetic Dice or old CC PAI.
  No threshold tuning, object filtering, new training or prediction results in this entry.

  Full real annotation parsing/rasterization passed. Legacy zero-radius ellipse warnings
  occur in16 pairs:4,5,9,13,23,29,31,37,45,48,51,53,57,58,73,87. Warnings are captured
  in per-case source provenance, preserving legacy raster semantics for review.
  Initial focused checks exposed a Windows case-insensitive duplicate-folder check and
  incorrect checkpoint-epoch metadata source; both fixed and checks rerun successfully.
  Final suite:33 tests passed in7.733s from the study CWD, including3 clinical regression
  tests (all rotations/asymmetric landmarks, signed/empty metric semantics, mocked
  end-to-end NIfTI geometry/provenance/export). Bash launcher syntax and editor
  diagnostics clean. Initial full-suite run from repository root failed existing
  relative-config lookups; rerun from required study directory passed without changes.
  Actual clinical model inference remains pending on campus; mock outputs were temporary
  test fixtures and are NOT clinical findings. Code is not committed/pushed here.
  Noninteractive SSH to the known campus node cas703 was attempted with BatchMode and
  strict host-key checking; Windows could not resolve the hostname. No credential prompt
  or remote execution occurred. Publishing approval was unavailable, so no commit/push
  was performed; the user must publish/transfer the scoped implementation before launch.

- **2026-10-17 - User flagged faint GT in the rotated-mask review.**
  Exact case train_10887_a_2/pair0/variant0, panel0c9745237261bcb7: GT peak0.05740356,
  GT area0.0002326965 =61/262144pixels (0.02327%); prediction mask14pixels,
  peak0.6153715, no GT overlap (Dice~1.33e-8). Verified exporter: top Signed GT and
  GT zoom panels plot ORIGINAL continuous cached GT, NOT the training signed mask;
  fixed [-1,1] and shared zoom+/-0.615 make the0.057peak faint. Bottom Signed GT mask
  uses where(abs(gt)>0.05,sign(gt),0) and is saturated +/-1 but spatially tiny.
  Verified change_map_loss and its training call: signed_mask conversion is performed
  inside loss, so training does NOT receive faded0.057-valued foreground targets.
  Main concern is threshold-dependent target support: a cutoff only just below the
  peak retains very few pixels. No display bug in mask amplitudes was found. Recommend
  clearer continuous-GT/training-target labels and cropped mask views; assess GT-threshold
  sensitivity separately before changing target definitions. No plotting or model code
  changed. This case was inspected from the user's attached panel plus exported metrics.

- **2026-10-17 - Rotated signed-mask validation export reviewed locally.**
  Artifacts: `review_outputs/rot90cw_mask_fp075_val_best_20261017_102112`.
  Metadata confirms best checkpoint epoch45, SHA256
  `0b7b49cc916806453ae5a2348e8b0546dd59dfbe5ff0449c034ba43d214a8053`, rotation3,
  signed_mask targets, prediction threshold0.5, GT threshold0.05;2108 evaluated pairs,
  338 exported panels. Report Dice0.6001953754, IoU0.5031012869 on1293 pathology pairs.
  Compared every pair with the previous native-last1 review: identical pair-ID sets,
  identical GT areas and identical manifest hash. These checks establish matched
  exported pairs, not an independent audit of source-patient split leakage.

  | Pathology | n | Old Dice | New Dice |
  |---|---:|---:|---:|
  | Consolidation | 313 | 0.416202 | 0.492340 |
  | Pneumothorax | 377 | 0.636580 | 0.688640 |
  | Pleural effusion | 294 | 0.636367 | 0.682573 |
  | Fluid overload | 309 | 0.499961 | 0.523160 |

  For1237 nonempty-GT pathology pairs, mean Dice0.543122 ->0.601498. GT-area bins:
  <=1% image (n207):0.048512 ->0.119054;1-5% (n319):0.389334 ->0.473113;
  >5% (n711):0.756121 ->0.799558. GT-peak0.05-0.1 (n241):0.089883 ->0.157605;
  0.1-0.2 (n567):0.549014 ->0.621431;0.2-0.4 (n418):0.787603 ->0.823471;
  >0.4 (n11):0.879161 ->0.864306. Empty thresholded predictions191 ->93; near-zero
  Dice<1e-5 cases240 ->142. Pairwise improvement/worsening/tie at tolerance1e-5:
  717/399/121; variants are correlated, so these are descriptive counts, not independent
  significance tests. Positive/negative/mixed Dice0.556944/0.502432/0.589475 ->
  0.619642/0.551651/0.656029. Of93 remaining empty predictions,45 have peak<0.001,
  60<0.01,82<0.1,88<0.3; a sweep down to0.1 cannot recover most complete misses.
  Empty predictions by type: consolidation34,pneumothorax14,effusion12,fluid33.

  Across815 controls, FP mean0.2395% ->0.1298%, p951.3332% ->0.6176%,
  max17.3973% ->8.0521% of image area. Controls exceeding1% FP50 ->26, BUT entirely
  empty controls643 ->635: fewer large FP regions, not fewer controls with any FP.
  Different score semantics/thresholds and three simultaneous recipe changes prevent
  matched-calibration robustness or isolated loss/rotation causal claims.

  Visually inspected input_preview plus11 actual case panels, not all338. Case stems:
  94f1c03858bf06e9 (near-zero consolidation miss), eff02aa2adb6c777 (negative localization
  with coarse smooth support),83b6196883d30886 (mixed signs partly localized, peripheral
  negative support missed),c12e42a257417945 (broad pneumothorax support, some missing
  contour and tiny opposite-sign artifacts),f4d045c4b8ab9c3b (good basal negative
  effusion localization),3ed8b865077b0b52 (high-overlap consolidation),c281aa5441f64ed3
  and aa48afe44cc6f7de (fluid signs captured, smooth masks miss patchy target detail),
  0eec5562d8bac8b0 and29eb5a9714201de5 (large nuisance FPs),e376d93a8b89e32b
  (device-rich control without thresholded FP). Preview is upright and aligned.
  Some zero-GT controls show prominent visible opacity differences; source-pair
  construction/projection effects need checking before attributing FPs solely to
  devices or calling these label errors. This review does not establish that explanation.

  Recommendation: keep this checkpoint as the new baseline. Next compare validation
  thresholds with small-region recall and nuisance tails, then one controlled
  small-region-aware loss ablation (e.g. modest sign-separated Dice with binary targets)
  while keeping the encoder frozen and the head/rotation/targets fixed. No retraining,
  cleanup or threshold selection was performed here. CC filtering cannot repair
  near-zero misses or the large coherent nuisance blobs. Test remains held out.

- **2026-10-17 - Legacy Itamar loss and prediction cleanup audited.**
  User asked whether Itamar removed small objects or refined predictions. Findings refer
  to the checked-in legacy implementation, not independently recovered historical run
  settings or proof that every existing prediction folder used this script.
  `python_files/longitudinal_MIM_training.py:363` computes mean L1 on outputs and GT;
  line374 computes mean MSE on BOTH tensors multiplied by128. With current
  `python_files/constants.py:101` flags and unit lambdas, the objective before gradient
  accumulation division is mean(abs(error)) +16384*mean(error**2), NOT equally scaled
  ordinary L1+MSE. Perceptual/style, SSIM, Fourier and GAN flags are off. No explicit
  foreground/background, Dice, component-area or sign weighting in the active loss.
  `python_files/datasets.py:1040` loads continuous DRR diff_map directly, and the active
  __getitem__ returns it; no component removal or signed-mask conversion there. This
  does not establish how the on-disk synthetic GT was originally generated.

  `python_files/archive/evaluation/Evaluation/Prediction.py:247` postprocess, in order:
  zero abs(output)>0.75 (not clipping); iterate ten overlapping width0.05 intensity
  bands with starts [-0.15,-0.12,-0.09,-0.06,-0.03,0,0.03,0.06,0.09,0.12], zeroing
  any current band occupying >25% of512x512; remove nonzero components of <=250pixels;
  then zero the open interval (-0.0075,0.02). Components use8-connectivity on nonzero
  values, so adjacent opposite signs merge. The band operations are sequential and
  can suppress weak values beyond the final dead zone. Postprocess runs BEFORE
  output.nii.gz is saved (lines366-371), not merely on plots. Input segmentation is
  used for a padded bounding-box square crop, not a hard lung-output mask.

  Separately, `Observer_Variability_sq_crop.py:223` inverse-maps saved output and filters
  positive/negative components independently: defaults reject area<50 or mean absolute
  intensity<0.03 in original-image coordinates. This changes evaluation masks, not the
  saved continuous output. Setting both thresholds to0 (no_cc) disables ONLY this
  evaluator filter; it cannot restore pixels already removed before saving.
  Executed the actual two prediction helper functions via AST extraction under the
  workspace venv:100/250pixels removed,251retained; touching125positive+126negative
  pixels survive together;300x300 at0.1 removed by band rule;0.9 region zeroed. Verified
  MSE scaling identity numerically. All assertions passed. No model code changed.
  Decision: do not copy these amplitude rules to signed-mask scores (confident values
  near +/-1 would be erased). Component removal cannot recover our small-object misses;
  any cleanup comparison must retain raw predictions and be tuned on validation only.

- **2026-10-17 — Campus rotated-mask run completed; reported best validation Dice 0.6002.**
  User supplied epochs43-60 and `[RQ1] done. best val dice = 0.6002`, followed by
  successful background-job completion. This establishes the run was executed on campus;
  local review artifacts/checkpoint/complete metrics have not yet been retrieved.
  Best shown epoch **45**: Dice **0.6002**, IoU **0.5031**, sens+ **0.661**, sens- **0.633**,
  mean nuisance FP area **0.0013** (0.13% image area), p95 **0.0062** (0.62%),
  max **0.0805** (8.05%); changed-region L1/L2 **0.2559/0.2493**.
  Epoch60: Dice **0.5955**, IoU **0.4998**, sens+ **0.645**, sens- **0.611**,
  mean/p95/max FP **0.0010/0.0039/0.0533**; changed-region L1/L2 **0.2738/0.2685**.
  The full training best epoch still needs checkpoint metadata confirmation; an earlier
  epoch could have tied the reported rounded best. Review best.pt, not last.pt.

  | Epoch | Train total | Val Dice | sens+ | sens- | Nuisance FP mean | p95 | max |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 43 | 0.0484 | 0.5984 | 0.670 | 0.633 | 0.0013 | 0.0061 | 0.0733 |
  | 44 | 0.0475 | 0.5977 | 0.669 | 0.628 | 0.0012 | 0.0056 | 0.0701 |
  | 45 | 0.0469 | 0.6002 | 0.661 | 0.633 | 0.0013 | 0.0062 | 0.0805 |
  | 46 | 0.0457 | 0.5986 | 0.660 | 0.631 | 0.0012 | 0.0048 | 0.0745 |
  | 47 | 0.0460 | 0.5974 | 0.651 | 0.621 | 0.0011 | 0.0043 | 0.0668 |
  | 48 | 0.0447 | 0.5961 | 0.650 | 0.622 | 0.0011 | 0.0043 | 0.0680 |
  | 49 | 0.0445 | 0.5974 | 0.655 | 0.620 | 0.0011 | 0.0040 | 0.0700 |
  | 50 | 0.0447 | 0.5969 | 0.653 | 0.617 | 0.0011 | 0.0041 | 0.0615 |
  | 51 | 0.0436 | 0.5951 | 0.646 | 0.618 | 0.0011 | 0.0040 | 0.0607 |
  | 52 | 0.0437 | 0.5965 | 0.648 | 0.615 | 0.0011 | 0.0041 | 0.0605 |
  | 53 | 0.0431 | 0.5965 | 0.646 | 0.613 | 0.0010 | 0.0042 | 0.0568 |
  | 54 | 0.0430 | 0.5967 | 0.646 | 0.612 | 0.0010 | 0.0040 | 0.0554 |
  | 55 | 0.0431 | 0.5957 | 0.646 | 0.611 | 0.0010 | 0.0038 | 0.0539 |
  | 56 | 0.0430 | 0.5957 | 0.645 | 0.612 | 0.0010 | 0.0041 | 0.0552 |
  | 57 | 0.0437 | 0.5955 | 0.645 | 0.611 | 0.0010 | 0.0039 | 0.0539 |
  | 58 | 0.0419 | 0.5954 | 0.645 | 0.611 | 0.0010 | 0.0039 | 0.0532 |
  | 59 | 0.0427 | 0.5955 | 0.645 | 0.611 | 0.0010 | 0.0039 | 0.0533 |
  | 60 | 0.0417 | 0.5955 | 0.645 | 0.611 | 0.0010 | 0.0039 | 0.0533 |

  Versus the previous exported best0.5505348, reported best improves by **0.0496652**
  (~4.97 percentage points, ~9.02% relative). Original GT support remains0.05, but
  prediction thresholds differ (old0.05/new0.5), output semantics changed, and rotation,
  target masking and background weight changed together. This supports the combined
  recipe at its chosen operating point, NOT isolated loss or rotation causation.
  Lower new FP area is descriptive, not a matched-calibration nuisance robustness claim.
  Late training decreases train loss and FP area while signed recall and changed-region
  validation reconstruction worsen: consistent with conservative drift/overfitting,
  not proof of a unique mechanism. More epochs are not the next recommendation.
  Printed training dice is an inactive loss component, not validation Dice.

  **Next:** export validation review from best.pt using checkpoint rotation/thresholds,
  with the large stratified review budget. Include input_preview.png, metrics.csv and
  run_config.json (not weights/cache). Compare small/faint-change misses, consolidation,
  mixed signs and high-FP controls with the old run before a new loss or architecture.
  A validation-only threshold sweep with original GT support fixed is a subsequent
  calibration option. Keep test held out. No new training changes or job launched.

- **2026-10-16 — Prepared requested combined run: upright scans, signed-mask GT, modest FP discount.**
  User requested a new run combining rotated scans, +/-1 GT and slightly less FP
  punishment. Config: `configs/last1_rot90cw_mask_fp075.yaml`; launcher:
  `training/run_rotated_mask.sh`. This is a combined recipe, not evidence isolating
  a loss effect. Campus launch has not been executed from this local Windows session.
  A three-way visual comparison of an exported source image established **90 degrees
  clockwise (rotation_k=3)** as upright. Apply that same transform to both scans and
  continuous GT before frozen extraction; no interpolation or left/right flip is added.
  Native GPU cache: `/cs/labs/josko/sahar_aharon/fcd_cache_last1_rot90cw_inference`.
  Run: `/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_fp075`.
  Cache contains continuous GT plus rotation tags; native extraction writes atomically,
  records manifest/layers/orientation and exports an actual input preview. Dataset,
  launcher and resume guards reject incompatible rotated caches/checkpoints.
  Training converts `abs(GT)>0.05` to sign(GT), otherwise zero. Signed L1+L2 uses
  foreground **11** and background **0.75** (25% background-only reduction, not the
  earlier proposed region-normalized loss). No-change pairs retain a nonzero penalty.
  Keep frozen native last-one RAD-DINO, the same 2,060,033-parameter cross-attention
  head, optimizer/splits and 60 epochs; seed42; Dice/sign weights remain zero.
  Prediction threshold **0.5**, original GT support **0.05**, stored in checkpoint
  metadata and reused automatically by evaluation/review; source panels rotate too.
  These outputs are signed change scores rather than calibrated intensity maps.
  Validation logs/checkpoints additionally track nuisance FP p95/max. Inspect inputs
  and early validation maps by epochs3-5; leave test untouched. Do not add more changes
  yet; use later separate ablations for attribution and threshold calibration.
  **Validation:** 30 CPU tests passed, including real NIfTI aligned rotation, signed
  target/FP weighting, training/validation target consistency, native cache guards,
  and end-to-end legacy plus rotated review export. Both launchers pass bash -n.
  Config parsing resolves rotation3/target=signed_mask/BG0.75/pred0.5/GT0.05.
  Real campus GPU/cache throughput and this run's learning quality remain unverified.

- **2026-10-16 — User proposes GT normalization/masking and a lower FP penalty; experiments not yet approved or run.**
  Motivation: weak/empty predictions may reflect conservative optimization. This is a
  hypothesis, not proof that the current FP term causes the misses. Recommended target
  ablation: signed ternary mask {-1,0,+1}, using an explicit support threshold (0.05
  would match the existing evaluation GT definition), rather than per-image max/min-max
  normalization. Zero must remain zero and no-change GT must stay empty. Mask targets
  change output semantics to change scores, not reconstructed DRR intensity; retain
  original continuous GT and a fixed reference mask for comparable evaluation.
  Recommended independent loss ablation: per-image foreground/background-mean signed
  reconstruction, starting with background coefficient 0.1 relative to foreground 1
  as a tunable hypothesis, not an established best value. Retain an explicit nonzero
  penalty on nuisance-only pairs. Do not train only inside the GT mask. A lower global
  background penalty also tolerates distant hallucinations; if boundary displacement
  is the issue, edge softness/spatial tolerance is a separate targeted option.
  Compare target-only and loss-only changes before combining them; keep encoder frozen
  and cache/head/splits fixed. Tune prediction thresholds on validation without changing
  the original GT reference, and report signed recall plus nuisance FP tails. No training
  code, config, cache, or checkpoint changed in response to this proposal.

- **2026-10-16 — Prediction/loss review: under-response is supported; loss-only causation is not established.**
  Reviewed all 2,108 rows of the epoch-39 validation CSV and visually inspected
  **12 selected panels** spanning all four pathologies, successes, failures, and two controls.
  This is not a visual review of all 338 panels. Selected panels are intentionally
  enriched for extremes; quantitative results below use the full validation CSV.
  Source: `review_outputs/last1_dpt_cross_attention_val_best_20261016_171153/`.
  No model, feature cache, loss, configuration, or held-out test data was changed.

  **Observed error pattern (threshold 0.05):**
  1,293 pathology-labelled pairs include 56 with no GT pixels above threshold.
  On the remaining **1,237 nonempty-GT pairs**, mean Dice is **0.5431**;
  **191 (15.4%)** have empty thresholded predictions and **240 (19.4%)** have
  effectively zero overlap (Dice < 1e-6). Empty prediction does not mean a zero-valued map.
  These rows cover repeated variants, not independent patients/experiments.

  | Nonempty-GT subset | Pairs | Mean Dice | Empty predictions |
  |---|---:|---:|---:|
  | GT area <=1% of image | 207 | 0.0485 | 128 |
  | GT area >1% and <=5% | 319 | 0.3893 | 55 |
  | GT area >5% | 711 | 0.7561 | 8 |
  | GT peak >0.05 and <=0.10 | 241 | 0.0899 | 124 |
  | GT peak >0.10 and <=0.20 | 567 | 0.5490 | 61 |
  | GT peak >0.20 and <=0.40 | 418 | 0.7876 | 6 |
  | GT peak >0.40 | 11 | 0.8792 | 0 |

  Area/intensity are related and defined from GT at the evaluation threshold, not
  the training mask threshold. A joint check still shows an intensity gradient for
  area >5%: Dice **0.4188 / 0.7064 / 0.8034** for peaks <=0.10 / <=0.20 / >0.20
  (15 / 287 / 409 pairs). This is descriptive evidence, not an isolated causal effect.
  Median per-pair prediction/GT peak ratio: **0.8168** overall, **0.6834** consolidation,
  **0.7088** fluid overload, **0.8856** pleural effusion, **0.9481** pneumothorax.
  A global peak ratio does not establish correct spatial placement.
  On 970 single-sign, nonempty-GT pairs, mean unsigned GT recall is **0.54836**
  versus correct-sign recall **0.54679**. Unsigned recall was reconstructed from
  Dice and mask areas (negligible smoothing error). Thus thresholded sign reversal
  is not the main aggregate deficit on this subset; mixed-sign pairs need separate care.
  The exported wrong_sign_fraction counts even subthreshold opposite-sign values,
  so it must not be read as a thresholded direction-error rate.

  **Concrete panels (filenames under the source folder's cases/):**
  - `abaf17cd83edf328.png`: fluid-overload resolution; broad correctly signed response,
    peak **0.04869** versus GT **0.18054**; thresholded prediction empty, Dice ~0.
  - `153f403a42fa51b7.png`: fluid-overload resolution; roughly localized but smoother
    and weaker; predicted area **8.89%** versus GT **21.71%**, Dice **0.5631**.
  - `abd371b8b6a476d6.png`: consolidation appearance; reduced support and detail,
    predicted area **2.02%** versus GT **4.59%**, Dice **0.4985**.
  - `bbc2ae99c05ab107.png`: mixed consolidation; textured signed GT is largely missed,
    GT/pred peaks **0.2089/0.03284**, Dice ~0; attenuation alone need not explain it.
  - `755dd28b383c02b7.png`: mixed pneumothorax; misses the negative region and gives
    faint positive responses; peak **0.04589**, Dice ~0. Not merely global gain.
  - `0bb99a810d41c6fa.png`: mixed pleural effusion; large off-target positive response,
    poor overlap with the negative GT region, Dice **0.0284**.
  - `d96dee84e1a18d8d.png`, `ce63e745db6c39dc.png`, `9e7fc242917b5203.png`,
    `f90aa7f118002274.png`: broad successful changes across the four pathologies,
    Dice **0.9467 / 0.9697 / 0.9560 / 0.9493**, respectively; fine texture is smoother.
  - `0d66731b9434938a.png`: worst nuisance control; FP area **17.397%**, not immunity
    to nuisance. `e376d93a8b89e32b.png`: random control, no thresholded detections.
    All 815 controls have zero GT: 643 have no detections; 50 exceed 1% FP area;
    mean FP area **0.2395%**, 95th percentile **1.3332%**. Monitor tails, not only mean.

  **Loss audit and executed checks:**
  Current config/call site uses signed L1+L2 with
  `mean((1 + 10 * (abs(gt)>0.02)) * (abs(pred-gt) + (pred-gt)**2))`.
  Dice/sign coefficients are zero. Sign is already supervised by signed reconstruction.
  Small/weak targets have less aggregate influence despite 11x changed-pixel weighting;
  L2 adds little relative to L1 for small residuals. A CPU check of the actual function
  gave zero-prediction loss **0.009504** for 1% area at GT=0.08, versus **0.264**
  for 10% area at GT=0.20 (**27.78x**). A correctly localized half-amplitude prediction
  at 0.04 gives loss **0.004576** but an empty mask at evaluation threshold 0.05.
  Perfect reconstruction gives zero active loss: do not claim zero output is its optimum.
  The inactive existing Dice compares abs(pred) to a **binary** GT>0.02 mask;
  at perfect reconstruction of the 0.08 target it still gives **0.851852** loss.
  Enabling it unchanged can push magnitudes toward one and conflict with continuous
  target calibration. Its mere presence is not evidence of a suitable localization term.

  **Important competing explanation: input orientation.**
  Reviewed source CXRs are sideways. `_load_nii_2d` uses squeeze without NIfTI affine
  reorientation; `_to_chw` only reshapes/resizes. Native cache generation passes these
  tensors directly to the backbone, whose `_prep` repeats/resizes/normalizes without
  rotation. The review uses the same dataset arrays and imshow without rotation.
  Thus the current code path does not explain the sideways views as a display-only fix.
  Historical cache input pixels were not available locally for direct inspection;
  verify an actual encoder-input preview on campus before choosing an orientation fix.
  A frozen encoder's representation, coarse feature grid, smoothing, and nuisance
  robustness remain alternatives to loss-only causation; no benefit from rotation is proven.

  **Recommended next decision, not implemented:**
  First check actual encoder-input orientation. Use the existing checkpoint for a
  validation-only prediction-threshold/gain sweep while holding GT threshold fixed
  at 0.05, tracking signed recall and nuisance FP tails as well as Dice. This requires
  raw maps or rerunning the head on campus: the CSV/PNG export cannot supply an exact sweep.
  Then isolate a loss ablation with the same frozen encoder/cache/head/splits and matched
  initialization/training budget: signed L1+L2 averaged separately over foreground and
  background per image, with fixed documented coefficients and background-only handling
  for no-change pairs. Keep a real background penalty. Normalizing by the sum of pixel
  weights alone does not give region-balanced means. Do not combine orientation changes,
  a new loss, and an architecture change in one comparison. Do not enable the existing
  raw-magnitude Dice or prioritize a larger sign penalty based on these results.
  Runtime assertions passed for CSV/report counts and Dice, analytical loss examples,
  the Dice calibration counterexample, and existence of all 338 referenced panel files.

- **2026-10-16 — Campus review push succeeded; artifacts and exact validation results verified locally.**
  Campus first push successfully published 87a5dba2. Subsequent authentication failures
  came from explicitly disabling the working credential helper; the later git -C ..
  command ran from the repo root and incorrectly targeted its parent. Neither invalidates
  the successful push; no regeneration/recommit needed. Local pull reports up to date
  and the review folder last1_dpt_cross_attention_val_best_20261016_171153 is present.
  Metadata: **epoch 39 best.pt**, threshold **0.05**, **2,108 validation pairs**, **338 panels**.
  Report: change-only **Dice 0.5505348**, IoU **0.4633572** on 1,293 pathology pairs;
  nuisance mean FP area **0.00239527** on 815 controls (0.2395% mean image area).
  Per-pathology Dice: consolidation **0.4162**, pneumothorax **0.6366**,
  pleural_effusion **0.6364**, fluid_overload **0.5000**. Consolidation sens+/sens-
  **0.4150/0.3253**. Checkpoint SHA256 begins c3ef61f0722f6709; full value in metadata.
  Opened two consolidation/negative source/GT/prediction panels: selected high-Dice
  case 0.947 has broad signed overlap but smoother prediction; selected zero-Dice
  case misses thresholded GT regions and adds opposite-sign off-target regions.
  These illustrate possible errors, not a systematic review or causal diagnosis.

- **2026-10-16 — Export a broad validation review into the repository for campus-to-local Git transfer.**
  User requests repo-local artifacts and enough examples to avoid repeated small exports.
  Existing evaluator gains optional --review: all validation-pair metrics plus deterministic
  low/middle/high-Dice and random cases per pathology/GT-sign group (24 each), and 48
  highest-FP/random nuisance controls. Distinct change groups within each review group
  limit duplicate variants; small groups can yield fewer examples. Typical positive/negative
  coverage across four pathologies targets up to 240 panels; mixed groups can add more.
  Panels show actual prior/current images, fixed-scale signed GT/prediction, matched
  amplitude zooms, signed masks and current-image overlays. Repo-relative contact sheets,
  index, CSV/JSON and loaded-checkpoint/manifest SHA256 provenance support offline review.
  Complete-cache preflight and nonempty-output rejection guard against partial/mixed reviews.
  The full split is scored once; selected cases alone are rerun for rendering, without
  retaining all validation tensors. No model weights/caches/NIfTI files are exported.
  Commands write to foundation_contrastive_diff/review_outputs/<timestamped run>, then
  stage/commit/push only that folder from campus; pull here for inspection. Keep test held out.
  Existing 21 head/cache tests plus four review tests pass locally (25 total), including
  actual temporary NIfTI/checkpoint-to-PNG rendering and deterministic selection.
  Actual cluster review images, counts and runtime remain unverified until the user runs it.

- **2026-10-16 — By epoch 55, best reported validation Dice is approximately 0.55.**
  User reports the native-last-one cross-attention run has reached epoch 55 with
  best validation Dice **0.55**; this is not necessarily the epoch-55 score. Best
  checkpoint epoch, exact metric values, sensitivities, nuisance FP and timing remain
  unavailable. Next decision should follow qualitative validation-output review,
  not an immediate architecture/loss change. Use best.pt at the existing threshold
  0.05; keep test data untouched while making development decisions. Existing
  evaluator provides per-pathology metrics, nuisance FP and one GT/prediction
  example per pathology, but no source images or representative failure sampling.
  Review source-image context and both change signs as well as nuisance controls
  before attributing remaining errors to localization, amplitude, direction or
  nuisance leakage. Documented a first-pass validation export; no cluster evaluation
  or model-output inspection has been performed by the assistant.

- **2026-10-16 — Single-layer cross-attention reaches approximately 0.49 by epoch 8.**
  User reports epoch 1: **0.28**, epoch 2: **0.34**, epoch 8: **0.49** for the current
  native-last-one GPU-cache cross-attention run. Interpreted as validation change-only
  Dice from the conversation; metric identity, exact values and full logs have not been
  independently checked. Reported epoch-8 value exceeds the previous last-four DPT's
  best supplied 0.4647 by approximately 0.0253 (2.53 percentage points). This does not
  isolate cross-attention: feature-layer count and native versus sliced/hidden-state
  representation differ as well. Epochs 3-7, other metrics, checkpoint status and
  epoch timing are unavailable. Keep the current experiment unchanged while tracking
  validation Dice, signed sensitivities and nuisance false positives; no test-set claim.

- **2026-10-16 — Native last-one GPU cache generation initializes on cas703.**
  User reports successful pull through c80d0ecc and **24.8 / 25.3 GB free GPU memory**
  before launch. Background shell job 2323798 prints
  `[cache] 20755 pairs -> /cs/labs/josko/sahar_aharon/fcd_cache_last1_inference on cuda (last_n_layers=1)`.
  This confirms CUDA/native last-one cache initialization, not a completed batch,
  measured throughput, cache completion, or training startup. The slow-image-processor
  message is a Transformers warning, not a reported failure; no processor setting changed.
  Cache progress is expected every 200 pairs with batch size 8. Keep this job running;
  the chained training command starts only after cache generation succeeds.

- **2026-10-16 — Switch single-layer cache creation to frozen GPU inference.**
  User requests stopping last-four-to-last-one CPU conversion and generating native
  last-one features directly from images. Reuse the existing cache CLI with
  `--last_n_layers 1 --device cuda --batch_size 8 --num_workers 4`; no encoder
  fine-tuning. Config and documented launch now use `fcd_cache_last1_inference` and
  `run_last1_dpt_cross_attention_inference`, separate from partial sliced files and
  previous checkpoints. Native extraction reads `last_hidden_state`, while slicing
  used `hidden_states[-1]`; normalization equivalence is not established. Preserve
  both old caches and logs. Documented exact-process SIGTERM targets only the earlier
  conversion; its nonzero exit prevents the old `&&`-chained training from starting.
  Already-started training requires separate identification. Direct inference's
  existing-file skip and non-atomic saves require care if interrupted. CPU conversion
  versus GPU inference throughput is unmeasured; this switch is the user's operational
  choice, not a measured speedup. No remote process stopped or GPU job launched locally.
  Local checks pass for native-cache CLI flags, isolated YAML/run paths, Bash syntax
  for stop/launch blocks, and the conversion-only process-matching expression.

- **2026-10-16 — Single-layer launch blocked before job execution by log redirection.**
  User's cluster pull succeeded through a1e2e6c2. GPU query reports **24.8 / 25.3 GB free**.
  Background job 2320567 exits 1 with zsh `no such file or directory` while opening
  `run_last1_dpt_cross_attention/train.log`; no cache-conversion/training startup is
  demonstrated. Missing log plus `>>` failure despite a preceding mkdir is consistent
  with zsh APPEND_CREATE disabled; directory existence/permissions were not independently
  checked. Remedy: explicitly mkdir and touch the log before append redirection, preserving
  any existing contents. Stop if either fails. An unmatched quote subsequently entered
  `dquote>` mode; the supplied prompt shows that state was cancelled (exit 130).
  Corrected launch documentation; do not treat the failed shell job as a training failure.

- **2026-10-16 — Last-four DPT reaches epoch 15; recommend stopping to reallocate compute.**
  User reports epoch-15 validation Dice **0.458**. Epochs 13-14 and other epoch-15
  metrics are unavailable; best previously supplied is epoch 11 **0.4647**, not an
  independently verified best over the entire run. Recommend stopping after confirming
  a completed checkpoint save, rather than waiting for the earlier epoch-20 reassessment.
  At the reported 45 minutes/epoch, five more epochs cost about 3 hours 45 minutes.
  This is a compute-budget decision in favor of testing single-layer temporal attention,
  not proof of overfitting, a plateau, or superiority of the new architecture.
  Preserve best.pt, last.pt, metrics.csv, run config and logs. Preserve the last-four
  cache at least until compact-cache conversion and verification have completed.
  Identify the exact old training PID before sending SIGTERM; do not kill unrelated
  Python processes. No cluster process has been inspected or stopped by the assistant.

  New config `configs/last1_dpt_cross_attention.yaml`: one 768-dimensional stream,
  shared projection to 96 channels, two bidirectional four-head attention blocks,
  signed fusion, existing decoder. Four decoder scales are derived from one stream;
  they are not four backbone layers. **2,060,033** trainable parameters. Metadata
  records `num_layers=1`; old four-layer checkpoint specs/defaults remain compatible.
  Batch 8/accumulation 4 and loss/optimizer settings unchanged; workers set to zero
  to avoid worker replication/prefetch overhead. Separate checkpoint/run directory.
  Existing cache CLI gains `--slice_last_layer_from`: CPU-only final-768 extraction
  from completed last-four files into `fcd_cache_last1_from_last4`. No encoder pass,
  source modification, or split regeneration. Compact cloned tensor storage, original
  dtype/CLS/GT/labels retained, per-file source provenance, atomic writes, and exact
  existing-file verification on resume. Missing source/mismatched target files fail.
  This is the final last-four stream, not a claim of normalization equivalence with a
  natively generated last-one cache. A distinct cache path prevents accidental mixing.
  Launcher now selects/audits one- or four-layer physical cache and reports disk bytes.
  RAM use is OS page caching of compact files, NOT explicit preload; cache residency,
  GPU batch-8 memory, and speed are unmeasured. Token bytes shrink fourfold, but GT
  and metadata do not. Cache conversion still needs additional disk space and reads
  the full source cache once (again when verifying files on a restarted conversion).
  Twenty-one CPU tests pass, including single-layer train/resume/eval/reload,
  attention/order gradients, exact compact cache conversion/resume/failure cases,
  and real launcher preflight for both dimensions with CUDA mocked. Full-size
  [1,1369,768] -> [1,1,512,512] plus [1,256] backward is finite. No cluster job launched
  or stopped; scientific performance and actual cache size remain unverified.

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
