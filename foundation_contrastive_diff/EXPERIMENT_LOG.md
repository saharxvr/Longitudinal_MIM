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

- **2026-10-20 - Added component cutoff500; compared250/500 at fixed0.75.**
  Expanded exploratory sweep to44settings/model; all30baseline F1 values
  reproduced,3tests pass. Fixed threshold.75, mean clinician detection F1
  original/augmented: raw .333182/.329690; area250 .437635/.439917;
  area500 .461260/.457385; area1000 .476557/.488400.
  Area500 is close to but below historical Itamar100 mean.464630 at this
  operating point. Augmented500 best tested threshold.9 gives.474662, but this
  is a different exploratory threshold selection. Mean predicted area changes
  very little250->1000 (original13.34->13.25%, augmented13.99->13.88%):
  detection counts sensitive to tiny fragments, not solved broad false positives.
  Full report/results: results/clinical100_postprocessing_cc500_20261020.
  No clinically safe cutoff or independent-test improvement claimed.

- **2026-10-20 - Best-filtered clinical heatmap gallery rendered locally.**
  User requests scans/heatmaps after best tested filtering. Rendered all100pairs,
  both CE models, threshold0.75/sign-specific area1000 in
  review_outputs/clinical100_best_filtered_20261020/index.html.
  Each15panel image includes prior/current, consensus, continuous raw scores,
  threshold-only and filtered overlays for each model, removed-pixel comparison,
  and five clinicians. Prediction/annotation hashes checked; all30 rendered-mask
  directional/combined F1 values exactly reproduce sweep results. Three helper
  tests passed. Gallery is local-only; no source-scan panels uploaded to Git.
  First case illustrates complete removal of thresholded predictions; this is
  not proof of a clinically correct negative. Existing exploratory-tuning and
  small-change-loss caveats still apply.

- **2026-10-20 - Exploratory postprocessing sweep on both CE clinical100 maps.**
  User requested literal Itamar cleanup, other methods and thresholds against
  clinicians; confirmed both baseline and augmentation. Ran39settings/model:
  raw, sign-specific area50/250/1000, Gaussian1/2, Gaussian1+area250 at
  thresholds.1/.25/.5/.75/.9; actual legacy cleanup at0/.1/.25/.5.
  All input prediction and annotation hashes verified (explicit Windows newline
  normalization only). Reproduced all30 published baseline clinician F1 values.
  102 randomized/empty matcher comparisons reproduce reference-first legacy
  greedy matching;3unit tests passed. Best tested: threshold.75+area1000
  equivalent512pixels, mean clinician F1 .476557 original/.488400 augmented
  versus .314082/.321089 raw. Area1000 at unchanged.5 already .471767/.480313.
  Literal legacy function on restored native maps gives .264976/.260316 at0:
  not historical full model-space pipeline; high-score suppression unsuitable
  for CE probability differences. Best empty-consensus mean predicted area
  still8.01%/8.37%, max40.47%/46.41%. Largest tested cutoff wins: boundary
  result, not global optimum; potential loss of true small changes and broad
  any-overlap detections remain concerns. All results exploratory tuning on
  same reused100cases, no independent-test/statistical-significance claim.
  Full report/tables: results/clinical100_postprocessing_sweep_20261020_verified.

- **2026-10-20 - Implemented shared-backbone block6/9/12 ablation.**
  User requests implementation, Git push and campus launch command.
  New configs/layer_depth_6_9_12_aug_30.yaml and
  training/run_layer_ablation.sh: online frozen RAD-DINO, identical initialized
  independent cross-attention+DPT heads, own CLS at each depth, frozen final
  LayerNorm,30epochs/twoepoch warmup/cosine, batch16/accum2, existing augmentation
  and CE+classwiseDice. No cross-layer fusion. Sequential branch backward shares
  one encoder pass per image. Existing manifest/input preflight reused.
  Gated benchmark10warmup+100measured batches includes all heads/diagnostics,
  then clean validation timing sample; real training starts in a fresh process.
  Primary threshold sweeps narrowed to pred0.5 and GT0.001/0.05 to limit CPU
  overhead; per-pair and component diagnostics remain enabled every epoch.
  Atomic shared epoch-boundary snapshot stores all optimizer/scheduler states,
  RNG/generators, histories and best heads. Per-layer best/last inference
  checkpoints carry explicit backbone_layer/final_layernorm metadata; synthetic
  and clinical evaluators honor it. Resume rejects config/code/manifest/model
  identity mismatches. Nonfinite loss/output/gradients fail explicitly.
  Completion publishes only dedicated metrics/diagnostic/provenance artifacts;
  clinical inference is separate after synthetic-validation layer selection.
  Local8new tests pass: cached actual RAD-DINO weights confirm layer indexing,
  normalization/final-output parity, frozen extraction; miniature real DPT
  backward, independent heads, shared calls, branch-order/RNG/resume continuity,
  partial accumulation policy, atomic-save failure, benchmark/train/resume/
  publication lifecycle and selected-layer clinical inference. Campus GPU speed,
  memory and scientific results remain unmeasured; no campus job launched here.
  Final combined regression suite:76tests PASS83.790s; both launch scripts pass
  Bash syntax checks and scoped staged whitespace checks pass.

- **2026-10-20 - Augmentation best20 clinical100 results pulled and checked.**
  Campus publication bb606b75 contains
  results/clinical100_upright_ce_dice_thr0p001_aug_b16_20261020_131848.
  Verification records early-stop at completed epoch46/60, selected epoch20,
  synthetic best pooled signed Dice0.812495 versus baseline0.814512.
  Locally checked all100 signed-output hashes and300 probability hashes,
  probability geometry/range/sum and signed-score identity, and100 thresholded
  masks. Clinical source and500 annotation hashes match the baseline exactly.
  Published OV combined F1 baseline->augmentation:
  Avi .299795->.295409, Benny .297710->.304833,
  Sigal .305785->.321285, Smadar .308911->.327553,
  Nitzan .358209->.356364. Unweighted clinician mean .314082->.321089;
  positive .320778->.303580, negative .306636->.345320.
  Consensus signed pixel Dice .161313->.158062; positive .207159->.185962,
  negative .106521->.115858. Empty-consensus12 mean predicted FP area
  .092229->.097676. Mixed result: improved negative detection and modest mean
  component F1, not broad localization/false-positive improvement or evidence
  of statistical significance. Scores read from published reports; OV matcher
  not independently rerun in this check. Clinical cohort remains exploratory.

- **2026-10-20 - User authorizes stopping augmentation and clinical export.**
  User reports latest epoch45, best target-matched pooled signed Dice0.812495
  at20, latest0.807493. Original nonaugmented best0.814512: no demonstrated
  gain on this metric; clinical/nuisance comparison remains pending.
  User requests stop, best-checkpoint clinical100 prediction and Git publication,
  then a later layer-depth ablation. Added explicit --evaluate-only to the CE
  pipeline: no training/benchmark subprocess, unchanged60epoch provenance,
  validated best/last ordering and configuration, actual completed-epoch
  diagnostics, early-stop metadata, and both pipeline/training locks on CLI.
  Existing map/probability/F1 validation and scoped publication retained.
  Local project-venv tests: three targeted tests passed, covering early-stop
  export/publication, default completion guard and baseline/augmented pipelines.
  Campus stop/export has NOT been executed from Windows.

- **2026-10-20 - Proposed shared-backbone single-layer depth ablation.**
  User requested a plan for one job training independent heads at different
  depths, with fewer epochs. Proposed blocks6/9/12, identical initialized
  cross-attention+DPT heads, shared augmented inputs and frozen extraction,
  independent optimizers/checkpoints, and a fixed30epoch screening schedule.
  Apply the frozen final LayerNorm to each selected hidden state and use that
  depth's CLS token; require block12 parity with existing final output.
  Benchmark before predicting runtime. Existing60epoch augmentation run remains
  unchanged; new block12 head is the matched internal control. Full scope,
  validation gates and limitations: LAYER_DEPTH_ABLATION_PLAN.md.
  Planning only: no trainer changes, campus launch or measured speedup.

- **2026-10-19 - Campus online augmentation batch-size benchmark.**
  User stopped the initial augmentation run to benchmark sequentially on RTX4090.
  Each candidate ran32 batches: batch8/accum4 .457s/batch,17.50pairs/s,2.82GiB;
  batch16/accum2 .783s/batch,20.44pairs/s,5.27GiB;
  batch32/accum1 1.716s/batch,18.65pairs/s,10.17GiB. Select16/2: about17%
  more throughput than8/4 and about9% shorter estimated training pass (13.4min
  versus14.7min) than32/1. Earlier500-batch run took11.2min: do not extrapolate
  these short warmed benchmarks as guaranteed sustained epoch time or attribute
  the difference to a specific cause. Preserve old batch8 run and start a fresh
  `_aug_b16` run. Effective batch32 is unchanged; batchwise Dice/RNG and final
  drop_last behavior mean this is not numerically identical optimization.

- **2026-10-19 - Stop weight-decay ablation; prepare online augmentation run.**
  User confirmed cas703 launcher2787709 and trainer2787717 terminated (empty
  process listing). Preserve completed checkpoints/logs; automatic clinical
  export/publication did not run. Supplied wd.05 epochs37-42 show train loss
  .165601 -> .154543, validation loss .727785 -> .741115 and target-matched
  signed Dice .807289-.808999 (epoch42 .808457). Earlier supplied epoch23 was
  .813015; original wd.01 best epoch25 is .814512. Missing wd epochs28-36:
  do not claim its actual best epoch or clinical inferiority. No demonstrated
  synthetic-validation improvement justifies stopping per user's preference.
  New dedicated `_aug` config returns to wd.01 and retains CE+classwise Dice,
  cutoff.001, frozen RAD-DINO last-one, upright transform, seed42, batch8/accum4,
  60 epochs and existing patient splits/data only. Training independently
  perturbs prior/current with probability.8 per image: additive brightness
  +/- .05, mean-centered contrast .9-1.1, gamma .9-1.1, Gaussian noise with
  std sampled0-.01, clamped[0,1]. No spatial or GT transform is added.
  Both train and validation extract online float32 features (no feature cache);
  validation and clinical inference receive no augmentation. This changes
  execution/numerics versus cached extraction, so do not describe it as an
  exactly identical numerical control. The pipeline first benchmarks32 training
  batches in an isolated process, saves timings/peak GPU memory and a synthetic
  image/GT preview, then reinitializes for the real run. Benchmark failure gates
  training; later clinical export/F1/scoped publication retain existing guards.
  Campus speed and memory remain unmeasured until that benchmark actually runs.
  Local validation:67 CPU regression tests passed, including online augmented
  train/clean-val execution, benchmark without saved training state, resume
  guards, no-cache train/val preflight without held-out test images, and complete
  mocked clinical export/publication. Bash syntax and scoped whitespace checks
  passed. No campus process has been started from this Windows session.

- **2026-10-19 - Authorized sequence: weight-decay-only run, THEN augmentation.**
  User requests augmentations using existing data only, no additional clinical
  cohort, and to run stronger head regularization before augmentation changes.
  Prepared configs/last1_upright_ce_dice_thr0p001_wd0p05.yaml: only weight_decay
  0.01->0.05 plus isolated checkpoint/plot paths differ from the completed CE
  recipe. Same frozen RAD-DINO upright last1 cache, manifest/splits, seed42,
  fresh head, CE/classwise-Dice, GT0.001, prediction0.5, batch8/accum4,
  learning-rate schedule and60epoch budget. No new early stopping or schedule
  shortening; compare BEST selected checkpoint, not final60. Existing gallery
  already displays BEST25, so late overfitting alone does not explain its errors.
  CE pipeline now allowlists the two run names with matching weight decay and
  contained checkpoint/plot paths; regularized results get a distinct
  clinical100_upright_ce_dice_thr0p001_wd0p05_TIMESTAMP prefix. Original naming
  and artifacts preserved. Same completion/provenance/probability/F1 checks and
  scoped artifact-only publication. No patient images, weights or cache upload.
  Two focused CPU tests PASS18.647s: both baseline/variant failure gates and
  complete100pair/300probability packaging, exact config-difference assertions,
  cross-run weight-decay/path rejection before training. Campus launch command
  and separate augmentation plan are in configs/README.md. Full61test CPU suite
  PASS117.995s; actual trainer YAML parsing confirms wd0.05/cached/60epochs,
  documented campus Bash syntax passes, touched code/config diagnostics clean.
  No campus job has
  been started from this local Windows session; actual new results are pending.
  Augmentation is PLAN ONLY until this ablation finishes: train-only mild
  independent per-scan photometric changes after min/max normalization, signed
  GT unchanged, using existing patients. Recompute features online with frozen
  eval/no_grad RAD-DINO, not random manipulation of cached tokens; trainer has
  --no_cache but existing launcher needs a guarded online path. Fix validation,
  head initialization, sampler RNG and evaluation; preview/seed/gradient/parity
  and memory checks before launch. Shared small geometry only in a later
  separate ablation with explicit GT interpolation/valid-border loss policy.
  No claimed simulation of true3D projection by photometric transforms, no
  invented device overlays or independent geometric warps. Clinical100 remains
  reused exploratory evidence, no independent-test or threshold-tuning claim.

- **2026-10-19 - Completed CE/classwise-Dice clinical100 results VERIFIED.**
  Pulled de169225, results/clinical100_upright_ce_dice_thr0p001_20261019_091535;
  compared against actual upright regression result
  results/clinical100_upright_cutoff0001_20261018_214717 (not morning epoch49).
  CE best checkpoint is epoch25, SHA4390c97c8cb114f05b0256ce828da015226af891d3e23863e36db5da9c8be717;
  regression best is epoch54, SHA8e5e3e913209c10d90e33d3ed9ac6bf6e96ade559a47d3d16783cbaff412509b.
  Full CE metrics history has exactly epochs1-60; maximum matched-target pooled
  signed Dice0.8145118753 at25, final0.8041646117 at60. Train loss0.204985->
  0.131504 while val loss0.679594->0.777921 from25->60: late generalization
  warning, not reason to extend training. Clinical maps are from BEST25, not60.
  Independently verified all recorded CE code/config hashes, manifest78d18b1c...
  54468, three-logit head and frozen-encoder recipe, all200local source hashes,
  identical500annotation hashes/protocol/cohort across runs, both100map/mask
  geometry/range/finiteness/exact threshold masks/hashes, and all300CE probability
  maps' native geometry/range/sum-to-one/signed-score identity/hashes. Known
  reviewed campus constants snapshot accepted where local comment-only bytes
  differ. Checkpoint weights remain on campus; no local weight reload/inference.
  ALL600per-pair overlap rows and EVERY observer-summary field reproduced
  EXACTLY for EACH run from actual saved maps plus local annotations. Independently
  reran REAL legacy detection scorer on both100pairs in disposable temp folders;
  entire F1 reports/CSVs and matching protocols EXACTLY match campus artifacts.
  Same original-space0.5 threshold, sign-separated8-connected greedy one-to-one
  overlap>=1pixel, no component filtering; detection F1 is NOT pixel Dice.

  | Clinician | Upright regression F1 all | CE/classwise-Dice F1 all |
  |---|---|---|
  | Avi | 0.140021 | 0.299795 |
  | Benny | 0.152918 | 0.297710 |
  | Sigal | 0.146751 | 0.305785 |
  | Smadar | 0.135385 | 0.308911 |
  | Nitzan | 0.182903 | 0.358209 |

  Equal-weight5clinician mean F1 positive0.133123->0.320778, negative0.191688->
  0.306636, combined0.151595->0.314082; BOTH signs improve for EVERY clinician.
  Mean clinician signed pixel Dice0.139293->0.172643, all5 improve. Consensus3/5
  signed Dice0.129975->0.161313 (same88activepairs); positiveDice0.154194->
  0.207159, negativeDice0.109455->0.106521 (slightly worse). Positive/negative
  consensus recall0.476505/0.429886 ->0.578404/0.451668.
  Predicted components573+/236-=809 ->169+/170-=339; mean predicted area
  15.3858%->15.1530%, empty predictions21->14. Fewer components without much
  area shrinkage is consistent with reduced fragmentation, not proof of anatomy.
  Union-of-any-clinician HMDR+0.230366->0.473373, HMDR-0.305085->0.500000;
  unmatched detections/pair4.41+1.64=6.05 ->0.89+0.85=1.74. These union metrics
  are distinct from individual-reader F1. Among12empty-consensus cases mean
  predicted/FP area10.5947%->9.2229%, max46.5795%->42.7958%; still substantial,
  and empty coarse annotations do not prove true absence of clinical pathology.
  Recommendation: CE best25 is the stronger NEXT clinical baseline relative to
  this upright regression run; do not extend/retrain just for more epochs.
  Inspect paired false positives/misses before another model change. Improvement
  cannot be attributed to CE alone: cutoff/loss/head-output/checkpoint criterion
  changed together. Reused100pair exploratory cohort/coarse ellipses, not fresh
  held-out confirmation or clinical threshold tuning; no statistical claim.

- **2026-10-18 - Next fresh run: three-class CE plus classwise Dice requested.**
  Added separate configs/last1_upright_ce_dice_thr0p001.yaml and
  training/run_ce_dice_clinical.py. Existing upright frozen-RAD-DINO last1
  cache retained; new three-raw-logit DPT head, no tanh. Class order is
  unchanged/positive/negative, strict threshold0.001, CE weights1/1/1.
  Objective CE + mean(positive,negative batchwise soft-Dice losses), weights1/1;
  nonzero background CE includes empty maps. No L1/L2/sign penalty, GT
  normalization, component deletion, encoder fine-tuning, or cache rebuild.
  Dedicated run_last1_upright_ce_dice_thr0p001 preserves seed42/60epochs,
  batch8/accum4/lr0.0003/weight_decay0.01/warmup2/clip1. Best checkpoint uses
  matched GT0.001 signed Dice; fixed GT0.05 remains a separate reference.
  New checkpoints identify head output mode and reject incompatible resume;
  populated CE directories missing last.pt cannot silently start afresh.
  Clinical export saves p_positive-p_negative plus all3 probabilities, with
  original geometry, sum-to-one, finite/range and signed-identity verification.
  Pipeline requires full configured training completion and unchanged code,
  config, manifest, source scans and annotations before clinical100/F1; errors
  stop downstream inference/publication. Only dedicated result artifacts are
  published, not weights/cache/source scans; active regression run untouched.
  CPU fixture tests passed for class boundaries, signs, empty-background
  gradients, two-epoch cached training/resume/logging, probability restoration,
  failure gating and scoped packaging. Complete training.test_dpt_head suite:
  60tests PASS in61.315s with bounded CPU threading; all8 touched Python files
  have clean editor diagnostics. Campus Bash command syntax/config agreement
  PASS. Fixed only a test's loss-sum tolerance to1e-6 for float32 rounding
  (observed difference5.96e-8); no objective change. Full campus training NOT
  launched here.
  Complete guarded nohup command is in configs/README.md, Next Run: CE And
  Classwise Dice. Wait for existing GPU job to finish; no repeated orientation
  audit. New cutoff plus loss is not an isolated CE-vs-regression ablation;
  clinical100 remains exploratory, with no clinical threshold tuning.

- **2026-10-18 - FULL candidate-threshold CC audit retrieved and verified.**
  Background watcher detected campus pushd989b2e0; pulled ff-only. Actual
  results/gt_cc_thresholds_20261018_203559 contains all18572pairs
  (train16464/val2108),55716native/float32/simulated-float16 rows, issues0.
  Verified expected split coverage, no invalid/out-of-range values,512/k3/flip
  configuration, source CSV/report and comparison-code hashes (allowing only
  Git LF/Windows CRLF byte normalization), and EXACTLY recomputed all36
  comparison summaries from per_pair.csv. Manifest78d18b1c...54468 matches
  current study. This is GT-only preprocessing with simulated cache float16,
  not an independent read of every actual cache record; no alignment re-audit
  or test-split evaluation. The monitoring process has stopped successfully.
  Validation sign-separated8-connected CC totals, including current baseline:

  | Cutoff | Positive CC | Negative CC | Total CC | CC per active map | CC under50px | Under50px foreground pixel fraction | Maps erased |
  |---|---|---|---|---|---|---|---|
  | 0.0001 | 28471 | 7296 | 35767 | 27.662 | 21785 | 0.2574% | 0 |
  | 0.001 | 8348 | 4367 | 12715 | 9.834 | 4264 | 0.0703% | 0 |
  | 0.01 | 3777 | 3984 | 7761 | 6.044 | 745 | 0.0196% | 9 |
  | 0.02 | 9233 | 10598 | 19831 | 15.505 | 12748 | 0.2794% | 14 |

  Train total CC269053 ->91899/56098/146479 at0.001/0.01/0.02;
  erased0/42/82 of9508originally nonempty maps. Validation erased0/9/14
  of1293; all9/14 are pneumothorax. Positive/negative/background PIXEL areas
  over ALL2108validation images (including815nuisance controls):
  0.001=7.6234/7.4725/84.9041%;0.01=6.4478/6.3154/87.2369%;
  0.02=5.4139/5.2064/89.3796%. Train0.001=6.9758/7.1862/85.8379%.
  Positive/negative pixel classes are broadly balanced with each other, NOT
  with background. Do not confuse sign-separated CC counts with pixel balance.
  Validation CC under50px fractions33.535/9.599/64.283%; their corresponding
  foreground pixel fractions0.0703/0.0196/0.2794% are small. Thus fragmentation
  can dominate component metrics while barely affecting pixel area; no automatic
  GT component deletion is justified. Higher cutoff can split broad regions.
  By type, consolidation and fluid_overload are EXACTLY unchanged between
  0.001 and0.01; at0.02 validation CC rise2248->4010 and2726->13486.
  Effusion CC4066->1274->1146; pneumothorax3675->1513->1189. Entire-sign
  losses also occur before whole-map loss (potential weak tails, not proven
  disappearance of true lesions); retain paired visual review before choosing.
  Recommendation:0.001 conservative next target baseline, markedly fewer CC
  than current0.0001 without erasing any nonempty map.0.01 is a separate
  cleaner-component ablation with explicit faint-pneumothorax tradeoff; avoid
  0.02 as default because fragmentation rises and more maps disappear. This
  is target-quality evidence, NOT proof of improved learned/clinical results.
  Keep current running training unchanged. Any future three-class CE run must
  use fresh3-logit head/checkpoint guards, background supervision, frozen
  encoder/same feature cache and fixed references; no CE implemented/launched.

- **2026-10-18 - Requested0.001/0.01/0.02 GT component/target comparison.**
  Extended audit_gt sign-separated8-connected component statistics with per-sign
  pixel area, tiny-component pixel mass, largest-component fraction, and each
  threshold's float16 support flips. New evaluation/compare_gt_thresholds.py
  summarizes existing full train/val audit CSVs, explicitly marks missing CC
  columns, and verifies actual cached-example GT against fresh float16 before
  rendering paired cutoff overlays. No source images/masks are modified.
  Local report: results/gt_threshold_comparison_20261018_001_01_02_final/index.md.
  Full18572train/val audit: active train9508 ->9508/9466/9426, validation1293
  ->1293/1284/1279 at0.001/0.01/0.02. Thus erased maps0/42/82 train and0/9/14
  val. Validation absolute mass retained99.937/98.631/94.808%, support retained
  85.821/72.559/60.377%. All9 and14 erased validation maps at0.01/0.02 are
  pneumothorax; this does not prove all their signal is biological rather than
  residual. Exact three TRAIN-example positive/negative CC counts:
  DRR1 2/0 ->5/0 ->2/0; DRR2 3/13 ->3/13 ->7/33;
  DRR3 18/5 ->4/1 ->5/1. Higher cutoffs can split regions, not just remove
  fragments. These example CC counts are NOT cohort estimates: old full audit
  measured CC only at0 and0.05. Candidate0.001 preserves all nonempty maps;
  defer final selection until full-cohort per-sign CC statistics are available.
  configs/README has full CPU-only GT-audit command at the three candidates
  plus current0.0001 baseline; no alignment re-audit, overlays, arrays, test
  data, cache rebuild, encoder inference or active-job changes. A dedicated
  fresh audit/comparison parent can be published with --publish using existing
  exact-folder publisher; unrelated contents reject publication before writing.
  Cross-entropy explanation/migration checklist documented:3logits for mutually
  exclusive unchanged/positive/negative, integer thresholded targets, CE on raw
  logits, nonzero background weight, optional later classwiseDice, signed export
  p_positive-p_negative plus retained probability maps. Frozen encoder/cache
  unchanged; new head/run and checkpoint guards required, no CE enabled yet.
  Two focused CPU tests passed including boundary/diagonal/fragment/empty masks,
  actual NIfTI end-to-end old/new audit schemas, held-out split exclusion and
  scoped publisher invocation/rejection. Campus full-cohort CC execution remains
  pending; do not claim all requested component measurements are complete.

- **2026-10-18 - Current upright run epoch44/60, user-supplied console log.**
  Config remains signed_mask cutoff0.0001, L1=L2=1, Dice=sign=0,
  foreground11/background0.75; learning rate5.29e-5. Online sampled train
  batch-mean loss0.253945 (L1 .118063, L2 .135881); fixed-head validation
  loss0.912615 (L1 .420428, L2 .492186). Optional Dice/sign numbers are
  diagnostics, not active objective terms. At prediction0.5 and matched raw
  GT0.0001, validation1293pathology pairs: macro supportDice0.6767,
  signedDice0.6526, micro signed precision0.7147/recall0.8169. Train matched
  signedDice0.8004, precision0.7608/recall0.9779; online sampled training is
  not a fixed-checkpoint evaluation of the whole train set. Validation815
  nuisance pairs: FP area mean1.50%, p95 10.69%, max48.12%; train3344 sampled
  nuisance occurrences: mean0.09%, p95 zero, max19.95%. Invalid pairs zero
  in reported summaries; this does not certify every numerical health field.
  Compared with this run's epoch12, validation matched signedDice0.6345 to
  0.6526 (+0.0181), nuisance mean6.10% to1.50%, but loss0.825217 to0.912615.
  Fits improving background suppression alongside a foreground/generalization
  warning, not all-zero collapse or clear late-stage overall improvement.
  Raw GT0.05 is a DIFFERENT reference: checkpoint supportDice0.3736,
  active-reference signedDice0.3679, micro precision0.3066/recall0.9581,
  component signed F1 .1555. Do not compare that F1 with matched-target pixel
  Dice or infer clinical F1. Current epoch44 matched signedDice0.6526 is close
  to previous non-upright run's epoch49 .654103, but nuisance mean1.50% is
  above its1.10598%; different epochs, not an isolated orientation effect or
  final best-checkpoint comparison. No evidence here for promoting the upright
  run over the previous run. Continue only to the planned60 if budget permits;
  no extension, mid-run target/loss changes, or clinical claims. Checkpoint
  identity/best epoch and full diagnostic artifacts not independently inspected.

- **2026-10-18 - Primary-paper loss review and current cached-GT threshold sweep.**
  Read Lin et al., Robust Scene Change Detection Using Visual Foundation Models
  and Cross-Attention Mechanisms, sections IV-C and V-C:
  https://arxiv.org/html/2409.16850v2 . Frozen DINOv2, two directional
  cross-attention comparators, two-class change/unchanged logits, weighted softmax
  cross-entropy; change weight0.975, unchanged0.025 (ratio39:1). No L1/L2 or Dice
  in the stated training objective. This is binary scene segmentation, not signed
  clinical change regression. Li et al., BAN, section III-D:
  https://arxiv.org/html/2312.01163v2 uses pixel-wise cross-entropy and a frozen
  foundation model PLUS a trainable image-processing Bi-TAB branch; not a
  cached-latent-only architecture. Neither paper validates a DRR noise cutoff.
  Independently measured abs(cached_continuous_gt)>cutoff from all three actual
  results/cache_input_evidence_20261018_160345/drr_*/exact_tensors.npz files:

  | Cutoff | DRR1 area % | DRR2 area % | DRR3 area % |
  |---|---|---|---|
  | 0.0001 | 42.205 | 30.255 | 40.361 |
  | 0.001 | 40.855 | 30.255 | 29.763 |
  | 0.005 | 38.816 | 30.255 | 25.182 |
  | 0.01 | 36.255 | 30.255 | 20.956 |
  | 0.02 | 32.813 | 26.379 | 16.024 |
  | 0.05 | 25.843 | 9.893 | 10.969 |

  These are three selected TRAIN examples, not full-split estimates or proof
  that excluded pixels are noise. Proposed next candidate:0.01, subject to
  paired visual review and existing full train/val audit; faint real changes can
  also disappear. Avoid per-image max/minmax GT normalization, which can amplify
  residuals and destroy amplitude comparability. Current change_map_loss signs
  every retained pixel, then uses foreground11/background0.75 pixel-weighted
  L1+L2; its optional Dice is unsigned abs(pred) support Dice, not signed Dice.
  First isolate target cutoff with that loss unchanged. Then compare a proposed
  three-class negative/unchanged/positive cross-entropy formulation, optionally
  adding classwise Dice as a separate ablation. This requires three logits and
  an explicit signed-score export rule, not CE applied to existing tanh scores;
  it is not a paper-exact reproduction or a pure loss-only change. Do not copy
  Lin's39:1 weighting without testing nuisance false positives. Preserve nonzero
  background supervision, frozen RAD-DINO, split/cache/input orientation, fixed
  evaluation reference and prediction protocol. No code/config/cache/checkpoint
  changes, new training, or interruption of the current upright run performed.

- **2026-10-18 - Fresh model-free cache-input/GT evidence exporter requested.**
  User correctly distinguished the morning epoch49 clinical collage from the
  current upright-cache run; old-model predictions are NOT evidence of current
  model inputs. Added evaluation/export_cache_input_evidence.py for campus export
  of3realized synthetic TRAIN pairs (distinct cases/anomaly types) and3clinical
  TEST pairs (defaults1,2,3 including the disputed pair3). This is not the DRR
  manifest's held-out test split. Uses actual raw dataset/DataLoader and selected
  CachedPairDataset records, actual clinical image/annotation loaders, and actual
  cache backbone factory/forward preprocessing. ONLY HF AutoModel.from_pretrained
  is replaced by a parameter-free capture sink; no weights or neural inference.
  Original processor metadata loading/fallback, RGB repeat,518resize and channel
  normalization execute unchanged; effective mean/std are recorded. Exact input
  and normalized pixel_values are exported as compressed NPZ; PNG encoder views
  reverse normalization for visibility, not orientation or anatomy.
  Current config/cache/manifest hashes must agree; active run_config.json is also
  compared when present. Fresh oriented GT converted through float16 must equal
  the actual cached GT. Exported signed target is verified against the actual loss
  implementation at0.0001, with checkpoint reference0.05 shown separately.
  Clinical doctors/consensus stay native-coordinate measurement references;
  resized/oriented consensus is explicitly display-only, not a new scoring GT.
  Six source/model/encoder/reference review PNGs, separate input PNGs, exact NPZ,
  source hashes and script/config/code provenance go to a fresh results/
  cache_input_evidence_TIMESTAMP. --publish reuses existing exact-folder Git
  publication; authorized six clinical/DRR input views and reference masks are
  uploaded, not source NIfTIs/annotationJSONs/cache/checkpoints. CPU-only capture
  leaves current training and existing caches untouched; no repeated full audit.
  Tests passed: exact encoder preprocessing without weights plus full6pair
  non-square real-NIfTI/raw+cached-loader/clinical-annotation export, actual-loss
  target equality, cache preservation, wrong-cache rejection and scoped publisher
  forwarding. ACTUAL campus six-example export is PENDING user command in
  configs/README.md Exact Input And GT Export; do not claim images reviewed yet.

- **2026-10-18 - Prediction-only horizontal mirror worsens clinical epoch49 agreement.**
  User requested checking clinician placement in the collages and mirroring the
  predictions before OV. Annotation tool displays native image.T; legacy OV
  rasterizes cx/cy/rx/ry in native array coordinates, and clinical collages
  transpose the scan and human masks together. This is coordinate-consistent;
  it does not certify every clinical label or prove anatomical laterality.
  Existing clinical inference already inverse-transforms model-space predictions,
  so mirrored encoder inputs alone do not justify a second output reflection.
  Ran a separate exploratory ablation of all100 saved cutoff.0001 epoch49 maps
  (checkpoint45a84390eefe4e233bb63ee0de7ba49429684d49ea950b9ab827e69fbd02ee7b).
  Horizontal display reflection equals native array axis0 flip, NOT np.fliplr
  on the native array. No sign inversion, resizing, threshold change or CC cleanup.
  Preserved native shape/affine and original maps; used prediction threshold.5,
  same500 annotation hashes and unchanged legacy one-to-one8CC overlap matching.
  All100 mirrored maps/masks and collages generated in review_outputs/
  clinical100_epoch49_mirror_ablation_20261018; index.md links the100 collages,
  original/mirrored OV matrices and f1_comparison.csv. Both complete OV runs
  processed100pairs; all6human-human matrix blocks are exactly unchanged.

  | Clinician | Original F1+ | Mirror F1+ | Original F1- | Mirror F1- | Original F1 all | Mirror F1 all |
  |---|---:|---:|---:|---:|---:|---:|
  |Avi|.146565|.119084|.181818|.064516|.158635|.100402|
  |Benny|.160000|.148148|.167598|.100559|.162633|.131655|
  |Sigal|.142857|.136646|.217765|.108883|.169184|.126888|
  |Smadar|.144578|.117470|.182857|.097143|.157791|.110454|
  |Nitzan|.182891|.179941|.256131|.179837|.208612|.179904|

  Every clinician's positive, negative and combined detection F1 worsens.
  Equal-weight clinician mean combined F1 .171371 -> .129861; pixelwise3of5
  consensus signed Dice .138688 -> .060207 (same88active-reference pairs).
  Original F1 values exactly reproduce the previous epoch49 evaluation.
  Components stay567positive/281negative; union-of-any-clinician HMDRpositive
  .236332 -> .253968 improves slightly, HMDRnegative .316726 -> .202847 worsens;
  total unmatched detections/pair6.25 ->6.47. These are different references and
  aggregations, so do not claim every OV statistic worsens. Do NOT promote the
  mirrored maps as corrected predictions or modify current upright training.
  Validation: 11clinical CPUtests pass, including non-square asymmetric mirror
  axis/involution, real NIfTI export and OV integration, annotation-hash rejection,
  original-map preservation and exact human-human agreement invariance. All100
  collages and12OV matrices decode; pair1 mirrored collage and combined OV matrix
  visually inspected. Encoder, trainer, bound orientation code and old outputs
  unchanged. Generated maps and patient-image collages remain local/untracked.

- **2026-10-18 - Corrected campus cache input preview reviewed.**
  Pulled bd0ee0b8: results/upright_cache_orientation_20261018_115435 contains
  cache_config.json and input_preview.png. Programmatic checks confirm frozen
  backbone identity microsoft/rad-dino, last_n_layers=1, rotation_k=3,
  flip_horizontal=true, image_size=512; manifest SHA256 exactly matches the
  reviewed orientation_preflight_20261018_104010 provenance. Visually inspected
  all three preview rows: scans upright, previous horizontal mirroring corrected,
  continuous and signed GT in the same apparent orientation as prior/current.
  The middle row has no visible GT change; the other rows show broad positive
  targets at the configured 0.0001 cutoff. Preview SHA256:
  e2c420f79d24f6d8772f7d34038b88c6162aba1e8e9f5bb5054a9e2b9dc6bcf8.
  Together with the earlier exhaustive alignment audit, this supports proceeding
  with the corrected orientation. These exports do NOT establish cache completion,
  independently verify every latent record, certify biological GT correctness,
  or demonstrate improved clinical performance. No encoder/model/config changes.

- **2026-10-18 - User explicitly requested skipping repeated audit and moving on.**
  Reuse the already inspected orientation_preflight_20261018_104010 (18572aligned,
  zeroissues,all37previews reviewed), not an unaudited run. Added optional explicit
  --reviewed-code-update plus exact artifact SHA256 to pipeline approval. This
  binds the original approved preflight digest and exact old/new hashes of ONLY
  training/train_foundation_diff.py and training/run_upright_clinical.py. Default
  strict approval behavior is unchanged. Config/manifest/report/panels/all other
  code remain checked; original audit/provenance stays immutable. Campus constants
  remain the exact inspected comment-only variant. The separate review artifact
  results/orientation_code_review_20261018/reviewed_code_update.json documents
  the decision; its record/digest persists in final clinical verification.json.
  New focused guard test passes: explicit review accepts precisely reviewed code,
  rejects mismatched review hash/parent audit/old code or non-trainer overrides,
  and still rejects config/report/panel/manifest/new-code tampering. Existing
  default approval guard test also passes. Actual artifact verified against all
  locally available old/new code/config/report/all37PNG hashes; campus manifest
  and clinical scans remain runtime checks. README provides exact GPU campus
  phase2 command: corrected cache then training,clinical100,F1,scopedGitpush. No
  repeat exhaustive audit required; no encoder fine-tuning or global guard bypass.
  Campus training has not been launched from this PC.

- **2026-10-18 - Campus constants mismatch resolved as comment-only.**
  Pulledb1a7d951: results/orientation_source_20261018_112551/constants.py and
  constants.diff. The exported source exactly matches the original preflight's
  constants SHA256be9e6bc7c28d8d6d2456aa34636eea1f14794bb99500905a5992d0467937931e.
  Full AST comparison against tracked constants.py is IDENTICAL: only the inline
  comment beside LAST_N_LAYERS=1 was removed. No parameter/encoder/feature-size
  change occurred; do not overwrite the campus file to restore a cosmetic comment.
  This resolves the unexplained constants blocker, not the stale trainer hash.
  Fresh phase1 is still required after ad10c245 logging changes; preserve the old
  audit/provenance and withhold phase2 approval until the new artifact is verified.
  For that next audit, compare constants against this exact reviewed campus
  snapshot (be9e6bc7...), not only Git's cosmetically different constants blob;
  remaining code must match the current reviewed trainer/preprocessing files.
  Existing complete phase1 Bash command syntax verified. No campus audit, feature
  extraction or training was started from this PC; next user action is phase1.

- **2026-10-18 - Retrieved/reviewed campus upright preflight; approval withheld.**
  Pulled34204688: results/orientation_preflight_20261018_104010. Campus report:
  selected=audited=alignment_verified18572 (train16464,val2108),issues0;
  rotation3+horizontalflipTrue,target0.0001. Provenance records clinical100pairs,
  200source hashes and500annotation hashes. At training float32, cutoff0.0001
  retains93.0516%/93.1807% pooled nonzero support (train/val), no nonempty pair
  entirely removed; cutoff0.05 instead removes430train/56val nonempty pairs.
  Inspected all37 real previews (all prior/current/continuous overlay/actual
  signed-target panels via temporary review sheets, plus three originals). No
  obvious extra transpose/mirror or GT placement mismatch observed. Some cases
  have severely cropped source fields of view (including train_1022_a_3,
  train_1022_a_5,train_1022_a_6); targets can be broad or fragmented. This is NOT
  expert biological-label validation or proof of laterality for unmarked scans.
  Verified50 published hash checks against exact Git bytes and PNG files:
  48match, including config/report/all37panels. TWO code mismatches block approval:
  training/train_foundation_diff.py hash7bb2a6273138c02126bf2a58371c4b95900bc248fbd485a34c17402c4f71fa50
  matches the pre-ad10c245 trainer (96d61a83/a4f2cd71), not the new logging trainer.
  constants.py campus hashbe9e6bc7c28d8d6d2456aa34636eea1f14794bb99500905a5992d0467937931e
  differs from trackedf38f94be2416a6a769f987b9a8a7c0dd8dfb9351f9b2e75c294cd4e2150627bb,
  and is NOT explained by LF/CRLF conversion of the compared revisions. Actual
  campus constants contents are not yet available; do not guess or overwrite them.
  NEXT: exact-path Git export of current campus constants.py and its diff for
  inspection, then a fresh approved preflight using current trainer/code. Do not
  patch the old provenance hash or issue phase2 approval from this artifact.
  No training/cache jobs started/stopped, caches removed or source data modified.

- **2026-10-18 - Explicit train/validation objective and metric reporting.**
  User requested correct losses and metrics printed for BOTH train and val.
  Validation now computes/reports total and all components with the actual train
  coefficients, including effective Dice warmup, target cutoff and pixel weights.
  Epoch headers state the objective; component values are pre-coefficient and
  loss aggregation is explicitly batch-mean. Detached train/val diagnostics print
  active-reference pathology macro support Dice/IoU and signed Dice, pooled signed
  pixel precision/recall, nuisance predicted-area mean/p95/max, sample counts and
  invalid counts; optional connected-component signed detection F1 is separate.
  Both primary raw-GT cutoff and signed-mask target cutoff are shown/collected;
  target0.0001 is NOT mislabeled as checkpointGT0.05. Train is explicitly online
  sampled/updating-head (not whole-train fixed-head); val is fixed-head. Undefined
  metrics are n/a, and skipped diagnostic epochs are not reported as zero. Total/
  component loss summaries persist in diagnostic JSON metadata and val checkpoints;
  existing metrics.csv schema is preserved for resume compatibility. No loss,
  gradient, sampling, encoder or best-checkpoint decision changed. Trainer source
  edits invalidate old orientation approvals: pull before fresh phase1 audit.
  All seven focused CPU checks passed: active-loss-weight parity; sign reversal,
  target-cutoff labels, nuisance separation and undefined-reference handling;
  cached-training/output/resume for last4 DPT, last4 cross-attention and last1
  cross-attention; diagnostics leave optimizer updates exactly unchanged; original
  mask-target/raw-GT validation behavior preserved. Touched-file diagnostics clean.
  Campus execution remains pending; already running trainers do not acquire these
  logging changes by pulling Git and must not be stopped solely for prettier logs.

- **2026-10-18 - User authorized removal of obsolete last-four/last-one latents.**
  Campus cleanup scope is exactly /cs/labs/josko/sahar_aharon/ directories:
  fcd_cache_last4,fcd_cache_last1,fcd_cache_last1_from_last4,
  fcd_cache_last1_inference,fcd_cache_last1_rot90cw_inference. These are the known
  legacy sideways/mirrored caches; their current campus existence and byte sizes
  have not been inspected from this PC. Keep fcd_cache_last1_upright_inference,
  all source DRRs/signed GT, trained heads, configs, metrics and clinical outputs.
  Added exact-path cleanup command with local job check; user must also ensure no
  jobs on other campus nodes use the shared caches. No wildcard deletion and no
  automatic deletion in the training pipeline. Cleanup execution remains pending
  on campus; do not claim space recovered. Old cache-dependent runs would require
  feature regeneration if revisited. Corrected features are rebuilt, not flipped
  or sliced from these obsolete vectors; encoder remains frozen.

- **2026-10-18 - Corrected-orientation retraining and DRR/GT approval gate prepared.**
  User requested retraining with correct positions, explicitly including DRRs AND
  GT, followed automatically by all100 clinical predictions and a Git push.
  Added opt-in flip_horizontal after rotation3: this equals the conventional raw
  array transpose. Prior/current/signed GT share the exact transform; clinical
  restoration unflips before inverse rotation. Old defaults remain unchanged.
  Separate last1_upright_mask_thr0p0001_fp075 config/cache/run retains the latest
  target0.0001 recipe,seed42,60epochs,head,loss and thresholds. Encoder stays frozen;
  old mirrored features cannot be reused or fixed by flipping tokens. Cache metadata
  and every loaded record, checkpoint resume and evaluation carry flip guards.
  Existing GT audit now supports exhaustive source-grid/finite checks and exact
  transpose/inverse alignment of all three tensors. Selected corrected overlays
  show continuous GT AND the actual0.0001 signed training mask, stratified by
  pathology/sign/support category in train/val, never synthetic held-out test.
  Added training.run_upright_clinical: phase1 publishes this real campus audit and
  checks all100 clinical input transforms/source hashes/500 original annotations.
  Phase2 requires the reviewed provenance hash, rejects changed code/config/report/
  panels/manifest, rebuilds features, trains only the corrected run, then evaluates
  the new best checkpoint on the same100 clinical pairs at0.5. Before automatic
  exact-folder Git commit/push it verifies all100 output/mask shapes,affines,signed
  values and mask thresholds, and runs existing per-clinician component F1.
  Publishes derived maps/masks,reports,provenance,metrics and input preview, not
  source clinical scans,clinical review PNGs,weights or the100GB feature cache.
  Failed preflights also publish their diagnostic evidence but never approve or
  train. Initial full47CPUtests passed40.037s, including a deliberately shifted GT
  affine rejected by the source-grid audit, exact corrected signed-GT arrays,
  old/new clinical inference original-space parity, all100 export guards and scoped
  Git publication. Actual local clinical preprocessing preflight passed ALL100
  pairs/200source scan hashes/500matching annotations: every corrected input equals
  the conventional resized display, with exact signed-landmark inverse placement.
  No inference weights were loaded in that real-data preprocessing check.
  Final full regression suite:48CPUtests passed16.790s; touched-file diagnostics
  clean. The new failed-preflight test confirms evidence publication without
  approval or any training call. Campus DRR/GT audit and GPU execution are pending.
  Shape/affine/transform checks alone do not establish anatomical laterality or
  biological GT correctness; real corrected DRR/GT overlays still need campus
  export and visual approval BEFORE training. No campus training launched here.

- **2026-10-18 - Fresh campus training samples retrieved and inspected.**
  Pulled campus commit245c65ed, containing only results/
  fcd_orientation_20261018_100044/training_inputs.png (512560 bytes).
  This is the requested split=train export using rotation_k=3. Inspected all
  three prior/current rows and compared with the actual clinical encoder-input
  panel in review_outputs/clinical100_cutoff0001_epoch49_20261018/pair1/review.png.
  Training samples show the same visually mirrored appearance relative to the
  conventional chest display, including left-sided cardiac silhouettes. No
  training laterality markers are visible; this does not establish every case's
  anatomical orientation. Decoded RGBA pixels are EXACTLY identical to the earlier
  epoch45 cache input_preview.png (1080x1440). Fresh PNG SHA256:
  784bc5daf0a7ce1dbdc7fb5045249e62d17bc271805a15827ee92e0fcf0d0c08.
  Both epoch45 and epoch49 metadata specify the same rotated cache, so these
  observations support shared training/clinical input orientation in the inspected
  examples, not an epoch49-only flip or demonstrated train/clinical mismatch.
  A possible frozen-encoder orientation effect remains unmeasured and cannot alone
  explain the old/new performance difference. GT preview mask is explicitly @0.05,
  not the new target cutoff. No preprocessing, cached features, weights, prediction
  coordinates or metrics were changed. Earlier fresh-preview transfer-pending
  notes are superseded by this completed retrieval and inspection.

- **2026-10-18 - Campus artifact transfer must use Git; training preview upload authorized.**
  User explicitly requires artifacts needed from campus to be pushed to Git for
  local retrieval, rather than requested as private chat attachments. Saved this
  workflow preference in persistent user memory. This explicitly supersedes the
  earlier private-only instruction for the new fcd_orientation_*/training_inputs.png
  preview. Prepared exact-file staging/commit/push commands, excluding unrelated
  files. Actual campus push and fresh-preview inspection remain pending.

- **2026-10-18 - Training-orientation preview found; previous epoch45 shares epoch49 inputs.**
  Inspected the existing campus-exported review_outputs/
  rot90cw_mask_fp075_val_best_20261017_102112/input_preview.png: three paired
  dataset inputs rendered directly, without display transpose, at rotation_k=3.
  Several cardiac silhouettes appear on the image left, consistent with mirroring
  relative to the conventional chest display; no laterality markers are visible,
  so this is not definitive anatomical left/right verification. The preview alone
  does not identify its split or individual pair IDs. Old epoch45 clinical metadata
  and new epoch49 metadata both record no_cache=false, rotation_k=3 and the same
  fcd_cache_last1_rot90cw_inference cache. Thus the previous foundation model used
  the same input-orientation convention, not a new epoch49-only transformation.
  Earlier unrotated models and Itamar are not covered by this comparison.
  Prepared a campus command using LongitudinalPairDataset(split="train",
  rotation_k=3) and the existing save_input_preview helper to export three distinct
  training cases privately, without loading the encoder or modifying the cache.
  Fresh campus execution and transfer remain pending; do not commit preview images.
  The helper's GT mask column is explicitly thresholded at0.05, not the new
  epoch49 training cutoff0.0001; the scan inputs are common to both models.
  No preprocessing, predictions, evaluation metrics or trained weights changed.

- **2026-10-18 - New epoch49 visual reviews rendered; mirrored encoder input identified.**
  User requested the same visual reviews as old epoch45, then noticed mirroring in
  the new image. Generated all100 local cases in review_outputs/
  clinical100_cutoff0001_epoch49_20261018 with index.md, metadata/render provenance,
  per-pair output/mask/source/review.png and copied/recomputed scalar summaries.
  Reused the original clinical exporter, with an optional explicit NEW MODEL/epoch49/
  target.0001/prediction.5 label and increased row spacing to prevent heading overlap.
  Inspected actual pair1 PNG after layout repair; all100 PNGs pass readability checks.
  All100 source-map hashes match the F1-run provenance; exported array values and
  original-space affines are unchanged; masks match strict +/-0.5 thresholding.
  Reproduced all600 published observer rows and the complete report without changes.
  No training, encoder loading, new inference, threshold tuning or component cleanup.
  Review images/volumes remain local and are not staged/published in this update.

  IMPORTANT orientation finding: the third panel is the ACTUAL encoder input, not an
  arbitrary mirrored display. Runtime check on the real pair1 gives exactly
  prepare_image(current,512,3)[0] == fliplr(normalized_resized_current.T), maximum
  difference0. Current and the original-space prediction/clinician overlays display
  with .T, whereas the input panel directly shows the raw-array clockwise90 rotation.
  Thus the clinical encoder does receive a horizontally mirrored view relative to
  the displayed chest. Both training configuration and clinical inference use
  rotation_k=3; the dataset applies it to prior/current/GT. Matching rotation flags
  alone do NOT prove training DRRs and clinical scans have matching anatomical
  orientation; actual DRR anatomy still needs inspection before a preprocessing fix.
  Asymmetric +/- landmark tests recover exact original locations through the inverse
  prediction transform for all4 rotations. This establishes no extra reflection in
  that coordinate-mapping step, not that mirrored clinical inputs are appropriate
  for RAD-DINO or that all possible registration problems have been excluded.
  Do not call this merely cosmetic, dismiss the user's observation, flip predictions
  to improve scores, or silently change inference orientation for the trained cached
  head. Keep raw inputs visible and label the discrepancy in the local review index.
  No training/cache/inference transforms or existing checkpoint metrics changed.

- **2026-10-18 - Actual one-pixel detection F1 computed locally; lower for all five clinicians.**
  Campus upload298e8984 supplies100 epoch49 original-space output.nii.gz maps and
  inference metadata in results/clinical100_cutoff0001_epoch49_predictions. Supersedes
  the F1-pending notes below. Metadata exactly matches the published clinical run,
  checkpoint45a84390eefe4e233bb63ee0de7ba49429684d49ea950b9ab827e69fbd02ee7b, epoch49.
  Verified100 finite map shapes/affines against local current scans and all500 annotation
  hashes. Independently recomputed every metric in all600 published observer rows from
  the actual transferred maps/local annotations: maximum absolute difference0, including
  empty-reference handling. No source-image byte-hash comparison was available.
  Ran evaluation.compare_clinical_models --foundation-only locally on CPU,100 pairs,
  prediction.5, original coordinates,8-connected components, no size/intensity filtering.
  No retraining, backbone loading, inference rerun, threshold tuning or annotation edits.
  Scores/provenance in results/clinical100_f1_cutoff0001_epoch49_local_20261018_092936:
  f1_per_clinician.csv, report.json, metadata.json and verification.json. Legacy outputs
  and aggregate plots remain local; no new raw maps, clinical scans or case images are
  staged for this results review. User separately authorized the preceding map upload.
  Old epoch45 comparator: review_outputs/clinical100_20261017_112038/itamar_comparison/
  foundation_100, pinned by comparison.json to the old metadata checkpoint. Verified
  exact old/new500 annotation-hash equality and identical complete legacy protocol.
  Every exported F1 was checked against its actual per_label matrix, not per_pair PAI.
  Same-sign >=1-pixel overlap qualifies for the unchanged greedy one-to-one matching;
  F1=2M/(2M+U), pooled across pairs; combined pools positive/negative counts.

  | Clinician | Old F1+ | New F1+ | Old F1- | New F1- | Old combined | New combined |
  |---|---|---|---|---|---|---|
  |Avi|.268852|.146565|.236967|.181818|.255814|.158635|
  |Benny|.233846|.160000|.236842|.167598|.235081|.162633|
  |Sigal|.272109|.142857|.273973|.217765|.272904|.169184|
  |Smadar|.286624|.144578|.281818|.182857|.284644|.157791|
  |Nitzan|.329268|.182891|.345992|.256131|.336283|.208612|
  |Equal-weight clinician mean|.278140|.155378|.275118|.201234|.276945|.171371|

  Thus new combined F1 drops for5/5 clinicians, including Nitzan despite his improved
  pixel Dice. Positive and negative detection F1 also fall for every clinician. This
  corrects the earlier unknown-F1 status with an actual measured result, not an
  inference from Dice. Clinician mean is not pooled across clinicians or consensus F1.
  Model component counts: positive217->567, negative151->281, combined368->848 across
  100 pairs (3.68->8.48 per pair). HMDR against any-clinician union positive.506912->
  .236332, negative.476821->.316726; unmatched detections per pair positive1.07->4.33,
  negative.79->1.92, combined1.86->6.25. Union HMDR/UDPP are descriptive auxiliary
  measures, not per-clinician F1 precision. More predicted components and a much larger
  unmatched burden accompany the previously measured larger total predicted area.
  Do not claim a specific anatomical failure or that component-size filtering fixes
  this: no cleanup ablation or new-case visual review has been performed.
  Historical Itamar100 combined F1 (unchanged saved operating point): Avi.467236,
  Benny.407216, Sigal.494253, Smadar.444444, Nitzan.510000; clinician mean.464630.
  Historical100 uses lmm5 substitutes for pairs99/100 and threshold0/pre-save cleanup,
  whereas both foundation comparisons use.5/no extra cleanup. Not a threshold-matched
  architecture-only comparison. No new Itamar inference/evaluation was run here.
  Decision: retain the old frozen-head model as the stronger current clinical baseline
  on BOTH measured pixel-overlap and one-pixel component-F1 evidence. Next inspect
  false components/target mismatch privately before another full training run; do not
  optimize this exploratory reused clinical cohort or portray it as a fresh test.
  Coarse ellipse references, greedy matching and different checkpoint trajectories
  remain limitations; no significance, anatomical-validity or clinical-outcome claim.

- **2026-10-18 - One-pixel clinical detection F1 requested; campus scorer prepared.**
  User requested actual detection F1 for cutoff.0001 epoch49, not pixel Dice. Added
  --foundation-only to evaluation.compare_clinical_models: reuse the unchanged
  run_legacy/OV matcher at prediction.5, original coordinates,8-connectivity, no CC
  filtering. Any same-sign pixel overlap qualifies for a greedy one-to-one component
  match; pooled F1all uses combined positive/negative counts, not mean directional F1
  or boosted per-pair PAI. No Itamar files needed in this mode. Preserve the existing
  full-comparison CLI's Itamar requirement. Verify annotation hashes/all saved maps,
  requested cohort size and complete legacy processing; use a fresh output directory.
  Export f1_per_clinician.csv, report.json and metadata.json with checkpoint identity,
  annotation/prediction hashes and actual protocol. No medical images in those exports.
  Nine clinical tests pass, including real legacy scoring on synthetic saved NIfTI:
  one-pixel hits, FP/FN components yield F1+=.5,F1-=1,F1all=2/3; changed annotations and
  missing predictions are rejected before output creation. Documented guarded campus
  command and scalar-only publication in configs/README.md. Actual new-model F1 is
  STILL PENDING: new raw maps are on campus, not in the local scalar upload; this
  session cannot execute there. No inference, training, threshold tuning or fabricated
  clinical F1. The prior clinical comparison establishes worse pixel overlap/FP area,
  not worse detection F1. Defer that metric's conclusion until campus scores arrive.

- **2026-10-18 - Clinical cutoff.0001 epoch49 completed; no clinical Dice improvement.**
  Pulled7d7bb709, results/clinical100_cutoff0001_epoch49_20261018_085458: metadata.json,
  per_pair.csv and report.json. Supersedes the pending-inference notes below. Reported
  checkpoint epoch49, SHA25645a84390eefe4e233bb63ee0de7ba49429684d49ea950b9ab827e69fbd02ee7b.
  Compare old epoch45 in review_outputs/clinical100_20261017_112038 (SHA2560b7b49cc
  916806453ae5a2348e8b0546dd59dfbe5ff0449c034ba43d214a8053; concatenate these lines).
  Verified identical head specs, rotation3, prediction.5, preprocessing, original-space
  coordinates, no postprocessing and scan ordering, plus all500 annotation hashes.
  Both exports contain100 pairs/600 unique pair-observer rows (five clinicians and
  3-of-5 consensus), identical pair IDs/reference-change flags. Recomputed every report
  mean and valid count from CSV, checked finite0..1 values. Known config differences
  are target cutoff.05->.0001, output/config paths and added diagnostic settings;
  active loss weights and model architecture remain unchanged. Metadata does not hash
  source-image bytes; their identity has not been independently verified here. Actual
  checkpoint bytes/images stay on campus; this review uses exported provenance/metrics.

  | Observer | Old signed Dice | New signed Dice | Delta | Valid reference pairs | New better/worse/tied |
  |---|---|---|---|---|---|
  |Avi|.119250|.109810|-.009441|92|37/37/18|
  |Benny|.124693|.111020|-.013673|97|35/36/26|
  |Sigal|.176209|.157974|-.018235|94|31/42/21|
  |Smadar|.167233|.126348|-.040885|92|26/45/21|
  |Nitzan|.212946|.257144|+.044198|98|51/32/15|
  |Consensus3-of-5|.184913|.138688|-.046224|88|23/44/21|

  Equal-weight mean of five clinicians' mean signed Dice .160066->.152459; do not
  confuse this with consensus or pooled/pixel Dice. Empty-reference pairs have no Dice
  value and are excluded rather than credited1; FP area is reported separately.

  | Consensus metric | Old | New | Valid pairs |
  |---|---|---|---|
  |Positive Dice|.199052|.183930|56|
  |Negative Dice|.160946|.105274|46|
  |Positive recall|.330136|.558207|56|
  |Negative recall|.313608|.441216|46|
  |FP area on empty-consensus references|.035821|.122080|12|
  |Mean predicted area over all cases|.060944|.170209|100|

  Median predicted area2.44%->16.60%, max33.49%->49.32%; zero-prediction cases12->14.
  On the12 empty-consensus cases worst FP area27.13%->47.68%. Such consensus-negative
  masks mean no qualifying3-of-5 overlap, not proven absence of clinical pathology.
  Clinical predictions occupy much more area, recall rises but consensus and4/5
  clinician Dice fall. This supports broader/excess predictions relative to these
  references, not successful clinical nuisance suppression or an all-zero collapse.
  Contrast synthetic FP improvement with clinical empty-reference errors cautiously:
  these are different populations/reference definitions, not interchangeable scores.
  No threshold sweep, checkpoint selection or training was performed on this cohort.
  Clinician ellipses are coarse regions, and this is a reused exploratory100-case set,
  not a fresh confirmatory test. No statistical-significance, anatomical-validity or
  clinical-outcome claim is made. Different checkpoint epochs/training trajectories
  also mean the observed result is not an isolated causal proof about cutoff choice.
  Component detection F1/PAI and the historical Itamar comparison were not included in
  this new upload; pixel Dice is not a substitute for those detection metrics.
  Largest consensus signed-Dice drops: pair31(-.548399),85(-.441072),69(-.310513),
  47(-.297378),95(-.290385); gains34(+.556537),33(+.323282),13(+.256410),89(+.203285),
  51(+.202386). Useful visual-review targets; new medical images were not uploaded or
  inspected locally, so do not invent anatomical explanations for individual cases.
  Decision: do not promote this candidate over the old model on current clinical
  evidence. Retain the old model as the clinical baseline, inspect the largest failures
  and synthetic/clinical target mismatch before another full training run. Do not tune
  thresholds or choose another checkpoint on this cohort and call it an untouched test.
  No new inference, model/config/annotation changes or medical-image publication here.

- **2026-10-18 - Fixed clinical launch guard's checkpoint/JSON spec comparison.**
  Campus user hit an unlabelled AssertionError at line11 of the supplied guard before
  inference. In that command, line11 directly compares native checkpoint head_spec to
  exported JSON. Trainer uses DECODER_OUT_RANGE=(-1.0,1.0), while JSON restores that
  tuple as a list; direct dictionary equality rejects an otherwise identical head.
  Normalize checkpoint spec through json.dumps/loads before equality, retaining the
  real architecture check. Added descriptive messages for every other config/cohort
  assertion. Regression executes the actual README guard with a saved native checkpoint
  and500 synthetic annotation references: matching normalized spec passes, changing
  head channels96->97 still fails. Focused regression and all7 clinical tests passed.
  This fixes the local
  reproduction, not proof of the uninspected campus checkpoint's contents. Retry the
  corrected guarded launch; do not bypass epoch/config/cohort/hash verification.
  Campus verification and clinical inference remain pending; no clinical results yet.

- **2026-10-18 - Clinical evaluation of cutoff.0001 best checkpoint approved/prepared.**
  User approved the next step after final-run review. Reuse evaluate_clinical and
  run_clinical100.sh for expected epoch49 in run_last1_rot90cw_mask_thr0p0001_fp075.
  Existing launcher was pinned to the old epoch45 hash; added an explicit
  EXPECTED_SHA256 override while preserving that default, threshold.5 and100 cases.
  Focused native-Bash test confirms both default and overridden hash argument paths.
  Documented full campus launch in configs/README.md: verify epoch49, signed target
  .0001, identical head/rotation3/prediction.5, and all500 annotation hashes against
  review_outputs/clinical100_20261017_112038/metadata.json, then pin checkpoint SHA256,
  run data preflight and start inference in a fresh clinical100_cutoff0001_epoch49
  output directory. No clinical tuning, preprocessing, postprocessing, model or loss
  change. Known local checkpoint locations have no new weights; campus verification
  and inference remain pending. Do not claim clinical scores or checkpoint hash yet.
  Validation: all6 clinical tests passed, both documented-launch/launcher Bash syntax
  checks and inline-guard Python parse passed, touched files have no editor errors.
  Fresh local DNS check cannot resolve cas703, so campus execution requires the user.
  Keep medical NIfTI/PNG outputs private; return only report/per-pair metrics/metadata
  for comparison. Do not stage unrelated clinician/LLM work or whole review folders.

- **2026-10-18 - Cutoff.0001 run completed60 epochs; next candidate is logged best49.**
  Pulled5dda4d5a, results/cutoff0001_final_20261018_082525: validation JSON and sweeps
  38-60, train JSON60, metrics.csv exactly1-60. Combined the earlier snapshots for1-37.
  Verified identical configurations, fixed validation coverage2108=1293 pathology+
  815 nuisance, all46080 sweep rows' integer TP/FP/FN and pooled overlap formulas,
  all validation group/epoch weighted-loss decompositions, epoch60 train decomposition,
  and nuisance means matching metrics.csv. No threshold optimization or new inference.

  | Epoch | Matched unsigned macro Dice | Matched signed macro Dice | Signed precision | Signed recall | Nuisance FP mean | FP P95 | Printed GT.05 Dice |
  |---|---|---|---|---|---|---|---|
  |23|.68613|.66068|.62552|.79500|.02960|.19057|.35720|
  |37|.68346|.65954|.67656|.75023|.01432|.10285|.37312|
  |49|.67819|.65410|.68470|.73396|.01106|.06334|.37739|
  |60|.67682|.65185|.68219|.73174|.01113|.06521|.37700|

  Matched columns are pathology pair means with valid denominators at raw-GT>.0001,
  prediction>.5. FP columns are nuisance image-area fractions, not affected-case rates.
  Across1-60, best unsigned/signed matched macro Dice is23 (.6861285/.6606753), best
  printed selection Dice49 (.3773876), best matched pooled signed Dice49 (.7679145),
  lowest mean FP and FP P95 both57 (.0110187/.0612087), lowest diagnostic val loss17
  (.8092077). Do not substitute one definition for another or declare one universally
  optimal epoch. Difference in macro signed Dice23->49 is only-.00657 observed; no
  uncertainty estimate or statistical superiority claim is available from aggregates.
  Late training primarily improves precision/nuisance suppression at a recall cost.
  Compared with23, epoch49 signed precision rises.62552->.68470, recall falls.79500
  ->.73396; nuisance FP mean2.96%->1.11%, P9519.06%->6.33%. Worst nuisance FP at49
  is49.81% of image area, so severe outliers remain. Epoch60 is1.11%/6.52%/47.80%.
  Pooled signed Dice improves23 .75053->49 .76791, then60 .76669; per-pair macro
  Dice and foreground fidelity do not show the same improvement. Epoch49 pooled
  unsigned P/R/Dice=.75315/.83644/.79261; signed=.72968/.81038/.76791. Pathology
  predicted area29.68% vs target26.73%, ratio1.1106, versus33.87%/ratio1.2671 at23.

  | Anomaly | Epoch23 signed macro Dice | Epoch49 signed macro Dice | Epoch60 signed macro Dice | Epoch49 signed precision | Epoch49 signed recall |
  |---|---|---|---|---|---|
  |Consolidation|.58822|.56399|.56045|.59804|.66932|
  |Fluid overload|.68855|.69378|.69093|.66876|.83508|
  |Pleural effusion|.71103|.70302|.70142|.75959|.72055|
  |Pneumothorax|.65872|.65825|.65706|.70604|.71521|

  Consolidation remains weakest and loses the most matched Dice/recall late in training.
  Epoch49 area ratios: consolidation1.2453, fluid1.2931, effusion.9398, pneumothorax
  1.0793. Aggregate area agreement does not imply anatomically correct localization.
  Foreground generalization deteriorates: diagnostic val loss17 .80921->23 .82254
  ->49 .93114->60 .94393, while train23 .39882->49 .23075->60 .22079. Val foreground
  contribution23 .71387->60 .87190 rises while background.10867->.07203 falls. Printed
  foreground L1/L2 rise23 .21169/.25427->60 .26425/.31205. This supports overfitting
  to training foreground plus a precision/recall tradeoff, not numerical collapse or
  proof that every quality metric worsens. Train is online replacement-sampled and
  val fixed-model; the absolute train/val gap is not a like-for-like estimate.
  Numerical health: no invalid validation pairs or nonfinite loss/gradient steps in
  all60 summaries; each2058 successful batches/514 optimizer steps/2 pending batches.
  Known accumulation remainder unchanged, not fixed during the experiment. Final train
  invalid pairs0;16464 draws/9725 unique pairs/4909 groups,13109 pathology/3355 nuisance.
  Final grad norm mean.60184/max.78337, epoch518.93s, peak2.45GiB, recorded LR0.
  Source still overwrites last.pt every epoch and best.pt only when original GT.05
  Dice improves. Expected retained weights: best.pt epoch49, last.pt epoch60. Weights
  were not uploaded/loaded; verify checkpoint metadata before inference. Epoch23 was
  not an original-metric best at that time, so its weights are not known to be retained
  and cannot be reconstructed from diagnostics. Do not assume they can be selected.
  Final recommendation: do not extend this run. Evaluate the stored best candidate49
  against clinician annotations with the same fixed protocol as the earlier model;
  inspect severe nuisance/consolidation errors before deciding another training run.
  Synthetic performance is promising but clinical benefit and improvement over the old
  model remain unproven. For future runs save matched-target best separately, define
  checkpoint/early-stopping criteria before launch, and test regularization as a
  separate ablation, not an inferred guaranteed fix. No training/loss/sampler/threshold/
  selection changes, checkpoint manipulation, or clinical/test evaluation performed.

- **2026-10-17 - Retrieved epochs15-37; matched macro Dice plateaus while FP falls.**
  Pulled c6c1aca0, results/cutoff0001_epoch37_20261017_192724: full validation JSON
  and threshold sweeps15-37, train JSON37, metrics.csv through38. Combined with the
  earlier epoch14 snapshot to verify all37 configurations and validation coverage
  (2108 pairs=1293 pathology+815 nuisance), all28416 sweep rows' pooled overlap formulas
  and integer counts, all validation groups' weighted-loss decompositions, and epoch37
  train decomposition. Config, GT and prediction thresholds remain unchanged.

  | Epoch | Matched unsigned macro Dice | Matched signed macro Dice | Signed precision | Signed recall | Nuisance FP mean | FP P95 | Printed GT.05 Dice |
  |---|---|---|---|---|---|---|---|
  |14|.67329|.64951|.62850|.77965|.02737|.18812|.35716|
  |23|.68613|.66068|.62552|.79500|.02960|.19057|.35720|
  |36|.68354|.66040|.67512|.75376|.01406|.09152|.37284|
  |37|.68346|.65954|.67656|.75023|.01432|.10285|.37312|

  Matched scores use raw-GT>.0001 and prediction>.5 on pathology pairs; FP values are
  nuisance image-area fractions, not affected-case rates. Among full diagnostics1-37,
  epoch23 is best on matched unsigned/signed macro Dice, epoch33 on mean nuisance FP
  (.0136005), epoch36 on FP P95 (.0915218), epoch17 on validation diagnostic loss
  (.8092077), and epoch37 on original printed Dice. Different objectives do not select
  the same checkpoint. Epoch36 signed Dice.66040 is very close to23 .66068; do not
  present these small observed differences as statistically established superiority.
  By37, predictions are more conservative than23: signed precision rises.62552->.67656
  while recall falls.79500->.75023. Mean pathology predicted area33.87%->30.45% against
  unchanged GT26.73%; pooled predicted/reference ratio1.267->1.139. Macro Dice is flat,
  but pooled unsigned Dice improves.77419->.79151 and signed.75053->.76757, so it is
  incorrect to say every overlap measure has stopped improving. Epoch37 pooled
  unsigned precision/recall=.74310/.84667 and signed=.72062/.82105. Macro emphasizes
  pairs equally; pooled metrics weight by pixel counts. Both must be reported clearly.
  Epoch37 nuisance FP mean1.43%, P9510.29%, max42.98%, versus epoch14 2.74%/18.81%/
  48.41%. Large worst-case errors remain despite substantial suppression of the mean.

  | Anomaly | Epoch14 signed macro Dice | Epoch23 signed macro Dice | Epoch37 signed macro Dice | Epoch37 signed precision | Epoch37 signed recall |
  |---|---|---|---|---|---|
  |Consolidation|.56728|.58822|.57590|.59195|.70217|
  |Fluid overload|.67889|.68855|.69652|.66063|.85031|
  |Pleural effusion|.69806|.71103|.71072|.75746|.73265|
  |Pneumothorax|.65586|.65872|.65874|.69071|.72181|

  Consolidation remains weakest and loses Dice/recall relative to23. Fluid improves;
  effusion/pneumothorax are approximately flat in Dice, with precision/recall tradeoffs.
  Epoch37 area ratios: consolidation1.3273, fluid1.3238, effusion.9606, pneumothorax
  1.0960. No source of medical/anatomical validity is established by these synthetic
  reference masks. This is not proof of clinical benefit or superiority over the
  earlier .05-target model; do not compare different-reference Dice as equivalent.
  Numerical health across1-37: invalid pairs0, nonfinite losses/gradient steps0;
  2058 successful batches/514 optimizer steps/2 pending batches each epoch, unchanged
  known remainder. Epoch37 grad norm mean1.070/max3.127, epoch510.29s, peak2.45GiB.
  Train total.53938 at14->.39882 at23->.28375 at37, while validation diagnostic loss
  .83100->.82254->.89119; FG contribution.72260->.71387->.81371 and BG.10840->.10867
  ->.07748. Foreground validation errors worsen while train loss falls: overfitting/
  generalization warning, not numerical collapse. Train is online replacement-sampled,
  so the train/val means are not an exact like-for-like generalization-gap estimate.
  Epoch38 has metrics only: printed Dice.371234, nuisance FP.013591, train.273675,
  foreground L1/L2.249095/.290015; no full health/matched diagnostics for38 received.
  Under the existing selection rule, best in metrics1-38 is37, not latest38. Checkpoint
  weights were not uploaded. Only best/last are automatically retained; epoch23 was not
  an original-metric best then, so do not assume its weights remain recoverable from
  these diagnostic files or propose selecting an unavailable checkpoint.
  Recommendation: no mid-run loss/sampler/threshold/selection changes. The head is a
  promising but largely plateaued synthetic candidate with better nuisance control;
  do not expect large macro-Dice gains from more epochs alone. Finish the planned60
  unchanged if the remaining budget is acceptable, without extending solely for the
  printed Dice. Next useful assessment is the stored best checkpoint on clinician
  annotations under the same comparison protocol, not clinical-set tuning. No clinical/
  test evaluation, new training run, checkpoint manipulation or running-job change made.

- **2026-10-17 - Retrieved epochs1-14 diagnostics; metrics extend through epoch17.**
  Pulled e03d46eb, results/cutoff0001_epoch14_20261017_162415. Full validation JSON
  and threshold sweeps cover1-14; training JSON covers14; metrics.csv covers1-17.
  Configuration matches run_config.json in every validation summary: training target
  cutoff.0001, frozen encoder, FG11/BG.75, L1+L2 only, original selection GT.05/pred.5.
  Each validation pass has2108 pairs,1293 pathology and815 nuisance. Independently
  checked pooled precision/recall/Dice/IoU formulas in all10752 sweep rows, and weighted
  L1/L2 foreground+background sums in every validation group and epoch14 train.

  | Epoch | Matched unsigned Dice | Matched signed Dice | Unsigned precision | Unsigned recall | Nuisance FP area mean | Printed GT.05 Dice |
  |---|---|---|---|---|---|---|
  |4|.62901|.60165|.55122|.84518|.11820|.31182|
  |8|.65814|.63503|.61492|.81489|.04934|.33997|
  |10|.65758|.63034|.59508|.84583|.06215|.32808|
  |11|.66479|.63709|.60527|.84159|.05025|.33488|
  |12|.65676|.63107|.60764|.83106|.04453|.33271|
  |13|.65933|.63513|.61575|.82565|.03777|.33918|
  |14|.67329|.64951|.65561|.80578|.02737|.35716|

  Matched metrics are pathology per-pair means at raw-GT>.0001 and prediction>.5,
  not the printed GT.05 metric. Epoch14 is best among uploaded1-14 on unsigned and
  signed matched Dice, original printed Dice and lowest mean nuisance FP. Improvement
  since4 is mainly precision with a modest recall tradeoff, not expanded foreground.
  Epoch14 signed macro precision/recall=.62850/.77965; pooled unsigned P/R/Dice=
  .68581/.86920/.76669, pooled signed=.66602/.84412/.74457. Pooled and macro scores
  answer different questions; do not interchange them. Pathology predicted area falls
  from40.86% to33.88%, with target26.73% unchanged; pooled predicted/reference area
  ratio1.529->1.267. Nuisance FP mean improves11.82%->2.74%, P9540.11%->18.81%,
  max88.19%->48.41%; those are image-area fractions, not affected-case percentages.
  Important false-positive tails therefore remain despite the mean improvement.

  | Anomaly | Epoch4 matched signed Dice | Epoch14 matched signed Dice | Epoch14 signed precision | Epoch14 signed recall | Epoch14 pooled predicted/reference area |
  |---|---|---|---|---|---|
  |Consolidation|.51555|.56728|.54604|.73305|1.5132|
  |Fluid overload|.64855|.67889|.61415|.88050|1.4537|
  |Pleural effusion|.66325|.69806|.71788|.74596|1.0446|
  |Pneumothorax|.58667|.65586|.63666|.76194|1.2426|

  All four pathologies improve signed Dice; consolidation remains weakest and has the
  largest excess predicted area, followed by fluid overload. No proof of anatomical
  validity or clinical improvement follows from these synthetic validation scores.
  Do not compare matched .0001 Dice directly to the earlier run's GT.05 Dice.6002.
  No new clinical100/test evaluation or validation-threshold tuning was performed.
  Numerical health is stable across1-14: invalid pairs0, nonfinite losses0, nonfinite
  gradient steps0; each epoch2058 successful batches/514 optimizer steps with the
  previously documented2-batch accumulation remainder unchanged. Epoch14 preclip grad
  norm mean2.268/max8.775; epoch528.73s, CUDA peak2.45GiB. Training loss falls1.00781
  at4->.53938 at14; train FG share73.08%. Validation diagnostic loss.91519->.83100,
  mostly from BG contribution.21330->.10840; FG.70189->.72260 is slightly worse.
  Validation loss has largely plateaued around.825-.837 at10-14 despite falling train
  loss. This cautions against claiming uniformly improving generalization; online
  replacement-sampled train means and fixed-model validation means are not directly
  comparable. No collapse, but monitor the growing optimization/generalization gap.
  Additional metrics-only epochs:15 printed Dice.36344/nuisance FP.01994/train.51555;
  16 .36131/.02534/.50283;17 .34438/.03384/.48704. Under the unchanged selection rule,
  epoch15 is the best in this metrics snapshot, not14 or the latest17. Checkpoint files
  were not uploaded, so their contents are not inspected. No matched-target metrics or
  detailed health were received for15-17; their target-matched ranking is unknown.
  Recommendation: continue the unchanged run to a review around25, compare the same
  fixed-threshold matched metrics and FP tails, and retain best rather than assume
  the last checkpoint is best. No reason yet to change loss, sampler or threshold
  mid-run. No training/config/optimizer/selection or running-job changes made.

- **2026-10-17 - Retrieved epoch4 diagnostics; metrics snapshot also includes epoch5.**
  Pulled08c75764, results/cutoff0001_epoch4_20261017_144309: train/val epoch4 JSON,
  val threshold CSV, run_config.json and metrics.csv through epoch5. Confirmed signed
  target cutoff.0001, FG11/BG.75, L1+L2 only, eval prediction.5/GT.05, seed42 and
  balanced replacement sampling. Independently recomputed micro precision/recall/Dice/
  IoU from integer-valued counts for all768 threshold rows; CSV count fields serialize
  with .0 suffixes. Verified weighted L1/L2 equal the sum of recorded foreground and
  background contributions for overall/pathology/nuisance groups in both splits.

  | Epoch4 validation reference | Unsigned macro Dice | Signed macro Dice | Unsigned precision | Unsigned recall |
  |---|---|---|---|---|
  | Raw GT>.0001, prediction>.5 |.62901|.60165|.55122|.84518|
  | Raw GT>.05, prediction>.5 |.32108|.31743|.21149|.91689|

  Table values are pathology per-pair means with valid denominators. Printed legacy
  Dice.3118158 includes empty-reference pathology cases differently from diagnostic
  Dice.32108; do not substitute them or confuse unsigned with signed overlap. Matched
  .0001 scores must NOT be compared numerically to the prior run's .05-reference .6002
  as evidence of improvement. Checkpoint selection remains on the original metric.
  At GT.0001/pred.5, pooled unsigned precision/recall/Dice=.58107/.88821/.70254,
  while signed=.56154/.85835/.67892. All1293 pathology references are nonempty;
  16 predictions are empty and1268 pairs have at least one unsigned GT hit. Mean
  predicted pathology area40.86% vs target26.73%, ratio1.529: model learns expanded
  support but overpredicts it. Wrong-sign overlap fraction averages3.14% on valid
  pairs; overlap/false-positive control is a larger visible deficit than polarity.
  Matched signed macro Dice by anomaly: consolidation.51555, fluid overload.64855,
  effusion.66325, pneumothorax.58667. Consolidation pooled predicted/reference area
  ratio2.074 is the largest, versus1.685 fluid,1.237 effusion and1.488 pneumothorax.
  Raising prediction threshold.5->.75 in this exploratory sweep increases matched
  unsigned Dice.62901->.63759 and signed.60165->.61549, but nuisance FP remains7.46%
  vs11.82%. Do not change the active threshold or claim this validation sweep is a
  held-out result. Component F1 in JSON still refers to primary GT.05, not GT.0001.
  Epoch4 weighted train loss has79.33% foreground contribution; val76.69%. Observed
  training sampling is13207 pathology/3257 nuisance (80.22%/19.78%), versus natural
  val1293/815 (61.34%/38.66%). Source confirms inverse-anomaly-frequency replacement
  sampler equalizes four pathology categories plus one nuisance category; it does not
  balance total pathology versus nuisance 50/50. Epoch train16464 rows cover9691
  unique pairs/4852 groups, whereas val2108 unique pairs cover703 groups. Online
  train and fixed-model val means are therefore not directly comparable estimates.
  Health: no invalid prediction/GT pairs, nonfinite losses or nonfinite gradient norms;
  2058 successful batches,514 steps, preclip grad norm mean4.633/max14.731. Clip1 is
  active. Metadata reports579.67s per epoch (train526.11s), peak CUDA allocation2.45GiB.
  A pre-existing accumulation remainder leaves2 batches unstepped at epoch end;
  train_one_epoch zeros them on next entry. This affects16 sampled occurrences
  (<0.1%), not a plausible sole cause of large FP. No mid-run optimizer fix applied.
  Additional epoch5 metrics: train total.8851623, val Dice.3314069/IoU.2281510,
  sens+.860919/-.826529, nuisance FP.0612955, down from.1181964 at4. Best printed
  selection Dice is now epoch5, superseding epoch2 from the earlier log-only report.
  No epoch5 threshold/health artifacts were uploaded; do not infer their scores.
  Recommendation: continue the current experiment unchanged and inspect the same
  matched-target/precision/nuisance diagnostics around epoch10. Current data support
  learning plus excess foreground, not training collapse or proven clinical benefit.
  Sampling and weight exposure are plausible contributors, not isolated causal proof.
  No training/loss/config/selection or running-job changes made after this review.

- **2026-10-17 - Campus cutoff0.0001 run started; user supplied epochs1-4.**
  Supersedes the preceding launch-pending note. Run uses published32668a5b and starts
  fresh with no last.pt in run_last1_rot90cw_mask_thr0p0001_fp075. Cache has20755 pairs,
  98.3GB; audited split coverage train16464/val2108/test2183 and manifest SHA256
  78d18b1c8c6c34aae2023f2c8342fba28b8109ca0343f81be4939ee849c54468. Cached test coverage
  does not mean test evaluation: reported loader is2058 train/264 val batches. Head
  has2.06M trainable parameters, native-last1 patch_dim768. Diagnostics report16464
  online sampled train rows and2108 val rows each epoch, not16464 unique train IDs.
  Runtime now reports torch2.14.1+cu130, GPU free24.8/25.3GB; this differs from earlier
  recorded campus torch2.5.1+cu121 and adds an environment difference for comparisons.

  | Epoch | LR | Train total | L1 | L2 | Inactive Dice term | Inactive sign term | Val Dice | IoU | L1ch | L2ch | Sens+ | Sens- | Nuisance FP mean | P95 | Max |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  |1|1.50e-4|2.2310|1.1056|1.1254|.6318|.0603|.2861|.1897|.3357|.3758|.758|.760|.1573|.4367|.7225|
  |2|3.00e-4|1.4228|.6929|.7299|.5527|.0332|.3190|.2178|.3003|.3127|.792|.748|.0902|.3043|.6538|
  |3|3.00e-4|1.1437|.5559|.5879|.5158|.0267|.3150|.2156|.2449|.2948|.845|.815|.1282|.4060|.8336|
  |4|2.99e-4|1.0078|.4866|.5212|.4947|.0249|.3118|.2112|.2109|.2508|.874|.831|.1182|.4011|.8819|

  Observations are from user logs, not downloaded epoch artifacts. Optimization and
  target-region errors improve. Best printed selection Dice so far is epoch2 .3190;
  epoch4 correct-direction sensitivity is .874/.831, but nuisance FP occupies11.82%
  of image area on average,40.11% atP95 and88.19% worst case. This FP metric is an area
  fraction, not a percentage of cases with false positives. No nonfinite warning is
  shown, but gradient/step health cannot be certified without diagnostics JSON.
  Verified evaluate(): printed Dice/IoU are unsigned support overlap at prediction.5
  versus raw-GT.05; signed sensitivities are separate. Training targets use raw-GT
  .0001, so intended support expansion can reduce the old-reference Dice. That mismatch
  cannot explain nuisance FP because zero-GT controls remain zero at both cutoffs.
  Larger foreground support with unchanged11/.75 weights plausibly encourages broad
  predictions, but logs alone do not establish the cause. Train Dice/sign terms remain
  observational (weights0); they are not active additions to L1+L2. Early training is
  not comparable to the previous converged best checkpoint, and train losses across
  the two different target definitions are not directly comparable.
  Recommendation: retain this run unchanged for an early review around epoch10,
  inspect epoch4 train/val JSON plus val threshold CSV now for precision, matched-GT
  .0001 signed/unsigned overlap, per-anomaly behavior and nuisance FP/health. If high
  nuisance FP persists, consider a separate weighting ablation, not a mid-run change.
  No model/config/threshold or running-job changes made in response to these logs.

- **2026-10-17 - User chose training target cutoff0.0001 for the next ablation.**
  Added configs/last1_rot90cw_mask_thr0p0001_fp075.yaml with signed-mask cutoff0.0001,
  fresh run_last1_rot90cw_mask_thr0p0001_fp075 checkpoint/plot paths, diagnostics every
  epoch, GT diagnostic thresholds0/.0001/.02/.05 and connected-component diagnostics.
  Shared run_rotated_mask.sh now honors HEAD_CONFIG while preserving its original
  default. Campus command explicitly exports both HEAD_CONFIG and RUN; guarded cache
  checks, split audits, locking and same-run resume remain in place. Reuse existing
  native-last1 rotated features; do not copy or resume the previous run's checkpoint.
  Model/frozen encoder, rotation3, seed42, optimizer,60epochs, L1+L2, foreground11 and
  background0.75 unchanged. Primary evaluation/selection stays prediction0.5/GT0.05
  for comparison; extra pixelwise sweeps include GT0.0001. Components remain at primary
  thresholds. Signed conversion occurs before tau0.02, so low-amplitude pixels surviving
  the new cutoff remain unit foreground targets, not discarded by tau. Boundary test
  verifies strict > cutoff, correct signed targets, weighted objective/gradients,
  active background supervision, diagnostic parity, and unchanged baseline settings.
  Audit at .0001 retained93.05%/93.18% train/val nonzero support, >99.99% pooled absolute
  GT mass, and erased no nonempty maps. This is user-selected, not proven optimal;
  selected spatial GT examples remain unreviewed locally. No actual campus training
  launched. Full setup/launch instructions saved in configs/README.md.
  Verification: all40 CPU regression tests passed in44.125s; both rotated-mask and
  audited training launchers passed bash -n. The new boundary/config test passed.

- **2026-10-17 - Full campus GT audit retrieved and independently verified.**
  Pulled campus result commit64c07dbd, results/gt_audit_20261017_125959. Audit selected
  and processed all18572 records: train16464 (9508 nonempty GT,6956 zero controls),
  val2108 (1293 nonempty,815 zero controls); no test rows and no reported issues.
  Independently streamed all55716 per-pair/stage rows and reproduced every summary
  pair/nonempty count and support/entire-map-loss statistic at cutoffs0,.0001,.001,
  .01,.02,.05. Pair/stage identities are unique. Every native GT is already512x512;
  native and training-float32 aggregate support statistics match, so resizing in this
  pipeline does not explain the tiny tails. No out-of-range or nonfinite GT pixels.
  Viewed cutoff_diagnostics.png, but selected actual GT overlays/NPZ were intentionally
  not pushed and have not been inspected locally. Statistical conclusions below do
  not certify anatomical validity of every nonzero pixel or explain clinical F1 alone.

  | Split | Support retained at .05 | Absolute GT mass retained, pooled | Entire nonempty maps erased |
  |---|---|---|---|
  | Train |33.6271%|75.8989%|430/9508 (4.52%)|
  | Val |34.0876%|77.3766%|56/1293 (4.33%)|

  Validation support retained at .05 by pathology: consolidation48.0582%, fluid
  overload49.6760%, pleural effusion38.2633%, pneumothorax16.6947%. Entire maps erased
  are5,14,9,28 respectively, reproducing the prior56-case observation while revealing
  substantial partial-support loss in other maps. Exact nonzero masks average28.63%
  (train) /28.68% (val) of image area on nonempty cases, versus9.63%/9.78% at .05.
  At .001, support retained is85.6609%/85.7988%, pooled absolute mass99.9335%/99.9366%,
  and no nonempty maps disappear. At .0001, support93.0516%/93.1807%, mass99.9948%/
  99.9951%, no maps disappear. These are sensitivity checks, not validated optimal
  thresholds. Per-pair mean mass retention at .05 is56.58%/56.12%; do not confuse this
  macro statistic with pooled mass retention75.90%/77.38%.
  Exact-nonzero masks contain fragmentation: pneumothorax averages118.06/115.04 signed
  8-connected components per nonempty pair. Tiny tails therefore cannot be assumed
  negligible once sign(GT) promotes them to unit labels. Mixed-sign raw GT occurs in
  5038/9508 train and661/1293 val nonempty maps; sign direction should be reviewed too.
  Simulated float16 removes only0.02685%/0.02529% of nonzero pixels and flips13404/1708
  pixel decisions at .05, without changing erased-map counts430/56. It is not the main
  explanation for large .05 support loss; actual cache read-back remains unaudited.
  No nonempty native GT array hash is shared across train/val; the one shared hash is
  the empty array. This is not proof of patient/change-group split independence.
  Recommendation: prioritize GT support/cutoff review before a more complex loss.
  Inspect actual selected masks, then make a controlled target-cutoff ablation with
  frozen encoder, same head/L1+L2/background penalty and fixed evaluation definitions.
  Do not declare exact-zero targets correct from these scalar statistics. Holding
  weights11/.75 while moving .05 to0 also raises average all-pixel weight budget from
  about1.320 to2.445 on train; target expansion changes effective weighting even when
  flags stay fixed. Region-normalized weighting would be a separate ablation. No
  config, loss or training changes made after receiving the audit.

- **2026-10-17 - Implemented original-GT audit and loss-independent epoch diagnostics.**
  User requested evidence for choosing the next loss and rich train/val statistics,
  including terms not used by the objective. Added evaluation/audit_gt.py and
  evaluation/training_diagnostics.py; integrated detached telemetry into the existing
  trainer forward passes. No real training or full campus audit was launched. No loss,
  frozen encoder, target cutoff, optimizer policy, legacy metrics.csv or best-checkpoint
  selection change. No new claim that a no-cutoff loss is better.
  Audit defaults to complete train/val only, with native/preprocessed/simulated-float16
  GT statistics, cutoff support/mass retention, entirely removed maps, magnitude
  histograms, components, provenance hashes, GT-only contrast overlays and real selected
  NPZ arrays. Float16 is a simulation, not an actual cached-GT read-back. Output must be
  new; per-case issues are retained and fail the CLI. Selected medical arrays must not
  be published automatically. Campus setup/commands and output semantics documented in
  configs/README.md. CPU shared-storage runtime remains unmeasured on the full dataset.
  Epoch diagnostics default on: per-pair gzip CSV, threshold-sweep gzip CSV and grouped
  JSON for train/val. Logs raw/sign/configured-target errors, weighted FG/BG terms,
  inactive Dice/sign terms, signed/unsigned/positive/negative overlap, nuisance FP,
  score/area tails, direction/anomaly groups, unique IDs and training-health metadata.
  Optional components report pooled +/-/combined F1, HMDR and unmatched detections,
  using reference-first largest-overlap greedy matching without PAI boost or cleanup.
  Synthetic component F1 is not the clinical per-clinician metric. Empty-reference
  Dice/recall are undefined and carry valid denominators. Training telemetry is online
  sampler-observed, not a fixed-model full-train evaluation. Existing pending-gradient
  and nonfinite-step behavior is exposed, not repaired. Diagnostics incur CPU overhead;
  every-N-epoch and disable flags are available. Regression fixtures verify actual
  NIfTIs, preprocessing parity, cutoff removal, held-out exclusion, real NPZ contents,
  new-output guard, legacy matching equivalence, detached stats, train/resume reports,
  and bit-identical optimizer updates with diagnostics enabled/disabled.
  Final local verification: all39 training.test_dpt_head CPU tests passed in10.864s;
  touched Python diagnostics and scoped git diff --check were clean. Visually checked
  nonempty negative-GT and empty-GT fixture overlays plus the cutoff summary figure.
  Fixture figures are not real dataset findings; campus original-array results remain
  pending. Related earlier clinical-comparison source is included for test dependencies,
  without publishing generated clinical images or unrelated clinician/LLM work.

- **2026-10-17 - Raw-GT visual review requested and gallery saved.**
  User requested raw GT overlays. Local samples/reuse_test contains18 original
  current_with_differences.png previews across6 projection/sample folders and3
  variants each, but no diff_map.nii.gz arrays. Created
  review_outputs/raw_gt_overlays_20261017/index.md with all18 preview links,
  six embedded examples and two below-cutoff synthetic-validation panels.
  Original previews are unchanged; their colorbars differ. Do not reconstruct
  exact GT support or tiny-value tails from PNG colors. Viewed three generator
  previews (10000_a_1,10001_a_1,10002_a_1 variant0) and two validation panels.
  Recomputed counts from saved per_pair_val.csv:2108 total,1293 nonzero GT maps,
  56 with peak<=.05 (pneumothorax28,fluid_overload14,pleural_effusion9,
  consolidation5). Their nonzero GT is entirely removed by abs(GT)>.05; this
  does not establish that all low-amplitude pixels are true lesions or measure
  partial support loss in the other maps. Example peaks .0451355 and .0427856
  show empty thresholded masks despite nonzero raw GT. Existing GT zoom panels
  share limits with prediction and can remain +/-1, hiding faint GT visually.
  Verified all18 PNGs decode and all27 gallery references resolve. No new raw
  rendering, zero-cutoff array audit, training/config changes or inference run.

- **2026-10-17 - Discussed threshold-free signed GT as the next loss ablation.**
  User asked why clinical per-clinician F1 is poor and proposed masking GT inside
  the loss without an amplitude threshold. Verified current config: signed_mask,
  target_threshold .05, tau .02, foreground weight11/background .75, L1+L2 only.
  Existing implementation with target_threshold0 already equals sign(raw_GT):
  exact zeros remain background and all nonzero values become +/-1. After this
  conversion tau .02 does not discard any nonzero target. Numerical probe checked
  both cutoff settings against direct weighted L1+L2, including +/-1e-8 inputs,
  and confirmed background prediction gradients remain active at cutoff0.
  Caveat: GT is bilinearly resized before loss; raw nonzero support must be checked
  for low-amplitude tails/noise before treating every nonzero as a true finding.
  No local GT support audit has been performed in this discussion. More unmatched
  components and low clinician F1 do not isolate target cutoff as the cause;
  synthetic-to-clinical domain/target mismatch and native postprocessing differences
  remain plausible. Proposed controlled sign-target ablation with frozen encoder
  and otherwise unchanged recipe; keep current evaluation definitions fixed for
  comparability. Do not restrict loss to foreground only: background supervision
  is still needed. No loss/config/prediction changes or real training performed.

- **2026-10-17 - Clarified requested F1: each individual clinician, not mean.**
  User pointed to no_cc_itamar_plus_lmm5_100 and requested our model against each
  clinician in the same style. Exported itamar_comparison/f1_per_clinician.md/.csv:
  positive/negative/combined pooled detection F1 for both models vs each of five
  clinicians, on100 and98 pairs (20 rows,60 exact score cells). Linked all six
  existing foundation per-detection/per-pair matrix plots per cohort, preserving
  reference layout and clearly separating detection F1 from boosted per-pair PAI.
  Previous averaged columns now explicitly say Mean F1; not substituted for the
  requested individual scores. Our100 combined F1 vs Avi/Benny/Sigal/Smadar/Nitzan:
  .255814/.235081/.272904/.284644/.336283. Every cell matches JSON and CSV matrices;
  all linked plots verified present. Added links from both review indexes; no
  inference, metric-formula, annotation, operating-point or training changes.

- **2026-10-17 - Requested compact HMDR / UDPP / F1 tables saved.**
  Added export_metric_tables to the comparison runner and exported
  itamar_comparison/hmdr_udpp_f1.md and .csv, linked from both review indexes.
  Each cohort has five clinician rows plus historical Itamar and foundation rows;
  reports sign-specific HMDR, mean UDPP (SD in CSV), and positive/negative/combined
  detection F1. This is pooled component-matching F1 averaged vs reference readers,
  NOT pixel Dice or boosted per-pair PAI. Human references exclude that human and
  both models (four peers); each model uses all five humans. 100-pair combined F1:
  Itamar .4646300560 vs foundation .2769454384; pure98 .4682591540 vs .2778287261.
  Reused existing validated report matrices; no prediction reload, threshold tuning
  or inference rerun. Verified all14 rows, reference pools, F1 bounds and model100
  values against source reports. 100-pair baseline still includes lmm5 pairs99/100.

- **2026-10-17 - Full clinical comparison against Itamar completed locally.**
  User requested all statistics previously made for Itamar on the 100 pairs.
  Added evaluation/compare_clinical_models.py; extended the existing legacy OV
  loader/main with original-space and explicit threshold options, preserving old
  square-crop/zero-threshold defaults and all metric formulas. Saved six numerical
  agreement matrices/CSVs and cohort protocol alongside the original reports/plots.
  Outputs: review_outputs/clinical100_20261017_112038/itamar_comparison/index.md,
  comparison.csv/json, 1400 spatial metric rows, four complete legacy profiles
  (both models on 98 and 100 pairs), and 100 six-panel matched reviews.
  Total 208 artifacts / 51.4 MB. Profiles contain HMDR, UDPP mean/std, consensus
  sensitivity levels1-5, six PAI matrices/plots and six sensitivity plots, plus
  component-count summaries. All five historical JSON reports reproduced exactly
  for BOTH the 98-pair and 100-pair baseline. All human-only metrics and agreement
  matrix blocks are invariant across model runs. All 600 foundation CSV metric rows
  match the original clinical export. All 500 annotation hashes checked with exact
  raw or LF-normalized bytes. All 100 foundation masks match thresholded raw maps.
  Historical baseline on 100 pairs is MIXED: original Itamar pairs1-98, lmm5
  substitutes99/100. Separate matched98 results are included; do not label the
  mixed100 cohort as pure Itamar. No additional CC filtering for either model.
  Itamar's prior pre-save cleanup remains baked in; threshold its saved scores at0,
  foundation signed scores at0.5. This is native-operating-point comparison, not
  matched-FP calibration or an isolated encoder/head effect. Encoder stays frozen.
  100-pair results, historical baseline vs foundation:
    any-human correct-sign pair hits85/100 vs76/100;
    pixelwise3/5 consensus pair hits64/88 vs55/88;
    consensus mean signedDice .2150684 vs .1849128;
    consensus Dice>=.25:36 vs30; Dice>=.5:9 vs10;
    empty predictions10 vs12; positive/negative components116/87 vs217/151;
    HMDR positive .6551724 vs .5069124; negative .6896552 vs .4768212;
    UDPP positive .40 vs1.07; negative .27 vs .79;
    model-human PAI per-detection-all mean .4646301 vs .2769454;
    model-human legacy PAI per-pair-all mean .5505730 vs .4284842.
  Legacy per-pair-all retains empty-empty=1 and one-sided boost, NOT pixel Dice.
  Legacy consensus sensitivity counts connected annotation regions, NOT pair hits.
  On pure98: any-human hits83 vs74; consensus hits63/86 vs53/86; consensus Dice
  .2186374 vs .1880236. Of88 consensus-positive100 pairs, foundation Dice is higher
  on31, historical higher on42, equal on15 (tolerance1e-10); on86 pure98:30/41/15.
  Viewed paired panels1,30,85,94 only: foundation-localized/Itamar-empty pair30;
  Itamar-localized/foundation-empty pair94; pair85 favors foundation vs narrow
  consensus but baseline covers more of the any-human union. No clinical diagnosis
  inferred from these views. All16 existing ellipse-raster warning pairs retained.
  Added crop/original coordinate regression and fixed existing missing-seaborn test
  isolation (preload dependencies before patch.dict(sys.modules)); all35 CPU tests
  passed9.008s. No real model training, threshold tuning, annotation changes or
  added prediction cleanup. Reports/code/log are local and not committed/pushed
  by the agent in this analysis task. Main result: lower average clinical agreement
  and more unmatched components for the frozen-head run, with case-specific wins.

- **2026-10-17 - Clinical100 annotation-hit analysis verified.**
  User asked how many predictions hit clinician references. At fixed checkpoint
  threshold0.5, define a PAIR-level signed hit as >=1overlapping pixel of matching
  polarity; this is a lenient case-level overlap measure, NOT lesion detection rate.
  Recomputed100raw/mask threshold identities,500annotation hashes and600existing CSV
  metric rows. All500Windows JSON hashes differ only by CRLF conversion: LF-normalized
  bytes match campus hashes; first example also verified against the HEAD Git blob.
  Same-sign hits: any-clinician union76/100; pixelwise3-of-5consensus55/88 (62.5%).
  Ignoring sign: union78/100, consensus56/88. Per-clinician signed hits/annotated:
  Avi51/92, Benny52/97, Sigal59/94, Smadar57/92, Nitzan69/98.
  Consensus mean signedDice0.1849128, median0.1000495; Dice>=0.1:44/88,
  >=0.25:30/88, >=0.5:10/88. Of55hits,11haveDice<0.1;53have>=50matching pixels.
  Consensus direction-specific pair hits: positive31/56, negative26/46; a pair may
  contain both signs, so these do not sum to the overall unique-pair count.
  Consensus misses33/88:12empty predictions and21nonempty predictions with no signed
  overlap. Empty outputs: pairs10,24,34,42,46,59,61,66,67,70,77,94.
  Twelve cases have no pixelwise3-of-5reference region; all12have nonempty predictions
  (mean predicted area3.5821%). They are NOT necessarily clinically normal: all100
  cases have some individual-clinician annotation. Do not call these definite FP cases.
  Highest consensus Dice: pair85 .6335,53 .6125,52 .6061,69 .5963,20 .5948,
  96 .5843,93 .5603,31 .5484. No visual review of these clinical panels yet.
  Same16known raster-warning pairs remain; no annotation repair, cleanup, threshold
  tuning or model changes performed. Ellipse references are not dense pathology GT;
  results describe exploratory overlap, not clinical sensitivity or validated accuracy.

- **2026-10-17 - Clinical100 inference results received locally.**
  User published campus results; pulled main at d81eeda4 with --ff-only.
  Verified review_outputs/clinical100_20261017_112038: report says100completed pairs;
  all100review PNGs, raw output maps, signed masks and source JSONs are present,
  together with per_pair.csv, index.md and metadata.json. Metadata confirms epoch45,
  threshold0.5 and checkpoint SHA256
  0b7b49cc916806453ae5a2348e8b0546dd59dfbe5ff0449c034ba43d214a8053.
  Actual clinical inference is now complete; clinical performance/visual analysis
  has not yet been performed. Local HEAD and origin/main matched after the pull.

- **2026-10-17 - Campus clinical preflight blocked by unused seaborn import; fixed locally.**
  User successfully pulled main through7d32da40 on cas703, confirming publication of
  the clinical runner. Preflight then failed importing seaborn from the legacy OV
  module, before discovery/inference; guarded nohup command did not start a job.
  Moved seaborn import into its sole consumer, plot_matrix; annotation resolution,
  ellipse rasterization, checkpoint settings and predictions are unchanged.
  Full LOCAL preflight passed100pairs/500annotations with seaborn deliberately blocked;
  the same16known raster-warning pairs remain. Four clinical unit tests passed,
  including a new blocked-seaborn annotation-reader regression. No real clinical
  inference results yet. User subsequently requested immediate commit/push of this
  fix, regression test and log entries, leaving unrelated work untouched.
  Immediate campus workaround for the already-pulled code: install seaborn using
  the activated fcd_venv's python -m pip, then rerun guarded preflight/inference.

- **2026-10-17 - Apparently empty GT panel7bfdd75204f78981 analyzed.**
  User asked why Signed GT and GT zoom look empty and whether this affects loss.
  Case train_10833_a_1/pair1/variant1, negative pleural effusion: continuous GT
  peak0.06982421875, prediction peak1.0. Exporter's shared zoom=max(GTpeak,predpeak,
  threshold)=1, so GT zoom is identical to fixed[-1,1] and both are very pale, NOT empty.
  Actual GT mask has20721pixels (7.9044%); prediction mask18652pixels (7.1152%).
  Dice0.5735403, negative recall0.5449061: partial localization with extra/missing regions,
  not absent GT. Training call/config verified signed_mask at0.05; retained negative
  GT becomes-1, background0, foreground loss weight11/background0.75. Actual loss-function
  check with representative GT[-0.0698242,-0.04,0] ->target[-1,0,0] gave zero loss for
  exact target and positive loss for zero prediction. This illustrative check is NOT
  the full case's loss; original GT/prediction arrays are not in the review export.
  Display contrast does not affect training; thresholding changes supervised support.
  Exported per-pair l1 currently compares prediction to CONTINUOUS GT and is not the
  signed-mask training loss. Recommend explicit Continuous GT and Training target labels
  and a GT-only contrast view, keeping numerical values and scored masks unchanged.
  No plotting/loss code changed for this question.

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
