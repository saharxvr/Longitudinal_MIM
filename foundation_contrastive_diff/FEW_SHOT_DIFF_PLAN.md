# Few-shot signed change heatmaps (FSDiff) on frozen RAD-DINO blocks 6/9/12

Status (2026-11-27): **implemented and unit-tested locally on CPU; nothing has run on campus
yet** (no corpus, no benchmark, no training, no ICU numbers). The run order is in section 9.
Commands are in [configs/README.md](configs/README.md#few-shot-fsdiff-pipeline).

## 1. Question and design

Given a query pair (prior, current) and a support set of `n` pairs with signed GT maps that all
show the same kind of change (the "mission"), predict the signed heatmap of that mission in
the query. `n` ranges over 0-64 at test time. The training support size is the largest of
64/32/16 that passes the GPU benchmark. The backbone is frozen.

- Backbone: RAD-DINO, blocks 6/9/12, frozen final LayerNorm, concatenated
  (2304-d, 37x37, own-depth CLS), the same features as arm A.
- Training data: new synthetic 3D-CT "supervoxel change" scenes from train-split patients only.
  The changes are arbitrary regions and anatomy-based edits (section 4).
- Episodes are selective: support and query share one task. Queries also contain changes of
  other types that must not be marked (VECD-style).
- Validation: existing synthetic pathology pairs (consolidation / effusion / pneumothorax /
  fluid overload x direction). These are never generated for training, so they test unseen tasks.
- Test: ICU-100, with support sets drawn from the other pairs (patient-group-disjoint).
  - Primary support GT = clinician consensus (>=3/5); secondary = >=2/5 and each reader alone.
  - n in {1,2,4,8,16,32,64}, 10 draws per n, with draw SD and a pair-bootstrap 95% CI.
- ICU is a reused cohort and stays exploratory; no model or setting is selected on it.

## 2. State of the art: how few-shot methods do this

### 2.1 Few-shot / in-context segmentation families

| Family | Representative work | Mechanism | Used here as |
|---|---|---|---|
| Prototypes | PANet (Wang, ICCV'19), PFENet | Masked-average class prototypes; cosine to query; PANet prototype-alignment (PAR) loss | `panet` arm (+PAR), training-free `prototype` |
| Superpixel self-supervised FSS (medical) | SSL-ALPNet (Ouyang, ECCV'20/TMI'22), ADNet (Hansen, MedIA'22) | Episodic training on superpixel/supervoxel pseudo-labels; adaptive local prototypes | Our training data lifts this idea to *changes*; `alpnet` arm |
| Dense correlation / label attention | HSNet (ICCV'21), DCAMA (Shi, ECCV'22) | Query x support pixel attention; DCAMA aggregates the support **mask** as attention values (fg and bg) | `dcama` arm, FSDiff branch M1, training-free `knn` |
| In-context universal medical | UniverSeg (Butoi, ICCV'23), Tyche (CVPR'24), MultiverSeg (ICCV'25), Neuralizer (CVPR'23) | CrossBlock query/support interaction averaged over supports; any n; synthetic tasks and task augmentation gave the largest gain; Dice rises steeply 1->16 then saturates; ensembling over support draws helps most at small n | `universeg` arm; log-uniform n curriculum; 10-draw ensemble |
| Visual in-context generalists | Painter (CVPR'23), SegGPT (ICCV'23); prompt selection (Zhang et al., NeurIPS'23) | Retrieving similar prompts helps | ICU retrieval-support arm |
| Training-free on frozen foundations | Matcher (ICLR'24), PerSAM (ICLR'24), No-Time-To-Train (2025), FSSDINO (2026) | Dense matching / prototypes of frozen features | `training_free_prototype`, `training_free_knn` |
| Set conditioning | Set Transformer (PMA/ISAB), FiLM, SAM 2 memory attention | Permutation-invariant set pooling; task-conditioned decoder; memory bank | FSDiff branches M2 and M4 |

### 2.2 Few-shot *change* detection

No benchmark or SOTA exists for few-shot, mask-supported, **signed longitudinal** change maps
(CXR or otherwise). The closest work:

- **VECD**, "Show Me the Change You Want: Visual Example-Guided Change Detection"
  ([arXiv 2611.33756](https://arxiv.org/html/2611.33756v1), Nov 2026).
  - Frozen CLIP + DINOv2. The example pair gives a change profile that selects matching
    query changes.
  - Trained on selective multi-change scenes, with an empty mask when the change is absent.
  - Tested on held-out attribute families.
  - VECD uses unannotated examples; we have support masks, which is a stronger signal.
- **AnyChange** (Zheng, NeurIPS'24): training-free bitemporal SAM latent matching plus a
  one-click "point query" for a change of interest.
- Remote-sensing few-shot change detection (IEEE 2025/2026): prototype/meta-learning
  episodes for binary change.
- Frozen-feature change heads (Lin et al. 2024): the lineage of arm A.
- Synthetic change data from single images: STAR/ChangeStar (ICCV'21), Changen (ICCV'23),
  CutPaste, DRAEM, AnatPaste.

FSDiff is therefore "UniverSeg/DCAMA for pairs". It is trained like SSL-ALPNet on
pseudo-changes, with VECD-style selective episodes. Each SOTA family is re-implemented as a
comparator on the same pair encoder, decoder, episodes and update budget.

## 3. Model ([models/few_shot_head.py](models/few_shot_head.py))

```
frozen RAD-DINO 6/9/12 tokens (2304-d, 37x37) for every query and support image
pair encoder (shared): 1x1 conv 2304->96, 2x bidirectional cross-attention,
                       z = MLP([cur-pri, cur, pri])
support labels: signed GT -> (unchanged, positive, negative) fractions per 37x37 cell
matching (permutation invariant over supports, any n, n = 0 -> learned null context):
  M1 DCAMA label attention   Q=z_q, K=z_s(+pos), V=support label fractions   (SDPA)
  M2 memory attention (2x)   Q=z_q, K=z_s(+pos), V=z_s+label embedding
  M3 prototypes              global + ALPNet local (unchanged/positive/negative) cosine maps
  M4 task tokens             per-support PMA -> set self-attention -> mean -> FiLM
fuse [z_q, M1, M2, M3] -> 96 ch -> FiLM(M4) -> existing three-class DPT decoder (512x512)
signed map = p(positive) - p(negative); CLS difference feeds the decoder as in arm A
```

- Variants on the identical encoder and decoder:
  - `fsdiff` (M1-M4), 2.98M parameters
  - `dcama` (M1), 2.51M
  - `panet` (global prototypes + PAR), 2.36M
  - `alpnet` (global + local prototypes), 2.36M
  - `universeg` (CrossBlocks), 2.77M
  - `none` (no supports), 2.35M
  - Arm A has 2.31M.
- Loss: three-class CE (arm A recipe) at GT cutoff 0.001. Ignored pixels: opposite-sign
  overlaps between the task change and distractors. Two auxiliary terms:
  - 0.1 x soft CE of an "all changes" token head (VECD L_change);
  - PAR for `panet`.
- Not implemented from the earlier sketch: time-role embeddings (the `[cur-pri, cur, pri]`
  mixing carries the direction instead).

## 4. Training data ([data_generation/](data_generation/))

### 4.1 Rendering and GT

- [supervoxel_render.py](data_generation/supervoxel_render.py) reproduces the existing DRR
  pipeline:
  - crop to the lungs;
  - rotate about the volume centre;
  - crop to the rotated lung box (ext 7);
  - ray-sum of HU+1000 along AP;
  - resize to 512 with antialiasing;
  - save the raw projection, which is min-max normalized at load time.
- Projection and rotation are linear, so each change's HU delta is projected separately. Its GT
  is `sign x P(delta) / max(P(base) x lung footprint)`. This is the linearized
  `calculate_diff_map`, with the same 0.015 and >50-pixel cleanup. The GT therefore matches
  the images exactly, per change.
- The prior is rendered at an independent angle. Nuisance enters only the prior render and the
  shared current/registered base, never the GT:
  - inspiration depth (both domes + lung density);
  - existing `ExternalDevices`;
  - angle.
- Tests check projection linearity and that the GT sign follows the timepoint.
- The rotation acts on voxel offsets, exactly like the existing generator, so training and
  validation DRRs share the same geometry. This is a rigid rotation only for near-isotropic CTs;
  the campus cohort is filtered to spacing <= 1 mm on every axis.

### 4.2 Change catalog ([supervoxel_tasks.py](data_generation/supervoxel_tasks.py))

- Inputs: only the existing segmentations (lungs = 5 lobes, heart, bronchi, lung vessels) and
  HU-derived compartments. No TotalSegmentator (user decision).
- HU-derived compartments: body, bone, subcutaneous fat, mediastinum, sub-diaphragm, chest wall,
  pleural rim, fissures, hila.
- Supervoxels: 3D SLIC, computed once per CT and cached. Regions are random connected supervoxel
  unions (70%) or smooth-noise blobs (30%), with log-uniform size.
- Units: every size, distance, smoothing width and rotation is in millimetres from the CT voxel
  spacing. Real CTs are strongly anisotropic (e.g. CT-RATE 0.35 x 0.35 x 1.5 mm), so voxel units
  would stretch changes along z by a different factor per CT.
- Head/foot direction comes from the heart centroid (inferior to the lung centre), then the
  trachea (superior), then +z. Lung-area profiles are not used: the diaphragm domes make the
  basal slices small.
- Sub-diaphragm and diaphragm changes use only dome columns (columns whose lowest lung voxel is
  in the bottom 40% of that lung). Chest-wall candidates are restricted to the lung projection
  footprint, because the DRR crop (lung box + 7 voxels) excludes the lateral wall.
- Visibility gate: every change must reach peak |GT| >= 0.035 and >= 80 GT pixels in an
  approximate AP projection (`CTContext.min_visibility`), otherwise it is resampled. The
  estimates are stored as `ap_peak` / `ap_area_px` in the change params.
- T1 grow/shrink/displace (and non-lung cut-paste) are seeded within 10 mm of the lung interface.
  Grow and shrink move only into tissue that differs by more than 150 HU, because edits inside
  homogeneous tissue are invisible in projection. Texture patterns are non-negative, because
  zero-mean patterns cancel along the ray.

| Tier | Families (`FAMILIES` keys) | Use |
|---|---|---|
| T1 (35) | `t1_{density,replace,grow,shrink,displace,texture,cutpaste}@{lungs,mediastinum,heart,chest_wall,sub_diaphragm}` | main training |
| T2 (21) | heart_resize, pericardial_shell, mediastinal_widening/mass/shift, tracheal_deviation, bronchial_dilation, mucus_plug, hilar_mass, pa_dilation, diaphragm_shift, free_air, gastric_bubble, subcutaneous_emphysema, chest_wall_thickening, rib_lesion, nodule, cavity, bulla, calcified_dot, lobar_hyperlucency | main training |
| T2near (6) | lobar_collapse, fissural_fluid, pleural_thickening, pneumomediastinum, bronchial_wall_thickening, vessel_change | **ablation corpus only** (separate `--output`, `--tiers T1 T2 T2near`) |
| Validation | consolidation, pleural_effusion, pneumothorax, fluid_overload | never generated here |
| Nuisance | inspiration depth, devices, angle | never a task, never in GT |

Scene generation:

- 1-3 voxel-disjoint changes per scene, each from a different family; 10% nuisance-only scenes.
- Each change is applied to prior or current with probability 0.5.
- Changes below 50 GT pixels are kept as unlabeled sub-threshold changes (`realized=false`).

Simplifications relative to the plan table:

- Heart enlargement uses centroid scaling, not the existing Cardiomegaly B-spline.
- There is no local chamber bulge.

### 4.3 Label styles (decision deferred, user request)

- Every change stores both labels:
  - the tight signed map (`changes.npz`);
  - annotation-tool ellipses (`ellipses.json`), fitted per component and drawn with the same
    `skimage.draw.ellipse` call as `load_labels_map`. Fit and draw round-trip at IoU >= 0.99.
- [episodes.py](datasets/episodes.py) supports three modes, plus on-the-fly bbox, dilate,
  erode and drop-one-component variants:
  - `tight`;
  - `coarse_support` (p=0.3 episodes have coarse supports, tight query targets);
  - `coarse_both`.
- The choice is made by a synthetic-validation comparison declared in advance, never on ICU.

## 5. Episodes

- A task is one of four levels:
  - fine: family + GT sign;
  - family: family, either sign;
  - compartment: compartment + GT sign;
  - any: every realized change, the ICU-like mission. Only this level allows n = 0.
- Sampling weights for the four levels: 0.45 / 0.25 / 0.15 / 0.15.
- Queries:
  - 15% contain no task change (empty target);
  - 50% are drawn from scenes that also contain distractors.
- Supports always contain the task and never share a patient with the queries.
- n is log-uniform in [1, n_max].
- Episodes are deterministic per (seed, epoch, index), so resume is exact (tested).

## 6. Validation and test protocols ([evaluation/](evaluation/))

### Synthetic manifests ([few_shot_manifests.py](evaluation/few_shot_manifests.py))

- Each task's query patients are split into groups. Each (task, n, draw, group) has one support
  set that shares no patient with that group.
- Every query is scored exactly once per (n, draw).
- Manifests are hashed, so every model sees identical draws.
- Selection manifest: n in {1,4,16}, 3 draws, <=64 queries/task, 16 nuisance pairs/task.
  - Score: task-macro pooled signed Dice (GT 0.001, prediction 0.5), averaged over n.
- Final manifest: all queries, n in {1..64}, 10 draws.
- Also reported:
  - GT cutoff 0.05;
  - nuisance FP area (mean / p95 / max);
  - a support-shuffle check (supports from another task) that detects heads ignoring the supports.

### ICU ([evaluate_few_shot.py](evaluation/evaluate_few_shot.py))

- Supports are drawn from the other pairs, in 10 groups; pool = non-empty consensus by default.
- Postprocessing and matching are unchanged: |score| > 0.75, sign-specific CC > 250
  (512²-equivalent), and the presentation one-to-many / one-to-one and legacy one-to-one terms
  from `rescore_subset_clinical_many`.
- Reported for each n:
  - mean over the five readers per draw, then draw mean and SD;
  - 95% pair-bootstrap CI (10,000 resamples, same resamples for all models);
  - F1 versus the consensus;
  - the 5x5 support-reader x reference matrix (`per_reference`);
  - paired difference versus the n=0 reference (arm A if `--arm_a_checkpoint` is given).
- Extra arms:
  - 10-draw ensemble;
  - retrieval supports (nearest frozen CLS-difference);
  - synthetic pathology supports;
  - arm-A few-shot fine-tuning (`--finetune_steps`);
  - training-free prototype and kNN;
  - FSDiff with n = 0.
- A threshold sweep is not automated; rerun with `--threshold` to get one.

## 7. Comparators and ablations

| Ablation | How to run |
|---|---|
| Non-selective (every change labeled) | `level_weights: [0, 0, 0, 1]` |
| Label style | `label_mode: tight / coarse_support / coarse_both` |
| n_max | `max_support` (benchmark-limited) |
| Block 9 only | `feature_layers: [9]` |
| T2-near | ablation corpus and `corpus_tiers: [T1, T2, T2near]` |
| T1 only | `corpus_tiers: [T1]` on a T1-only corpus |

Not yet implemented:

- 2D-DRR superpixel generator;
- "validation families seen in training" upper bound;
- initializing the encoder from arm A;
- the held-out shape/location audit of training families against the validation pathologies.

## 8. Compute

- Images are stored as float16 memmaps:
  - scenes: `[S,2,512,512]` and per-change maps;
  - validation pairs.
- Frozen features are extracted online, once per episode for supports and queries together.
  The benchmark reports bf16-vs-fp32 token cosine parity.
- `--benchmark` tries n_max 64/32/16 for all six arms and picks the largest under 21 GB peak.
  Training refuses `max_support` above it.
- Validation tokens are cached (default 1,200 pairs, about 15 GB RAM in fp16).

## 9. Campus run order

1. `bash training/run_few_shot.sh audit` (set `CLINICAL=...`). Review:
   - segmentation coverage;
   - per-family success and GT areas on sample CTs;
   - linearity error;
   - validation task pool sizes;
   - ICU non-empty consensus/reader counts.
   Tune magnitude ranges if families rarely pass 50 px.
2. Visual preview before mass generation:
   - On campus, `python -m data_generation.export_preview_cases --source_manifest
     $DATA/manifest_split.jsonl --output /cs/labs/josko/sahar_aharon/fewshot_preview_cases --count 2`
     bundles 1-2 train-split CTs. Cases come from different sources, with all segmentations and
     one existing pathology pair per type for comparison.
   - Copy the bundle to the local machine.
   - Run `python -m data_generation.preview_supervoxel_families --cases_root <bundle> --output
     review_outputs/fewshot_preview_<date>`, about 5 minutes per CT on CPU.
   - Output: one panel per family (CT before/after, DRR prior/current, image difference, tight GT,
     ellipse label), per-tier contact sheets, composed scenes with episode target vs distractors,
     the nuisance example, and `index.md` with families that were not realized.
   - Review it, adjust magnitudes, and only then generate at scale.
   - Without campus access, `python -m data_generation.fetch_ctrate_preview_cases --output <folder
     outside the repo> --totalseg <TotalSegmentator.exe> --count 2` downloads public CT-RATE CTs
     (Hugging Face login required; no redistribution). It keeps only CTs with every spacing
     <= 1 mm, the same rule as the campus cohort filter. The renderer rotates in voxel units like
     the existing DRR generator, so CTs with thicker slices are sheared by the rotation and are
     not representative.
3. `generate <slice> <num_slices>` on several GPU nodes (resumable), then `manifest`, then `stores`.
4. `benchmark`; set `max_support` to `chosen_max_support`.
5. `train` (locked, resumable), then `eval_synthetic`.
6. Freeze the recipe, then `eval_icu` (set `ARM_A=<arm A best checkpoint>`). Publish aggregates only.

## 10. Tests and limitations

Local CPU tests:

- [models/test_few_shot_head.py](models/test_few_shot_head.py): shapes, permutation invariance,
  n = 0, gradients.
- [data_generation/test_supervoxel_tasks.py](data_generation/test_supervoxel_tasks.py):
  - all families on a phantom CT;
  - bounded HU;
  - linearity;
  - timepoint signs;
  - disjoint scenes;
  - nuisance excluded from GT;
  - ellipse round trip;
  - corpus files;
  - NIfTI load/crop;
  - audit.
- [datasets/test_episodes.py](datasets/test_episodes.py):
  - selective targets and ignore;
  - patient-disjoint supports;
  - coarse labels;
  - store parity;
  - determinism.
- [training/test_train_few_shot.py](training/test_train_few_shot.py), using a fake frozen backbone:
  - training, validation, checkpoints;
  - exact resume;
  - benchmark;
  - manifests;
  - synthetic CLI;
  - ICU runner with ensemble/retrieval/bootstrap;
  - presentation-metric parity.

Not verified locally:

- the real RAD-DINO;
- real CT volumes (speed, memory, family realism, magnitudes);
- GPU memory;
- ICU loading (Benny's annotations are not in the local checkout).

Phantom success says nothing about clinical realism. Review generated scenes visually before
training.
