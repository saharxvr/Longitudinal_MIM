# Joint block-6/9/12 projection, temporal cross-attention, and CE

Status: **implemented for joint A/B screening; not launched**. The user requested
training A and B in one shared-backbone job. Configuration:
[layer_fusion_6_9_12_ce_30.yaml](configs/layer_fusion_6_9_12_ce_30.yaml).
Campus preflight and the real GPU benchmark remain required before training.
Local validation:91passing CPU tests, exact old DPT initialization/output parity
for eight legacy architecture/output combinations, and valid Bash launch blocks.

## 1. Question and scope

Can one jointly trained spatial head use normalized RAD-DINO blocks6/9/12 to
localize signed change better than matched single-layer heads?

The selected32case map review motivates this hypothesis: all three depths have
some exclusive GT-supported predictions, but also unsupported/faint-target
regions. Arithmetic score averaging is not a clear solution: it can suppress
useful isolated detections as well as nuisance predictions. Those enriched
examples do not establish population performance or pathology specialization.

Here, "projection head" means the trainable **spatial feature projection,
temporal cross-attention, global CLS branch and segmentation decoder**.
It does not mean the contrastive pathology/nuisance MLPs in
[models/projection_heads.py](models/projection_heads.py). Pixel CE alone does not
train an output-only latent projection that is not on the segmentation path.
No contrastive, orthogonality, triplet, or pathology-classification objective is
added. Outputs remain unchanged / positive change / negative change, not
pathology categories.

## 2. Architecture A: early channel fusion

Keep RAD-DINO frozen, in eval mode, under no-grad:

```text
Prior and current images: same existing upright preprocessing
          |
Shared frozen RAD-DINO: one pass per image
          |
Blocks 6, 9, 12, each with its own patch and CLS tokens
Frozen final LayerNorm per depth; no double normalization of block12
          |
Per image: concatenate patch features along CHANNELS
[B, 1369, 768] x 3 -> [B, 1369, 2304]
          |
Trainable shared prior/current 1x1 projection: 2304 -> 96
          |
Existing bidirectional temporal cross-attention
2 blocks, 4 attention heads, 1369 tokens, channel width96
          |
Current-context minus prior-context
          |
Existing four-scale DPT reassembly/refinement from the one fused feature map
          |
Three raw logits per pixel at512x512
          |
Pixel CE; inference score = p_positive - p_negative
```

The CLS branch concatenates the three depth-specific normalized CLS tokens for
each image. Project their current-minus-prior difference from2304 to96 and inject
it through the existing decoder's global-change path. There is no hidden block12
CLS shortcut for earlier-layer controls.

This is learned feature fusion BEFORE temporal attention, not averaging maps,
not concatenating layers along the token axis, and not cross-layer attention.
Keeping1369tokens avoids the quadratic cost of a4107token attention sequence.

The existing [DPT head](models/dpt_head.py) still accepts one or four input layers.
The separate DPTLayerFusionHead has an explicit three-source input contract,
retains backbone_dim768 and ordered source metadata, and shares the projected-map
decoder with the old heads without bypassing their input guard. Architecture B
uses separate per-depth projection/attention streams before learned288->96fusion,
as illustrated in section8. Both use the same concatenated own-depth CLS branch.

## 3. Loss: CE-only for this experiment

Interpret the requested CE objective literally:

- Classes:0unchanged,1positive,2negative.
- Targets: rawGT>0.001 is positive; rawGT<-0.001 is negative; otherwise unchanged.
- Equal CE class weights[1,1,1], CE coefficient1, Dice coefficient0.
- Regression/sign/contrastive coefficients0.
- Keep the original balanced pair sampler. Do not change class weighting or GT
  cutoff during this experiment to compensate for disappointing results.

The current `ce_dice` loss path already permits `w_ce=1, w_dice=0`; no new public
loss mode is required just to express CE-only. Dice may still be logged as a
diagnostic without contributing gradients.

Earlier block6/9/12 results used CE+Dice. They are contextual evidence, NOT matched
controls for this CE-only run. If CE+Dice is desired instead, preregister it for
every arm before launch; do not mix loss recipes in the primary comparison.

## 4. Matched arms and initialization

Joint screening: seven branches in one shared-backbone job:

| Arm | Feature sources into its own trainable head | Purpose |
|---|---|---|
| S6 | Block6 | Fresh CE-only single-layer control |
| S9 | Block9 | Primary single-layer reference chosen before this run |
| S12 | Block12 | Final-layer reference; important given the GT0.05 ranking |
| A | Blocks6,9,12; concatenate then project/attend | Early-fusion model |
| R9A | Block9 repeated three times; architecture A | Shape/parameter-matched early-fusion control |
| B | Blocks6,9,12; project/attend separately then fuse | Depth-preserving model |
| R9B | Block9 repeated through three streams; architecture B | Shape/parameter-matched late-fusion control |

Extract unique depths6/9/12 once per image and batch, for all seven arms.
R9A/R9B repeat block9's already normalized features downstream without another
encoder pass or normalization, including block9's own CLS token.

Primary comparisons: A versus S9 and B versus S9; also compare A versus B.
Report all single-layer comparisons, A versus R9A and B versus R9B; do not hide
a stronger S6/S12 outcome. Each repeat control helps distinguish
additional feature sources from merely expanding the input projection. It matches
architecture/parameter count, not input rank or optimization geometry, so it is
not an information-theoretic proof.

CE-path trainable parameters at width96: S6/S9/S12=2,010,723 each;
A/R9A=2,305,635 each; B/R9B=2,780,931 each. A's spatial/CLS matrices add
294,912 weights versus a single-layer head. The unused z_proj latent branch is
frozen and excluded from optimizers/counts for every arm in this experiment.
Benchmark output records actual counts/specifications for the selected arms.

Initialize all heads fresh; do not warm-start from CE+Dice checkpoints. Copy
identical named decoder/attention initialization where shapes match. Copy complete
initial states within S6/S9/S12, A/R9A and B/R9B. Do not claim
identical complete initialization across incompatible projection shapes.

Use the same sampled pairs and the same realized photometric perturbations for
all branches. Optimizers, schedulers, gradients and best checkpoints are
independent. Backpropagate branches sequentially to bound activation memory.

## 5. Fixed training recipe and budget

- Existing patient/source-disjoint manifests and clean validation; no new data.
- Upright rotation_k3 plus horizontal flip, unchanged GT preprocessing.
- Existing independent prior/current photometric augmentation, train only.
- Frozen backbone and frozen final LayerNorm; only segmentation-path head
  parameters train.
- AdamW, learning rate0.0003, weight decay0.01, gradient clipping1.
- Seed42 screening, batch16/accumulation2 as the benchmark starting point.
- Fixed30epochs, two-epoch linear warmup, cosine decay declared for30epochs.
- Complete the same budget for all arms unless an explicit numerical or
  operational failure occurs. Save best and final checkpoints separately.

Preflight/benchmark seven arms with10warmup and100measured shared batches,
including complete optimizer steps, validation and diagnostics timing. Report
measured memory/runtime; do not scale the previous14hour run by head count and
call that a prediction. If memory forces a microbatch change, apply it to every
arm before starting in a new isolated run directory and record the deviation.

After screening, confirm the promising comparison with seeds43and44 using the
same declared recipe. Do not interpret one seed or correlated epoch fluctuations
as a robust fusion advantage.

## 6. Checkpoint selection, evaluation and decision

Select each checkpoint using the same validation target-matched pooled signed
Dice: prediction threshold0.5, rawGT cutoff0.001, pathology pairs only.

Report, without switching the selection metric post hoc:

- Target-matched pooled and macro signed Dice; positive/negative metrics.
- RawGT0.05 results separately, including per-pathology and per-sign results.
- Same-sign, reference-first, one-to-one8-connected component detection F1.
- Nuisance predicted FP area mean/p95/max; empty masks do not become perfect Dice.
- Loss/metric curves, actual epochs, parameter counts, wall time and memory.
- Full-validation spatial disagreement and GT-supported exclusive coverage.
  Distinguish permissive component coverage from one-to-one detection.

Predeclared secondary operating points: prediction0.5/no CC filtering,
prediction0.75/CC>50, and prediction0.75/CC>250. Apply filters separately by sign
in512x512-equivalent pixels. A wider sweep may be descriptive but must not silently
replace these endpoints or select depth on the32case review gallery.

Fusion is promising if its target-matched gain survives matched controls/seeds
and is not merely a pooled-area gain accompanied by worse macro/component
performance or nuisance errors. Report mixed outcomes explicitly. A GT0.001-only
gain is not automatically better localization of high-magnitude change atGT0.05.

Use the full fixed synthetic validation split for selection/diagnosis. Only then
evaluate selected checkpoints on clinical100 under the unchanged protocol;
those reused clinical cases remain exploratory. Do not use synthetic pathology
labels or the enriched map gallery to claim clinical specialization.

## 7. Implementation and validation gates before launch

1. Add explicit ordered feature-source/channel-fusion metadata and a trainable
   projection/CLS stem; wire trainer, head builder, resume and all evaluators.
2. Verify exact block indexing, frozen normalization parity and depth-specific
   CLS sourcing; all three feature sources must influence the fused logits.
3. Verify CE gradients reach spatial/CLS projection, temporal attention and
   decoder; backbone and unused output-only latent branches must not be claimed
   as trained. Verify no gradients/optimizer state leak between arms.
4. Test zero-Dice CE behavior, target classes, shapes and signed-map inference;
   preserve existing single/four-layer regressions.
5. Test branch-order-independent sampling/augmentation and save/resume continuity,
   with one authoritative shared completed-epoch boundary and independent states.
6. Reject mismatched feature order, normalization, fusion, loss, config, manifest,
   backbone revision and code provenance; never fall back to final-layer inference.
7. Run miniature end-to-end export/evaluation and campus benchmark before training.
   Publish only explicitly scoped metrics/provenance/maps, never raw scans,
   weights or caches by broad Git staging.

See [the complete campus launch and monitoring blocks](configs/README.md#joint-ab-layer-fusion-ce-only)
after the implementation has been committed and pushed to main. The launcher
preflights, benchmarks, then starts/resumes independent heads from one shared
completed-epoch checkpoint. No campus connection or real training was performed
from the development workstation. Training diagnostics include unfiltered
0.5/0.75 sweeps; the sign-specific CC>50/250 secondary evaluation remains a
separate evaluation step, not checkpoint selection.

## 8. Literature support and fusion-placement alternative

Primary papers and official code checked for this review. These sources support
individual design principles, not the exact combination of frozen RAD-DINO,
blocks 6/9/12, width 96, signed three-class CE and longitudinal chest X-rays.

| Source | Actual multi-layer / temporal mechanism | Relation to this plan |
|---|---|---|
| [DPT: Vision Transformers for Dense Prediction, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html) | Reassembles features from several ViT depths into spatial maps at different resolutions and progressively fuses them. Official plain ViT-B hooks are zero-based 2/5/8/11, i.e. blocks 3/6/9/12. | Strong precedent for intermediate-depth features and DPT decoding, but NOT evidence for collapsing all depths before temporal attention. |
| [SegFormer, 2021](https://arxiv.org/abs/2105.15203) | Projects each hierarchical feature level with an MLP, resizes to a common grid, concatenates along channels, and applies learned 1x1 fusion before pixel classification. | Direct precedent for project/concatenate/learned-fusion operations. Its encoder stages have different spatial resolutions; ours do not. |
| [DINOv2, 2023](https://arxiv.org/abs/2304.07193) | Official dense heads support multi-level feature inputs. Its DPT depth head selects blocks 3/6/9/12 for ViT-B; its segmentation BNHead supports resize-concatenation. | Relevant foundation-feature precedent. Does not establish our layer subset, width, intermediate final-LayerNorm policy or temporal fusion as optimal. |
| [ChangeFormer, 2022](https://arxiv.org/abs/2201.01293) | A Siamese hierarchical transformer provides four feature levels. At each level, a learned convolutional difference module processes concatenated before/after features, followed by multi-level decoding. | Supports comparing times at each level before fusing levels. Its temporal difference module is not our proposed bidirectional full-grid cross-attention. |
| [BIT: Remote Sensing Image Change Detection with Transformers](https://arxiv.org/abs/2103.00208) | Contextualizes compact semantic tokens from both dates, cross-attends spatial features to token memory, then differences decoded features for binary change detection. | Supports attention-based temporal reasoning, not an exact three-depth full-grid cross-attention architecture. Its absolute differencing/binary output must not be copied blindly into signed change detection. |

Official implementation details:

- [DPT block hooks, readout and reassembly](https://github.com/isl-org/DPT/blob/main/dpt/vit.py).
  Projected readout incorporates the CLS token of the corresponding depth into
  patch features; our separate global CLS-difference branch is a different choice.
- [SegFormer projection and channel fusion](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py).
  Its [official ADE20K configuration](https://github.com/NVlabs/SegFormer/blob/master/local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py)
  explicitly uses softmax cross entropy. This supports CE as a standard dense
  classification objective, not a claim that CE-only is best for our imbalance.
- [DINOv2 depth-head layer selection](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/depthers.py)
  and [segmentation resize-concatenation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/models/decode_heads/linear_head.py).
- [ChangeFormer code](https://github.com/wgcban/ChangeFormer/blob/main/models/ChangeFormer.py).
- [BIT model flow](https://github.com/justchenhao/BIT_CD/blob/master/models/networks.py)
  and [cross-attention decoder](https://github.com/justchenhao/BIT_CD/blob/master/models/help_funcs.py).

### Important distinction: depth is not spatial scale

RAD-DINO blocks 6/9/12 all have the same 37x37 patch grid. They are multi-depth
representations, not three naturally different image resolutions. DPT creates
decoder scales by reassembly; SegFormer/ChangeFormer start from an actual
hierarchical spatial pyramid. Do not equate the architectures or assert that
blocks 6/9/12 intrinsically correspond to specific pathology types.

### Candidate B: depth-preserving temporal processing

Architecture B implements this depth-preserving alternative to the
early-concatenation architecture in section 2:

```text
Block6: shared prior/current projection 768->96 -> temporal cross-attention -> D6
Block9: shared prior/current projection 768->96 -> temporal cross-attention -> D9
Block12: shared prior/current projection 768->96 -> temporal cross-attention -> D12
                 |
Concatenate the three temporal-difference maps: 96+96+96
                 |
Learned 1x1 fusion: 288->96 -> common DPT decoder -> three-class CE
```

This is motivated by per-level processing in DPT/ChangeFormer and learned
concatenation fusion in SegFormer. The per-level temporal cross-attention itself
remains our adaptation, not a reproduction of those papers. It preserves depth
identity until after temporal comparison but requires three attention streams;
whether that helps is an empirical hypothesis.

User decision after literature review: train both A and B in the same job,
with separate R9A/R9B controls. R9B repeats block9 through B's full three-stream
projection/attention/fusion architecture; R9A is not its matched parameter-count
control. Shared-backbone extraction, signed targets, CE recipe, data, schedule,
evaluator and clinical caveats stay fixed. Resource feasibility is established
by the campus benchmark, not inferred from the old three-head run.
