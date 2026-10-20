# Single-layer depth ablation with a shared frozen backbone

Status: proposed implementation; no training launched and no active run changed.

## Question and scope

Which RAD-DINO depth provides the most useful features for longitudinal change
localization when each head receives exactly one layer?

Train three independent, identical DPT cross-attention heads on blocks 6, 9 and
12 in one job. This is a depth ablation, not multi-layer fusion. Each head has
its own bidirectional prior/current attention and decoder; there is no attention
or feature sharing between heads. RAD-DINO remains frozen and in eval mode.
All selected layers have a 37x37 token grid with 768 channels.

## Shared extraction and fair comparison

- Preprocess and augment each prior/current pair once. All heads receive features
  from the same realized images and the same balanced sampler sequence.
- Extract the selected hidden states during one backbone pass per image under
  no_grad. Block numbers are one-based transformer outputs, not the embedding
  state at hidden_states[0].
- Apply the frozen backbone's final LayerNorm separately to each selected hidden
  state. Verify that normalized block 12 reproduces the existing
  last_hidden_state patch and CLS outputs within numerical tolerance. Do not
  normalize the already normalized final output again.
- Use the selected layer's CLS token as well as its patch tokens, so the global
  branch cannot leak final-layer information into earlier-layer experiments.
- Initialize one head and copy its initial state to the other heads. Keep head
  architecture, parameter count and hyperparameters identical.
- Retain only selected detached outputs after extraction. Forward/backward heads
  sequentially to avoid retaining three head activation graphs simultaneously.
  Keep independent optimizers, gradient accumulation, clipping and schedulers.
- Keep sampling/augmentation RNG independent of head construction/execution.
  If stochastic head operations are present, isolate and checkpoint per-head RNG
  state rather than allowing branch order to change augmentation randomness.

## Shorter training budget

Recommended first stage: **30 epochs for all three heads**, with the same
two-epoch linear warmup and cosine decay configured for 30 epochs from the start.
This halves the number of dataset passes versus the current 60-epoch recipe,
but does not imply half its wall time because three heads must be trained.

This is a fixed-budget screening experiment: it compares depth under a shorter
schedule, not definitive convergence. Do not eliminate a branch early based on
noisy initial rankings. Inspect curves at epochs 10, 20 and 30, while preserving
the predeclared budget unless there is a numerical or operational failure.

If leading scores are close or still improving at the end, treat the ranking as
inconclusive. A longer confirmation run should have its schedule declared in
advance; do not present an extension after completed cosine decay as equivalent
to a fresh longer-schedule run. Additional seeds on promising depths are needed
before claiming a robust depth advantage.

## Fixed settings

Use the current online augmentation recipe as the reference:

- Existing patient-disjoint manifests; no new patients or datasets.
- Upright rotation/flip preprocessing; existing train-only photometric
  augmentation; clean validation and clinical inference.
- CE + classwise Dice, coefficients 1/1; equal three-class CE weights.
- Signed targets at raw-GT cutoff 0.001; signed score p_positive - p_negative.
- AdamW, learning rate 0.0003, weight decay 0.01, gradient clip 1, seed 42.
- Start benchmarking batch 16 / accumulation 2, effective batch 32.
  If memory requires a smaller microbatch, apply it to all branches and record
  that batchwise Dice makes the recipe different from the current baseline.
- Same diagnostics and evaluation thresholds; select best checkpoints by
  validation target-matched pooled signed Dice, separately for each branch.

The fresh block-12 branch is the internal control. The existing 60-epoch
augmentation run is contextual evidence, not the matched control.

## Benchmark and validation gates

Before full training:

1. Test exact block indexing, token/CLS shapes, and final-output normalization
   parity using the actual RAD-DINO model.
2. Test that backbone parameters stay frozen and each loss updates only its own
   head. Verify copied initial weights and independent optimizer state.
3. Test that shared features and clean validation outputs match independent
   extraction; changing branch order must not change sampled/augmented inputs.
4. Test accumulation boundaries, incomplete-batch policy and per-head optimizer
   step counts against the existing trainer behavior.
5. Save and resume all heads, optimizers, schedulers, RNG/sampler state,
   normalization/layer specification, epoch and best metrics. Test resume
   continuity against an uninterrupted miniature run, and reject incompatible
   configs explicitly.
6. Benchmark at least 100 batches after warmup, including complete optimizer
   updates for every head. Measure seconds per shared batch, peak allocated and
   reserved GPU memory, and separately time validation/diagnostics. Record batch
   count and estimate total wall time from measurements, not head count alone.
7. Run a miniature end-to-end pipeline through per-layer checkpoint loading,
   prediction export and metrics. Evaluation must select the trained block and
   normalization policy, rather than silently falling back to the final layer.

## Outputs and evaluation

Use a new isolated run directory. Persist one shared configuration and provenance
record, per-layer metrics CSVs/diagnostics, and per-layer best/last checkpoints.
A resumable shared snapshot must identify one consistent training boundary
across all three branches.

Report a summary table for blocks 6/9/12:

- Best target-matched pooled signed Dice and selected epoch.
- Macro signed Dice, positive/negative metrics and component F1.
- Nuisance false-positive area mean, p95 and maximum.
- Training/validation loss curves, runtime and memory.

Choose promising depths using synthetic validation, not repeated clinical
selection. Subsequently evaluate selected best checkpoints using the unchanged
clinician comparison protocol. These clinical cases have already been reused;
describe their results as exploratory rather than an untouched clinical test.

## Implementation order

1. Add explicit selected-layer extraction without changing existing defaults.
2. Add shared-backbone multi-head training and complete provenance/resume support.
3. Wire per-layer validation, inference and checkpoint-selection metadata.
4. Add tests and benchmark/config-driven launch support.
5. Run benchmark on campus, then launch the fixed 30-epoch experiment.

Do not modify or stop the current augmentation run as part of this plan.
