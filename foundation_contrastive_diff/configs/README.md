# Run configs

YAML configs for `training.train_foundation_diff`. A config's values become the arg
defaults; anything you pass on the CLI still overrides it. Each run also writes its fully
resolved parameters to `<save_folder>/run_config.json` (a record of exactly what ran).

## Exact Input And GT Export

This is a fresh, model-free check of the CURRENT corrected cache configuration,
not a collage of the morning epoch49 model's predictions. It exports three
realized synthetic TRAIN pairs from different cases/anomaly types, and three
clinical TEST pairs (default1,2,3). The clinical cohort is not the synthetic
manifest's held-out test split. No neural weights, head, predictions or embeddings
are loaded/generated, and the existing cache and active training are untouched.

Synthetic images use the SAME LongitudinalPairDataset/DataLoader as cache creation.
Clinical images and annotations use the actual clinical inference/evaluation
loaders. The actual cache backbone factory and forward preprocessing run with
ONLY AutoModel.from_pretrained replaced by a capture-only, parameter-free sink:
the capture stops at pixel_values before any neural inference. The original
processor mean/std loading and fallback, repeat-to-RGB, resize and normalization
therefore execute unchanged. Their effective values are recorded in report.json.

Each folder includes source/model-coordinate review.png, separate prior/current
cache-input PNGs, final encoder RGB previews, exact_tensors.npz and source hashes.
Encoder PNGs are de-normalized for visibility; NPZ preserves actual normalized
pixel_values without scaling. Actual cached continuous GT is read by
CachedPairDataset and required to equal fresh GT converted through cache float16.
The actual signed training target at0.0001 is verified with change_map_loss;
checkpoint GT at0.05 is shown separately. Clinical human masks/3of5consensus
are shown in their NATIVE measurement coordinates; the model-space consensus
panel is for orientation inspection ONLY, not a replacement scoring reference.

Run on campus while training continues:

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
python -u -m evaluation.export_cache_input_evidence \
	--config configs/last1_upright_mask_thr0p0001_fp075.yaml \
	--clinical-pairs 1 2 3 \
	--publish
```

The script checks cache/manifest/config agreement and, when present, the active
run_config.json. It writes a fresh results/cache_input_evidence_TIMESTAMP folder
and uses the existing exact-path Git publisher to pullff-only, stage/commit ONLY
that folder, and pushmain. This explicitly uploads the six requested input/GT
examples, including clinical image views and exact tensors, but no source NIfTIs,
annotation JSONs, feature caches or checkpoints. The previous large orientation
audit is NOT repeated. On completion, pull locally and inspect index.md. Existing
bound preprocessing/training files are unchanged, so this export does not replace
or invalidate the approved training preflight.

## Epoch Reporting

Each epoch prints train/validation total loss and pixel-weighted L1/L2, soft-Dice
loss and sign penalty, using the SAME active coefficients (including Dice warmup),
target transform and foreground/background weights. Coefficients are printed in
the epoch header; the component numbers are before multiplying by those
coefficients. Losses are means of batch losses, not sample-weighted epoch means.
Total/component losses are also saved in diagnostic JSON metadata and validation
checkpoint statistics. The existing metrics.csv column schema stays unchanged.

When diagnostics are enabled for an epoch, both splits print pathology-only macro
support Dice/IoU and signed Dice over pairs with active reference support, pooled
signed pixel precision/recall, and nuisance predicted-area mean/p95/max. Undefined
ratios print `n/a`; empty-reference pairs never earn a perfect diagnostic Dice.
Connected-component signed detection F1 is printed when components are enabled;
this is not pixel F1. Train values are ONLINE SAMPLED observations from updating
weights (with sampler repeats), NOT a fixed-head evaluation of the whole train set.
Validation uses the fixed end-of-epoch head. Disabled/skipped diagnostics are
explicitly marked as not collected, never printed as zero.

Prediction and raw-GT cutoffs are printed on every metric line. Signed-mask runs
always collect and print the target cutoff as well as the primary cutoff. For the
upright run these are prediction0.5, targetGT0.0001 and primaryGT0.05. The separately
labeled validation checkpoint metric retains the ORIGINAL change-only support
Dice definition and raw-GT0.05 threshold, including its existing handling of empty
pathology references. Reporting does not change losses, gradients, sampling,
checkpoint selection or the frozen encoder. Because the approved audit hashes the
trainer code, pull reporting changes BEFORE phase1; any old approval must be
regenerated or paired with an explicit hash-bound code review after a trainer edit.
An already running Python trainer will
not acquire new logging by pulling Git.

## Reuse Reviewed Audit

The user explicitly authorized skipping the repeated phase1 audit on2026-10-18.
The actual18572-pair alignment report and ALL37 DRR/GT overlays were already
reviewed. The campus constants difference was verified comment-only. A separate
results/orientation_code_review_20261018/reviewed_code_update.json binds this
decision to the ORIGINAL approval hash and the exact old/new hashes of ONLY the
trainer reporting change and launcher approval handling. The original report and
provenance are not rewritten. Configuration, manifest, report, overlays and every
other code hash must still match; later source changes require a new review.
Clinical source/annotation checks, checkpoint checks, prediction verification and
scoped publication remain enabled. The review record is included in the final
clinical verification.json. No global skip-checks flag is provided.

Run on the campus GPU node to create the corrected cache, train/resume only the
dedicated upright run, then automatically predict clinical100, evaluate F1 and
push the derived outputs/reports:

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_upright_mask_thr0p0001_fp075
export PREFLIGHT="$PWD/results/orientation_preflight_20261018_104010"
export REVIEW="$PWD/results/orientation_code_review_20261018/reviewed_code_update.json"
mkdir -p "$RUN"
touch "$RUN/pipeline.log"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info())"
nohup python -u -m training.run_upright_clinical \
	--preflight-dir "$PREFLIGHT" \
	--approved-preflight-sha256 d43232afe9413a1c7957e83fb2be05f3a640273fd654511b0653da4709fa41e4 \
	--reviewed-code-update "$REVIEW" \
	--reviewed-code-update-sha256 bb5b6cca6edfd6ca1146c4ccf389dc0477ec8833849238b80c789fdb509a7fa5 \
	>> "$RUN/pipeline.log" 2>&1 &
tail -f "$RUN/pipeline.log"
```

Ctrl+C stops tail, not the job. If a retained evidence/code/data guard fails, inspect
the exception rather than editing the audit or review hashes to force approval.

## Remove Obsolete Feature Caches

The user authorized removing old last-four and last-one latent caches with the
obsolete sideways/mirrored input convention. Delete only the five exact directories
below, never a wildcard such as `fcd_cache_last*`. Preserve the new corrected cache
`fcd_cache_last1_upright_inference`, source DRRs/GT in `fcd_train`, all checkpoints,
reports and predictions. Historical head evaluation/training that depends on a
deleted cache will require regenerating those features. Check that NO job on ANY
campus node is reading/writing these shared directories before running cleanup;
the local process check below cannot inspect other nodes. The CPU orientation-only
audit does not need the old caches and can be completed before cleanup.

```bash
bash <<'BASH'
set -euo pipefail
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
if pgrep -u "$(id -u)" -af 'training[./](train_foundation_diff|cache_features|run_last4_dpt|run_rotated_mask)'; then
	echo "Stop or finish the cache/training jobs above before cleanup."
	exit 1
fi
BASE=/cs/labs/josko/sahar_aharon
for CACHE in \
	"$BASE/fcd_cache_last4" \
	"$BASE/fcd_cache_last1" \
	"$BASE/fcd_cache_last1_from_last4" \
	"$BASE/fcd_cache_last1_inference" \
	"$BASE/fcd_cache_last1_rot90cw_inference"
do
	if [[ -d "$CACHE" ]]; then
		du -sh "$CACHE"
		rm -rf -- "$CACHE"
	fi
done
df -h "$BASE"
BASH
```

## Corrected DRR/GT Orientation

`last1_upright_mask_thr0p0001_fp075.yaml` is an orientation-only follow-up to the
latest cutoff0.0001 experiment. Clockwise90 plus a horizontal flip equals the
normalized/resized raw-array transpose, matching the conventional chest display.
Prior, current and continuous signed GT receive the SAME transform. The frozen
RAD-DINO features must be regenerated in `fcd_cache_last1_upright_inference`;
flipping old tokens or reusing the old cache is not valid. The head starts fresh in
`run_last1_upright_mask_thr0p0001_fp075`; resume is only within that corrected run.
Loss, target cutoff0.0001, foreground11/background0.75, head, seed42,60epochs,
checkpoint selection and clinical prediction threshold0.5 are unchanged.

Phase1 checks every train/val prior/current/GT for finite2D arrays, identical source
shape/affine, unchanged signs/locations through the forward/inverse transform, and
exact equality to the intended transpose. It exports stratified GT-on-DRR overlays
with the ACTUAL0.0001 training target mask. No held-out synthetic test is evaluated.
It also checks all100 clinical pairs, their input transpose/inverse placement, source
hashes and the same500 reference annotations. Automated checks cannot prove that a
synthetic lesion is biologically correct, so the actual DRR/GT overlays MUST be
visually reviewed before training. The phase1 artifacts are automatically committed
and pushed through Git for that review, without feature extraction or training.

Complete campus phase1 command:

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_upright_mask_thr0p0001_fp075
mkdir -p "$RUN"
touch "$RUN/orientation_preflight.log"
nohup python -u -m training.run_upright_clinical --preflight-only >> "$RUN/orientation_preflight.log" 2>&1 &
tail -f "$RUN/orientation_preflight.log"
```

Wait for `[publish] pushed` and `Approval SHA256`. Ctrl+C stops tail, not the job.
Failed audits also push their reports for investigation, but never issue approval
or start training. Check the log for an exception, not only a successful Git push.
Pull the published `results/orientation_preflight_<timestamp>` locally and inspect
its `index.md`, report, and all GT overlays. Only after approval, use that exact
directory and printed provenance hash in phase2 below. Do not approve a preview
only because it looks upright: check signed GT overlays against the DRRs too.
Changed configuration, source manifest, audited preprocessing code, report or
preview images invalidate approval. Clinical source scans/annotations are rehashed
after training and must match phase1. Grid checks do not establish anatomical
laterality for unmarked images or exclude every source-generation labeling error.

Complete campus phase2 template (fill the reviewed directory/hash, never guess):

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_upright_mask_thr0p0001_fp075
export PREFLIGHT="/cs/labs/josko/sahar_aharon/repos/foundation_contrastive_diff/results/orientation_preflight_REVIEWED_TIMESTAMP"
export APPROVED_SHA256="REVIEWED_APPROVAL_SHA256"
mkdir -p "$RUN"
touch "$RUN/pipeline.log"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info())"
nohup python -u -m training.run_upright_clinical \
	--preflight-dir "$PREFLIGHT" --approved-preflight-sha256 "$APPROVED_SHA256" \
	>> "$RUN/pipeline.log" 2>&1 &
tail -f "$RUN/pipeline.log"
```

Phase2 automatically rebuilds frozen features, trains/resumes ONLY the corrected
head, checks training reached60epochs, and predicts all100 clinical pairs using
the new best.pt. Prediction restoration reverses the horizontal flip BEFORE
undoing the rotation; output shape and affine must match each original current
NIfTI. All100 raw maps and signed masks are validated, including finite/range and
exact strict+/-0.5 mask checks. The existing no-CC-filter one-pixel detection F1
protocol runs afterward, without tuning. After all checks succeed, the pipeline
automatically commits/pushes exactly `results/clinical100_upright_cutoff0001_<timestamp>`:
100 output.nii.gz,100 mask.nii.gz,source JSONs,clinical CSV/report/metadata,
per-clinician F1,checkpoint/prediction hashes,training config/metrics,best/final
diagnostic summaries and corrected training-input preview. It does not upload
source clinical scans,clinical review PNGs,encoder/checkpoint weights or cache.
Clinical review images remain under the campus run directory and can be rebuilt
locally from the published maps plus the existing clinical images. Both publication
steps stage/commit exact artifact paths only, never unrelated work. Git/network
failure exits nonzero and preserves artifacts; do not regenerate predictions just
to retry publication. No actual campus execution is implied by these local changes.

## GT Cutoff 0.0001 Ablation

User-selected follow-up: `last1_rot90cw_mask_thr0p0001_fp075.yaml` changes the training
signed-mask cutoff from 0.05 to 0.0001. Pixels with abs(GT)>0.0001 become +/-1;
all others remain zero. Conversion precedes tau=0.02, so tau does not remove surviving
unit-valued targets. Foreground weight11, background0.75, L1+L2, frozen encoder,
native-last1 cache, head, rotation3, seed42 and60-epoch budget are unchanged.
Reuse the rotated cache; start a fresh head in a separate run, not from the old model.

For comparison, best.pt selection and primary evaluation retain prediction threshold
0.5 and raw-GT threshold0.05. Additional diagnostic sweeps include raw-GT threshold
0.0001, but do not change checkpoint selection. Full train/val diagnostics run every
epoch, including connected-component statistics enabled in this config. Components
still use the primary evaluation thresholds; the threshold CSV contains the additional
pixelwise GT-cutoff comparisons. Expect extra CPU work for diagnostics.

Complete campus launch (check GPU availability; do not overlap another training job):

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE_DIR=/cs/labs/josko/sahar_aharon/fcd_cache_last1_rot90cw_inference
export HEAD_CONFIG=configs/last1_rot90cw_mask_thr0p0001_fp075.yaml
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_thr0p0001_fp075
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free {f/1e9:.1f}GB / {t/1e9:.1f}GB')"
if mkdir -p "$RUN" && touch "$RUN/train.log"; then
	nohup bash training/run_rotated_mask.sh >> "$RUN/train.log" 2>&1 &
	tail -f "$RUN/train.log"
fi
```

Always set both HEAD_CONFIG and RUN as above: the shared launcher keeps its original
0.05 config/run defaults when they are not supplied. It checks cache/split metadata,
uses the existing pipeline lock, and resumes only the selected run if interrupted.
Do not copy old checkpoints into this new run. Ctrl+C stops tail, not training.
Outputs include checkpoints/best.pt, checkpoints/metrics.csv and
checkpoints/diagnostics/epoch_NNNN_{train,val}.json/.csv.gz/_thresholds.csv.gz.
No campus training is launched by editing this config locally.

## GT Audit Before Choosing a Loss

Run this CPU audit on the campus machine with the original NIfTIs. It does not train
or require a checkpoint. Default scope is all train/val records, never held-out test.
The output directory must be new. It exports actual arrays for selected examples;
PNG colors are not used to estimate GT values.

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export OUT="$PWD/results/gt_audit_$(date +%Y%m%d_%H%M%S)"
export LOG="${OUT}.log"
mkdir -p "$(dirname "$OUT")"
touch "$LOG"
nohup python -u -m evaluation.audit_gt --dataset-root "$DATA" --out-dir "$OUT" \
		--splits train val --img-size 512 --rotation-k 3 >> "$LOG" 2>&1 &
tail -f "$LOG"
```

Ctrl+C stops tail, not the audit. The log is outside OUT to preserve the new-directory
guard. A complete audit can take time on shared storage. `--limit 300` gives a seeded
random smoke sample, not a stratified or full-cohort result; default limit 0 audits all.

Outputs: `index.md`, `report.json`, `summary.csv`, `per_pair.csv`,
`cutoff_diagnostics.png`, and selected `cases/*.png` / `cases/*.npz`.
Each example compares prior/current, fixed-scale continuous GT, GT-only contrast,
sign(GT) without a cutoff, the current abs(GT)>0.05 mask, and support removed at 0.05.
NPZ files retain native GT, preprocessed float32 GT, simulated float16 GT, and model
input images; load with `numpy.load(path, allow_pickle=False)`. These contain medical
images: transfer privately, and do not commit them or whole audit folders by default.
`--no-save-arrays` disables NPZ export; `--overlays-per-group 0` disables examples.

Statistics separate native arrays, training resize/clip/rotation, and simulated
float16 quantization (not a read-back audit of the actual feature cache). They include
sign/support/area, magnitude quantiles, pooled log histograms, tiny connected components,
TV per edge, out-of-range/nonfinite values, support and absolute GT mass retained across
cutoffs 0 through 0.5, entirely removed maps, and float16 support/threshold flips.
Manifest and native-array hashes record provenance. Failures are listed in report.json
and cause a nonzero exit, rather than silently presenting a complete audit.

First inspect whether exact-nonzero support is anatomically plausible or consists of
tiny interpolation tails. Then compare cutoff retention and per-anomaly/group size
before deciding between continuous targets, sign targets, or region-normalized loss.
Do not infer that every nonzero pixel is true pathology, or tune on held-out test.

## Loss-Independent Epoch Diagnostics

Diagnostics are now enabled by default for both train and val, using existing forward
passes. Frozen encoder, loss, optimizer behavior, original metrics.csv, and best.pt
selection by the existing validation Dice are unchanged. The original recipe still
uses GT cutoff 0.05; this change does not launch a no-cutoff experiment.

Each recorded epoch writes `<save_folder>/diagnostics/epoch_NNNN_{train,val}.json`,
per-pair `.csv.gz`, and `_thresholds.csv.gz`. Reports contain:

- Raw-GT, sign-GT and configured-target MAE/MSE; weighted L1/L2 and separate
	foreground/background contributions; region-normalized MAE, soft support Dice and
	sign penalty even when their loss weights are zero. Sign-GT errors are observational,
	not a claim about how a model trained with that alternative objective would perform.
- Positive, negative, unsigned-support and correct-sign Dice (pixel F1), IoU,
	precision/recall, TP/FP/FN, empty predictions, pair hits, wrong-sign overlap,
	predicted area, score percentiles and saturation. Sweep defaults are prediction
	thresholds 0.1/0.25/0.5/0.75 crossed with GT thresholds 0/0.02/0.05; configured
	primary thresholds are always included. These thresholds do not alter training.
- Overall, pathology/nuisance, anomaly and GT-direction groups, unique pair/change-group
	counts, macro means with valid denominators, pooled micro scores, and FP-area tails.
	Undefined empty-reference Dice/recall are null, not automatically perfect scores;
	empty-reference FP area remains visible. This differs from the legacy selection metric.
- Optional `--diagnostics_components`: positive/negative/combined component detection
	F1, HMDR and unmatched detections per pair at the primary thresholds. Uses synthetic
	GT, 8-connectivity and legacy reference-first maximum-overlap greedy matching, without
	the clinical one-sided PAI boost or component cleanup. This is not clinician F1.
- Gradient norm, nonfinite loss/gradient counts, optimizer steps, pending accumulation
	batches, epoch duration, learning rate, peak CUDA allocation and resolved configuration.
	Health logging exposes existing behavior; it does not change step/skipping policy.

Training rows are online, augmented/sampler-observed predictions as weights change,
not fixed-model inference over every training pair; balanced sampling can repeat IDs.
Validation rows come from the usual eval pass. CPU metric sweeps and compression add
overhead; components add more. Use `--diagnostics_every 5` for epoch 1 and every fifth
epoch, customize `--diagnostics_pred_thresholds` / `--diagnostics_gt_thresholds`, or
disable with `--no-diagnostics`. No extra model evaluation pass is performed.

When a new training run is agreed, this records the unchanged baseline plus optional
component diagnostics in a separate run directory, using the existing guarded launcher:

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE_DIR=/cs/labs/josko/sahar_aharon/fcd_cache_last1_rot90cw_inference
export RUN="/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_diagnostics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
touch "$RUN/train.log"
nohup bash training/run_rotated_mask.sh --diagnostics_components >> "$RUN/train.log" 2>&1 &
tail -f "$RUN/train.log"
```

Do not run alongside another GPU job without checking memory. Audit results should
guide the next loss ablation before launching another full training run.

## Clinical Cutoff0.0001 Best-Checkpoint Evaluation

The completed cutoff0.0001 run selects epoch49 by the unchanged original-reference
validation Dice. Evaluate that stored best checkpoint, not last.pt, with the same
100 cases, five observers, prediction threshold0.5, geometry and postprocessing as
the earlier clinical run. The command below verifies checkpoint epoch/config/head
and all500 annotation hashes against the earlier metadata before preflight. It pins
the verified new checkpoint bytes via EXPECTED_SHA256; the old launcher default is
unchanged. No clinical threshold tuning or training is performed.

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLINICAL_ROOT="/cs/labs/josko/sahar_aharon/repos/python_files/annotation tool"
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_thr0p0001_fp075
export OUT="$PWD/review_outputs/clinical100_cutoff0001_epoch49_$(date +%Y%m%d_%H%M%S)"
export LOG="${OUT}.log"
if EXPECTED_SHA256=$(python - <<'PY'
import hashlib, io, json, os
from pathlib import Path
import torch
from evaluation.evaluate_clinical import discover_cases, PHYSICIANS

reference = json.loads(Path("review_outputs/clinical100_20261017_112038/metadata.json").read_text())
payload = (Path(os.environ["RUN"]) / "checkpoints/best.pt").read_bytes()
state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
config = state["run_config"]
assert state["epoch"] == 49, "Expected the selected epoch49 checkpoint"
assert config["target_mode"] == "signed_mask" and config["target_threshold"] == 0.0001, "Training target changed"
assert json.loads(json.dumps(state["head_spec"])) == reference["head_spec"], "Head architecture changed after JSON normalization"
assert config["rotation_k"] == reference["rotation_k"] == 3, "Rotation changed"
assert config["eval_threshold"] == reference["prediction_threshold"] == 0.5, "Prediction threshold changed"
assert reference["num_pairs"] == 100, "Reference cohort size changed"
cases = discover_cases(Path(os.environ["CLINICAL_ROOT"]), 100)
annotation_hashes = {
	f"{case['pair_id']}/{doctor}": hashlib.sha256(Path(case["annotations"][doctor]).read_bytes()).hexdigest()
	for case in cases for doctor in PHYSICIANS
}
assert len(annotation_hashes) == 500 and annotation_hashes == reference["annotation_sha256"], "Clinical annotation cohort changed"
print(hashlib.sha256(payload).hexdigest())
PY
); then
	export EXPECTED_SHA256
	mkdir -p "$(dirname "$OUT")"
	touch "$LOG"
	if bash training/run_clinical100.sh --preflight; then
		nohup bash training/run_clinical100.sh >> "$LOG" 2>&1 &
		tail -f "$LOG"
	fi
fi
```

The guard loads only this trusted study checkpoint. If epoch/config/cohort validation
fails, do not bypass it; investigate the mismatch. Ctrl+C exits tail without stopping
inference. Keep per-case NIfTI/PNG outputs private. Share report.json, per_pair.csv and
metadata.json for comparison with the earlier clinical run; do not stage the whole
review directory or publish medical images. Actual campus checkpoint verification,
GPU inference and clinical comparison remain pending until this command is run there.

## One-Pixel Clinical Detection F1

Score already-saved original-space predictions, without retraining, encoder loading or
another inference pass. This uses the unchanged legacy greedy one-to-one matcher on
8-connected components: at least one same-sign overlapping pixel qualifies, with
threshold0.5 and no component filtering. Combined F1 pools positive/negative counts;
it is not pixel Dice or the boosted legacy per-pair agreement. Reports are against each
of the five clinicians, not a pixelwise consensus. Lower Dice does not establish lower
detection F1. The command selects the completed epoch49 clinical run, checking its
published metadata before scoring. If the run was moved elsewhere, set CLINICAL_RUN
to the private directory containing metadata.json and pair1/output.nii.gz.

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
set -o pipefail
export CLINICAL_ROOT="/cs/labs/josko/sahar_aharon/repos/python_files/annotation tool"
export CLINICAL_RUN="$PWD/review_outputs/clinical100_cutoff0001_epoch49_20261018_085458"
if [ ! -f "$CLINICAL_RUN/pair1/output.nii.gz" ]; then
	export CLINICAL_RUN="$PWD/results/clinical100_cutoff0001_epoch49_20261018_085458"
fi
export OUT="$PWD/results/clinical100_f1_cutoff0001_epoch49_$(date +%Y%m%d_%H%M%S)"
export LOG="${OUT}.log"
mkdir -p "$(dirname "$OUT")"
if python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["CLINICAL_RUN"])
published = Path("results/clinical100_cutoff0001_epoch49_20261018_085458/metadata.json")
actual = json.loads((run / "metadata.json").read_text())
assert actual == json.loads(published.read_text()), "Saved run differs from published clinical provenance"
assert actual["checkpoint_epoch"] == 49 and actual["run_config"]["target_threshold"] == .0001, "Wrong experiment"
assert actual["num_pairs"] == 100 and actual["prediction_threshold"] == .5, "Clinical protocol changed"
PY
then
	if python -u -m evaluation.compare_clinical_models \
		--foundation-only --clinical-run "$CLINICAL_RUN" \
		--clinical-root "$CLINICAL_ROOT" --num-pairs 100 \
		--out-dir "$OUT" 2>&1 | tee -a "$LOG"
	then
		git add -- "$OUT/f1_per_clinician.csv" "$OUT/report.json" "$OUT/metadata.json"
		git commit -m "Publish cutoff 0.0001 epoch49 clinical detection F1"
		git push
	fi
fi
```

The scorer verifies all requested annotations against inference-time SHA256 values,
requires every saved output and rejects a reused output directory. It fails if the
legacy evaluator skips a case. The printed scores and f1_per_clinician.csv give F1+,
F1- and pooled F1all for each clinician. metadata.json records checkpoint identity,
annotation/prediction hashes and the actual legacy protocol. Share only the three
files explicitly staged above. Raw NIfTI, case images and logs remain private;
per-case overlays are disabled. This is a CPU scoring pass, not a GPU training job.

## Clinical 100-Case Inference

`evaluation.evaluate_clinical` applies the frozen encoder and trained head to the100 ICU
pairs in `python_files/annotation tool/Pairs1..Pairs8`, using Avi, Benny, Sigal, Smadar
and Nitzan annotations. No training, synthetic GT cutoff, clinical threshold optimization,
lung cropping, sharpening or component cleanup is performed. The launcher defaults to
the verified epoch45 best-checkpoint SHA256 and prediction threshold0.5; an explicit
EXPECTED_SHA256 override supports a separately verified checkpoint.

Inputs use the SAME raw-NIfTI resize/minmax/rotation sequence as training, NOT the
legacy predictor's transpose. The prediction is inverse-rotated and resized to the
current image's original array coordinates before masks/metrics are computed. Main
review panels are transposed for familiar upright clinical display; the model-input
panel explicitly shows the actual checkpoint input coordinates. No affine canonicalization
or registration is introduced. Scan ordering matches the existing evaluator's sorted
filenames; source paths are saved for auditing chronology.

Publish the local implementation before the campus `git pull`. Only stage the clinical
entry point, launcher, touched tests and study documentation; do not stage unrelated
clinician/LLM experiments or raw medical data. Ensure the campus CLINICAL_ROOT contains
the100 pairs and Annotations tree; these files are verified locally, not on campus.
Override CLINICAL_ROOT if that data lives elsewhere. Parent OV dependencies include
nibabel, scipy, scikit-image, matplotlib, pandas and seaborn in addition to the study
requirements.

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLINICAL_ROOT="/cs/labs/josko/sahar_aharon/repos/python_files/annotation tool"
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_fp075
export OUT="$PWD/review_outputs/clinical100_rot90cw_mask_fp075_$(date +%Y%m%d_%H%M%S)"
export LOG="${OUT}.log"
mkdir -p "$(dirname "$OUT")"
touch "$LOG"
if bash training/run_clinical100.sh --preflight; then
	nohup bash training/run_clinical100.sh >> "$LOG" 2>&1 &
	tail -f "$LOG"
fi
```

The log lives alongside OUT, not inside it, because inference refuses nonempty output
directories. Ctrl+C exits tail without stopping background inference.

Outputs: index.md, report.json, per_pair.csv, checkpoint/config/annotation-hash metadata;
each pair gets raw `output.nii.gz`, thresholded `mask.nii.gz`, source provenance and a
review.png showing both images, model input, prediction and all five clinicians.
Annotation raster warnings are retained per case. Signed Dice/recall are reported
against each clinician and sign-specific3-of-5 consensus. Empty reference masks do not
inflate Dice; their FP area is reported separately. Ellipses are coarse human regions,
not dense pathology masks, so these metrics are exploratory and not comparable to
synthetic validation Dice or the older connected-component PAI. Do not use these100
cases to select the checkpoint or tune thresholds and then call them an untouched test.

## Usage
```bash
python -u -m training.train_foundation_diff --config configs/last1_l1l2.yaml
# override anything on the CLI:
python -u -m training.train_foundation_diff --config configs/last1_l1l2.yaml --epochs 40 --lr 2e-4
```

## Available
| config | features | loss | notes |
|---|---|---|---|
| `last1_l1l2.yaml` | last-1 (90 GB cache, fast) | L1+L2 | needs `fcd_cache_last1` (re-cache with LAST_N_LAYERS=1) |
| `last4_l1l2.yaml` | last-4 (360 GB cache) | L1+L2 | NFS-bound ~20 min/epoch |
| `last4_dpt.yaml` | last-4 | L1+L2 | 2.06M-parameter DPT-style head; separate per-layer fusion |
| `last4_dpt_cross_attention.yaml` | last-4 | L1+L2 | 2.95M-parameter head; bidirectional attention before signed fusion |
| `last1_dpt_cross_attention.yaml` | native last-1 from frozen GPU inference | L1+L2 | 2.06M parameters; two bidirectional blocks, separate cache/run |
| `last1_rot90cw_mask_fp075.yaml` | native last-1, scans and GT rotated 90 degrees clockwise | signed-mask L1+L2 | foreground weight 11, background 0.75; separate cache/run |
| `overfit_l1l2.yaml` | last-4 | L1+L2 | capacity sanity test (16 pairs, 400 ep) |

Loss default here is **L1+L2 (no Dice)** — the overfit test showed Dice hurts on the faint GT.

## Rotated Signed-Mask Run

This is a combined follow-up recipe, not an isolated loss ablation. Keep the frozen
RAD-DINO encoder, single-layer cross-attention head, split, optimizer and 60-epoch
budget unchanged. Do not run it alongside another GPU training job without checking
available GPU memory first.

- `rotation_k: 3`: apply the same 90-degree clockwise quarter-turn to prior, current,
	and continuous GT before frozen feature extraction. No image resampling for rotation.
- Fresh native GPU cache: `fcd_cache_last1_rot90cw_inference`. Do not copy old features.
	Cache metadata records orientation/layers/manifest; mismatches fail. New native
	writes are atomic. Input previews show up to three distinct cases before extraction.
- Cache retains continuous signed GT. During training, `abs(GT)>0.05` becomes
	`sign(GT)` and all other pixels become zero. No-change targets stay zero.
- Signed L1+L2, averaged over all pixels, with weight **11** on masked change and
	**0.75** on background. This is a 25% background discount, not foreground/background
	region normalization. Nuisance-only pairs retain a nonzero penalty.
- Dice and extra sign terms remain off; fixed initial seed 42. Predictions are signed
	change scores, not reconstructed DRR intensities or calibrated probabilities.
- Validation prediction threshold **0.5**, original continuous GT threshold **0.05**.
	Thus the GT reference stays comparable to the old run. Evaluation/review reads both
	thresholds and rotation from checkpoint metadata; CLI overrides remain available.
- Mean, 95th-percentile and maximum nuisance FP area are printed and stored in the
	checkpoint validation dictionary. Do not select improvements using Dice alone.
- Separate run: `run_last1_rot90cw_mask_fp075`; old checkpoints are not reused.
	Relaunching the same command resumes only this run after cache checks.

Complete campus launch after these changes have been published to the remote:

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE_DIR=/cs/labs/josko/sahar_aharon/fcd_cache_last1_rot90cw_inference
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_fp075
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free {f/1e9:.1f}GB / {t/1e9:.1f}GB')"
if mkdir -p "$RUN" && touch "$RUN/train.log"; then
	nohup bash training/run_rotated_mask.sh >> "$RUN/train.log" 2>&1 &
	tail -f "$RUN/train.log"
fi
```

The launcher writes `$RUN/input_preview.png` before inference, uses direct frozen GPU
extraction (not CPU cache conversion), then runs the audited training wrapper. Leaving
`tail` with Ctrl+C does not stop the background pipeline. Inspect the input preview
and early validation maps, ideally by epochs 3-5. Keep the test split held out.
Do not add Dice, boundary relaxation or another architecture change to this run.
If the combined recipe helps, subsequent ablations can identify which change helped.

## DPT Experiment
`training/run_last4_dpt.sh` activates the cluster venv, sets library/data paths, locks the
run directory against duplicate launches, and audits split metadata and cache coverage.
It samples three cached tensors per split for shape/finite checks, then resumes training
if `last.pt` exists. All four cached layers are required; the backbone is never trained.
The final four adjacent blocks are reassembled separately, not canonical spaced DPT taps.
Architecture metadata is saved in new checkpoints and read automatically by RQ1 evaluation.
Use evaluation threshold **0.05** for this config (the evaluator CLI default is 0.1).
The last-four cache exceeds node RAM, so training can remain NFS-bound.

Complete cluster launch (do not run alongside another GPU job without checking memory):
```bash
cd /cs/labs/josko/sahar_aharon/repos && git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE4=/cs/labs/josko/sahar_aharon/fcd_cache_last4
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last4_dpt
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free {f/1e9:.1f}GB / {t/1e9:.1f}GB')"
mkdir -p "$RUN"
nohup bash training/run_last4_dpt.sh >| "$RUN/train.log" 2>&1 &
tail -f "$RUN/train.log"
```

## Single-Layer Cross-Attention
`last1_dpt_cross_attention.yaml` reads [1369,768] tokens per image. One shared
768-to-96 projection and two bidirectional four-head attention blocks precede signed
fusion. The decoder builds its four working scales from that one feature stream, not
four distinct backbone layers. It contains 2,060,033 trainable parameters.

Generate this cache directly from the images with frozen RAD-DINO on the GPU, using
`--last_n_layers 1 --device cuda`. No encoder training or split regeneration occurs.
The wrapper reads `last_hidden_state` for native last-one extraction, rather than
`hidden_states[-1]` used by last-four slicing; do not assume normalization equivalence.
Use the new `fcd_cache_last1_inference` cache and
`run_last1_dpt_cross_attention_inference` run. Do not copy sliced files or checkpoints
into them: inference skips existing cache files and training resumes existing checkpoints.
Preserve the old last-four cache, partial sliced cache and logs for now.

To stop the earlier CPU conversion, first leave `tail -f` with Ctrl+C. On the cluster,
list the exact conversion process before terminating it (this does not target the
parent `bash -c` or unrelated Python jobs):
```bash
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE4=/cs/labs/josko/sahar_aharon/fcd_cache_last4
export SLICE_CACHE=/cs/labs/josko/sahar_aharon/fcd_cache_last1_from_last4
export STOP_PATTERN="^python -u -m training[.]cache_features --dataset_root $DATA --cache_dir $SLICE_CACHE --last_n_layers 1 --slice_last_layer_from $CACHE4 --device cpu$"
pgrep -a -u "$USER" -f "$STOP_PATTERN"
pkill -TERM -u "$USER" -f "$STOP_PATTERN"
pgrep -a -u "$USER" -f "$STOP_PATTERN"
```
Continue only when the final check shows no matching process. If none matched initially,
inspect `ps -u "$USER" -o pid,ppid,etime,args` before assuming the old pipeline is stopped.
SIGTERM makes the conversion exit unsuccessfully, preventing its `&&`-chained training
command from starting. If training already started, identify and stop that exact job
separately before launching GPU inference. No cluster process is stopped by this document.

Only the token payload shrinks fourfold; GT and metadata stay unchanged. The launcher
prints actual cache bytes. Training uses disk files and the OS page cache, not a forced
RAM preload. With enough available RAM, repeated reads can be served from memory;
residency and epoch speed are not guaranteed on a shared node. `num_workers: 0` avoids
worker replication/prefetch overhead. A separate float32 preload could double tensor RAM
and is intentionally not enabled. The existing four-layer runs remain unchanged.

This sequence generates the native last-one cache first, then trains only if inference
succeeds. Use a free GPU for both stages. Batch size 8 and four data workers are starting
settings, not a measured throughput optimum. If caching is interrupted, inspect the last
written file before resuming: unlike slicing, direct extraction does not write atomically
or validate existing files before skipping them.
The explicit `touch` permits append redirection even when zsh has `APPEND_CREATE` disabled;
it preserves existing log contents. If directory/log creation fails, resolve that error
before launching the background job.
```bash
cd /cs/labs/josko/sahar_aharon/repos && git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE1=/cs/labs/josko/sahar_aharon/fcd_cache_last1_inference
export CACHE_DIR="$CACHE1"
export HEAD_CONFIG=configs/last1_dpt_cross_attention.yaml
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_dpt_cross_attention_inference
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free {f/1e9:.1f}GB / {t/1e9:.1f}GB')"
if mkdir -p "$RUN" && touch "$RUN/train.log"; then
	nohup bash -c 'python -u -m training.cache_features --dataset_root "$DATA" --cache_dir "$CACHE1" --last_n_layers 1 --device cuda --batch_size 8 --num_workers 4 && exec bash training/run_last4_dpt.sh' >> "$RUN/train.log" 2>&1 &
	tail -f "$RUN/train.log"
fi
```

## Validation Output Review
Export inside this repository so the campus PC can push the review and this PC can
pull it. Use best.pt at threshold 0.05, with validation only; leave test data untouched
while choosing model/loss changes. Run after GPU training finishes. No retraining occurs.
The optional review mode scores the complete validation split, then runs only selected
pairs again to render source images. Defaults select up to 24 cases per pathology/GT-sign
group (positive, negative, mixed, or below threshold) and 48 nuisance controls:
- Low, middle, high Dice and seeded-random cases in each pathology/sign group.
- Highest-FP and seeded-random nuisance cases.
- One variant per change_group_id per review group, with pair_id fallback.

For four pathologies and both signs, this targets up to 240 panels; mixed-sign groups
can increase this to 336, and below-threshold groups can add more. Actual counts depend
on available distinct change groups. All validation-pair metrics are exported, not just
those selected for visualization. Selection is deliberately enriched, not representative.

Each panel contains prior/current images, signed GT/prediction with fixed [-1,1] limits,
matched zoom views for faint maps, signed masks and current-image overlays. Red is
positive; blue is negative. Contact sheets and index.md navigate the full panels.
Source images use the existing dataset loader's normalization and resizing.

Complete campus launch (OUT is relative to foundation_contrastive_diff):
```bash
cd /cs/labs/josko/sahar_aharon/repos && git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE_DIR=/cs/labs/josko/sahar_aharon/fcd_cache_last1_inference
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_dpt_cross_attention_inference
export OUT="review_outputs/last1_dpt_cross_attention_val_best_$(date +%Y%m%d_%H%M%S)"
python -u -m evaluation.evaluate_rq1 --dataset_root "$DATA" --cache_dir "$CACHE_DIR" --ckpt "$RUN/checkpoints/best.pt" --split val --threshold 0.05 --device cuda --batch_size 8 --num_workers 0 --review --review_per_group 24 --review_nuisance 48 --review_seed 42 --out_dir "$OUT"
```
Outputs include index.md, contact sheets, cases/*.png, report_val.json, per_pair_val.csv,
review_cases.json and review_metadata.json. Provenance records the exact loaded checkpoint
SHA256/epoch/head specification, manifest SHA256, split, threshold and sampling seed.
No checkpoint weights, NIfTI files or feature tensors are exported. Existing nonempty
review directories are rejected to avoid mixing experiments. A missing split cache fails
before inference rather than silently reporting a partial validation set.

After the evaluator succeeds, inspect the index and size, then push only this folder:
```bash
du -sh "$OUT"
git -C .. add -- "foundation_contrastive_diff/$OUT"
git -C .. diff --cached --stat -- "foundation_contrastive_diff/$OUT"
git -C .. commit --only -m "Add single-layer attention validation output review" -- "foundation_contrastive_diff/$OUT"
git -C .. push
```
Do not use git add . or stage external cache/run directories. Each PNG is a separate
small artifact; the evaluator also reports total review size. After the campus push,
on this Windows PC run from the repository root:
```powershell
git pull --ff-only
Get-ChildItem foundation_contrastive_diff/review_outputs -Filter index.md -Recurse
```
Open the generated index in VS Code Markdown Preview and follow contact-sheet/panel
links. The original one-example-per-pathology plots remain for compatibility, but the
full review should use the index and all-pair CSV rather than those first examples.

## Cross-Attention Ablation
The optional `dpt_cross_attention` head retains the frozen last-four cache and the
DPT reassembly/decoder. At each layer, the same projection processes prior/current
features into 96 channels. Two bidirectional cross-attention blocks (four heads each,
with residual paths and feed-forward layers) run on the original 37x37 token grid.
Weights are shared between temporal directions; updates use both streams before either
is overwritten. The resulting current-minus-prior features feed the DPT decoder.
There is no absolute-difference-only path or encoder adaptation. The signed intermediate
features reverse under temporal swapping; exact antisymmetry of the final map is not
enforced. This is Lin-inspired, not a reproduction of their concatenation-based head.

All loss/optimizer/batch settings match the DPT config. The separate run directory avoids
incompatible checkpoint reuse. GPU peak memory, speed and clinical performance remain
unmeasured; full-token attention adds quadratic token cost even at reduced channel width.
Do not stop or overwrite the current DPT run merely because this variant is available.

Complete cluster launch, on a free GPU:
```bash
cd /cs/labs/josko/sahar_aharon/repos && git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE4=/cs/labs/josko/sahar_aharon/fcd_cache_last4
export CACHE1=/cs/labs/josko/sahar_aharon/fcd_cache_last1
export HEAD_CONFIG=configs/last4_dpt_cross_attention.yaml
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last4_dpt_cross_attention
python -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'GPU free {f/1e9:.1f}GB / {t/1e9:.1f}GB')"
mkdir -p "$RUN"
nohup bash training/run_last4_dpt.sh >| "$RUN/train.log" 2>&1 &
tail -f "$RUN/train.log"
```
