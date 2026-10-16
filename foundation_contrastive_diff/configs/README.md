# Run configs

YAML configs for `training.train_foundation_diff`. A config's values become the arg
defaults; anything you pass on the CLI still overrides it. Each run also writes its fully
resolved parameters to `<save_folder>/run_config.json` (a record of exactly what ran).

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
| `overfit_l1l2.yaml` | last-4 | L1+L2 | capacity sanity test (16 pairs, 400 ep) |

Loss default here is **L1+L2 (no Dice)** — the overfit test showed Dice hurts on the faint GT.

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
Use the best checkpoint, not necessarily the last epoch, and the existing threshold
0.05. Keep the test split untouched while choosing model/loss changes. Run after the
current GPU training finishes to avoid competing for memory. This does not resume training:
```bash
cd /cs/labs/josko/sahar_aharon/repos && git pull --ff-only
cd foundation_contrastive_diff
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE_DIR=/cs/labs/josko/sahar_aharon/fcd_cache_last1_inference
export RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_dpt_cross_attention_inference
python -u -m evaluation.evaluate_rq1 --dataset_root "$DATA" --cache_dir "$CACHE_DIR" --ckpt "$RUN/checkpoints/best.pt" --split val --threshold 0.05 --device cuda --batch_size 8 --num_workers 0 --out_dir "$RUN/review_val_best"
```
Outputs are `report_val.json`, `per_type_val.png` and `examples_val.png`. The last
contains continuous signed GT/prediction maps at a fixed [-1,1] color scale, one
example per pathology. It is not a best/worst/random sample and contains neither
source prior/current images nor nuisance examples; do not base the next experiment
solely on this panel. Also inspect training curves and per-direction sensitivities.
A fuller review should include source images, thresholded signed masks, each pathology,
both appearance/resolution, and nuisance controls, with reproducible case selection.

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
