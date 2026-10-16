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
