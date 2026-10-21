#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HEAD_CONFIG="${HEAD_CONFIG:-configs/layer_depth_6_9_12_aug_30.yaml}"
config_data=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["dataset_root"])' "$HEAD_CONFIG")
if [[ -n "${DATA:-}" && "$DATA" != "$config_data" ]]; then
    echo "DATA differs from the training config; refusing to preflight a different dataset"
    exit 1
fi
export DATA="$config_data"
export BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-$(python -c 'import sys,yaml; a=int(yaml.safe_load(open(sys.argv[1]))["accum"]); assert a>0; print(((10+a-1)//a)*a)' "$HEAD_CONFIG")}"
export RUN="$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(open(sys.argv[1]))["save_folder"]).parent)' "$HEAD_CONFIG")"
mkdir -p "$RUN"
exec 8>"$RUN/.layer_pipeline.lock"
flock -n 8 || { echo "Layer-ablation launcher already running"; exit 1; }
PREFLIGHT_ONLY=1 bash training/run_last4_dpt.sh
if [[ ! -f "$RUN/checkpoints/shared_last.pt" ]]; then
    python -u -m training.train_layer_ablation --config "$HEAD_CONFIG" --benchmark_batches 100 \
        --benchmark-warmup "$BENCHMARK_WARMUP"
fi
exec python -u -m training.train_layer_ablation --config "$HEAD_CONFIG" --resume --publish
