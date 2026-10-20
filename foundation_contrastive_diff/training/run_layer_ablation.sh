#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HEAD_CONFIG="${HEAD_CONFIG:-configs/layer_depth_6_9_12_aug_30.yaml}"
export RUN="$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(open(sys.argv[1]))["save_folder"]).parent)' "$HEAD_CONFIG")"
mkdir -p "$RUN"
exec 8>"$RUN/.layer_pipeline.lock"
flock -n 8 || { echo "Layer-ablation launcher already running"; exit 1; }
PREFLIGHT_ONLY=1 bash training/run_last4_dpt.sh
if [[ ! -f "$RUN/checkpoints/shared_last.pt" ]]; then
    python -u -m training.train_layer_ablation --config "$HEAD_CONFIG" --benchmark_batches 100
fi
exec python -u -m training.train_layer_ablation --config "$HEAD_CONFIG" --resume --publish
