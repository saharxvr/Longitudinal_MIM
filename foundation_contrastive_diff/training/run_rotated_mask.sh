#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-/cs/labs/josko/sahar_aharon/fcd_train}"
export CACHE_DIR="${CACHE_DIR:-/cs/labs/josko/sahar_aharon/fcd_cache_last1_rot90cw_inference}"
export RUN="${RUN:-/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_fp075}"
export HEAD_CONFIG="${HEAD_CONFIG:-configs/last1_rot90cw_mask_fp075.yaml}"
mkdir -p "$RUN"
exec 8>"$RUN/.pipeline.lock"
flock -n 8 || { echo "The rotated-mask pipeline is already running in $RUN"; exit 1; }
git rev-parse HEAD
python -u -m training.cache_features --dataset_root "$DATA" --cache_dir "$CACHE_DIR" \
    --last_n_layers 1 --rotation_k 3 --device cuda --batch_size 8 --num_workers 4 \
    --preview_path "$RUN/input_preview.png"
exec bash training/run_last4_dpt.sh "$@"