#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(git -C .. branch --show-current)" != main ]]; then
    echo "Run this from main; no branch was changed."
    exit 1
fi
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0))'
RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_layer_depth_6_9_12_aug_30/checkpoints
OUT="results/layer_prediction_review_$(date +%Y%m%d_%H%M%S)_$$"
python -u -m evaluation.export_layer_predictions --checkpoints "$RUN" --out-dir "$OUT"
REL="foundation_contrastive_diff/$OUT"
git -C .. add -- "$REL"
git -C .. diff --cached --check -- "$REL"
git -C .. commit --only -m "Export selected layer-ablation prediction maps" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -- "$REL"
git -C .. push origin HEAD:main
echo "Pushed $REL"
