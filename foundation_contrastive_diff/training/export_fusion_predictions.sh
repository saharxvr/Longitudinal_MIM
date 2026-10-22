#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="results/clinical100_AB_Itamar_masks_$(date +%Y%m%d_%H%M%S)_$$"
python -u -m evaluation.export_fusion_predictions --out-dir "$OUT" "$@"
git add -- "$OUT"
git commit --only -m "Publish compact A B Itamar Clinical100 prediction masks" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -- "$OUT"
git push origin HEAD:main
echo "[published] $OUT"
