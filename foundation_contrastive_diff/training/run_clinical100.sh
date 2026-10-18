#!/usr/bin/env bash
set -euo pipefail

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$STUDY_ROOT")"
cd "$STUDY_ROOT"
source "${FCD_VENV:-/cs/labs/josko/sahar_aharon/fcd_venv}/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -f /proc/driver/nvidia/version ]]; then
    DRIVER="$(awk '{print $8}' /proc/driver/nvidia/version)"
    export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$DRIVER/lib:${LD_LIBRARY_PATH:-}"
fi
export CLINICAL_ROOT="${CLINICAL_ROOT:-$REPO_ROOT/python_files/annotation tool}"
export RUN="${RUN:-/cs/labs/josko/sahar_aharon/fcd_runs/run_last1_rot90cw_mask_fp075}"
export OUT="${OUT:-$STUDY_ROOT/review_outputs/clinical100_rot90cw_mask_fp075_$(date +%Y%m%d_%H%M%S)}"

python -u -m evaluation.evaluate_clinical \
    --clinical_root "$CLINICAL_ROOT" \
    --ckpt "$RUN/checkpoints/best.pt" \
    --expected_sha256 "${EXPECTED_SHA256:-0b7b49cc916806453ae5a2348e8b0546dd59dfbe5ff0449c034ba43d214a8053}" \
    --out_dir "$OUT" --device cuda --threshold 0.5 --num_pairs 100 "$@"