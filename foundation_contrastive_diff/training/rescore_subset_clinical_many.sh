#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: bash training/rescore_subset_clinical_many.sh"
    echo "CPU-only one-to-many rescoring of the saved seed42 ICU100 predictions."
    echo "No training is started/stopped. Aggregate reports are pushed on success."
    exit 0
fi
[[ $# -eq 0 ]] || { echo "Unexpected arguments; use --help" >&2; exit 2; }
STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
INFERENCE="${INFERENCE_DIR:-/cs/labs/josko/sahar_aharon/fcd_runs/clinical100_subset_seed42_20261127_153538_98466/predictions}"
PREVIOUS="${ORIGINAL_REPORT:-$STUDY_ROOT/results/clinical100_subset_seed42_ov_20261127_153538_98466}"
test -f "$INFERENCE/complete.json"
test -f "$PREVIOUS/summary.json"
test "$(git branch --show-current)" = main
exec 8>"${INFERENCE}.rescoring_launcher.lock"
flock -n 8 || { echo "A CPU rescoring job already holds this lock." >&2; exit 1; }
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="results/clinical100_subset_seed42_many_$STAMP"
LOG="$(dirname "$INFERENCE")/rescoring_$STAMP.log"
nohup python -u -m evaluation.rescore_subset_clinical_many \
    --inference-dir "$INFERENCE" --original-report "$PREVIOUS" \
    --clinical-root "$STUDY_ROOT/../python_files/annotation tool" \
    --out-dir "$OUT" --publish > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$LOG.pid"
echo "PID=$PID"
echo "LOG=$LOG"
echo "REPORT=$OUT"
sleep 10
if ! kill -0 "$PID" 2>/dev/null; then
    tail -n 80 "$LOG"
    exit 1
fi
ps -p "$PID" -o pid,etime,args
tail -n 30 "$LOG"
echo "CPU rescoring started; successful completion pushes eight aggregate files."
