#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_layer_fusion_6_9_12_ce_30
exec 8>"$RUN/.clinical_ov.lock"
flock -n 8 || { echo "An OV evaluation already holds the run lock"; exit 1; }
SOURCE="${CLINICAL_RUN:-$(python -c 'from pathlib import Path; import sys; p=sorted(p for p in Path(sys.argv[1]).glob("clinical100_*") if p.is_dir()); assert p, "No clinical inference directory"; print(p[-1])' "$RUN")}"
OUT="results/clinical100_fusion_ov_thr075_cc250_$(date +%Y%m%d_%H%M%S)_$$"
python -u -m evaluation.evaluate_fusion_ov \
    --clinical-run "$SOURCE" \
    --clinical-root "/cs/labs/josko/sahar_aharon/repos/python_files/annotation tool" \
    --out-dir "$OUT"
python -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); files=[f for f in p.rglob("*") if f.is_file()]; assert files and all(f.suffix in (".json", ".csv", ".png", ".log", ".md") for f in files), "Unexpected export file"; assert sum(f.stat().st_size for f in files)<100*1024**2, "OV export exceeds 100 MiB"; print("Compact OV export verified:", p)' "$OUT"
git add -- "$OUT"
git commit --only -m "Publish seven-arm Clinical100 OV at threshold 0.75 and CC cutoff 250" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>" -- "$OUT"
git push origin HEAD:main
echo "[published] $OUT"
