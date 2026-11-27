#!/usr/bin/env bash
set -euo pipefail

STOP_TRAINING=0
case "${1:-}" in
    --stop-training) STOP_TRAINING=1; shift ;;
    --help|-h)
        echo "Usage: bash training/run_subset_clinical_ov.sh [--stop-training]"
        echo "Launch seed42 ICU100 OV in the background and push aggregate results."
        echo "--stop-training terminates only the verified owned subset-sweep process tree."
        exit 0 ;;
esac
[[ $# -eq 0 ]] || { echo "Unexpected arguments; use --help" >&2; exit 2; }
[[ "$(hostname -s)" == cas703 ]] || { echo "Run this launcher on cas703 only." >&2; exit 1; }

STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STUDY_ROOT"
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run_architecture_data_block9_aug_subset_epochs
CFG=/cs/labs/josko/sahar_aharon/fcd_runs/run_architecture_data_block9_aug/locked_rank8.yaml
CLINICAL_ROOT="$STUDY_ROOT/../python_files/annotation tool"
test -f "$CFG"
test -d "$CLINICAL_ROOT/Annotations"
test "$(git branch --show-current)" = main
for FRACTION in 0.1 0.25 0.5 1; do
    test -f "$RUN/fraction_${FRACTION}_seed_42/complete.json"
done
exec 8>"$RUN/.icu_seed42_launcher.lock"
flock -n 8 || { echo "An ICU100 evaluation already holds the launcher lock." >&2; exit 1; }

python - "$RUN" "$STOP_TRAINING" <<'PY_STOP'
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def owned_processes():
    rows = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,args="], text=True
    ).splitlines()
    result = {}
    for row in rows:
        pid, parent, command = row.split(None, 2)
        result[int(pid)] = (int(parent), shlex.split(command))
    return result


def process_identity(pid):
    try:
        path = Path(f"/proc/{pid}")
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        command = tuple(part.decode() for part in (path / "cmdline").read_bytes().split(b"\0") if part)
        return path.stat().st_uid, fields[19], command, fields[0]
    except FileNotFoundError:
        return None


def option(tokens, name):
    if name in tokens and tokens.index(name) + 1 < len(tokens):
        return tokens[tokens.index(name) + 1]
    return None


def stop_training(run_root, allowed):
    processes = owned_processes()
    roots = [
        pid for pid, (_, tokens) in processes.items()
        if option(tokens, "-m") == "training.train_subset_architecture_ablation"
        and option(tokens, "--mode") == "sweep"
        and option(tokens, "--run-root") == run_root
    ]
    if len(roots) > 1:
        raise SystemExit("Multiple matching launchers; inspect before stopping.")
    if roots and not allowed:
        raise SystemExit("Training is running. Use --stop-training only when ready to stop it.")
    targets = set(roots)
    while True:
        expanded = targets | {
            pid for pid, (parent, _) in processes.items() if parent in targets
        }
        if expanded == targets:
            break
        targets = expanded
    original = {pid: process_identity(pid) for pid in targets}
    print("Stopping verified training PIDs:", sorted(targets), flush=True)
    for pid in roots + sorted(targets - set(roots), reverse=True):
        current, expected = process_identity(pid), original[pid]
        if current is None or current[3] == "Z":
            continue
        if expected is None or current[0] != os.getuid() or current[:3] != expected[:3]:
            raise SystemExit("Process identity changed; refusing termination.")
        if tuple(processes[pid][1]) != current[2]:
            raise SystemExit("Process command changed; refusing termination.")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"PID {pid} already exited.", flush=True)
    for _ in range(30):
        remaining = [
            pid for pid in targets
            if (state := process_identity(pid)) is not None
            and original[pid] is not None
            and state[:2] == original[pid][:2] and state[3] != "Z"
        ]
        if not remaining:
            break
        time.sleep(1)
    else:
        raise SystemExit(f"Still exiting: {remaining}; evaluation not started.")
    print("Saved checkpoints preserved; partial-epoch work may be lost.", flush=True)


if __name__ == "__main__":
    stop_training(sys.argv[1], sys.argv[2] == "1")
PY_STOP

if pgrep -u "$USER" -af 'training[.](train_|run_)'; then
    echo "Another training process remains. Evaluation not started." >&2
    exit 1
else
    STATUS=$?
    [[ $STATUS -eq 1 ]] || { echo "Process check failed." >&2; exit "$STATUS"; }
fi

export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
df -h /cs/labs/josko

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WORK="/cs/labs/josko/sahar_aharon/fcd_runs/clinical100_subset_seed42_$STAMP"
OUT="results/clinical100_subset_seed42_ov_$STAMP"
mkdir "$WORK"
LOG="$WORK/evaluation.log"

nohup python -u -m evaluation.evaluate_subset_clinical_ov \
    --config "$CFG" --run-root "$RUN" --clinical-root "$CLINICAL_ROOT" \
    --inference-dir "$WORK/predictions" --out-dir "$OUT" \
    --seed 42 --fractions 0.1 0.25 0.5 1.0 --device cuda --publish \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$WORK/evaluation.pid"
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
echo "Launcher is alive; completion will print '[ICU publish] pushed 8 aggregate files'."
echo "Training will not restart automatically. This launcher never pulls code."
