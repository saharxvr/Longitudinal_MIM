#!/usr/bin/env bash
# Few-shot pipeline stages (see FEW_SHOT_DIFF_PLAN.md and configs/README.md "Few-shot ...").
# Usage: bash training/run_few_shot.sh <stage> [extra args]
#   audit | generate <slice> <num_slices> | generate_bench | generate_launch <workers> | generate_status
#   | manifest | stores | benchmark | train | eval_synthetic | eval_icu
# Every stage is resumable/idempotent where possible; no process is stopped by this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${FCD_VENV:-/cs/labs/josko/sahar_aharon/fcd_venv}/bin/activate"
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version 2>/dev/null || echo none)/lib:${LD_LIBRARY_PATH:-}"
# cupy (gryds B-spline CUDA, used by the DRR device insertion) compiles kernels with the venv's NVRTC.
for nvrtc in "${VIRTUAL_ENV:-}"/lib/python3*/site-packages/nvidia/cuda_nvrtc/lib; do
  [[ -d "$nvrtc" ]] && export LD_LIBRARY_PATH="$nvrtc:$LD_LIBRARY_PATH"
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
CONFIG="${CONFIG:-configs/few_shot_6_9_12.yaml}"
value() { python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])' "$CONFIG" "$1"; }
CORPUS="$(value corpus_root)"
RUN="$(value run_root)"
DATA="$(value val_dataset_root)"
VAL_STORE="$(value val_store)"
CLINICAL="${CLINICAL:-}"
ARM_A="${ARM_A:-}"
stage="${1:?stage required}"
shift || true
mkdir -p "$RUN"
case "$stage" in
  audit)
    python -u -m data_generation.audit_fewshot_inputs --source_manifest "$DATA/manifest_split.jsonl" \
      ${CLINICAL:+--clinical_root "$CLINICAL"} --device cuda --output "$RUN/audit_fewshot.json" "$@" ;;
  generate)
    slice="${1:?slice index}"; slices="${2:?number of slices}"; shift 2
    mkdir -p "$CORPUS/logs"
    python -u -m data_generation.generate_supervoxel_corpus --source_manifest "$DATA/manifest_split.jsonl" \
      --output "$CORPUS" --slice_index "$slice" --num_slices "$slices" --device cuda "$@" \
      >> "$CORPUS/logs/generate_slice${slice}.log" 2>&1 ;;
  generate_bench)
    # One worker, 3 CTs x 4 scenes into a scratch folder: per-stage times, GPU/RAM peak, full-corpus projection.
    python -u -m data_generation.generate_supervoxel_corpus --source_manifest "$DATA/manifest_split.jsonl" \
      --output "${CORPUS}_bench" --device cuda --benchmark --scenes_per_ct 4 "$@" ;;
  generate_launch)
    # N background workers on this PC (slice i of N each; resumable). Keep N fixed for one corpus:
    # stop all workers before relaunching with another N. GPUS="0 1" spreads workers over GPUs.
    workers="${1:?number of workers}"; shift
    mkdir -p "$CORPUS/logs"
    read -r -a gpus <<< "${GPUS:-0}"
    for ((i = 0; i < workers; i++)); do
      pid_file="$CORPUS/logs/generate_slice$i.pid"
      if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "worker $i already running (pid $(cat "$pid_file"))"; continue
      fi
      gpu="${gpus[$((i % ${#gpus[@]}))]}"
      echo "=== $(date '+%F %T') launched worker $i/$workers (args: $*) ===" >> "$CORPUS/logs/generate_slice$i.log"
      CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="${WORKER_THREADS:-2}" nohup python -u \
        -m data_generation.generate_supervoxel_corpus --source_manifest "$DATA/manifest_split.jsonl" \
        --output "$CORPUS" --slice_index "$i" --num_slices "$workers" --device cuda "$@" \
        >> "$CORPUS/logs/generate_slice$i.log" 2>&1 &
      echo $! > "$pid_file"
      echo "worker $i/$workers: pid $!, GPU $gpu, log $CORPUS/logs/generate_slice$i.log"
    done
    echo "Stop all workers: kill \$(cat $CORPUS/logs/generate_slice*.pid)" ;;
  generate_status)
    python -u -m data_generation.generate_supervoxel_corpus --output "$CORPUS" --status
    for log in "$CORPUS"/logs/generate_slice*.log; do
      [[ -f "$log" ]] && echo "$(basename "$log"): $(tail -n 1 "$log")"
    done
    if command -v nvidia-smi > /dev/null; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi ;;
  manifest)
    python -u -m data_generation.generate_supervoxel_corpus --output "$CORPUS" --build_manifest ;;
  stores)
    python -u -m datasets.image_store scenes --root "$CORPUS" --destination "$CORPUS/_store"
    python -u -m datasets.image_store pairs --root "$DATA" --split val --destination "$VAL_STORE" ;;
  benchmark)
    python -u -m training.train_few_shot --config "$CONFIG" --benchmark "$@" | tee "$RUN/benchmark.log" ;;
  train)
    exec 8>"$RUN/.few_shot.lock"
    flock -n 8 || { echo "Few-shot training already running for $RUN"; exit 1; }
    resume=""; [[ -f "$RUN/last.pt" ]] && resume="--resume"
    python -u -m training.train_few_shot --config "$CONFIG" $resume "$@" >> "$RUN/train.log" 2>&1 ;;
  eval_synthetic)
    out="$RUN/eval_synthetic_$(date +%Y%m%d_%H%M%S)"
    python -u -m evaluation.evaluate_few_shot synthetic --run_root "$RUN" --dataset_root "$DATA" \
      --split val --store "$VAL_STORE" --output "$out" "$@" ;;
  eval_icu)
    : "${CLINICAL:?set CLINICAL to the ICU annotation-tool root}"
    out="$RUN/eval_icu_$(date +%Y%m%d_%H%M%S)"
    python -u -m evaluation.evaluate_few_shot icu --run_root "$RUN" --clinical_root "$CLINICAL" \
      --support_modes consensus consensus_2 reader:Avi reader:Benny reader:Sigal reader:Smadar reader:Nitzan \
      --retrieval --synthetic_supports "$DATA" --store "$VAL_STORE" \
      ${ARM_A:+--arm_a_checkpoint "$ARM_A" --finetune_steps 20} --output "$out" "$@" ;;
  *) echo "Unknown stage $stage"; exit 2 ;;
esac
