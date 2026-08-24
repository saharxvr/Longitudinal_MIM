#!/bin/bash
# SLURM launcher for the foundation_contrastive_diff synthetic training set.
# Runs on the university Linux PCs (HUJI josko cluster), inside the SAME venv
# used to generate the original DRRs.
#
# Submit a single job:
#     sbatch run_generate_pairs.slurm
#
# Submit a parallel split across 8 array tasks (each task does 1/8 of the CTs):
#     sbatch --array=0-7 run_generate_pairs.slurm
#
# Adjust N_PAIRS / OUT_DIR / VENV below before submitting.

#SBATCH --job-name=fcd_drr_gen
#SBATCH --output=/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/logs/fcd_drr_gen_%A_%a.out
#SBATCH --error=/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/logs/fcd_drr_gen_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# --- Configuration (edit these) ---------------------------------------------
# Point VENV at the SAME virtualenv itamar uses for DRR generation.
VENV="${VENV:-/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/.venv}"
N_PAIRS="${N_PAIRS:-12000}"
OUT_DIR="${OUT_DIR:-/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans/final/fcd_train}"

# Repo root is two levels up from this script (foundation_contrastive_diff/data_generation).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/generate_training_set.py"

# --- Parallel slicing via SLURM array ---------------------------------------
SLICE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SLICES="${SLURM_ARRAY_TASK_COUNT:-1}"

# --- Environment ------------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

mkdir -p "$(dirname "${OUT_DIR}")" "$(dirname "${SLURM_SUBMIT_DIR:-$OUT_DIR}")" || true

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Slice ${SLICE_INDEX}/${NUM_SLICES} -> ${OUT_DIR}"
nvidia-smi || true

python "${DRIVER}" \
    -n "${N_PAIRS}" \
    -o "${OUT_DIR}" \
    --slice_index "${SLICE_INDEX}" \
    --num_slices "${NUM_SLICES}"

# After ALL array tasks finish, build the dataset manifest once, e.g.:
#     python build_manifest.py -o "${OUT_DIR}" --split-out
