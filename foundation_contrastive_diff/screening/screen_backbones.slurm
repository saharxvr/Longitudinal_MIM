#!/bin/bash
# SLURM launcher for the backbone-fitness screen (cache + cos-correlation + linear probe).
# Runs one or more frozen backbones on a fixed 2.5k/1.5k subset for a fair comparison.
#
#   mkdir -p /cs/labs/josko/sahar_aharon/logs        # output dir must exist first
#   sbatch screening/screen_backbones.slurm                       # rad_dino + torchxrayvision
#   BACKBONES="rad_dino:37" sbatch screening/screen_backbones.slurm   # just one
#
# itamar_ae is NOT here (its module clashes with this study's constants) — use
# screening/cache_itamar_ae.py from python_files/, then probe with --grid 32.

#SBATCH --job-name=fcd_screen
#SBATCH --output=/cs/labs/josko/sahar_aharon/logs/fcd_screen_%j.out
#SBATCH --error=/cs/labs/josko/sahar_aharon/logs/fcd_screen_%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-/cs/labs/josko/sahar_aharon/fcd_train}"
export SCREEN="${SCREEN:-/cs/labs/josko/sahar_aharon/fcd_screen}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # foundation_contrastive_diff

python -c "import torchxrayvision" 2>/dev/null || pip install -q torchxrayvision
python -c "import open_clip" 2>/dev/null || pip install -q open_clip_torch

# spec = "name:grid". Full frozen-foundation-model sweep (correlation + probe) on the 2.5k subset.
BACKBONES="${BACKBONES:-rad_dino:37 clip_vitb16:14 pubmedclip:7 biomedclip:14 sam_vitb:64 medsam:64}"
for spec in $BACKBONES; do
  bb="${spec%%:*}"; grid="${spec##*:}"
  echo "================ screening ${bb} (grid ${grid}) ================"
  python -m screening.cache_backbone_subset -o "$DATA" --backbone "$bb" \
      --cache_dir "$SCREEN/$bb" --n_train 2500 --n_val 1500
  python -m evaluation.feature_change_correlation -o "$DATA" \
      --cache_dir "$SCREEN/$bb" --split val --feat_last_k 0 --grid "$grid"
  python -m evaluation.linear_probe -o "$DATA" \
      --cache_dir "$SCREEN/$bb" --feat_last_k 0 --grid "$grid" --epochs 12
done
echo "================ screen done ================"
