#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HEAD_CONFIG="${HEAD_CONFIG:-configs/layer_fusion_6_9_12_ce_30.yaml}"
exec bash training/run_layer_ablation.sh
