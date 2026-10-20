#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH="/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA="${DATA:-/cs/labs/josko/sahar_aharon/fcd_train}"
export CACHE4="${CACHE4:-/cs/labs/josko/sahar_aharon/fcd_cache_last4}"
export CACHE1="${CACHE1:-/cs/labs/josko/sahar_aharon/fcd_cache_last1_from_last4}"
export HEAD_CONFIG="${HEAD_CONFIG:-configs/last4_dpt.yaml}"
default_run=$(python -c 'import pathlib,sys,yaml; print(pathlib.Path(yaml.safe_load(open(sys.argv[1]))["save_folder"]).parent)' "$HEAD_CONFIG")
export RUN="${RUN:-$default_run}"
cache_layers=$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["feat_last_k"])' "$HEAD_CONFIG")
case "$cache_layers" in
    1) export CACHE_DIR="${CACHE_DIR:-$CACHE1}" ;;
    4) export CACHE_DIR="${CACHE_DIR:-$CACHE4}" ;;
    *) echo "Expected a one- or four-layer config"; exit 1 ;;
esac
export CACHE_LAYERS="$cache_layers"
export NO_CACHE="$(python -c 'import sys,yaml; print(int(bool(yaml.safe_load(open(sys.argv[1])).get("no_cache", False))))' "$HEAD_CONFIG")"
export ROTATION_K="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("rotation_k", 0))' "$HEAD_CONFIG")"
export FLIP_HORIZONTAL="$(python -c 'import sys,yaml; print(int(bool(yaml.safe_load(open(sys.argv[1])).get("flip_horizontal", False))))' "$HEAD_CONFIG")"
mkdir -p "$RUN"
exec 9>"$RUN/.train.lock"
flock -n 9 || { echo "A DPT run already holds the lock in $RUN"; exit 1; }
git rev-parse HEAD
python -u - <<'PY'
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

from data_generation.split_dataset import patient_key

manifest = Path(os.environ['DATA']) / 'manifest_split.jsonl'
payload = manifest.read_bytes()
records = [json.loads(line) for line in payload.splitlines() if line.strip()]
if not records:
    raise SystemExit('Empty split manifest')
print('manifest SHA256:', hashlib.sha256(payload).hexdigest(), flush=True)
rotation_k = int(os.environ.get('ROTATION_K', '0'))
flip_horizontal = bool(int(os.environ.get('FLIP_HORIZONTAL', '0')))
metadata = Path(os.environ['CACHE_DIR']) / 'cache_config.json'
online = bool(int(os.environ.get('NO_CACHE', '0')))
if not online and (rotation_k or flip_horizontal or metadata.is_file()):
    if not metadata.is_file():
        raise SystemExit('Rotated run requires cache_config.json from native extraction')
    cache_config = json.loads(metadata.read_text())
    cache_config.setdefault('flip_horizontal', False)
    for key, expected in (('rotation_k', rotation_k),
                          ('flip_horizontal', flip_horizontal),
                          ('last_n_layers', int(os.environ['CACHE_LAYERS'])),
                          ('manifest_sha256', hashlib.sha256(payload).hexdigest())):
        if cache_config.get(key) != expected:
            raise SystemExit(f'Cache configuration mismatch: {key}')
patients, groups = {}, {}
counts = Counter()
samples = defaultdict(list)
seen_pairs = set()
cache_bytes = 0
patch_dim = int(os.environ['CACHE_LAYERS']) * 768
for record in records:
    split = record.get('split')
    if split not in ('train', 'val', 'test'):
        raise SystemExit(f'Invalid split: {split}')
    patient = patient_key(record.get('case', '')) or record.get('patient_key')
    if not patient:
        raise SystemExit('Missing patient identity; cannot audit split')
    if patients.setdefault(patient, split) != split:
        raise SystemExit(f'Patient crosses splits: {patient}')
    group = record.get('change_group_id')
    if group and groups.setdefault(group, split) != split:
        raise SystemExit(f'Change group crosses splits: {group}')
    pair_id = record['pair_id']
    if pair_id in seen_pairs:
        raise SystemExit(f'Duplicate pair: {pair_id}')
    seen_pairs.add(pair_id)
    counts[split] += 1
    if online:
        continue
    path = Path(os.environ['CACHE_DIR']) / pair_id / 'feat.pt'
    if not path.is_file():
        raise SystemExit(f'Missing cache file: {path}')
    cache_bytes += path.stat().st_size
    if len(samples[split]) < 3:
        samples[split].append(path)
for split in ('train', 'val', 'test'):
    if not counts[split]:
        raise SystemExit(f'Empty split: {split}')
    if online:
        continue
    for path in samples[split]:
        cached = torch.load(path, map_location='cpu', weights_only=False)
        if cached.get('rotation_k', 0) != rotation_k:
            raise SystemExit(f'Cache rotation mismatch in {path}')
        if cached.get('flip_horizontal', False) != flip_horizontal:
            raise SystemExit(f'Cache horizontal flip mismatch in {path}')
        for key, shape in (('p_prior', (1369, patch_dim)), ('p_curr', (1369, patch_dim)),
                           ('cls_prior', (768,)), ('cls_curr', (768,)),
                           ('gt_diff', (1, 512, 512))):
            tensor = cached[key]
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise SystemExit(f'Invalid {key} in {path}: {tuple(tensor.shape)}')
if online:
    from datasets import LongitudinalPairDataset
    for split in ('train', 'val'):
        dataset = LongitudinalPairDataset(os.environ['DATA'], split=split, rotation_k=rotation_k,
                                          flip_horizontal=flip_horizontal)
        for record in dataset.records:
            for key, fallback in (('prior_path', 'prior.nii.gz'), ('current_path', 'current.nii.gz'),
                                  ('diff_map_path', 'diff_map.nii.gz')):
                path = Path(dataset._abspath(record, key, fallback))
                if not path.is_file():
                    raise SystemExit(f'Missing online input: {path}')
        for index in range(min(3, len(dataset))):
            sample = dataset[index]
            for key in ('img_prior', 'img_curr', 'gt_diff'):
                tensor = sample[key]
                if tuple(tensor.shape) != (1, 512, 512) or not torch.isfinite(tensor).all():
                    raise SystemExit(f'Invalid online {key}: {sample["pair_id"]}')
    print('PASS: disjoint manifest; complete train/val image/GT paths; sampled finite upright tensors:',
          dict(counts), flush=True)
    print('Online frozen encoder: no feature cache read; held-out test images not loaded', flush=True)
else:
    print('PASS: disjoint manifest, complete cache coverage:', dict(counts), flush=True)
    print('PASS: sampled shapes/finite values (up to three files per split)', flush=True)
    print(f"cache: {os.environ['CACHE_DIR']} ({cache_bytes/1e9:.1f} GB on disk); OS page cache, not forced RAM preload", flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; check venv and NVIDIA library path')
free, total = torch.cuda.mem_get_info()
print(f'torch {torch.__version__}; GPU free {free/1e9:.1f} / {total/1e9:.1f} GB', flush=True)
PY
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi
exec python -u -m training.train_foundation_diff --config "$HEAD_CONFIG" \
    --dataset_root "$DATA" --cache_dir "$CACHE_DIR" \
    --save_folder "$RUN/checkpoints" --plots_folder "$RUN/plots" --resume "$@"