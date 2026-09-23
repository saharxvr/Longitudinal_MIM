# RQ1 Training & Evaluation — Campus PC Runbook

How to train the D-GLoRI change-map head (frozen RAD-DINO backbone) and evaluate it
on the HUJI josko cluster. Assumes generation + manifest + split are already done
(see `data_generation/RUN_ON_CLUSTER.md`).

Paths below assume:
- Repo:    `/cs/labs/josko/sahar_aharon/repos`
- venv:    `/cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new`
- Dataset: `/cs/labs/josko/sahar_aharon/fcd_train`  (contains `manifest_split.jsonl`)
- Cache:   `/cs/labs/josko/sahar_aharon/fcd_cache`
- Outputs: `/cs/labs/josko/sahar_aharon/fcd_runs/run1`

Adjust if yours differ.

---

## 0. One-time setup (per PC / login)

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only

source /cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/bin/activate
export LD_LIBRARY_PATH=/cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RAD-DINO backbone deps (transformers >= 4.40). It is usually ALREADY in the shared
# venv, so check first and only install if missing/old:
python -c "import transformers; print('transformers', transformers.__version__)" || \
    PIP_USER=0 pip install -r foundation_contrastive_diff/requirements.txt
# NOTE: if pip errors "Will not install to the user site ...", transformers is already
# present — either skip, or force a venv install with the PIP_USER=0 prefix shown above.

# sanity: GPU present?
python -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,round(p.total_memory/1024**3,1),'GiB');print('cuda:',torch.cuda.is_available())"
```

All commands below run from the package dir so `-m` module paths resolve:

```bash
cd /cs/labs/josko/sahar_aharon/repos/foundation_contrastive_diff
```

---

## 1. Cache frozen features (run once, per split)

The backbone is frozen, so we run RAD-DINO **once** and store patch/CLS tokens per pair.
Training then reads these `feat.pt` tensors — no backbone, no NIfTIs, GPU barely used.

> First RAD-DINO use downloads ~346 MB from HuggingFace — the node needs internet.
> If the compute node is offline, see **Offline RAD-DINO** at the bottom.

```bash
for S in train val test; do
  python -m training.cache_features \
      -o /cs/labs/josko/sahar_aharon/fcd_train \
      --cache_dir /cs/labs/josko/sahar_aharon/fcd_cache \
      --split $S --batch_size 8 --num_workers 4
done
```

It is **resumable** (existing `feat.pt` are skipped). Check progress:

```bash
find /cs/labs/josko/sahar_aharon/fcd_cache -name feat.pt | wc -l
```

---

## 2. Train the head

```bash
cd /cs/labs/josko/sahar_aharon/repos/foundation_contrastive_diff
source /cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run1
mkdir -p $RUN
rm -f $RUN/train.log $RUN/train.pid          # zsh noclobber-safe

nohup python -u -m training.train_foundation_diff \
    -o /cs/labs/josko/sahar_aharon/fcd_train \
    --cache_dir /cs/labs/josko/sahar_aharon/fcd_cache \
    --save_folder $RUN/checkpoints \
    --plots_folder $RUN/plots \
    --epochs 50 --batch_size 8 --accum 4 --lr 3e-4 --balanced \
    > $RUN/train.log 2>&1 &
echo $! > $RUN/train.pid
```

Notes:
- `--balanced` = class-balanced sampler over anomaly type (counters the `none` majority).
- Cheap head-only training: bump `--batch_size` (e.g. 32–64) if the GPU has room.
- Full backbone-on-the-fly (skip step 1): add `--no_cache` and drop `--cache_dir`.

---

## 3. Track training MID-RUN

Every epoch the trainer appends a row to `metrics.csv` and redraws `train_curves.png`,
so you can watch progress live without stopping anything:

```bash
RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run1

# live stdout (unbuffered thanks to -u)
tail -f $RUN/train.log

# metrics table so far (epoch, lr, train loss, val Dice/IoU, sens+/-)
column -s, -t $RUN/checkpoints/metrics.csv | less -S

# is it still running?
ps -o pid,etime,%cpu,%mem -p $(cat $RUN/train.pid)
```

Pull the curve + sample plots to your laptop to eyeball them:

```bash
# from YOUR machine
scp <user>@<pc>.cs.huji.ac.il:/cs/labs/josko/sahar_aharon/fcd_runs/run1/plots/train_curves.png .
scp <user>@<pc>.cs.huji.ac.il:/cs/labs/josko/sahar_aharon/fcd_runs/run1/plots/val_best_samples.png .
```

Checkpoints: `checkpoints/last.pt` (every epoch) and `checkpoints/best.pt` (best val Dice).

Stop early: `kill $(cat $RUN/train.pid)` — `best.pt` is already saved.

---

## 4. Evaluate (held-out test split)

```bash
RUN=/cs/labs/josko/sahar_aharon/fcd_runs/run1

python -m evaluation.evaluate_rq1 \
    -o /cs/labs/josko/sahar_aharon/fcd_train \
    --cache_dir /cs/labs/josko/sahar_aharon/fcd_cache \
    --ckpt $RUN/checkpoints/best.pt \
    --split test \
    --out_dir $RUN/plots/eval
```

Prints and writes to `$RUN/plots/eval/`:
- `report_test.json` — overall + **per-anomaly-type** Dice/IoU, direction sens+/sens-,
  and **nuisance false-positive area** (change wrongly flagged on nuisance-only pairs; lower = better).
- `per_type_test.png` — Dice/IoU bar chart per anomaly type.
- `examples_test.png` — GT-vs-pred heatmap panel, one example per type.

---

## 5. Tips / troubleshooting

- **CUDA OOM** (11 GB nodes): lower `--batch_size`; keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  Cached training is light — OOM usually only bites during step 1 caching.
- **Resume caching / training**: caching skips done pairs; for training, re-launch and it
  starts fresh (load `last.pt` manually if you want to continue — not automated yet).
- **Offline RAD-DINO** (compute node has no internet): on a login node with internet run
  `python -c "from transformers import AutoModel,AutoImageProcessor; AutoModel.from_pretrained('microsoft/rad-dino'); AutoImageProcessor.from_pretrained('microsoft/rad-dino')"`
  to populate `~/.cache/huggingface`, then on the compute node
  `export HF_HUB_OFFLINE=1` before caching/training.
- **Sanity smoke test** before a long run: add `--limit 200` to step 1 and `--epochs 2`
  to step 2 to confirm the whole path works end-to-end.
```
