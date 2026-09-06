# Synthetic Pair Generation — Campus PC Runbook

Commands to generate the `foundation_contrastive_diff` synthetic DRR pairs on the
HUJI josko cluster, in parallel across several PCs. Copy‑paste per machine.

Paths below assume:
- Repo:   `/cs/labs/josko/sahar_aharon/repos`
- venv:   `/cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new`
- Output: `/cs/labs/josko/sahar_aharon/fcd_train`

Adjust if yours differ.

---

## 0. One‑time setup (run once per PC / login)

```bash
cd /cs/labs/josko/sahar_aharon/repos
git pull --ff-only

source /cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/bin/activate
export LD_LIBRARY_PATH=/cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduces CUDA OOM

# sanity: GPU present?
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, round(p.total_memory/1024**3,1),'GiB'); print('cuda:', torch.cuda.is_available())"
```

---

## 1. (Optional, once) Validate --reuse_change on a scratch dir

`--reuse_change` generates the pathology once per change and only varies devices +
angle across variants (faster, identical change). Validate before large use:

```bash
python foundation_contrastive_diff/data_generation/generate_training_set.py \
    -o /cs/labs/josko/sahar_aharon/fcd_reuse_test \
    --pairs_per_ct 1 --fixed_change_variants 3 --reuse_change \
    --num_slices 200 --slice_index 0
# once .../pair0/variant2/ exists, Ctrl-C

find /cs/labs/josko/sahar_aharon/fcd_reuse_test -name current_with_differences.png
```
Open `variant0/1/2` — expect: same change region, different angle + devices, and the
diff overlay shows only the pathology (devices cancel).

---

## 2. Launch the real run (one PC per slice)

Run the SAME block on each machine, changing only `I`. `N` = number of PCs.
All PCs write to the same `-o` (they work on disjoint CTs, so no collisions).

```bash
cd /cs/labs/josko/sahar_aharon/repos
source /cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/bin/activate
export LD_LIBRARY_PATH=/cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

I=0            # <-- PC 1=0, PC 2=1, PC 3=2, PC 4=3
N=4            # <-- total number of PCs

nohup python foundation_contrastive_diff/data_generation/generate_training_set.py \
    -o /cs/labs/josko/sahar_aharon/fcd_train \
    --pairs_per_ct 20 --fixed_change_variants 3 --reuse_change \
    --num_slices $N --slice_index $I \
    > /cs/labs/josko/sahar_aharon/fcd_train_pc$I.log 2>&1 &
echo $! > /cs/labs/josko/sahar_aharon/fcd_train_pc$I.pid
```

Sizing: total files ≈ `num_CTs × pairs_per_ct × fixed_change_variants`.
`pairs_per_ct 20`, K=3, ~400 CTs → ~24,000 pairs (~8,000 change‑groups).
The startup log prints `Total number of available CTs`.

---

## 3. Monitor

```bash
# live log (stream thanks to unbuffered -u)
tail -f /cs/labs/josko/sahar_aharon/fcd_train_pc0.log

# how many pairs done (all PCs, shared output)
find /cs/labs/josko/sahar_aharon/fcd_train -name current.png | wc -l

# is a PC's process alive / busy?
ps -o pid,etime,%cpu,%mem -p $(cat /cs/labs/josko/sahar_aharon/fcd_train_pc0.pid)
```

---

## 4. Stop / resume

```bash
# stop one PC
kill $(cat /cs/labs/josko/sahar_aharon/fcd_train_pc0.pid)
```
It's **resumable** — finished pairs are skipped, so just re‑run the same launch
block to continue after a stop, crash, or reclaimed node.

---

## 5. When done — build the dataset manifest (run once)

```bash
python foundation_contrastive_diff/data_generation/build_manifest.py \
    -o /cs/labs/josko/sahar_aharon/fcd_train --split-out

cat /cs/labs/josko/sahar_aharon/fcd_train/manifest_summary.json
```
Produces `manifest.jsonl` / `manifest.csv` / `manifest_summary.json` and
case‑disjoint `train/val/test` id lists. Loaders group contrastive positives by
`change_group_id`; use `effective_anomaly_type` / `realized_change` for labels.

---

## Troubleshooting

- **Log looks frozen but no error**: stdout was buffered — fixed via `-u`; judge by
  the `current.png` count instead.
- **`CUDA out of memory`**: keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  one generation process per GPU; use a bigger‑VRAM GPU if available (11 GB is tight).
- **`nvidia-smi: command not found`**: not on PATH here — use the torch snippet in §0.
- **Wrong Python** (e.g. itamar's venv): pass `--python /cs/usr/sahar_aharon/Desktop/sahar_aharon/venv_new/bin/python`.
