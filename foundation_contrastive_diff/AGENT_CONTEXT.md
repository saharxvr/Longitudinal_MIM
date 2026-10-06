# Project context & working preferences (for the AI assistant)

Read this first. It captures the cluster layout, paths, environment quirks, timing facts,
conventions, and key findings so we don't re-derive them each session.

## Communication preferences
- Keep answers concise. Summarize each step. User is non-native English (typos common).
- Default to implementing; commit + push changes to git (`saharxvr/Longitudinal_MIM`, branch main).
- Save results with short descriptions under `foundation_contrastive_diff/results/` for backlook.

## Campus / cluster layout (HUJI josko lab)
- Nodes: `casNNN` (e.g. `cas703`). **128 GB RAM, ~24 GB GPU.** GPU userspace is at
  `/usr/local/APP/nvidia/<driver>/lib`.
- Storage: `/cs/labs/josko` is a **shared 21 TB Isilon NFS** (gets full; quota is lab-wide).
  Local node disk `/tmp` ≈ 89 GB. Home `/cs/usr/...` is small and often full.
- **RAM trick:** cached training is fast only if the cache fits in page cache (free RAM).
  - last-1 cache ≈ **90 GB → fits 128 GB RAM → ~1–3 min/epoch (fast).**
  - last-4 cache ≈ **360 GB → does NOT fit → NFS-bound → ~20 min/epoch (slow).**
  - A fat-RAM node (≥384 GB) would make last-4 fast too.

## Paths
| What | Path |
|---|---|
| repo | `/cs/labs/josko/sahar_aharon/repos` (run `-m` cmds from `.../repos/foundation_contrastive_diff`) |
| dataset (`$DATA`) | `/cs/labs/josko/sahar_aharon/fcd_train` (has `manifest_split.jsonl`) |
| last-4 cache (`$CACHE4`) | `/cs/labs/josko/sahar_aharon/fcd_cache_last4` (~360 GB) |
| last-1 cache (`$CACHE1`) | `/cs/labs/josko/sahar_aharon/fcd_cache_last1` (~90 GB, fast) |
| runs | `/cs/labs/josko/sahar_aharon/fcd_runs/<run_name>/` |
| **our venv** | `/cs/labs/josko/sahar_aharon/fcd_venv` (py3.13, torch 2.5.1+cu121) |

## Per-session environment setup (post OS-update, Oct 2026)
The OS update bumped system Python 3.11→3.13 (broke itamar's `venv_new`) and left the
NVIDIA userspace off the default path. Every new shell:
```bash
source /cs/labs/josko/sahar_aharon/fcd_venv/bin/activate
export LD_LIBRARY_PATH=/usr/local/APP/nvidia/$(awk '{print $8}' /proc/driver/nvidia/version)/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA=/cs/labs/josko/sahar_aharon/fcd_train
export CACHE4=/cs/labs/josko/sahar_aharon/fcd_cache_last4
```
Verify: `python -c "import torch; print(torch.cuda.is_available())"` → must be True.

## Shell / ops quirks
- zsh **noclobber**: `>` on an existing file fails ("file exists"). Use `>|` or `rm -f` first.
- Nodes can **reboot** (OS maintenance) → always launch with `--resume` available and `nohup`.
- Big deletes of many small files are slow over NFS → background them (`nohup rm -rf ... &`).
- `conda` is not reliably on PATH; prefer the `fcd_venv`.

## Running convention
- `nohup python -u -m training.train_foundation_diff --config configs/<run>.yaml >| $RUN/train.log 2>&1 &`
- Always `-u` (unbuffered), `>|` (zsh), `--grad_clip 1.0` (last-4 NaN'd without it), `--resume` to survive reboots.
- Checkpoints: `last.pt` (every epoch, full state), `best.pt` (best val Dice). `metrics.csv` + `train_curves.png` update each epoch.

## Key findings so far (RQ1: signed longitudinal change map, frozen RAD-DINO + D-GLoRI head)
- **Loss matters most:** L1+Dice overfit-16 → 0.71; **L1+L2 (Itamar-style) → 0.94.** Dice
  *hurts* because the synthetic GT is **faint (mag_median ~0.08)** — Dice pushes magnitude
  up while L1 matches the low GT; they conflict. Use **L1+L2, no Dice/sign** going forward.
- **Features:** last-4 > last-1 (Dice loss: 0.46 vs 0.43). (L1+L2 ablation pending.)
- **Data is learnable:** resolution-ceiling Dice ~0.79 (effusion/pneumothorax ~0.85);
  overfit 0.94 → capacity is NOT the limit. 0.46 was mostly a loss-design problem.
- **Metric hygiene:** report CHANGE-ONLY Dice (is_pathology==1); route no-change pairs to a
  separate nuisance false-positive-area metric (empty-vs-empty scores Dice 1.0 and inflates).
- **Prediction head now:** `DifferenceHead` → single 37×37 grid → `UPerNetDecoder` (14× upsample,
  no fine-res skips). Candidate next step: DPT/ViT-Adapter multi-scale head (reuses last-4 cache).

## Results log
See `results/README.md` (run1 last-1/Dice 0.43, run4 last-1/Dice 0.43, run5 last-4/Dice 0.46).
