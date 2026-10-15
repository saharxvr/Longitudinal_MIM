# Backbone-fitness screen

Rank candidate frozen backbones **cheaply** (no full-head training) using the two diagnostics
that already correlate with full-head Dice:

- `evaluation/feature_change_correlation.py` — cos/L2 feature-movement vs GT change (AUROC)
- `evaluation/linear_probe.py` — best *linear* readout of ΔP (Dice)

Calibration point (RAD-DINO last-1): cos AUROC 0.62, probe 0.27, full head 0.449.
Rank backbones by probe Dice; **full-train only the winner** to confirm the ranking held.

> After each screen, append the numbers to `../EXPERIMENT_LOG.md` (standing rule).

## Pipeline per backbone
1. Cache a small subset of features (train + val).
2. Run correlation (val) + linear probe (train→val).
3. Record `(cos AUROC, probe Dice)`.

### Non-conflicting backbones (one generic script)
`rad_dino`, `dinov2_imagenet`, `torchxrayvision`:

```bash
# preamble (see ../AGENT_CONTEXT.md): git pull, venv, LD_LIBRARY_PATH, exports
export SCREEN=/cs/labs/josko/sahar_aharon/fcd_screen

for BB in dinov2_imagenet torchxrayvision; do
  python -m screening.cache_backbone_subset -o $DATA --backbone $BB \
      --cache_dir $SCREEN/$BB --n_train 2500 --n_val 1500
done
```
Then for each (grid: dinov2=37, xrv=7):
```bash
python -m evaluation.feature_change_correlation -o $DATA --cache_dir $SCREEN/$BB \
    --split val --feat_last_k 0 --grid <GRID>
python -m evaluation.linear_probe -o $DATA --cache_dir $SCREEN/$BB \
    --feat_last_k 0 --grid <GRID> --epochs 12
```

### Itamar reconstruction AE (standalone — avoids the constants/models import clash)
Run the cache step from `python_files/` (Itamar's env/deps), then probe in the fcd env:
```bash
cd /cs/labs/josko/sahar_aharon/repos/python_files   # or wherever python_files lives
python /cs/.../foundation_contrastive_diff/screening/cache_itamar_ae.py \
    --dataset_root $DATA --cache_dir $SCREEN/itamar_ae \
    --ckpt /cs/casmip/itamar_sab/LongitudinalCXRAnalysis/saved_models/Longitudinal_MIM/Checkpoint_id32_Epoch8_Longitudinal_MassesRotationInvariance_Dropout_ExtendedConvNet_1Channel_single128_Sched_Decoder6_Eff_ViT_L1L2_GN.pt \
    --n_train 2500 --n_val 1500
# back in the fcd env + repo:
python -m evaluation.feature_change_correlation -o $DATA --cache_dir $SCREEN/itamar_ae \
    --split val --feat_last_k 0 --grid 32
python -m evaluation.linear_probe -o $DATA --cache_dir $SCREEN/itamar_ae \
    --feat_last_k 0 --grid 32 --epochs 12
```

## Grids
| backbone | grid | dim | source (frozen, no fine-tune) |
|---|---|---|---|
| rad_dino | 37 | 768 | `microsoft/rad-dino` (HF) |
| dinov2_imagenet | 37 | 768 | torch.hub (control) |
| torchxrayvision | 7 | 1024 | `densenet121-res224-all` |
| clip_vitb16 | 14 | 768 | `openai/clip-vit-base-patch16` (control) |
| pubmedclip | 7 | 768 | `flaviagiammarino/pubmed-clip-vit-base-patch32` |
| biomedclip | 14 | 768 | `microsoft/BiomedCLIP…` (open_clip) |
| sam_vitb | 64 | 256 | `facebook/sam-vit-base` |
| medsam | 64 | 256 | `flaviagiammarino/medsam` (medical SAM) |
| itamar_ae | 32 | 640 | ORACLE only (in-domain, not a foundation model) |

Grid caveat: cos/L2 AUROC is only cleanly comparable at the SAME grid (coarse grids inflate it,
fine grids deflate it). Compare within similar grids; use the probe Dice (all upsample to 512)
as the grid-fairer final signal.

## Interpreting
- **Higher probe Dice than RAD-DINO's 0.27** → that backbone exposes our change signal better.
  The money result is **itamar_ae > rad_dino**: proves a reconstruction objective keeps the
  intensity deltas DINO discards. Then full-train the head on that backbone.
- Low grid (xrv=7) caps spatial Dice — read xrv's probe as a *feature-quality* signal, not a
  final-Dice prediction (coarse grid handicaps it).
