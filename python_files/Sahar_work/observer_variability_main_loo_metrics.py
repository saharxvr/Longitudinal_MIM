"""Leave-one-out recall, false-negative pairs, and consensus-based specificity
for the MAIN experiment (physicians Avi, Benny, Sigal, Smadar, Nitzan + Model).

Reuses the repo's established annotation IO (`get_disagreement_levels`) and the
metric helpers added to the OV pipeline:
  - base.evaluate_recall_fn_spec  (per-pair building blocks)
  - ov._new_loo_acc / _accumulate_loo / _loo_rates / plot_loo_summary

Metrics (all sign-split into positive / negative changes):
  - Leave-One-Out Recall: each physician scored against the consensus of the OTHER
    physicians. For the Model, BOTH conventions are reported:
      * full_panel (A): Model vs the consensus of ALL physicians.
      * swap_in  (B): Model swapped in for each held-out physician, scored against
        the remaining physicians, then averaged -> comparable to a human LOO score.
  - False-Negative Pairs: pairs where the reference consensus has >=1 change but the
    observer detected none.
  - Consensus-Based Specificity: among pairs where the reference consensus has no
    change (of that sign), the fraction where the observer also marked none.

Usage (from repo root):
    python python_files/Sahar_work/observer_variability_main_loo_metrics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_disagreement_levels as gdl  # noqa: E402  (annotation IO + consensus)
import observer_variability_nathalia_thaer_model as base  # noqa: E402  (metric helpers)
import observer_variability_pnimit_humans as ov  # noqa: E402  (safe_div)


def load_model_maps(model_path: Path, shape: tuple[int, int],
                    min_cc_size: int, min_cc_intensity: float) -> tuple[np.ndarray, np.ndarray]:
    """Load a model output into (pos, neg) binary maps aligned to the human-map shape."""
    from scipy.ndimage import label as _label

    out = nib.load(str(model_path)).get_fdata().astype(np.float32)
    if out.ndim == 3:
        out = out[..., out.shape[2] // 2]
    if out.shape != tuple(shape):
        from PIL import Image
        h, w = shape
        out = np.asarray(
            Image.fromarray(out, mode="F").resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    pos = (out > 0).astype(np.uint8)
    neg = (out < 0).astype(np.uint8)
    if min_cc_size > 0 or min_cc_intensity > 0:
        for bm in (pos, neg):
            ccs, n = _label(bm, gdl.STRUCT)
            for cc in range(1, n + 1):
                m = ccs == cc
                sz = int(np.sum(m))
                inten = float(np.abs(out[m]).mean()) if sz > 0 else 0.0
                if sz < min_cc_size or inten < min_cc_intensity:
                    bm[m] = 0
    return pos, neg


def load_model_maps_crop_info(model_path: Path, crop_info_path: Path, shape: tuple[int, int],
                              min_cc_size: int, min_cc_intensity: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-map a 512x512 square-crop model output back to full scan space via crop_info.json.

    Replicates Observer_Variability_sq_crop.load_model_labels_map: the crop coordinates live in
    transposed (torch) space, so we transpose the output, resize to the recorded square, place it
    on the (orig_h, orig_w) canvas, then transpose back to nifti space.
    """
    from skimage.transform import resize as sk_resize
    from scipy.ndimage import label as _label

    ci = json.loads(Path(crop_info_path).read_text(encoding="utf-8"))["current"]
    sq_x_min, sq_y_min = ci["sq_x_min"], ci["sq_y_min"]
    square_size, orig_h, orig_w = ci["square_size"], ci["orig_h"], ci["orig_w"]

    out = nib.load(str(model_path)).get_fdata()
    if out.ndim == 3:
        out = out[:, :, 0]
    out = out.T
    resized = sk_resize(out, (square_size, square_size), order=1, preserve_range=True, anti_aliasing=False)
    canvas = np.zeros((orig_h, orig_w), dtype=resized.dtype)
    canvas[sq_x_min:sq_x_min + square_size, sq_y_min:sq_y_min + square_size] = resized
    full = canvas.T
    if full.shape != tuple(shape):  # only if the scan isn't at the crop_info's orig resolution
        from PIL import Image
        h, w = shape
        full = np.asarray(Image.fromarray(full.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)

    pos = (full > 0).astype(np.uint8)
    neg = (full < 0).astype(np.uint8)
    if min_cc_size > 0 or min_cc_intensity > 0:
        for bm in (pos, neg):
            ccs, n = _label(bm, gdl.STRUCT)
            for cc in range(1, n + 1):
                m = ccs == cc
                sz = int(np.sum(m))
                inten = float(np.abs(full[m]).mean()) if sz > 0 else 0.0
                if sz < min_cc_size or inten < min_cc_intensity:
                    bm[m] = 0
    return pos, neg


def _resolve_pred(root: Path, i: int) -> tuple[Path | None, Path | None]:
    """Return (output.nii.gz, crop_info.json) for pair i, trying 'pair'/'Pair' prefixes."""
    for prefix in ("pair", "Pair"):
        d = root / f"{prefix}{i}"
        out = d / "output.nii.gz"
        if out.exists():
            ci = d / "crop_info.json"
            return out, (ci if ci.exists() else None)
    return None, None


def _new_level_acc(num_levels: int) -> dict:
    """Per-consensus-level accumulator for one observer (recall + FN-pairs per level; specificity at level 1)."""
    L = num_levels
    return {
        "rec_det_pos": [0] * L, "rec_tot_pos": [0] * L,
        "rec_det_neg": [0] * L, "rec_tot_neg": [0] * L,
        "fn_pairs_pos": [0] * L, "fn_denom_pos": [0] * L,
        "fn_pairs_neg": [0] * L, "fn_denom_neg": [0] * L,
        "tn_pos": 0, "cn_denom_pos": 0, "tn_neg": 0, "cn_denom_neg": 0,
    }


def _accumulate_levels(acc: dict, obs_pos, obs_neg, refs_pos, refs_neg) -> None:
    """Fold one pair into an observer accumulator, one entry per consensus level.

    A consensus level k means "at least k reference observers agreed on the finding".
    Recall and false-negative pairs are tracked per level; specificity uses level-1
    (the natural true-negative case: the reference panel agreed on NO change at all).
    """
    for sign, obs_map, ref_maps in (("pos", obs_pos, refs_pos), ("neg", obs_neg, refs_neg)):
        sens = base.get_sensitivity_at_consensus_levels(obs_map, ref_maps)  # [(det, tot)] per level
        for k, (det, tot) in enumerate(sens):
            acc[f"rec_det_{sign}"][k] += det
            acc[f"rec_tot_{sign}"][k] += tot
            if tot > 0:
                acc[f"fn_denom_{sign}"][k] += 1
                if det == 0:
                    acc[f"fn_pairs_{sign}"][k] += 1
        total_lvl1 = sens[0][1] if sens else 0
        if total_lvl1 == 0:                                  # consensus-negative pair
            acc[f"cn_denom_{sign}"] += 1
            _, obs_count = label(obs_map != 0, gdl.STRUCT)
            if obs_count == 0:
                acc[f"tn_{sign}"] += 1


def _level_rates(acc: dict) -> dict:
    """Derive per-level recall / FN-pair rates and level-1 specificity from an accumulator."""
    def _r(num, den):
        return [ov.safe_div(num[k], den[k]) for k in range(len(den))]
    return {
        "recall_by_level_pos": _r(acc["rec_det_pos"], acc["rec_tot_pos"]),
        "recall_by_level_neg": _r(acc["rec_det_neg"], acc["rec_tot_neg"]),
        "false_negative_pairs_by_level_pos": list(acc["fn_pairs_pos"]),
        "false_negative_pairs_by_level_neg": list(acc["fn_pairs_neg"]),
        "false_negative_pair_rate_by_level_pos": _r(acc["fn_pairs_pos"], acc["fn_denom_pos"]),
        "false_negative_pair_rate_by_level_neg": _r(acc["fn_pairs_neg"], acc["fn_denom_neg"]),
        "consensus_findings_by_level_pos": list(acc["rec_tot_pos"]),
        "consensus_findings_by_level_neg": list(acc["rec_tot_neg"]),
        "specificity_pos": ov.safe_div(acc["tn_pos"], acc["cn_denom_pos"]),
        "specificity_neg": ov.safe_div(acc["tn_neg"], acc["cn_denom_neg"]),
        "consensus_negative_pairs_pos": int(acc["cn_denom_pos"]),
        "consensus_negative_pairs_neg": int(acc["cn_denom_neg"]),
    }


def _avg_level_list(dicts: list[dict], key: str) -> list[float]:
    arrs = [d[key] for d in dicts]
    L = len(arrs[0])
    return [float(np.mean([a[k] for a in arrs])) for k in range(L)]


def plot_recall_by_level(loo_metrics: dict, physicians: list[str], model_name: str, out_path: Path) -> None:
    """Leave-one-out recall vs consensus level (one line per physician + both model conventions)."""
    import matplotlib.pyplot as plt

    per_obs = loo_metrics["per_observer"]
    swap = loo_metrics["model_swap_in"]["averaged"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    for ax, sign, title in ((axes[0], "pos", "Positive changes"), (axes[1], "neg", "Negative changes")):
        for phy in physicians:
            y = per_obs[phy][f"recall_by_level_{sign}"]
            ax.plot(range(1, len(y) + 1), y, marker="o", linewidth=1.6, alpha=0.85, label=phy)
        ys = swap[f"recall_by_level_{sign}"]
        ax.plot(range(1, len(ys) + 1), ys, marker="s", linewidth=2.8, color="black", label=f"{model_name} (swap-in)")
        yf = per_obs[model_name][f"recall_by_level_{sign}"]
        ax.plot(range(1, len(yf) + 1), yf, marker="^", linewidth=2.0, linestyle="--", color="dimgray", label=f"{model_name} (full panel)")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Consensus level (# agreeing reference observers)")
        ax.set_ylabel("Leave-one-out recall")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, edgecolor="black", loc="upper left", fontsize=9)
    fig.suptitle("Leave-One-Out Recall by Consensus Level", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    physicians: list[str] = list(args.physicians)
    model_name = args.model_name
    observers = physicians + [model_name]

    person_idx = {phy: gdl._build_pair_index(args.annotations_dir / phy) for phy in physicians}
    coverage = {phy: len(person_idx[phy]) for phy in physicians}

    loo_acc = {obs: _new_level_acc(len(physicians) - (0 if obs == model_name else 1)) for obs in observers}
    model_swapin_acc = {h: _new_level_acc(len(physicians) - 1) for h in physicians}

    pairs_processed = 0
    skipped: list[tuple[int, str]] = []

    for i in range(1, args.num_pairs + 1):
        pair_dir = gdl.find_pair_path(args.pairs_roots, i)
        if pair_dir is None:
            skipped.append((i, "no pair folder"))
            continue
        nii_files = sorted(
            p for p in pair_dir.iterdir()
            if p.name.endswith(".nii.gz") and not p.name.endswith("_lung_seg.nii.gz")
        )
        if len(nii_files) < 2:
            skipped.append((i, "missing scans"))
            continue
        model_path, crop_info_path = _resolve_pred(args.model_preds_root, i)
        if model_path is None:
            skipped.append((i, "no model output"))
            continue
        if args.model_space == "crop_info" and crop_info_path is None:
            skipped.append((i, "no crop_info.json"))
            continue

        shape = tuple(nib.load(str(nii_files[1])).get_fdata().shape)
        pos_maps: dict[str, np.ndarray] = {}
        neg_maps: dict[str, np.ndarray] = {}
        missing = None
        for phy in physicians:
            ann_p = person_idx[phy].get(i)
            if ann_p is None:
                missing = phy
                break
            p_map, n_map = gdl.load_labels_map(ann_p, shape)
            pos_maps[phy] = p_map
            neg_maps[phy] = n_map
        if missing is not None:
            skipped.append((i, f"missing annotation: {missing}"))
            continue

        try:
            if args.model_space == "crop_info":
                m_pos, m_neg = load_model_maps_crop_info(
                    model_path, crop_info_path, shape, args.min_cc_size, args.min_cc_intensity)
            else:
                m_pos, m_neg = load_model_maps(model_path, shape, args.min_cc_size, args.min_cc_intensity)
        except Exception as e:  # noqa: BLE001
            skipped.append((i, f"model load failed: {e}"))
            continue
        pos_maps[model_name] = m_pos
        neg_maps[model_name] = m_neg

        # Per-observer leave-one-out (humans: other physicians; Model: full panel = convention A).
        for obs in observers:
            refs = [p for p in physicians if p != obs]
            _accumulate_levels(
                loo_acc[obs],
                pos_maps[obs], neg_maps[obs],
                [pos_maps[o] for o in refs], [neg_maps[o] for o in refs],
            )

        # Model convention B: swap it in for each held-out physician, average later.
        for h in physicians:
            refs = [p for p in physicians if p != h]
            _accumulate_levels(
                model_swapin_acc[h],
                pos_maps[model_name], neg_maps[model_name],
                [pos_maps[o] for o in refs], [neg_maps[o] for o in refs],
            )

        pairs_processed += 1
        print(f"[pair {i:>3}] processed (total {pairs_processed})", flush=True)

    if pairs_processed == 0:
        raise RuntimeError("No pairs were processed. Check annotation coverage and paths.")

    loo_metrics = {
        "convention_notes": {
            "consensus_level": "Consensus level k = a change region that at least k reference observers agreed on. "
                               "recall_by_level[k-1] is the leave-one-out recall against those level>=k findings.",
            "humans": "Leave-one-out: each physician is scored against the consensus of all OTHER physicians (4 levels).",
            "model_full_panel": "Convention A: model scored against the consensus of ALL physicians (5 levels).",
            "model_swap_in": "Convention B: model swapped in for each held-out physician and scored "
                             "against the remaining physicians (4 levels); values averaged over the held-out physician. "
                             "Directly comparable to a physician's leave-one-out score.",
            "false_negative_pairs_by_level": "Per level k: pairs where the reference consensus has >=1 change at "
                                             "level>=k but the observer detected none of them.",
            "specificity": "Level-1 only: among pairs where the reference consensus has no change (of that sign), "
                           "fraction where the observer also marked none (true-negative rate).",
        },
        "pairs_processed": pairs_processed,
        "annotation_coverage": coverage,
        "per_observer": {obs: _level_rates(loo_acc[obs]) for obs in observers},
    }
    per_held_out = {h: _level_rates(model_swapin_acc[h]) for h in physicians}
    ph_vals = list(per_held_out.values())
    swapin_avg = {
        "recall_by_level_pos": _avg_level_list(ph_vals, "recall_by_level_pos"),
        "recall_by_level_neg": _avg_level_list(ph_vals, "recall_by_level_neg"),
        "false_negative_pair_rate_by_level_pos": _avg_level_list(ph_vals, "false_negative_pair_rate_by_level_pos"),
        "false_negative_pair_rate_by_level_neg": _avg_level_list(ph_vals, "false_negative_pair_rate_by_level_neg"),
        "specificity_pos": float(np.mean([per_held_out[h]["specificity_pos"] for h in physicians])),
        "specificity_neg": float(np.mean([per_held_out[h]["specificity_neg"] for h in physicians])),
    }
    loo_metrics["model_full_panel"] = loo_metrics["per_observer"][model_name]
    loo_metrics["model_swap_in"] = {"per_held_out_physician": per_held_out, "averaged": swapin_avg}
    loo_metrics["skipped"] = skipped

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leave_one_out_metrics.json").write_text(json.dumps(loo_metrics, indent=4), encoding="utf-8")
    plot_recall_by_level(loo_metrics, physicians, model_name, args.out_dir / "recall_by_consensus_level.png")

    # Console summary: leave-one-out recall by consensus level (columns c1..c4).
    print(f"\nPairs processed: {pairs_processed} / {args.num_pairs}")
    print("Annotation coverage (pairs indexed per physician): "
          + ", ".join(f"{p}={coverage[p]}" for p in physicians))

    def _print_recall_table(sign: str, title: str) -> None:
        ncols = 4
        hdr = f"{'observer':<18}" + "".join(f"{'c'+str(k):>7}" for k in range(1, ncols + 1))
        print(f"\nLOO recall by consensus level - {title}")
        print(hdr)
        print("-" * len(hdr))
        rows = [(obs, loo_metrics["per_observer"][obs][f"recall_by_level_{sign}"]) for obs in physicians]
        rows.append((f"{model_name} (full)", loo_metrics["per_observer"][model_name][f"recall_by_level_{sign}"]))
        rows.append((f"{model_name} (swap-in)", swapin_avg[f"recall_by_level_{sign}"]))
        for name, arr in rows:
            cells = "".join(f"{arr[k]:>7.3f}" if k < len(arr) else f"{'-':>7}" for k in range(ncols))
            print(f"{name:<18}{cells}")

    _print_recall_table("pos", "positive changes")
    _print_recall_table("neg", "negative changes")
    print(f"\nResults saved to: {args.out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} pairs (see run output / JSON 'skipped').")
    return 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    ann_tool = repo_root / "annotation tool"
    p = argparse.ArgumentParser(description="Main-experiment leave-one-out / FN-pairs / specificity metrics.")
    p.add_argument("--annotations-dir", type=Path, default=ann_tool / "Annotations")
    p.add_argument("--pairs-roots", nargs="+", type=Path,
                   default=[ann_tool / f"Pairs{i}" for i in range(1, 9)])
    p.add_argument("--model-preds-root", type=Path,
                   default=repo_root / "Sahar_work" / "files" / "predictions_1_100")
    p.add_argument("--physicians", nargs="+", default=["Avi", "Benny", "Sigal", "Smadar", "Nitzan"])
    p.add_argument("--model-name", default="Model")
    p.add_argument("--num-pairs", type=int, default=100)
    p.add_argument("--model-space", choices=["full", "crop_info"], default="full",
                   help="'full' resizes output.nii.gz to the scan; 'crop_info' inverse-maps a square-crop "
                        "output via crop_info.json (matches Observer_Variability_sq_crop).")
    p.add_argument("--min-cc-size", type=int, default=0, help="Model CC size filter (0 = off, matches main experiment).")
    p.add_argument("--min-cc-intensity", type=float, default=0.0, help="Model CC intensity filter (0 = off).")
    p.add_argument("--out-dir", type=Path,
                   default=repo_root / "Sahar_work" / "files" / "ov_results_main_loo")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
