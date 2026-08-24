"""Observer Variability for PNIMIT among human annotators only (no model).

Reuses the exact metric helpers from `observer_variability_nathalia_thaer_model`
(PAI per-label / per-pair with one-sided boost, HMDR/UDPP, sensitivity at
consensus levels) but generalizes to an arbitrary list of human observers.

Default observers: Benny, Nathalia, Thaer.

Pair matching: each annotator's JSON filename contains a pair token like
`A1_1_2` (regardless of any name prefix / case). Pairs used are those that ALL
selected observers annotated AND that have a scan folder under --pairs-root.

Usage (from repo root):
    python python_files/Sahar_work/observer_variability_pnimit_humans.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402

# Import the exact metric + IO helpers from the sibling module so definitions
# (incl. the one-sided per-pair "all" boost) stay identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import observer_variability_nathalia_thaer_model as base  # noqa: E402

PAIR_TOKEN_RE = re.compile(r"A\d+_\d+_\d+", re.IGNORECASE)


def _overlay_diff(ax, current: np.ndarray, diff_map: np.ndarray, title: str) -> None:
    ax.imshow(current, cmap="gray")
    if np.any(diff_map):
        ax.imshow(
            diff_map,
            alpha=base.generate_alpha_map(diff_map),
            cmap=base.DIFF_CMAP,
            norm=colors.TwoSlopeNorm(
                vmin=min(float(np.min(diff_map)), -0.01),
                vcenter=0.0,
                vmax=max(float(np.max(diff_map)), 0.01),
            ),
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_axis_off()


def make_pair_collage(
    current: np.ndarray,
    annot_maps: dict[str, np.ndarray],
    model_full: np.ndarray | None,
    panel_order: list[str],
    pair_token: str,
    out_dir: Path,
) -> None:
    """One row: Original | <each observer overlay>. Green=increase/appearance, red=decrease/disappearance."""
    n_panels = 1 + len(panel_order)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    _overlay_diff(axes[0], current, np.zeros_like(current), "Original")
    for ax, name in zip(axes[1:], panel_order):
        _overlay_diff(ax, current, annot_maps[name], name)
    fig.suptitle(f"pair_{pair_token}", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"pair_{pair_token}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)



def extract_pair_token(name: str) -> str | None:
    m = PAIR_TOKEN_RE.search(name)
    return m.group(0) if m else None


def discover_observer_pairs(ann_root: Path, observer: str) -> dict[str, Path]:
    obs_dir = ann_root / observer
    out: dict[str, Path] = {}
    if not obs_dir.is_dir():
        return out
    for p in sorted(obs_dir.glob("*.json")):
        tok = extract_pair_token(p.stem)
        if tok is not None:
            out[tok] = p
    return out


def resolve_pair_dir(pairs_root: Path, token: str) -> Path | None:
    for prefix in ("pair_", "Pair_", "pair", "Pair"):
        d = pairs_root / f"{prefix}{token}"
        if d.is_dir():
            return d
    return None


def resolve_model_output(preds_root: Path, token: str) -> Path | None:
    for prefix in ("pair_", "Pair_", "pair", "Pair"):
        p = preds_root / f"{prefix}{token}" / "output.nii.gz"
        if p.exists():
            return p
    return None


# ── Square-crop reconstruction & inverse mapping (matches Prediction.py) ──────

def _square_from_bbox(x0: int, y0: int, x1: int, y1: int, H: int, W: int) -> tuple[int, int, int]:
    """Replicate Prediction.py: expand bbox to a centered square, clamped to image."""
    side = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    sx = cx - side // 2
    sy = cy - side // 2
    ex = sx + side
    ey = sy + side
    if sx < 0:
        ex -= sx; sx = 0
    if sy < 0:
        ey -= sy; sy = 0
    if ex > H:
        sx -= (ex - H); ex = H
    if ey > W:
        sy -= (ey - W); ey = W
    sx = max(sx, 0)
    sy = max(sy, 0)
    return sx, sy, side


def crop_info_noseg(H: int, W: int) -> tuple[int, int, int]:
    """Whole-image centered square (Prediction.preprocess_no_seg): side=max(H,W)."""
    return _square_from_bbox(0, 0, H, W, H, W)


def crop_info_from_seg(seg_T: np.ndarray, crop_pad_val: int = 15, margin: int = 0) -> tuple[int, int, int] | None:
    """Square around segmentation bbox + pad (Prediction.preprocess)."""
    coords = np.argwhere(seg_T > 0)
    if coords.size == 0:
        return None
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    H, W = seg_T.shape
    pad = crop_pad_val + margin
    x0 = max(int(x0) - pad, 0)
    y0 = max(int(y0) - pad, 0)
    x1 = min(int(x1) + pad, H - 1)
    y1 = min(int(y1) + pad, W - 1)
    return _square_from_bbox(x0, y0, x1, y1, H, W)


def _resize_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    h, w = out_hw
    if arr.shape == (h, w):
        return arr.astype(np.float32)
    pil = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(pil.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)


def inverse_map_square(output_sq_T: np.ndarray, crop: tuple[int, int, int], H: int, W: int) -> np.ndarray:
    """Place a square (512x512) model output back into full (H, W) scan space."""
    sx, sy, side = crop
    full = np.zeros((H, W), dtype=np.float32)
    resized = _resize_2d(output_sq_T, (side, side))
    ex = min(sx + side, H)
    ey = min(sy + side, W)
    full[sx:ex, sy:ey] = resized[: ex - sx, : ey - sy]
    return full


def load_current_and_lungseg(pair_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return current scan (.T) and its lung-seg (.T) if present."""
    import nibabel as nib
    scans = sorted([p for p in pair_dir.glob("*.nii.gz") if "_seg" not in p.name])
    current = base.load_current_scan(pair_dir)
    seg = None
    if len(scans) >= 2:
        stem = scans[1].name[:-7] if scans[1].name.endswith(".nii.gz") else scans[1].stem
        seg_path = pair_dir / f"{stem}_lung_seg.nii.gz"
        if seg_path.exists():
            seg = nib.load(str(seg_path)).get_fdata().T.astype(np.float32)
    return current, seg


def load_model_map_spaced(
    model_path: Path, pair_dir: Path, shape_hw: tuple[int, int],
    model_space: str, margin: int, min_cc_size: int, min_cc_intensity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model output into full scan space, honoring the square-crop pipeline.

    model_space: 'full' (already full-res), 'square_noseg' (whole-image square),
                 'square_seg' (square around lung seg bbox + margin).
    Returns (pos_map, neg_map, full_signed) all at shape_hw.
    """
    import nibabel as nib
    from scipy.ndimage import label as _label
    out = nib.load(str(model_path)).get_fdata().T.astype(np.float32)
    if out.ndim == 3:
        out = out[:, :, out.shape[2] // 2]
    H, W = shape_hw

    if model_space == "full":
        full = base.resize_to_shape(out, shape_hw)
    elif model_space == "square_noseg":
        crop = crop_info_noseg(H, W)
        full = inverse_map_square(out, crop, H, W)
    elif model_space == "square_seg":
        _, seg = load_current_and_lungseg(pair_dir)
        if seg is None:
            full = base.resize_to_shape(out, shape_hw)
        else:
            crop = crop_info_from_seg(seg, crop_pad_val=15, margin=margin)
            full = inverse_map_square(out, crop, H, W) if crop else base.resize_to_shape(out, shape_hw)
    else:
        raise ValueError(f"Unknown model_space: {model_space}")

    pos = (full > 0).astype(np.uint8)
    neg = (full < 0).astype(np.uint8)
    for bm in (pos, neg):
        ccs, n = _label(bm, base.STRUCT_3x3)
        for cc in range(1, n + 1):
            m = ccs == cc
            sz = int(np.sum(m))
            inten = float(np.abs(full[m]).mean()) if sz > 0 else 0.0
            if sz < min_cc_size or inten < min_cc_intensity:
                bm[m] = 0
    return pos, neg, full


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _new_loo_acc() -> dict:
    """Accumulator for leave-one-out recall / false-negative-pairs / specificity (one observer)."""
    return {
        "rec_det_pos": 0, "rec_tot_pos": 0, "rec_det_neg": 0, "rec_tot_neg": 0,
        "fn_pairs_pos": 0, "fn_denom_pos": 0, "fn_pairs_neg": 0, "fn_denom_neg": 0,
        "tn_pos": 0, "cn_denom_pos": 0, "tn_neg": 0, "cn_denom_neg": 0,
    }


def _accumulate_loo(acc: dict, rp: dict, rn: dict) -> None:
    """Fold one pair's positive/negative building blocks into an observer accumulator."""
    for sign, r in (("pos", rp), ("neg", rn)):
        acc[f"rec_det_{sign}"] += r["detected"]
        acc[f"rec_tot_{sign}"] += r["ref_total"]
        if r["ref_total"] > 0:                       # reference agreed a change exists
            acc[f"fn_denom_{sign}"] += 1
            if r["detected"] == 0:
                acc[f"fn_pairs_{sign}"] += 1          # observer missed the whole pair
        else:                                        # consensus-negative pair
            acc[f"cn_denom_{sign}"] += 1
            if r["obs_count"] == 0:
                acc[f"tn_{sign}"] += 1                # observer correctly abstained


def _loo_rates(acc: dict) -> dict:
    """Derive interpretable rates from an accumulator."""
    return {
        "recall_pos": safe_div(acc["rec_det_pos"], acc["rec_tot_pos"]),
        "recall_neg": safe_div(acc["rec_det_neg"], acc["rec_tot_neg"]),
        "false_negative_pairs_pos": int(acc["fn_pairs_pos"]),
        "false_negative_pairs_neg": int(acc["fn_pairs_neg"]),
        "false_negative_pair_rate_pos": safe_div(acc["fn_pairs_pos"], acc["fn_denom_pos"]),
        "false_negative_pair_rate_neg": safe_div(acc["fn_pairs_neg"], acc["fn_denom_neg"]),
        "specificity_pos": safe_div(acc["tn_pos"], acc["cn_denom_pos"]),
        "specificity_neg": safe_div(acc["tn_neg"], acc["cn_denom_neg"]),
        "consensus_negative_pairs_pos": int(acc["cn_denom_pos"]),
        "consensus_negative_pairs_neg": int(acc["cn_denom_neg"]),
    }


def plot_loo_summary(loo_metrics: dict, observers: list[str], model_name: str,
                     include_model: bool, out_path: Path) -> None:
    """Grouped bars: leave-one-out recall and consensus-based specificity per observer."""
    labels = list(observers)
    per_obs = loo_metrics["per_observer"]
    rec_pos = [per_obs[o]["recall_pos"] for o in observers]
    rec_neg = [per_obs[o]["recall_neg"] for o in observers]
    spec_pos = [per_obs[o]["specificity_pos"] for o in observers]
    spec_neg = [per_obs[o]["specificity_neg"] for o in observers]
    if include_model:
        sw = loo_metrics["model_swap_in"]["averaged"]
        labels = labels + [f"{model_name}\n(swap-in)"]
        rec_pos.append(sw["recall_pos"]); rec_neg.append(sw["recall_neg"])
        spec_pos.append(sw["specificity_pos"]); spec_neg.append(sw["specificity_neg"])

    x = np.arange(len(labels))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(6 + 1.6 * len(labels), 5), dpi=180)
    for ax, pos_vals, neg_vals, title in (
        (axes[0], rec_pos, rec_neg, "Leave-One-Out Recall"),
        (axes[1], spec_pos, spec_neg, "Consensus-Based Specificity"),
    ):
        ax.bar(x - w / 2, pos_vals, w, label="Positive changes", color="#003366")
        ax.bar(x + w / 2, neg_vals, w, label="Negative changes", color="#800000")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(title)
        ax.set_title(title, pad=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.legend(frameon=True, edgecolor="black", loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    humans: list[str] = list(args.observers)
    model_name = args.model_name
    include_model = not args.no_model
    # Model (if any) is appended LAST so the matrix separator splits humans | model.
    observers: list[str] = humans + ([model_name] if include_model else [])
    idx = {n: i for i, n in enumerate(observers)}
    num_obs = len(observers)
    if len(humans) < 2:
        raise ValueError("Need at least 2 human observers.")

    def ref_names(obs: str) -> list[str]:
        """Reference pool for HMDR/UDPP/sensitivity: other HUMANS only (never model)."""
        return [h for h in humans if h != obs]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    make_collages: bool = args.make_collages
    if args.collage_dir is None:
        args.collage_dir = out_dir / "pair_collages"

    # Discover each human observer's pairs, then intersect.
    obs_pairs = {obs: discover_observer_pairs(args.annotations_json_root, obs) for obs in humans}
    for obs in humans:
        if not obs_pairs[obs]:
            raise FileNotFoundError(
                f"No JSON annotations found for observer '{obs}' under {args.annotations_json_root / obs}"
            )
    common_tokens = set.intersection(*[set(obs_pairs[obs]) for obs in humans])
    all_tokens = sorted(common_tokens, key=lambda t: [int(x) for x in re.findall(r"\d+", t)])
    if args.max_pairs > 0:
        all_tokens = all_tokens[: args.max_pairs]

    # Accumulators.
    pairwise_agreement_mat_pos = np.eye(num_obs, dtype=np.float64)
    pairwise_agreement_mat_neg = np.eye(num_obs, dtype=np.float64)
    pairwise_disagreement_mat_pos = np.zeros((num_obs, num_obs), dtype=np.float64)
    pairwise_disagreement_mat_neg = np.zeros((num_obs, num_obs), dtype=np.float64)
    pairwise_agreement_per_pair_mat_pos = np.eye(num_obs, dtype=np.float64)
    pairwise_agreement_per_pair_mat_neg = np.eye(num_obs, dtype=np.float64)

    agreements_num_list_pos = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    agreements_num_list_neg = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    disagreements_num_list_pos = [[[] for _ in range(num_obs)] for __ in range(num_obs)]
    disagreements_num_list_neg = [[[] for _ in range(num_obs)] for __ in range(num_obs)]

    total_labels_pos = {obs: [] for obs in observers}
    total_labels_neg = {obs: [] for obs in observers}

    total_preds_pos = {obs: 0 for obs in observers}
    total_preds_neg = {obs: 0 for obs in observers}
    total_overlapping_pos = {obs: 0 for obs in observers}
    total_overlapping_neg = {obs: 0 for obs in observers}
    not_overlapping_pos = {obs: [] for obs in observers}
    not_overlapping_neg = {obs: [] for obs in observers}

    # Sensitivity: each observer vs its reference humans -> that many consensus levels.
    sens_pos = {obs: [[0, 0] for _ in range(max(1, len(ref_names(obs))))] for obs in observers}
    sens_neg = {obs: [[0, 0] for _ in range(max(1, len(ref_names(obs))))] for obs in observers}

    # Leave-one-out recall / false-negative-pairs / consensus-based specificity.
    # Humans: reference = other humans (true leave-one-out). Model: reference = all humans
    # (Convention A, full panel). Convention B for the model is the swap-in loop below.
    loo_acc = {obs: _new_loo_acc() for obs in observers}
    model_swapin_acc = {h: _new_loo_acc() for h in humans} if include_model else {}

    pairs_processed = 0
    skipped = []

    for tok in all_tokens:
        pair_dir = resolve_pair_dir(args.pairs_root, tok)
        if pair_dir is None:
            skipped.append((tok, "no pair folder"))
            continue
        model_path = None
        if include_model:
            model_path = resolve_model_output(args.model_preds_root, tok)
            if model_path is None:
                skipped.append((tok, "no model output"))
                continue
        try:
            current = base.load_current_scan(pair_dir)
        except Exception as e:  # noqa: BLE001
            skipped.append((tok, f"scan load failed: {e}"))
            continue
        shape_hw = tuple(current.shape)

        pos_maps = {}
        neg_maps = {}
        for obs in humans:
            p, n = base.load_labels_map(obs_pairs[obs].get(tok, Path("")), shape_hw)
            pos_maps[obs] = p
            neg_maps[obs] = n
        model_full = None
        if include_model:
            try:
                m_pos, m_neg, model_full = load_model_map_spaced(
                    model_path, pair_dir, shape_hw,
                    model_space=args.model_space, margin=args.model_margin,
                    min_cc_size=args.min_cc_size, min_cc_intensity=args.min_cc_intensity,
                )
            except Exception as e:  # noqa: BLE001
                skipped.append((tok, f"model load failed: {e}"))
                continue
            pos_maps[model_name] = m_pos
            neg_maps[model_name] = m_neg

        if make_collages:
            annot_maps = {
                obs: pos_maps[obs].astype(np.float32) + neg_maps[obs].astype(np.float32)
                for obs in humans
            }
            if include_model:
                annot_maps[model_name] = (
                    model_full if model_full is not None
                    else pos_maps[model_name].astype(np.float32) - neg_maps[model_name].astype(np.float32)
                )
            make_pair_collage(current, annot_maps, model_full, observers, tok, args.collage_dir)


        # PAI counts (per-label accumulators + per-pair value lists).
        for maps, ag_mat, dis_mat, ag_pp_mat, ag_lists, dis_lists, totals in [
            (pos_maps, pairwise_agreement_mat_pos, pairwise_disagreement_mat_pos,
             pairwise_agreement_per_pair_mat_pos, agreements_num_list_pos, disagreements_num_list_pos, total_labels_pos),
            (neg_maps, pairwise_agreement_mat_neg, pairwise_disagreement_mat_neg,
             pairwise_agreement_per_pair_mat_neg, agreements_num_list_neg, disagreements_num_list_neg, total_labels_neg),
        ]:
            for n1, n2 in combinations(observers, 2):
                a, d = base.get_pairwise_detections(maps[n1], maps[n2])
                i1, i2 = idx[n1], idx[n2]
                ag_mat[i1, i2] += 2 * a
                ag_mat[i2, i1] += 2 * a
                dis_mat[i1, i2] += d
                dis_mat[i2, i1] += d
                c_pai = np.nan_to_num(2 * a / np.array(2 * a + d), nan=1.0, posinf=1.0, neginf=1.0)
                ag_pp_mat[i1, i2] += c_pai
                ag_pp_mat[i2, i1] += c_pai
                ag_lists[i1][i2].append(2 * a)
                ag_lists[i2][i1].append(2 * a)
                dis_lists[i1][i2].append(d)
                dis_lists[i2][i1].append(d)
            for obs in observers:
                _, ccs_num = base.label(maps[obs] != 0, base.STRUCT_3x3)
                totals[obs].append(int(ccs_num))

        # HMDR / UDPP + sensitivity: each observer vs its reference humans (never the model).
        for obs in observers:
            others = ref_names(obs)
            if not others:
                continue
            ov, nov, tp = base.get_hmdr_udpp_counts(pos_maps[obs], [pos_maps[o] for o in others])
            total_preds_pos[obs] += tp
            total_overlapping_pos[obs] += ov
            not_overlapping_pos[obs].append(nov)
            ov, nov, tp = base.get_hmdr_udpp_counts(neg_maps[obs], [neg_maps[o] for o in others])
            total_preds_neg[obs] += tp
            total_overlapping_neg[obs] += ov
            not_overlapping_neg[obs].append(nov)

            sp = base.get_sensitivity_at_consensus_levels(pos_maps[obs], [pos_maps[o] for o in others])
            sn = base.get_sensitivity_at_consensus_levels(neg_maps[obs], [neg_maps[o] for o in others])
            for k, (dd, cc) in enumerate(sp):
                sens_pos[obs][k][0] += dd
                sens_pos[obs][k][1] += cc
            for k, (dd, cc) in enumerate(sn):
                sens_neg[obs][k][0] += dd
                sens_neg[obs][k][1] += cc

            # Leave-one-out recall / FN-pairs / specificity vs the same reference pool.
            rp = base.evaluate_recall_fn_spec(pos_maps[obs], [pos_maps[o] for o in others])
            rn = base.evaluate_recall_fn_spec(neg_maps[obs], [neg_maps[o] for o in others])
            _accumulate_loo(loo_acc[obs], rp, rn)

        # Convention B for the model: swap it in for each held-out human, score against
        # the remaining humans. Averaging over the held-out human makes the model number
        # directly comparable to a human's leave-one-out score (both use N-1 references).
        if include_model:
            for h in humans:
                refs = [o for o in humans if o != h]
                rp = base.evaluate_recall_fn_spec(pos_maps[model_name], [pos_maps[o] for o in refs])
                rn = base.evaluate_recall_fn_spec(neg_maps[model_name], [neg_maps[o] for o in refs])
                _accumulate_loo(model_swapin_acc[h], rp, rn)

        pairs_processed += 1

    if pairs_processed == 0:
        raise RuntimeError("No pairs were processed. Check paths and file availability.")

    pairwise_agreement_per_pair_mat_pos /= pairs_processed
    pairwise_agreement_per_pair_mat_neg /= pairs_processed

    # Sensitivity outputs.
    sens_dict = {}
    for obs in observers:
        for i, s in enumerate(sens_pos[obs]):
            sens_dict[f"{obs} Sensitivity Consensus Level {i + 1} (Positive)"] = safe_div(s[0], s[1])
        for i, s in enumerate(sens_neg[obs]):
            sens_dict[f"{obs} Sensitivity Consensus Level {i + 1} (Negative)"] = safe_div(s[0], s[1])
        for i, s in enumerate(sens_pos[obs]):
            sens_dict[f"{obs} Total detections & changes at Consensus Level {i + 1} (Positive)"] = (int(s[0]), int(s[1]))
        for i, s in enumerate(sens_neg[obs]):
            sens_dict[f"{obs} Total detections & changes at Consensus Level {i + 1} (Negative)"] = (int(s[0]), int(s[1]))
    (out_dir / "sensitivity_measures.json").write_text(json.dumps(sens_dict, indent=4), encoding="utf-8")
    for obs in observers:
        base.plot_curves(
            [safe_div(s[0], s[1]) for s in sens_pos[obs]],
            [safe_div(s[0], s[1]) for s in sens_neg[obs]],
            obs,
            out_dir / f"sensitivity_consensus_levels_{obs.lower()}.png",
        )

    # Leave-one-out recall / false-negative-pairs / consensus-based specificity.
    loo_metrics = {
        "convention_notes": {
            "humans": "Leave-one-out: each human is scored against the consensus of all OTHER humans.",
            "model_full_panel": "Convention A: model scored against the consensus of ALL humans.",
            "model_swap_in": "Convention B: model swapped in for each held-out human and scored "
                             "against the remaining humans; rates averaged over the held-out human. "
                             "Directly comparable to a human's leave-one-out score (both use N-1 references).",
            "false_negative_pairs": "Pairs where the reference consensus has >=1 change but the observer detected none.",
            "specificity": "Among pairs where the reference consensus has no change (of that sign), "
                           "fraction where the observer also marked none (true-negative rate).",
        },
        "per_observer": {obs: _loo_rates(loo_acc[obs]) for obs in observers},
    }
    if include_model:
        per_held_out = {h: _loo_rates(model_swapin_acc[h]) for h in humans}
        avg_keys = [
            "recall_pos", "recall_neg",
            "false_negative_pair_rate_pos", "false_negative_pair_rate_neg",
            "specificity_pos", "specificity_neg",
        ]
        swapin_avg = {k: float(np.mean([per_held_out[h][k] for h in humans])) for k in avg_keys}
        loo_metrics["model_full_panel"] = loo_metrics["per_observer"][model_name]
        loo_metrics["model_swap_in"] = {"per_held_out_human": per_held_out, "averaged": swapin_avg}
    (out_dir / "leave_one_out_metrics.json").write_text(json.dumps(loo_metrics, indent=4), encoding="utf-8")
    plot_loo_summary(loo_metrics, observers, model_name, include_model, out_dir / "leave_one_out_summary.png")

    # HMDR / UDPP outputs.
    precision = {}
    for obs in observers:
        precision[f"{obs} HMDR (Positive)"] = safe_div(total_overlapping_pos[obs], total_preds_pos[obs])
        precision[f"{obs} HMDR (Negative)"] = safe_div(total_overlapping_neg[obs], total_preds_neg[obs])
        precision[f"UDPP {obs} (Positive)"] = safe_div(sum(not_overlapping_pos[obs]), pairs_processed)
        precision[f"UDPP {obs} (Negative)"] = safe_div(sum(not_overlapping_neg[obs]), pairs_processed)
        precision[f"UDPP STD {obs} (Positive)"] = float(np.std(not_overlapping_pos[obs])) if not_overlapping_pos[obs] else 0.0
        precision[f"UDPP STD {obs} (Negative)"] = float(np.std(not_overlapping_neg[obs])) if not_overlapping_neg[obs] else 0.0
    (out_dir / "precision_measures.json").write_text(json.dumps(precision, indent=4), encoding="utf-8")

    # PAI matrices.
    mat_per_label_pos = pairwise_agreement_mat_pos / (pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos)
    mat_per_label_neg = pairwise_agreement_mat_neg / (pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg)
    mat_per_label_all = (pairwise_agreement_mat_pos + pairwise_agreement_mat_neg) / (
        pairwise_agreement_mat_pos + pairwise_disagreement_mat_pos
        + pairwise_agreement_mat_neg + pairwise_disagreement_mat_neg
    )

    mat_per_pair_pos = pairwise_agreement_per_pair_mat_pos
    mat_per_pair_neg = pairwise_agreement_per_pair_mat_neg
    mat_per_pair_all = np.eye(num_obs, dtype=np.float64)
    for n1, n2 in combinations(observers, 2):
        i1, i2 = idx[n1], idx[n2]
        pai_per_pair = 0.0
        for k in range(len(agreements_num_list_pos[i1][i2])):
            ag_pos = agreements_num_list_pos[i1][i2][k]
            ag_neg = agreements_num_list_neg[i1][i2][k]
            ag_all = ag_pos + ag_neg
            dis_pos = disagreements_num_list_pos[i1][i2][k]
            dis_neg = disagreements_num_list_neg[i1][i2][k]
            dis_all = dis_pos + dis_neg
            c_pai = np.nan_to_num(ag_all / np.array(ag_all + dis_all), nan=1.0, posinf=1.0, neginf=1.0)
            if (ag_pos + dis_pos == 0 and ag_neg + dis_neg > 0) or (ag_pos + dis_pos > 0 and ag_neg + dis_neg == 0):
                c_pai = c_pai * 0.5 + 0.5
            pai_per_pair += float(c_pai)
        mat_per_pair_all[i1, i2] = pai_per_pair / pairs_processed
        mat_per_pair_all[i2, i1] = pai_per_pair / pairs_processed

    for mat, fname, title in [
        (mat_per_label_pos, "per_label_agreement_pos.png", "Pairwise Agreement Index Per Detection (positive)"),
        (mat_per_label_neg, "per_label_agreement_neg.png", "Pairwise Agreement Index Per Detection (negative)"),
        (mat_per_label_all, "per_label_agreement_all.png", "Pairwise Agreement Index Per Detection (all)"),
        (mat_per_pair_pos, "per_pair_agreement_pos.png", "Pairwise Agreement Index Per Pair (positive)"),
        (mat_per_pair_neg, "per_pair_agreement_neg.png", "Pairwise Agreement Index Per Pair (negative)"),
        (mat_per_pair_all, "per_pair_agreement_all.png", "Pairwise Agreement Index Per Pair (all)"),
    ]:
        df = pd.DataFrame(mat, index=observers, columns=observers)
        base.plot_matrix(df, out_dir / fname, title)

    # Total label statistics.
    def stats_dict(arr_list: list[int]):
        arr = np.array(arr_list, dtype=np.int32)
        return (
            int(np.sum(arr)),
            float(np.mean(arr)) if arr.size else 0.0,
            float(np.std(arr)) if arr.size else 0.0,
            int(np.max(arr)) if arr.size else 0,
            int(np.min(arr)) if arr.size else 0,
        )

    total_labels_data_pos = {obs: stats_dict(total_labels_pos[obs]) for obs in observers}
    total_labels_data_neg = {obs: stats_dict(total_labels_neg[obs]) for obs in observers}
    total_labels_data_all = {
        obs: stats_dict([total_labels_pos[obs][k] + total_labels_neg[obs][k] for k in range(len(total_labels_pos[obs]))])
        for obs in observers
    }
    (out_dir / "total_labels_marked_pos.json").write_text(json.dumps(total_labels_data_pos, indent=4), encoding="utf-8")
    (out_dir / "total_labels_marked_neg.json").write_text(json.dumps(total_labels_data_neg, indent=4), encoding="utf-8")
    (out_dir / "total_labels_marked_all.json").write_text(json.dumps(total_labels_data_all, indent=4), encoding="utf-8")

    summary = {
        "pairs_processed": pairs_processed,
        "observers": observers,
        "humans": humans,
        "model_included": include_model,
        "model_preds_root": str(args.model_preds_root) if include_model else None,
        "annotations_root": str(args.annotations_json_root),
        "pairs_root": str(args.pairs_root),
        "skipped": skipped,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNIMIT Observer Variability among human annotators (no model).")
    parser.add_argument("--annotations-json-root", type=Path,
                        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named",
                        help="Root containing one subfolder of JSONs per observer.")
    parser.add_argument("--pairs-root", type=Path,
                        default=Path("python_files") / "annotation tool" / "Pairs_PNIMIT_1_pairs",
                        help="Root containing pair folders with scans.")
    parser.add_argument("--observers", nargs="+", default=["Benny", "Nathalia", "Thaer"],
                        help="Human observer subfolder names (order defines matrix order).")
    parser.add_argument("--model-preds-root", type=Path,
                        default=Path("python_files") / "annotation tool" / "predictions_pnimit_lung",
                        help="Root containing per-pair model outputs (pair_<token>/output.nii.gz).")
    parser.add_argument("--model-name", default="Model", help="Display name for the model observer.")
    parser.add_argument("--no-model", action="store_true", help="Exclude the model observer (humans only).")
    parser.add_argument("--min-cc-size", type=int, default=50, help="Model CC size filter.")
    parser.add_argument("--min-cc-intensity", type=float, default=0.03, help="Model CC intensity filter.")
    parser.add_argument("--model-space", default="full",
                        choices=["full", "square_noseg", "square_seg"],
                        help="How the model output maps to the scan: 'full' (already full-res, e.g. lung), "
                             "'square_noseg' (whole-image square crop, e.g. full_thorax), "
                             "'square_seg' (square around lung-seg bbox + margin, e.g. lungs_med5).")
    parser.add_argument("--model-margin", type=int, default=5,
                        help="Extra pixels added to the seg bbox before the 15px crop pad (square_seg only).")
    parser.add_argument("--make-collages", action="store_true", default=True,
                        help="Generate one annotation+model collage PNG per pair (default on).")
    parser.add_argument("--no-collages", dest="make_collages", action="store_false",
                        help="Disable per-pair collage generation.")
    parser.add_argument("--collage-dir", type=Path, default=None,
                        help="Directory for per-pair collages (default: <out-dir>/pair_collages).")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("python_files") / "annotation tool" / "Annotations_Pnimit" / "JSON_only_named" / "ov_benny_nathalia_thaer_model",
                        help="Output directory for all OV results.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit number of pairs (0 = all).")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
