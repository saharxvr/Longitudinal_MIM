"""Human-only observer variability for the main experiment (5 radiologists, no model).

Produces everything the v2 paper needs, with NO model reference:
  1. Pairwise Agreement Index heatmaps (5x5), per-detection and per-pair, pos/neg/all.
  2. Leave-one-out recall by consensus level, per reader (no model lines).
  3. Reader-subgroup analysis: for every subset of readers of size m (1..5), the fraction
     of the full 5-reader reference findings recovered ("coverage"), to show how many
     radiologists are needed before the read stabilizes.

Run (from repo root):
    python python_files/Sahar_work/observer_variability_human_only.py
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import get_disagreement_levels as gdl  # noqa: E402
import observer_variability_nathalia_thaer_model as base  # noqa: E402
import observer_variability_main_loo_metrics as loo  # noqa: E402  (_new_level_acc, _accumulate_levels, _level_rates)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def plot_pai_heatmap(mat: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    """5x5 human-only PAI heatmap (no model separator)."""
    df = pd.DataFrame(mat, index=labels, columns=labels)
    n = len(labels)
    plt.figure(figsize=(max(6, n * 1.3), max(5, n * 1.15)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="vlag", vmin=0, vmax=1, center=0.5,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.9, "label": "PAI"},
                annot_kws={"fontsize": 12, "fontweight": "bold"})
    plt.xticks(rotation=0, fontsize=11, fontweight="bold")
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_recall_by_level(per_reader: dict, readers: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    for ax, sign, title in ((axes[0], "pos", "Positive changes"), (axes[1], "neg", "Negative changes")):
        for r in readers:
            y = per_reader[r][f"recall_by_level_{sign}"]
            ax.plot(range(1, len(y) + 1), y, marker="o", linewidth=1.8, label=r)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(1, 5))
        ax.set_xlabel("Consensus level (# of the other readers who agreed)")
        ax.set_ylabel("Leave-one-out sensitivity")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(frameon=True, edgecolor="black", loc="lower right", fontsize=9)
    fig.suptitle("Reader Sensitivity by Consensus Level (leave-one-out)", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(cov_by_m: dict, out_path: Path) -> None:
    m_vals = sorted(cov_by_m.keys())
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=180)
    styles = {"all": ("#111111", "o", "-"), "pos": ("#003366", "s", "--"), "neg": ("#800000", "^", "--")}
    labels = {"all": "All change", "pos": "Positive change", "neg": "Negative change"}
    for sign in ("all", "pos", "neg"):
        mean = [cov_by_m[m][sign]["mean"] for m in m_vals]
        lo = [cov_by_m[m][sign]["min"] for m in m_vals]
        hi = [cov_by_m[m][sign]["max"] for m in m_vals]
        color, marker, ls = styles[sign]
        ax.plot(m_vals, mean, marker=marker, linestyle=ls, color=color, linewidth=2.2, label=labels[sign])
        ax.fill_between(m_vals, lo, hi, color=color, alpha=0.12)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(m_vals)
    ax.set_xlabel("Number of radiologists in the subgroup")
    ax.set_ylabel("Coverage of the full 5-reader reference")
    ax.set_title("How Many Radiologists Stabilize the Read?", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, edgecolor="black", loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_hmdr_udpp(hmdr_by_m: dict, udpp_by_m: dict, out_path: Path) -> None:
    """Subgroup corroboration (HMDR) and solo-finding rate (UDPP) vs subgroup size."""
    m_vals = sorted(hmdr_by_m.keys())
    styles = {"all": ("#111111", "o", "-"), "pos": ("#003366", "s", "--"), "neg": ("#800000", "^", "--")}
    labels = {"all": "All change", "pos": "Positive change", "neg": "Negative change"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    for src, ax, ylab, title in ((hmdr_by_m, axes[0], "HMDR (fraction corroborated)",
                                  "Corroboration within the subgroup (HMDR)"),
                                 (udpp_by_m, axes[1], "UDPP (solo findings per reader per pair)",
                                  "Solo findings within the subgroup (UDPP)")):
        for sign in ("all", "pos", "neg"):
            mean = [src[m][sign]["mean"] for m in m_vals]
            lo = [src[m][sign]["min"] for m in m_vals]
            hi = [src[m][sign]["max"] for m in m_vals]
            color, marker, ls = styles[sign]
            ax.plot(m_vals, mean, marker=marker, linestyle=ls, color=color, linewidth=2.2, label=labels[sign])
            ax.fill_between(m_vals, lo, hi, color=color, alpha=0.12)
        ax.set_xticks(m_vals)
        ax.set_xlabel("Number of radiologists in the subgroup")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=True, edgecolor="black", loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> int:
    readers: list[str] = list(args.physicians)
    R = len(readers)
    disp = list(args.display_labels) if args.display_labels else list(readers)
    if len(disp) != R:
        raise ValueError("--display-labels must match the number of physicians.")
    ridx = {r: i for i, r in enumerate(readers)}
    person_idx = {r: gdl._build_pair_index(args.annotations_dir / r) for r in readers}
    coverage_counts = {r: len(person_idx[r]) for r in readers}

    # LOO accumulators (each reader vs the other R-1).
    loo_acc = {r: loo._new_level_acc(R - 1) for r in readers}

    # PAI accumulators (per detection: summed agreements/disagreements; per pair: summed c_pai).
    def zmat():
        return np.zeros((R, R), dtype=np.float64)
    ag = {"pos": zmat(), "neg": zmat()}
    dis = {"pos": zmat(), "neg": zmat()}
    pp = {"pos": zmat(), "neg": zmat()}          # per-pair PAI sum (pos/neg)
    pp_all = zmat()                               # per-pair PAI sum (all, one-sided boost)

    # Subgroup coverage accumulators: per subset -> summed overlapped / summed n_full.
    subsets = [S for m in range(1, R + 1) for S in combinations(range(R), m)]
    cov = {S: {"pos_ov": 0, "pos_den": 0, "neg_ov": 0, "neg_den": 0} for S in subsets}

    # Subgroup HMDR/UDPP accumulators: each member scored against the OTHER members of the subset.
    #   ov = findings corroborated by >=1 other member; tp = member's total findings; nov = solo findings.
    sg = {S: {"pos_ov": 0, "pos_tp": 0, "pos_nov": 0, "neg_ov": 0, "neg_tp": 0, "neg_nov": 0} for S in subsets}

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
        shape = tuple(nib.load(str(nii_files[1])).get_fdata().shape)

        pos_maps: dict[str, np.ndarray] = {}
        neg_maps: dict[str, np.ndarray] = {}
        missing = None
        for r in readers:
            ann_p = person_idx[r].get(i)
            if ann_p is None:
                missing = r
                break
            p_map, n_map = gdl.load_labels_map(ann_p, shape)
            pos_maps[r] = p_map
            neg_maps[r] = n_map
        if missing is not None:
            skipped.append((i, f"missing annotation: {missing}"))
            continue

        # LOO recall by consensus level.
        for r in readers:
            refs = [o for o in readers if o != r]
            loo._accumulate_levels(loo_acc[r], pos_maps[r], neg_maps[r],
                                   [pos_maps[o] for o in refs], [neg_maps[o] for o in refs])

        # PAI (per detection + per pair), pos and neg.
        for sign, maps in (("pos", pos_maps), ("neg", neg_maps)):
            for r1, r2 in combinations(readers, 2):
                a, d = base.get_pairwise_detections(maps[r1], maps[r2])
                i1, i2 = ridx[r1], ridx[r2]
                ag[sign][i1, i2] += 2 * a
                ag[sign][i2, i1] += 2 * a
                dis[sign][i1, i2] += d
                dis[sign][i2, i1] += d
                c = float(np.nan_to_num(2 * a / (2 * a + d) if (2 * a + d) else 1.0, nan=1.0))
                pp[sign][i1, i2] += c
                pp[sign][i2, i1] += c

        # Per-pair PAI (all) with the one-sided boost (matches the OV pipeline).
        for r1, r2 in combinations(readers, 2):
            i1, i2 = ridx[r1], ridx[r2]
            ap, dp = base.get_pairwise_detections(pos_maps[r1], pos_maps[r2])
            an, dn = base.get_pairwise_detections(neg_maps[r1], neg_maps[r2])
            ag_all, dis_all = 2 * (ap + an), (dp + dn)
            c = float(np.nan_to_num(ag_all / (ag_all + dis_all) if (ag_all + dis_all) else 1.0, nan=1.0))
            one_sided = ((2 * ap + dp == 0) and (2 * an + dn > 0)) or ((2 * ap + dp > 0) and (2 * an + dn == 0))
            if one_sided:
                c = c * 0.5 + 0.5
            pp_all[i1, i2] += c
            pp_all[i2, i1] += c

        # Subgroup coverage.
        for sign, maps in (("pos", pos_maps), ("neg", neg_maps)):
            bins = [(maps[r] != 0).astype(np.uint8) for r in readers]
            full = np.zeros_like(bins[0])
            for b in bins:
                full |= b
            full_ccs, n_full = label(full, gdl.STRUCT)
            if n_full == 0:
                continue
            for S in subsets:
                sub = np.zeros_like(bins[0])
                for k in S:
                    sub |= bins[k]
                overlapped = len(np.unique(full_ccs * sub)) - 1
                cov[S][f"{sign}_ov"] += overlapped
                cov[S][f"{sign}_den"] += n_full

        # Subgroup HMDR / UDPP: each member vs the OTHER members of the subgroup.
        for sign, maps in (("pos", pos_maps), ("neg", neg_maps)):
            for S in subsets:
                for k in S:
                    others = [maps[readers[o]] for o in S if o != k]
                    if others:
                        ov, nov, tp = base.get_hmdr_udpp_counts(maps[readers[k]], others)
                    else:
                        _, tp = label(maps[readers[k]] != 0, gdl.STRUCT)
                        ov, nov = 0, int(tp)
                    sg[S][f"{sign}_ov"] += ov
                    sg[S][f"{sign}_tp"] += tp
                    sg[S][f"{sign}_nov"] += nov

        pairs_processed += 1
        print(f"[pair {i:>3}] processed (total {pairs_processed})", flush=True)

    if pairs_processed == 0:
        raise RuntimeError("No pairs processed.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── PAI matrices + heatmaps ────────────────────────────────────────────────
    mat_label_pos = ag["pos"] / np.where(ag["pos"] + dis["pos"] == 0, 1, ag["pos"] + dis["pos"])
    mat_label_neg = ag["neg"] / np.where(ag["neg"] + dis["neg"] == 0, 1, ag["neg"] + dis["neg"])
    denom_all = ag["pos"] + dis["pos"] + ag["neg"] + dis["neg"]
    mat_label_all = (ag["pos"] + ag["neg"]) / np.where(denom_all == 0, 1, denom_all)
    mat_pair_pos = pp["pos"] / pairs_processed
    mat_pair_neg = pp["neg"] / pairs_processed
    mat_pair_all = pp_all / pairs_processed
    for m in (mat_label_pos, mat_label_neg, mat_label_all, mat_pair_pos, mat_pair_neg, mat_pair_all):
        np.fill_diagonal(m, 1.0)

    heatmaps = [
        (mat_label_all, "per_label_agreement_all_humans.png", "Pairwise Agreement Index per Detection (all change)"),
        (mat_label_pos, "per_label_agreement_pos_humans.png", "Pairwise Agreement Index per Detection (positive)"),
        (mat_label_neg, "per_label_agreement_neg_humans.png", "Pairwise Agreement Index per Detection (negative)"),
        (mat_pair_all, "per_pair_agreement_all_humans.png", "Pairwise Agreement Index per Pair (all change)"),
        (mat_pair_pos, "per_pair_agreement_pos_humans.png", "Pairwise Agreement Index per Pair (positive)"),
        (mat_pair_neg, "per_pair_agreement_neg_humans.png", "Pairwise Agreement Index per Pair (negative)"),
    ]
    for mat, fname, title in heatmaps:
        plot_pai_heatmap(mat, disp, args.out_dir / fname, title)

    def offdiag_stats(mat):
        vals = [mat[a, b] for a in range(R) for b in range(R) if a < b]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

    pai_summary = {
        "per_detection_all": offdiag_stats(mat_label_all),
        "per_detection_pos": offdiag_stats(mat_label_pos),
        "per_detection_neg": offdiag_stats(mat_label_neg),
        "per_pair_all": offdiag_stats(mat_pair_all),
        "per_pair_pos": offdiag_stats(mat_pair_pos),
        "per_pair_neg": offdiag_stats(mat_pair_neg),
        "matrices": {
            "per_detection_all": mat_label_all.round(3).tolist(),
            "per_pair_all": mat_pair_all.round(3).tolist(),
            "readers": disp,
        },
    }

    # ── LOO recall by level ─────────────────────────────────────────────────────────────
    per_reader = {disp[ridx[r]]: loo._level_rates(loo_acc[r]) for r in readers}
    plot_recall_by_level(per_reader, disp, args.out_dir / "recall_by_consensus_level_humans.png")

    # ── Subgroup coverage ──────────────────────────────────────────────────────
    cov_by_m: dict[int, dict] = {}
    for m in range(1, R + 1):
        subs_m = [S for S in subsets if len(S) == m]
        by_sign = {}
        for sign in ("pos", "neg", "all"):
            vals = []
            for S in subs_m:
                if sign == "all":
                    ov = cov[S]["pos_ov"] + cov[S]["neg_ov"]
                    den = cov[S]["pos_den"] + cov[S]["neg_den"]
                else:
                    ov = cov[S][f"{sign}_ov"]
                    den = cov[S][f"{sign}_den"]
                vals.append(safe_div(ov, den))
            by_sign[sign] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                             "max": float(np.max(vals)), "n_subsets": len(vals)}
        cov_by_m[m] = by_sign
    plot_coverage(cov_by_m, args.out_dir / "reader_subgroup_coverage.png")

    marginal = {sign: {m: cov_by_m[m][sign]["mean"] - cov_by_m[m - 1][sign]["mean"]
                       for m in range(2, R + 1)} for sign in ("pos", "neg", "all")}

    # ── Subgroup HMDR / UDPP ───────────────────────────────────────────────────
    hmdr_by_m: dict[int, dict] = {}
    udpp_by_m: dict[int, dict] = {}
    for m in range(1, R + 1):
        subs_m = [S for S in subsets if len(S) == m]
        h_by_sign, u_by_sign = {}, {}
        for sign in ("pos", "neg", "all"):
            h_vals, u_vals = [], []
            for S in subs_m:
                if sign == "all":
                    ov = sg[S]["pos_ov"] + sg[S]["neg_ov"]
                    tp = sg[S]["pos_tp"] + sg[S]["neg_tp"]
                    nov = sg[S]["pos_nov"] + sg[S]["neg_nov"]
                else:
                    ov, tp, nov = sg[S][f"{sign}_ov"], sg[S][f"{sign}_tp"], sg[S][f"{sign}_nov"]
                h_vals.append(safe_div(ov, tp))                       # HMDR = corroborated / total
                u_vals.append(nov / (pairs_processed * m))            # UDPP per reader per pair
            h_by_sign[sign] = {"mean": float(np.mean(h_vals)), "min": float(np.min(h_vals)), "max": float(np.max(h_vals))}
            u_by_sign[sign] = {"mean": float(np.mean(u_vals)), "min": float(np.min(u_vals)), "max": float(np.max(u_vals))}
        hmdr_by_m[m] = h_by_sign
        udpp_by_m[m] = u_by_sign
    plot_hmdr_udpp(hmdr_by_m, udpp_by_m, args.out_dir / "reader_subgroup_hmdr_udpp.png")

    out = {
        "pairs_processed": pairs_processed,
        "readers": readers,
        "annotation_coverage": coverage_counts,
        "pai": pai_summary,
        "loo_recall_by_level": per_reader,
        "subgroup_coverage": {str(m): cov_by_m[m] for m in cov_by_m},
        "subgroup_marginal_gain": {s: {str(m): marginal[s][m] for m in marginal[s]} for s in marginal},
        "subgroup_hmdr": {str(m): hmdr_by_m[m] for m in hmdr_by_m},
        "subgroup_udpp": {str(m): udpp_by_m[m] for m in udpp_by_m},
        "skipped": skipped,
    }
    (args.out_dir / "human_ov_metrics.json").write_text(json.dumps(out, indent=4), encoding="utf-8")

    # Console summary.
    print(f"\nPairs processed: {pairs_processed}/{args.num_pairs}")
    print(f"Mean per-detection PAI (all): {pai_summary['per_detection_all'][0]:.3f} "
          f"[{pai_summary['per_detection_all'][1]:.2f}-{pai_summary['per_detection_all'][2]:.2f}]")
    print(f"Mean per-pair PAI (all):      {pai_summary['per_pair_all'][0]:.3f} "
          f"[{pai_summary['per_pair_all'][1]:.2f}-{pai_summary['per_pair_all'][2]:.2f}]")
    print("\nSubgroup coverage of the full 5-reader reference (mean [min-max]):")
    print(f"{'m':>2} {'all':>18} {'pos':>18} {'neg':>18}")
    for m in range(1, R + 1):
        row = "  ".join(
            f"{cov_by_m[m][s]['mean']:.3f} [{cov_by_m[m][s]['min']:.2f}-{cov_by_m[m][s]['max']:.2f}]"
            for s in ("all", "pos", "neg"))
        print(f"{m:>2} {row}")
    print("\nSubgroup HMDR (corroboration) and UDPP (solo findings/pair), all change, mean [min-max]:")
    print(f"{'m':>2} {'HMDR':>20} {'UDPP':>20}")
    for m in range(1, R + 1):
        h, u = hmdr_by_m[m]["all"], udpp_by_m[m]["all"]
        print(f"{m:>2} {h['mean']:.3f} [{h['min']:.2f}-{h['max']:.2f}]   {u['mean']:.3f} [{u['min']:.2f}-{u['max']:.2f}]")
    print("\nMarginal gain per added reader (all):")
    for m in range(2, R + 1):
        print(f"  {m-1}->{m}: +{marginal['all'][m]*100:.1f}%")
    print(f"\nSaved to {args.out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    ann_tool = repo_root / "annotation tool"
    p = argparse.ArgumentParser(description="Human-only observer variability (PAI + LOO + subgroup coverage).")
    p.add_argument("--annotations-dir", type=Path, default=ann_tool / "Annotations")
    p.add_argument("--pairs-roots", nargs="+", type=Path, default=[ann_tool / f"Pairs{i}" for i in range(1, 9)])
    p.add_argument("--physicians", nargs="+", default=["Avi", "Benny", "Sigal", "Smadar", "Nitzan"])
    p.add_argument("--display-labels", nargs="+", default=["R1", "R2", "R3", "R4", "R5"],
                   help="Anonymized labels for figures (aligned with --physicians order).")
    p.add_argument("--num-pairs", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=repo_root / "Sahar_work" / "files" / "ov_results_human_only")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
