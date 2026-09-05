"""Build a dataset manifest from a generated pair tree.

Scans an output directory produced by `generate_training_set.py` (i.e. the
enriched `params.json` written by DRR_generator.py, schema_version >= 2) and
emits:
    * manifest.jsonl  -- one JSON record per pair (full metadata + absolute paths)
    * manifest.csv    -- flat columns for quick filtering / pandas
    * manifest_summary.json -- counts per anomaly_type / direction / has_devices

This makes the set reusable across research projects and across every stage of
the foundation_contrastive_diff study (RQ1 seg, RQ2 contrastive, RQ3
disentanglement) without re-reading any NIfTI files: a loader can select pairs
by anomaly_type, direction, has_devices, pathology_vs_nuisance, angle_delta, etc.

Usage:
    python build_manifest.py -o /cs/labs/.../final/fcd_train
    python build_manifest.py -o <dir> --split-out   # also write train/val/test lists
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from typing import Any, Dict, Iterator, List

# Columns promoted to the flat CSV (everything is still in the JSONL).
_CSV_COLUMNS = [
    "pair_id", "case", "pair_index",
    "change_group_id", "variant_index", "num_variants",
    "anomaly_type", "effective_anomaly_type", "realized_change",
    "direction", "pathology_vs_nuisance",
    "num_pathologies", "has_devices", "single_pathology_mode",
    "direction_score", "changed_fraction", "angle_delta_l2_deg",
    "prior_path", "current_path", "diff_map_path", "params_path",
]


def _iter_param_files(root: str) -> Iterator[str]:
    for dirpath, _dirs, files in os.walk(root):
        if "params.json" in files:
            yield os.path.join(dirpath, "params.json")


def _load(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] could not read {path}: {e}")
        return None


def _record(params_path: str, root: str) -> Dict[str, Any] | None:
    meta = _load(params_path)
    if meta is None:
        return None
    pair_dir = os.path.dirname(params_path)
    files = meta.get("files", {})
    prior = os.path.join(pair_dir, files.get("prior", "prior.nii.gz"))
    current = os.path.join(pair_dir, files.get("current", "current.nii.gz"))
    diff = os.path.join(pair_dir, files.get("diff_map", "diff_map.nii.gz"))
    # Skip incomplete pairs (missing image outputs).
    if not (os.path.isfile(prior) and os.path.isfile(current) and os.path.isfile(diff)):
        return None

    rec = dict(meta)
    rec["pair_id"] = os.path.relpath(pair_dir, root).replace(os.sep, "/")
    rec["prior_path"] = os.path.abspath(prior)
    rec["current_path"] = os.path.abspath(current)
    rec["diff_map_path"] = os.path.abspath(diff)
    rec["params_path"] = os.path.abspath(params_path)
    return rec


def _flat_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    diff_stats = rec.get("diff_stats", {}) or {}
    return {
        "pair_id": rec.get("pair_id"),
        "case": rec.get("case"),
        "pair_index": rec.get("pair_index"),
        "change_group_id": rec.get("change_group_id"),
        "variant_index": rec.get("variant_index"),
        "num_variants": rec.get("num_variants"),
        "anomaly_type": rec.get("anomaly_type"),
        "effective_anomaly_type": rec.get("effective_anomaly_type"),
        "realized_change": rec.get("realized_change"),
        "direction": rec.get("direction"),
        "pathology_vs_nuisance": rec.get("pathology_vs_nuisance"),
        "num_pathologies": rec.get("num_pathologies"),
        "has_devices": rec.get("has_devices"),
        "single_pathology_mode": rec.get("single_pathology_mode"),
        "direction_score": rec.get("direction_score"),
        "changed_fraction": diff_stats.get("changed_fraction"),
        "angle_delta_l2_deg": rec.get("angle_delta_l2_deg"),
        "prior_path": rec.get("prior_path"),
        "current_path": rec.get("current_path"),
        "diff_map_path": rec.get("diff_map_path"),
        "params_path": rec.get("params_path"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-o", "--output", required=True, help="Root directory of the generated pair tree.")
    p.add_argument("--manifest-name", default="manifest", help="Base name for manifest files.")
    p.add_argument("--split-out", action="store_true", help="Also write case-disjoint train/val/test id lists.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _write_splits(records: List[Dict[str, Any]], out_base: str, val_frac: float, test_frac: float, seed: int) -> None:
    # Case-disjoint split so no CT leaks across train/val/test.
    cases = sorted({r.get("case") for r in records if r.get("case")})
    rng = random.Random(seed)
    rng.shuffle(cases)
    n = len(cases)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_cases = set(cases[:n_test])
    val_cases = set(cases[n_test:n_test + n_val])
    split_of = {}
    for r in records:
        c = r.get("case")
        split = "test" if c in test_cases else ("val" if c in val_cases else "train")
        split_of.setdefault(split, []).append(r["pair_id"])
    for split, ids in split_of.items():
        with open(f"{out_base}_{split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        print(f"[splits] {split}: {len(ids)} pairs")


def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.output)
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")

    records: List[Dict[str, Any]] = []
    for pp in _iter_param_files(root):
        rec = _record(pp, root)
        if rec is not None:
            records.append(rec)

    if not records:
        raise SystemExit(f"No complete pairs found under {root}")

    out_base = os.path.join(root, args.manifest_name)
    with open(f"{out_base}.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(f"{out_base}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for rec in records:
            w.writerow(_flat_row(rec))

    summary = {
        "num_pairs": len(records),
        "num_cases": len({r.get("case") for r in records}),
        "num_change_groups": len({r.get("change_group_id") for r in records}),
        "by_anomaly_type": dict(Counter(r.get("anomaly_type") for r in records)),
        "by_effective_anomaly_type": dict(Counter(r.get("effective_anomaly_type") for r in records)),
        "realized_change": int(sum(1 for r in records if r.get("realized_change"))),
        "by_direction": dict(Counter(r.get("direction") for r in records)),
        "with_devices": int(sum(1 for r in records if r.get("has_devices"))),
        "pathology_vs_nuisance": dict(Counter(r.get("pathology_vs_nuisance") for r in records)),
    }
    with open(f"{out_base}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[manifest] wrote {out_base}.jsonl / .csv / _summary.json")

    if args.split_out:
        _write_splits(records, out_base, args.val_frac, args.test_frac, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
