"""Patient-disjoint train/val/test split for the generated pair dataset.

Reads `manifest.jsonl` (produced by build_manifest.py) and assigns every pair to
a split such that:
  * NO patient appears in more than one split (prevents leakage of same-patient
    CT-RATE reconstructions, e.g. train_10000_a_1 vs train_10000_a_2),
  * ALL variants of a change_group (contrastive positives) stay together
    (guaranteed, since a change_group belongs to one case -> one patient).

Patient key:
  * CT-RATE `<'train'|'valid'>_<patient>_<scan>_<recon>` (e.g. train_10000_a_1)
    -> patient = `train_10000`  (strip the trailing _<scan>_<recon>).
  * Everything else (LUNA UIDs, volume-X, ABD_LYMPH_00X, ...) -> the case name
    itself (each is its own patient; no reconstructions).

Outputs (next to the manifest):
  * split_train.txt / split_val.txt / split_test.txt  -- pair_ids, one per line
  * manifest_split.jsonl                              -- records + `split` + `patient_key`
  * split_summary.json                                -- per-split patient/pair/group counts

Usage:
    python split_dataset.py -o /path/to/fcd_train
    python split_dataset.py -o <dir> --train 0.8 --val 0.1 --test 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

# CT-RATE case: <split>_<patient>_<scan>_<recon>, e.g. train_10000_a_1 / valid_53_b_2
_CTRATE_RE = re.compile(r"^((?:train|valid)_\d+)_[A-Za-z]+_\d+$")


def patient_key(case_name: str) -> str:
    """Map a case name to its patient-level key (see module docstring)."""
    if not case_name:
        return case_name
    m = _CTRATE_RE.match(case_name)
    if m:
        return m.group(1)
    return case_name


def _load_manifest(root: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(
            f"manifest.jsonl not found in {root}. Run build_manifest.py first:\n"
            f"    python build_manifest.py -o {root}"
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"manifest.jsonl in {root} is empty.")
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-o", "--output", required=True, help="Dataset root containing manifest.jsonl.")
    p.add_argument("--train", type=float, default=0.8, help="Train fraction of patients.")
    p.add_argument("--val", type=float, default=0.1, help="Val fraction of patients.")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction of patients.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _assign_patients(patients: List[str], train: float, val: float, test: float, seed: int) -> Dict[str, str]:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"train+val+test must sum to 1.0 (got {total}).")
    rng = random.Random(seed)
    shuffled = sorted(patients)          # deterministic base order
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    split_of: Dict[str, str] = {}
    for idx, pat in enumerate(shuffled):
        if idx < n_test:
            split_of[pat] = "test"
        elif idx < n_test + n_val:
            split_of[pat] = "val"
        else:
            split_of[pat] = "train"
    return split_of


def main() -> int:
    args = parse_args()
    root = os.path.abspath(args.output)
    records = _load_manifest(root)

    # Map each record -> patient, then split at the patient level.
    for r in records:
        r["patient_key"] = patient_key(r.get("case", ""))
    patients = sorted({r["patient_key"] for r in records})
    split_of = _assign_patients(patients, args.train, args.val, args.test, args.seed)

    for r in records:
        r["split"] = split_of[r["patient_key"]]

    # Write per-split id lists.
    by_split_ids: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_split_ids[r["split"]].append(r.get("pair_id"))
    for split in ("train", "val", "test"):
        with open(os.path.join(root, f"split_{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(by_split_ids.get(split, []))) + "\n")

    # Write the enriched manifest (records + split + patient_key).
    with open(os.path.join(root, "manifest_split.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary + leakage assertions.
    pats_by_split = defaultdict(set)
    groups_by_split = defaultdict(set)
    for r in records:
        pats_by_split[r["split"]].add(r["patient_key"])
        if r.get("change_group_id") is not None:
            groups_by_split[r["split"]].add(r["change_group_id"])

    # Hard guarantees: patients and change_groups never cross splits.
    all_pats = [p for s in pats_by_split.values() for p in s]
    assert len(all_pats) == len(set(all_pats)), "LEAK: a patient appears in multiple splits."
    all_groups = [g for s in groups_by_split.values() for g in s]
    assert len(all_groups) == len(set(all_groups)), "LEAK: a change_group spans splits."

    summary = {
        "seed": args.seed,
        "fractions": {"train": args.train, "val": args.val, "test": args.test},
        "num_patients": len(patients),
        "num_pairs": len(records),
        "per_split": {
            s: {
                "patients": len(pats_by_split.get(s, set())),
                "pairs": len(by_split_ids.get(s, [])),
                "change_groups": len(groups_by_split.get(s, set())),
                "by_effective_anomaly_type": dict(
                    Counter(r.get("effective_anomaly_type") for r in records if r["split"] == s)
                ),
            }
            for s in ("train", "val", "test")
        },
    }
    with open(os.path.join(root, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[split] wrote split_{{train,val,test}}.txt, manifest_split.jsonl, split_summary.json in {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
