"""Compute per-pair disagreement levels analogous to consensus levels.

For each pair, this script:
1. Loads the 5 physician annotation maps (positive and negative).
2. Builds a consensus map per pair (same logic as Observer_Variability_*.py).
3. Counts detections at consensus level >= k for k = 1..5
   (where level k means at least k physicians marked the same finding).
4. Derives disagreement metrics and buckets each pair into a disagreement
   level (1..5), where:
     - disagreement_level = 6 - max_consensus_level_present_in_pair
     - higher disagreement_level => more disagreement
     - 0 means the pair has no human detections at all.

Defaults match the existing 98-pair OV setup.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from skimage.draw import ellipse


PHYSICIANS = ["Avi", "Benny", "Sigal", "Smadar", "Nitzan"]
NUM_HUMANS = 5
STRUCT = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

_PAIR_RE = re.compile(r"pair\D*(\d+)", re.IGNORECASE)
_LEADING_NUM_RE = re.compile(r"^(\d+)")
_ANY_NUM_RE = re.compile(r"(\d+)")


def _parse_pair_number(stem: str) -> int | None:
    for rx in (_PAIR_RE, _LEADING_NUM_RE, _ANY_NUM_RE):
        m = rx.search(stem)
        if m:
            return int(m.group(1))
    return None


def _build_pair_index(person_dir: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    if not person_dir.exists():
        return index
    for p in sorted(person_dir.rglob('*.json')):
        n = _parse_pair_number(p.stem)
        if n is not None and n not in index:
            index[n] = p
    return index

LABEL_MAP_RULES = {
    ('Appearance', None, None): 3,
    ('Disappearance', None, None): -3,
    ('Persistence', 'Increase', 'Increase'): 2,
    ('Persistence', 'Decrease', 'Decrease'): -2,
    ('Persistence', 'Increase', 'None'): 1,
    ('Persistence', 'Decrease', 'None'): -1,
    ('Persistence', 'None', 'Increase'): 1,
    ('Persistence', 'None', 'Decrease'): -1,
    ('Persistence', 'None', 'None'): 0,
    ('Persistence', 'Increase', 'Decrease'): (1, -1),
    ('Persistence', 'Decrease', 'Increase'): (1, -1),
}


def load_labels_map(json_path: Path, shape: tuple[int, int], audit: dict | None = None):
    """Load a physician annotation JSON into positive/negative ellipse maps.

    Mapping is identical to the OV scripts (raises KeyError on unknown combos).
    If `audit` is provided, increments counters per category so callers can
    verify how Persistence is being split between pos/neg/both/skipped.
    """
    pos = np.zeros(shape)
    neg = np.zeros(shape)
    with json_path.open('r', encoding='utf-8') as f:
        items = json.load(f)
    for l in items[1:]:
        rr, cc = ellipse(
            l['cx'], l['cy'], l['rx'], l['ry'],
            shape=shape, rotation=np.deg2rad(l['angle']),
        )
        label_type = l['label']
        s = l['size_change'] if label_type == 'Persistence' else None
        i = l['intensity_change'] if label_type == 'Persistence' else None
        c = LABEL_MAP_RULES[(label_type, s, i)]  # strict lookup, matches OV

        if audit is not None:
            if label_type == 'Appearance':
                audit['appearance'] += 1
            elif label_type == 'Disappearance':
                audit['disappearance'] += 1
            elif label_type == 'Persistence':
                if c == 0:
                    audit['persistence_skipped'] += 1
                elif isinstance(c, tuple):
                    audit['persistence_both'] += 1
                elif c > 0:
                    audit['persistence_pos'] += 1
                else:
                    audit['persistence_neg'] += 1

        if c == 0:
            continue
        if isinstance(c, int):
            (pos if c > 0 else neg)[rr, cc] = c
        else:
            pos[rr, cc] = c[0]
            neg[rr, cc] = c[1]
    return pos, neg


def build_consensus_map(human_maps):
    """Replicates the consensus-map construction from the OV script.

    For each physician CC, assign it the maximum number of overlapping
    physicians (its consensus level). Then take the elementwise max
    across all physician CC maps.
    """
    bin_maps = [(m != 0).astype(int) for m in human_maps]
    sum_map = np.sum(bin_maps, axis=0)
    labeled = [cc_label(m, STRUCT)[0] for m in bin_maps]
    consensus_map = np.zeros_like(sum_map)
    for hm in labeled:
        c_consensus = np.zeros_like(hm)
        vals = np.unique(hm)
        for v in vals:
            if v == 0:
                continue
            inter = sum_map * (hm == v)
            c_consensus[hm == v] = np.amax(inter)
        consensus_map = np.maximum(consensus_map, c_consensus)
    return consensus_map


def consensus_counts_ge(human_maps):
    """Return counts of CCs at consensus level >= k for k = 1..5."""
    consensus_map = build_consensus_map(human_maps)
    counts_ge = []
    for k in range(1, NUM_HUMANS + 1):
        _, n = cc_label(consensus_map >= k, STRUCT)
        counts_ge.append(int(n))
    return counts_ge


def find_pair_path(pairs_roots, pair_num):
    for root in pairs_roots:
        cand = root / f'pair{pair_num}'
        if cand.is_dir():
            return cand
    return None


def compute_pair_metrics(counts_ge):
    """Given [N>=1, N>=2, N>=3, N>=4, N>=5], compute per-pair metrics."""
    n1, n2, n3, n4, n5 = counts_ge
    total = n1
    if total == 0:
        return {
            'total_detections_ge1': 0,
            'detections_ge1': 0,
            'detections_ge2': 0,
            'detections_ge3': 0,
            'detections_ge4': 0,
            'detections_ge5': 0,
            'solo_only_count': 0,
            'low_consensus_count': 0,
            'high_consensus_count': 0,
            'max_consensus_present': 0,
            'disagreement_level': 0,
        }

    max_present = 0
    for k, val in enumerate(counts_ge, start=1):
        if val > 0:
            max_present = k
    # disagreement_level: 1 = full agreement (max_present=5), 5 = strongest disagreement (max_present=1)
    disagreement_level = NUM_HUMANS + 1 - max_present

    return {
        'total_detections_ge1': n1,
        'detections_ge1': n1,
        'detections_ge2': n2,
        'detections_ge3': n3,
        'detections_ge4': n4,
        'detections_ge5': n5,
        'solo_only_count': n1 - n2,           # CCs that exist only at level 1
        'low_consensus_count': n1 - n3,        # CCs visible only at level <= 2
        'high_consensus_count': n4,            # CCs at level >= 4
        'max_consensus_present': max_present,
        'disagreement_level': disagreement_level,
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_pairs_roots = [repo_root / 'annotation tool' / f'Pairs{i}' for i in range(1, 9)]
    default_anns_dir = repo_root / 'annotation tool' / 'Annotations'
    default_out_csv = repo_root / 'Sahar_work/files/ov_results_sq/no_cc_itamar_segs_98/disagreement_levels.csv'

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs-roots', nargs='+', type=Path, default=default_pairs_roots)
    ap.add_argument('--annotations-dir', type=Path, default=default_anns_dir)
    ap.add_argument('--num-pairs', type=int, default=98)
    ap.add_argument('--out-csv', type=Path, default=default_out_csv)
    args = ap.parse_args()

    person_indices = {phy: _build_pair_index(args.annotations_dir / phy) for phy in PHYSICIANS}

    audit = {
        'appearance': 0,
        'disappearance': 0,
        'persistence_pos': 0,
        'persistence_neg': 0,
        'persistence_both': 0,
        'persistence_skipped': 0,
    }

    rows = []
    for i in range(1, args.num_pairs + 1):
        pair_dir = find_pair_path(args.pairs_roots, i)
        if pair_dir is None:
            print(f'[skip] pair{i} not found')
            continue
        nii_files = sorted([
            p for p in pair_dir.iterdir()
            if p.name.endswith('.nii.gz') and not p.name.endswith('_lung_seg.nii.gz')
        ])
        if len(nii_files) < 2:
            print(f'[skip] pair{i} missing scans')
            continue
        current = nib.load(str(nii_files[1])).get_fdata()
        shape = current.shape

        pos_maps, neg_maps = [], []
        ok = True
        for phy in PHYSICIANS:
            ann_p = person_indices[phy].get(i)
            if ann_p is None:
                print(f'[skip] pair{i} missing annotation for {phy}')
                ok = False
                break
            p_map, n_map = load_labels_map(ann_p, shape, audit=audit)
            pos_maps.append(p_map)
            neg_maps.append(n_map)
        if not ok:
            continue

        pos_ge = consensus_counts_ge(pos_maps)
        neg_ge = consensus_counts_ge(neg_maps)
        all_ge = [pos_ge[k] + neg_ge[k] for k in range(NUM_HUMANS)]

        pos_metrics = compute_pair_metrics(pos_ge)
        neg_metrics = compute_pair_metrics(neg_ge)
        all_metrics = compute_pair_metrics(all_ge)

        row = {'pair': i}
        for prefix, m in [('pos', pos_metrics), ('neg', neg_metrics), ('all', all_metrics)]:
            for k, v in m.items():
                row[f'{prefix}_{k}'] = v
        rows.append(row)
        print(
            f'pair{i:>3}  '
            f'all_ge=[1:{all_ge[0]} 2:{all_ge[1]} 3:{all_ge[2]} 4:{all_ge[3]} 5:{all_ge[4]}]  '
            f"max={all_metrics['max_consensus_present']}  "
            f"disagreement_level={all_metrics['disagreement_level']}"
        )

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f'\nWrote {len(df)} pairs to {args.out_csv}')

    # Print disagreement-level distribution for the combined (pos+neg) view.
    if not df.empty and 'all_disagreement_level' in df.columns:
        dist = df['all_disagreement_level'].value_counts().sort_index()
        print('\nDisagreement level distribution (combined pos+neg):')
        for level, count in dist.items():
            print(f'  level {level}: {count} pairs')

    # Persistence audit: confirms that Persistence cases are split between
    # positive (Appearance-like), negative (Disappearance-like), or both,
    # exactly as the OV scripts do.
    print('\nLabel-type audit (across all loaded annotations):')
    print(f"  Appearance              -> pos map:   {audit['appearance']}")
    print(f"  Disappearance           -> neg map:   {audit['disappearance']}")
    print(f"  Persistence (Increase)  -> pos map:   {audit['persistence_pos']}")
    print(f"  Persistence (Decrease)  -> neg map:   {audit['persistence_neg']}")
    print(f"  Persistence (Inc+Dec)   -> both maps: {audit['persistence_both']}")
    print(f"  Persistence (None/None) -> skipped:   {audit['persistence_skipped']}")


if __name__ == '__main__':
    main()
