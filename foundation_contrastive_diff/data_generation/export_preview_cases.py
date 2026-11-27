"""Campus side: bundle 1-2 TRAIN-split CTs (+ all segmentations) for local synthetic previews.

Copies, per case:  <output>/<case>/ct.nii.gz and <output>/<case>/<structure>.nii.gz
(structures = generate_supervoxel_corpus.SEG_TEMPLATES keys), plus up to ``--reference_pairs``
existing fcd_train pairs of the same case per pathology (prior/current/diff_map/params) so the
new families can be compared with the current synthetic pathologies on the same anatomy.

    python -m data_generation.export_preview_cases \
        --source_manifest /cs/labs/josko/sahar_aharon/fcd_train/manifest_split.jsonl \
        --output /cs/labs/josko/sahar_aharon/fewshot_preview_cases --count 2
    tar -czf fewshot_preview_cases.tgz -C /cs/labs/josko/sahar_aharon fewshot_preview_cases
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_generation import generate_supervoxel_corpus as corpus


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_cases(source_manifest, templates):
    cases = []
    for case, ct_path, patient in corpus.train_cases(source_manifest):
        paths = {name: template.format(case=case) for name, template in templates.items() if name != "lungs"}
        if os.path.isfile(ct_path) and all(os.path.isfile(path) for path in paths.values()):
            cases.append((case, ct_path, patient, paths))
    return cases


def reference_pairs(source_manifest, case, per_type):
    chosen = defaultdict(list)
    with open(source_manifest, encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line) if line.strip() else None
            if (not record or record.get("case") != case or record.get("split") != "train"
                    or not record.get("realized_change")):
                continue
            anomaly = record.get("effective_anomaly_type", "none")
            if len(chosen[anomaly]) < per_type:
                chosen[anomaly].append(record)
    return [record for records in chosen.values() for record in records]


def source_kind(case):
    if case.startswith(("train_", "valid_")):
        return "ct_rate"
    if case.startswith("1.3.6"):
        return "luna"
    if case.startswith("volume"):
        return "lits"
    return "other"


def diverse_choice(available, count, seed):
    """Seeded order, preferring one case per CT source (CT-RATE / LUNA / LiTS / other)."""
    order = [available[index] for index in np.random.default_rng(seed).permutation(len(available))]
    chosen, kinds = [], set()
    for item in order:
        if len(chosen) < count and source_kind(item[0]) not in kinds:
            chosen.append(item)
            kinds.add(source_kind(item[0]))
    chosen += [item for item in order if item not in chosen][:count - len(chosen)]
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="*", help="Explicit train-split case names (default: seeded choice)")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference_pairs", type=int, default=1)
    parser.add_argument("--seg_root", default=corpus.SEG_ROOT)
    args = parser.parse_args(argv)
    templates = {name: template.replace(corpus.SEG_ROOT, args.seg_root)
                 for name, template in corpus.SEG_TEMPLATES.items()}
    available = complete_cases(args.source_manifest, templates)
    if not available:
        raise SystemExit("No train-split case has the CT and all segmentations under the configured "
                         "templates; check --seg_root (see generate_supervoxel_corpus.SEG_TEMPLATES)")
    if args.cases:
        selected = [item for item in available if item[0] in set(args.cases)]
        missing = set(args.cases) - {item[0] for item in selected}
        if missing:
            raise SystemExit(f"Not train-split or missing segmentations: {sorted(missing)}")
    else:
        selected = diverse_choice(available, args.count, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, ct_path, patient, paths in selected:
        folder = output / case
        folder.mkdir()
        files = {"ct": (ct_path, folder / "ct.nii.gz")}
        files.update({name: (path, folder / f"{name}.nii.gz") for name, path in paths.items()})
        for source, destination in files.values():
            shutil.copy2(source, destination)
        pairs = []
        for record in reference_pairs(args.source_manifest, case, args.reference_pairs):
            pair_folder = folder / "reference_pairs" / record["pair_id"].replace("/", "__")
            pair_folder.mkdir(parents=True)
            for key, name in (("prior_path", "prior.nii.gz"), ("current_path", "current.nii.gz"),
                              ("diff_map_path", "diff_map.nii.gz"), ("params_path", "params.json")):
                if record.get(key) and os.path.isfile(record[key]):
                    shutil.copy2(record[key], pair_folder / name)
            pairs.append(dict(pair_id=record["pair_id"], anomaly=record.get("effective_anomaly_type"),
                              direction=record.get("direction"), folder=pair_folder.relative_to(output).as_posix()))
        records.append(dict(case=case, patient_key=patient, split="train",
                            files={name: dict(source=source, sha256=sha256(destination))
                                   for name, (source, destination) in files.items()},
                            reference_pairs=pairs))
    (output / "cases.json").write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"Exported {len(records)} cases ({size / 2 ** 20:.0f} MB) to {output}")
    print(f"tar -czf {output.name}.tgz -C {output.parent} {output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
