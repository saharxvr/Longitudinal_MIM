"""Stage-0 audit for the few-shot experiment (run on campus before generating/training).

Checks: train-split CT list and segmentation coverage, compartments/supervoxels/family
success rates and projection linearity on a few sample CTs, synthetic validation task pool
sizes, and ICU-100 consensus/per-reader non-empty counts. Writes one JSON report.

    python -m data_generation.audit_fewshot_inputs \
        --source_manifest /cs/labs/josko/sahar_aharon/fcd_train/manifest_split.jsonl \
        --clinical_root "<ICU annotation tool root>" --sample_cts 4 --output audit_fewshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from data_generation import generate_supervoxel_corpus as corpus
from data_generation import supervoxel_render as render
from data_generation import supervoxel_tasks as tasks


def segmentation_coverage(cases, templates):
    coverage = Counter()
    complete = 0
    for case, _, _ in cases:
        present = {name for name, template in templates.items() if os.path.isfile(template.format(case=case))}
        coverage.update(present)
        complete += all(name in present for name in corpus.REQUIRED_SEGS)
    return dict(cases=len(cases), with_all_lobes=complete, per_structure=dict(coverage))


def sample_ct_checks(cases, templates, count, device, seed=0):
    reports = []
    rng = np.random.default_rng(seed)
    for case, path, _ in [cases[index] for index in rng.permutation(len(cases))[:count]]:
        started = time.time()
        try:
            ct, segs, affine, missing = corpus.load_case(case, path, templates)
            ctx = tasks.CTContext(ct, segs, spacing=corpus.spacing_from_affine(affine))
            setup = time.time() - started
            families = {}
            for family in sorted(tasks.FAMILIES):
                change = tasks.sample_change(ctx, family, np.random.default_rng(1)) if "cutpaste" not in family else None
                if change is None:
                    families[family] = None
                    continue
                scene = render.render_scene(ctx, [(change, "current")], np.random.default_rng(2), device=device)
                families[family] = render.realized_statistics(scene["maps"][0])
            change = None
            for name in ("nodule", "heart_resize", "t1_density@lungs"):
                change = change or tasks.sample_change(ctx, name, np.random.default_rng(3))
            linearity = None
            if change is not None:
                frame = render.FrameRenderer(ctx.segs["lungs"], render.sample_angles(rng), device)
                delta = change.delta.full(ctx.shape)
                separate = frame.base(ctx.ct) + frame.delta(delta)
                combined = frame.base(ctx.ct + delta)
                linearity = float((combined - separate).abs().max() / (frame.delta(delta).abs().max() + 1e-6))
            reports.append(dict(case=case, shape=list(ct.shape), missing_segs=missing, inferior=ctx.inferior,
                                supervoxels=int(len(np.unique(ctx.supervoxels)) - 1),
                                compartment_voxels={name: int(mask.sum()) for name, mask in ctx.compartments.items()},
                                setup_seconds=setup, total_seconds=time.time() - started,
                                linearity_relative_error=linearity, families=families,
                                family_success=sum(value is not None and value["area_001"] > render.MIN_COMPONENT
                                                   for value in families.values())))
        except Exception as error:
            reports.append(dict(case=case, error=repr(error)))
    return reports


def validation_pools(dataset_root):
    from datasets.pair_dataset import _read_manifest
    from evaluation.few_shot_manifests import synthetic_task_pools

    pools, nuisance = synthetic_task_pools(_read_manifest(dataset_root), "val")
    return dict(tasks={task: dict(pairs=len(items), patients=len({patient for _, patient in items}))
                       for task, items in sorted(pools.items())}, nuisance_pairs=len(nuisance))


def icu_counts(clinical_root):
    from evaluation.few_shot_eval import ICUPairSource

    source = ICUPairSource(clinical_root)
    keys = sorted(source.cases)
    report = dict(pairs=len(keys))
    for mode in ("consensus", "consensus_2", *[f"reader:{reader}" for reader in source.readers]):
        report[f"nonempty_{mode}"] = sum(source.nonempty(key, mode) for key in keys)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--seg_root", default=corpus.SEG_ROOT)
    parser.add_argument("--sample_cts", type=int, default=4)
    parser.add_argument("--clinical_root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    templates = {name: template.replace(corpus.SEG_ROOT, args.seg_root)
                 for name, template in corpus.SEG_TEMPLATES.items()}
    cases = corpus.train_cases(args.source_manifest)
    report = dict(segmentation_coverage=segmentation_coverage(cases, templates),
                  validation_pools=validation_pools(str(Path(args.source_manifest).parent)),
                  sample_cts=sample_ct_checks(cases, templates, args.sample_cts, args.device))
    if args.clinical_root:
        report["icu"] = icu_counts(args.clinical_root)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sample_cts"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
