"""Local preview of every synthetic change family on 1-2 real CTs (DRRs + labels per type).

Input: a folder exported by export_preview_cases.py (<case>/ct.nii.gz + <case>/<structure>.nii.gz,
optional <case>/reference_pairs/*). Output (new folder; contains medical images, do not publish):
    families/<case>/<family>_<k>.png   CT coronal before/after, DRR prior/current, image difference,
                                       tight GT, ellipse label (all in the upright model frame)
    contact_<tier>.png                 one thumbnail per family (current DRR + GT)
    scenes/<case>_scene<k>.png         composed scenes: all changes, one task target, distractors
    modes/<case>_<family>.png          one change in every timepoint mode (appear, resolve, progress,
                                       regress, stable) with prior/current DRR and GT
    nuisance/<case>_inspiration.png    breathing-depth nuisance (never in GT)
    reference/<pair>.png               existing synthetic pathology pairs of the same CT
    index.md, stats.json               per-family realized areas and failures

Family panels show the appear mode (change in the current only; resolve only flips the sign).
Devices are not available locally (campus device library) and are omitted.

    python -m data_generation.preview_supervoxel_families --cases_root <dir> --output <new dir>
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from data_generation import generate_supervoxel_corpus as corpus
from data_generation import supervoxel_render as render
from data_generation import supervoxel_tasks as tasks

WINDOW = (-1000.0, 400.0)


def norm(image):
    image = np.asarray(image, np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def seed_for(*parts):
    return [zlib.crc32(str(part).encode()) & 0x7FFFFFFF for part in parts]


def case_folders(root):
    root = Path(root)
    return sorted(path for path in root.iterdir() if (path / "ct.nii.gz").is_file())


def load_context(folder, cache_dir, n_supervoxels):
    templates = {name: str(folder.parent / "{case}" / f"{name}.nii.gz") for name in corpus.SEG_TEMPLATES}
    started = time.time()
    ct, segs, affine, missing = corpus.load_case(folder.name, str(folder / "ct.nii.gz"), templates)
    cache = Path(cache_dir) / f"{folder.name}_supervoxels_mm.npy"
    supervoxels = np.load(cache) if cache.is_file() else None
    ctx = tasks.CTContext(ct, segs, spacing=corpus.spacing_from_affine(affine), supervoxels=supervoxels,
                          n_supervoxels=n_supervoxels, seed=seed_for(folder.name)[0])
    if supervoxels is None or supervoxels is not ctx.supervoxels:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, ctx.supervoxels)
    info = dict(case=folder.name, cropped_shape=list(ct.shape), spacing_mm=[round(v, 3) for v in ctx.spacing],
                compartment_grid=ctx.ds.tolist(), supervoxel_grid=ctx.sv_ds.tolist(), lung_scale_mm=round(ctx.scale, 1),
                missing_segs=missing, inferior=ctx.inferior, inferior_rule=ctx.inferior_rule,
                supervoxels=int(len(np.unique(ctx.supervoxels)) - 1), setup_seconds=round(time.time() - started, 1),
                compartment_voxels={name: int(mask.sum()) for name, mask in ctx.compartments.items()})
    return ctx, info


def coronal(ctx, change):
    """Coronal slice (z, x) through the AP index with the most changed voxels: before, after, support."""
    support = change.delta.support()
    box = change.delta.box
    column = int(np.argmax(support.sum((0, 1)))) + box[2].start
    before = ctx.ct[:, :, column].copy()
    after = before.copy()
    mask = np.zeros_like(before, bool)
    local = column - box[2].start
    after[box[0], box[1]] += change.delta.values[:, :, local]
    mask[box[0], box[1]] = support[:, :, local]
    return before, after, mask


def overlay(axis, image, signed, title, alpha=0.6, limit=0.3):
    axis.imshow(norm(image), cmap="gray")
    masked = np.ma.masked_where(np.abs(signed) < 1e-6, signed)
    axis.imshow(masked, cmap="bwr", vmin=-limit, vmax=limit, alpha=alpha)
    axis.set_title(title, fontsize=8)


def family_panel(ctx, change, current_image, prior_image, base_projection, gt, ellipses, stats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before, after, mask = coronal(ctx, change)
    figure, axes = plt.subplots(1, 7, figsize=(24, 4))
    for axis, image, title in ((axes[0], before, "CT coronal before"), (axes[1], after, "CT coronal after")):
        axis.imshow(np.clip(image, *WINDOW), cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1],
                    aspect=float(ctx.spacing[0] / ctx.spacing[1]))
        axis.set_title(title, fontsize=8)
    axes[1].contour(mask, levels=[0.5], colors="yellow", linewidths=0.6)
    axes[2].imshow(norm(prior_image), cmap="gray")
    axes[2].set_title("DRR prior (other angle)", fontsize=8)
    axes[3].imshow(norm(current_image), cmap="gray")
    axes[3].set_title("DRR current", fontsize=8)
    difference = norm(current_image) - norm(base_projection)
    axes[4].imshow(difference, cmap="bwr", vmin=-0.08, vmax=0.08)
    axes[4].set_title("current - registered prior (image)", fontsize=8)
    overlay(axes[5], current_image, gt, f"tight GT  area>0.001={stats['area_001']}  >0.05={stats['area_005']}")
    positive, negative = render.rasterize_ellipses(ellipses, gt.shape)
    overlay(axes[6], current_image, positive * 0.3 - negative * 0.3, f"ellipse label ({len(ellipses)})", alpha=0.35)
    for axis in axes:
        axis.axis("off")
    params = {key: (round(value, 2) if isinstance(value, float) else value)
              for key, value in change.params.items() if not isinstance(value, (list, dict))}
    figure.suptitle(f"{change.family} | {change.tier} | {change.compartment} | effect={change.effect} | "
                    f"GT sign={render.gt_sign(stats)} | realized={stats['area_001'] > render.MIN_COMPONENT} | "
                    f"max|GT|={stats['max_abs']:.3f} | {params}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=70)
    plt.close(figure)


def contact_sheet(cells, path, columns=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not cells:
        return
    rows = (len(cells) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3.2 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (title, image, gt) in zip(axes.ravel(), cells):
        overlay(axis, image[::2, ::2], gt[::2, ::2], title, alpha=0.7)
    figure.tight_layout()
    figure.savefig(path, dpi=60)
    plt.close(figure)


def scene_panel(ctx, scene, changes, records, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datasets.episodes import Task, task_target

    maps = torch.from_numpy(np.stack(scene["maps"])) if scene["maps"] else torch.zeros((0, 512, 512))
    task = Task("family", family=records[0]["family"]) if records else Task("any")
    target, ignore, everything, _ = task_target(maps, records, task)
    figure, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(norm(scene["prior"]), cmap="gray")
    axes[0].set_title("prior (other angle, prior-timepoint changes)", fontsize=8)
    axes[1].imshow(norm(scene["current"]), cmap="gray")
    axes[1].set_title("current", fontsize=8)
    overlay(axes[2], scene["current"], everything.numpy(), "all realized changes")
    overlay(axes[3], scene["current"], target.numpy(), f"episode target: family={task.family}")
    axes[3].contour(ignore.numpy(), levels=[0.5], colors="yellow", linewidths=0.8)
    distractors = (everything - target).numpy() * (np.abs(target.numpy()) < 1e-6)
    overlay(axes[4], scene["current"], distractors, "distractors (unlabeled in this episode)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(" | ".join(f"{r['family']}@{r['timepoint']}:{r['gt_sign']}:{r['area_001']}px" for r in records)
                    or "nuisance-only scene", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=70)
    plt.close(figure)


def modes_panel(ctx, change, rng, angles, path):
    """One change in every timepoint mode: prior DRR, current DRR, GT (rows = modes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    partial, method = tasks.make_partial(ctx, change, rng, 0.5)
    change.partial = partial
    modes = [mode for mode in tasks.TIMEPOINT_MODES if partial is not None or mode not in ("progress", "regress")]
    figure, axes = plt.subplots(len(modes), 3, figsize=(10, 3.3 * len(modes)), squeeze=False)
    summary = []
    for row, mode in zip(axes, modes):
        scene = render.render_scene(ctx, [(change, mode)], rng, angles=angles)
        gt = scene["maps"][0]
        stats = render.realized_statistics(gt)
        row[0].imshow(norm(scene["prior"]), cmap="gray")
        row[0].set_title(f"{mode}: prior (other angle)", fontsize=8)
        row[1].imshow(norm(scene["current"]), cmap="gray")
        row[1].set_title(f"{mode}: current", fontsize=8)
        overlay(row[2], scene["current"], gt, f"GT {render.gt_sign(stats)} {stats['area_001']}px max {stats['max_abs']:.3f}")
        for axis in row:
            axis.axis("off")
        summary.append(dict(mode=mode, gt_sign=render.gt_sign(stats), area_001=stats["area_001"]))
    figure.suptitle(f"{change.family}: timepoint modes (progress/regress smaller version: {method}, fraction 0.5)",
                    fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=60)
    plt.close(figure)
    return summary


def nuisance_panel(ctx, prior_angles, rng, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta, params = tasks.inspiration_nuisance(ctx, rng)
    frame = render.FrameRenderer(ctx.segs["lungs"], prior_angles)
    clean = frame.base(ctx.ct).numpy()
    breathing = frame.base(ctx.ct + delta.full(ctx.shape)).numpy()
    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(norm(clean), cmap="gray")
    axes[0].set_title("prior without nuisance", fontsize=8)
    axes[1].imshow(norm(breathing), cmap="gray")
    axes[1].set_title(f"prior with inspiration nuisance {params}", fontsize=8)
    axes[2].imshow(norm(breathing) - norm(clean), cmap="bwr", vmin=-0.08, vmax=0.08)
    axes[2].set_title("difference (never in any GT)", fontsize=8)
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=70)
    plt.close(figure)


def reference_panels(folder, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datasets.pair_dataset import _load_nii_2d, orient_tensor

    written = []
    for pair in sorted((folder / "reference_pairs").glob("*")) if (folder / "reference_pairs").is_dir() else []:
        try:
            arrays = [orient_tensor(torch.from_numpy(_load_nii_2d(str(pair / name))), 3, True).numpy()
                      for name in ("prior.nii.gz", "current.nii.gz", "diff_map.nii.gz")]
        except Exception:
            continue
        params = json.loads((pair / "params.json").read_text()) if (pair / "params.json").is_file() else {}
        figure, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(norm(arrays[0]), cmap="gray")
        axes[0].set_title("prior", fontsize=8)
        axes[1].imshow(norm(arrays[1]), cmap="gray")
        axes[1].set_title("current", fontsize=8)
        overlay(axes[2], arrays[1], np.clip(arrays[2], -1, 1),
                f"existing GT  area>0.001={int((np.abs(arrays[2]) > 0.001).sum())}")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(f"existing pair {pair.name}: {params.get('effective_anomaly_type')} / {params.get('direction')}",
                        fontsize=9)
        figure.tight_layout()
        path = destination / f"{pair.name}.png"
        figure.savefig(path, dpi=70)
        plt.close(figure)
        written.append(dict(pair=pair.name, anomaly=params.get("effective_anomaly_type"),
                            direction=params.get("direction"), png=path.name))
    return written


def selected_families(tiers, filters):
    names = [name for name in sorted(tasks.FAMILIES) if tasks.FAMILIES[name]["tier"] in tiers]
    if filters:
        names = [name for name in names if any(item in name for item in filters)]
    return names


def preview_case(folder, args, output, families):
    ctx, info = load_context(folder, output / "_cache", args.n_supervoxels)
    rng = np.random.default_rng(seed_for(args.seed, folder.name))
    current_angles, prior_angles = render.sample_angles(rng), render.sample_angles(rng)
    started = time.time()
    frame = render.FrameRenderer(ctx.segs["lungs"], current_angles)
    base_projection = frame.base(ctx.ct)
    prior_image = render.FrameRenderer(ctx.segs["lungs"], prior_angles).base(ctx.ct).numpy()
    scale = float((base_projection * frame.footprint).max())
    base_projection = base_projection.numpy()
    info.update(current_angles=list(current_angles), prior_angles=list(prior_angles),
                frame_seconds=round(time.time() - started, 1))
    donors = corpus.DonorBank(seed=args.seed)
    donors.add(ctx)
    destination = output / "families" / folder.name
    destination.mkdir(parents=True, exist_ok=True)
    rows, cells = [], {}
    for family in families:
        for sample in range(args.samples):
            family_rng = np.random.default_rng(seed_for(args.seed, folder.name, family, sample))
            started = time.time()
            change = tasks.sample_change(ctx, family, family_rng, donors if "cutpaste" in family else None)
            if change is None:
                rows.append(dict(case=folder.name, family=family, sample=sample, tier=tasks.FAMILIES[family]["tier"],
                                 produced=False))
                continue
            projection = frame.delta(change.delta.full(ctx.shape)).numpy()
            gt = render.clean_map(projection / scale)
            current_image = base_projection + projection
            ellipses = render.fit_ellipses(gt)
            stats = render.realized_statistics(gt)
            png = destination / f"{family.replace('@', '_at_')}_{sample}.png"
            family_panel(ctx, change, current_image, prior_image, base_projection, gt, ellipses, stats, png)
            rows.append(dict(case=folder.name, family=family, sample=sample, tier=change.tier,
                             compartment=change.compartment, effect=change.effect, produced=True,
                             realized=stats["area_001"] > render.MIN_COMPONENT, gt_sign=render.gt_sign(stats),
                             ellipses=len(ellipses), seconds=round(time.time() - started, 1),
                             png=png.relative_to(output).as_posix(), **stats))
            if sample == 0:
                cells.setdefault(change.tier, []).append(
                    (f"{family}\n{stats['area_001']}px {render.gt_sign(stats)}", current_image, gt))
        print(f"[{folder.name}] {family}: {sum(r['produced'] for r in rows if r['family'] == family)}"
              f"/{args.samples} produced", flush=True)
    for tier, tier_cells in cells.items():
        contact_sheet(tier_cells, output / f"contact_{folder.name}_{tier}.png")
    scenes = []
    (output / "scenes").mkdir(exist_ok=True)
    for index in range(args.scenes):
        scene_rng = np.random.default_rng(seed_for(args.seed, folder.name, "scene", index))
        changes = tasks.compose_scene(ctx, scene_rng, tiers=("T1", "T2"), n_changes=(2, 3), donor_fn=donors)
        scene = render.render_scene(ctx, changes, scene_rng)
        records = []
        for (change, timepoint), signed in zip(changes, scene["maps"]):
            stats = render.realized_statistics(signed)
            records.append(dict(family=change.family, compartment=change.compartment, timepoint=timepoint,
                                gt_sign=render.gt_sign(stats), realized=stats["area_001"] > render.MIN_COMPONENT,
                                area_001=stats["area_001"]))
        path = output / "scenes" / f"{folder.name}_scene{index}.png"
        scene_panel(ctx, scene, changes, records, path)
        scenes.append(dict(png=path.relative_to(output).as_posix(), changes=records))
    (output / "modes").mkdir(exist_ok=True)
    for family in [name for name in args.mode_families if name in tasks.FAMILIES]:
        mode_rng = np.random.default_rng(seed_for(args.seed, folder.name, "modes", family))
        change = tasks.sample_change(ctx, family, mode_rng, donors if "cutpaste" in family else None)
        if change is not None:
            path = output / "modes" / f"{folder.name}_{family.replace('@', '_at_')}.png"
            summary = modes_panel(ctx, change, mode_rng, (current_angles, prior_angles), path)
            info.setdefault("modes", []).append(dict(family=family, png=path.relative_to(output).as_posix(),
                                                     modes=summary))
    (output / "nuisance").mkdir(exist_ok=True)
    nuisance_panel(ctx, prior_angles, np.random.default_rng(seed_for(args.seed, folder.name, "nuisance")),
                   output / "nuisance" / f"{folder.name}_inspiration.png")
    (output / "reference").mkdir(exist_ok=True)
    references = reference_panels(folder, output / "reference")
    return info, rows, scenes, references


def write_index(output, infos, rows, scenes, references):
    lines = ["# Synthetic few-shot change families: preview", "",
             "Local review only (contains CT-derived images; do not publish). "
             "All families rendered at the current timepoint; devices omitted locally.", ""]
    for info in infos:
        lines += [f"- **{info['case']}**: cropped CT {info['cropped_shape']}, missing segs {info['missing_segs']}, "
                  f"inferior={info['inferior']}, supervoxels={info['supervoxels']}, setup {info['setup_seconds']} s, "
                  f"angles current {np.round(info['current_angles'], 1).tolist()} / prior "
                  f"{np.round(info['prior_angles'], 1).tolist()}"]
    lines += ["", "Contact sheets: " + ", ".join(f"[{path.name}]({path.name})"
                                                for path in sorted(output.glob("contact_*.png"))), ""]
    for tier in ("T1", "T2", "T2near"):
        tier_rows = [row for row in rows if row["tier"] == tier]
        if not tier_rows:
            continue
        lines += [f"## {tier}", "", "| family | case | sample | realized | GT sign | area>0.001 | area>0.05 | max|GT| | s | panel |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for row in tier_rows:
            if not row["produced"]:
                lines.append(f"| {row['family']} | {row['case']} | {row['sample']} | **not produced** | | | | | | |")
                continue
            lines.append(f"| {row['family']} | {row['case']} | {row['sample']} | {'yes' if row['realized'] else '**no**'} "
                         f"| {row['gt_sign']} | {row['area_001']} | {row['area_005']} | {row['max_abs']:.3f} "
                         f"| {row['seconds']} | [png]({row['png']}) |")
        lines.append("")
    lines += ["## Composed scenes (selective episode target vs distractors)", ""]
    lines += [f"- [{scene['png']}]({scene['png']}): " + "; ".join(
        f"{c['family']}@{c['timepoint']} {c['gt_sign']} {c['area_001']}px" for c in scene["changes"])
        for scene in scenes]
    lines += ["", "## Timepoint modes (appear / resolve / progress / regress / stable)", ""]
    lines += [f"- [{item['png']}]({item['png']}) {info['case']} {item['family']}: " + "; ".join(
        f"{mode['mode']} {mode['gt_sign']} {mode['area_001']}px" for mode in item["modes"])
        for info in infos for item in info.get("modes", [])]
    lines += ["", "## Nuisance", ""] + [f"- [{path.name}](nuisance/{path.name})"
                                        for path in sorted((output / "nuisance").glob("*.png"))]
    lines += ["", "## Existing synthetic pathology pairs (same CTs, for comparison)", ""]
    lines += [f"- [{item['pair']}](reference/{item['png']}): {item['anomaly']} / {item['direction']}" for item in references]
    failures = sorted({row["family"] for row in rows if not row["produced"] or not row.get("realized")})
    lines += ["", f"## Families needing attention ({len(failures)})", "", ", ".join(failures) or "none"]
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cases_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tiers", nargs="+", default=["T1", "T2", "T2near"])
    parser.add_argument("--families", nargs="*", help="Substring filters, e.g. heart nodule t1_grow")
    parser.add_argument("--mode_families", nargs="*",
                        default=["t1_density@lungs", "nodule", "heart_resize", "diaphragm_shift", "mediastinal_shift"],
                        help="Families shown in every timepoint mode (modes/ panels); empty list disables")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--n_supervoxels", type=int, default=1500)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    families = selected_families(args.tiers, args.families)
    infos, rows, scenes, references = [], [], [], []
    for folder in case_folders(args.cases_root):
        info, case_rows, case_scenes, case_references = preview_case(folder, args, output, families)
        infos.append(info)
        rows += case_rows
        scenes += case_scenes
        references += case_references
    (output / "stats.json").write_text(json.dumps(dict(cases=infos, families=rows, scenes=scenes,
                                                       references=references), indent=1) + "\n", encoding="utf-8")
    write_index(output, infos, rows, scenes, references)
    print(f"Preview written to {output / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
