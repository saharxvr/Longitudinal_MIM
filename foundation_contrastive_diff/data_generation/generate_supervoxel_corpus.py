"""Generate the supervoxel few-shot training corpus from TRAIN-split CTs only.

Output layout (one directory per scene):
    <out>/<case>/scene0007/prior.nii.gz      raw projection, same convention as fcd_train
    <out>/<case>/scene0007/current.nii.gz
    <out>/<case>/scene0007/changes.npz        maps: float16 [K,512,512] per-change signed GT
    <out>/<case>/scene0007/ellipses.json      per-change annotation-tool ellipses (coarse labels)
    <out>/<case>/scene0007/params.json        scene metadata (manifest record)
then ``--build_manifest`` writes <out>/manifest_scenes.jsonl.

Each change record's ``timepoint`` is its mode (supervoxel_tasks.TIMEPOINT_MODES): appear, resolve,
progress, regress or stable. Stable changes have an all-zero map (realized=false), so they are
never episode targets; they teach the model to ignore findings present in both images.

Scenes are deterministic per (case, scene index) and resumable. T2-near families are only
generated with ``--tiers T1 T2 T2near`` (use a separate --output for the ablation corpus).

Example (one campus GPU node, slice 0 of 8):
    python -m data_generation.generate_supervoxel_corpus \
        --source_manifest /cs/labs/josko/sahar_aharon/fcd_train/manifest_split.jsonl \
        --output /cs/labs/josko/sahar_aharon/fcd_fewshot_supervoxel \
        --scenes_per_ct 8 --slice_index 0 --num_slices 8 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from data_generation import supervoxel_render as render
from data_generation import supervoxel_tasks as tasks

ROOT = Path(__file__).resolve().parents[1]
CT_ENTITIES = ROOT.parent / "python_files" / "CT_entities"
GENERATOR_VERSION = 2  # 2: timepoint modes (appear / resolve / progress / regress / stable)
SEG_ROOT = "/cs/labs/josko/itamar_sab/LongitudinalCXRAnalysis/CT_scans"
SEG_TEMPLATES = {
    "lungs": f"{SEG_ROOT}/scans_segs/{{case}}_seg.nii.gz",
    "heart": f"{SEG_ROOT}/scans_heart_segs/{{case}}_heart_seg.nii.gz",
    "bronchi": f"{SEG_ROOT}/scans_bronchi_segs/{{case}}_bronchia_seg.nii.gz",
    "lung_vessels": f"{SEG_ROOT}/scans_vessels_segs/{{case}}_vessels_seg.nii.gz",
    **{lobe: f"{SEG_ROOT}/scans_lobes_segs/{{case}}_lung_{name}.nii.gz" for lobe, name in (
        ("upper_right_lobe", "upper_lobe_right"), ("middle_right_lobe", "middle_lobe_right"),
        ("lower_right_lobe", "lower_lobe_right"), ("upper_left_lobe", "upper_lobe_left"),
        ("lower_left_lobe", "lower_lobe_left"))},
}
REQUIRED_SEGS = ("upper_right_lobe", "middle_right_lobe", "lower_right_lobe",
                 "upper_left_lobe", "lower_left_lobe")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: sha256(ROOT / "data_generation" / name) for name in (
        "supervoxel_tasks.py", "supervoxel_render.py", "generate_supervoxel_corpus.py")}


def train_cases(source_manifest):
    """Unique (case, ct path, patient) from the TRAIN split of the existing pair manifest."""
    cases, splits = {}, {}
    with open(source_manifest, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            case, patient = record.get("case"), record.get("patient_key") or record.get("case")
            splits.setdefault(patient, set()).add(record.get("split"))
            if record.get("split") == "train" and case and record.get("ct_source_path"):
                cases.setdefault(case, (record["ct_source_path"], patient))
    leaked = sorted(patient for patient, values in splits.items() if len(values) > 1)
    if leaked:
        raise ValueError(f"Source manifest has patients in several splits: {leaked[:5]}")
    return [(case, path, patient) for case, (path, patient) in sorted(cases.items())]


def load_volume(path):
    """Same axis convention as DRR_generator.load_scan_and_seg: transpose (2,0,1), flip z."""
    import nibabel as nib

    image = nib.load(path)
    data = np.asarray(image.dataobj, dtype=np.float32)
    return np.ascontiguousarray(np.flip(np.transpose(data, (2, 0, 1)), 0)), image.affine


def crop_like_generator(ct, segs):
    """crop_according_to_seg(lungs, ext=15, h_low_ext=lungs_h//5, tight_y=False)."""
    lungs = segs["lungs"]
    coordinates = np.argwhere(lungs)
    low, high = coordinates.min(0), coordinates.max(0)
    height = int(high[0] - low[0])
    box = (slice(max(int(low[0]) - 15, 0), min(int(high[0]) + height // 5, ct.shape[0])),
           slice(max(int(low[1]) - 15, 0), min(int(high[1]) + 15, ct.shape[1])),
           slice(None))
    return ct[box], {name: mask[box] for name, mask in segs.items()}


def spacing_from_affine(affine):
    """Voxel spacing (mm) in the (z, x, y) axis order produced by load_volume."""
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(0))
    return (float(zooms[2]), float(zooms[0]), float(zooms[1]))


def load_case(case, ct_path, templates):
    ct, affine = load_volume(ct_path)
    segs, missing = {}, []
    for name, template in templates.items():
        path = template.format(case=case)
        if os.path.isfile(path):
            segs[name] = load_volume(path)[0] > 0.5
        else:
            missing.append(name)
    if any(name in missing for name in REQUIRED_SEGS):
        raise FileNotFoundError(f"{case}: missing lobe segmentations {missing}")
    segs["lungs"] = np.logical_or.reduce([segs[name] for name in REQUIRED_SEGS])
    ct, segs = crop_like_generator(ct, segs)
    return ct, segs, affine, missing


def save_nifti(array, affine, path):
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.ascontiguousarray(array.astype(np.float32)), affine), str(path))


def make_devices_hook(probability, lungs, rng_seed):
    if probability <= 0:
        return None
    sys.path.insert(0, str(CT_ENTITIES))
    sys.path.insert(0, str(CT_ENTITIES.parent))
    import torch
    from External_Devices import ExternalDevices  # campus DRR environment (gryds, devices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lungs_tensor = torch.from_numpy(lungs.astype(np.float32)).to(device)
    generator = random.Random(rng_seed)

    def hook(base):
        if generator.random() >= probability:
            return base, base, []
        random.seed(generator.randrange(2 ** 31))
        torch.manual_seed(generator.randrange(2 ** 31))
        volume = (base.to(device, torch.float32) if torch.is_tensor(base)
                  else torch.from_numpy(np.array(base, np.float32)).to(device))
        output = ExternalDevices.add_to_CT_pair(
            [volume.clone(), volume.clone()], [{"lungs": lungs_tensor}, {"lungs": lungs_tensor}],
            registrated_prior=volume.clone(), log_params=False)
        prior, current = output["scans"]
        return prior.float(), current.float(), ["ExternalDevices"]

    return hook


class DonorBank:
    """HU cubes from previously processed TRAIN CTs (3D CutPaste donors)."""

    def __init__(self, size=64, seed=0):
        self.items, self.size, self.rng = [], size, np.random.default_rng(seed)

    def add(self, ctx, count=8, edge_mm=None):
        edge_mm = edge_mm or max(20.0, 0.15 * ctx.scale)
        points = ctx.full_points(ctx.compartments["body"])
        for _ in range(count):
            center = points[self.rng.integers(len(points))]
            box = tasks.box_around(center, ctx.vox(edge_mm / 2), ctx.shape)
            self.items.append(ctx.ct[box].copy())
        self.items = self.items[-self.size:]

    def __call__(self, rng):
        return self.items[int(rng.integers(len(self.items)))] if self.items else None


def scene_seed(case, index):
    return zlib.crc32(f"{case}/scene{index:04d}".encode()) & 0x7FFFFFFF


def generate_scene(ctx, case, patient, index, args, affine, devices_hook, donors, out_dir):
    timing = {}
    started = time.time()
    rng = np.random.default_rng(scene_seed(case, index))
    empty = rng.random() < args.empty_scene_prob
    changes = [] if empty else tasks.compose_scene(
        ctx, rng, tiers=tuple(args.tiers), n_changes=tuple(args.changes_per_scene), donor_fn=donors)
    nuisance, nuisance_params = (None, None)
    if rng.random() < args.inspiration_prob:
        nuisance, nuisance_params = tasks.inspiration_nuisance(ctx, rng)
    timing["changes"] = time.time() - started
    started = time.time()
    scene = render.render_scene(ctx, changes, rng, device=args.device,
                                devices_hook=devices_hook, prior_nuisance=nuisance)
    timing["render"] = time.time() - started
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nifti(scene["prior"].T, affine, out_dir / "prior.nii.gz")
    save_nifti(scene["current"].T, affine, out_dir / "current.nii.gz")
    maps = np.stack([item.T for item in scene["maps"]]) if scene["maps"] else np.zeros((0, 512, 512))
    np.savez_compressed(out_dir / "changes.npz", maps=maps.astype(np.float16))
    ellipses, records = [], []
    for position, ((change, mode), signed) in enumerate(zip(changes, maps)):
        statistics = render.realized_statistics(signed)
        ellipses.append(render.fit_ellipses(signed))
        records.append(dict(index=position, family=change.family, tier=change.tier,
                            compartment=change.compartment, effect=change.effect,
                            timepoint=mode, gt_sign=render.gt_sign(statistics),
                            realized=statistics["area_001"] > render.MIN_COMPONENT,
                            analogue=tasks.FAMILIES[change.family]["analogue"],
                            params=_jsonable(change.params), **statistics))
    (out_dir / "ellipses.json").write_text(json.dumps(ellipses) + "\n", encoding="utf-8")
    record = dict(schema="fewshot_scene_v2", generator_version=GENERATOR_VERSION,
                  scene_id=f"{case}/scene{index:04d}", case=case, patient_key=patient,
                  split="train", corpus_tiers=list(args.tiers), seed=scene_seed(case, index),
                  changes=records, num_changes=len(records),
                  realized_changes=sum(item["realized"] for item in records),
                  nuisance=dict(inspiration=nuisance_params, devices=scene["device_names"],
                                current_angles_deg=scene["current_angles"],
                                prior_angles_deg=scene["prior_angles"]),
                  projection_scale=scene["scale"],
                  files=dict(prior="prior.nii.gz", current="current.nii.gz",
                             changes="changes.npz", ellipses="ellipses.json"))
    timing["save"] = time.time() - started
    record["timing_s"] = {key: round(value, 2) for key, value in timing.items()}
    (out_dir / "params.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def build_manifest(output):
    output = Path(output)
    records = []
    for params in sorted(output.glob("*/scene*/params.json")):
        record = json.loads(params.read_text(encoding="utf-8"))
        files = [params.parent / name for name in record["files"].values()]
        if all(path.is_file() for path in files):
            record["scene_dir"] = params.parent.relative_to(output).as_posix()
            records.append(record)
    path = output / "manifest_scenes.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = dict(scenes=len(records), cases=len({r["case"] for r in records}),
                   patients=len({r["patient_key"] for r in records}),
                   changes=sum(r["num_changes"] for r in records),
                   realized=sum(r["realized_changes"] for r in records),
                   by_family={}, by_tier={})
    for record in records:
        for change in record["changes"]:
            summary["by_family"][change["family"]] = summary["by_family"].get(change["family"], 0) + 1
            summary["by_tier"][change["tier"]] = summary["by_tier"].get(change["tier"], 0) + 1
    (output / "manifest_scenes_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                          encoding="utf-8")
    return summary


def bundle_cases(root):
    """(case, ct path, patient) of a local preview bundle (<root>/<case>/ct.nii.gz + <structure>.nii.gz)."""
    return [(folder.name, str(folder / "ct.nii.gz"), folder.name) for folder in sorted(Path(root).iterdir())
            if (folder / "ct.nii.gz").is_file()]


def all_cases(args):
    cases_root = getattr(args, "cases_root", None)
    return bundle_cases(cases_root) if cases_root else train_cases(args.source_manifest)


def seg_templates(args):
    if getattr(args, "cases_root", None):
        return {name: str(Path(args.cases_root) / "{case}" / f"{name}.nii.gz") for name in SEG_TEMPLATES}
    return {name: template.replace(SEG_ROOT, args.seg_root) for name, template in SEG_TEMPLATES.items()}


def _cpu_count():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _memory_gb():
    """(peak RSS of this process, total RAM) in GB; None where unavailable (e.g. Windows)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # KB on Linux
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        return round(peak, 2), round(total, 1)
    except (ImportError, ValueError, OSError, AttributeError):
        return None, None


def speed_summary(setups, timings, elapsed, device):
    import torch

    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    stages = {key: round(float(np.mean([timing[key] for timing in timings])), 2)
              for key in ("changes", "render", "save")} if timings else {}
    peak_rss, total_ram = _memory_gb()
    return dict(cases=len(setups), scenes=len(timings), elapsed_s=round(elapsed, 1),
                setup_s_per_ct=round(float(np.mean(setups)), 2) if setups else None,
                scene_s=stages, scene_s_total=round(sum(stages.values()), 2) if stages else None,
                scenes_per_hour=round(3600 * len(timings) / elapsed, 1) if elapsed > 0 else None,
                gpu=torch.cuda.get_device_name(0) if cuda else None,
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None,
                gpu_total_gb=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1) if cuda else None,
                ram_peak_gb=peak_rss, ram_total_gb=total_ram,
                cpu_count=_cpu_count(), omp_threads=os.environ.get("OMP_NUM_THREADS"))


def projection(summary, n_cases, scenes_per_ct):
    """Full-corpus time from a single-worker benchmark (ideal parallel scaling) and a worker suggestion."""
    per_ct = (summary["setup_s_per_ct"] or 0.0) + scenes_per_ct * (summary["scene_s_total"] or 0.0)
    hours = n_cases * per_ct / 3600.0
    limits = dict(cpu=max(1, summary["cpu_count"] - 2))
    if summary.get("gpu_peak_gb"):
        # + ~0.6 GB CUDA context per process; keep 15% headroom for device insertion spikes.
        limits["gpu"] = max(1, int(0.85 * summary["gpu_total_gb"] // (summary["gpu_peak_gb"] + 0.6)))
    if summary.get("ram_peak_gb"):
        limits["ram"] = max(1, int(0.8 * summary["ram_total_gb"] // summary["ram_peak_gb"]))
    workers = min(limits.values())
    return dict(train_cases=n_cases, scenes_per_ct=scenes_per_ct, scenes=n_cases * scenes_per_ct,
                single_worker_hours=round(hours, 1), worker_limits=limits, suggested_workers=workers,
                hours_with_suggested_workers=round(hours / workers, 1),
                note="Ideal scaling; CPU/memory contention makes the real rate lower. Launch, wait ~15 min, "
                     "then use --status for the measured rate and ETA.")


def status(output):
    output = Path(output)
    config = json.loads((output / "generator_config.json").read_text(encoding="utf-8"))
    stored = argparse.Namespace(**config["args"])
    n_cases = len(all_cases(stored))
    expected = n_cases * stored.scenes_per_ct
    now = time.time()
    times = [path.stat().st_mtime for path in output.glob("*/scene*/params.json")]
    recent = [stamp for stamp in times if now - stamp < 1800]
    rate = len(recent) * 2.0  # scenes per hour over the last 30 minutes
    errors = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in output.glob("errors_slice*.jsonl"))
    return dict(done=len(times), expected=expected, percent=round(100.0 * len(times) / max(1, expected), 1),
                scenes_per_hour_last_30min=rate,
                eta_hours=round((expected - len(times)) / rate, 1) if rate else None,
                cases_with_errors=errors, generator_version=config.get("version"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source_manifest", required=False)
    parser.add_argument("--cases_root", help="Local bundle (<case>/ct.nii.gz + <structure>.nii.gz) instead of "
                                             "--source_manifest, e.g. the CT-RATE preview cases")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes_per_ct", type=int, default=8)
    parser.add_argument("--tiers", nargs="+", default=["T1", "T2"], choices=["T1", "T2", "T2near"])
    parser.add_argument("--changes_per_scene", nargs=2, type=int, default=[1, 3])
    parser.add_argument("--empty_scene_prob", type=float, default=0.1)
    parser.add_argument("--inspiration_prob", type=float, default=0.3)
    parser.add_argument("--devices_prob", type=float, default=0.8)
    parser.add_argument("--n_supervoxels", type=int, default=1500)
    parser.add_argument("--slice_index", type=int, default=0)
    parser.add_argument("--num_slices", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seg_root", default=SEG_ROOT)
    parser.add_argument("--build_manifest", action="store_true")
    parser.add_argument("--status", action="store_true", help="Progress, measured rate and ETA of --output")
    parser.add_argument("--benchmark", action="store_true",
                        help="Time a few CTs (default --max_cases 3) and project the full corpus; use a scratch --output")
    parser.add_argument("--target_scenes_per_ct", type=int, default=8, help="Corpus size for the --benchmark projection")
    args = parser.parse_args(argv)
    if not (args.build_manifest or args.status) and not (args.source_manifest or args.cases_root):
        parser.error("--source_manifest or --cases_root is required for generation")
    if not 0 <= args.slice_index < args.num_slices:
        parser.error("Invalid slice")
    if args.benchmark and not args.max_cases:
        args.max_cases = 3
    return args


def check_environment(args):
    """Fail at startup (not once per CT) when a dependency or the GPU is missing."""
    import torch

    try:
        import skimage.segmentation  # noqa: F401  (SLIC supervoxels)
    except ImportError as error:
        raise SystemExit(f"Missing {error.name}: pip install scikit-image") from error
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch sees no GPU (check LD_LIBRARY_PATH / CUDA_VISIBLE_DEVICES)")
    if args.devices_prob > 0:
        sys.path.insert(0, str(CT_ENTITIES))
        sys.path.insert(0, str(CT_ENTITIES.parent))
        try:
            from External_Devices import ExternalDevices  # noqa: F401
        except ImportError as error:
            raise SystemExit(f"Device nuisance needs the DRR generator dependencies (missing {error.name}): "
                             "pip install kornia gryds in this venv, use FCD_VENV=<DRR venv>, "
                             "or pass --devices_prob 0") from error


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if args.build_manifest:
        print(json.dumps(build_manifest(output), indent=1))
        return 0
    if args.status:
        print(json.dumps(status(output), indent=1))
        return 0
    check_environment(args)
    templates = seg_templates(args)
    every = all_cases(args)
    cases = every[args.slice_index::args.num_slices]
    if args.max_cases:
        cases = cases[:args.max_cases]
    output.mkdir(parents=True, exist_ok=True)
    (output / "generator_config.json").write_text(json.dumps(dict(
        args=vars(args), code_sha256=code_hashes(), version=GENERATOR_VERSION), indent=1) + "\n",
        encoding="utf-8")
    donors = DonorBank(seed=args.slice_index)
    started = time.time()
    setups, timings = [], []
    for number, (case, ct_path, patient) in enumerate(cases):
        pending = [index for index in range(args.scenes_per_ct)
                   if not (output / case / f"scene{index:04d}" / "params.json").is_file()]
        if not pending:
            continue
        try:
            setup_started = time.time()
            ct, segs, affine, missing = load_case(case, ct_path, templates)
            cache = output / "_supervoxels" / f"{case}_mm.npy"
            supervoxels = np.load(cache) if cache.is_file() else None
            ctx = tasks.CTContext(ct, segs, spacing=spacing_from_affine(affine), supervoxels=supervoxels,
                                  n_supervoxels=args.n_supervoxels, seed=scene_seed(case, 0))
            if supervoxels is None or supervoxels is not ctx.supervoxels:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, ctx.supervoxels)
            hook = make_devices_hook(args.devices_prob, ctx.segs["lungs"], scene_seed(case, 9999))
            setups.append(time.time() - setup_started)
            for index in pending:
                record = generate_scene(ctx, case, patient, index, args, affine, hook, donors,
                                        output / case / f"scene{index:04d}")
                timings.append(record["timing_s"])
            donors.add(ctx)
            print(f"[{number + 1}/{len(cases)}] {case}: {len(pending)} scenes, setup {setups[-1]:.0f}s, "
                  f"missing segs={missing}, elapsed {time.time() - started:.0f}s, "
                  f"{3600 * len(timings) / (time.time() - started):.0f} scenes/h", flush=True)
            del ctx, hook
        except Exception as error:  # keep going; record the failure
            print(f"[error] {case}: {type(error).__name__}: {error}", flush=True)
            with (output / f"errors_slice{args.slice_index}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(case=case, error=repr(error))) + "\n")
    summary = speed_summary(setups, timings, time.time() - started, args.device)
    (output / "logs").mkdir(exist_ok=True)
    (output / "logs" / f"speed_slice{args.slice_index}.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                                         encoding="utf-8")
    print("[speed] " + json.dumps(summary), flush=True)
    if args.benchmark and timings:
        print("[projection] " + json.dumps(projection(summary, len(every), args.target_scenes_per_ct), indent=1),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
