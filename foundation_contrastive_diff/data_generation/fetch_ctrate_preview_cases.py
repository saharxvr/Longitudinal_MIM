"""Download 1-2 public CT-RATE chest CTs and build a local preview bundle (no campus needed).

Requires your own Hugging Face access to the gated dataset (accept the terms on
https://huggingface.co/datasets/ibrahimhamamci/CT-RATE, then `hf auth login`). The CT-RATE
terms forbid redistribution: keep the output OUTSIDE the git repository.

Per case it writes the same layout as export_preview_cases.py:
    <output>/<case>/ct.nii.gz                 HU (int16), canonical RAS orientation
    <output>/<case>/<lobe>.nii.gz, heart       from CT-RATE ts_seg/ts_total (v1 or v2 label map, auto-detected)
    <output>/<case>/bronchi.nii.gz, lung_vessels.nii.gz   TotalSegmentator `lung_vessels` task, run locally
    <output>/cases.json                       provenance (source files, revision, HU/spacing fixes, label map)

    python -m data_generation.fetch_ctrate_preview_cases --output C:/Users/<you>/thesis/fewshot_preview_cases \
        --totalseg C:/Users/<you>/thesis/totalseg_venv/Scripts/TotalSegmentator.exe --count 2
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = "ibrahimhamamci/CT-RATE"
VOLUME = "dataset/train_fixed/{patient}/{scan}/{name}"
SEGMENTATION = "dataset/ts_seg/ts_total/train_fixed/{patient}/{scan}/{name}"
METADATA = "dataset/metadata/train_metadata.csv"
NO_CHEST = "dataset/metadata/no_chest_train.txt"
LABEL_MAPS = {
    "total_v2": dict(upper_left_lobe=[10], lower_left_lobe=[11], upper_right_lobe=[12], middle_right_lobe=[13],
                     lower_right_lobe=[14], trachea=[16], heart=[51]),
    "total_v1": dict(upper_left_lobe=[13], lower_left_lobe=[14], upper_right_lobe=[15], middle_right_lobe=[16],
                     lower_right_lobe=[17], trachea=[43], heart=[44, 45, 46, 47, 48]),
}
LOBES = ("upper_right_lobe", "middle_right_lobe", "lower_right_lobe", "upper_left_lobe", "lower_left_lobe")


def volume_paths(name):
    """train_12_a_1.nii.gz -> (volume path, ts_total path) inside the CT-RATE repo."""
    stem = name.removesuffix(".nii.gz")
    patient, scan = "_".join(stem.split("_")[:2]), "_".join(stem.split("_")[:3])
    return (VOLUME.format(patient=patient, scan=scan, name=name),
            SEGMENTATION.format(patient=patient, scan=scan, name=name))


def column(row, *needles):
    for key, value in row.items():
        if all(needle in key.lower().replace(" ", "") for needle in needles):
            return value
    return None


def number(value, default=None):
    try:
        text = str(value).strip().strip("[]()").split(",")[0]
        return float(text)
    except (TypeError, ValueError):
        return default


def choose(rows, no_chest, count, seed, explicit):
    by_name = {row.get("VolumeName") or column(row, "volume", "name"): row for row in rows}
    if explicit:
        missing = [name for name in explicit if name not in by_name]
        if missing:
            raise SystemExit(f"Not in train metadata: {missing}")
        return [(name, by_name[name]) for name in explicit]
    candidates = []
    for name, row in by_name.items():
        z_spacing = number(column(row, "zspacing")) or number(column(row, "slicethickness"))
        slices = number(column(row, "numberofslices"))
        if (name and name.endswith("_1.nii.gz") and name not in no_chest and z_spacing
                and 0.8 <= z_spacing <= 1.5 and (slices is None or 150 <= slices <= 600)):
            candidates.append(name)
    rng = np.random.default_rng(seed)
    chosen, patients = [], set()
    for index in rng.permutation(len(candidates)):
        name = candidates[index]
        patient = "_".join(name.split("_")[:2])
        if patient not in patients:
            chosen.append((name, by_name[name]))
            patients.add(patient)
        if len(chosen) == count:
            break
    return chosen


def to_hu(image, row):
    """Apply CT-RATE rescale metadata when the stored values are not already HU; fix spacing."""
    import nibabel as nib

    data = np.asarray(image.dataobj, dtype=np.float32)
    notes = {}
    if float(np.percentile(data, 1)) > -200:  # raw detector values, no air at about -1000
        slope = number(column(row, "rescaleslope"), 1.0)
        intercept = number(column(row, "rescaleintercept"), -1024.0)
        data = data * slope + intercept
        notes["rescaled"] = dict(slope=slope, intercept=intercept)
    xy, z = number(column(row, "xyspacing")), number(column(row, "zspacing"))
    affine = image.affine.copy()
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(0))
    if xy and z:
        target = np.array([xy, xy, z])
        if np.any(np.abs(zooms - target) / target > 0.02):
            affine[:3, :3] = affine[:3, :3] / zooms * target
            notes["spacing_fixed"] = dict(header=zooms.round(4).tolist(), metadata=target.tolist())
    data = np.clip(data, -1024, 3071).round().astype(np.int16)
    return nib.as_closest_canonical(nib.Nifti1Image(data, affine)), notes


def decode_total(segmentation, ct):
    """Choose the TotalSegmentator label map whose lobes are air-like and heart soft-tissue-like."""
    labels = np.asarray(segmentation.dataobj).astype(np.int16)
    hu = np.asarray(ct.dataobj, np.float32)
    report = {}
    for version, mapping in LABEL_MAPS.items():
        masks = {name: np.isin(labels, ids) for name, ids in mapping.items()}
        lobe_hu = [float(hu[masks[name]].mean()) if masks[name].any() else 0.0 for name in LOBES]
        heart_hu = float(hu[masks["heart"]].mean()) if masks["heart"].any() else -1000.0
        valid = all(value < -500 for value in lobe_hu) and -100 < heart_hu < 200
        report[version] = dict(lobe_mean_hu=[round(value) for value in lobe_hu], heart_mean_hu=round(heart_hu),
                               valid=valid)
        if valid:
            return version, masks, report
    raise ValueError(f"ts_total labels do not match a known TotalSegmentator map: {report}")


def run_lung_vessels(executable, ct_path, device):
    """TotalSegmentator lung_vessels; supports the old (trachea_bronchia/vessels) and new (airways/arteries/veins) outputs."""
    import nibabel as nib

    with tempfile.TemporaryDirectory() as folder:
        command = [str(executable), "-i", str(ct_path), "-o", folder, "-ta", "lung_vessels", "--device", device]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"TotalSegmentator failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")

        def load(*names):
            paths = [Path(folder) / f"{name}.nii.gz" for name in names]
            present = [path for path in paths if path.is_file()]
            if not present:
                return None
            return np.logical_or.reduce([np.asarray(nib.load(str(path)).dataobj) > 0 for path in present])

        old = load("lung_trachea_bronchia"), load("lung_vessels")
        if old[0] is not None and old[1] is not None:
            return dict(bronchi=old[0], lung_vessels=old[1]), "lung_trachea_bronchia / lung_vessels"
        new = load("lung_airways"), load("lung_arteries", "lung_veins")
        if new[0] is None or new[1] is None:
            raise RuntimeError(f"Unexpected lung_vessels outputs: {sorted(p.name for p in Path(folder).glob('*'))}")
        return dict(bronchi=new[0], lung_vessels=new[1]), "lung_airways / lung_arteries+lung_veins"


def save_mask(mask, reference, path):
    import nibabel as nib

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), str(path))


def main(argv=None):
    import nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--totalseg", help="TotalSegmentator executable (separate venv) for bronchi/vessels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--cases", nargs="*", help="Explicit CT-RATE volume names, e.g. train_123_a_1.nii.gz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if (Path(__file__).resolve().parents[2] / ".git").exists() and Path(__file__).resolve().parents[2] in output.parents:
        raise SystemExit("Output is inside the git repository; CT-RATE forbids redistribution. Choose a folder outside it.")
    downloads = output / "_downloads"
    output.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(REPO).sha

    def fetch(path):
        return Path(hf_hub_download(REPO, path, repo_type="dataset", revision=revision, local_dir=downloads))

    with open(fetch(METADATA), newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    no_chest = {line.strip() for line in fetch(NO_CHEST).read_text(encoding="utf-8").splitlines() if line.strip()}
    records = []
    for name, row in choose(rows, no_chest, args.count, args.seed, args.cases):
        case = name.removesuffix(".nii.gz")
        folder = output / case
        if (folder / "lung_vessels.nii.gz").is_file() or (folder / "ct.nii.gz").is_file() and not args.totalseg:
            print(f"[skip] {case} already prepared")
            continue
        volume_path, segmentation_path = volume_paths(name)
        print(f"[download] {volume_path}", flush=True)
        ct, notes = to_hu(nib.load(str(fetch(volume_path))), row)
        segmentation = nib.as_closest_canonical(nib.load(str(fetch(segmentation_path))))
        if segmentation.shape != ct.shape or not np.allclose(segmentation.affine, ct.affine, atol=1e-2):
            notes["segmentation_affine_differs"] = True
            if segmentation.shape != ct.shape:
                raise ValueError(f"{case}: ts_total shape {segmentation.shape} != CT {ct.shape}")
        version, masks, label_report = decode_total(segmentation, ct)
        folder.mkdir(parents=True, exist_ok=True)
        nib.save(ct, str(folder / "ct.nii.gz"))
        for structure in (*LOBES, "heart"):
            save_mask(masks[structure], ct, folder / f"{structure}.nii.gz")
        airways, airway_source = {}, "missing"
        if args.totalseg:
            print(f"[TotalSegmentator lung_vessels] {case} on {args.device} (slow on CPU)", flush=True)
            airways, airway_source = run_lung_vessels(args.totalseg, folder / "ct.nii.gz", args.device)
            for structure, mask in airways.items():
                save_mask(mask, ct, folder / f"{structure}.nii.gz")
                if int(mask.sum()) < 1000:
                    notes[f"{structure}_nearly_empty"] = int(mask.sum())
                    print(f"[warning] {case}: {structure} has only {int(mask.sum())} voxels", flush=True)
        records.append(dict(case=case, patient_key="_".join(case.split("_")[:2]), split="public_preview",
                            source=dict(repo=REPO, revision=revision, volume=volume_path, ts_total=segmentation_path),
                            shape=list(ct.shape), zooms=[round(float(value), 4) for value in ct.header.get_zooms()],
                            conversions=notes, ts_total_map=version, label_check=label_report,
                            bronchi_vessels=f"TotalSegmentator lung_vessels: {airway_source}",
                            note="Public CT-RATE case, not from the fcd_train cohort; preview only"))
        print(f"[done] {case}: map={version}, notes={notes}, voxels heart={int(masks['heart'].sum())}", flush=True)
    path = output / "cases.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    known = {record["case"] for record in records}
    path.write_text(json.dumps([item for item in previous if item["case"] not in known] + records, indent=1) + "\n",
                    encoding="utf-8")
    print(f"Bundle ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
