"""Export compact, prediction-only masks for local A/B/Itamar collages."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from evaluation.evaluate_clinical import clinical_tools, discover_cases
from evaluation.evaluate_fusion_ov import filtered_scores


STUDY = Path(__file__).resolve().parents[1]
OV = STUDY / "results/clinical100_fusion_ov_thr075_cc250_20261022_190306_3338552"
REFERENCE = STUDY / "results/clinical100_itamar_prediction_hashes.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(run, clinical, baseline_root, destination, ov=OV, reference=REFERENCE, count=100):
    if destination.exists():
        raise ValueError("Use a new export directory")
    cases = discover_cases(clinical, count)
    if len(cases) != count:
        raise ValueError(f"Expected {count} clinical pairs")
    expected_baseline = read(reference)
    expected = {
        arm: read(ov / f"arm_{arm}/input_provenance.json")["prediction_sha256"]
        for arm in ("A", "B")
    }
    sources = {}
    for arm in expected:
        for folder in sorted(run.glob("clinical100_*/arm_" + arm), reverse=True):
            paths = {c["pair_id"]: folder / c["pair_id"] / "output.nii.gz" for c in cases}
            if all(p.is_file() for p in paths.values()) and all(
                    sha(p) == expected[arm][pair] for pair, p in paths.items()):
                sources[arm] = paths
                break
        if arm not in sources:
            raise FileNotFoundError(f"No {arm} inference matches published OV hashes under {run}")

    wanted = {expected_baseline[c["pair_id"]] for c in cases}
    baseline = {}
    print(f"Finding {count} historical baseline predictions under {baseline_root}", flush=True)
    for path in baseline_root.rglob("output.nii.gz"):
        crop = path.parent / "crop_info.json"
        if not crop.is_file():
            continue
        digest = sha(path)
        if digest in wanted:
            baseline[digest] = path
        if wanted <= baseline.keys():
            break
    if not wanted <= baseline.keys():
        raise FileNotFoundError(
            f"Missing {len(wanted - baseline.keys())} historical baseline predictions "
            f"with crop_info.json under {baseline_root}")

    tools = clinical_tools()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="prediction_export_", dir=destination.parent) as temporary:
        output = Path(temporary) / "bundle"
        output.mkdir()
        provenance = {}
        for index, case in enumerate(cases, 1):
            pair = case["pair_id"]
            current = nib.load(case["current"])
            shape = tuple(size for size in current.shape if size != 1)
            masks = {}
            hashes = {}
            for arm in ("A", "B"):
                path = sources[arm][pair]
                image = nib.load(path)
                scores = np.asarray(image.dataobj, dtype=np.float32).squeeze()
                if scores.shape != shape or not np.allclose(image.affine, current.affine):
                    raise ValueError(f"Geometry mismatch: {pair}/{arm}")
                masks[arm] = filtered_scores(scores).astype(np.int8)
                hashes[arm] = expected[arm][pair]
            path = baseline[expected_baseline[pair]]
            crop = path.parent / "crop_info.json"
            restored = tools._restore_model_output(str(path), str(crop), shape)
            if restored.shape != shape or not np.isfinite(restored).all():
                raise ValueError(f"Invalid restored baseline: {pair}")
            masks["Itamar"] = np.sign(restored).astype(np.int8)
            np.savez_compressed(output / f"{pair}.npz", **masks)
            with np.load(output / f"{pair}.npz", allow_pickle=False) as saved:
                if set(saved.files) != set(masks) or any(
                        not np.array_equal(saved[key], masks[key]) for key in masks):
                    raise ValueError(f"Export verification failed: {pair}")
            provenance[pair] = dict(
                shape=list(shape), prediction_sha256={**hashes, "Itamar": expected_baseline[pair]},
                crop_sha256=sha(crop),
                baseline="lmm5 substitute" if pair in ("pair99", "pair100") else "Itamar")
            print(f"[{index}/{count}] exported {pair}", flush=True)
        manifest = dict(
            num_pairs=count, arrays=["A", "B", "Itamar"],
            coordinates="Original current-image NIfTI array space; transpose for display",
            values={"-1": "negative change", "0": "no change", "1": "positive change"},
            fusion_postprocessing="Strict |score| >0.75; sign-specific 8-connected CC >250 "
                                  "equivalent 512x512 pixels",
            baseline_postprocessing="Historical saved cleanup; restored with legacy crop mapping; "
                                    "threshold zero. Not the A/B postprocessing.",
            contains="Prediction masks only; no scans, annotations, weights, or raw scores",
            pairs=provenance)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
        if sum(p.stat().st_size for p in output.iterdir()) >= 100 * 1024**2:
            raise ValueError("Export exceeds 100 MiB; refusing publication")
        output.rename(destination)
    print(f"Verified prediction-only bundle: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(
        "/cs/labs/josko/sahar_aharon/fcd_runs/run_layer_fusion_6_9_12_ce_30"))
    parser.add_argument("--clinical-root", type=Path,
                        default=STUDY.parent / "python_files" / "annotation tool")
    parser.add_argument("--baseline-root", type=Path, default=STUDY.parent / "python_files")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.clinical_root, args.baseline_root, args.out_dir)


if __name__ == "__main__":
    main()
