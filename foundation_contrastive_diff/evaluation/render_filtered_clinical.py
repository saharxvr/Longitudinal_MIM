"""Render both CE models before/after exploratory filtering against clinicians."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import find_objects, label

from evaluation.evaluate_clinical import PHYSICIANS, clinical_tools, discover_cases, load_case
from evaluation.sweep_clinical_postprocessing import STRUCTURE, masks_for, match_counts, f1


ROOT = Path(__file__).resolve().parents[1]


def render(models, sweep, destination):
    destination.mkdir(parents=True, exist_ok=False)
    tools = clinical_tools()
    cases = discover_cases(ROOT.parent / "python_files/annotation tool", 100)
    verifications = [json.loads((model / "verification.json").read_text()) for model in models]
    with (sweep / "per_clinician.csv").open(newline="") as stream:
        expected = {(row["model"], row["clinician"]): row for row in csv.DictReader(stream)
                    if row["method"] == "components_1000" and float(row["threshold"]) == .75}
    totals = np.zeros((2, 5, 2, 2), dtype=int)
    cards = []
    for number, case in enumerate(cases, 1):
        prior, current, _, annotations = load_case(case, tools)
        figure, axes = plt.subplots(3, 5, figsize=(20, 12))

        def scan(axis, image, title, signed=None):
            axis.imshow(image.T, cmap="gray")
            if signed is not None:
                axis.imshow(np.ma.masked_where(signed.T == 0, signed.T),
                            cmap="bwr", vmin=-1, vmax=1, alpha=.65)
            axis.set_title(title, fontsize=10)
            axis.axis("off")

        scan(axes[0, 0], prior, "Prior")
        scan(axes[0, 1], current, "Current")
        votes = np.asarray(list(annotations.values())).sum(0)
        consensus = (votes[0] >= 3).astype(float) - (votes[1] >= 3).astype(float)
        scan(axes[0, 2], current, "Clinician consensus (3/5)", consensus)
        for model_index, (model, verification, model_name) in enumerate(
                zip(models, verifications, ("Original CE", "Augmented CE"))):
            path = model / case["pair_id"] / "output.nii.gz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != verification["prediction_sha256"][case["pair_id"]]:
                raise ValueError(f"Prediction hash mismatch: {path}")
            scores = np.asarray(nib.load(path).dataobj, dtype=np.float32).squeeze()
            if scores.shape != current.shape:
                raise ValueError("Prediction/current geometry mismatch")
            for doctor, annotation in case["annotations"].items():
                payload = Path(annotation).read_bytes()
                expected_hash = verification["annotation_sha256"][f"{case['pair_id']}/{doctor}"]
                if expected_hash not in (hashlib.sha256(payload).hexdigest(),
                                         hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()):
                    raise ValueError(f"Annotation hash mismatch: {annotation}")
            raw = np.where(np.abs(scores) > .05, scores, 0)
            scan(axes[0, 3 + model_index], current,
                 f"{model_name}: continuous scores\n(display hides |score| <= 0.05)", raw)
            threshold = masks_for(scores, "raw", .75)
            filtered = masks_for(scores, "components_1000", .75)
            binary = threshold[0].astype(float) - threshold[1].astype(float)
            kept = filtered[0].astype(float) - filtered[1].astype(float)
            removed = binary - kept
            scan(axes[1, model_index * 2], current, f"{model_name}: threshold 0.75", binary)
            scan(axes[1, model_index * 2 + 1], current,
                 f"{model_name}: FILTERED\narea={np.mean(kept != 0):.1%}", kept)
            for reader, doctor in enumerate(PHYSICIANS):
                for sign in (0, 1):
                    reference, _ = label(annotations[doctor][sign], STRUCTURE)
                    prediction, count = label(filtered[sign], STRUCTURE)
                    totals[model_index, reader, sign] += match_counts(
                        reference, find_objects(reference), prediction, count)
            if model_index == 0:
                removed_original = removed
            else:
                removed_augmented = removed
        # The fifth middle panel distinguishes removed areas from retained predictions.
        removed_union = (removed_original != 0).astype(float) - (removed_augmented != 0).astype(float)
        scan(axes[1, 4], current,
             "Removed pixels: red=original only\nblue=augmented only; shared shown yellow",
             removed_union)
        shared = (removed_original != 0) & (removed_augmented != 0)
        axes[1, 4].imshow(np.ma.masked_where(~shared.T, np.ones_like(shared.T)),
                         cmap=matplotlib.colors.ListedColormap(["yellow"]), alpha=.65)
        for axis, doctor in zip(axes[2], PHYSICIANS):
            positive, negative = annotations[doctor]
            signed = positive.astype(float) - negative.astype(float)
            scan(axis, current, doctor, signed)
            if (positive & negative).any():
                axis.contour((positive & negative).T, levels=[.5], colors="black", linewidths=.7)
        figure.suptitle(
            f"{case['pair_id']} | Red: positive change; blue: negative change (except removed panel)\n"
            "Threshold 0.75; keep sign-specific components >1000 equivalent 512x512 pixels. "
            "Exploratory clinical tuning; small true changes may be removed.",
            fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .94))
        name = f"pair{number:03d}.jpg"
        figure.savefig(destination / name, dpi=110, pil_kwargs={"quality": 90})
        plt.close(figure)
        cards.append(f'<section id="pair{number}"><h2>Pair {number}</h2>'
                     f'<a href="{name}"><img loading="lazy" src="{name}" alt="Pair {number}"></a></section>')
        if number % 20 == 0:
            print(f"[gallery] {number}/100", flush=True)
    rows = []
    for index, model in enumerate(models):
        for reader, doctor in enumerate(PHYSICIANS):
            values = dict(f1_positive=f1(*totals[index, reader, 0]),
                          f1_negative=f1(*totals[index, reader, 1]),
                          f1_all=f1(*totals[index, reader].sum(0)))
            for key, value in values.items():
                if not np.isclose(value, float(expected[model.name, doctor][key]), rtol=0, atol=1e-12):
                    raise ValueError(f"Rendered-mask F1 mismatch: {model.name}/{doctor}/{key}")
            rows.append(dict(model=model.name, clinician=doctor, **values))
    (destination / "verification.json").write_text(json.dumps(dict(
        pairs=100, rendered_mask_f1_reproduced=True, scores=rows,
        method="components_1000", threshold=.75), indent=2))
    navigation = " ".join(f'<a href="#pair{number}">{number}</a>' for number in range(1, 101))
    (destination / "index.html").write_text(
        '<!doctype html><html><meta charset="utf-8"><title>Filtered clinical heatmaps</title>'
        '<style>body{font-family:Arial;margin:24px;background:#eee}img{width:100%;max-width:2200px}'
        'section{margin:30px 0}a{margin-right:8px}</style>'
        '<h1>Original and augmented CE: filtered heatmaps vs clinicians</h1>'
        '<p>100 cases; threshold 0.75; sign-specific component cutoff 1000 equivalent pixels. '
        'Exploratory selection on these same cases, not independent test results.</p>'
        f'<nav>{navigation}</nav>' + "".join(cards) + '</html>', encoding="utf-8")
    print(f"[complete] {destination / 'index.html'}; all 30 rendered-mask F1 values reproduced", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs=2, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.models, args.sweep, args.out_dir)
