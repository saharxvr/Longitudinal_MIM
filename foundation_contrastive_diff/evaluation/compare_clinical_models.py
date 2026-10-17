"""Reproduce legacy observer statistics and compare clinical model predictions."""

import argparse
import contextlib
import csv
import hashlib
import json
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.evaluate_clinical import (
    PHYSICIANS, clinical_tools, discover_cases, load_case, score_masks,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equivalent(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int)):
        return bool(np.isclose(left, right, atol=1e-12, rtol=1e-10, equal_nan=True))
    return left == right


def run_legacy(tools, predictions, clinical_root, destination, count, space, threshold):
    tools.struct = tools.STRUCT_3x3
    with destination.with_suffix(".log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            tools.main(str(predictions), str(clinical_root / "Annotations"),
                       [str(clinical_root / f"Pairs{number}") for number in range(1, 9)],
                       str(destination), None, num_pairs=count, min_cc_size=0,
                       min_cc_intensity=0, model_space=space, prediction_threshold=threshold)
    protocol = read_json(destination / "protocol.json")
    if protocol["pairs_processed"] != count:
        raise ValueError(f"Incomplete evaluation: {destination}; see its log")


def legacy_summary(destination):
    precision = read_json(destination / "precision_measures.json")
    sensitivity = read_json(destination / "sensitivity_measures.json")
    matrices = read_json(destination / "agreement_matrices.json")
    result = {}
    for direction in ("Positive", "Negative"):
        result[f"HMDR {direction}"] = precision[f"Model HMDR ({direction})"]
        result[f"UDPP {direction}"] = precision[f"UDPP Model ({direction})"]
        result[f"UDPP STD {direction}"] = precision[f"UDPP STD Model ({direction})"]
        sign = "pos" if direction == "Positive" else "neg"
        result[f"Components {direction}"] = read_json(
            destination / f"total_labels_marked_{sign}.json")["Model"][0]
        for level in range(1, 6):
            key = f"Sensitivity Consensus Level {level} ({direction})"
            result[key] = sensitivity[key]
    for kind in ("per_label", "per_pair"):
        for sign in ("pos", "neg", "all"):
            values = np.asarray(matrices[f"{kind}_{sign}"])[-1, :-1]
            finite = values[np.isfinite(values)]
            result[f"PAI {kind} {sign} mean vs humans"] = float(finite.mean()) if finite.size else None
    return result


def export_per_clinician_f1(destination, cohorts=(98, 100)):
    rows = []
    lines = ["# Detection F1 Against Each Clinician", "",
             "Each entry compares one model with ONE clinician across all image pairs.",
             "No averaging across clinicians. F1 = 2M / (2M + U), using the same",
             "one-to-one component matching as the reference Itamar per-label agreement plots.",
             "Positive and negative components are matched separately; combined F1 pools their counts.",
             "This is detection F1, not pixel Dice or legacy boosted per-pair agreement.", "",
             "The 100-pair historical baseline includes lmm5 substitutes for pairs99/100.",
             "Itamar retains its saved-score threshold0 and pre-save cleanup; our frozen RAD-DINO",
             "head uses threshold0.5 with no added component filtering.", "",
             "[Download exact scores](f1_per_clinician.csv)", ""]
    for count in sorted(cohorts, reverse=True):
        scores = {}
        for model in ("foundation", "itamar"):
            matrices = read_json(destination / f"{model}_{count}" / "agreement_matrices.json")
            model_index = matrices["observers"].index("Model")
            scores[model] = {}
            for index, clinician in enumerate(matrices["observers"]):
                if clinician == "Model":
                    continue
                row = dict(cohort=count, model=model, clinician=clinician)
                for sign in ("pos", "neg", "all"):
                    row[f"F1 {sign}"] = matrices[f"per_label_{sign}"][model_index][index]
                scores[model][clinician] = row
                rows.append(row)
        lines += [f"## {count} Pairs", "",
                  "| Clinician | Our F1 + | Our F1 - | Our F1 all | Itamar F1 + | Itamar F1 - | Itamar F1 all |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for clinician in scores["foundation"]:
            values = [scores[model][clinician][f"F1 {sign}"]
                      for model in ("foundation", "itamar") for sign in ("pos", "neg", "all")]
            lines.append(f"| {clinician} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        lines += ["", "Full clinician-clinician/model matrices, matching the reference folder:", "",
                  f"[Our combined detection F1 matrix](foundation_{count}/per_label_agreement_all.png) | "
                  f"[Positive](foundation_{count}/per_label_agreement_pos.png) | "
                  f"[Negative](foundation_{count}/per_label_agreement_neg.png)", "",
                  f"[Our legacy per-pair agreement](foundation_{count}/per_pair_agreement_all.png) | "
                  f"[Positive](foundation_{count}/per_pair_agreement_pos.png) | "
                  f"[Negative](foundation_{count}/per_pair_agreement_neg.png)", "",
                  "The per-pair plots retain the historical empty-empty agreement and combined",
                  "one-sided boost. They are provided for reproduction, not labeled detection F1.", ""]
    write_csv(destination / "f1_per_clinician.csv", rows)
    (destination / "f1_per_clinician.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def export_metric_tables(destination, cohorts=(98, 100)):
    rows = []
    lines = ["# HMDR, UDPP and Detection F1", "",
             "HMDR: fraction of detections overlapping at least one reference clinician.",
             "UDPP: unmatched detections per image pair (mean; SD included in CSV).",
             "Detection F1 = 2M / (2M + U), using one-to-one spatial component matches,",
             "pooled over image pairs for each reader comparison, then averaged over readers.",
             "Human rows use the other four humans; model rows use all five humans.",
             "F1 is not pixel Dice or boosted per-pair PAI. Higher HMDR/F1 and lower UDPP are better.",
             "Positive and negative detections are matched separately; combined F1 pools their counts.", "",
             "[Individual model-vs-clinician F1 scores and full matrices](f1_per_clinician.md)", "",
             "The 100-pair historical baseline uses lmm5 substitutes for pairs99/100.",
             "The 98-pair table uses original Itamar only. Models retain their existing operating points:",
             "Itamar threshold0 and baked-in cleanup; foundation threshold0.5, no added cleanup.", "",
             "[Download numeric table, including UDPP SD](hmdr_udpp_f1.csv)", ""]
    for count in sorted(cohorts, reverse=True):
        profiles = {name: (read_json(destination / f"{name}_{count}" / "precision_measures.json"),
                           read_json(destination / f"{name}_{count}" / "agreement_matrices.json"))
                    for name in ("itamar", "foundation")}
        humans = [name for name in profiles["itamar"][1]["observers"] if name != "Model"]
        cohort_rows = []
        for observer, profile, label in [
                *((name, "itamar", name) for name in humans),
                ("Model", "itamar", "Itamar / historical baseline"),
                ("Model", "foundation", "Frozen RAD-DINO head")]:
            precision, matrices = profiles[profile]
            observer_index = matrices["observers"].index(observer)
            reference_indices = [matrices["observers"].index(name) for name in humans if name != observer]
            row = dict(cohort=count, observer=label)
            for direction in ("Positive", "Negative"):
                row[f"HMDR {direction}"] = precision[f"{observer} HMDR ({direction})"]
                row[f"UDPP {direction}"] = precision[f"UDPP {observer} ({direction})"]
                row[f"UDPP SD {direction}"] = precision[f"UDPP STD {observer} ({direction})"]
            for sign in ("pos", "neg", "all"):
                matrix = np.asarray(matrices[f"per_label_{sign}"])
                row[f"Detection F1 {sign}"] = float(matrix[observer_index, reference_indices].mean())
            cohort_rows.append(row)
        rows.extend(cohort_rows)
        lines += [f"## {count} Pairs", "",
                  "| Observer | HMDR + | HMDR - | UDPP + | UDPP - | Mean F1 + | Mean F1 - | Mean F1 all |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        fields = ("HMDR Positive", "HMDR Negative", "UDPP Positive", "UDPP Negative",
                  "Detection F1 pos", "Detection F1 neg", "Detection F1 all")
        for row in cohort_rows:
            lines.append("| " + row["observer"] + " | " +
                         " | ".join(f"{row[field]:.3f}" for field in fields) + " |")
        lines.append("")
    write_csv(destination / "hmdr_udpp_f1.csv", rows)
    (destination / "hmdr_udpp_f1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    export_per_clinician_f1(destination, cohorts)
    return rows


def spatial_summary(rows, model, count, observer):
    subset = [row for row in rows if row["model"] == model and row["observer"] == observer
              and int(row["pair_id"][4:]) <= count]
    active = [row for row in subset if row["reference_has_change"]]
    return dict(annotated_pairs=len(active), signed_hit_pairs=sum(row["signed_hit"] for row in active),
                empty_predictions=sum(row["predicted_area"] == 0 for row in subset),
                mean_signed_dice=float(np.mean([row["signed_dice"] for row in active])) if active else None,
                dice_ge_0_25=sum(row["signed_dice"] >= 0.25 for row in active),
                dice_ge_0_5=sum(row["signed_dice"] >= 0.5 for row in active))


def plot_pair(prior, current, references, masks, pair_id, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 9))
    signed = {name: positive.astype(int) - negative.astype(int)
              for name, (positive, negative) in {**references, **masks}.items()}
    for axis, image, title in ((axes[0, 0], prior, "Prior"), (axes[0, 1], current, "Current")):
        axis.imshow(image.T, cmap="gray")
        axis.set_title(title)
    for axis, name, title in ((axes[0, 2], "consensus_3_of_5", "3-of-5 reference"),
                              (axes[1, 0], "itamar", "Itamar / historical baseline"),
                              (axes[1, 1], "foundation", "Frozen RAD-DINO comparison head"),
                              (axes[1, 2], "any_clinician_union", "Any-clinician reference")):
        axis.imshow(current.T, cmap="gray")
        axis.imshow(np.ma.masked_where(signed[name].T == 0, signed[name].T),
                    cmap="bwr", vmin=-1, vmax=1, alpha=0.55)
        axis.set_title(title)
    for axis in axes.ravel():
        axis.axis("off")
    figure.suptitle(f"{pair_id}: red = positive; blue = negative")
    figure.tight_layout()
    figure.savefig(destination, dpi=110)
    plt.close(figure)


def run(args):
    if not 1 <= args.num_pairs <= 100:
        raise ValueError("num_pairs must be between 1 and 100")
    if args.out_dir.exists():
        raise ValueError("Use a new output directory to preserve previous comparisons")
    metadata = read_json(args.clinical_run / "metadata.json")
    if metadata["prediction_threshold"] != 0.5:
        raise ValueError("This comparison requires the fixed clinical threshold 0.5")
    cases = discover_cases(args.clinical_root, args.num_pairs)
    tools = clinical_tools()
    for case in cases:
        number = int(case["pair_id"][4:])
        for filename in ("output.nii.gz", "crop_info.json"):
            if not (args.itamar_preds / f"pair{number}" / filename).is_file():
                raise FileNotFoundError(args.itamar_preds / f"pair{number}" / filename)
        for doctor, path in case["annotations"].items():
            payload = Path(path).read_bytes()
            expected = metadata["annotation_sha256"][f"{case['pair_id']}/{doctor}"]
            if expected not in (hashlib.sha256(payload).hexdigest(),
                               hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()):
                raise ValueError(f"Changed annotation: {case['pair_id']}/{doctor}")
    args.out_dir.mkdir(parents=True)
    cohorts = (98, 100) if args.num_pairs == 100 else (args.num_pairs,)
    summaries = {}
    parity = {}
    for count in cohorts:
        summaries[count] = {}
        for name, predictions, space, threshold in (
                ("itamar", args.itamar_preds, "square_crop", 0.0),
                ("foundation", args.clinical_run, "original", 0.5)):
            destination = args.out_dir / f"{name}_{count}"
            print(f"[comparison] legacy metrics: {name}, pairs1-{count}", flush=True)
            run_legacy(tools, predictions, args.clinical_root, destination, count, space, threshold)
            summaries[count][name] = legacy_summary(destination)
        reference = args.legacy_reference_100 if count == 100 else args.legacy_reference_98
        if reference is not None:
            checked = []
            for filename in ("precision_measures.json", "sensitivity_measures.json",
                             "total_labels_marked_all.json", "total_labels_marked_pos.json",
                             "total_labels_marked_neg.json"):
                if not equivalent(read_json(reference / filename),
                                  read_json(args.out_dir / f"itamar_{count}" / filename)):
                    raise ValueError(f"Historical reference mismatch: {reference / filename}")
                checked.append(filename)
            parity[count] = checked
            print(f"[comparison] reproduced all5 historical JSON reports for {count}pairs", flush=True)
    spatial = []
    hashes = {}
    gallery = args.out_dir / "cases"
    gallery.mkdir()
    for case in cases:
        prior, current, current_nifti, references = load_case(case, tools)
        pair_id = case["pair_id"]
        votes = np.asarray(list(references.values())).sum(axis=0)
        references["consensus_3_of_5"] = (votes[0] >= 3, votes[1] >= 3)
        references["any_clinician_union"] = (votes[0] >= 1, votes[1] >= 1)
        masks = {}
        hashes[pair_id] = {}
        for name, root, crop, threshold in (
                ("itamar", args.itamar_preds, args.itamar_preds / pair_id / "crop_info.json", 0.0),
                ("foundation", args.clinical_run, None, 0.5)):
            path = root / pair_id / "output.nii.gz"
            with warnings.catch_warnings(record=True):
                positive, negative, raw = tools.load_model_labels_map(path, crop, current.shape,
                                                                     0, 0, threshold)
            masks[name] = (positive > 0, negative > 0)
            hashes[pair_id][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name == "foundation":
                saved = nib.load(root / pair_id / "mask.nii.gz")
                if not np.array_equal(positive - negative, np.asarray(saved.dataobj)):
                    raise ValueError(f"Thresholded mask mismatch: {pair_id}")
                if not np.allclose(saved.affine, current_nifti.affine):
                    raise ValueError(f"Prediction affine mismatch: {pair_id}")
            for observer, reference in references.items():
                score = score_masks(masks[name], reference)
                spatial.append(dict(model=name, pair_id=pair_id, observer=observer,
                                    signed_hit=int(bool(score["signed_dice"])), **score))
        plot_pair(prior, current, references, masks, pair_id, gallery / f"{pair_id}.png")
        print(f"[comparison] spatial metrics/review {pair_id}", flush=True)
    write_csv(args.out_dir / "spatial_per_pair.csv", spatial)
    comparisons = []
    for count in cohorts:
        for name in ("itamar", "foundation"):
            for observer in ("consensus_3_of_5", "any_clinician_union", *PHYSICIANS):
                values = spatial_summary(spatial, name, count, observer)
                summaries[count][name].update({f"{observer} {key}": value for key, value in values.items()})
        for metric, old in summaries[count]["itamar"].items():
            new = summaries[count]["foundation"][metric]
            comparisons.append(dict(cohort=count, metric=metric, itamar=old, foundation=new,
                                    delta=new - old if old is not None and new is not None else None))
    write_csv(args.out_dir / "comparison.csv", comparisons)
    report = dict(cohorts=summaries, baseline_json_parity=parity,
                  foundation_checkpoint=metadata["checkpoint_sha256"],
                  foundation_threshold=0.5, itamar_threshold=0.0,
                  postprocessing="no additional CC filtering; Itamar pre-save cleanup cannot be undone",
                  caveat="Pairs99/100 in the historical 100-pair baseline use lmm5, not original Itamar",
                  annotation_warnings={case["pair_id"]: case["annotation_warnings"]
                                       for case in cases if case["annotation_warnings"]},
                  prediction_sha256=hashes)
    write_json(args.out_dir / "comparison.json", report)
    export_metric_tables(args.out_dir, cohorts)
    lines = ["# Clinical Model Comparison", "", report["caveat"] + ".", "",
             "[F1 against each individual clinician and full matrices](f1_per_clinician.md)", "",
             "[HMDR, UDPP and detection F1 tables](hmdr_udpp_f1.md)", "",
             "Both models use the same full-image annotations. No extra CC filtering.",
             "Foundation scores use the fixed threshold0.5; historical saved Itamar uses sign at0.",
             "Legacy consensus sensitivity counts annotated components, not pixelwise Dice.",
             "Legacy per-pair-all PAI retains empty-empty=1 and the one-sided boost.",
             "Ellipses are exploratory references, not dense pathology GT.", "",
             "[Comparison CSV](comparison.csv) | [Full report](comparison.json) | "
             "[Per-pair spatial metrics](spatial_per_pair.csv)", ""]
    for count in cohorts:
        lines += [f"## Pairs1-{count}", "", "| Metric | Historical baseline | Foundation |",
                  "|---|---:|---:|"]
        for row in comparisons:
            if row["cohort"] == count:
                old = f"{row['itamar']:.4f}" if row["itamar"] is not None else "NA"
                new = f"{row['foundation']:.4f}" if row["foundation"] is not None else "NA"
                lines.append(f"| {row['metric']} | {old} | {new} |")
        lines += ["", f"[Historical agreement matrix](itamar_{count}/per_pair_agreement_all.png) | "
                  f"[Foundation agreement matrix](foundation_{count}/per_pair_agreement_all.png)", ""]
    lines += ["## Matched Case Reviews", ""]
    lines += [f"- [{case['pair_id']}](cases/{case['pair_id']}.png)" for case in cases]
    (args.out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[comparison] complete: {args.out_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-run", type=Path, required=True)
    parser.add_argument("--clinical-root", type=Path, required=True)
    parser.add_argument("--itamar-preds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--legacy-reference-100", type=Path)
    parser.add_argument("--legacy-reference-98", type=Path)
    parser.add_argument("--num-pairs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())