"""Evaluate few-shot heads on the synthetic pathology tasks (final manifest) or ICU-100.

Synthetic (held-out tasks, val or test split):
    python -m evaluation.evaluate_few_shot synthetic --run_root RUN --dataset_root DATA \
        --split val --output OUT [--val_store STORE]
ICU-100 (exploratory, reused cohort; nothing is selected here):
    python -m evaluation.evaluate_few_shot icu --run_root RUN --clinical_root CLIN --output OUT \
        --support_modes consensus consensus_2 reader:Avi ... --arm_a_checkpoint A_BEST.pt \
        [--synthetic_supports DATA] [--retrieval] [--finetune_steps 20]

Predictors: every best_<arm>.pt in RUN (FSDiff also at n=0), training-free prototype/kNN,
optional supervised arm A (n=0) and arm-A few-shot fine-tuning. Same manifests for all.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import few_shot_eval, few_shot_manifests
from evaluation.training_free_baselines import build_baselines
from models import build_change_head
from models.few_shot_head import build_few_shot_head


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_few_shot_heads(run_root, arms=None, device="cpu"):
    heads, config, provenance = {}, None, {}
    for path in sorted(Path(run_root).glob("best_*.pt")):
        arm = path.stem.removeprefix("best_")
        if arms and arm not in arms:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        head = build_few_shot_head(state["spec"], device).eval()
        head.load_state_dict(state["state_dict"])
        heads[arm] = head
        config = config or state["config"]
        if state["config"]["feature_layers"] != config["feature_layers"]:
            raise ValueError("All few-shot heads must use the same frozen feature depths")
        provenance[arm] = dict(sha256=sha256(path), epoch=state["epoch"], val_selection_score=state["score"])
    if not heads:
        raise ValueError(f"No best_<arm>.pt checkpoints in {run_root}")
    return heads, config, provenance


def load_arm_a(path, config, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    spec, run_config = state["head_spec"], state.get("run_config", {})
    if (spec["type"] != "dpt_layer_fusion" or list(spec["kwargs"]["feature_layers"]) != list(config["feature_layers"])
            or int(run_config.get("rotation_k", 0)) != int(config["rotation_k"])
            or bool(run_config.get("flip_horizontal", False)) != bool(config["flip_horizontal"])):
        raise ValueError("Arm A checkpoint must use the same depths and orientation as the few-shot heads")
    head = build_change_head(spec, device).eval()
    head.load_state_dict(state["head"])
    return head, dict(sha256=sha256(path), epoch=state.get("epoch"))


def predictors_for(heads, args, config, device):
    predictors = {}
    for arm, head in heads.items():
        predictors[arm] = few_shot_eval.HeadPredictor(head, uses_support=arm != "none", device=device)
        if arm == "fsdiff":
            predictors["fsdiff_n0"] = few_shot_eval.HeadPredictor(head, uses_support=False, device=device)
    if args.training_free:
        predictors.update(build_baselines())
    provenance = {}
    if args.arm_a_checkpoint:
        head, provenance["arm_a"] = load_arm_a(args.arm_a_checkpoint, config, device)
        predictors["arm_a_supervised"] = few_shot_eval.HeadPredictor(head, uses_support=False, device=device)
        if args.finetune_steps:
            predictors["arm_a_finetune"] = few_shot_eval.FineTunePredictor(
                head, steps=args.finetune_steps, lr=args.finetune_lr, device=device)
    return predictors, provenance


def make_extractor(config, args):
    from models import FrozenCXRBackbone

    backbone = FrozenCXRBackbone(model_id=config["model_id"], selected_layers=list(config["feature_layers"]),
                                 freeze=True).to(args.device).eval()
    return few_shot_eval.TokenExtractor(backbone, args.device, args.precision, args.chunk, args.cache_pairs)


class CombinedSource:
    """ICU queries; supports keyed 'syn:<pair_id>' come from synthetic pathology pairs."""

    def __init__(self, icu, synthetic):
        self.icu, self.synthetic = icu, synthetic
        self.readers, self.rotation_k, self.flip = icu.readers, icu.rotation_k, icu.flip

    def item(self, key, mode="consensus"):
        if key.startswith("syn:"):
            return self.synthetic.item(key[4:])
        return self.icu.item(key, mode)

    def load(self, key):
        return self.icu.load(key)

    def consensus(self, key, k=None):
        return self.icu.consensus(key, k)


def per_reference(run):
    """Draw-mean one-to-many F1 for every reference (readers + consensus) per predictor/n."""
    output = {}
    for predictor, by_key in run["results"].items():
        sizes = sorted({n for n, draw in by_key if not isinstance(draw, str)})
        for n in sizes:
            draws = [per_pair for (size, draw), per_pair in by_key.items() if size == n and not isinstance(draw, str)]
            values = []
            for per_pair in draws:
                terms = np.stack([per_pair[key] for key in run["pairs"]]).sum((0, 2))
                values.append(few_shot_eval.vector_scores(terms)["one_to_many"])
            output[f"{predictor}|n{n}"] = dict(zip(run["reference_names"], np.mean(values, 0).tolist()))
    return output


def write_table(summary, path):
    rows = []
    for key, row in summary.items():
        predictor, n = key.rsplit("|n", 1)
        for kind, values in row.items():
            rows.append(dict(predictor=predictor, n=int(n), kind=kind, draws=values["draws"],
                             **{f"{metric}_{stat}": (values[metric].get(stat) if stat != "ci95"
                                                     else json.dumps(values[metric].get("ci95")))
                                for metric in few_shot_eval.METRICS
                                for stat in ("mean", "sd", "consensus", "ci95")}))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(summary, path, metric="one_to_many"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    curves = {}
    for key, row in summary.items():
        predictor, n = key.rsplit("|n", 1)
        if "draws" in row:
            curves.setdefault(predictor, []).append((int(n), row["draws"][metric]))
    for predictor, points in sorted(curves.items()):
        points.sort()
        sizes = [n for n, _ in points]
        means = np.array([value["mean"] for _, value in points])
        if sizes == [0]:
            axis.axhline(means[0], linestyle="--", label=f"{predictor} (no support)")
            continue
        sds = np.array([value["sd"] for _, value in points])
        axis.plot(sizes, means, marker="o", label=predictor)
        axis.fill_between(sizes, means - sds, means + sds, alpha=0.15)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("support size n")
    axis.set_ylabel(f"mean clinician {metric} F1")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_synthetic_mode(args, heads, config, provenance):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = few_shot_manifests.synthetic_manifest(args.dataset_root, args.split, selection=False,
                                                     seed=args.seed, draws=args.draws)
    few_shot_manifests.write_manifest(manifest, output / "manifest.json")
    source = few_shot_eval.SyntheticPairSource(args.dataset_root, args.split, args.store,
                                               config["rotation_k"], config["flip_horizontal"])
    extractor = make_extractor(config, args)
    predictors, extra = predictors_for(heads, args, config, args.device)
    result = few_shot_eval.run_synthetic(manifest, source, extractor, predictors, config["eval_threshold"],
                                         shuffle_check=True, seed=args.seed)
    (output / "summary.json").write_text(json.dumps(dict(
        summary=result["summary"], nuisance_fp_area=result["nuisance_fp_area"], heads=provenance, **extra,
        manifest_sha256=manifest["meta"]["sha256"], split=args.split), indent=1) + "\n", encoding="utf-8")
    with (output / "rows.json").open("w", encoding="utf-8") as stream:
        json.dump(result["rows"], stream)
    return result["summary"]


def run_icu_mode(args, heads, config, provenance):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    icu = few_shot_eval.ICUPairSource(args.clinical_root, config["rotation_k"], config["flip_horizontal"],
                                      args.consensus_k)
    pairs = sorted(icu.cases)
    extractor = make_extractor(config, args)
    predictors, extra = predictors_for(heads, args, config, args.device)
    report = dict(heads=provenance, **extra, pairs=len(pairs), consensus_k=args.consensus_k,
                  threshold=args.threshold, cc_cutoff=args.cc_cutoff, modes={})
    for mode in args.support_modes:
        pool = [key for key in pairs if icu.nonempty(key, mode)] if args.pool == "nonempty" else pairs
        manifest = few_shot_manifests.icu_manifest(pairs, pool, n_values=tuple(args.n_values), draws=args.draws,
                                                   groups=args.groups, seed=args.seed)
        name = mode.replace(":", "_")
        few_shot_manifests.write_manifest(manifest, output / f"manifest_{name}.json")
        run = few_shot_eval.run_icu(manifest, icu, extractor, predictors, mode, args.threshold, args.cc_cutoff,
                                    ensemble=True, retrieval=args.retrieval and mode == "consensus")
        summary = few_shot_eval.summarize_icu(run, bootstrap=args.bootstrap, seed=args.seed)
        differences = {}
        reference = "arm_a_supervised" if "arm_a_supervised" in predictors else "fsdiff_n0"
        for predictor in predictors:
            if predictor in run["results"] and predictor != reference and reference in run["results"]:
                for n in manifest["meta"]["n_values"]:
                    if any(size == n for size, _ in run["results"][predictor]):
                        differences[f"{predictor}-{reference}|n{n}"] = few_shot_eval.paired_difference(
                            run, predictor, reference, n, bootstrap=args.bootstrap, seed=args.seed)
        report["modes"][mode] = dict(pool=len(pool), manifest_sha256=manifest["meta"]["sha256"],
                                     summary=summary, per_reference=per_reference(run),
                                     paired_differences=differences)
        write_table(summary, output / f"table_{name}.csv")
        plot_curves(summary, output / f"curves_{name}.png")
        np.savez_compressed(output / f"terms_{name}.npz", **{
            f"{predictor}|{n}|{draw}": np.stack([per_pair[key] for key in run["pairs"]])
            for predictor, by_key in run["results"].items() for (n, draw), per_pair in by_key.items()})
    if args.synthetic_supports:
        synthetic = few_shot_eval.SyntheticPairSource(args.synthetic_supports, args.synthetic_split, args.store,
                                                      config["rotation_k"], config["flip_horizontal"])
        records = [record for record in synthetic.dataset.records] if synthetic.dataset else []
        pool = [f"syn:{record['pair_id']}" for record in records
                if record.get("realized_change") and record.get("effective_anomaly_type", "none") != "none"]
        manifest = few_shot_manifests.icu_manifest(pairs, pool, patients={**{key: key for key in pairs},
                                                                         **{key: key for key in pool}},
                                                   n_values=tuple(args.n_values), draws=args.draws,
                                                   groups=args.groups, seed=args.seed)
        few_shot_manifests.write_manifest(manifest, output / "manifest_synthetic_supports.json")
        run = few_shot_eval.run_icu(manifest, CombinedSource(icu, synthetic), extractor, predictors, "consensus",
                                    args.threshold, args.cc_cutoff, ensemble=False)
        summary = few_shot_eval.summarize_icu(run, bootstrap=args.bootstrap, seed=args.seed)
        report["modes"]["synthetic_supports"] = dict(pool=len(pool), summary=summary)
        write_table(summary, output / "table_synthetic_supports.csv")
    (output / "summary.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("mode", choices=("synthetic", "icu"))
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="*")
    parser.add_argument("--dataset_root")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--store")
    parser.add_argument("--clinical_root")
    parser.add_argument("--support_modes", nargs="+", default=["consensus", "consensus_2"])
    parser.add_argument("--pool", choices=("nonempty", "all"), default="nonempty")
    parser.add_argument("--consensus_k", type=int, default=3)
    parser.add_argument("--n_values", nargs="+", type=int, default=list(few_shot_manifests.N_VALUES))
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--cc_cutoff", type=int, default=250)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--synthetic_supports")
    parser.add_argument("--synthetic_split", default="val")
    parser.add_argument("--arm_a_checkpoint")
    parser.add_argument("--finetune_steps", type=int, default=0)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--training_free", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("fp32", "bf16"))
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--cache_pairs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args(argv)
    heads, config, provenance = load_few_shot_heads(args.run_root, args.arms, args.device)
    if args.mode == "synthetic":
        if not args.dataset_root:
            parser.error("--dataset_root is required for synthetic evaluation")
        run_synthetic_mode(args, heads, config, provenance)
    else:
        if not args.clinical_root:
            parser.error("--clinical_root is required for ICU evaluation")
        run_icu_mode(args, heads, config, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
