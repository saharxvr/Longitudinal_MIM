"""Lock an explicitly chosen rank, benchmark B2, and optionally publish compact metrics."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import yaml

from training import train_architecture_data_ablation as trainer
from training.run_upright_clinical import publish


ROOT = trainer.ROOT


def load_diagnostic(config):
    basis = Path(config["diagnostic_file"])
    report = json.loads(basis.with_suffix(".json").read_text(encoding="utf-8"))
    if report["diagnostic_sha256"] != trainer.sha256(basis):
        raise ValueError("Saved diagnostic JSON and basis hashes disagree")
    for name, digest in report["provenance"]["code_sha256"].items():
        if trainer.sha256(ROOT / name) != digest:
            raise ValueError(f"Code changed since diagnostics: {name}")
    return report


def lock_config(source, rank):
    config = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    trainer.validate_config(trainer.parse_args(["--config", str(source)]))
    report = load_diagnostic(config)
    if type(rank) is not int or rank <= 0 or rank not in {row["rank"] for row in report["rows"]}:
        raise ValueError("Choose an explicitly diagnosed nonzero rank")
    for key, value in (("debias_rank", rank), ("diagnostic_sha256", report["diagnostic_sha256"])):
        if config.get(key) not in (None, value):
            raise ValueError(f"Source config already locks a different {key}")
        config[key] = value
    destination = Path(config["run_root"]) / f"locked_rank{rank}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if yaml.safe_load(destination.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing locked config differs; do not overwrite it")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    return destination.resolve(), config, report


def export_report(config, diagnostic):
    benchmark_path = Path(config["run_root"]) / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    identity = benchmark["identity"]
    if (identity["provenance"] != diagnostic["provenance"]
            or identity["config"]["debias_rank"] != config["debias_rank"]
            or identity["config"]["diagnostic_sha256"] != diagnostic["diagnostic_sha256"]
            or benchmark["training"]["measured_batches"] < trainer.MIN_BENCHMARK_BATCHES):
        raise ValueError("Benchmark does not match the locked diagnostic/rank")
    for key, value in config.items():
        if key not in ("config", "resume", "benchmark_batches") and identity["config"].get(key) != value:
            raise ValueError(f"Benchmark config differs in {key}")
    diagnostic_keys = (
        "rank", "same_position", "other_position", "positional_gap",
        "train_energy_retained_mean", "train_energy_retained_min", "noise_energy_removed",
    )
    metric_keys = (
        "optimizer_updates", "sample_draws", "seconds", "measured_batches",
        "loss_total", "loss_ce", "loss_dice", "target_signed_dice",
        "raw_signed_dice", "nuisance_fp_mean", "nuisance_fp_p95", "nuisance_fp_max", "component_f1",
    )
    rows = [{key: row[key] for key in diagnostic_keys} for row in diagnostic["rows"]]
    metrics = {split: {key: benchmark[split][key] for key in metric_keys}
               for split in ("training", "validation")}
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("Nonfinite diagnostic values cannot be published")
    for values in metrics.values():
        if any(value is not None and not math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite benchmark values cannot be published")
    features = identity["provenance"]["features"]
    report = dict(
        selected_rank=config["debias_rank"], diagnostic_sha256=diagnostic["diagnostic_sha256"],
        benchmark_sha256=trainer.sha256(benchmark_path), diagnostic_rows=rows,
        provenance=dict(
            **{key: features[key] for key in (
                "manifest_sha256", "model_id", "backbone_revision", "backbone_layer",
                "feature_normalization", "image_size", "backbone_image_size",
                "rotation_k", "flip_horizontal", "noise_method")},
            runtime=identity["provenance"]["runtime"],
            code_sha256=identity["provenance"]["code_sha256"]),
        recipe={key: config[key] for key in (
            "batch_size", "accum", "epochs", "lr", "weight_decay", "warmup_epochs",
            "photometric_augmentation", "dpt_channels", "target_threshold", "eval_threshold")},
        updates_per_slot=identity["updates_per_slot"], **metrics,
        peak_cuda_bytes=benchmark["peak_cuda_bytes"],
        note="Disposable B2 benchmark, not trained-model performance. No full sweep launched. "
             "Export excludes images, patient/pair IDs, paths, basis and model weights.")
    destination = ROOT / "results" / f"architecture_data_benchmark_{time.time_ns()}"
    destination.mkdir(parents=True, exist_ok=False)
    trainer.write_json(destination / "summary.json", report)
    if (destination / "summary.json").stat().st_size > 1_000_000:
        raise ValueError("Compact benchmark export exceeds 1 MB")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--publish", action="store_true", help="Commit/push only the compact result export")
    parser.add_argument("--export-only", action="store_true", help="Export an existing benchmark without GPU work")
    options = parser.parse_args(argv)
    source = Path(options.config).resolve()
    initial = yaml.safe_load(source.read_text(encoding="utf-8"))
    with trainer.run_lock(Path(initial["run_root"]) / ".benchmark_export.lock"):
        locked, config, diagnostic = lock_config(source, options.rank)
        print(f"[rank lock] {locked}", flush=True)
        if not options.export_only:
            subprocess.run(
                [sys.executable, "-u", "-m", "training.train_architecture_data_ablation",
                 "--config", str(locked), "--mode", "benchmark"], cwd=ROOT, check=True)
        # Recheck source/basis identity after the child exits, before exporting.
        if load_diagnostic(config) != diagnostic:
            raise ValueError("Diagnostic changed during the benchmark")
        destination = export_report(config, diagnostic)
        print(f"[export] {destination / 'summary.json'}", flush=True)
        if options.publish:
            publish(destination)


if __name__ == "__main__":
    main()
