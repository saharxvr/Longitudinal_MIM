from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SEED = 20261001
PAIR_COUNT = 12
FRESH_RUNS = 5
MODEL = "gpt-5.4"
SAMPLE = sorted(random.Random(SEED).sample(range(1, 99), PAIR_COUNT - 2) + [99, 100])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    annotation_root = repo_root / "python_files" / "annotation tool"
    parser = argparse.ArgumentParser(description="Create the GPT-5.4 repeatability study snapshot.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--source-experiment",
        type=Path,
        default=annotation_root
        / "LLM_Pathology_Heatmaps"
        / "gpt_5_4_full_experiment",
    )
    parser.add_argument(
        "--shared-input-root",
        type=Path,
        default=annotation_root
        / "LLM_Pathology_Heatmaps"
        / "claude_sonnet_5_full_experiment"
        / "inputs",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=annotation_root / "llm_prompts" / "precision_heatmap_annotator.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=annotation_root
        / "LLM_Pathology_Heatmaps"
        / "gpt_5_4_precision_repeatability",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    inputs_root = args.output_root / "inputs"
    baseline_root = args.output_root / "baseline_original" / "responses"
    runs_root = args.output_root / "runs"
    inputs_root.mkdir(exist_ok=True)
    baseline_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(exist_ok=True)

    prompt_snapshot = args.output_root / "precision_heatmap_annotator.txt"
    shutil.copy2(args.prompt, prompt_snapshot)

    files = []
    for pair_number in SAMPLE:
        pair_id = f"pair{pair_number}"
        source_image = args.shared_input_root / pair_id / "prior_current.png"
        target_dir = inputs_root / pair_id
        target_dir.mkdir(exist_ok=True)
        target_image = target_dir / "prior_current.png"
        shutil.copy2(source_image, target_image)
        baseline_source = (
            args.source_experiment
            / "responses_heatmap_precise"
            / f"{pair_id}.json"
        )
        baseline_target = baseline_root / f"{pair_id}.json"
        shutil.copy2(baseline_source, baseline_target)
        files.extend(
            [
                {
                    "role": "input_image",
                    "pair_id": pair_id,
                    "path": str(target_image.relative_to(args.output_root)),
                    "sha256": sha256(target_image),
                },
                {
                    "role": "original_baseline_response",
                    "pair_id": pair_id,
                    "path": str(baseline_target.relative_to(args.output_root)),
                    "sha256": sha256(baseline_target),
                },
            ]
        )

    run_manifests = []
    base_prompt = prompt_snapshot.read_text(encoding="utf-8").rstrip()
    for run_number in range(1, FRESH_RUNS + 1):
        run_id = f"run_{run_number:02d}"
        nonce = hashlib.sha256(
            f"{MODEL}|precision-repeatability|{SEED}|{run_id}".encode()
        ).hexdigest()[:24]
        run_root = runs_root / run_id
        responses_root = run_root / "responses"
        responses_root.mkdir(parents=True, exist_ok=True)
        run_prompt = run_root / "prompt.txt"
        run_prompt.write_text(
            base_prompt
            + "\n\n"
            + f"REPEATABILITY_RUN_NONCE: {nonce}\n"
            + "The nonce is cache-busting metadata only. Ignore it when interpreting the images.\n",
            encoding="utf-8",
        )
        run_manifest = {
            "run_id": run_id,
            "model": MODEL,
            "model_delivery": "Copilot SDK isolated general-purpose agent",
            "fresh_session_required": True,
            "prior_outputs_visible": False,
            "cache_busting_nonce": nonce,
            "prompt_path": str(run_prompt.relative_to(args.output_root)),
            "prompt_sha256": sha256(run_prompt),
            "pair_numbers": SAMPLE,
            "response_directory": str(responses_root.relative_to(args.output_root)),
            "generation_parameters": {
                "temperature": "not exposed by the agent runtime",
                "seed": "not exposed by the agent runtime",
            },
        }
        (run_root / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2),
            encoding="utf-8",
        )
        run_manifests.append(run_manifest)

    protocol = {
        "study": "GPT-5.4 precision heatmap repeatability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "algorithm": "Python random.Random(seed).sample(range(1, 99), 10), plus pairs 99 and 100",
            "seed": SEED,
            "sample_size": PAIR_COUNT,
            "pair_numbers": SAMPLE,
            "forced_coordinate_stress_cases": [99, 100],
        },
        "design": {
            "fresh_runs": FRESH_RUNS,
            "original_output_role": "secondary frozen baseline; excluded from primary fresh-run repeatability",
            "blinding": "each run sees only its prompt snapshot and the 12 snapshotted input sheets",
            "cache_mitigation": (
                "fresh isolated session per run plus a unique non-semantic nonce; "
                "provider-side cache state cannot be directly inspected or disabled by this runtime"
            ),
        },
        "model": MODEL,
        "prompt_snapshot": {
            "path": str(prompt_snapshot.relative_to(args.output_root)),
            "sha256": sha256(prompt_snapshot),
        },
        "runs": run_manifests,
        "files": files,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_value(args.repo_root, "rev-parse", "HEAD"),
            "git_status_short": git_value(args.repo_root, "status", "--short"),
        },
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(args.output_root), "sample": SAMPLE}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
