from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PILOT_DIRECTORIES = [
    "_pilot_inputs",
    "all_pairs_compare_pathologists_model_llm",
    "pilot_claude_sonnet_5",
    "pilot_claude_sonnet_5_appearance_disappearance",
    "pilot_claude_sonnet_5_forced_best_effort",
    "pilot_claude_sonnet_5_heatmap_only",
    "pilot_claude_sonnet_5_signed_heatmap_only",
    "pilot_compare_pathologists_model",
]
SLIDE_COLLAGES = {
    "pair15_collage.png",
    "pair45_collage.png",
    "pair51_collage.png",
    "pair96_collage.png",
    "pair98_collage.png",
}
PRESENTATIONS_TO_KEEP = {
    "Why_Not_Use_LLM_Experiment.pptx",
    "Why_Not_Use_LLM_With_Size_Stats.pptx",
}
REPEATABILITY_COLLAGES = {"pair47_repeatability.png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class Cleaner:
    def __init__(self, root: Path, apply: bool) -> None:
        self.root = root.resolve()
        self.apply = apply
        self.deleted_paths: list[str] = []
        self.deleted_bytes = 0
        if self.root.name != "LLM_Pathology_Heatmaps":
            raise ValueError(f"Refusing unexpected root: {self.root}")

    def validate(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved == self.root or not resolved.is_relative_to(self.root):
            raise ValueError(f"Refusing unsafe cleanup target: {resolved}")
        return resolved

    def delete_directory(self, path: Path) -> None:
        resolved = self.validate(path)
        if not resolved.is_dir():
            return
        self.deleted_bytes += directory_size(resolved)
        self.deleted_paths.append(str(resolved.relative_to(self.root)))
        if self.apply:
            shutil.rmtree(resolved)

    def delete_file(self, path: Path) -> None:
        resolved = self.validate(path)
        if not resolved.is_file():
            return
        self.deleted_bytes += resolved.stat().st_size
        self.deleted_paths.append(str(resolved.relative_to(self.root)))
        if self.apply:
            resolved.unlink()


def classify(relative_path: Path) -> str:
    parts = set(relative_path.parts)
    if relative_path.suffix.lower() == ".pptx":
        return "presentation"
    if "responses" in parts or any(part.startswith("responses_") for part in parts):
        return "raw_llm_response"
    if relative_path.name in {"protocol.json", "run_manifest.json"}:
        return "reproducibility_manifest"
    if relative_path.suffix.lower() in {".csv", ".json"}:
        return "metric_or_audit"
    if relative_path.suffix.lower() == ".png":
        return "slide_evidence_image"
    if relative_path.suffix.lower() in {".md", ".txt"}:
        return "documentation_or_prompt"
    return "supporting_artifact"


def write_manifest(root: Path, cleaner: Cleaner) -> None:
    manifest_path = root / "FINAL_EVIDENCE_MANIFEST.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        relative_path = path.relative_to(root)
        files.append(
            {
                "path": str(relative_path),
                "role": classify(relative_path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cleanup_profile": "reproducible_compact",
        "kept_file_count": len(files),
        "kept_bytes": sum(item["bytes"] for item in files),
        "deleted_bytes": cleaner.deleted_bytes,
        "deleted_paths": cleaner.deleted_paths,
        "files": files,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cleanup(root: Path, apply: bool) -> Cleaner:
    cleaner = Cleaner(root, apply)
    for directory in PILOT_DIRECTORIES:
        cleaner.delete_directory(root / directory)

    claude_root = root / "claude_sonnet_5_full_experiment"
    for directory in (
        "collages",
        "inputs",
        "presentation",
        "rendered_ellipses",
        "rendered_heatmap",
        "rendered_heatmap_precise",
    ):
        cleaner.delete_directory(claude_root / directory)
    cleaner.delete_directory(claude_root / "collages_precision" / "per_pair")

    gpt_root = root / "gpt_5_4_full_experiment"
    for directory in (
        "rendered_ellipses",
        "rendered_heatmap",
        "rendered_heatmap_precise",
    ):
        cleaner.delete_directory(gpt_root / directory)
    cleaner.delete_directory(gpt_root / "collages_precision" / "per_pair")

    cross_root = root / "claude_gpt_comparison"
    collage_root = cross_root / "collages"
    if collage_root.is_dir():
        for path in collage_root.iterdir():
            if path.is_file() and path.name not in SLIDE_COLLAGES:
                cleaner.delete_file(path)
    presentation_root = cross_root / "presentation"
    if presentation_root.is_dir():
        for path in presentation_root.iterdir():
            if path.is_file() and path.name not in PRESENTATIONS_TO_KEEP:
                cleaner.delete_file(path)

    repeatability_root = root / "gpt_5_4_precision_repeatability"
    cleaner.delete_directory(repeatability_root / "baseline_original" / "rendered")
    for run_number in range(1, 6):
        cleaner.delete_directory(
            repeatability_root / "runs" / f"run_{run_number:02d}" / "rendered"
        )
    repeatability_collages = repeatability_root / "analysis" / "collages"
    if repeatability_collages.is_dir():
        for path in repeatability_collages.iterdir():
            if path.is_file() and path.name not in REPEATABILITY_COLLAGES:
                cleaner.delete_file(path)

    if apply:
        write_manifest(root, cleaner)
    return cleaner


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Remove regenerable LLM experiment artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root
        / "python_files"
        / "annotation tool"
        / "LLM_Pathology_Heatmaps",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion. Without this flag, only print the dry-run plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cleaner = cleanup(args.root, args.apply)
    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry_run",
                "paths": len(cleaner.deleted_paths),
                "megabytes": round(cleaner.deleted_bytes / (1024 * 1024), 1),
                "targets": cleaner.deleted_paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
