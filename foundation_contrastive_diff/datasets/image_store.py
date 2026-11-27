"""Float16 array stores so few-shot episodes do not decompress NIfTIs at every step.

Scene store (supervoxel corpus):  images [S,2,512,512] (prior,current; min-max normalized,
saved-NIfTI frame, NOT oriented) and maps [C,512,512] (all per-change GT maps concatenated).
Pair store (synthetic val/test pairs): images [P,2,512,512] and maps [P,512,512] (clipped GT).
Orientation is applied at load time by the episode code, exactly like LongitudinalPairDataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from datasets.pair_dataset import _load_nii_2d, _norm01, _read_manifest, _to_chw

SIZE = 512


def _image(path):
    return _norm01(_to_chw(_load_nii_2d(str(path)), SIZE))[0].numpy().astype(np.float16)


def build_scene_store(corpus_root, destination=None):
    root = Path(corpus_root)
    destination = Path(destination or root / "_store")
    records = [json.loads(line) for line in (root / "manifest_scenes.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Scene manifest is empty")
    destination.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(destination / "images.f16.npy", mode="w+", dtype=np.float16,
                                       shape=(len(records), 2, SIZE, SIZE))
    total = sum(record["num_changes"] for record in records)
    maps = np.lib.format.open_memmap(destination / "maps.f16.npy", mode="w+", dtype=np.float16,
                                     shape=(total, SIZE, SIZE))
    index, offset = {}, 0
    for row, record in enumerate(records):
        folder = root / record["scene_dir"]
        images[row, 0] = _image(folder / record["files"]["prior"])
        images[row, 1] = _image(folder / record["files"]["current"])
        scene_maps = np.load(folder / record["files"]["changes"])["maps"]
        if len(scene_maps) != record["num_changes"]:
            raise ValueError(f"{record['scene_id']}: change maps do not match the manifest")
        maps[offset:offset + len(scene_maps)] = scene_maps
        index[record["scene_id"]] = dict(row=row, map_offset=offset, num_changes=len(scene_maps))
        offset += len(scene_maps)
    images.flush()
    maps.flush()
    (destination / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    return destination


def build_pair_store(dataset_root, split, destination, clip_gt=1.0):
    records = [record for record in _read_manifest(dataset_root) if record.get("split") == split]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(destination / "images.f16.npy", mode="w+", dtype=np.float16,
                                       shape=(len(records), 2, SIZE, SIZE))
    maps = np.lib.format.open_memmap(destination / "maps.f16.npy", mode="w+", dtype=np.float16,
                                     shape=(len(records), SIZE, SIZE))
    index = {}
    root = Path(dataset_root)
    for row, record in enumerate(records):
        folder = root / record["pair_id"]

        def path(key, name):
            value = record.get(key)
            return value if value and Path(value).is_file() else folder / name

        images[row, 0] = _image(path("prior_path", "prior.nii.gz"))
        images[row, 1] = _image(path("current_path", "current.nii.gz"))
        gt = _to_chw(_load_nii_2d(str(path("diff_map_path", "diff_map.nii.gz"))), SIZE)[0]
        maps[row] = gt.clamp(-clip_gt, clip_gt).numpy().astype(np.float16)
        index[record["pair_id"]] = dict(row=row)
    images.flush()
    maps.flush()
    (destination / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")
    return destination


class ArrayStore:
    """Read-only memory-mapped store."""

    def __init__(self, folder):
        folder = Path(folder)
        self.images = np.load(folder / "images.f16.npy", mmap_mode="r")
        self.maps = np.load(folder / "maps.f16.npy", mmap_mode="r")
        self.index = json.loads((folder / "index.json").read_text(encoding="utf-8"))

    def pair(self, key):
        entry = self.index[key]
        images = torch.from_numpy(np.asarray(self.images[entry["row"]], np.float32))
        return images[0:1], images[1:2]

    def scene_maps(self, key):
        entry = self.index[key]
        start = entry["map_offset"]
        return torch.from_numpy(np.asarray(self.maps[start:start + entry["num_changes"]], np.float32))

    def pair_map(self, key):
        return torch.from_numpy(np.asarray(self.maps[self.index[key]["row"]], np.float32))[None]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build float16 few-shot stores")
    parser.add_argument("kind", choices=("scenes", "pairs"))
    parser.add_argument("--root", required=True, help="Scene corpus root or pair dataset root")
    parser.add_argument("--destination")
    parser.add_argument("--split", default="val")
    args = parser.parse_args(argv)
    if args.kind == "scenes":
        print(build_scene_store(args.root, args.destination))
    else:
        if not args.destination:
            parser.error("--destination is required for pair stores")
        print(build_pair_store(args.root, args.split, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
