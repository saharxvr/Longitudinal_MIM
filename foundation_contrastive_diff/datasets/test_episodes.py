"""Selective episodes: task targets, distractor ignore, patient-disjoint supports, coarse labels."""

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from data_generation.supervoxel_render import fit_ellipses
from datasets import episodes
from datasets.image_store import build_scene_store

FAMILIES = (("nodule", "lungs"), ("heart_resize", "heart"), ("free_air", "sub_diaphragm"))


def square(row, column, value, size=40):
    signed = np.zeros((512, 512), np.float32)
    signed[row:row + size, column:column + size] = value
    return signed


def build_corpus(root, scenes=18, patients=6, seed=0):
    rng = np.random.default_rng(seed)
    lines = []
    for index in range(scenes):
        patient = f"p{index % patients}"
        folder = root / f"case{index}" / "scene0000"
        folder.mkdir(parents=True)
        for name in ("prior", "current"):
            nib.save(nib.Nifti1Image(rng.random((512, 512)).astype(np.float32), np.eye(4)),
                     str(folder / f"{name}.nii.gz"))
        count = 1 + index % 3
        maps, changes = [], []
        for position in range(count):
            family, compartment = FAMILIES[(index + position) % len(FAMILIES)]
            sign = 1 if (index + position) % 2 == 0 else -1
            maps.append(square(60 + 130 * position, 60 + 20 * (index % 5), 0.2 * sign))
            changes.append(dict(index=position, family=family, compartment=compartment, tier="T2",
                                gt_sign="positive" if sign > 0 else "negative", realized=True))
        maps = np.stack(maps)
        np.savez_compressed(folder / "changes.npz", maps=maps.astype(np.float16))
        (folder / "ellipses.json").write_text(json.dumps([fit_ellipses(item) for item in maps]))
        lines.append(json.dumps(dict(
            scene_id=f"case{index}/scene0000", scene_dir=f"case{index}/scene0000", case=f"case{index}",
            patient_key=patient, split="train", corpus_tiers=["T1", "T2"], changes=changes,
            num_changes=count, realized_changes=count,
            files=dict(prior="prior.nii.gz", current="current.nii.gz", changes="changes.npz",
                       ellipses="ellipses.json"))))
    (root / "manifest_scenes.jsonl").write_text("\n".join(lines) + "\n")


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        build_corpus(cls.root)
        cls.corpus = episodes.SceneCorpus(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_task_target_marks_only_tau_and_ignores_opposite_overlap(self):
        maps = torch.from_numpy(np.stack([square(10, 10, 0.3), square(30, 30, -0.4), square(200, 200, 0.2)]))
        changes = [dict(family="a", compartment="x", gt_sign="positive", realized=True),
                   dict(family="b", compartment="x", gt_sign="negative", realized=True),
                   dict(family="a", compartment="y", gt_sign="positive", realized=True)]
        task = episodes.Task("family", family="a")
        target, ignore, everything, _ = episodes.task_target(maps, changes, task)
        self.assertTrue(bool((target[200:240, 200:240] > 0).all()))
        self.assertTrue(bool(ignore[30:50, 30:50].all()))
        self.assertFalse(bool(ignore[10:30, 10:30].any()))
        self.assertTrue(bool((target[60:70, 60:70] == 0).all()))
        self.assertLess(float(everything[60:70, 60:70].min()), 0)

    def test_sampler_supports_disjoint_patients_and_contain_task(self):
        config = episodes.EpisodeConfig(max_support=4, queries=3, empty_query_prob=0.3)
        sampler = episodes.EpisodeSampler(self.corpus, config)
        rng = np.random.default_rng(0)
        for _ in range(40):
            episode = sampler.sample(rng)
            task = episode["task"]
            query_patients = {self.corpus.patients[row] for row in episode["queries"]}
            self.assertFalse(query_patients & {self.corpus.patients[row] for row in episode["supports"]})
            self.assertLessEqual(len(episode["supports"]), 4)
            if task.level != "any":
                self.assertGreaterEqual(len(episode["supports"]), 1)
            for row in episode["supports"]:
                self.assertTrue(any(task.matches(change) for change in self.corpus.records[row]["changes"]))

    def test_materialized_episode_shapes_orientation_and_determinism(self):
        config = episodes.EpisodeConfig(max_support=3, queries=2)
        dataset = episodes.EpisodeDataset(self.corpus, config, episodes_per_epoch=5, seed=7)
        first, again = dataset[2], dataset[2]
        self.assertEqual(first["support_ids"], again["support_ids"])
        torch.testing.assert_close(first["query_gt"], again["query_gt"])
        self.assertEqual(first["query_prior"].shape, (2, 1, 512, 512))
        self.assertEqual(first["support_gt"].shape[0], first["n"])
        self.assertEqual(first["query_ignore"].dtype, torch.bool)
        dataset.set_epoch(1)
        self.assertNotEqual([dataset[index]["query_ids"] for index in range(5)],
                            [first["query_ids"]] + [None] * 4)
        row = self.corpus.records.index(next(r for r in self.corpus.records if r["scene_id"] == first["query_ids"][0]))
        raw = episodes.task_target(self.corpus.maps(row), self.corpus.records[row]["changes"],
                                   self.corpus.tasks[first["task"]])[0]
        torch.testing.assert_close(first["query_gt"][0, 0], episodes.orient(config, raw[None])[0])

    def test_coarse_supports_with_tight_queries(self):
        config = episodes.EpisodeConfig(max_support=3, queries=2, label_mode="coarse_support", coarse_prob=1.0)
        sampler = episodes.EpisodeSampler(self.corpus, config)
        rng = np.random.default_rng(3)
        episode = sampler.sample(rng, support_size=2)
        episode["coarse_variant"] = "ellipse"
        batch = episodes.materialize(self.corpus, episode, config, rng)
        self.assertTrue(batch["coarse"])
        support = batch["support_gt"]
        self.assertTrue(set(torch.unique(support.abs()).tolist()) <= {0.0, 0.5})
        self.assertLess(float(batch["query_gt"].abs().max()), 0.25)
        tight = episodes.EpisodeConfig(max_support=3, queries=2)
        reference = episodes.materialize(self.corpus, dict(episode, coarse=False, coarse_variant=None), tight, rng)
        covered = ((support != 0) | (reference["support_gt"] == 0)).float().mean()
        self.assertGreater(float(covered), 0.999)
        for variant in ("bbox", "dilate", "erode", "drop"):
            target, ignore = episodes.coarse_target(
                torch.from_numpy(square(50, 50, 0.3)), [fit_ellipses(square(50, 50, 0.3))],
                [dict(family="a", gt_sign="positive", realized=True)], episodes.Task("any"), variant, rng)
            self.assertEqual(target.shape, (512, 512))
            self.assertFalse(bool(ignore.any()))

    def test_store_matches_files(self):
        store = build_scene_store(self.root, self.root / "_store_test")
        stored = episodes.SceneCorpus(self.root, store=store)
        for row in (0, 5):
            for a, b in zip(stored.images(row), self.corpus.images(row)):
                torch.testing.assert_close(a, b, atol=1e-3, rtol=0)
            torch.testing.assert_close(stored.maps(row), self.corpus.maps(row))


if __name__ == "__main__":
    unittest.main()
