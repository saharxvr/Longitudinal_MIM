"""Few-shot trainer/evaluation end to end on tiny fixtures with a fake frozen backbone (CPU)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from datasets.test_episodes import build_corpus, square
from evaluation import few_shot_eval, few_shot_manifests
from evaluation.training_free_baselines import build_baselines
from training import train_few_shot as trainer

GRID, DIM = 6, 16


class FakeBackbone(torch.nn.Module):
    """Frozen stand-in: pooled image -> 3 'depths' of 16 channels on a 6x6 grid."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.projection = torch.nn.Conv2d(1, 3 * DIM, 1)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, images):
        pooled = F.adaptive_avg_pool2d(images.float(), GRID)
        features = torch.tanh(self.projection(pooled * 4 - 2)).flatten(2).transpose(1, 2)
        return dict(patch_tokens=features, cls_token=features.mean(1))


def build_val_dataset(root, pairs_per_task=6, nuisance=4):
    rng = np.random.default_rng(1)
    lines = []
    tasks = (("consolidation", "appearance", 0.2), ("pneumothorax", "disappearance", -0.2))
    index = 0
    for anomaly, direction, value in tasks:
        for number in range(pairs_per_task):
            lines.append(_pair(root, rng, index, f"p{number}", anomaly, direction, value))
            index += 1
    for number in range(nuisance):
        lines.append(_pair(root, rng, index, f"q{number}", "none", "none", 0.0))
        index += 1
    (root / "manifest_split.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _pair(root, rng, index, patient, anomaly, direction, value):
    pair_id = f"{patient}_{index}/pair0"
    folder = root / pair_id
    folder.mkdir(parents=True)
    base = rng.random((512, 512)).astype(np.float32)
    gt = square(100 + 10 * (index % 4), 120, value, 80)
    for name, array in (("prior", base), ("current", base + gt), ("diff_map", gt)):
        nib.save(nib.Nifti1Image(array, np.eye(4)), str(folder / f"{name}.nii.gz"))
    return dict(pair_id=pair_id, case=patient, patient_key=patient, split="val",
                change_group_id=pair_id, effective_anomaly_type=anomaly, anomaly_type=anomaly,
                direction=direction, realized_change=anomaly != "none")


class FewShotTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        (root / "corpus").mkdir()
        (root / "val").mkdir()
        build_corpus(root / "corpus", scenes=12, patients=6)
        build_val_dataset(root / "val")
        self.root = root
        self.cfg = trainer.load_config(
            None, corpus_root=str(root / "corpus"), run_root=str(root / "run"),
            val_dataset_root=str(root / "val"), arms=["fsdiff", "panet", "none"], grid=GRID,
            backbone_dim=DIM, channels=16, num_heads=2, local_pool=2, episodes_per_epoch=3, epochs=2,
            max_support=3, queries=2, precision="fp32", device="cpu", num_workers=0, extract_chunk=8,
            val_shuffle_check=True, benchmark_support=[3, 2], benchmark_episodes=1)

    def tearDown(self):
        self.folder.cleanup()

    def _context(self, cfg=None, run="run"):
        cfg = cfg or self.cfg
        cfg.run_root = str(self.root / run)
        original = few_shot_manifests.synthetic_manifest

        def small(*args, **kwargs):
            return original(*args, **dict(kwargs, n_values=(1, 2), draws=2, groups=2, max_queries=4,
                                          nuisance_per_task=2))

        with patch.object(few_shot_manifests, "synthetic_manifest", side_effect=small):
            return trainer.prepare(cfg, FakeBackbone())

    def test_training_validation_checkpoints_and_exact_resume(self):
        context = self._context()
        initial = {arm: {k: v.clone() for k, v in branch["head"].state_dict().items()}
                   for arm, branch in context["branches"].items()}
        summary = trainer.train(self.cfg, context)
        run = Path(self.cfg.run_root)
        for arm in self.cfg.arms:
            self.assertTrue((run / f"best_{arm}.pt").is_file())
            self.assertGreater(summary[arm]["best_epoch"], 0)
            changed = any(not torch.equal(initial[arm][key], value)
                          for key, value in context["branches"][arm]["head"].state_dict().items())
            self.assertTrue(changed)
        validation = json.loads((run / "validation_epoch002.json").read_text())
        self.assertIn("selection_score", validation["summary"]["fsdiff"])
        self.assertIn("shuffled:fsdiff", validation["summary"])
        self.assertIn("n0_gt0.001", validation["summary"]["none"])
        self.assertTrue((run / "metrics.csv").is_file())
        with self.assertRaises(ValueError):
            trainer.train(self.cfg, self._context())

        interrupted = self._context(run="interrupted")
        original = trainer.run_epoch
        calls = []

        def stop_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return original(*args, **kwargs)

        with patch.object(trainer, "run_epoch", side_effect=stop_second):
            with self.assertRaises(KeyboardInterrupt):
                trainer.train(self.cfg, interrupted)
        resumed = self._context(run="interrupted")
        trainer.train(self.cfg, resumed, resume=True)
        for arm in self.cfg.arms:
            for key, value in context["branches"][arm]["head"].state_dict().items():
                torch.testing.assert_close(resumed["branches"][arm]["head"].state_dict()[key], value)

    def test_benchmark_reports_chosen_support(self):
        context = self._context(run="bench")
        report = trainer.benchmark(self.cfg, context)
        self.assertEqual(report["chosen_max_support"], 3)
        self.assertEqual([row["max_support"] for row in report["results"]], [3, 2])

    def test_evaluation_cli_synthetic_and_icu_runner(self):
        from evaluation import evaluate_few_shot

        self.cfg.epochs = 1
        trainer.train(self.cfg, self._context(run="eval_run"))
        extractor = few_shot_eval.TokenExtractor(FakeBackbone(), "cpu", "fp32", 8, 64)
        original = few_shot_manifests.synthetic_manifest

        def small(*args, **kwargs):
            return original(*args, **dict(kwargs, n_values=(1, 2), draws=2, groups=2, nuisance_per_task=2))

        with patch.object(evaluate_few_shot, "make_extractor", return_value=extractor), \
                patch.object(few_shot_manifests, "synthetic_manifest", side_effect=small):
            evaluate_few_shot.main(["synthetic", "--run_root", str(self.root / "eval_run"), "--output",
                                    str(self.root / "eval_out"), "--dataset_root", str(self.root / "val"),
                                    "--device", "cpu", "--precision", "fp32"])
        summary = json.loads((self.root / "eval_out" / "summary.json").read_text())["summary"]
        for name in ("fsdiff", "fsdiff_n0", "panet", "none", "training_free_knn", "shuffled:fsdiff"):
            self.assertIn(name, summary)

        class FakeICU:
            readers = ("r1", "r2")
            rotation_k, flip = 3, True

            def __init__(self):
                self.rng = np.random.default_rng(0)
                self.cases = {f"pair{index}": index for index in range(6)}

            def load(self, key):
                index = self.cases[key]
                mask = np.zeros((300, 260), bool)
                mask[40 + 10 * index:110 + 10 * index, 60:140] = True
                image = torch.from_numpy(self.rng.random((1, 512, 512)).astype(np.float32))
                return dict(images=(image, image), shape=mask.shape,
                            annotations={"r1": (mask, np.zeros_like(mask)), "r2": (mask, np.zeros_like(mask))})

            def consensus(self, key, k=None):
                return self.load(key)["annotations"]["r1"]

            def item(self, key, mode="consensus"):
                loaded = self.load(key)
                signed = torch.zeros(1, 512, 512)
                signed[:, 100:200, 100:200] = 0.5
                return loaded["images"][0], loaded["images"][1], signed, torch.zeros(1, 512, 512, dtype=torch.bool)

        source = FakeICU()
        manifest = few_shot_manifests.icu_manifest(sorted(source.cases), sorted(source.cases), n_values=(1, 2),
                                                   draws=2, groups=3, seed=0)
        heads, config, _ = evaluate_few_shot.load_few_shot_heads(self.root / "eval_run", ["fsdiff", "none"])
        predictors = {"fsdiff": few_shot_eval.HeadPredictor(heads["fsdiff"]),
                      "none": few_shot_eval.HeadPredictor(heads["none"], uses_support=False),
                      **build_baselines(["knn"])}
        run = few_shot_eval.run_icu(manifest, source, extractor, predictors, retrieval=True)
        summary = few_shot_eval.summarize_icu(run, bootstrap=50)
        self.assertEqual(set(summary["fsdiff|n2"]), {"draws", "ensemble", "retrieval"})
        self.assertEqual(summary["fsdiff|n1"]["draws"]["draws"], 2)
        self.assertIn("none|n0", summary)
        matrix = evaluate_few_shot.per_reference(run)
        self.assertEqual(set(matrix["training_free_knn|n1"]), {"r1", "r2", "consensus"})
        evaluate_few_shot.write_table(summary, self.root / "table.csv")
        evaluate_few_shot.plot_curves(summary, self.root / "curves.png")
        self.assertTrue((self.root / "curves.png").is_file())

    def test_manifest_queries_once_per_draw_and_disjoint_supports(self):
        from datasets.image_store import build_pair_store

        store = build_pair_store(str(self.root / "val"), "val", self.root / "val_store")
        files = few_shot_eval.SyntheticPairSource(str(self.root / "val"), "val")
        stored = few_shot_eval.SyntheticPairSource(str(self.root / "val"), "val", str(store))
        key = files.dataset.records[3]["pair_id"]
        for a, b in zip(files.item(key), stored.item(key)):
            torch.testing.assert_close(a.float(), b.float(), atol=2e-3, rtol=0)
        manifest = few_shot_manifests.synthetic_manifest(str(self.root / "val"), "val", selection=False, seed=3,
                                                         n_values=(1, 4), draws=3, groups=2, nuisance_per_task=2)
        written = few_shot_manifests.write_manifest(manifest, self.root / "m.json")
        self.assertEqual(few_shot_manifests.read_manifest(written)["meta"]["sha256"], manifest["meta"]["sha256"])
        patient = lambda key: key.split("_")[0]
        seen = {}
        for entry in manifest["entries"]:
            self.assertFalse({patient(key) for key in entry["support"]} & {patient(key) for key in entry["queries"]})
            for key in entry["queries"]:
                seen[(entry["task"], entry["n"], entry["draw"], key)] = seen.get(
                    (entry["task"], entry["n"], entry["draw"], key), 0) + 1
        self.assertEqual(set(seen.values()), {1})
        self.assertEqual(len({key[3] for key in seen}), 12)

    def test_synthetic_runner_scores_training_free_and_zero_shot(self):
        manifest = few_shot_manifests.synthetic_manifest(str(self.root / "val"), "val", selection=True, seed=0,
                                                         n_values=(1, 2), draws=2, groups=2, max_queries=4,
                                                         nuisance_per_task=2)
        source = few_shot_eval.SyntheticPairSource(str(self.root / "val"), "val")
        extractor = few_shot_eval.TokenExtractor(FakeBackbone(), "cpu", "fp32", 8, cache_items=64)
        head = trainer.FewShotChangeHead(**trainer.head_spec(self.cfg, "none")["kwargs"])
        predictors = dict(build_baselines(), zero=few_shot_eval.HeadPredictor(head, uses_support=False))
        result = few_shot_eval.run_synthetic(manifest, source, extractor, predictors, shuffle_check=True)
        summary = result["summary"]
        for name in ("training_free_prototype", "training_free_knn"):
            self.assertIn("n2_gt0.001", summary[name])
            self.assertGreaterEqual(summary[name]["selection_score"], 0)
        self.assertEqual(sorted(key for key in summary["zero"] if key.startswith("n")), ["n0_gt0.001", "n0_gt0.05"])
        self.assertTrue(any(key.startswith("training_free_prototype|") for key in result["nuisance_fp_area"]))


class ICUScoringTests(unittest.TestCase):
    def test_training_free_predictors_transfer_support_labels(self):
        from models.few_shot_head import grid_labels

        generator = torch.Generator().manual_seed(0)

        def pair(changed, sign):
            prior = torch.randn(1, GRID * GRID, 3 * DIM, generator=generator)
            current = prior + 0.3 * torch.randn(prior.shape, generator=generator)
            direction = torch.zeros(3 * DIM)
            direction[::DIM] = 1.0
            current[0, changed] = prior[0, changed] + sign * 8 * direction
            return prior, current

        supports = [pair([7, 8, 13], 1), pair([20, 21], 1)]
        support_gt = torch.zeros(2, 1, GRID * 10, GRID * 10)
        for index, tokens in enumerate(([7, 8, 13], [20, 21])):
            for token in tokens:
                row, column = divmod(token, GRID)
                support_gt[index, 0, row * 10:(row + 1) * 10, column * 10:(column + 1) * 10] = 0.3
        labels = grid_labels(support_gt, grid=GRID)
        support = (torch.cat([item[0] for item in supports]), torch.cat([item[1] for item in supports]))
        prior, current = pair([27, 28], 1)
        query = (prior, current, prior.mean(1), current.mean(1))
        for name, predictor in build_baselines().items():
            prediction = predictor(query, support, labels)
            self.assertEqual(prediction.shape, (1, 1, 512, 512))
            row, column = divmod(27, GRID)
            cell = 512 / GRID
            center = prediction[0, 0, int((row + 0.5) * cell), int((column + 0.5) * cell)]
            self.assertGreater(float(center), 0.5)
            self.assertGreater(float(center), float(prediction[0, 0, 5, 5]) + 0.2)
            if name.endswith("knn"):
                self.assertLess(float(prediction[0, 0, 5, 5]), 0.5)

    def test_vector_scores_match_presentation_scores(self):
        from evaluation.rescore_subset_clinical_many import scores

        rng = np.random.default_rng(0)
        for _ in range(50):
            terms = rng.integers(0, 6, 6)
            expected = scores(terms)
            actual = few_shot_eval.vector_scores(terms[None])
            for metric in few_shot_eval.METRICS:
                if expected[metric] is None:
                    self.assertTrue(np.isnan(actual[metric][0]))
                else:
                    self.assertAlmostEqual(float(actual[metric][0]), expected[metric])

    def test_icu_terms_and_summary_with_draws_ensemble_and_bootstrap(self):
        reference = np.zeros((300, 300), bool)
        reference[50:120, 50:120] = True
        references = [(reference, np.zeros_like(reference))] * 2 + [(reference, np.zeros_like(reference))]
        scores = np.zeros((300, 300), np.float32)
        scores[60:130, 60:130] = 0.9
        terms, area = few_shot_eval.icu_terms(scores, references)
        self.assertEqual(terms.shape, (3, 2, 6))
        self.assertEqual(few_shot_eval.vector_scores(terms[0].sum(0))["one_to_many"], 1.0)
        pairs = ["a", "b", "c"]
        perfect = {key: terms for key in pairs}
        empty_terms, _ = few_shot_eval.icu_terms(np.zeros_like(scores), references)
        run = dict(pairs=pairs, reference_names=["r1", "r2", "consensus"], results=dict(
            model={(4, 0): perfect, (4, 1): {key: empty_terms for key in pairs}, (4, "ensemble"): perfect},
            zero={(0, 0): {key: empty_terms for key in pairs}}))
        summary = few_shot_eval.summarize_icu(run, bootstrap=200)
        row = summary["model|n4"]
        self.assertAlmostEqual(row["draws"]["one_to_many"]["mean"], 0.5)
        self.assertEqual(row["draws"]["draws"], 2)
        self.assertEqual(row["ensemble"]["one_to_many"]["mean"], 1.0)
        self.assertEqual(len(row["draws"]["one_to_many"]["ci95"]), 2)
        difference = few_shot_eval.paired_difference(run, "model", "zero", 4, bootstrap=100)
        self.assertAlmostEqual(difference["difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
