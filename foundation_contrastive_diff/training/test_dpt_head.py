"""CPU checks: python -m unittest training.test_dpt_head."""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.evaluate_rq1 import _tokens, run_eval, select_review_cases
from losses import change_map_loss
from models import DPTCrossAttentionHead, DPTDifferenceHead, DifferenceHead, build_change_head
from training.train_foundation_diff import head_spec, parse_args
from training import train_foundation_diff as trainer
from training.cache_features import slice_last_layer_cache


class ReviewSelectionTests(unittest.TestCase):
    def test_review_export_end_to_end(self):
        import numpy as np
        import nibabel as nib
        from PIL import Image
        from evaluation import evaluate_rq1 as evaluator

        spec = {"type": "dpt_cross_attention", "kwargs": dict(
            backbone_dim=16, patch_dim=16, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12, num_layers=1)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for index, value in enumerate((0.2, -0.2, 0.0)):
                pair_id = f"case/pair{index}/variant0"
                folder = root / pair_id
                folder.mkdir(parents=True)
                prior = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
                gt = np.zeros((32, 32), dtype=np.float32)
                gt[8:20, 8:20] = value
                for name, array in (("prior", prior), ("current", prior + gt), ("diff_map", gt)):
                    nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f"{name}.nii.gz")
                records.append(dict(pair_id=pair_id, split="val", realized_change=bool(value),
                                    effective_anomaly_type="consolidation" if value else "none",
                                    change_group_id=str(index)))
                cache = root / "cache" / pair_id
                cache.mkdir(parents=True)
                torch.save(dict(p_prior=torch.randn(25, 16), p_curr=torch.randn(25, 16),
                                cls_prior=torch.randn(16), cls_curr=torch.randn(16),
                                gt_diff=torch.from_numpy(gt)[None], anomaly_type=1 if value else 0,
                                direction=0, is_pathology=int(bool(value)),
                                change_group_id=str(index), pair_id=pair_id), cache / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records), encoding="utf-8")
            checkpoint = root / "best.pt"
            torch.save(dict(head=build_change_head(spec).state_dict(), head_spec=spec,
                            epoch=55), checkpoint)
            output = root / "review"
            arguments = ["eval", "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--ckpt", str(checkpoint), "--split", "val", "--threshold", "0.05",
                         "--device", "cpu", "--batch_size", "2", "--num_workers", "0",
                         "--out_dir", str(output), "--review"]
            with patch.object(sys, "argv", arguments):
                evaluator.main()
            cases = json.loads((output / "review_cases.json").read_text())
            metadata = json.loads((output / "review_metadata.json").read_text())
            self.assertEqual(len(cases), 3)
            self.assertEqual(metadata["evaluated_pairs"], 3)
            self.assertEqual(metadata["checkpoint_epoch"], 55)
            self.assertEqual(metadata["checkpoint_sha256"], evaluator.file_sha256(checkpoint))
            self.assertTrue((output / "index.md").is_file())
            self.assertEqual(len((output / "per_pair_val.csv").read_text().splitlines()), 4)
            self.assertEqual(len(list(output.glob("contact_*.png"))), 3)
            for row in cases:
                with Image.open(output / row["panel"]) as image:
                    self.assertGreater(image.width, 1000)
                    self.assertTrue(any(low != high for low, high in image.convert("RGB").getextrema()))
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(ValueError, "empty"):
                evaluator.main()
            rows = [dict(row) for row in cases]
            rows.pop()
            dataset = evaluator.CachedPairDataset(str(root), str(root / "cache"), split="val")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                evaluator.export_review(build_change_head(spec), None, dataset, rows, [],
                                        str(root), "val", output, "cpu", 0.05, img_size=32)
            (root / "cache" / records[0]["pair_id"] / "feat.pt").unlink()
            arguments[arguments.index("--out_dir") + 1] = str(root / "new_review")
            with patch.object(sys, "argv", arguments), self.assertRaisesRegex(FileNotFoundError, "complete cache"):
                evaluator.main()

    def test_stratified_selection_is_reproducible_and_diverse(self):
        rows = []
        for anomaly in ("consolidation", "pleural_effusion"):
            for direction in ("positive", "negative", "mixed"):
                for index in range(40):
                    rows.append(dict(pair_id=f"{anomaly}_{direction}_{index:02}",
                                     change_group_id=f"{anomaly}_{direction}_{index // 2}",
                                     anomaly_type=anomaly, direction=direction,
                                     is_pathology=1, dice=index / 40, nuisance_fp=None))
        for index in range(60):
            rows.append(dict(pair_id=f"nuisance_{index:02}", change_group_id="",
                             anomaly_type="none", direction="below_threshold",
                             is_pathology=0, dice=None, nuisance_fp=index / 60))
        chosen = select_review_cases(rows, per_group=12, nuisance_count=20)
        self.assertEqual(chosen, select_review_cases(list(reversed(rows)), 12, 20))
        self.assertEqual(len(chosen), 6 * 12 + 20)
        self.assertEqual(len({row["pair_id"] for row in chosen}), len(chosen))
        self.assertEqual({row["selection"] for row in chosen},
                         {"lowest_dice", "middle_dice", "highest_dice", "seeded_random", "highest_fp"})
        self.assertIn("nuisance_59", {row["pair_id"] for row in chosen})
        for group in {row["review_group"] for row in chosen} - {"nuisance"}:
            identities = [row["change_group_id"] for row in chosen if row["review_group"] == group]
            self.assertEqual(len(identities), len(set(identities)))

    def test_small_buckets_and_invalid_counts(self):
        row = dict(pair_id="one", anomaly_type="consolidation", direction="positive",
                   is_pathology=1, dice=0.5, nuisance_fp=None)
        self.assertEqual(len(select_review_cases([row])), 1)
        for count in (1, 2, 3):
            rows = [dict(row, pair_id=str(index)) for index in range(8)]
            self.assertEqual(len(select_review_cases(rows, per_group=count)), count)
        with self.assertRaises(ValueError):
            select_review_cases([row], per_group=0)

    def test_review_metrics_include_nuisance_and_direction(self):
        class FixedHead(torch.nn.Module):
            def forward(self, *inputs):
                return torch.tensor([[[[0.2, -0.2]]], [[[0.2, 0.0]]]]), None

        batch = dict(p_prior=torch.zeros(2, 1, 1), p_curr=torch.zeros(2, 1, 1),
                     cls_prior=torch.zeros(2, 1), cls_curr=torch.zeros(2, 1),
                     gt_diff=torch.tensor([[[[0.2, -0.2]]], [[[0.0, 0.0]]]]),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]),
                     pair_id=["pathology", "control"], change_group_id=["a", "b"])
        rows = []
        report, _ = run_eval(FixedHead(), None, [batch], "cpu", 0.05, review_rows=rows)
        self.assertEqual(rows[0]["direction"], "mixed")
        self.assertEqual(rows[0]["dice"], 1.0)
        self.assertEqual(rows[1]["nuisance_fp"], 0.5)
        self.assertIsNone(rows[1]["dice"])
        self.assertEqual(report["overall"]["n"], 1)


class DPTHeadTests(unittest.TestCase):
    head_type = "dpt"
    config_path = "configs/last4_dpt.yaml"
    num_layers = 4

    def setUp(self):
        torch.manual_seed(0)
        self.spec = {"type": self.head_type, "kwargs": dict(
            backbone_dim=16, patch_dim=16 * self.num_layers, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12)}
        if self.num_layers == 1:
            self.spec["kwargs"]["num_layers"] = 1
        self.prior = torch.randn(2, 25, 16 * self.num_layers)
        self.current = torch.randn_like(self.prior)
        self.cls_prior = torch.randn(2, 16)
        self.cls_current = torch.randn(2, 16)

    def test_forward_loss_backward_and_reload(self):
        head = build_change_head(self.spec)
        inputs = (self.prior, self.current, self.cls_prior, self.cls_current)
        prediction, embedding = head(*inputs)
        self.assertEqual(prediction.shape, (2, 1, 32, 32))
        self.assertEqual(embedding.shape, (2, 12))
        self.assertLessEqual(float(prediction.detach().abs().max()), 1.0)
        loss, _ = change_map_loss(prediction, torch.zeros_like(prediction),
                                  w_l1=1, w_l2=1, w_dice=0, w_sign=0)
        loss.backward()
        for project in head.projections:
            self.assertGreater(float(project.weight.grad.abs().sum()), 0)
        for parameter in head.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer = torch.optim.AdamW(head.parameters())
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"head": head.state_dict(), "head_spec": self.spec,
                    "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_change_head(checkpoint["head_spec"])
        restored.load_state_dict(checkpoint["head"])
        restored_optimizer = torch.optim.AdamW(restored.parameters())
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(restored(*inputs)[0], head(*inputs)[0])

    def test_reject_wrong_cache(self):
        with self.assertRaises(ValueError):
            DPTDifferenceHead(patch_dim=768)
        head = build_change_head(self.spec)
        with self.assertRaises(ValueError):
            head(self.prior[..., :-1], self.current[..., :-1])
        with self.assertRaises(ValueError):
            head(self.prior[:, :-1], self.current[:, :-1])

    def test_default_head_and_config_override(self):
        self.assertIsInstance(build_change_head(head_spec(768)), DifferenceHead)
        with patch.object(sys, "argv", ["train", "--config", self.config_path,
                                        "--batch_size", "2"]):
            args = parse_args()
        self.assertEqual((args.head, args.feat_last_k, args.batch_size),
                 (self.head_type, self.num_layers, 2))
        spec = head_spec(768 * self.num_layers, args.head)
        self.assertEqual(spec["type"], self.head_type)
        self.assertEqual(spec["kwargs"].get("num_layers", 4), self.num_layers)
        self.assertEqual((args.w_l1, args.w_l2, args.w_dice, args.w_sign), (1, 1, 0, 0))

    def test_evaluator_and_feature_slicing(self):
        batch = dict(p_prior=self.prior, p_curr=self.current,
                     cls_prior=self.cls_prior, cls_curr=self.cls_current,
                     gt_diff=torch.zeros(2, 1, 32, 32),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]))
        batch["gt_diff"][0, 0, 8:20, 8:20] = 0.2
        sliced = _tokens(batch, None, "cpu", 16)
        torch.testing.assert_close(sliced[0], self.prior[..., -16:])
        report, _ = run_eval(build_change_head(self.spec), None, [batch], "cpu", 0.05,
                     self.spec["kwargs"]["patch_dim"])
        self.assertEqual(report["overall"]["n"], 1)
        self.assertEqual(report["nuisance_false_positive_area"]["n"], 1)

    def test_cached_training_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for split in ("train", "val"):
                for index in range(2):
                    pair_id = f"{split}_{index}"
                    records.append(dict(pair_id=pair_id, split=split,
                                        effective_anomaly_type="none"))
                    folder = root / "cache" / pair_id
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=self.prior[index], p_curr=self.current[index],
                                    cls_prior=self.cls_prior[index], cls_curr=self.cls_current[index],
                                    gt_diff=torch.ones(1, 32, 32) * 0.2,
                                    anomaly_type=1, direction=1, is_pathology=1,
                                    change_group_id=pair_id, pair_id=pair_id), folder / "feat.pt")
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            arguments = ["train", "--config", self.config_path,
                         "--dataset_root", str(root), "--cache_dir", str(root / "cache"),
                         "--save_folder", str(root / "checkpoints"),
                         "--plots_folder", str(root / "plots"), "--batch_size", "2",
                         "--accum", "1", "--num_workers", "0", "--device", "cpu",
                         "--dpt_channels", "8"]
            with patch.multiple(trainer.C, BACKBONE_DIM=16, FEATURE_GRID=5,
                                IMG_SIZE=32, EMBED_DIM=12), \
                    patch.object(trainer, "plot_curves"), \
                    patch.object(trainer, "save_sample_plots"):
                with patch.object(sys, "argv", arguments + ["--epochs", "1"]):
                    trainer.main()
                first = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
                self.assertEqual(first["best_dice"], first["val"]["dice"])
                with patch.object(sys, "argv", arguments + ["--epochs", "2", "--resume"]):
                    trainer.main()
            checkpoint = torch.load(root / "checkpoints" / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["best_dice"], max(first["best_dice"], checkpoint["val"]["dice"]))
            self.assertEqual(checkpoint["head_spec"]["type"], self.head_type)
            self.assertEqual(len((root / "checkpoints" / "metrics.csv").read_text().splitlines()), 3)


class DPTCrossAttentionTests(DPTHeadTests):
    head_type = "dpt_cross_attention"
    config_path = "configs/last4_dpt_cross_attention.yaml"

    def test_temporal_attention_gradients_and_order(self):
        head = build_change_head(self.spec)
        prediction, _ = head(self.prior, self.current)
        prediction.square().mean().backward()
        for blocks in head.temporal_blocks:
            for block in blocks:
                gradient = block.attention.in_proj_weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
        with torch.no_grad():
            forward = head._project_streams(self.prior, self.current)
            reverse = head._project_streams(self.current, self.prior)
            identical = head._project_streams(self.prior, self.prior)
            offset = torch.randn_like(self.prior)
            shifted = head._project_streams(self.prior + offset, self.current + offset)
        for actual, swapped, unchanged in zip(forward, reverse, identical):
            torch.testing.assert_close(actual, -swapped)
            torch.testing.assert_close(unchanged, torch.zeros_like(unchanged))
        self.assertTrue(any(not torch.allclose(actual, shifted_actual)
                            for actual, shifted_actual in zip(forward, shifted)))

    def test_reject_invalid_attention_settings(self):
        for settings in ({"num_heads": 3}, {"num_heads": 0}, {"attention_depth": 0}):
            with self.assertRaises(ValueError):
                DPTCrossAttentionHead(**self.spec["kwargs"], **settings)


class SingleLayerCrossAttentionTests(DPTCrossAttentionTests):
    config_path = "configs/last1_dpt_cross_attention.yaml"
    num_layers = 1


class CacheSliceTests(unittest.TestCase):
    def test_layer_aware_launcher_preflight(self):
        script = Path(__file__).with_name("run_last4_dpt.sh").read_text()
        preflight = script.split("python -u - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(pair_id=split, split=split, patient_key=split)
                       for split in ("train", "val", "test")]
            (root / "manifest_split.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records), encoding="utf-8")
            for layers in (1, 4):
                cache = root / str(layers)
                for record in records:
                    folder = cache / record["pair_id"]
                    folder.mkdir(parents=True)
                    torch.save(dict(p_prior=torch.zeros(1369, 768 * layers).half(),
                                    p_curr=torch.zeros(1369, 768 * layers).half(),
                                    cls_prior=torch.zeros(768).half(), cls_curr=torch.zeros(768).half(),
                                    gt_diff=torch.zeros(1, 512, 512).half()), folder / "feat.pt")
                with patch.dict(os.environ, DATA=str(root), CACHE_DIR=str(cache), CACHE_LAYERS=str(layers)), \
                        patch.object(torch.cuda, "is_available", return_value=True), \
                        patch.object(torch.cuda, "mem_get_info", return_value=(12_000_000_000, 24_000_000_000)):
                    exec(compile(preflight, "launcher_preflight", "exec"), {})
                    if layers == 4:
                        with patch.dict(os.environ, CACHE_LAYERS="1"), self.assertRaises(SystemExit):
                            exec(compile(preflight, "launcher_preflight", "exec"), {})

    def test_compact_exact_resumable_slice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            (source / "pair").mkdir(parents=True)
            (root / "manifest_split.jsonl").write_text(
                json.dumps(dict(pair_id="pair", split="train")), encoding="utf-8")
            cached = dict(p_prior=torch.randn(25, 64).half(), p_curr=torch.randn(25, 64).half(),
                          cls_prior=torch.randn(16).half(), cls_curr=torch.randn(16).half(),
                          gt_diff=torch.randn(1, 32, 32).half(), pair_id="pair",
                          anomaly_type=1, direction=1, is_pathology=1, change_group_id="group")
            source_path = source / "pair" / "feat.pt"
            torch.save(cached, source_path)
            before = source_path.read_bytes()
            with patch.multiple(trainer.C, NUM_PATCH_TOKENS=25, BACKBONE_DIM=16):
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 1, "skipped": 0})
                self.assertEqual(slice_last_layer_cache(root, source, target), {"written": 0, "skipped": 1})
                result = torch.load(target / "pair" / "feat.pt", weights_only=True)
                for key in ("p_prior", "p_curr"):
                    torch.testing.assert_close(result[key], cached[key][..., -16:])
                    self.assertEqual(result[key].untyped_storage().nbytes(), result[key].numel() * 2)
                torch.testing.assert_close(result["gt_diff"], cached["gt_diff"])
                self.assertEqual(source_path.read_bytes(), before)
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, source)
                result["p_prior"].zero_()
                torch.save(result, target / "pair" / "feat.pt")
                with self.assertRaises(ValueError):
                    slice_last_layer_cache(root, source, target)
                source_path.unlink()
                with self.assertRaises(FileNotFoundError):
                    slice_last_layer_cache(root, source, target)


if __name__ == "__main__":
    unittest.main()