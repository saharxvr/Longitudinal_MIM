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

from evaluation.evaluate_rq1 import _tokens, run_eval
from losses import change_map_loss
from models import DPTCrossAttentionHead, DPTDifferenceHead, DifferenceHead, build_change_head
from training.train_foundation_diff import head_spec, parse_args
from training import train_foundation_diff as trainer
from training.cache_features import slice_last_layer_cache


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