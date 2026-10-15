"""CPU checks: python -m unittest training.test_dpt_head."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.evaluate_rq1 import _tokens, run_eval
from losses import change_map_loss
from models import DPTDifferenceHead, DifferenceHead, build_change_head
from training.train_foundation_diff import head_spec, parse_args
from training import train_foundation_diff as trainer


class DPTHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.spec = {"type": "dpt", "kwargs": dict(
            backbone_dim=16, patch_dim=64, cls_dim=16, grid=5,
            channels=8, out_size=32, embed_dim=12)}
        self.prior = torch.randn(2, 25, 64)
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
            head(self.prior[..., :16], self.current[..., :16])
        with self.assertRaises(ValueError):
            head(self.prior[:, :-1], self.current[:, :-1])

    def test_default_head_and_config_override(self):
        self.assertIsInstance(build_change_head(head_spec(768)), DifferenceHead)
        with patch.object(sys, "argv", ["train", "--config", "configs/last4_dpt.yaml",
                                        "--batch_size", "2"]):
            args = parse_args()
        self.assertEqual((args.head, args.feat_last_k, args.batch_size), ("dpt", 4, 2))
        self.assertEqual((args.w_l1, args.w_l2, args.w_dice, args.w_sign), (1, 1, 0, 0))

    def test_evaluator_and_feature_slicing(self):
        batch = dict(p_prior=self.prior, p_curr=self.current,
                     cls_prior=self.cls_prior, cls_curr=self.cls_current,
                     gt_diff=torch.zeros(2, 1, 32, 32),
                     anomaly_type=torch.tensor([1, 0]), is_pathology=torch.tensor([1, 0]))
        batch["gt_diff"][0, 0, 8:20, 8:20] = 0.2
        sliced = _tokens(batch, None, "cpu", 16)
        torch.testing.assert_close(sliced[0], self.prior[..., -16:])
        report, _ = run_eval(build_change_head(self.spec), None, [batch], "cpu", 0.05, 64)
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
            arguments = ["train", "--config", "configs/last4_dpt.yaml",
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
            self.assertEqual(checkpoint["head_spec"]["type"], "dpt")
            self.assertEqual(len((root / "checkpoints" / "metrics.csv").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()