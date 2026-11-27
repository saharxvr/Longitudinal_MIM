"""Few-shot change heads: shapes, support permutation invariance, variable n, gradients."""

import unittest

import torch

from models.few_shot_head import (MATCHING_TYPES, FewShotChangeHead, grid_labels,
                                  log_uniform_support_size)


def tokens(count, grid=6, layers=3, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (count, grid * grid, layers * dim)
    return (torch.randn(shape, generator=generator), torch.randn(shape, generator=generator),
            torch.randn(count, layers * dim, generator=generator),
            torch.randn(count, layers * dim, generator=generator))


def head(matching, **kwargs):
    torch.manual_seed(0)
    return FewShotChangeHead(matching=matching, backbone_dim=16, grid=6, channels=16,
                             num_heads=2, out_size=24, embed_dim=8, local_pool=2, **kwargs).eval()


class GridLabelTests(unittest.TestCase):
    def test_fractions_sum_to_one_and_ignore_is_excluded(self):
        gt = torch.zeros(1, 1, 12, 12)
        gt[..., :6, :6] = 0.5
        gt[..., 6:, 6:] = -0.5
        ignore = torch.zeros_like(gt, dtype=torch.bool)
        ignore[..., :6, 6:] = True
        labels = grid_labels(gt, ignore, grid=2)
        self.assertEqual(labels.shape, (1, 4, 3))
        torch.testing.assert_close(labels.sum(-1), torch.ones(1, 4))
        torch.testing.assert_close(labels[0, 0], torch.tensor([0., 1., 0.]))
        torch.testing.assert_close(labels[0, 3], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(labels[0, 1], torch.tensor([1., 0., 0.]))


class FewShotHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.query = tokens(2, seed=1)
        self.support = tokens(5, seed=2)
        gt = torch.zeros(5, 1, 24, 24)
        gt[:, :, 4:12, 4:12] = 0.3
        gt[1:3, :, 14:20, 10:20] = -0.3
        self.labels = grid_labels(gt, grid=6)

    def test_every_variant_handles_supports_and_empty_support(self):
        for matching in MATCHING_TYPES:
            with self.subTest(matching=matching):
                model = head(matching)
                logits = model(*self.query, *self.support[:2], self.labels)
                self.assertEqual(logits.shape, (2, 3, 24, 24))
                self.assertTrue(torch.isfinite(logits).all())
                empty = model(*self.query)
                self.assertEqual(empty.shape, (2, 3, 24, 24))

    def test_support_order_does_not_change_prediction(self):
        order = torch.tensor([3, 0, 4, 2, 1])
        for matching in MATCHING_TYPES:
            if matching == "none":
                continue
            with self.subTest(matching=matching):
                model = head(matching)
                with torch.no_grad():
                    first = model(*self.query, *self.support[:2], self.labels)
                    second = model(*self.query, self.support[0][order], self.support[1][order],
                                   self.labels[order])
                torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-5)

    def test_supports_change_prediction_and_receive_gradients(self):
        for matching in ("fsdiff", "dcama", "panet", "alpnet", "universeg"):
            with self.subTest(matching=matching):
                model = head(matching).train()
                flipped = self.labels[:, :, [0, 2, 1]]
                with torch.no_grad():
                    a = model(*self.query, *self.support[:2], self.labels)
                    b = model(*self.query, *self.support[:2], flipped)
                self.assertGreater(float((a - b).abs().max()), 1e-6)
                logits, aux = model(*self.query, *self.support[:2], self.labels, return_aux=True)
                loss = logits.logsumexp(1).mean() + aux["change_logits"].square().mean()
                if matching == "panet":
                    loss = loss + model.alignment_loss(aux["query_tokens"], logits,
                                                       aux["support_tokens"], self.labels)
                loss.backward()
                self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)
                frozen = [name for name, parameter in model.named_parameters()
                          if parameter.requires_grad and parameter.grad is None
                          and not name.startswith("decoder.z_proj")]
                frozen = [name for name in frozen
                          if not name.startswith(("task_encoder.null", "null_cross"))]
                self.assertEqual(frozen, [])

    def test_token_budget_keeps_changed_tokens(self):
        model = head("dcama", token_budget=40)
        memory, labels = model._select_tokens(torch.randn(5, 36, 16), self.labels)
        self.assertEqual(memory.shape, (40, 16))
        self.assertEqual(int((labels[:, 0] < 1).sum()), int((self.labels[..., 0] < 1).sum()))

    def test_support_size_sampler_covers_range(self):
        generator = torch.Generator().manual_seed(0)
        values = {log_uniform_support_size(16, generator) for _ in range(500)}
        self.assertEqual(min(values), 1)
        self.assertEqual(max(values), 16)


if __name__ == "__main__":
    unittest.main()
