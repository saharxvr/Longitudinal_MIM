"""Checks for exploratory postprocessing and exact legacy matching."""

import unittest

import numpy as np
from scipy.ndimage import find_objects, label

from evaluation.sweep_clinical_postprocessing import (
    STRUCTURE, clinical_tools, configurations, legacy_postprocess, masks_for, match_counts,
)


class PostprocessingTests(unittest.TestCase):
    def test_legacy_matcher_parity(self):
        tools = clinical_tools()
        tools.struct = STRUCTURE
        rng = np.random.default_rng(42)
        examples = [(rng.random((20, 23)) > .7, rng.random((20, 23)) > .7) for _ in range(100)]
        examples += [(np.zeros((5, 5)), np.zeros((5, 5))),
                     (np.ones((5, 5)), np.zeros((5, 5)))]
        for reference, prediction in examples:
            actual, _ = label(reference, STRUCTURE)
            predicted, count = label(prediction, STRUCTURE)
            self.assertEqual(match_counts(actual, find_objects(actual), predicted, count),
                             tools.get_pairwise_detections(reference, prediction))

    def test_sign_specific_area_filter_and_no_mutation(self):
        scores = np.zeros((512, 512), dtype=np.float32)
        scores[:10, :10] = .8
        scores[10:20, :10] = -.8
        before = scores.copy()
        positive, negative = masks_for(scores, "components_50", .5)
        self.assertEqual(positive.sum(), 100)
        self.assertEqual(negative.sum(), 100)
        self.assertFalse(any(mask.any() for mask in masks_for(scores, "components_250", .5)))
        np.testing.assert_array_equal(scores, before)
        self.assertEqual(len(configurations()), 39)

    def test_literal_legacy_suppresses_high_scores(self):
        import torch

        cleanup, _ = legacy_postprocess()
        scores = torch.zeros(512, 512)
        scores[:30, :30] = .9
        scores[60:90, :30] = -.5
        output = cleanup(scores)
        self.assertEqual(int((output > 0).sum()), 0)
        self.assertEqual(int((output < 0).sum()), 900)


if __name__ == "__main__":
    unittest.main()
