import unittest

import torch

from src.utils.probability_fusion import (
    arithmetic_probability_fusion,
    fuse_probabilities,
    rms_probability_fusion,
)


class ProbabilityFusionTest(unittest.TestCase):
    def setUp(self):
        self.task = torch.tensor(
            [[0.80, 0.10, 0.10], [0.20, 0.70, 0.10]], dtype=torch.float64
        )
        self.clip = torch.tensor(
            [[0.20, 0.70, 0.10], [0.10, 0.20, 0.70]], dtype=torch.float64
        )

    def test_arithmetic_matches_released_duet_formula(self):
        fused = arithmetic_probability_fusion(self.task, self.clip)
        self.assertTrue(torch.allclose(fused, (self.task + self.clip) / 2.0))

    def test_rms_matches_formula_and_is_probability_distribution(self):
        raw = torch.sqrt((self.task.square() + self.clip.square()) / 2.0)
        expected = raw / raw.sum(dim=1, keepdim=True)
        fused = rms_probability_fusion(self.task, self.clip)

        self.assertTrue(torch.allclose(fused, expected))
        self.assertTrue(torch.allclose(fused.sum(dim=1), torch.ones(2, dtype=torch.float64)))
        self.assertTrue(torch.all(fused >= 0))

    def test_rms_is_symmetric_and_preserves_identical_probabilities(self):
        self.assertTrue(
            torch.allclose(
                rms_probability_fusion(self.task, self.clip),
                rms_probability_fusion(self.clip, self.task),
            )
        )
        self.assertTrue(torch.allclose(rms_probability_fusion(self.task, self.task), self.task))

    def test_dispatch_rejects_unknown_mode(self):
        self.assertTrue(
            torch.allclose(
                fuse_probabilities(self.task, self.clip, mode="rms"),
                rms_probability_fusion(self.task, self.clip),
            )
        )
        with self.assertRaises(ValueError):
            fuse_probabilities(self.task, self.clip, mode="unknown")


if __name__ == "__main__":
    unittest.main()
