import unittest

import numpy as np
import torch

from src.utils.conflict_boundary import (
    boundary_choice_and_separation,
    fixed_fraction_mask,
    paired_accuracy_bootstrap_ci,
    pairwise_first_order_boundary,
)


class ConflictBoundaryTest(unittest.TestCase):
    def test_first_order_radius_matches_linear_boundary(self):
        inputs = torch.tensor([[2.0, 1.0], [1.0, 3.0]], requires_grad=True)
        weights = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        logits = inputs @ weights
        own = torch.tensor([0, 1])
        other = torch.tensor([1, 0])

        radius, margin, gradient_norm = pairwise_first_order_boundary(
            logits, inputs, own, other
        )

        expected_norm = torch.full((2,), 5.0**0.5)
        expected_margin = torch.tensor([3.0, 1.0])
        self.assertTrue(torch.allclose(margin, expected_margin))
        self.assertTrue(torch.allclose(gradient_norm, expected_norm))
        self.assertTrue(torch.allclose(radius, expected_margin / expected_norm))
        self.assertIsNone(inputs.grad)

    def test_choice_and_top_fraction_do_not_depend_on_labels(self):
        task = torch.tensor([4.0, 1.0, 3.0, 1.0, 2.0])
        clip = torch.tensor([1.0, 4.0, 2.0, 2.0, 2.0])
        choose_task, separation = boundary_choice_and_separation(task, clip)
        selected = fixed_fraction_mask(separation, 0.4)

        self.assertEqual(choose_task.tolist(), [True, False, True, False, True])
        self.assertEqual(int(selected.sum()), 2)
        self.assertEqual(selected.tolist(), [True, True, False, False, False])

    def test_paired_bootstrap_detects_uniform_improvement(self):
        candidate = np.ones(20, dtype=bool)
        baseline = np.zeros(20, dtype=bool)
        low, high = paired_accuracy_bootstrap_ci(
            candidate, baseline, repeats=200, batch_size=20
        )
        self.assertEqual(low, 100.0)
        self.assertEqual(high, 100.0)


if __name__ == "__main__":
    unittest.main()
