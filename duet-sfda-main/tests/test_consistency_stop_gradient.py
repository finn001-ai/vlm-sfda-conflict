import unittest

import torch

from src.utils.consistency import prediction_consistency_kl


class ConsistencyStopGradientTest(unittest.TestCase):
    def test_stop_gradient_updates_only_strong_branch(self):
        weak_logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
        )
        strong_logits = torch.tensor(
            [[0.5, 1.5], [1.5, 0.5]], requires_grad=True
        )

        loss = prediction_consistency_kl(
            weak_logits, strong_logits, stop_gradient=True
        )
        loss.backward()

        self.assertIsNone(weak_logits.grad)
        self.assertIsNotNone(strong_logits.grad)
        self.assertGreater(float(strong_logits.grad.abs().sum()), 0.0)

    def test_legacy_mode_updates_both_branches(self):
        weak_logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]], requires_grad=True
        )
        strong_logits = torch.tensor(
            [[0.5, 1.5], [1.5, 0.5]], requires_grad=True
        )

        loss = prediction_consistency_kl(
            weak_logits, strong_logits, stop_gradient=False
        )
        loss.backward()

        self.assertIsNotNone(weak_logits.grad)
        self.assertIsNotNone(strong_logits.grad)
        self.assertGreater(float(weak_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(strong_logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
