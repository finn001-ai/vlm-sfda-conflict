import unittest

import torch

from src.utils.target_head import bounded_residual_logits


class TargetHeadTest(unittest.TestCase):
    def test_zero_residual_preserves_source_logits(self):
        source = torch.tensor([[1.0, -1.0, 0.5]])
        output = bounded_residual_logits(
            source,
            torch.zeros_like(source),
            torch.tensor(-2.0),
            max_gate=0.3,
            epsilon=1e-6,
        )
        self.assertTrue(torch.equal(output, source))

    def test_residual_displacement_is_bounded_by_source_scale(self):
        source = torch.tensor([[2.0, -1.0, 0.0]])
        residual = torch.tensor([[100.0, -100.0, 100.0]])
        output = bounded_residual_logits(
            source,
            residual,
            torch.tensor(20.0),
            max_gate=0.3,
            epsilon=1e-6,
        )
        source_scale = source.std(dim=1, keepdim=True, unbiased=False)
        self.assertTrue(torch.all((output - source).abs() <= 0.3 * source_scale + 1e-6))

    def test_rejects_unbounded_gate(self):
        with self.assertRaises(ValueError):
            bounded_residual_logits(
                torch.zeros(1, 2),
                torch.zeros(1, 2),
                torch.tensor(0.0),
                max_gate=1.1,
                epsilon=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
