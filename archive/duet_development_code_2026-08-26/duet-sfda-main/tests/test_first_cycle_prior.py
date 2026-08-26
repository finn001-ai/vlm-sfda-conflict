import unittest

import torch

from src.utils.first_cycle_prior import (
    apply_first_cycle_prior,
    prior_calibrate,
)


class FirstCyclePriorTest(unittest.TestCase):
    def setUp(self):
        self.source = torch.tensor(
            [[0.90, 0.10], [0.80, 0.20], [0.60, 0.40]]
        )
        self.clip = torch.tensor(
            [[0.70, 0.30], [0.55, 0.45], [0.40, 0.60]]
        )

    def test_first_cycle_calibrates_both_views(self):
        source, clip, active = apply_first_cycle_prior(
            self.source,
            self.clip,
            curr_cycle=0,
            power=0.5,
            epsilon=1e-6,
        )

        self.assertTrue(active)
        self.assertTrue(
            torch.allclose(
                source,
                prior_calibrate(
                    self.source,
                    power=0.5,
                    epsilon=1e-6,
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                clip,
                prior_calibrate(
                    self.clip,
                    power=0.5,
                    epsilon=1e-6,
                ),
            )
        )
        self.assertFalse(torch.allclose(source, self.source))

    def test_later_cycles_are_exact_identity(self):
        source, clip, active = apply_first_cycle_prior(
            self.source,
            self.clip,
            curr_cycle=1,
            power=0.5,
            epsilon=1e-6,
        )

        self.assertFalse(active)
        self.assertIs(source, self.source)
        self.assertIs(clip, self.clip)

    def test_zero_power_keeps_probabilities(self):
        source, clip, active = apply_first_cycle_prior(
            self.source,
            self.clip,
            curr_cycle=0,
            power=0.0,
            epsilon=1e-6,
        )

        self.assertTrue(active)
        self.assertTrue(torch.allclose(source, self.source))
        self.assertTrue(torch.allclose(clip, self.clip))


if __name__ == "__main__":
    unittest.main()
