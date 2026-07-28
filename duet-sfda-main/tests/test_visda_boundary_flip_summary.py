import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.analyze_visda_boundary_flip_duet import (
    read_visda_mechanism,
    summarize,
)


VALID_MECHANISM = {
    "snapshots": 8,
    "candidates": 100,
    "stable": 40,
    "active": 20,
    "loss_batches": 12,
    "positive_loss_records": 4,
    "valid": True,
}


class VisDABoundaryFlipSummaryTest(unittest.TestCase):
    def test_mechanism_requires_active_samples_and_real_loss_batches(self):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            np.savez(
                directory / "TV_cycle0.npz",
                boundary_flip_candidate_mask=np.array([True, True]),
                boundary_flip_stable_mask=np.array([True, False]),
                boundary_flip_active_mask=np.array([True, False]),
            )
            (directory / "run.txt").write_text(
                "boundary_flip_loss=0.125000; boundary_flip_batches=3\n"
            )

            mechanism = read_visda_mechanism(
                str(directory / "*.npz"), str(directory / "*.txt")
            )

        self.assertTrue(mechanism["valid"])
        self.assertEqual(mechanism["active"], 1)
        self.assertEqual(mechanism["loss_batches"], 3)

    def test_matched_gain_and_active_mechanism_pass(self):
        report = summarize(
            {"TV": 88.30},
            {"TV": 88.00},
            VALID_MECHANISM,
        )

        self.assertEqual(
            report["decision"], "pass_visda_boundary_flip_preflight"
        )
        self.assertAlmostEqual(report["delta"], 0.3)

    def test_inactive_loss_fails_even_with_accuracy_gain(self):
        mechanism = dict(VALID_MECHANISM, loss_batches=0, valid=False)
        report = summarize(
            {"TV": 89.00},
            {"TV": 88.00},
            mechanism,
        )

        self.assertEqual(
            report["decision"], "fail_visda_boundary_flip_preflight"
        )

    def test_candidate_only_reports_control_pending(self):
        report = summarize({"TV": 88.30}, {}, VALID_MECHANISM)

        self.assertEqual(
            report["decision"], "candidate_complete_control_pending"
        )


if __name__ == "__main__":
    unittest.main()
