import unittest

from tools.summarize_visda_matched_preflight import summarize_preflight


def records(max_cycle, accuracies):
    return [
        {
            "cycle": cycle,
            "max_cycle": max_cycle,
            "accuracy": accuracy,
        }
        for cycle, accuracy in accuracies
    ]


class VisDAMatchedPreflightTest(unittest.TestCase):
    def setUp(self):
        self.baseline = records(
            8,
            [(1, 86.01), (2, 88.49), (3, 89.51), (4, 90.15), (8, 91.07)],
        )

    def test_passes_candidate_with_sufficient_matched_and_projected_gain(self):
        candidate = records(4, [(1, 86.1), (2, 88.8), (3, 89.9), (4, 90.5)])

        result = summarize_preflight(self.baseline, candidate, 0.3, 0.4)

        self.assertEqual(result["decision"], "pass_full_training_gate")
        self.assertAlmostEqual(result["matched_improvement"], 0.35)
        self.assertAlmostEqual(result["projected_candidate_full_peak"], 91.42)

    def test_rejects_gain_too_small_to_project_past_reference(self):
        candidate = records(4, [(1, 86.1), (2, 88.6), (3, 89.8), (4, 90.4)])

        result = summarize_preflight(self.baseline, candidate, 0.3, 0.4)

        self.assertEqual(result["decision"], "fail_full_training_gate")
        self.assertTrue(result["checks"]["matched_improvement"])
        self.assertFalse(result["checks"]["projected_to_beat_reference"])

    def test_rejects_wrong_candidate_mix(self):
        candidate = records(4, [(1, 86.1), (2, 88.8), (3, 89.9), (4, 90.6)])

        result = summarize_preflight(self.baseline, candidate, 0.3, 0.5)

        self.assertEqual(result["decision"], "fail_full_training_gate")
        self.assertFalse(result["checks"]["config_valid"])

    def test_late_window_validates_stability_config_and_mechanism(self):
        baseline = records(
            8,
            [
                (1, 86.01),
                (2, 88.49),
                (3, 89.51),
                (4, 90.15),
                (5, 90.36),
                (8, 91.07),
            ],
        )
        candidate = records(
            5,
            [(1, 85.9), (2, 88.2), (3, 89.4), (4, 90.4), (5, 90.7)],
        )

        result = summarize_preflight(
            baseline,
            candidate,
            0.3,
            0.3,
            matched_cycles=5,
            matched_start_cycle=4,
            min_improvement=0.2,
            baseline_pl_stable_cycles=2,
            candidate_pl_stable_cycles=3,
            baseline_gtr_stable_cycles=2,
            candidate_gtr_stable_cycles=3,
            expected_candidate_mix=0.3,
            expected_baseline_pl_stable_cycles=2,
            expected_candidate_pl_stable_cycles=3,
            expected_baseline_gtr_stable_cycles=2,
            expected_candidate_gtr_stable_cycles=3,
            mechanism_valid=True,
            pass_command="run stability full",
        )

        self.assertEqual(result["decision"], "pass_full_training_gate")
        self.assertEqual(result["matched_start_cycle"], 4)
        self.assertAlmostEqual(result["matched_improvement"], 0.34)
        self.assertAlmostEqual(result["projected_candidate_full_peak"], 91.41)
        self.assertEqual(result["next"], "run stability full")

    def test_rejects_failed_mechanism_gate(self):
        candidate = records(4, [(1, 86.1), (2, 88.8), (3, 89.9), (4, 90.5)])

        result = summarize_preflight(
            self.baseline,
            candidate,
            0.3,
            0.4,
            mechanism_valid=False,
        )

        self.assertEqual(result["decision"], "fail_full_training_gate")
        self.assertFalse(result["checks"]["mechanism_valid"])


if __name__ == "__main__":
    unittest.main()
