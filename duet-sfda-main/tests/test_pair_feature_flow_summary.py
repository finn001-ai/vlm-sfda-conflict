import unittest

from tools.summarize_pair_feature_flow import summarize_log


class PairFeatureFlowSummaryTest(unittest.TestCase):
    def test_extracts_activation_pairs_and_router_norm(self):
        log = """
DCCL fixed-candidate pair flow: cycle=1; valid=2400; candidate_mass=1700.50; active_rank=0; frozen=False; resolved_flow_mass=1700.50
DCCL fixed-candidate pair flow: cycle=2; valid=2400; candidate_mass=1800.25; active_rank=2; frozen=True; resolved_flow_mass=3500.75
DCCL pair-feature directions frozen: pairs=[(0, 1), (2, 3)]
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%; pair_feature_gate=0.006; pair_feature_router_norm=0.42; pair_flow_active_rank=2
"""
        summary = summarize_log(log, "pair_feature_method")

        self.assertEqual(summary["activation_cycle"], 2)
        self.assertEqual(summary["frozen_pairs"], [[0, 1], [2, 3]])
        self.assertTrue(summary["mechanism_active"])

    def test_accepts_exact_fallback_below_coverage_threshold(self):
        log = """
DCCL fixed-candidate pair flow: cycle=1; valid=1200; candidate_mass=700.0; active_rank=0; frozen=False; resolved_flow_mass=700.0
DCCL fixed-candidate pair flow: cycle=2; valid=1200; candidate_mass=750.0; active_rank=2; frozen=True; resolved_flow_mass=1450.0
DCCL pair-feature directions frozen: pairs=[(0, 1), (2, 3)]
Task: PA, Iter:40/40; Cycle: 4/4; Accuracy = 83.10%; pair_feature_gate=0.000000; pair_feature_router_norm=0.000000; pair_flow_active_rank=2; pair_feature_effective=False
"""
        summary = summarize_log(
            log,
            "coverage_method",
            min_active_rank=8,
            allow_fallback=True,
        )

        self.assertFalse(summary["mechanism_active"])
        self.assertTrue(summary["mechanism_valid"])
        self.assertFalse(summary["pair_feature_effective"])

        strict_summary = summarize_log(
            log, "coverage_method", min_active_rank=8
        )
        self.assertFalse(strict_summary["mechanism_valid"])


if __name__ == "__main__":
    unittest.main()
