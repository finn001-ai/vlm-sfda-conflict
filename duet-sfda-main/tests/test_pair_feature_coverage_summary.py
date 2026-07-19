import unittest

from tools.summarize_office_home_pair_feature_coverage import (
    DUET,
    TASKS,
    summarize_rows,
)
from tools.summarize_pair_feature_coverage_preflight import (
    STAGE19,
    summarize_rows as summarize_preflight,
)


def make_row(task, accuracy, rank, effective):
    return {
        "method": "coverage_method",
        "task": task,
        "record_type": "standard",
        "selection": "peak",
        "accuracy": str(accuracy),
        "target_head_variant": "blend",
        "pair_feature_adapt": "True",
        "pair_feature_min_active_rank": "8",
        "pair_flow_active_rank": str(rank),
        "pair_feature_effective": str(effective),
        "pair_feature_router_norm": "0.2" if effective else "0.0",
    }


class PairFeatureCoverageSummaryTest(unittest.TestCase):
    def test_preflight_passes_recovered_target_art_tasks(self):
        rows = [
            make_row("CA", 83.68, 2, False),
            make_row("PA", 83.11, 2, False),
            make_row("RA", 83.52, 1, False),
        ]

        summary = summarize_preflight(rows)

        self.assertEqual(summary["decision"], "pass_coverage_preflight")
        self.assertGreater(summary["projected_full_mean_diagnostic"], 84.7225)

    def test_full_gate_accepts_active_and_fallback_paths(self):
        rows = []
        for index, task in enumerate(TASKS):
            fallback = task in {"CA", "PA", "RA"}
            rows.append(
                make_row(
                    task,
                    DUET[task] + 0.2,
                    2 if fallback else 12,
                    not fallback,
                )
            )

        summary = summarize_rows(rows)

        self.assertEqual(summary["decision"], "pass_seed2022_gate")
        self.assertEqual(summary["coverage_fallback_tasks"], 3)

    def test_full_gate_rejects_router_on_undercovered_basis(self):
        rows = [make_row(task, DUET[task] + 0.2, 12, True) for task in TASKS]
        rows[0] = make_row("AC", DUET["AC"] + 0.2, 2, True)

        summary = summarize_rows(rows)

        self.assertEqual(summary["decision"], "fail_seed2022_gate")
        self.assertFalse(summary["checks"]["coverage_policy_valid"])


if __name__ == "__main__":
    unittest.main()
