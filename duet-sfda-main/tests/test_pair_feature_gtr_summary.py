import unittest

from tools.summarize_pair_feature_gtr import (
    MATCHED_ONLINE,
    PREFLIGHT_TASKS,
    route_summary,
    summarize_full,
    summarize_preflight,
)
from tools.summarize_office_home_pair_feature import DUET, TASKS


def make_rows(offset=0.2):
    return [
        {
            "method": "pair_feature_gtr_preflight",
            "task": task,
            "record_type": "standard",
            "selection": "peak",
            "accuracy": str(MATCHED_ONLINE[task] + offset),
            "target_head_variant": "blend",
            "pair_feature_adapt": "True",
            "pair_feature_min_active_rank": "1",
            "pair_feature_gradient_mode": "gtr_only",
            "pair_feature_effective": "True",
            "pair_flow_active_rank": "2",
            "pair_feature_router_norm": "0.2",
            "pair_feature_gtr_active": "50",
            "pair_feature_gtr_loss": "0.3",
            "pair_feature_gtr_batches": "4",
        }
        for task in PREFLIGHT_TASKS
    ]


def make_full_rows(offset=0.2):
    rows = []
    for task in TASKS:
        row = make_rows()[0]
        row.update(
            {
                "task": task,
                "accuracy": str(DUET[task] + offset),
                "pair_feature_gate_final": "0.006",
            }
        )
        rows.append(row)
    return rows


class PairFeatureGtrSummaryTest(unittest.TestCase):
    def test_preflight_passes_with_valid_route_and_accuracy(self):
        summary = summarize_preflight(make_rows())

        self.assertEqual(summary["decision"], "pass_gtr_preflight")
        self.assertTrue(summary["checks"]["gtr_route_passes"])

    def test_route_fails_when_generic_losses_train_adapter(self):
        rows = make_rows()
        rows[0]["pair_feature_gradient_mode"] = "joint"

        summary = route_summary(rows)

        self.assertEqual(summary["decision"], "fail_gtr_route")
        self.assertEqual(summary["valid_rows"], 2)

    def test_full_gate_requires_valid_gtr_route(self):
        rows = make_full_rows()
        rows[0]["pair_feature_gtr_batches"] = "0"

        summary = summarize_full(rows)

        self.assertEqual(summary["decision"], "fail_gtr_seed2022_gate")
        self.assertFalse(summary["checks"]["gtr_route_passes"])


if __name__ == "__main__":
    unittest.main()
