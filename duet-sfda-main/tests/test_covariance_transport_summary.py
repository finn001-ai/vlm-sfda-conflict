import unittest

from tools.summarize_covariance_transport_flow import summarize_log
from tools.summarize_covariance_transport_preflight import summarize
from tools.summarize_office_home_covariance_transport import (
    DUET as FULL_DUET,
    TASKS as FULL_TASKS,
    summarize as summarize_full,
)


TASKS = ["AC", "PA", "RA"]


def make_row(task, accuracy):
    return {
        "task": task,
        "record_type": "standard",
        "selection": "peak",
        "accuracy": str(accuracy),
        "target_head_variant": "blend",
        "cov_transport_adapt": "True",
        "cov_transport_min_anchors": "8",
        "cov_transport_rank": "4",
        "cov_transport_max_gate": "0.05",
    }


def make_flow():
    return {
        "decision": "pass_transport_diagnostics",
        "tasks": [
            {
                "task": task,
                "active_classes": 40,
                "eligible_coverage": 0.7,
                "mean_relative_shift": 0.04,
            }
            for task in TASKS
        ],
    }


class CovarianceTransportSummaryTest(unittest.TestCase):
    def test_extracts_valid_geometry(self):
        log = """
DCCL agreement covariance geometry frozen: anchors=1200; active_classes=40; fixed_conflicts=2400; eligible_conflicts=1700; eligible_coverage=0.708333; mean_relative_shift=0.040000
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%
"""

        summary = summarize_log(log, "covariance_method")

        self.assertTrue(summary["mechanism_valid"])
        self.assertEqual(summary["active_classes"], 40)
        self.assertAlmostEqual(summary["eligible_coverage"], 0.708333)

    def test_preflight_passes_mechanism_and_accuracy(self):
        rows = [
            make_row("AC", 73.9),
            make_row("PA", 83.3),
            make_row("RA", 83.7),
        ]

        summary = summarize(rows, make_flow())

        self.assertEqual(summary["decision"], "pass_covariance_preflight")
        self.assertTrue(summary["checks"]["mean_beats_matched_base"])

    def test_preflight_fails_inactive_geometry(self):
        rows = [make_row("AC", 73.9), make_row("PA", 83.3), make_row("RA", 83.7)]
        flow = make_flow()
        flow["decision"] = "fail_transport_diagnostics"

        summary = summarize(rows, flow)

        self.assertEqual(summary["decision"], "fail_covariance_preflight")
        self.assertFalse(summary["checks"]["mechanism_valid"])

    def test_full_gate_passes_complete_valid_run(self):
        rows = []
        flow_tasks = []
        for task in FULL_TASKS:
            row = make_row(task, FULL_DUET[task] + 0.2)
            rows.append(row)
            flow_tasks.append(
                {
                    "task": task,
                    "active_classes": 40,
                    "eligible_coverage": 0.7,
                    "mean_relative_shift": 0.04,
                }
            )
        flow = {
            "decision": "pass_transport_diagnostics",
            "tasks": flow_tasks,
        }

        summary = summarize_full(rows, flow)

        self.assertEqual(summary["decision"], "pass_seed2022_gate")
        self.assertTrue(summary["checks"]["mean_passes"])


if __name__ == "__main__":
    unittest.main()
