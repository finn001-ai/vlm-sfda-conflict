import unittest

from tools.summarize_office_home_whitened_transport import (
    DUET as FULL_DUET,
    TASKS as FULL_TASKS,
    summarize as summarize_full,
)
from tools.summarize_whitened_transport_flow import summarize_log
from tools.summarize_whitened_transport_preflight import summarize


TASKS = ["AC", "PA", "RA"]


def make_row(task, accuracy):
    return {
        "task": task,
        "record_type": "standard",
        "selection": "peak",
        "accuracy": str(accuracy),
        "target_head_variant": "blend",
        "cov_transport_adapt": "True",
    }


def make_diagnostic(task):
    return {
        "task": task,
        "selected_strength": 0.025,
        "heldout_loss_improvement": 0.02,
        "mean_relative_shift": 0.024,
        "mechanism_valid": True,
        "config": {
            "min_anchors": 512,
            "shrinkage": 0.1,
            "holdout_ratio": 0.2,
            "max_gate": 0.05,
            "min_improvement": 0.001,
            "start_cycle": 1,
        },
    }


def make_flow(tasks=TASKS):
    return {
        "decision": "pass_whitened_diagnostics",
        "tasks": [make_diagnostic(task) for task in tasks],
    }


class WhitenedTransportSummaryTest(unittest.TestCase):
    def test_extracts_valid_label_free_selection(self):
        log = """
DCCL agreement-whitened transport enabled: min_anchors=512; shrinkage=0.1000; holdout_ratio=0.2000; max_gate=0.0500; min_improvement=0.001000; start_cycle=1
DCCL agreement-whitened geometry frozen: anchors=1500; train_anchors=1200; heldout_anchors=300; active_classes=55; selected_strength=0.025000; heldout_loss_improvement=0.020000; heldout_accuracy_delta=0.001000; mean_relative_shift=0.024000
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%
"""

        result = summarize_log(log, "whitened_method")

        self.assertTrue(result["mechanism_valid"])
        self.assertEqual(result["heldout_anchors"], 300)
        self.assertAlmostEqual(result["selected_strength"], 0.025)

    def test_zero_selection_fails_mechanism_gate(self):
        log = """
DCCL agreement-whitened transport enabled: min_anchors=512; shrinkage=0.1000; holdout_ratio=0.2000; max_gate=0.0500; min_improvement=0.001000; start_cycle=1
DCCL agreement-whitened geometry frozen: anchors=1500; train_anchors=1200; heldout_anchors=300; active_classes=55; selected_strength=0.000000; heldout_loss_improvement=0.000000; heldout_accuracy_delta=0.000000; mean_relative_shift=0.000000
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%
"""

        result = summarize_log(log, "whitened_method")

        self.assertFalse(result["mechanism_valid"])
        self.assertFalse(result["checks"]["label_free_selection_active"])

    def test_preflight_passes_mechanism_and_accuracy(self):
        rows = [
            make_row("AC", 73.9),
            make_row("PA", 83.3),
            make_row("RA", 83.7),
        ]

        result = summarize(rows, make_flow())

        self.assertEqual(result["decision"], "pass_whitened_preflight")

    def test_full_gate_requires_all_tasks_and_mean(self):
        rows = [
            make_row(task, FULL_DUET[task] + 0.2) for task in FULL_TASKS
        ]

        result = summarize_full(rows, make_flow(FULL_TASKS))

        self.assertEqual(result["decision"], "pass_seed2022_gate")
        self.assertTrue(result["checks"]["mean_passes"])


if __name__ == "__main__":
    unittest.main()
