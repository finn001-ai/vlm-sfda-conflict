import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.summarize_three_view_em_flow import summarize_log, summarize_paths
from tools.summarize_three_view_em_preflight import summarize
from tools.summarize_three_view_em_stability import (
    TASKS as STABILITY_TASKS,
    summarize as summarize_stability,
)


def accuracy_rows(value=84.0):
    return [
        {"task": task, "accuracy": str(value), "selection": "peak"}
        for task in ("AC", "PA", "RA")
    ]


def flow_task(task, valid=True):
    return {
        "task": task,
        "config": {
            "start_cycle": 1,
            "steps": 5,
            "dirichlet": 5.0,
            "min_class_anchors": 3,
            "par": 0.05,
            "gradient_scope": "target_head_only",
        },
        "mechanism_valid": valid,
        "final_head_loss": 0.2,
    }


def stability_records(value=85.0):
    rows = []
    diagnostics = []
    for seed in ("2020", "2021", "2022"):
        method = f"temporal_precision_head_seed{seed}_three_view_em"
        for task in STABILITY_TASKS:
            rows.append(
                {
                    "method": method,
                    "task": task,
                    "accuracy": str(value),
                    "selection": "peak",
                }
            )
            diagnostic = flow_task(task)
            diagnostic["method"] = method
            diagnostics.append(diagnostic)
    return rows, {"tasks": diagnostics}


class ThreeViewEMSummaryTest(unittest.TestCase):
    def test_flow_parser_accepts_complete_training_log(self):
        consensus = "\n".join(
            (
                "DCCL three-view EM consensus: cycle={}; anchors=900; "
                "active_classes=50; conflicts=1800; weighted_conflicts=1700; "
                "mean_conflict_weight=0.420000; changed_top1=300; "
                "source_diag=0.800000; clip_diag=0.810000; graph_diag=0.790000"
            ).format(cycle)
            for cycle in (2, 3, 4)
        )
        log = (
            "DCCL three-view EM enabled: start_cycle=1; steps=5; "
            "dirichlet=5.000; min_class_anchors=3; par=0.050; "
            "gradient_scope=target_head_only\n"
            f"{consensus}\n"
            "Task: AC, Iter:100/100; Accuracy = 74.00%; "
            "three_view_em_loss=0.123000; three_view_em_batches=17\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "stage22.txt"
            path.write_text(log)
            result = summarize_log(path)
        self.assertEqual(result["task"], "AC")
        self.assertTrue(result["mechanism_valid"])
        self.assertEqual(result["final_head_loss_batches"], 17)

    def test_preflight_passes_valid_high_accuracy_rows(self):
        flow = {
            "decision": "pass_three_view_em_diagnostics",
            "tasks": [flow_task(task) for task in ("AC", "PA", "RA")],
        }
        result = summarize(accuracy_rows(), flow)
        self.assertEqual(result["decision"], "pass_three_view_em_preflight")

    def test_preflight_fails_invalid_mechanism(self):
        flow = {
            "decision": "fail_three_view_em_diagnostics",
            "tasks": [flow_task("AC", False), flow_task("PA"), flow_task("RA")],
        }
        result = summarize(accuracy_rows(), flow)
        self.assertEqual(result["decision"], "fail_three_view_em_preflight")
        self.assertFalse(result["checks"]["mechanism_valid"])

    def test_flow_parser_requires_paths(self):
        result = summarize_paths([])
        self.assertEqual(result["decision"], "fail_three_view_em_diagnostics")

    def test_stability_requires_valid_fixed_mechanism(self):
        rows, flow = stability_records()
        result = summarize_stability(rows, flow)
        self.assertEqual(result["decision"], "pass_three_view_em_stability_gate")
        flow["tasks"][0]["mechanism_valid"] = False
        result = summarize_stability(rows, flow)
        self.assertEqual(result["decision"], "fail_three_view_em_stability_gate")

    def test_stability_rejects_duplicate_records(self):
        rows, flow = stability_records()
        rows.append(dict(rows[0]))
        with self.assertRaises(ValueError):
            summarize_stability(rows, flow)


if __name__ == "__main__":
    unittest.main()
