import unittest

from tools.summarize_office_home_pair_flow import DUET, TASKS, summarize_rows


def make_rows(mean_offset=0.2, active_rank=4):
    return [
        {
            "method": "temporal_precision_head_seed2022_pair_flow_probe",
            "task": task,
            "record_type": "standard",
            "selection": "peak",
            "accuracy": str(DUET[task] + mean_offset),
            "target_head_variant": "pair_flow",
            "pair_flow_active_rank": str(active_rank),
            "pair_flow_gate_final": "0.04",
        }
        for task in TASKS
    ]


class PairFlowSummaryTest(unittest.TestCase):
    def test_passes_accuracy_and_mechanism_gates(self):
        summary = summarize_rows(make_rows(), 84.72, 10, -1.5)

        self.assertEqual(summary["decision"], "pass_seed2022_gate")
        self.assertEqual(summary["active_pair_flow_tasks"], 12)

    def test_fails_when_basis_never_activates(self):
        summary = summarize_rows(make_rows(active_rank=0), 84.72, 10, -1.5)

        self.assertEqual(summary["decision"], "fail_seed2022_gate")
        self.assertFalse(summary["checks"]["mechanism_passes"])


if __name__ == "__main__":
    unittest.main()
