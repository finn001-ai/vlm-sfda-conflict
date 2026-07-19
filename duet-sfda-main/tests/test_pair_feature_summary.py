import unittest

from tools.summarize_office_home_pair_feature import DUET, TASKS, summarize_rows


def make_rows(mean_offset=0.2, active_rank=4, router_norm=0.2):
    return [
        {
            "method": "temporal_precision_head_seed2022_pair_feature_probe",
            "task": task,
            "record_type": "standard",
            "selection": "peak",
            "accuracy": str(DUET[task] + mean_offset),
            "target_head_variant": "blend",
            "pair_feature_adapt": "True",
            "pair_flow_active_rank": str(active_rank),
            "pair_feature_gate_final": "0.006",
            "pair_feature_router_norm": str(router_norm),
        }
        for task in TASKS
    ]


class PairFeatureSummaryTest(unittest.TestCase):
    def test_passes_accuracy_activation_and_training_gates(self):
        summary = summarize_rows(make_rows(), 84.72, 10, -1.5)

        self.assertEqual(summary["decision"], "pass_seed2022_gate")
        self.assertEqual(summary["trained_pair_feature_tasks"], 12)

    def test_fails_when_router_does_not_train(self):
        summary = summarize_rows(make_rows(router_norm=0.0), 84.72, 10, -1.5)

        self.assertEqual(summary["decision"], "fail_seed2022_gate")
        self.assertFalse(summary["checks"]["router_training_passes"])


if __name__ == "__main__":
    unittest.main()
