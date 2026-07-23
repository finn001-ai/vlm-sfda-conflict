import unittest

from tools.summarize_visda_plmatch_proxy_control import summarize_control


def record(accuracy, classes, cycle=4, max_cycle=4, iteration=868):
    return {
        "iteration": iteration,
        "max_iteration": 868,
        "cycle": cycle,
        "max_cycle": max_cycle,
        "accuracy": accuracy,
        "class_accuracy": classes,
    }


class VisdaPlmatchProxyControlTest(unittest.TestCase):
    def test_reports_plmatch_above_dccl(self):
        classes = [90.0] * 12
        result = summarize_control([record(88.10, classes)])
        self.assertEqual(result["decision"], "plmatch_above_dccl")
        self.assertAlmostEqual(result["final_delta_vs_dccl"], 0.27)

    def test_reports_matched_control_inside_margin(self):
        classes = [
            97.56,
            86.01,
            85.61,
            75.36,
            96.27,
            95.81,
            93.50,
            80.68,
            91.91,
            94.56,
            91.34,
            65.34,
        ]
        result = summarize_control([record(87.83, classes)])
        self.assertEqual(result["decision"], "matched_within_margin")
        self.assertAlmostEqual(result["plmatch_hard_mean"], 73.7933, places=4)
        self.assertAlmostEqual(result["plmatch_other9_mean"], 92.5078, places=4)

    def test_rejects_incomplete_control(self):
        classes = [90.0] * 12
        with self.assertRaisesRegex(ValueError, "did not finish"):
            summarize_control(
                [record(87.83, classes, cycle=3, max_cycle=4)]
            )


if __name__ == "__main__":
    unittest.main()
