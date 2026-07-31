import unittest

from tools.analyze_duet_fcp_visda_proxy import analyze


def summary(accuracy, classes):
    return {
        "num_checkpoints": 16,
        "final": {
            "cycle": 4,
            "accuracy": accuracy,
            "class_accuracy": classes,
        },
    }


class DuetFCPVisDAProxyTest(unittest.TestCase):
    def setUp(self):
        self.control_classes = [90.0] * 12
        self.control = summary(87.93, self.control_classes)

    def test_passes_balanced_final_improvement(self):
        candidate = summary(88.13, [90.2] * 12)
        result = analyze(self.control, candidate)

        self.assertEqual(result["decision"], "pass_duet_fcp_proxy_gate")
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_hard_class_compensation(self):
        classes = [90.4] * 12
        classes[3] = 88.0
        candidate = summary(88.20, classes)
        result = analyze(self.control, candidate)

        self.assertEqual(result["decision"], "fail_duet_fcp_proxy_gate")
        self.assertFalse(result["checks"]["no_hard_class_compensation"])

    def test_rejects_incomplete_contract(self):
        candidate = summary(88.20, [90.2] * 12)
        candidate["num_checkpoints"] = 15
        result = analyze(self.control, candidate)

        self.assertFalse(result["checks"]["matched_contract"])


if __name__ == "__main__":
    unittest.main()
