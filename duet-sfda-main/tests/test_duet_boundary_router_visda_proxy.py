import unittest

from tools.analyze_duet_boundary_router_visda_proxy import analyze


def summary(accuracy, classes):
    return {
        "num_checkpoints": 16,
        "final": {
            "cycle": 4,
            "accuracy": accuracy,
            "class_accuracy": classes,
        },
    }


class DuetBoundaryRouterVisDAProxyTest(unittest.TestCase):
    def setUp(self):
        self.control = summary(87.93, [90.0] * 12)

    def test_car_truck_exchange_is_reported_not_automatically_rejected(self):
        classes = [90.2] * 12
        classes[3] = 92.0
        classes[7] = 90.0
        classes[11] = 88.5
        result = analyze(self.control, summary(88.20, classes))

        self.assertEqual(result["decision"], "pass_boundary_router_proxy_gate")
        self.assertTrue(result["car_truck_exchange_observed"])
        self.assertTrue(all(result["checks"].values()))

    def test_rejects_hard_class_mean_regression(self):
        classes = [90.4] * 12
        classes[3] = 89.0
        classes[7] = 89.0
        classes[11] = 89.0
        result = analyze(self.control, summary(88.20, classes))

        self.assertEqual(result["decision"], "fail_boundary_router_proxy_gate")
        self.assertFalse(result["checks"]["hard_mean_noninferior"])

    def test_rejects_incomplete_contract(self):
        candidate = summary(88.20, [90.2] * 12)
        candidate["num_checkpoints"] = 15
        result = analyze(self.control, candidate)

        self.assertFalse(result["checks"]["matched_contract"])


if __name__ == "__main__":
    unittest.main()
