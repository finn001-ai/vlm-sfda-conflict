import unittest

from tools.analyze_visda_unlabeled_route_proxies import PROXY_FIELDS, evaluate_proxies


def row(name, confidence):
    result = {"class": name}
    for field in PROXY_FIELDS:
        result[field] = confidence if field == "teacher_confidence" else 0.5
    return result


class VisDAUnlabeledRouteProxiesTest(unittest.TestCase):
    def test_accepts_single_proxy_that_ranks_oracle_supported_classes(self):
        rows = [row("a", 0.9), row("b", 0.8), row("c", 0.2), row("d", 0.1)]
        gain = {"a": 4.0, "b": 3.0, "c": 0.0, "d": -1.0}

        result = evaluate_proxies(
            rows,
            gain,
            {"a", "b"},
            min_spearman=0.5,
            min_topk_overlap=2,
        )

        self.assertEqual(result["decision"], "supports_unlabeled_class_router")
        self.assertEqual(result["best_proxy"]["proxy"], "teacher_confidence")

    def test_rejects_constant_proxies(self):
        rows = [row("a", 0.5), row("b", 0.5), row("c", 0.5), row("d", 0.5)]
        gain = {"a": 4.0, "b": 3.0, "c": 0.0, "d": -1.0}

        result = evaluate_proxies(
            rows,
            gain,
            {"a", "b"},
            min_spearman=0.5,
            min_topk_overlap=2,
        )

        self.assertEqual(result["decision"], "rejects_unlabeled_class_router")


if __name__ == "__main__":
    unittest.main()
