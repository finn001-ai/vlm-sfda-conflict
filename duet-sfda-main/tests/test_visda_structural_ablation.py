import unittest

from tools.summarize_visda_structural_ablation import (
    parse_refresh_metrics,
    summarize_ablation,
)


CONTROL_CLASSES = [
    97.45,
    83.63,
    84.82,
    80.42,
    96.87,
    93.98,
    93.91,
    84.18,
    92.46,
    93.56,
    91.19,
    62.74,
]


def records(final_accuracy, final_classes):
    rows = []
    for cycle in range(1, 5):
        for checkpoint in range(1, 5):
            is_final = cycle == 4 and checkpoint == 4
            rows.append(
                {
                    "iteration": checkpoint * 217,
                    "max_iteration": 868,
                    "cycle": cycle,
                    "max_cycle": 4,
                    "accuracy": (
                        final_accuracy if is_final else final_accuracy - 1.0
                    ),
                    "class_accuracy": final_classes,
                }
            )
    return rows


def refresh(coverage=94.0, precision=90.0, mix=88.0):
    total = 13847
    return {
        "num_refreshes": 4,
        "selected_count": round(total * coverage / 100.0),
        "total_count": total,
        "coverage": coverage,
        "selected_source_label_precision": precision - 0.5,
        "pseudo_label_precision": precision,
        "mix_accuracy": mix,
    }


class VisdaStructuralAblationTest(unittest.TestCase):
    def setUp(self):
        self.control = records(87.93, CONTROL_CLASSES)
        self.control_refresh = refresh(94.84, 90.36, 88.29)

    def test_selects_only_variant_passing_all_gates(self):
        improved = [value + 0.20 for value in CONTROL_CLASSES]
        candidates = {
            "v1_monotonic_head": records(88.13, improved),
            "v2_stable_nohead": records(87.90, CONTROL_CLASSES),
            "v3_monotonic_nohead": records(87.80, CONTROL_CLASSES),
        }
        result = summarize_ablation(
            self.control,
            candidates,
            self.control_refresh,
            {name: refresh() for name in candidates},
            {name: True for name in candidates},
        )

        self.assertEqual(result["decision"], "pass_proxy_gate")
        self.assertEqual(result["passing_variant"], "v1_monotonic_head")
        self.assertTrue(
            result["variants"]["v1_monotonic_head"]["pass_proxy_gate"]
        )

    def test_rejects_macro_gain_built_on_hard_class_compensation(self):
        compensated = [value + 0.40 for value in CONTROL_CLASSES]
        compensated[11] = CONTROL_CLASSES[11] - 0.60
        candidates = {
            "v1_monotonic_head": records(88.20, compensated),
            "v2_stable_nohead": records(87.90, CONTROL_CLASSES),
            "v3_monotonic_nohead": records(87.80, CONTROL_CLASSES),
        }
        result = summarize_ablation(
            self.control,
            candidates,
            self.control_refresh,
            {name: refresh() for name in candidates},
            {name: True for name in candidates},
        )

        v1 = result["variants"]["v1_monotonic_head"]
        self.assertEqual(result["decision"], "fail_proxy_gate")
        self.assertFalse(v1["checks"]["no_hard_class_compensation"])

    def test_rejects_invalid_candidate_configuration(self):
        improved = [value + 0.20 for value in CONTROL_CLASSES]
        candidates = {
            name: records(88.13, improved)
            for name in (
                "v1_monotonic_head",
                "v2_stable_nohead",
                "v3_monotonic_nohead",
            )
        }
        result = summarize_ablation(
            self.control,
            candidates,
            self.control_refresh,
            {name: refresh() for name in candidates},
            {
                "v1_monotonic_head": False,
                "v2_stable_nohead": False,
                "v3_monotonic_nohead": False,
            },
        )

        self.assertEqual(result["decision"], "fail_proxy_gate")
        self.assertIsNone(result["passing_variant"])

    def test_accepts_exact_final_improvement_boundary(self):
        improved = [value + 0.15 for value in CONTROL_CLASSES]
        candidates = {
            "v1_monotonic_head": records(88.08, improved),
            "v2_stable_nohead": records(87.90, CONTROL_CLASSES),
            "v3_monotonic_nohead": records(87.80, CONTROL_CLASSES),
        }
        result = summarize_ablation(
            self.control,
            candidates,
            self.control_refresh,
            {name: refresh() for name in candidates},
            {name: True for name in candidates},
        )

        self.assertEqual(result["decision"], "pass_proxy_gate")
        self.assertEqual(result["passing_variant"], "v1_monotonic_head")

    def test_rejects_incomplete_candidate(self):
        candidates = {
            "v1_monotonic_head": records(88.13, CONTROL_CLASSES)[:-1],
            "v2_stable_nohead": records(87.90, CONTROL_CLASSES),
            "v3_monotonic_nohead": records(87.80, CONTROL_CLASSES),
        }
        with self.assertRaisesRegex(ValueError, "16 checkpoints"):
            summarize_ablation(
                self.control,
                candidates,
                self.control_refresh,
                {name: refresh() for name in candidates},
                {name: True for name in candidates},
            )

    def test_parses_cycle_four_precision_coverage(self):
        text = "\n".join(
            [
                (
                    "Number of valid pseudo-labeled samples: "
                    f"{1000 + cycle}/2000; Accuracy = {90 + cycle:.2f}%"
                )
                + "\n"
                + f"Mixed output with valid mask: {90.5 + cycle:.2f}%"
                + "\n"
                + f"all_mix_output Accuracy = {80 + cycle:.2f}%;"
                for cycle in range(1, 5)
            ]
        )

        result = parse_refresh_metrics(text)

        self.assertEqual(result["selected_count"], 1004)
        self.assertEqual(result["total_count"], 2000)
        self.assertAlmostEqual(result["coverage"], 50.2)
        self.assertEqual(result["selected_source_label_precision"], 94.0)
        self.assertEqual(result["pseudo_label_precision"], 94.5)
        self.assertEqual(result["mix_accuracy"], 84.0)


if __name__ == "__main__":
    unittest.main()
