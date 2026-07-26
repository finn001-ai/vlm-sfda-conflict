import unittest
from pathlib import Path

from tools.summarize_visda_stage14_prior_memory import (
    causal_conclusion,
    factorial_summary,
    parse_refresh_metrics,
    read_candidate_config,
    reproduction_summary,
    validate_candidate_config,
)


CONTROL_CLASSES = [
    97.5,
    84.0,
    85.0,
    80.0,
    96.5,
    94.0,
    93.0,
    84.0,
    92.0,
    93.5,
    91.0,
    63.0,
]


def run(name, final, *, calib_mode=None, pl_memory=None, coverage=90.0):
    config = None
    if calib_mode is not None:
        config = {
            "calib_mode": calib_mode,
            "calib_power": 0.5,
            "pl_memory": pl_memory,
            "target_head": True,
            "gtr_par": 0.05,
        }
    return {
        "name": name,
        "log": f"{name}.txt",
        "final": final,
        "oracle_peak": final,
        "oracle_peak_cycle": 4,
        "oracle_peak_iteration": 100,
        "class_accuracy": CONTROL_CLASSES,
        "refresh": {
            "selected_count": round(55388 * coverage / 100),
            "total_count": 55388,
            "coverage": coverage,
            "selected_source_label_precision": 92.0,
            "pseudo_label_precision": 92.0,
            "mix_accuracy": 89.0,
        },
        "config": config,
    }


class VisdaStage14PriorMemoryTest(unittest.TestCase):
    def setUp(self):
        self.control = run("duet", 90.33)

    def test_reproduction_gate_requires_material_negative_gap(self):
        reproduced = reproduction_summary(
            self.control,
            run(
                "both_prior_stable",
                89.98,
                calib_mode="both_prior",
                pl_memory="stable",
            ),
            -0.15,
        )
        not_reproduced = reproduction_summary(
            self.control,
            run(
                "both_prior_stable",
                90.25,
                calib_mode="both_prior",
                pl_memory="stable",
            ),
            -0.15,
        )

        self.assertTrue(reproduced["gap_reproduced"])
        self.assertEqual(
            reproduced["decision"], "gap_reproduced_run_factorial"
        )
        self.assertFalse(not_reproduced["gap_reproduced"])
        self.assertEqual(
            not_reproduced["decision"], "gap_not_reproduced_stop"
        )

    def test_factorial_identifies_consistent_prior_effect(self):
        runs = {
            "both_prior_stable": run(
                "both_prior_stable",
                89.98,
                calib_mode="both_prior",
                pl_memory="stable",
            ),
            "none_stable": run(
                "none_stable",
                90.48,
                calib_mode="none",
                pl_memory="stable",
            ),
            "both_prior_monotonic": run(
                "both_prior_monotonic",
                90.00,
                calib_mode="both_prior",
                pl_memory="monotonic",
            ),
            "none_monotonic": run(
                "none_monotonic",
                90.45,
                calib_mode="none",
                pl_memory="monotonic",
            ),
        }

        result = factorial_summary(
            self.control,
            runs,
            max_reproduction_delta=-0.15,
            min_material_effect=0.10,
            min_duet_gain=0.10,
        )

        self.assertEqual(
            result["causal_conclusion"],
            "both_prior_is_primary_consistent_cause",
        )
        self.assertEqual(result["best_ablation"], "none_stable")
        self.assertEqual(result["decision"], "candidate_beats_duet_at_cycle4")

    def test_factorial_reports_prior_memory_interaction(self):
        effects = {
            "remove_prior_with_stable": 0.02,
            "remove_prior_with_monotonic": 0.45,
            "remove_prior_average": 0.235,
            "monotonic_with_both_prior": 0.03,
            "monotonic_with_no_prior": 0.46,
            "monotonic_average": 0.245,
            "interaction_remove_prior_x_monotonic": 0.43,
            "remove_both_jointly": 0.48,
        }

        self.assertEqual(
            causal_conclusion(effects, 0.10),
            "prior_memory_interaction_is_primary_cause",
        )

    def test_parses_full_data_refresh_and_fixed_config(self):
        refresh_lines = []
        for cycle in range(1, 5):
            refresh_lines.extend(
                [
                    (
                        "Number of valid pseudo-labeled samples: "
                        f"{50000 + cycle}/55388; Accuracy = 92.00%"
                    ),
                    "Mixed output with valid mask: 92.00%",
                    "all_mix_output Accuracy = 89.00%;",
                ]
            )
        refresh = parse_refresh_metrics(
            "\n".join(refresh_lines), "both_prior_stable"
        )
        config = read_candidate_config(
            "\n".join(
                [
                    "  CALIB_MODE: both_prior",
                    "  CALIB_POWER: 0.5",
                    "  PL_MEMORY: stable",
                    "  TARGET_HEAD_ADAPT: True",
                    "  GTR_PAR: 0.05",
                ]
            ),
            "both_prior_stable",
        )

        self.assertEqual(refresh["total_count"], 55388)
        self.assertEqual(refresh["selected_count"], 50004)
        validate_candidate_config(config, "both_prior_stable")

    def test_single_gpu_runner_does_not_override_string_gpu_id(self):
        script = Path(
            "tools/run_visda_stage14_prior_memory_full4.sh"
        ).read_text()

        self.assertNotIn("GPU_ID ", script)


if __name__ == "__main__":
    unittest.main()
