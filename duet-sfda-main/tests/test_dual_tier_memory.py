import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils.pseudo_label_memory import (
    dual_tier_supervision,
    weighted_cross_entropy,
)
from tools.summarize_visda_stage14_dual_tier import (
    dual_tier_summary,
    parse_memory_dynamics,
)


def make_run(final, *, dual=False):
    run = {
        "name": "run",
        "log": "run.txt",
        "final": final,
        "oracle_peak": final,
        "oracle_peak_cycle": 4,
        "oracle_peak_iteration": 100,
        "cycle4_peak": final,
        "cycle4_peak_iteration": 100,
        "cycle4_peak_to_final": 0.0,
        "class_accuracy": [final] * 12,
        "refresh": {
            "coverage": 90.0,
            "pseudo_label_precision": 92.0,
            "mix_accuracy": 89.0,
        },
    }
    if dual:
        run["config"] = {"pl_memory": "dual_tier"}
        run["memory_dynamics"] = [
            {
                "cycle": cycle,
                "stable_memory": "reversible",
                "warmup": cycle == 1,
                "current": 48000,
                "stable": 26000,
                "pending": 22000,
                "conflict": 7388,
                "low_confidence": 0,
                "selected": 48000,
                "effective_weight": 35000.0,
                "pending_mean_weight": 0.4091,
            }
            for cycle in range(1, 5)
        ]
    return run


class DualTierMemoryTest(unittest.TestCase):
    def test_builds_three_states_and_confidence_weighted_pending(self):
        current = torch.tensor([True, True, False, True])
        stable = torch.tensor([True, False, False, False])
        confidence = torch.tensor([0.9, 0.8, 0.95, 0.4])

        selected, pending, weights = dual_tier_supervision(
            current,
            stable,
            confidence,
            0.5,
            warmup=False,
        )

        self.assertTrue(torch.equal(selected, current))
        self.assertTrue(torch.equal(pending, torch.tensor([False, True, False, True])))
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 0.4, 0.0, 0.2])))

    def test_cycle_one_warmup_preserves_full_current_ce(self):
        current = torch.tensor([True, True, False])
        stable = torch.tensor([False, False, False])
        confidence = torch.tensor([0.8, 0.6, 0.9])

        _, _, weights = dual_tier_supervision(
            current,
            stable,
            confidence,
            0.5,
            warmup=True,
        )

        self.assertTrue(torch.equal(weights, torch.tensor([1.0, 1.0, 0.0])))

    def test_weighted_ce_uses_effective_weight_denominator(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0])
        weights = torch.tensor([1.0, 0.25])
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        expected = (per_sample * weights).sum() / weights.sum()

        actual = weighted_cross_entropy(logits, labels, weights)

        self.assertTrue(torch.allclose(actual, expected))

    def test_memory_log_partition_is_verified(self):
        lines = []
        for cycle in range(1, 5):
            lines.append(
                "DCCL pseudo-label memory: mode=dual_tier; "
                "stable_memory=reversible; "
                f"warmup={int(cycle == 1)}; current=48000; "
                "stable=26000; pending=22000; conflict=7388; "
                "low_confidence=0; selected=48000; "
                "effective_weight=35000.0000; "
                "pending_mean_weight=0.409100"
            )

        rows = parse_memory_dynamics("\n".join(lines), "candidate")

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["pending"], 22000)
        self.assertFalse(rows[1]["warmup"])

    def test_summary_advances_only_when_final_beats_duet(self):
        runs = {
            "duet": make_run(90.32),
            "both_prior_stable": make_run(89.98),
            "none_stable": make_run(89.99),
            "both_prior_monotonic": make_run(90.07),
            "none_monotonic": make_run(90.17),
            "both_prior_dual_tier": make_run(90.25, dual=True),
            "none_dual_tier": make_run(90.45, dual=True),
        }

        result = dual_tier_summary(
            runs,
            min_memory_gain=0.10,
            min_duet_gain=0.10,
        )

        self.assertEqual(result["decision"], "dual_tier_beats_duet_advance")
        self.assertEqual(result["best_dual_tier"], "none_dual_tier")

    def test_runner_keeps_single_gpu_config_untouched(self):
        script = Path("tools/run_visda_stage14_dual_tier_full4.sh").read_text()

        self.assertNotIn("GPU_ID ", script)
        self.assertIn("DCCL.PL_PENDING_WEIGHT 0.5", script)
        self.assertIn("DCCL.PL_MEMORY dual_tier", script)


if __name__ == "__main__":
    unittest.main()
