import unittest
from pathlib import Path

import torch

from src.utils.dcr_credit_memory import (
    initialize_delayed_credit,
    normalized_js_divergence,
    update_delayed_credit,
)
from src.utils.dcr_consensus import samplewise_distribution_alignment_loss


class DcrCreditMemoryTest(unittest.TestCase):
    def test_samplewise_alignment_handles_more_classes_than_samples(self):
        teacher_logits = torch.randn(4, 126)
        teacher = teacher_logits.softmax(dim=1)
        matched_logits = teacher_logits.detach().clone().requires_grad_(True)
        mismatched_logits = (-teacher_logits).detach().clone().requires_grad_(True)
        matched = samplewise_distribution_alignment_loss(
            matched_logits.softmax(dim=1), teacher
        )
        mismatched = samplewise_distribution_alignment_loss(
            mismatched_logits.softmax(dim=1), teacher
        )
        self.assertLess(float(matched.item()), float(mismatched.item()))
        matched.backward()
        self.assertTrue(torch.isfinite(matched_logits.grad).all())

    def test_normalized_js_is_symmetric_and_bounded(self):
        left = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
        right = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
        forward = normalized_js_divergence(left, right)
        reverse = normalized_js_divergence(right, left)
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-7))
        self.assertGreaterEqual(float(forward.min().item()), 0.0)
        self.assertLessEqual(float(forward.max().item()), 1.0)

    def test_identical_experts_are_a_fixed_point(self):
        probability = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        state = initialize_delayed_credit(probability, probability)
        updated, diagnostics = update_delayed_credit(
            state, probability, probability
        )
        self.assertTrue(torch.allclose(updated["memory"], probability, atol=1e-6))
        self.assertTrue(
            torch.allclose(updated["task_weight"], torch.full((2,), 0.5))
        )
        self.assertTrue(
            torch.allclose(diagnostics["memory_shift_l1"], torch.zeros(2))
        )

    def test_future_outcome_rewards_the_better_previous_expert(self):
        task = torch.tensor([[0.9, 0.1]])
        vlm = torch.tensor([[0.2, 0.8]])
        state = initialize_delayed_credit(task, vlm)
        outcome = torch.tensor([[0.82, 0.18]])
        updated, diagnostics = update_delayed_credit(
            state, outcome, outcome, credit_eta=4.0
        )
        self.assertLess(
            float(diagnostics["task_delayed_loss"].item()),
            float(diagnostics["clip_delayed_loss"].item()),
        )
        self.assertGreater(
            float(updated["task_weight"].item()),
            float(updated["clip_weight"].item()),
        )

    def test_uniform_ablation_keeps_equal_weights(self):
        task = torch.tensor([[0.9, 0.1]])
        vlm = torch.tensor([[0.2, 0.8]])
        state = initialize_delayed_credit(task, vlm)
        updated, _ = update_delayed_credit(
            state,
            torch.tensor([[0.82, 0.18]]),
            torch.tensor([[0.82, 0.18]]),
            credit_mode="uniform",
        )
        self.assertAlmostEqual(float(updated["task_weight"].item()), 0.5)
        self.assertAlmostEqual(float(updated["clip_weight"].item()), 0.5)

    def test_method_and_configs_use_only_dcr_names(self):
        source = Path("src/methods/oh/dcr_memory.py").read_text()
        self.assertIn("sample_self_history_only=True", source)
        self.assertIn("dcr_memory_state.pt", source)
        for path in (
            "cfgs/office-home/dcr.yaml",
            "cfgs/visda/dcr.yaml",
            "cfgs/domainnet126/dcr.yaml",
        ):
            config = Path(path).read_text()
            self.assertIn("METHOD: dcr_memory", config)
            self.assertIn("DCR_MEMORY:", config)
            self.assertIn("EPOCHS: 15", config)
            self.assertIn("CREDIT_MODE: delayed", config)
            self.assertIn("FEEDBACK_MODE: agreement_temporal", config)
            self.assertIn("HARD_LABEL_MODE: task_vlm_agreement", config)

    def test_all_formal_datasets_use_rank_adaptive_alignment(self):
        defaults = Path("conf.py").read_text()
        self.assertIn(
            '_C.DCR_MEMORY.ALIGNMENT_MODE = "rank_adaptive"',
            defaults,
        )
        for path in (
            "cfgs/office-home/dcr.yaml",
            "cfgs/visda/dcr.yaml",
            "cfgs/domainnet126/dcr.yaml",
        ):
            config = Path(path).read_text()
            self.assertIn("ALIGNMENT_MODE: rank_adaptive", config)
            self.assertIn("MIN_IIC_RANK_COVERAGE: 0.75", config)
            self.assertIn("DIVERSITY_DELTA: 0.1", config)


if __name__ == "__main__":
    unittest.main()
