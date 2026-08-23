import unittest
from pathlib import Path

import torch

from src.utils.duet_anchored_consensus import (
    average_rank,
    centered_log_probability,
    consensus_shift_factors,
    entropy_weighted_poe,
    iic_mutual_information_loss,
    modulate_anchored_consensus,
    prediction_diversity_entropy,
)


class DuetAnchoredConsensusTest(unittest.TestCase):
    def test_centered_coordinate_preserves_distribution(self):
        probability = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]])
        centered = centered_log_probability(probability, epsilon=1e-8)
        self.assertTrue(
            torch.allclose(
                centered.mean(dim=1),
                torch.zeros(2),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(centered.softmax(dim=1), probability, atol=1e-6)
        )

    def test_equal_experts_are_exact_fixed_point(self):
        probability = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        result = entropy_weighted_poe(probability, probability, epsilon=1e-8)
        self.assertTrue(
            torch.allclose(result["probability"], probability, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(result["left_weight"], torch.full((2,), 0.5))
        )

    def test_more_concentrated_expert_receives_more_weight(self):
        concentrated = torch.tensor([[0.95, 0.03, 0.02]])
        diffuse = torch.tensor([[0.40, 0.32, 0.28]])
        result = entropy_weighted_poe(concentrated, diffuse)
        self.assertGreater(
            float(result["left_weight"].item()),
            float(result["right_weight"].item()),
        )

    def test_average_rank_handles_ties(self):
        values = torch.tensor([3.0, 1.0, 1.0, 2.0])
        expected = torch.tensor([3.0, 0.5, 0.5, 2.0])
        self.assertTrue(torch.equal(average_rank(values), expected))

    def test_csm_bounds_and_final_epoch_identity(self):
        probability = torch.tensor(
            [
                [0.99, 0.01],
                [0.80, 0.20],
                [0.55, 0.45],
            ]
        )
        first = consensus_shift_factors(
            probability,
            epoch=0,
            total_epochs=5,
            strength=0.5,
        )
        self.assertGreaterEqual(float(first["gamma"].min()), 0.5)
        self.assertLessEqual(float(first["gamma"].max()), 1.5)
        final = consensus_shift_factors(
            probability,
            epoch=4,
            total_epochs=5,
            strength=0.5,
        )
        self.assertTrue(torch.equal(final["gamma"], torch.ones(3)))

    def test_unit_gamma_returns_dynamic_consensus(self):
        anchor = torch.tensor([[1.0, -1.0], [0.2, -0.2]])
        dynamic = torch.tensor([[0.5, -0.5], [-0.3, 0.3]])
        result = modulate_anchored_consensus(
            anchor,
            dynamic,
            torch.ones(2),
        )
        self.assertTrue(torch.equal(result["centered"], dynamic))

    def test_iic_prefers_informative_joint_over_uniform_joint(self):
        prediction = torch.eye(3).repeat(2, 1)
        aligned = prediction.clone()
        uniform = torch.full_like(prediction, 1.0 / 3.0)
        aligned_loss = iic_mutual_information_loss(prediction, aligned)
        uniform_loss = iic_mutual_information_loss(prediction, uniform)
        self.assertLess(float(aligned_loss.item()), float(uniform_loss.item()))

    def test_losses_propagate_prediction_gradients(self):
        logits = torch.randn(8, 4, requires_grad=True)
        prediction = logits.softmax(dim=1)
        teacher = torch.randn(8, 4).softmax(dim=1)
        loss = iic_mutual_information_loss(prediction, teacher)
        loss = loss - prediction_diversity_entropy(prediction)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_path_only_optimizes_prompt_context(self):
        source = Path(
            "src/methods/oh/duet_anchored_consensus.py"
        ).read_text()
        self.assertIn(
            "prompt_parameters = [prompt_model.prompt_learner.ctx]",
            source,
        )
        self.assertNotIn(
            "optim.SGD(\n            prompt_model.prompt_learner.parameters()",
            source,
        )
        self.assertNotIn("prompt_model.prompt_learner.train()", source)

    def test_training_path_has_no_comparator_or_coverage_gate(self):
        source = Path(
            "src/methods/oh/duet_anchored_consensus.py"
        ).read_text()
        self.assertNotIn("PairwiseConflictComparator", source)
        self.assertNotIn("COMPARATOR_COVERAGE_FRACTION", source)
        self.assertIn("soft_coverage=100.00%", source)


if __name__ == "__main__":
    unittest.main()
