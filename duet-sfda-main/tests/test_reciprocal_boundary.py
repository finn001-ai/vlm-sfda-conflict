import unittest

import torch

from src.utils.reciprocal_boundary import (
    ReciprocalBoundaryHead,
    reciprocal_boundary_consistency_loss,
    reciprocal_boundary_margin_loss,
    reciprocal_boundary_preservation_loss,
    update_reciprocal_boundary_state,
)


class ReciprocalBoundaryStateTest(unittest.TestCase):
    def test_persistent_unordered_pair_freezes_without_global_winner(self):
        source = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        clip = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1])
        state = None
        for _ in range(2):
            state = update_reciprocal_boundary_state(
                source,
                clip,
                state,
                num_classes=3,
                max_pairs=2,
                min_conflicts=4,
                min_cycles=2,
                min_anchors_per_side=2,
            )

        self.assertTrue(state["frozen"])
        self.assertEqual(state["pairs"], [(0, 1)])
        self.assertEqual(state["pair_anchor_counts"].tolist(), [[2, 2]])
        self.assertEqual(int(state["active_conflict_mask"].sum()), 4)

    def test_pair_does_not_activate_without_two_sided_anchors(self):
        source = torch.tensor([0, 0, 0, 0, 0, 0])
        clip = torch.tensor([1, 1, 1, 1, 0, 0])
        state = None
        for _ in range(2):
            state = update_reciprocal_boundary_state(
                source,
                clip,
                state,
                num_classes=2,
                max_pairs=1,
                min_conflicts=4,
                min_cycles=2,
                min_anchors_per_side=1,
            )

        self.assertFalse(state["frozen"])
        self.assertEqual(state["pairs"], [])


class ReciprocalBoundaryHeadTest(unittest.TestCase):
    def test_zero_initialized_head_is_exact_identity(self):
        head = ReciprocalBoundaryHead(4, 3, 3, 2, 0.5, 1e-6)
        head.set_pairs([(0, 1)])
        features = torch.randn(5, 4)
        logits = torch.randn(5, 3)

        self.assertTrue(torch.equal(head(features, logits), logits))

    def test_pair_residual_is_antisymmetric_and_bounded(self):
        head = ReciprocalBoundaryHead(2, 3, 2, 1, 0.5, 1e-6)
        head.set_pairs([(0, 1)])
        head.coefficient.weight.data.fill_(5.0)
        head.coefficient.bias.data.fill_(5.0)
        logits = torch.tensor([[2.0, 1.0, -1.0]])
        output = head(torch.ones(1, 2), logits)
        delta = output - logits
        scale = logits.std(dim=1, unbiased=False)

        self.assertAlmostEqual(
            float((delta[0, 0] + delta[0, 1]).detach()),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(float(delta[0, 2].detach()), 0.0, places=6)
        self.assertLessEqual(
            float(delta.abs().max().detach()),
            0.5 * float(scale) + 1e-6,
        )

    def test_overlapping_pairs_average_each_class_correction(self):
        head = ReciprocalBoundaryHead(2, 3, 2, 2, 0.5, 1e-6)
        head.set_pairs([(0, 1), (0, 2)])
        head.coefficient.weight.data.zero_()
        head.coefficient.bias.data.fill_(2.0)
        logits = torch.tensor([[1.0, 1.0, 1.0]])
        output = head(torch.ones(1, 2), logits)
        delta = output - logits

        self.assertAlmostEqual(float(delta.sum().detach()), 0.0, places=6)
        self.assertAlmostEqual(
            float(delta[0, 1].detach()),
            float(delta[0, 2].detach()),
            places=6,
        )
        self.assertAlmostEqual(
            float(delta[0, 0].detach()),
            -2.0 * float(delta[0, 1].detach()),
            places=6,
        )

    def test_detached_inference_residual_does_not_train_head(self):
        head = ReciprocalBoundaryHead(3, 2, 3, 1, 0.5, 1e-6)
        head.set_pairs([(0, 1)])
        head.coefficient.bias.data.fill_(0.5)
        features = torch.randn(4, 3, requires_grad=True)
        logits = torch.randn(4, 2, requires_grad=True)

        head(features, logits, detach_residual=True).sum().backward()

        self.assertIsNone(head.coefficient.weight.grad)
        self.assertIsNotNone(logits.grad)

    def test_boundary_only_path_does_not_train_backbone_inputs(self):
        head = ReciprocalBoundaryHead(3, 2, 3, 1, 0.5, 1e-6)
        head.set_pairs([(0, 1)])
        features = torch.randn(4, 3, requires_grad=True)
        logits = torch.randn(4, 2, requires_grad=True)

        output = head(features.detach(), logits.detach())
        output[:, 0].sum().backward()

        self.assertIsNotNone(head.coefficient.weight.grad)
        self.assertIsNone(features.grad)
        self.assertIsNone(logits.grad)


class ReciprocalBoundaryLossTest(unittest.TestCase):
    def test_margin_loss_balances_pair_sides_by_global_anchor_count(self):
        base = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        corrected = base.clone().requires_grad_()
        loss = reciprocal_boundary_margin_loss(
            corrected,
            base,
            torch.tensor([0, 1]),
            torch.tensor([True, True]),
            torch.tensor([[0, 1]]),
            torch.tensor([[10, 10]]),
            total_samples=20,
            margin=0.5,
            epsilon=1e-6,
        )

        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertAlmostEqual(
            float(corrected.grad[0, 0]),
            float(corrected.grad[1, 1]),
            places=6,
        )

    def test_margin_loss_supervises_residual_not_existing_base_margin(self):
        pairs = torch.tensor([[0, 1]])
        labels = torch.tensor([0, 1])
        mask = torch.tensor([True, True])
        counts = torch.tensor([[2, 2]])
        base_a = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        base_b = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

        loss_a = reciprocal_boundary_margin_loss(
            base_a.clone().requires_grad_(),
            base_a,
            labels,
            mask,
            pairs,
            counts,
            total_samples=2,
            margin=0.5,
            epsilon=1e-6,
        )
        loss_b = reciprocal_boundary_margin_loss(
            base_b.clone().requires_grad_(),
            base_b,
            labels,
            mask,
            pairs,
            counts,
            total_samples=2,
            margin=0.5,
            epsilon=1e-6,
        )

        self.assertAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), places=6)

    def test_consistency_loss_uses_only_matching_conflict_pair(self):
        weak = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
        strong = torch.tensor([[0.0, 2.0, -1.0], [2.0, 1.0, 0.0]])
        loss = reciprocal_boundary_consistency_loss(
            weak,
            strong,
            weak,
            strong,
            torch.tensor([0, 2]),
            torch.tensor([1, 1]),
            torch.tensor([[0, 1]]),
            epsilon=1e-6,
        )

        self.assertGreater(float(loss), 0.0)

    def test_preservation_loss_does_not_penalize_active_conflict(self):
        base = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        corrected = torch.tensor([[2.0, -1.0], [1.0, 0.0]])
        loss = reciprocal_boundary_preservation_loss(
            corrected,
            base,
            torch.tensor([False, True]),
            epsilon=1e-6,
        )

        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
