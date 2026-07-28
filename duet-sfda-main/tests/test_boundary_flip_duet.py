import unittest

import torch

from src.utils.boundary_flip import (
    apply_pair_budget,
    boundary_flip_loss,
    dynamic_logit_adjustment,
    init_boundary_flip_state,
    update_boundary_flip_memory,
    update_boundary_flip_state,
)


class DynamicLogitAdjustmentTest(unittest.TestCase):
    def test_overrepresented_class_can_flip_a_boundary_prediction(self):
        probability = torch.tensor([[0.55, 0.45]])
        adjusted, penalty = dynamic_logit_adjustment(
            probability,
            torch.tensor([90.0, 10.0]),
            torch.tensor([0.5, 0.5]),
            alpha=0.15,
        )

        self.assertEqual(int(probability.argmax(dim=1).item()), 0)
        self.assertEqual(int(adjusted.argmax(dim=1).item()), 1)
        self.assertGreater(float(penalty[0, 0]), float(penalty[0, 1]))


class BoundaryFlipMemoryTest(unittest.TestCase):
    def test_stable_transition_is_accepted_then_interruption_is_rejected(self):
        state = init_boundary_flip_state(2, 3)
        eligible = torch.tensor([True, False])
        late = torch.tensor([1, 2])

        first = update_boundary_flip_memory(
            state, eligible, late, stable_cycles=2, max_switches=0
        )
        second = update_boundary_flip_memory(
            state, eligible, late, stable_cycles=2, max_switches=0
        )
        interrupted = update_boundary_flip_memory(
            state,
            torch.tensor([False, False]),
            late,
            stable_cycles=2,
            max_switches=0,
        )

        self.assertFalse(bool(first[0]))
        self.assertTrue(bool(second[0]))
        self.assertFalse(bool(interrupted[0]))
        self.assertEqual(int(state["switch_count"][0]), 1)

    def test_pair_budget_is_independent_for_each_ordered_pair(self):
        selected = apply_pair_budget(
            torch.tensor([True, True, True, True]),
            torch.tensor([0, 0, 0, 1]),
            torch.tensor([1, 1, 1, 0]),
            torch.tensor([0.1, 0.9, 0.2, 0.3]),
            num_classes=2,
            max_per_pair=1,
        )

        self.assertEqual(selected.tolist(), [False, True, False, True])


class BoundaryFlipPipelineTest(unittest.TestCase):
    def _update(self, state, cycle, semantic_threshold=-0.1):
        model = torch.tensor([[0.80, 0.20]])
        clip = torch.tensor([[0.30, 0.70]])
        source_label = torch.tensor([0])
        clip_label = torch.tensor([1])
        text = torch.eye(2)
        return update_boundary_flip_state(
            model,
            clip,
            source_label,
            clip_label,
            torch.tensor([False]),
            text,
            state,
            curr_cycle=cycle,
            start_cycle=1,
            alpha=0.15,
            min_adjusted_confidence=0.0,
            min_margin=0.0,
            semantic_threshold=semantic_threshold,
            stable_cycles=2,
            max_switches=0,
            max_per_pair=8,
            min_weight=0.05,
        )

    def test_candidate_requires_temporal_stability(self):
        state = init_boundary_flip_state(1, 2)
        state["class_count"][:] = torch.tensor([90.0, 10.0])
        state["class_confidence_sum"][:] = state["class_count"] * 0.5

        state, warmup = self._update(state, 0)
        state, pending = self._update(state, 1)
        _, stable = self._update(state, 2)

        self.assertFalse(bool(warmup["candidate_mask"][0]))
        self.assertTrue(bool(pending["candidate_mask"][0]))
        self.assertFalse(bool(pending["active_mask"][0]))
        self.assertTrue(bool(stable["active_mask"][0]))
        self.assertEqual(int(stable["initial_label"][0]), 0)
        self.assertEqual(int(stable["adjusted_label"][0]), 1)

    def test_semantic_gate_blocks_unrelated_class(self):
        state = init_boundary_flip_state(1, 2)
        state["class_count"][:] = torch.tensor([90.0, 10.0])
        state["class_confidence_sum"][:] = state["class_count"] * 0.5

        state, _ = self._update(state, 0, semantic_threshold=0.1)
        _, result = self._update(state, 1, semantic_threshold=0.1)

        self.assertFalse(bool(result["candidate_mask"][0]))


class BoundaryFlipLossTest(unittest.TestCase):
    def test_loss_pushes_late_up_and_early_down(self):
        logits = torch.zeros((1, 3), requires_grad=True)
        value = boundary_flip_loss(
            logits,
            torch.tensor([0]),
            torch.tensor([1]),
            torch.tensor([1.0]),
            negative_weight=0.5,
        )
        value.backward()

        self.assertGreater(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
