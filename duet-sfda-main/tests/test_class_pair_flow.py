import unittest

import torch

from src.utils.class_pair_flow import (
    ClassPairFlowAdapter,
    update_class_pair_flow,
)


class ClassPairFlowTest(unittest.TestCase):
    def test_basis_uses_persistent_resolved_candidate_directions(self):
        source = torch.tensor([0, 0, 2, 2, 1, 1])
        clip = torch.tensor([1, 1, 1, 1, 2, 2])
        resolved = torch.tensor([1, 1, 2, 2, 0, 0])
        mask = torch.ones(6, dtype=torch.bool)

        state = None
        for _ in range(2):
            state = update_class_pair_flow(
                source,
                clip,
                resolved,
                mask,
                state,
                num_classes=3,
                rank=2,
                min_count=2,
                min_cycles=2,
            )

        self.assertTrue(state["frozen"])
        self.assertEqual(state["active_rank"], 2)
        self.assertEqual(set(state["pairs"]), {(0, 1), (1, 2)})
        self.assertTrue(torch.allclose(state["basis"].sum(dim=1), torch.zeros(2)))

    def test_frozen_basis_ignores_later_pair_changes(self):
        source = torch.tensor([0, 0])
        clip = torch.tensor([1, 1])
        resolved = clip.clone()
        mask = torch.ones(2, dtype=torch.bool)
        state = None
        for _ in range(2):
            state = update_class_pair_flow(
                source, clip, resolved, mask, state, 3, 1, 2, 2
            )
        frozen_basis = state["basis"].clone()

        state = update_class_pair_flow(
            torch.tensor([2, 2]),
            torch.tensor([1, 1]),
            torch.tensor([1, 1]),
            mask,
            state,
            3,
            1,
            2,
            2,
        )

        self.assertTrue(torch.equal(state["basis"], frozen_basis))

    def test_opposite_directions_do_not_duplicate_the_basis(self):
        source = torch.tensor([0, 0, 0, 1, 1])
        clip = torch.tensor([1, 1, 1, 0, 0])
        resolved = clip.clone()
        mask = torch.ones(5, dtype=torch.bool)
        state = None
        for _ in range(2):
            state = update_class_pair_flow(
                source,
                clip,
                resolved,
                mask,
                state,
                num_classes=3,
                rank=2,
                min_count=2,
                min_cycles=2,
            )

        self.assertEqual(state["pairs"], [(0, 1)])
        self.assertEqual(state["active_rank"], 1)

    def test_selected_pair_edges_are_linearly_independent(self):
        source = torch.tensor([0] * 5 + [1] * 4 + [0] * 3)
        clip = torch.tensor([1] * 5 + [2] * 4 + [2] * 3)
        resolved = clip.clone()
        mask = torch.ones(source.size(0), dtype=torch.bool)
        state = None
        for _ in range(2):
            state = update_class_pair_flow(
                source,
                clip,
                resolved,
                mask,
                state,
                num_classes=4,
                rank=3,
                min_count=3,
                min_cycles=2,
            )

        active_basis = state["basis"][: state["active_rank"]]
        self.assertEqual(state["active_rank"], 2)
        self.assertEqual(int(torch.linalg.matrix_rank(active_basis)), 2)

    def test_zero_initialized_adapter_preserves_source_logits(self):
        adapter = ClassPairFlowAdapter(2, 3, 2, 0.3, -2.0, 1e-6)
        source_logits = torch.tensor([[2.0, 0.0, -1.0]])
        output = adapter(torch.tensor([[1.0, -1.0]]), source_logits)

        self.assertTrue(torch.equal(output, source_logits))

    def test_adapter_delta_is_bounded_and_pair_constrained(self):
        adapter = ClassPairFlowAdapter(2, 3, 1, 0.3, 20.0, 1e-6)
        adapter.set_basis(torch.tensor([[-2 ** -0.5, 2 ** -0.5, 0.0]]))
        adapter.coefficient.weight.data.fill_(10.0)
        source_logits = torch.tensor([[2.0, 0.0, -1.0]])
        output = adapter(torch.ones(1, 2), source_logits)
        delta = output - source_logits
        source_scale = source_logits.std(dim=1, keepdim=True, unbiased=False)

        self.assertAlmostEqual(float(delta[0, 2].detach()), 0.0, places=6)
        self.assertTrue(torch.all(delta.abs() <= 0.3 * source_scale + 1e-6))


if __name__ == "__main__":
    unittest.main()
