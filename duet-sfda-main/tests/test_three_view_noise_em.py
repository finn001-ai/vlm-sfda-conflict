import unittest

import torch

from src.utils.three_view_noise_em import (
    three_view_class_conditional_em,
    weighted_soft_kl,
)


class ThreeViewNoiseEMTest(unittest.TestCase):
    def test_consensus_is_normalized_and_clamps_anchors(self):
        source = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        )
        clip = torch.tensor(
            [[0.85, 0.15], [0.3, 0.7], [0.25, 0.75], [0.1, 0.9]]
        )
        graph = torch.tensor(
            [[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8]]
        )
        base = (source + clip) / 2
        anchors = torch.tensor([True, False, False, True])
        labels = torch.tensor([0, 0, 1, 1])
        conflicts = source.argmax(1) != clip.argmax(1)

        posterior, weight, diagnostics = three_view_class_conditional_em(
            source,
            clip,
            graph,
            base,
            anchors,
            labels,
            conflicts,
            steps=3,
            dirichlet=1.0,
            min_class_anchors=1,
        )

        self.assertTrue(torch.allclose(posterior.sum(1), torch.ones(4)))
        self.assertTrue(torch.equal(posterior[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(posterior[3], torch.tensor([0.0, 1.0])))
        self.assertEqual(float(weight[0]), 0.0)
        self.assertGreater(float(weight[1]), 0.0)
        self.assertEqual(diagnostics["active_classes"], 2)

    def test_consensus_is_deterministic(self):
        generator = torch.Generator().manual_seed(4)
        raw = [torch.rand(12, 3, generator=generator) for _ in range(4)]
        source, clip, graph, base = [item / item.sum(1, keepdim=True) for item in raw]
        anchors = torch.tensor([True, True, True] + [False] * 9)
        labels = torch.tensor([0, 1, 2] + [0] * 9)
        conflicts = source.argmax(1) != clip.argmax(1)

        first = three_view_class_conditional_em(
            source, clip, graph, base, anchors, labels, conflicts,
            min_class_anchors=1,
        )
        second = three_view_class_conditional_em(
            source, clip, graph, base, anchors, labels, conflicts,
            min_class_anchors=1,
        )

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_minimum_anchor_support_controls_transition_estimation(self):
        generator = torch.Generator().manual_seed(7)
        raw = [torch.rand(12, 3, generator=generator) for _ in range(4)]
        source, clip, graph, base = [
            item / item.sum(1, keepdim=True) for item in raw
        ]
        anchors = torch.tensor([True, True, True] + [False] * 9)
        labels = torch.tensor([0, 1, 2] + [0] * 9)
        conflicts = source.argmax(1) != clip.argmax(1)

        supported = three_view_class_conditional_em(
            source, clip, graph, base, anchors, labels, conflicts,
            min_class_anchors=1,
        )
        unsupported = three_view_class_conditional_em(
            source, clip, graph, base, anchors, labels, conflicts,
            min_class_anchors=10,
        )

        self.assertEqual(supported[2]["active_classes"], 3)
        self.assertEqual(unsupported[2]["active_classes"], 0)
        self.assertFalse(torch.allclose(supported[0], unsupported[0]))

    def test_weighted_soft_kl_only_backpropagates_weighted_rows(self):
        logits = torch.zeros(2, 2, requires_grad=True)
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = weighted_soft_kl(logits, target, torch.tensor([1.0, 0.0]))
        loss.backward()

        self.assertGreater(float(logits.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)

    def test_rejects_missing_view(self):
        probability = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            three_view_class_conditional_em(
                probability,
                probability,
                probability[:, :1],
                probability,
                torch.tensor([True]),
                torch.tensor([0]),
                torch.tensor([False]),
            )


if __name__ == "__main__":
    unittest.main()
