import unittest

import torch
import torch.nn.functional as F

from src.utils.agreement_whitened_transport import AgreementWhitenedTransport


def make_rotated_problem(max_gate=0.5):
    weight = torch.eye(3)
    rotation = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    )
    labels = torch.arange(3).repeat_interleave(20)
    reference = weight[labels]
    offsets = torch.linspace(-0.03, 0.03, labels.numel()).unsqueeze(1)
    noise = torch.cat((offsets, -offsets, 0.5 * offsets), dim=1)
    features = F.normalize(reference @ rotation + noise, dim=1)
    transport = AgreementWhitenedTransport(
        num_classes=3,
        feature_dim=3,
        min_anchors=30,
        shrinkage=0.1,
        holdout_ratio=0.2,
        max_gate=max_gate,
        min_improvement=1e-4,
        epsilon=1e-6,
    )
    diagnostics = transport.fit(features, labels, labels, weight)
    return transport, features, labels, diagnostics


class AgreementWhitenedTransportTest(unittest.TestCase):
    def test_heldout_selection_activates_improving_global_map(self):
        transport, _, _, diagnostics = make_rotated_problem()

        self.assertEqual(diagnostics["anchors"], 60)
        self.assertEqual(diagnostics["active_classes"], 3)
        self.assertAlmostEqual(diagnostics["selected_strength"], 0.05)
        self.assertGreater(diagnostics["heldout_loss_improvement"], 0.0)
        self.assertGreaterEqual(diagnostics["heldout_accuracy_delta"], 0.0)
        self.assertEqual(sum(p.numel() for p in transport.parameters()), 0)

    def test_selected_transform_is_globally_bounded(self):
        transport, features, _, diagnostics = make_rotated_problem(max_gate=0.5)
        output = transport(features)
        relative_shift = (output - features).norm(dim=1) / features.norm(dim=1)

        self.assertGreater(float(relative_shift.mean()), 0.0)
        self.assertLessEqual(
            float(relative_shift.max()),
            diagnostics["selected_strength"] + 1e-6,
        )

    def test_forward_remains_differentiable(self):
        transport, features, _, _ = make_rotated_problem()
        selected = features[:2].clone().requires_grad_(True)

        transport(selected).sum().backward()

        self.assertIsNotNone(selected.grad)
        self.assertGreater(float(selected.grad.norm()), 0.0)

    def test_insufficient_agreements_selects_exact_identity(self):
        transport = AgreementWhitenedTransport(
            num_classes=3,
            feature_dim=3,
            min_anchors=10,
            shrinkage=0.1,
            holdout_ratio=0.2,
            max_gate=0.05,
            min_improvement=1e-3,
            epsilon=1e-6,
        )
        features = torch.eye(3)
        labels = torch.arange(3)

        diagnostics = transport.fit(features, labels, labels, torch.eye(3))

        self.assertEqual(diagnostics["selected_strength"], 0.0)
        self.assertTrue(torch.equal(transport(features), features))

    def test_geometry_can_only_be_frozen_once(self):
        transport, features, labels, _ = make_rotated_problem()

        with self.assertRaises(ValueError):
            transport.fit(features, labels, labels, torch.eye(3))


if __name__ == "__main__":
    unittest.main()
