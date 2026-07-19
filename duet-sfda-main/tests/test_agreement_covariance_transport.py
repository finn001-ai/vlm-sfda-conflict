import unittest

import torch

from src.utils.agreement_covariance_transport import (
    AgreementCovarianceTransport,
)


def make_problem(max_gate=0.1):
    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    source = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0])
    clip = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2])
    mix = torch.full((8, 3), 0.01)
    mix[:3, 0] = 0.98
    mix[3:6, 1] = 0.98
    mix[6] = torch.tensor([0.2, 0.7, 0.1])
    mix[7] = torch.tensor([0.5, 0.1, 0.4])
    transport = AgreementCovarianceTransport(
        num_classes=3,
        feature_dim=3,
        rank=1,
        min_anchors=3,
        max_gate=max_gate,
        epsilon=1e-6,
    )
    diagnostics = transport.fit(features, source, clip, mix)
    return transport, features, diagnostics


class AgreementCovarianceTransportTest(unittest.TestCase):
    def test_fit_builds_only_supported_class_geometry(self):
        transport, _, diagnostics = make_problem()

        self.assertEqual(diagnostics["active_classes"], 2)
        self.assertEqual(diagnostics["fixed_conflicts"], 2)
        self.assertEqual(diagnostics["eligible_conflicts"], 1)
        self.assertAlmostEqual(diagnostics["eligible_coverage"], 0.5)
        self.assertEqual(sum(p.numel() for p in transport.parameters()), 0)

    def test_agreements_and_unsupported_pairs_are_identity(self):
        transport, features, _ = make_problem()
        indices = torch.tensor([0, 7])
        selected = features[indices]

        output = transport(selected, indices)

        self.assertTrue(torch.equal(output, selected))

    def test_eligible_conflict_moves_within_relative_bound(self):
        transport, features, _ = make_problem(max_gate=0.1)
        feature = features[6:7]
        output = transport(feature, torch.tensor([6]))
        delta = output - feature

        self.assertGreater(float(delta.norm()), 0.0)
        self.assertLessEqual(
            float(delta.norm()), 0.1 * float(feature.norm()) + 1e-6
        )

    def test_transport_remains_differentiable_through_features(self):
        transport, features, _ = make_problem()
        feature = features[6:7].clone().requires_grad_(True)

        transport(feature, torch.tensor([6])).sum().backward()

        self.assertIsNotNone(feature.grad)
        self.assertGreater(float(feature.grad.norm()), 0.0)

    def test_geometry_can_only_be_frozen_once(self):
        transport, features, _ = make_problem()
        source = torch.zeros(features.size(0), dtype=torch.long)
        mix = torch.full((features.size(0), 3), 1.0 / 3.0)

        with self.assertRaises(ValueError):
            transport.fit(features, source, source, mix)


if __name__ == "__main__":
    unittest.main()
