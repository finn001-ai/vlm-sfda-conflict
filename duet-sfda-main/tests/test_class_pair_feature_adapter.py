import unittest

import torch
import torch.nn.functional as F

from src.utils.class_pair_feature_adapter import ClassPairFeatureAdapter


class ClassPairFeatureAdapterTest(unittest.TestCase):
    def test_zero_initialized_router_preserves_features(self):
        adapter = ClassPairFeatureAdapter(3, 2, 0.05, -2.0, 1e-6)
        features = torch.tensor([[1.0, -2.0, 0.5]])

        self.assertTrue(torch.equal(adapter(features), features))

    def test_delta_follows_classifier_pair_direction(self):
        adapter = ClassPairFeatureAdapter(3, 1, 0.05, 20.0, 1e-6)
        classifier_weight = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        adapter.set_pairs([(0, 1)], classifier_weight)
        adapter.router.weight.data.fill_(10.0)
        features = torch.ones(1, 3)
        delta = adapter(features) - features
        expected = torch.tensor([[-1.0, 1.0, 0.0]])

        self.assertGreater(
            float(F.cosine_similarity(delta.detach(), expected)), 0.999
        )
        self.assertEqual(int(adapter.active_rank), 1)

    def test_delta_norm_is_bounded_by_relative_feature_gate(self):
        adapter = ClassPairFeatureAdapter(3, 2, 0.05, 20.0, 1e-6)
        classifier_weight = torch.eye(3)
        adapter.set_pairs([(0, 1), (1, 2)], classifier_weight)
        adapter.router.weight.data.fill_(10.0)
        features = torch.tensor([[2.0, -1.0, 3.0]])
        delta = adapter(features) - features

        self.assertLessEqual(
            float(delta.detach().norm()),
            0.05 * float(features.norm()) + 1e-6,
        )

    def test_active_direction_gives_router_a_gradient(self):
        adapter = ClassPairFeatureAdapter(3, 1, 0.05, -2.0, 1e-6)
        adapter.set_pairs([(0, 1)], torch.eye(3))
        features = torch.ones(1, 3)

        adapter(features)[0, 1].backward()

        self.assertGreater(float(adapter.router.weight.grad.norm()), 0.0)

    def test_undercovered_basis_is_an_exact_identity(self):
        adapter = ClassPairFeatureAdapter(
            3, 2, 0.05, -2.0, 1e-6, min_active_rank=2
        )
        adapter.set_pairs([(0, 1)], torch.eye(3))
        adapter.router.weight.data.fill_(10.0)
        features = torch.tensor([[1.0, -2.0, 0.5]])

        self.assertTrue(torch.equal(adapter(features), features))
        self.assertFalse(adapter.is_effective())
        self.assertEqual(float(adapter.effective_gate()), 0.0)

    def test_coverage_threshold_activates_at_minimum_rank(self):
        adapter = ClassPairFeatureAdapter(
            3, 2, 0.05, -2.0, 1e-6, min_active_rank=2
        )
        adapter.set_pairs([(0, 1), (1, 2)], torch.eye(3))
        adapter.router.weight.data.fill_(1.0)
        features = torch.ones(1, 3)

        self.assertTrue(adapter.is_effective())
        self.assertFalse(torch.equal(adapter(features), features))


if __name__ == "__main__":
    unittest.main()
