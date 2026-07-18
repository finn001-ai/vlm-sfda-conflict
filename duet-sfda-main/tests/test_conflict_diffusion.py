import unittest

import torch

from src.utils.conflict_diffusion import (
    conflict_diffusion_evidence,
    dual_space_diffusion,
    select_class_balanced_anchors,
    transport_candidate_mass,
    update_temporal_resolution,
)


class ConflictDiffusionTest(unittest.TestCase):
    def test_class_balanced_anchor_selection(self):
        source_prob = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
        )
        clip_prob = source_prob.clone()
        labels = source_prob.argmax(dim=1)
        anchors, _ = select_class_balanced_anchors(
            source_prob, clip_prob, labels, labels, ratio=0.5, min_per_class=1
        )
        self.assertEqual(int(anchors.sum()), 2)
        self.assertTrue(anchors[0])
        self.assertTrue(anchors[3])

    def test_dual_diffusion_resolves_supported_candidate(self):
        task_features = torch.tensor(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]
        )
        clip_features = task_features.clone()
        source_prob = torch.tensor(
            [[0.95, 0.05], [0.55, 0.45], [0.05, 0.95], [0.45, 0.55]]
        )
        clip_prob = torch.tensor(
            [[0.90, 0.10], [0.45, 0.55], [0.10, 0.90], [0.55, 0.45]]
        )
        source_label = source_prob.argmax(dim=1)
        clip_label = clip_prob.argmax(dim=1)

        task_post, clip_post, fused, anchors = dual_space_diffusion(
            task_features,
            clip_features,
            source_prob,
            clip_prob,
            source_label,
            clip_label,
            anchor_ratio=1.0,
            anchor_min_per_class=1,
            k=2,
            temperature=0.1,
            alpha=0.8,
            steps=10,
            device=torch.device("cpu"),
        )
        evidence = conflict_diffusion_evidence(
            task_post,
            clip_post,
            fused,
            source_label,
            clip_label,
            candidate_mass_threshold=0.8,
            candidate_margin_threshold=0.1,
        )

        self.assertEqual(int(anchors.sum()), 2)
        self.assertTrue(evidence["eligible"][1])
        self.assertEqual(int(evidence["graph_label"][1]), 0)
        self.assertTrue(evidence["eligible"][3])
        self.assertEqual(int(evidence["graph_label"][3]), 1)

    def test_fixed_anchor_labels_override_current_predictions(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]]
        )
        source_prob = torch.tensor(
            [[0.05, 0.95], [0.55, 0.45], [0.95, 0.05], [0.45, 0.55]]
        )
        clip_prob = source_prob.clone()
        current_label = source_prob.argmax(dim=1)
        fixed_mask = torch.tensor([True, False, True, False])
        fixed_label = torch.tensor([0, -1, 1, -1])

        _, _, fused, anchors = dual_space_diffusion(
            features,
            features,
            source_prob,
            clip_prob,
            current_label,
            current_label,
            anchor_ratio=1.0,
            anchor_min_per_class=1,
            k=2,
            temperature=0.1,
            alpha=0.8,
            steps=10,
            device=torch.device("cpu"),
            anchor_mask=fixed_mask,
            anchor_label=fixed_label,
        )

        self.assertTrue(torch.equal(anchors, fixed_mask))
        self.assertEqual(int(fused[1].argmax()), 0)
        self.assertEqual(int(fused[3].argmax()), 1)

    def test_reversible_resolution_demotes_stale_label(self):
        pending_label = torch.full((2,), -1, dtype=torch.long)
        pending_count = torch.zeros(2, dtype=torch.long)
        resolved_label = torch.full((2,), -1, dtype=torch.long)
        eligible = torch.tensor([True, False])
        proposed = torch.tensor([1, 0])

        state = update_temporal_resolution(
            pending_label,
            pending_count,
            resolved_label,
            eligible,
            proposed,
            stable_cycles=2,
            memory="reversible",
        )
        state = update_temporal_resolution(
            state[0], state[1], state[2], eligible, proposed,
            stable_cycles=2, memory="reversible",
        )
        self.assertEqual(state[4].tolist(), [True, False])

        no_support = torch.tensor([False, False])
        state = update_temporal_resolution(
            state[0], state[1], state[2], no_support, proposed,
            stable_cycles=2, memory="reversible",
        )
        self.assertEqual(state[4].tolist(), [False, False])
        self.assertEqual(state[5].tolist(), [True, False])

    def test_candidate_transport_preserves_non_candidate_distribution(self):
        teacher = torch.tensor(
            [[0.10, 0.60, 0.20, 0.10], [0.20, 0.30, 0.10, 0.40]]
        )
        graph = torch.tensor(
            [[0.70, 0.20, 0.05, 0.05], [0.10, 0.20, 0.30, 0.40]]
        )
        corrected, shifted = transport_candidate_mass(
            teacher,
            graph,
            source_label=torch.tensor([0, 1]),
            clip_label=torch.tensor([1, 3]),
            mask=torch.tensor([True, False]),
        )

        self.assertTrue(torch.allclose(corrected.sum(dim=1), torch.ones(2)))
        self.assertTrue(torch.allclose(corrected[0, 2:], teacher[0, 2:]))
        self.assertTrue(torch.allclose(corrected[0, :2].sum(), teacher[0, :2].sum()))
        self.assertGreater(float(corrected[0, 0]), float(teacher[0, 0]))
        self.assertTrue(torch.equal(corrected[1], teacher[1]))
        self.assertGreater(float(shifted[0]), 0.0)
        self.assertEqual(float(shifted[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
