import unittest

import torch

from src.utils.conflict_diffusion import (
    conflict_diffusion_evidence,
    dual_space_diffusion,
    select_class_balanced_anchors,
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


if __name__ == "__main__":
    unittest.main()
