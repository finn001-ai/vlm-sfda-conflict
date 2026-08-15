"""Unit tests for duet_first_cycle_prior_context_transformer core modules.

Covers the 18 checks requested in section 18 of the task.  All tests run on
CPU with synthetic tensors; they never import the heavy training loop (which
needs cv2 / yacs / a GPU), only ``src.utils.duet_context`` and AST contracts
of the method entry file.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.utils.duet_context import (
    ClassBalancedAnchorBank,
    ComparatorReplayMemory,
    DuetContextConflictTransformer,
    PairwiseConflictComparator,
    PersistentConflictBeliefMemory,
    apply_pairwise_decision,
    apply_decision_rules,
    apply_weak_verification,
    build_comparator_features,
    build_delayed_transition_supervision,
    build_reliability_gated_fusion,
    build_real_conflict_multiview_supervision,
    build_synthetic_conflicts,
    cosine_knn_refine,
    fuse_transition_comparator_vote,
    prototype_refine,
    run_context_refinement,
    train_pairwise_comparator,
    train_pairwise_comparator_real_multiview,
    train_pairwise_comparator_early_stopping,
    train_pairwise_comparator_epochs,
    train_context_transformer,
    _exclude_query_anchors,
    _build_extended_real_conflict_probe_features,
    _log_agreement_ambiguity_eval_only,
    _log_agreement_candidate_probe_eval_only,
    _log_agreement_synthetic_feasibility_eval_only,
    _log_fixed_conflict_trajectory,
    _log_eval_only_metrics,
    _log_real_conflict_gt_feature_probe_eval_only,
    _log_real_comparator_margin_distribution,
    _stratified_binary_train_val_split,
    _zscore_filter,
)


def make_separable(n=512, num_classes=8, feature_dim=32, seed=0):
    """Craft features/probs with clear per-class structure + true labels."""
    torch.manual_seed(seed)
    feat = torch.randn(n, feature_dim)
    prototypes = torch.randn(num_classes, feature_dim)
    sim = feat @ prototypes.t()
    true_labels = sim.argmax(dim=1)
    task_prob = torch.softmax(sim * 4.0 + torch.randn(n, num_classes) * 0.1, dim=1)
    clip_prob = torch.softmax(sim * 3.5 + torch.randn(n, num_classes) * 0.3, dim=1)
    return feat, task_prob, clip_prob, true_labels


def make_context_cfg(**overrides):
    """构造与 DUET_CONTEXT 配置段等价的简单对象（测试不需要 yacs）。"""
    defaults = dict(
        USE_STRICT_CONFLICT=True,
        USE_WEAK_AGREEMENT=True,
        ANCHORS_PER_CLASS=8,
        ANCHOR_TASK_CONF=0.90,
        ANCHOR_CLIP_CONF=0.90,
        ANCHOR_TASK_ENTROPY=0.40,
        ANCHOR_CLIP_ENTROPY=0.40,
        ENTROPY_WEIGHT=1.0,
        REQUIRE_PRE_POST_PRIOR_AGREEMENT=True,
        WEAK_CONF_THRESHOLD=0.70,
        WEAK_ENTROPY_THRESHOLD=1.00,
        ACCEPT_CONF=0.75,
        ACCEPT_MARGIN=0.20,
        WEAK_ACCEPT_CONF=0.75,
        WEAK_ACCEPT_MARGIN=0.20,
        THIRD_CLASS_CONF=0.85,
        THIRD_CLASS_MARGIN=0.30,
        ALLOW_THIRD_CLASS=True,
        ABSTAIN_WHEN_UNCERTAIN=True,
        REFINER_TYPE="transformer",
        TRAIN_STEPS_PER_CYCLE=0,
        TRAIN_BATCH_SIZE=32,
        SEED=2020,
        EVAL_ONLY_LOGGING=False,
        KNN_K=5,
        COMPARATOR_HIDDEN=32,
        COMPARATOR_LAYERS=2,
        SIM_TOPK=3,
        MIN_RUNNER_PROB=0.10,
        MAX_TOP1_MARGIN=0.60,
        COMPARATOR_GATE=0.15,
        COMPARATOR_COVERAGE_FRACTION=0.0,
        REPLAY_PER_DIRECTION=64,
        REPLAY_MIX_FRACTION=0.25,
        EARLY_STOP_ENABLED=False,
        EARLY_STOP_VAL_FRACTION=0.20,
        EARLY_STOP_MIN_VAL_PER_DIRECTION=6,
        EARLY_STOP_CHECK_INTERVAL=10,
        EARLY_STOP_PATIENCE=3,
        EVAL_TRAJECTORY_ENABLED=False,
        EVAL_TRAJECTORY_INTERVAL=10,
        EVAL_TRAJECTORY_COVERAGES=[10, 20, 40, 60, 80],
        REAL_CONFLICT_GT_PROBE_ENABLED=False,
        REAL_CONFLICT_GT_PROBE_FOLDS=5,
        REAL_CONFLICT_GT_PROBE_STEPS=30,
        REAL_CONFLICT_GT_PROBE_HIDDEN=16,
        REAL_CONFLICT_GT_PROBE_LR=0.02,
        REAL_CONFLICT_GT_PROBE_EXTENDED_20D_ENABLED=False,
        REAL_MULTIVIEW_ENABLED=False,
        REAL_MULTIVIEW_RESIDUAL_FALLBACK=False,
        REAL_MULTIVIEW_TRAIN_FRACTION=0.60,
        REAL_MULTIVIEW_FINETUNE_STEPS=10,
        REAL_MULTIVIEW_TEMPERATURE=0.50,
        REAL_MULTIVIEW_SYNTHETIC_MIX_FRACTION=0.25,
        CONFLICT_MEMORY_ENABLED=False,
        CONFLICT_MEMORY_COVERAGE_FRACTION=0.80,
        CONFLICT_MEMORY_LOSS_WEIGHT=0.10,
        CONFLICT_MEMORY_TEMPERATURE=0.50,
        RELIABILITY_GATE_ENABLED=False,
        RELIABILITY_GATE_COVERAGE_FRACTION=0.80,
        RELIABILITY_GATE_TEMPERATURE=0.25,
        RELIABILITY_GATE_NEIGHBORS=5,
        RELIABILITY_GATE_NUM_VIEWS=1,
        RELIABILITY_GATE_LOSS_WEIGHT=0.10,
        TRANSITION_SUPERVISION_ENABLED=False,
        TRANSITION_MIN_VIEW_AGREEMENT=0.75,
        TRANSITION_MIN_PER_DIRECTION=2,
        TRANSITION_TRAIN_STEPS=20,
        TRANSITION_SYNTHETIC_MIX_FRACTION=0.25,
        TRANSITION_COMPARATOR_WEIGHT=0.50,
        AGREEMENT_AMBIGUITY_EVAL_ENABLED=False,
        AGREEMENT_AMBIGUITY_FRACTIONS=[10, 25, 50, 100],
        AGREEMENT_COMPARATOR_PROBE_ENABLED=False,
        AGREEMENT_SYNTHETIC_FEASIBILITY_ENABLED=False,
        COMPARATOR_EPOCHS=0,
        SOFT_ONLY_ADMISSION=False,
        DIST_MATCH_SYNTHETIC=False,
        DIST_MATCH_Z_MAX=1.5,
        DIST_MATCH_DIMS=[4, 5, 6, 7],
        MIN_DIST_MATCH_KEPT=16,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class AnchorBankTest(unittest.TestCase):
    def test_delayed_transition_supervision_uses_matured_real_conflicts(self):
        n, c, d = 12, 3, 6
        historical_task = torch.full((n, c), 0.01)
        historical_clip = torch.full((n, c), 0.01)
        historical_task[:8, 0] = 0.98
        historical_clip[:8, 1] = 0.98
        anchor_labels = torch.tensor([0, 1, 2, 0])
        for row, label in enumerate(anchor_labels, start=8):
            historical_task[row, label] = 0.98
            historical_clip[row, label] = 0.98
        historical_task /= historical_task.sum(dim=1, keepdim=True)
        historical_clip /= historical_clip.sum(dim=1, keepdim=True)
        current_task = historical_task.clone()
        current_clip = historical_clip.clone()
        for row in range(8):
            matured = 0 if row < 4 else 1
            current_task[row] = 0.01
            current_clip[row] = 0.01
            current_task[row, matured] = 0.98
            current_clip[row, matured] = 0.98
        current_task /= current_task.sum(dim=1, keepdim=True)
        current_clip /= current_clip.sum(dim=1, keepdim=True)
        views_task = current_task.unsqueeze(0).repeat(4, 1, 1)
        views_clip = current_clip.unsqueeze(0).repeat(4, 1, 1)
        snapshot = {
            "task_probs": historical_task,
            "clip_probs": historical_clip,
            "pre_prior_task_probs": historical_task,
            "pre_prior_clip_probs": historical_clip,
            "task_features": torch.randn(n, d),
            "clip_features": torch.randn(n, d),
        }
        result = build_delayed_transition_supervision(
            snapshot,
            current_task,
            current_clip,
            views_task,
            views_clip,
            num_classes=c,
            anchors_per_class=2,
            anchor_task_conf=0.90,
            anchor_clip_conf=0.90,
            anchor_task_entropy=0.40,
            anchor_clip_entropy=0.40,
            entropy_weight=1.0,
            require_pre_post_prior_agreement=True,
            sim_topk=2,
            min_view_agreement=0.75,
            min_per_direction=2,
            seed=2020,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["historical_conflicts"], 8)
        self.assertEqual(result["matured"], 8)
        self.assertEqual(result["features"].shape, (8, 16))
        self.assertEqual(int((result["targets"] == 0).sum()), 4)
        self.assertEqual(int((result["targets"] == 1).sum()), 4)

    def test_delayed_transition_supervision_keeps_imbalanced_majority(self):
        n, c, d = 14, 3, 5
        historical_task = torch.full((n, c), 0.01)
        historical_clip = torch.full((n, c), 0.01)
        historical_task[:10, 0] = 0.98
        historical_clip[:10, 1] = 0.98
        for row in range(10, n):
            label = row % c
            historical_task[row, label] = 0.98
            historical_clip[row, label] = 0.98
        historical_task /= historical_task.sum(dim=1, keepdim=True)
        historical_clip /= historical_clip.sum(dim=1, keepdim=True)
        current_task = historical_task.clone()
        current_clip = historical_clip.clone()
        # Two choose-A and eight choose-B matured transitions.
        for row in range(10):
            label = 0 if row < 2 else 1
            current_task[row] = 0.01
            current_clip[row] = 0.01
            current_task[row, label] = 0.98
            current_clip[row, label] = 0.98
        current_task /= current_task.sum(dim=1, keepdim=True)
        current_clip /= current_clip.sum(dim=1, keepdim=True)
        views_task = current_task.unsqueeze(0).repeat(4, 1, 1)
        views_clip = current_clip.unsqueeze(0).repeat(4, 1, 1)
        snapshot = {
            "task_probs": historical_task,
            "clip_probs": historical_clip,
            "pre_prior_task_probs": historical_task,
            "pre_prior_clip_probs": historical_clip,
            "task_features": torch.randn(n, d),
            "clip_features": torch.randn(n, d),
        }
        result = build_delayed_transition_supervision(
            snapshot,
            current_task,
            current_clip,
            views_task,
            views_clip,
            num_classes=c,
            anchors_per_class=2,
            anchor_task_conf=0.90,
            anchor_clip_conf=0.90,
            anchor_task_entropy=0.40,
            anchor_clip_entropy=0.40,
            entropy_weight=1.0,
            require_pre_post_prior_agreement=True,
            sim_topk=2,
            min_view_agreement=0.75,
            min_per_direction=2,
            seed=2020,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["features"].shape, (10, 16))
        self.assertEqual(int((result["targets"] == 0).sum()), 2)
        self.assertEqual(int((result["targets"] == 1).sum()), 8)
        self.assertEqual(result["balanced_per_direction"], 2)

    def test_transition_vote_fusion_preserves_fixed_coverage(self):
        weak_task = torch.tensor(
            [[0.70, 0.20, 0.10], [0.20, 0.70, 0.10], [0.60, 0.30, 0.10], [0.30, 0.60, 0.10]]
        )
        weak_clip = torch.tensor(
            [[0.20, 0.70, 0.10], [0.70, 0.20, 0.10], [0.30, 0.60, 0.10], [0.60, 0.30, 0.10]]
        )
        committee = {
            "q": torch.tensor([0.8, 0.2, 0.6, 0.4]),
            "source_agreement": torch.tensor([0.9, 0.9, 0.7, 0.7]),
            "candidate_a": weak_task.argmax(dim=1),
            "candidate_b": weak_clip.argmax(dim=1),
            "active": torch.ones(4, dtype=torch.bool),
            "target": 0.5 * (weak_task + weak_clip),
            "weight": torch.ones(4),
            "reliability": torch.ones(4),
        }
        fused = fuse_transition_comparator_vote(
            committee,
            torch.tensor([0.9, 0.1, 0.2, 0.8]),
            weak_task,
            weak_clip,
            comparator_weight=0.5,
            coverage_fraction=0.5,
        )
        self.assertEqual(int(fused["active"].sum()), 2)
        self.assertTrue(torch.allclose(fused["target"].sum(dim=1), torch.ones(4)))
        self.assertEqual(fused["transition_committee_agreement"].tolist(), [1.0, 1.0, 0.0, 0.0])

    def test_transition_supervision_runs_before_current_committee(self):
        torch.manual_seed(44)
        n, c, d = 40, 3, 8
        historical_task = torch.full((n, c), 0.01)
        historical_clip = torch.full((n, c), 0.01)
        historical_task[:16, 0] = 0.98
        historical_clip[:16, 1] = 0.98
        for row in range(16, n):
            label = row % c
            historical_task[row, label] = 0.98
            historical_clip[row, label] = 0.98
        historical_task /= historical_task.sum(dim=1, keepdim=True)
        historical_clip /= historical_clip.sum(dim=1, keepdim=True)
        current_task = historical_task.clone()
        current_clip = historical_clip.clone()
        for row in range(8):
            label = 0 if row < 4 else 1
            current_task[row] = 0.01
            current_clip[row] = 0.01
            current_task[row, label] = 0.98
            current_clip[row, label] = 0.98
        current_task /= current_task.sum(dim=1, keepdim=True)
        current_clip /= current_clip.sum(dim=1, keepdim=True)
        task_feature = torch.randn(n, d)
        clip_feature = torch.randn(n, d)
        task_views = current_task.unsqueeze(0).repeat(4, 1, 1)
        clip_views = current_clip.unsqueeze(0).repeat(4, 1, 1)
        snapshot = {
            "sample_indices": torch.arange(n),
            "task_probs": historical_task,
            "clip_probs": historical_clip,
            "pre_prior_task_probs": historical_task,
            "pre_prior_clip_probs": historical_clip,
            "task_features": torch.randn(n, d),
            "clip_features": torch.randn(n, d),
        }
        model = PairwiseConflictComparator(input_dim=16, hidden=16, layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logs = []
        result = run_context_refinement(
            current_task,
            current_clip,
            task_feature,
            num_classes=c,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator",
                RELIABILITY_GATE_ENABLED=True,
                RELIABILITY_GATE_NUM_VIEWS=4,
                RELIABILITY_GATE_COVERAGE_FRACTION=0.80,
                TRANSITION_SUPERVISION_ENABLED=True,
                TRANSITION_TRAIN_STEPS=3,
                TRANSITION_MIN_PER_DIRECTION=2,
                TRAIN_STEPS_PER_CYCLE=0,
            ),
            pre_prior_task_probs=current_task,
            pre_prior_clip_probs=current_clip,
            labels=torch.zeros(n, dtype=torch.long),
            sample_indices=torch.arange(n),
            clip_features=clip_feature,
            strong_task_probs=current_task,
            strong_clip_probs=current_clip,
            strong_task_features=task_feature,
            strong_clip_features=clip_feature,
            comparator=model,
            comparator_optimizer=optimizer,
            historical_conflict_snapshot=snapshot,
            reliability_task_view_probs=task_views,
            reliability_clip_view_probs=clip_views,
            reliability_task_view_features=task_feature.unsqueeze(0).repeat(4, 1, 1),
            reliability_clip_view_features=clip_feature.unsqueeze(0).repeat(4, 1, 1),
            cycle=2,
            log_fn=logs.append,
        )
        self.assertEqual(int(result["reliability_gate"]["active"].sum()), 6)
        self.assertTrue(any("DUET transition comparator training:" in line for line in logs))
        self.assertTrue(any("DUET transition-comparator fusion:" in line for line in logs))

    def test_reliability_gate_preserves_full_distribution_and_fixed_coverage(self):
        torch.manual_seed(31)
        n, c, d = 20, 4, 8
        labels = torch.arange(n) % c
        anchor_features = torch.randn(n, d)
        task_bank = ClassBalancedAnchorBank(c, 3, d, device=torch.device("cpu"))
        clip_bank = ClassBalancedAnchorBank(c, 3, d, device=torch.device("cpu"))
        scores = torch.linspace(0.1, 1.0, n)
        task_bank.update(anchor_features, labels, scores)
        clip_bank.update(anchor_features, labels, scores)
        weak_task = torch.softmax(torch.randn(n, c), dim=1)
        weak_clip = torch.roll(weak_task, shifts=1, dims=1)
        strong_task = 0.9 * weak_task + 0.1 / c
        strong_clip = torch.softmax(torch.randn(n, c), dim=1)
        result = build_reliability_gated_fusion(
            weak_task,
            weak_clip,
            strong_task,
            strong_clip,
            anchor_features,
            anchor_features,
            anchor_features,
            torch.randn(n, d),
            task_bank,
            clip_bank,
            num_classes=c,
            neighbors=3,
            temperature=0.25,
            coverage_fraction=0.80,
        )
        self.assertEqual(result["target"].shape, (n, c))
        self.assertEqual(int(result["active"].sum().item()), 16)
        self.assertTrue(torch.allclose(result["target"].sum(dim=1), torch.ones(n)))
        self.assertTrue(((result["q"] >= 0.0) & (result["q"] <= 1.0)).all())
        self.assertTrue((result["candidate_a"] != result["candidate_b"]).all())
        self.assertTrue((result["weight"][~result["active"]] == 0.0).all())

    def test_persistent_conflict_memory_accumulates_and_resets_changed_pair(self):
        memory = PersistentConflictBeliefMemory()
        first = memory.update(
            torch.tensor([10, 11]),
            torch.tensor([1, 2]),
            torch.tensor([3, 4]),
            torch.tensor([0.80, 0.20]),
            torch.ones(2),
            cycle=2,
            coverage_fraction=0.50,
        )
        self.assertEqual(int(first["active"].sum().item()), 1)
        self.assertTrue(torch.equal(first["observations"], torch.ones(2).long()))
        second = memory.update(
            torch.tensor([10, 11]),
            torch.tensor([1, 4]),
            torch.tensor([3, 2]),
            torch.tensor([0.90, 0.80]),
            torch.ones(2),
            cycle=3,
            coverage_fraction=1.0,
        )
        self.assertEqual(int(second["observations"][0].item()), 2)
        self.assertAlmostEqual(float(second["q"][0].item()), 0.85, places=6)
        self.assertEqual(int(second["observations"][1].item()), 1)
        self.assertEqual(second["pair_resets"], 1)

    def test_01_each_class_keeps_at_most_k(self):
        bank = ClassBalancedAnchorBank(num_classes=4, anchors_per_class=3, feature_dim=8)
        feats = torch.randn(100, 8)
        labels = torch.arange(4).repeat(25)
        scores = torch.rand(100)
        bank.update(feats, labels, scores, sample_indices=torch.arange(100))
        counts = bank.per_class_counts()
        self.assertEqual(counts.tolist(), [3, 3, 3, 3])
        self.assertEqual(int(counts.sum().item()), 12)

    def test_01b_top_k_by_reliability_and_deterministic(self):
        bank = ClassBalancedAnchorBank(num_classes=2, anchors_per_class=2, feature_dim=8)
        feats = torch.randn(10, 8)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = torch.tensor([1.0, 9.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0])
        indices = torch.arange(10)
        bank.update(feats, labels, scores, indices)
        self.assertEqual(bank.anchor_indices[0].tolist(), [1, 4])
        self.assertEqual(bank.anchor_indices[1].tolist(), [6, 8])

    def test_02_anchor_labels_are_hard_pseudo_labels(self):
        bank = ClassBalancedAnchorBank(num_classes=3, anchors_per_class=2, feature_dim=8)
        feats = torch.randn(12, 8)
        labels = torch.tensor([0, 1, 2] * 4)
        bank.update(feats, labels, torch.rand(12))
        for cls in range(3):
            stored = bank.anchor_labels[cls, bank.anchor_valid[cls]]
            self.assertTrue((stored == cls).all())
        self.assertEqual(int(bank.anchor_valid.sum().item()), 6)

    def test_06_self_exclusion_mask(self):
        query_ids = torch.tensor([10, 20, 30])
        anchor_ids = torch.tensor([10, 11, 20, 21, 30])
        mask = _exclude_query_anchors(query_ids, anchor_ids, 3, 5, torch.device("cpu"))
        self.assertEqual(mask[0].tolist(), [True, False, False, False, False])
        self.assertEqual(mask[1].tolist(), [False, False, True, False, False])
        self.assertEqual(mask[2].tolist(), [False, False, False, False, True])


class TransformerTest(unittest.TestCase):
    def setUp(self):
        self.model = DuetContextConflictTransformer(
            feature_dim=16, num_classes=4, model_dim=16, num_heads=4, ffn_dim=32
        )
        self.anchor_feat = torch.randn(12, 16)
        self.anchor_label = torch.tensor([0, 1, 2, 3] * 3)
        self.anchor_valid = torch.ones(12, dtype=torch.bool)

    def test_07_output_shape_b_by_c(self):
        out = self.model(torch.randn(5, 16), self.anchor_feat, self.anchor_label, self.anchor_valid)
        self.assertEqual(out["logits"].shape, (5, 4))
        self.assertEqual(out["probabilities"].shape, (5, 4))
        self.assertEqual(out["hidden"].shape, (5, 16))

    def test_05_one_image_feature_is_one_token(self):
        # [B, D] -> [B, 1, H] query token; keys are [B, A, H]
        out = self.model(torch.randn(3, 16), self.anchor_feat, self.anchor_label, self.anchor_valid)
        self.assertIsNotNone(out["attention"])
        self.assertEqual(out["attention"].shape[-1], 12)  # attends over anchors

    def test_08_empty_anchor_class_no_nan(self):
        # Class 3 has no anchor -> key_padding keeps it but values finite.
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        valid = torch.ones(6, dtype=torch.bool)
        out = self.model(torch.randn(4, 16), torch.randn(6, 16), labels, valid)
        self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_09_all_anchors_masked_safe_abstain(self):
        exclude = torch.ones(4, 12, dtype=torch.bool)
        out = self.model(
            torch.randn(4, 16), self.anchor_feat, self.anchor_label,
            self.anchor_valid, anchor_self_exclude=exclude,
        )
        self.assertTrue(torch.isfinite(out["logits"]).all())
        probs = out["probabilities"]
        self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5))

    def test_10_decision_supports_task(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.9, 0.05, 0.03, 0.02]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=True, abstain_when_uncertain=True,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertTrue(dec["resolved"].item())
        self.assertTrue(dec["is_task"].item())
        self.assertEqual(dec["context_top1"].item(), 0)

    def test_11_decision_supports_clip(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.05, 0.9, 0.03, 0.02]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=True, abstain_when_uncertain=True,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertTrue(dec["resolved"].item())
        self.assertTrue(dec["is_clip"].item())
        self.assertEqual(dec["context_top1"].item(), 1)

    def test_12_decision_supports_third_class(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.05, 0.05, 0.88, 0.02]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=True, abstain_when_uncertain=True,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertTrue(dec["resolved"].item())
        self.assertTrue(dec["is_third"].item())
        self.assertEqual(dec["context_top1"].item(), 2)

    def test_12b_third_class_blocked_when_disallowed(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.05, 0.05, 0.88, 0.02]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=False, abstain_when_uncertain=True,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertFalse(dec["resolved"].item())

    def test_13_uncertain_abstains(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.26, 0.25, 0.25, 0.24]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=True, abstain_when_uncertain=True,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertFalse(dec["resolved"].item())

    def test_13b_no_abstain_forces_resolution(self):
        task_top1 = torch.tensor([0])
        clip_top1 = torch.tensor([1])
        probs = torch.tensor([[0.26, 0.25, 0.25, 0.24]])
        dec = apply_decision_rules(
            probs, task_top1, clip_top1, accept_conf=0.75, accept_margin=0.2,
            allow_third_class=True, abstain_when_uncertain=False,
            third_class_conf=0.85, third_class_margin=0.3,
        )
        self.assertTrue(dec["resolved"].item())

    def test_leave_one_out_training(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss = train_context_transformer(
            self.model, optimizer, self.anchor_feat, self.anchor_label,
            self.anchor_valid, steps=10, batch_size=8, seed=42,
        )
        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)


class PipelineTest(unittest.TestCase):
    def _run(self, **overrides):
        feat, task_prob, clip_prob, true_label = make_separable()
        kwargs = dict(
            task_probs=task_prob,
            clip_probs=clip_prob,
            task_features=feat,
            num_classes=8,
            context_cfg=make_context_cfg(**overrides),
            pre_prior_task_probs=task_prob,
            pre_prior_clip_probs=clip_prob,
            labels=true_label,
            sample_indices=torch.arange(feat.size(0)),
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None,
            cycle=1,
        )
        return run_context_refinement(**kwargs)

    def test_03_ground_truth_never_affects_decisions(self):
        feat, task_prob, clip_prob, true_label = make_separable()
        base = dict(
            task_probs=task_prob, clip_probs=clip_prob, task_features=feat,
            num_classes=8, context_cfg=make_context_cfg(),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None,
        )
        with_gt = run_context_refinement(**base, labels=true_label,
                                        sample_indices=torch.arange(feat.size(0)))
        without_gt = run_context_refinement(**base, labels=None,
                                            sample_indices=torch.arange(feat.size(0)))
        for key in ("resolved_mask", "weak_rejected_mask", "context_labels", "refined_targets"):
            self.assertTrue(
                torch.equal(with_gt[key], without_gt[key]),
                "{} differs when ground truth is provided".format(key),
            )

    def test_04_conflicts_have_no_hard_label_and_are_not_anchors(self):
        result = self._run()
        anchor_mask = result["anchor_mask"]
        strict = result["strict_conflict_mask"]
        weak = result["weak_agreement_mask"]
        self.assertEqual(int((anchor_mask & strict).sum().item()), 0)
        self.assertEqual(int((anchor_mask & weak).sum().item()), 0)
        # 未 resolved 的 conflict 样本 context_labels 必须为 -1（无 hard label）
        unresolved = strict & ~result["resolved_mask"]
        if int(unresolved.sum().item()) > 0:
            self.assertTrue((result["context_labels"][unresolved] == -1).all())

    def test_14_weak_rejected_stays_out_of_hard_ce_and_keeps_clip_kl(self):
        feat, task_prob, clip_prob, _ = make_separable()
        result = run_context_refinement(
            task_prob, clip_prob, feat, num_classes=8,
            context_cfg=make_context_cfg(),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=torch.arange(feat.size(0)) % 8,
            sample_indices=torch.arange(feat.size(0)),
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None,
        )
        rejected = result["weak_rejected_mask"]
        if int(rejected.sum().item()) > 0:
            # weak 未通过验证的样本不得进入 hard CE（不在 resolved_mask），
            # 且 kl_soft 必须保持原始 clip 概率。
            self.assertTrue(
                torch.allclose(
                    result["refined_targets"][rejected],
                    clip_prob[rejected],
                    atol=1e-6,
                )
            )
            self.assertEqual(int((rejected & result["resolved_mask"]).sum().item()), 0)

    def test_15_unresolved_keeps_original_clip_target(self):
        result = self._run()
        untouched = ~result["resolved_mask"]
        feat, task_prob, clip_prob, _ = make_separable()
        self.assertTrue(
            torch.allclose(result["refined_targets"][untouched], clip_prob[untouched], atol=1e-6)
        )

    def test_16_resolved_rows_consistent_mem_kl(self):
        # 关闭 abstain 以强制 resolve，确保测试覆盖 resolved 一致性路径；
        # 与配置消融 #8（不允许 abstain）语义一致。
        result = self._run(ABSTAIN_WHEN_UNCERTAIN=False)
        resolved = result["resolved_mask"]
        self.assertGreater(int(resolved.sum().item()), 0)
        # mem_label 候选 = refined_targets.argmax；必须 == context_labels
        self.assertTrue(
            (
                result["refined_targets"][resolved].argmax(dim=1)
                == result["context_labels"][resolved]
            ).all()
        )

    def test_17_disabled_pipeline_returns_identity(self):
        feat, task_prob, clip_prob, true_label = make_separable()
        result = run_context_refinement(
            task_prob, clip_prob, feat, num_classes=8,
            context_cfg=make_context_cfg(
                USE_STRICT_CONFLICT=False, USE_WEAK_AGREEMENT=False
            ),
            labels=true_label, sample_indices=torch.arange(feat.size(0)),
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
        )
        self.assertEqual(int(result["resolved_mask"].sum().item()), 0)
        self.assertEqual(int(result["weak_rejected_mask"].sum().item()), 0)
        self.assertTrue(torch.allclose(result["refined_targets"], clip_prob, atol=1e-6))

    def test_correction_stats_and_logs(self):
        """每个激活 cycle 输出修正数量/比例/正确/错误统计。"""
        feat, task_prob, clip_prob, true_label = make_separable()
        logs = []
        result = run_context_refinement(
            task_prob, clip_prob, feat, num_classes=8,
            context_cfg=make_context_cfg(EVAL_ONLY_LOGGING=True),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=true_label, sample_indices=torch.arange(feat.size(0)),
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None, cycle=2, log_fn=logs.append,
        )
        stats = result["stats"]
        for key in (
            "resolved_rate_pct",
            "weak_defer_rate_pct",
            "final_admitted",
            "admitted_delta",
        ):
            self.assertIn(key, stats)
        self.assertEqual(
            stats["final_admitted"],
            stats["post_prior_agreement"] - stats["weak_deferred"]
            + stats["resolved_strict"],
        )
        self.assertEqual(
            stats["anchor_bank_total"], sum(stats["anchor_per_class_counts"])
        )
        self.assertGreaterEqual(stats["anchor_count"], stats["anchor_bank_total"])
        joined = "\n".join(logs)
        self.assertIn("DUET context correction: cycle=2;", joined)
        self.assertIn("DUET context correction eval-only: cycle=2;", joined)
        self.assertIn("anchor_candidates=", joined)
        self.assertIn("resolved_subset_task_acc=", joined)
        self.assertIn("resolved_subset_clip_acc=", joined)
        self.assertIn("resolved_comparator_acc=", joined)
        self.assertIn("resolved_candidate_oracle_acc=", joined)
        self.assertIn("conditional_arbitration_acc=", joined)
        self.assertIn("ground_truth_affects_training=False", joined)

    def test_eval_only_resolved_metrics_use_the_same_subset(self):
        """Task/CLIP/comparator/oracle must all use resolved_mask rows only."""
        logs = []
        resolved = torch.tensor([True, True, True, False, False])
        task_top1 = torch.tensor([0, 0, 2, 1, 0])
        clip_top1 = torch.tensor([1, 1, 1, 0, 1])
        context_labels = torch.tensor([0, 1, 1, -1, -1])
        labels = torch.tensor([0, 1, 2, 0, 0])
        _log_eval_only_metrics(
            {},
            resolved_mask=resolved,
            weak_rejected_mask=torch.zeros(5, dtype=torch.bool),
            context_labels=context_labels,
            task_top1=task_top1,
            clip_top1=clip_top1,
            duet_fallback_top1=torch.tensor([1, 1, 1, 0, 1]),
            all_label=labels,
            anchor_mask=torch.zeros(5, dtype=torch.bool),
            weak_agreement_mask=torch.zeros(5, dtype=torch.bool),
            strict_conflict_mask=task_top1 != clip_top1,
            cycle=4,
            log_fn=logs.append,
        )
        eval_line = next(
            line for line in logs if line.startswith("DUET context eval-only:")
        )
        self.assertIn("resolved_subset_task_acc=66.67%", eval_line)
        self.assertIn("resolved_subset_clip_acc=33.33%", eval_line)
        self.assertIn("resolved_subset_duet_fallback_acc=33.33%", eval_line)
        self.assertIn("resolved_comparator_acc=66.67%", eval_line)
        self.assertIn("resolved_gain_over_duet_fallback=+33.33pp", eval_line)
        self.assertIn(
            "coverage_weighted_gain_over_duet_fallback=+20.000pp", eval_line
        )
        self.assertIn("resolved_candidate_oracle_acc=100.00%", eval_line)
        self.assertIn("conditional_arbitration_acc=66.67%", eval_line)

    def test_conditional_arbitration_is_nan_without_correct_candidate(self):
        logs = []
        _log_eval_only_metrics(
            {},
            resolved_mask=torch.tensor([True]),
            weak_rejected_mask=torch.tensor([False]),
            context_labels=torch.tensor([0]),
            task_top1=torch.tensor([0]),
            clip_top1=torch.tensor([1]),
            all_label=torch.tensor([2]),
            anchor_mask=torch.tensor([False]),
            weak_agreement_mask=torch.tensor([False]),
            strict_conflict_mask=torch.tensor([True]),
            cycle=4,
            log_fn=logs.append,
        )
        eval_line = next(
            line for line in logs if line.startswith("DUET context eval-only:")
        )
        self.assertIn("resolved_candidate_oracle_acc=0.00%", eval_line)
        self.assertIn("conditional_arbitration_acc=nan", eval_line)


class ComparatorTest(unittest.TestCase):
    """Pairwise conflict-resolution（REFINER_TYPE=comparator）测试。"""

    def _small_bank_pair(self, num_classes=6, dim=8):
        task_bank = ClassBalancedAnchorBank(num_classes, 2, dim)
        clip_bank = ClassBalancedAnchorBank(num_classes, 2, dim)
        return task_bank, clip_bank

    def test_real_multiview_supervision_is_gt_free_and_fixed_coverage(self):
        weak_task = torch.tensor(
            [[0.80, 0.20], [0.70, 0.30], [0.60, 0.40], [0.55, 0.45]]
        )
        weak_clip = torch.tensor(
            [[0.45, 0.55], [0.40, 0.60], [0.30, 0.70], [0.20, 0.80]]
        )
        # Row 0 supports A across both branches/views; row 3 supports B.
        strong_task = torch.tensor(
            [[0.85, 0.15], [0.55, 0.45], [0.45, 0.55], [0.20, 0.80]]
        )
        strong_clip = torch.tensor(
            [[0.70, 0.30], [0.45, 0.55], [0.40, 0.60], [0.15, 0.85]]
        )
        result = build_real_conflict_multiview_supervision(
            weak_task,
            weak_clip,
            strong_task,
            strong_clip,
            torch.zeros(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            train_fraction=0.50,
            temperature=0.50,
        )
        self.assertEqual(int(result["selected"].sum().item()), 2)
        self.assertEqual(result["soft_targets"].shape, (2, 2))
        self.assertTrue(torch.allclose(result["soft_targets"].sum(1), torch.ones(2)))
        self.assertGreater(float(result["score"][0].item()), 0.0)
        self.assertLess(float(result["score"][3].item()), 0.0)
        self.assertAlmostEqual(float(result["weights"].mean().item()), 1.0, places=5)

    def test_real_multiview_soft_finetune_updates_comparator(self):
        torch.manual_seed(19)
        comparator = PairwiseConflictComparator(
            input_dim=4, hidden=8, layers=2, dropout=0.0
        )
        optimizer = torch.optim.Adam(comparator.parameters(), lr=1e-2)
        features = torch.randn(12, 4)
        soft_targets = torch.zeros(12, 2)
        soft_targets[:6, 0] = 0.9
        soft_targets[:6, 1] = 0.1
        soft_targets[6:, 0] = 0.1
        soft_targets[6:, 1] = 0.9
        before = [parameter.detach().clone() for parameter in comparator.parameters()]
        loss = train_pairwise_comparator_real_multiview(
            comparator,
            optimizer,
            features,
            soft_targets,
            torch.ones(12),
            steps=5,
            batch_size=12,
            seed=2020,
            synthetic_mix_fraction=0.0,
        )
        self.assertIsNotNone(loss)
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, comparator.parameters())
            )
        )

    def test_same_view_conflict_cases(self):
        """同 view：只有“一边保持 A、一边翻转”才造 synthetic 对；
        both_flip / no_conflict 都丢弃。"""
        num_classes, dim = 6, 8
        n = 4
        pool_labels = torch.tensor([0, 1, 2, 3])
        pool_task = torch.zeros(n, num_classes)
        pool_clip = torch.zeros(n, num_classes)
        pool_feat = torch.randn(n, dim)
        pool_clip_feat = torch.randn(n, dim)
        # 同一 strong 视图下的 Task/CLIP 分布（各自归一化）
        strong_task = torch.zeros(n, num_classes)
        strong_clip = torch.zeros(n, num_classes)
        strong_task[0] = torch.tensor([0.31, 0.42, 0.09, 0.06, 0.07, 0.05])  # 翻到1
        strong_clip[0] = torch.tensor([0.45, 0.20, 0.12, 0.10, 0.08, 0.05])  # 保持0
        strong_task[1] = torch.tensor([0.25, 0.40, 0.15, 0.08, 0.07, 0.05])  # 保持1
        strong_clip[1] = torch.tensor([0.10, 0.31, 0.42, 0.08, 0.05, 0.04])  # 翻到2
        strong_task[2] = torch.tensor([0.10, 0.15, 0.40, 0.15, 0.12, 0.08])  # 保持2
        strong_clip[2] = torch.tensor([0.10, 0.15, 0.42, 0.15, 0.10, 0.08])  # 保持2
        strong_task[3] = torch.tensor([0.35, 0.20, 0.15, 0.12, 0.10, 0.08])  # 翻到0
        strong_clip[3] = torch.tensor([0.20, 0.35, 0.15, 0.12, 0.10, 0.08])  # 翻到1
        task_bank, clip_bank = self._small_bank_pair(num_classes, dim)
        task_bank.update(pool_feat, pool_labels, torch.ones(n))
        clip_bank.update(pool_clip_feat, pool_labels, torch.ones(n))
        features, targets, counts = build_synthetic_conflicts(
            pool_labels,
            pool_strong_task_probs=strong_task,
            pool_strong_clip_probs=strong_clip,
            pool_strong_task_features=pool_feat,
            pool_strong_clip_features=pool_clip_feat,
            task_bank=task_bank,
            clip_bank=clip_bank,
            min_runner_prob=0.10, max_top1_margin=0.60, sim_topk=2,
        )
        self.assertEqual(counts["task_flip_only"], 1)
        self.assertEqual(counts["clip_flip_only"], 1)
        self.assertEqual(counts["both_flip"], 1)
        self.assertEqual(counts["no_conflict"], 1)
        self.assertEqual(counts["task_side"], 1)
        self.assertEqual(counts["clip_side"], 1)
        self.assertEqual(targets.tolist(), [1.0, 0.0])  # trust CLIP / trust Task
        # 证据里必须是真 conflict：Task 证据 Top1=候选A，CLIP 证据 Top1=候选B
        self.assertTrue((features[:, 0] > features[:, 1]).all())
        self.assertTrue((features[:, 3] > features[:, 2]).all())
        self.assertEqual(targets.device, features.device)

    def test_same_view_decisive_flip_gated(self):
        """同 view 下果断 flip（p_B - p_A 大）被 MAX_TOP1_MARGIN 拦掉。"""
        num_classes, dim = 6, 8
        n = 2
        pool_labels = torch.tensor([0, 1])
        pool_task = torch.zeros(n, num_classes)
        pool_clip = torch.zeros(n, num_classes)
        pool_feat = torch.randn(n, dim)
        pool_clip_feat = torch.randn(n, dim)
        strong_task = torch.zeros(n, num_classes)
        strong_clip = torch.zeros(n, num_classes)
        strong_task[0] = torch.tensor([0.01, 0.98, 0.002, 0.003, 0.003, 0.002])  # 果断翻到1
        strong_clip[0] = torch.tensor([0.45, 0.20, 0.12, 0.10, 0.08, 0.05])  # 保持0
        strong_task[1] = torch.tensor([0.42, 0.31, 0.09, 0.06, 0.07, 0.05])  # 勉强翻到0
        strong_clip[1] = torch.tensor([0.10, 0.45, 0.15, 0.12, 0.10, 0.08])  # 保持1
        task_bank, clip_bank = self._small_bank_pair(num_classes, dim)
        task_bank.update(pool_feat, pool_labels, torch.ones(n))
        clip_bank.update(pool_clip_feat, pool_labels, torch.ones(n))
        _, _, counts = build_synthetic_conflicts(
            pool_labels,
            pool_strong_task_probs=strong_task,
            pool_strong_clip_probs=strong_clip,
            pool_strong_task_features=pool_feat,
            pool_strong_clip_features=pool_clip_feat,
            task_bank=task_bank,
            clip_bank=clip_bank,
            min_runner_prob=0.10, max_top1_margin=0.60, sim_topk=2,
        )
        self.assertEqual(counts["task_flip_only"], 2)
        self.assertEqual(counts["task_side"], 1)  # 果断 flip 被 gate 掉

    def test_zscore_filter(self):
        reference = torch.tensor(
            [
                [0.4, 0.1, 0.1, 0.6, 2.0, 1.0, 0.3, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.3, 0.1, 0.1, 0.7, 2.5, 1.2, 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        features = torch.tensor(
            [
                [0.35, 0.1, 0.1, 0.65, 2.2, 1.1, 0.25, 0.45, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.90, 0.01, 0.01, 0.95, 0.3, 0.3, 0.85, 0.85, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.20, 0.1, 0.1, 0.80, 3.5, 1.5, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        keep, used_fallback = _zscore_filter(
            features, reference, [0, 4, 6], z_max=1.5, min_kept=1
        )
        # 第一行贴近 reference；第二行 p_task_A=0.90 太自信；第三行熵 3.5 太极端
        self.assertEqual(keep.tolist(), [True, False, False])
        self.assertFalse(used_fallback)

    def test_zscore_filter_zero_variance(self):
        reference = torch.zeros(3, 16)
        features = torch.zeros(2, 16)
        keep, used_fallback = _zscore_filter(
            features, reference, [0, 1], z_max=1.5, min_kept=1
        )
        self.assertEqual(keep.tolist(), [True, True])
        self.assertFalse(used_fallback)

    def test_zscore_filter_fallback_keeps_closest(self):
        reference = torch.tensor(
            [[0.4, 0.1, 0.1, 0.6, 2.0, 1.0, 0.3, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]] * 3
        )
        features = torch.tensor(
            [
                [0.90, 0.01, 0.01, 0.95, 0.3, 0.3, 0.85, 0.85, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.88, 0.01, 0.01, 0.92, 0.4, 0.4, 0.80, 0.80, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.95, 0.01, 0.01, 0.96, 0.2, 0.2, 0.90, 0.90, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        keep, used_fallback = _zscore_filter(
            features, reference, [0, 4, 6], z_max=1.0, min_kept=2
        )
        self.assertTrue(used_fallback)
        # 保留 mean|z| 最小的 2 个（第 1、2 行比第 3 行更接近 reference）
        self.assertEqual(keep.tolist(), [True, True, False])

    def test_forward_shape_two_way(self):
        model = PairwiseConflictComparator(input_dim=16, hidden=16, layers=2)
        logits = model(torch.randn(5, 16))
        self.assertEqual(logits.shape, (5, 2))

    def test_features_missing_anchor_uses_zero_and_flag(self):
        """缺失 anchor：support=0 + available=0，而不是负相似度。"""
        task_bank = ClassBalancedAnchorBank(4, 2, 8)
        clip_bank = ClassBalancedAnchorBank(4, 2, 8)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        task_bank.update(torch.randn(6, 8), labels, torch.rand(6))
        clip_bank.update(torch.randn(6, 8), labels, torch.rand(6))
        features = build_comparator_features(
            torch.randn(2, 4), torch.randn(2, 4),
            torch.randn(2, 8), torch.randn(2, 8),
            task_bank, clip_bank,
            class_a=torch.tensor([0, 1]),
            class_b=torch.tensor([3, 3]),  # class 3 无 anchor
            sim_topk=2,
        )
        self.assertEqual(features.shape, (2, 16))
        # class 3 缺失：B 侧两个 sim 和 B 侧 available 标志都是 0
        self.assertTrue((features[:, 9] == 0).all())
        self.assertTrue((features[:, 11] == 0).all())
        self.assertTrue((features[:, 13] == 0).all())
        self.assertTrue((features[:, 15] == 0).all())

    def test_training_reduces_loss(self):
        model = PairwiseConflictComparator(input_dim=16, hidden=16, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        features = torch.randn(32, 16)
        targets = torch.randint(0, 2, (32,)).float()
        loss1 = train_pairwise_comparator(
            model, optimizer, features, targets, steps=10, batch_size=32, seed=1
        )
        loss2 = train_pairwise_comparator(
            model, optimizer, features, targets, steps=10, batch_size=32, seed=1
        )
        self.assertIsNotNone(loss1)
        self.assertLess(loss2, loss1)

    def test_fixed_step_trajectory_captures_requested_checkpoints(self):
        torch.manual_seed(7)
        model = PairwiseConflictComparator(
            input_dim=4, hidden=8, layers=1, dropout=0.0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        features = torch.randn(12, 4)
        targets = torch.tensor([0.0] * 6 + [1.0] * 6)
        fixed_features = torch.randn(5, 4)
        trajectory = []
        train_pairwise_comparator(
            model,
            optimizer,
            features,
            targets,
            steps=5,
            batch_size=12,
            seed=3,
            trajectory_features=fixed_features,
            trajectory_interval=2,
            trajectory_sink=trajectory,
        )
        self.assertEqual([row["step"] for row in trajectory], [0, 2, 4, 5])
        self.assertTrue(
            all(row["fixed_logits"].shape == (5, 2) for row in trajectory)
        )

    def test_fixed_conflict_trajectory_is_same_subset_and_no_gate(self):
        trajectory = [
            {
                "step": 0,
                "synthetic_train_loss": 0.7,
                "fixed_logits": torch.tensor(
                    [[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]
                ),
            }
        ]
        logs = []
        _log_fixed_conflict_trajectory(
            trajectory,
            task_candidates=torch.tensor([0, 0, 1, 1]),
            clip_candidates=torch.tensor([1, 1, 0, 0]),
            duet_fallback_candidates=torch.tensor([1, 1, 1, 1]),
            labels=torch.tensor([0, 1, 0, 1]),
            coverages=[50, 100],
            cycle=2,
            log_fn=logs.append,
        )
        self.assertEqual(len(logs), 2)
        self.assertIn("fixed_conflicts=4", logs[0])
        self.assertIn("task_acc=50.00%", logs[0])
        self.assertIn("clip_acc=50.00%", logs[0])
        self.assertIn("duet_fallback_acc=50.00%", logs[0])
        self.assertIn("comparator_acc=100.00%", logs[0])
        self.assertIn("gate_used=False", logs[0])
        self.assertIn("coverage_50_n=2", logs[1])
        self.assertIn("coverage_50_duet_fallback_acc=50.00%", logs[1])
        self.assertIn("coverage_50_gain_over_duet_fallback=+50.00pp", logs[1])
        self.assertIn("coverage_50_coverage_weighted_gain=+25.000pp", logs[1])
        self.assertIn("coverage_100_n=4", logs[1])
        self.assertIn("checkpoint_selected_by_gt=False", logs[1])

    def test_real_conflict_gt_feature_probe_recovers_separable_ceiling(self):
        generator = torch.Generator().manual_seed(31)
        per_group, dim = 50, 16
        features = torch.randn(3 * per_group, dim, generator=generator) * 0.05
        features[:per_group, 0] -= 2.0  # GT is Task candidate
        features[per_group : 2 * per_group, 0] += 2.0  # GT is CLIP candidate
        # Final group has neither candidate as GT and must count as incorrect.
        task_candidates = torch.zeros(3 * per_group, dtype=torch.long)
        clip_candidates = torch.ones(3 * per_group, dtype=torch.long)
        labels = torch.cat(
            [
                torch.zeros(per_group, dtype=torch.long),
                torch.ones(per_group, dtype=torch.long),
                torch.full((per_group,), 2, dtype=torch.long),
            ]
        )
        current_logits = torch.tensor([[0.0, 1.0]]).repeat(3 * per_group, 1)
        features_before = features.clone()
        rng_before = torch.random.get_rng_state().clone()
        logs = []
        result = _log_real_conflict_gt_feature_probe_eval_only(
            features,
            task_candidates,
            clip_candidates,
            labels,
            current_logits,
            folds=5,
            steps=300,
            hidden=16,
            lr=0.01,
            seed=2020,
            cycle=2,
            log_fn=logs.append,
        )
        rng_after = torch.random.get_rng_state()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["total"], 150)
        self.assertEqual(result["oracle_count"], 100)
        self.assertEqual(result["neither_count"], 50)
        self.assertAlmostEqual(result["candidate_oracle_acc"], 200.0 / 3.0)
        self.assertAlmostEqual(
            result["current_comparator_acc"], 100.0 / 3.0, places=5
        )
        self.assertGreater(result["logistic_probe_acc"], 65.0)
        self.assertGreater(result["mlp_probe_acc"], 65.0)
        self.assertGreater(
            result["logistic_conditional_arbitration_acc"], 98.0
        )
        self.assertGreater(result["mlp_conditional_arbitration_acc"], 98.0)
        self.assertEqual(len(result["folds"]), 5)
        self.assertTrue(torch.equal(features, features_before))
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertIn("synthetic_comparator_acc=33.33%", logs[-1])
        self.assertIn("candidate_oracle_acc=66.67%", logs[-1])
        self.assertIn("probe_uses_gt=True", logs[-1])
        self.assertIn("formal_method_affected=False", logs[-1])

    def test_extended_real_conflict_probe_features_are_exact_and_gt_free(self):
        base = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        task = torch.tensor(
            [[0.60, 0.30, 0.10], [0.10, 0.30, 0.60]], dtype=torch.float32
        )
        clip = torch.tensor(
            [[0.60, 0.30, 0.10], [0.60, 0.30, 0.10]], dtype=torch.float32
        )
        task_reference = torch.tensor([0.50, 0.30, 0.20])
        clip_reference = torch.tensor([0.50, 0.30, 0.20])
        base_before = base.clone()
        task_before = task.clone()
        clip_before = clip.clone()

        extended = _build_extended_real_conflict_probe_features(
            base,
            task,
            clip,
            task_reference,
            clip_reference,
            ranking_chunk_size=1,
        )

        self.assertEqual(extended.shape, (2, 20))
        self.assertTrue(torch.equal(extended[:, :16], base))
        self.assertTrue(torch.equal(base, base_before))
        self.assertTrue(torch.equal(task, task_before))
        self.assertTrue(torch.equal(clip, clip_before))
        # Sorting removes the class reversal, so both Task rows have the same
        # concentration-profile drift from the supplied reference.
        self.assertTrue(torch.allclose(extended[:, 16], torch.tensor([0.2, 0.2])))
        self.assertTrue(torch.allclose(extended[:, 17], torch.tensor([0.2, 0.2])))
        self.assertAlmostEqual(float(extended[0, 18]), 0.0, places=7)
        self.assertGreater(float(extended[1, 18]), 0.0)
        self.assertAlmostEqual(float(extended[0, 19]), 0.0, places=7)
        self.assertAlmostEqual(float(extended[1, 19]), 1.0, places=7)
        self.assertTrue(torch.isfinite(extended).all())

    def test_real_conflict_gt_feature_probe_skips_one_direction(self):
        logs = []
        result = _log_real_conflict_gt_feature_probe_eval_only(
            torch.randn(8, 16),
            torch.zeros(8, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.zeros(8, dtype=torch.long),
            torch.zeros(8, 2),
            folds=5,
            steps=10,
            hidden=8,
            lr=0.01,
            seed=1,
            cycle=3,
            log_fn=logs.append,
        )
        self.assertEqual(result["status"], "skipped_insufficient_binary_targets")
        self.assertEqual(len(logs), 1)
        self.assertIn("effective_folds=0", logs[0])
        self.assertIn("formal_method_affected=False", logs[0])

    def test_trajectory_capture_does_not_change_training_result(self):
        torch.manual_seed(17)
        features = torch.randn(20, 4)
        targets = torch.tensor([0.0] * 10 + [1.0] * 10)
        fixed_features = torch.randn(7, 4)
        plain = PairwiseConflictComparator(
            input_dim=4, hidden=8, layers=1, dropout=0.2
        )
        diagnostic = PairwiseConflictComparator(
            input_dim=4, hidden=8, layers=1, dropout=0.2
        )
        diagnostic.load_state_dict(plain.state_dict())
        plain_optimizer = torch.optim.Adam(plain.parameters(), lr=1e-2)
        diagnostic_optimizer = torch.optim.Adam(
            diagnostic.parameters(), lr=1e-2
        )

        torch.manual_seed(123)
        train_pairwise_comparator(
            plain,
            plain_optimizer,
            features,
            targets,
            steps=8,
            batch_size=12,
            seed=5,
        )
        trajectory = []
        torch.manual_seed(123)
        train_pairwise_comparator(
            diagnostic,
            diagnostic_optimizer,
            features,
            targets,
            steps=8,
            batch_size=12,
            seed=5,
            trajectory_features=fixed_features,
            trajectory_interval=2,
            trajectory_sink=trajectory,
        )
        for plain_value, diagnostic_value in zip(
            plain.state_dict().values(), diagnostic.state_dict().values()
        ):
            self.assertTrue(torch.equal(plain_value, diagnostic_value))

    def test_replay_memory_per_direction_cap_and_update(self):
        memory = ComparatorReplayMemory(per_direction_capacity=4, feature_dim=8)
        features = torch.randn(10, 8)
        targets = torch.tensor([0.0] * 5 + [1.0] * 5)
        memory.update(features, targets)
        self.assertEqual(memory.task_features.size(0), 4)
        self.assertEqual(memory.clip_features.size(0), 4)
        mem_f, mem_t = memory.as_tensors()
        self.assertEqual(mem_f.size(0), 8)
        self.assertTrue(torch.allclose(mem_f[:4], features[1:5]))
        self.assertTrue(torch.allclose(mem_f[4:], features[6:]))
        # 继续 update 仍 cap，保留最新
        memory.update(features[5:6], torch.tensor([1.0]))
        self.assertEqual(memory.clip_features.size(0), 4)
        # 均衡采样：两方向各半
        generator = torch.Generator()
        generator.manual_seed(0)
        s_f, s_t = memory.sample(4, generator)
        self.assertEqual(s_f.size(0), 4)
        self.assertEqual(int((s_t == 0).sum()), int((s_t == 1).sum()))

    def test_train_pairwise_comparator_with_replay_mixing(self):
        torch.manual_seed(0)
        pos = torch.randn(20, 8) + 1.0
        neg = torch.randn(20, 8) - 1.0
        current_f = torch.cat([neg, pos])
        current_t = torch.tensor([0.0] * 20 + [1.0] * 20)
        mem_f = torch.cat([neg[:4], pos[:4]])
        mem_t = torch.tensor([0.0] * 4 + [1.0] * 4)
        model = PairwiseConflictComparator(input_dim=8, hidden=16, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss1 = train_pairwise_comparator(
            model, optimizer, current_f, current_t,
            steps=20, batch_size=32, seed=1,
            memory_features=mem_f, memory_targets=mem_t, memory_fraction=0.25,
        )
        loss2 = train_pairwise_comparator(
            model, optimizer, current_f, current_t,
            steps=20, batch_size=32, seed=1,
            memory_features=mem_f, memory_targets=mem_t, memory_fraction=0.25,
        )
        self.assertIsNotNone(loss1)
        self.assertLess(loss2, loss1)
        with torch.no_grad():
            probs = torch.softmax(model(current_f), 1)
            acc = (probs.argmax(1) == current_t.long()).float().mean().item()
        self.assertGreater(acc, 0.7)

    def test_train_pairwise_comparator_balances_batches_without_downsampling(self):
        class RecordingComparator(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.seen = []

            def forward(self, inputs):
                self.seen.append(inputs.detach().clone())
                return self.linear(inputs)

        features = torch.stack(
            [
                torch.cat([torch.zeros(2), torch.ones(18)]),
                torch.arange(20, dtype=torch.float32),
            ],
            dim=1,
        )
        targets = torch.cat([torch.zeros(2), torch.ones(18)])
        model = RecordingComparator()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        train_pairwise_comparator(
            model,
            optimizer,
            features,
            targets,
            steps=4,
            batch_size=10,
            seed=7,
            balance_current_directions=True,
        )
        self.assertEqual(len(model.seen), 4)
        for batch in model.seen:
            self.assertEqual(int((batch[:, 0] == 0).sum()), 5)
            self.assertEqual(int((batch[:, 0] == 1).sum()), 5)
        seen = torch.cat(model.seen)
        seen_majority_ids = torch.unique(seen[seen[:, 0] == 1, 1])
        self.assertEqual(seen_majority_ids.numel(), 18)

    def test_train_pairwise_comparator_epochs(self):
        """Full-batch epoch training performs exactly one update per epoch."""
        torch.manual_seed(0)
        pos = torch.randn(24, 8) + 1.0
        neg = torch.randn(24, 8) - 1.0
        current_f = torch.cat([neg, pos])
        current_t = torch.tensor([0.0] * 24 + [1.0] * 24)
        mem_f = torch.cat([neg[:4], pos[:4]])
        mem_t = torch.tensor([0.0] * 4 + [1.0] * 4)
        model = PairwiseConflictComparator(input_dim=8, hidden=16, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer_steps = 0
        original_step = optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1
            return original_step(*args, **kwargs)

        optimizer.step = counted_step
        loss1 = train_pairwise_comparator_epochs(
            model, optimizer, current_f, current_t,
            epochs=5, batch_size=32, seed=1,
            memory_features=mem_f, memory_targets=mem_t, memory_fraction=0.25,
        )
        loss2 = train_pairwise_comparator_epochs(
            model, optimizer, current_f, current_t,
            epochs=5, batch_size=32, seed=1,
            memory_features=mem_f, memory_targets=mem_t, memory_fraction=0.25,
        )
        self.assertIsNotNone(loss1)
        self.assertLess(loss2, loss1)
        self.assertEqual(optimizer_steps, 10)

    def test_train_pairwise_comparator_epochs_has_no_sample_count_step_cliff(self):
        """96 and 102 current rows must produce the same number of updates."""
        update_counts = []
        for sample_count in (96, 102):
            torch.manual_seed(0)
            features = torch.randn(sample_count, 8)
            targets = torch.arange(sample_count).remainder(2).float()
            memory_features = torch.randn(16, 8)
            memory_targets = torch.arange(16).remainder(2).float()
            model = PairwiseConflictComparator(input_dim=8, hidden=16, layers=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer_steps = 0
            original_step = optimizer.step

            def counted_step(*args, **kwargs):
                nonlocal optimizer_steps
                optimizer_steps += 1
                return original_step(*args, **kwargs)

            optimizer.step = counted_step
            train_pairwise_comparator_epochs(
                model,
                optimizer,
                features,
                targets,
                epochs=5,
                batch_size=64,
                seed=1,
                memory_features=memory_features,
                memory_targets=memory_targets,
                memory_fraction=0.25,
            )
            update_counts.append(optimizer_steps)
        self.assertEqual(update_counts, [5, 5])

    def test_stratified_synthetic_validation_split_is_balanced(self):
        targets = torch.tensor([0.0] * 31 + [1.0] * 31)
        split1 = _stratified_binary_train_val_split(
            targets,
            val_fraction=0.20,
            min_val_per_direction=6,
            seed=7,
        )
        split2 = _stratified_binary_train_val_split(
            targets,
            val_fraction=0.20,
            min_val_per_direction=6,
            seed=7,
        )
        self.assertIsNotNone(split1)
        self.assertEqual(split1["val_per_direction"], 6)
        self.assertEqual(split1["val_indices"].numel(), 12)
        self.assertEqual(split1["train_indices"].numel(), 50)
        val_targets = targets[split1["val_indices"]]
        self.assertEqual(int((val_targets == 0).sum()), 6)
        self.assertEqual(int((val_targets == 1).sum()), 6)
        self.assertTrue(
            torch.equal(split1["train_indices"], split2["train_indices"])
        )
        self.assertTrue(
            torch.equal(split1["val_indices"], split2["val_indices"])
        )

    def test_synthetic_validation_early_stop_restores_step_zero(self):
        targets = torch.tensor([0.0] * 12 + [1.0] * 12)
        split = _stratified_binary_train_val_split(
            targets,
            val_fraction=0.25,
            min_val_per_direction=3,
            seed=11,
        )
        features = torch.zeros(24, 2)
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
        features[train_indices, targets[train_indices].long()] = 1.0
        features[val_indices, 1 - targets[val_indices].long()] = 1.0
        model = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(model.weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        logs = []
        result = train_pairwise_comparator_early_stopping(
            model,
            optimizer,
            features,
            targets,
            max_steps=50,
            batch_size=32,
            seed=11,
            val_fraction=0.25,
            min_val_per_direction=3,
            check_interval=5,
            patience=2,
            memory_fraction=0.0,
            cycle=3,
            log_fn=logs.append,
        )
        self.assertEqual(result["best_step"], 0)
        self.assertEqual(result["optimizer_steps"], 10)
        self.assertTrue(result["stopped_early"])
        self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(result["val_per_direction"], 3)
        self.assertTrue(any("step=0" in line for line in logs))
        self.assertTrue(any("step=10" in line for line in logs))

    def test_replay_memory_integration_log(self):
        """带 replay memory 的 comparator 管线：日志出现 replay 行。"""
        torch.manual_seed(2)
        n, c, d = 512, 8, 32
        feat = torch.randn(n, d)
        proto = torch.randn(c, d)
        sim = feat @ proto.t()
        task_prob = torch.softmax(sim * 3.0 + torch.randn(n, c) * 0.2, dim=1)
        clip_prob = torch.softmax(sim * 2.8 + torch.randn(n, c) * 0.4, dim=1)
        clip_feat = torch.randn(n, d)
        true_label = sim.argmax(dim=1)
        strong_task = torch.softmax(sim * 1.5 + torch.randn(n, c) * 0.7, dim=1)
        strong_clip = torch.softmax(sim * 1.3 + torch.randn(n, c) * 0.8, dim=1)
        memory = ComparatorReplayMemory(
            per_direction_capacity=8, feature_dim=16, device=torch.device("cpu")
        )
        memory.update(torch.randn(10, 16), torch.tensor([0.0] * 5 + [1.0] * 5))
        model = PairwiseConflictComparator(input_dim=16, hidden=32, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logs = []
        run_context_refinement(
            task_prob, clip_prob, feat, num_classes=c,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator", TRAIN_STEPS_PER_CYCLE=30,
                COMPARATOR_EPOCHS=20,
            ),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=true_label, sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task, strong_clip_probs=strong_clip,
            strong_task_features=feat, strong_clip_features=clip_feat,
            comparator=model, comparator_optimizer=optimizer,
            replay_memory=memory, cycle=3, log_fn=logs.append,
        )
        self.assertTrue(
            any("DUET comparator replay" in line for line in logs)
        )
        training_log = next(
            line
            for line in logs
            if "DUET comparator training: cycle=3; mode=full_batch_epochs"
            in line
        )
        self.assertIn("current_samples=", training_log)
        self.assertIn("memory_samples=", training_log)
        self.assertIn("optimizer_steps_this_cycle=20", training_log)

    def test_decision_margin_gate(self):
        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.05, -0.05]])
        task_top1 = torch.tensor([0, 0, 0])
        clip_top1 = torch.tensor([1, 1, 1])
        decision = apply_pairwise_decision(logits, task_top1, clip_top1, gate=0.2)
        self.assertEqual(decision["resolved"].tolist(), [True, True, False])
        self.assertEqual(decision["chosen"][0].item(), 0)  # trust Task
        self.assertEqual(decision["chosen"][1].item(), 1)  # trust CLIP

    def test_decision_rank_coverage_ignores_absolute_gate(self):
        logits = torch.tensor(
            [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.2, 0.0], [0.0, 0.0]]
        )
        task_top1 = torch.zeros(5, dtype=torch.long)
        clip_top1 = torch.ones(5, dtype=torch.long)
        decision = apply_pairwise_decision(
            logits,
            task_top1,
            clip_top1,
            gate=0.99,
            coverage_fraction=0.40,
        )
        self.assertEqual(decision["selection_mode"], "rank_coverage")
        self.assertEqual(decision["selected_count"], 2)
        self.assertEqual(
            decision["resolved"].tolist(), [True, True, False, False, False]
        )

    def test_decision_rank_coverage_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            apply_pairwise_decision(
                torch.zeros(2, 2),
                torch.zeros(2, dtype=torch.long),
                torch.ones(2, dtype=torch.long),
                gate=0.2,
                coverage_fraction=1.1,
            )

    def test_all_real_conflict_margin_diagnostics(self):
        logs = []
        _log_real_comparator_margin_distribution(
            torch.tensor([0.05, 0.10, 0.15, 0.20, 0.30]),
            cycle=3,
            gate=0.20,
            log_fn=logs.append,
        )
        self.assertEqual(len(logs), 2)
        self.assertIn("cycle=3; total=5", logs[0])
        self.assertIn("mean=0.1600", logs[0])
        self.assertIn("p50=0.1500", logs[0])
        self.assertIn("p75=0.2000", logs[0])
        self.assertIn("p90=0.2600", logs[0])
        self.assertIn("p95=0.2800", logs[0])
        self.assertIn("max=0.3000", logs[0])
        self.assertIn("gate=0.20", logs[0])
        self.assertIn("margin_ge_0.10=4/5 (80.00%)", logs[1])
        self.assertIn("margin_ge_0.15=3/5 (60.00%)", logs[1])
        self.assertIn("margin_ge_0.20=2/5 (40.00%)", logs[1])
        self.assertIn("margin_ge_0.25=1/5 (20.00%)", logs[1])
        self.assertIn("margin_ge_0.30=1/5 (20.00%)", logs[1])
        self.assertTrue(
            all("ground_truth_affects_training=False" in line for line in logs)
        )

    def test_agreement_ambiguity_shared_top2_diagnostic(self):
        task_probs = torch.tensor(
            [
                [0.50, 0.40, 0.10],  # shared A=0, B=1, most ambiguous
                [0.80, 0.15, 0.05],  # shared A=0, B=1, least ambiguous
                [0.60, 0.10, 0.30],  # agreement, but different Top2
                [0.10, 0.55, 0.35],  # shared A=1, B=2
                [0.55, 0.35, 0.10],  # strict conflict; excluded
            ]
        )
        clip_probs = torch.tensor(
            [
                [0.45, 0.40, 0.15],
                [0.75, 0.20, 0.05],
                [0.58, 0.30, 0.12],
                [0.10, 0.52, 0.38],
                [0.35, 0.55, 0.10],
            ]
        )
        labels = torch.tensor([1, 0, 0, 0, 1])
        task_before = task_probs.clone()
        clip_before = clip_probs.clone()
        logs = []
        result = _log_agreement_ambiguity_eval_only(
            task_probs,
            clip_probs,
            labels,
            fractions=[10, 25, 50, 100],
            cycle=2,
            log_fn=logs.append,
        )

        self.assertEqual(result["agreement_total"], 4)
        self.assertEqual(result["shared_top2_agreement_count"], 3)
        self.assertEqual(result["different_top2_count"], 1)
        row_25 = next(
            row for row in result["fractions"] if row["fraction"] == 25
        )
        self.assertEqual(row_25["count"], 1)
        self.assertEqual(row_25["gt_is_top1_count"], 0)
        self.assertEqual(row_25["gt_is_top2_count"], 1)
        self.assertEqual(row_25["gt_neither_count"], 0)
        self.assertEqual(row_25["candidate_oracle_acc"], 100.0)
        self.assertEqual(len(logs), 5)
        self.assertIn("fraction=25%; n=1", logs[1])
        self.assertIn("top2_recovery_rate=100.00%", logs[1])
        self.assertIn("agreement_total=4", logs[-1])
        self.assertIn("shared_top2_agreement_count=3", logs[-1])
        self.assertIn("different_top2_count=1", logs[-1])
        self.assertIn("ambiguous_25_count=1", logs[-1])
        self.assertTrue(torch.equal(task_probs, task_before))
        self.assertTrue(torch.equal(clip_probs, clip_before))
        self.assertTrue(
            all("ground_truth_affects_training=False" in line for line in logs)
        )
        self.assertTrue(all("selection_uses_gt=False" in line for line in logs))

    def test_agreement_ambiguity_handles_no_shared_top2(self):
        task_probs = torch.tensor([[0.70, 0.10, 0.20]])
        clip_probs = torch.tensor([[0.70, 0.20, 0.10]])
        logs = []
        result = _log_agreement_ambiguity_eval_only(
            task_probs,
            clip_probs,
            torch.tensor([0]),
            fractions=[10, 25, 50, 100],
            cycle=3,
            log_fn=logs.append,
        )
        self.assertEqual(result["agreement_total"], 1)
        self.assertEqual(result["shared_top2_agreement_count"], 0)
        self.assertEqual(result["different_top2_count"], 1)
        self.assertEqual(result["fractions"], [])
        self.assertEqual(len(logs), 1)
        self.assertIn("ambiguous_25_count=0", logs[0])
        self.assertIn("candidate_oracle_acc=nan", logs[0])

    def test_agreement_candidate_probe_uses_a_b_positions_only(self):
        task_probs = torch.tensor(
            [
                [0.50, 0.40, 0.10],
                [0.80, 0.15, 0.05],
                [0.10, 0.55, 0.35],
            ]
        )
        clip_probs = torch.tensor(
            [
                [0.45, 0.40, 0.15],
                [0.75, 0.20, 0.05],
                [0.10, 0.52, 0.38],
            ]
        )
        task_features = torch.randn(3, 4)
        clip_features = torch.randn(3, 4)
        bank_features = torch.randn(6, 4)
        bank_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        bank_scores = torch.arange(6, dtype=torch.float32)
        task_bank = ClassBalancedAnchorBank(3, 2, 4).update(
            bank_features, bank_labels, bank_scores
        )
        clip_bank = ClassBalancedAnchorBank(3, 2, 4).update(
            bank_features, bank_labels, bank_scores
        )
        model = PairwiseConflictComparator(
            input_dim=16, hidden=4, layers=1, dropout=0.0
        )
        for parameter in model.parameters():
            nn.init.zeros_(parameter)
        # Force output position 1, which the probe interprets as choose B.
        model.mlp[-1].bias.data[1] = 1.0
        logs = []
        result = _log_agreement_candidate_probe_eval_only(
            task_probs,
            clip_probs,
            task_features,
            clip_features,
            torch.tensor([1, 0, 0]),
            comparator=model,
            task_bank=task_bank,
            clip_bank=clip_bank,
            sim_topk=2,
            fractions=[10, 25, 50, 100],
            cycle=2,
            log_fn=logs.append,
        )
        row_25 = next(
            row for row in result["fractions"] if row["fraction"] == 25
        )
        self.assertEqual(row_25["count"], 1)
        self.assertEqual(row_25["choose_b_count"], 1)
        self.assertEqual(row_25["recovered_top1_errors"], 1)
        self.assertEqual(row_25["overridden_correct_top1"], 0)
        self.assertEqual(row_25["net_corrections"], 1)
        self.assertEqual(row_25["comparator_acc"], 100.0)
        fraction_25_log = next(
            line for line in logs if "fraction=25%" in line
        )
        self.assertIn(
            "output_semantics=0_choose_A_1_choose_B", fraction_25_log
        )
        self.assertIn("admission_changed=False", fraction_25_log)
        self.assertIn("ambiguous_25_count=1", logs[-1])
        self.assertTrue(
            all("ground_truth_affects_training=False" in line for line in logs)
        )

    def test_agreement_synthetic_feasibility_builds_both_candidate_directions(self):
        pool_labels = torch.tensor([0, 1, 2, 1])
        strong_task_probs = torch.tensor(
            [
                [0.60, 0.30, 0.10],  # A=0, B=1, Y=A
                [0.48, 0.46, 0.06],  # A=0, B=1, Y=B; most ambiguous
                [0.55, 0.35, 0.10],  # A=0, B=1, Y neither
                [0.35, 0.55, 0.10],  # A=1, Task B=0
            ]
        )
        strong_clip_probs = torch.tensor(
            [
                [0.58, 0.32, 0.10],
                [0.49, 0.45, 0.06],
                [0.52, 0.38, 0.10],
                [0.10, 0.56, 0.34],  # A=1, CLIP B=2; excluded
            ]
        )
        task_features = torch.randn(4, 5)
        clip_features = torch.randn(4, 5)
        bank_features = torch.randn(6, 5)
        bank_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        bank_scores = torch.arange(6, dtype=torch.float32)
        task_bank = ClassBalancedAnchorBank(3, 2, 5).update(
            bank_features, bank_labels, bank_scores
        )
        clip_bank = ClassBalancedAnchorBank(3, 2, 5).update(
            bank_features, bank_labels, bank_scores
        )
        logs = []
        result = _log_agreement_synthetic_feasibility_eval_only(
            pool_labels,
            strong_task_probs,
            strong_clip_probs,
            task_features,
            clip_features,
            task_bank=task_bank,
            clip_bank=clip_bank,
            sim_topk=2,
            fractions=[10, 25, 50, 100],
            cycle=2,
            log_fn=logs.append,
            pool_gt_labels=torch.tensor([0, 1, 2, 1]),
        )
        self.assertEqual(result["anchor_pool_total"], 4)
        self.assertEqual(result["strong_shared_pair_count"], 3)
        self.assertEqual(result["pseudo_y_is_a_count"], 1)
        self.assertEqual(result["pseudo_y_is_b_count"], 1)
        self.assertEqual(result["pseudo_y_neither_count"], 1)
        self.assertEqual(result["usable_count"], 2)
        row_25 = next(
            row for row in result["fractions"] if row["fraction"] == 25
        )
        self.assertEqual(row_25["count"], 1)
        self.assertEqual(row_25["choose_b_count"], 1)
        self.assertEqual(row_25["choose_b_share"], 100.0)
        self.assertEqual(row_25["pseudo_target_precision"], 100.0)
        self.assertTrue(
            any("agreement-synthetic-choose-A" in line for line in logs)
        )
        self.assertTrue(
            any("agreement-synthetic-choose-B" in line for line in logs)
        )
        self.assertIn("strong_shared_pair_count=3", logs[-1])
        self.assertIn("pseudo_Y_is_B_count=1", logs[-1])
        self.assertIn("training_changed=False", logs[-1])
        self.assertTrue(
            all("ground_truth_affects_training=False" in line for line in logs)
        )

    def test_agreement_synthetic_feasibility_handles_no_shared_pair(self):
        probs_task = torch.tensor([[0.60, 0.30, 0.10]])
        probs_clip = torch.tensor([[0.60, 0.10, 0.30]])
        features = torch.randn(1, 4)
        bank_features = torch.randn(3, 4)
        bank_labels = torch.tensor([0, 1, 2])
        bank_scores = torch.ones(3)
        task_bank = ClassBalancedAnchorBank(3, 1, 4).update(
            bank_features, bank_labels, bank_scores
        )
        clip_bank = ClassBalancedAnchorBank(3, 1, 4).update(
            bank_features, bank_labels, bank_scores
        )
        logs = []
        result = _log_agreement_synthetic_feasibility_eval_only(
            torch.tensor([0]),
            probs_task,
            probs_clip,
            features,
            features,
            task_bank=task_bank,
            clip_bank=clip_bank,
            sim_topk=1,
            fractions=[10, 25, 50, 100],
            cycle=3,
            log_fn=logs.append,
        )
        self.assertEqual(result["strong_shared_pair_count"], 0)
        self.assertEqual(result["usable_count"], 0)
        self.assertEqual(result["fractions"], [])
        self.assertEqual(len(logs), 1)
        self.assertIn("strong_shared_pair_count=0", logs[0])
        self.assertIn("construction_uses_gt=False", logs[0])

    def test_agreement_ambiguity_pipeline_is_eval_only(self):
        torch.manual_seed(21)
        n, c, d = 256, 6, 16
        feat = torch.randn(n, d)
        proto = torch.randn(c, d)
        sim = feat @ proto.t()
        task_prob = torch.softmax(
            sim * 2.5 + torch.randn(n, c) * 0.4, dim=1
        )
        clip_prob = torch.softmax(
            sim * 2.2 + torch.randn(n, c) * 0.6, dim=1
        )
        clip_feat = torch.randn(n, d)
        labels = sim.argmax(dim=1)
        strong_task = torch.softmax(
            sim * 1.2 + torch.randn(n, c) * 0.8, dim=1
        )
        strong_clip = torch.softmax(
            sim * 1.0 + torch.randn(n, c) * 0.9, dim=1
        )
        base_model = PairwiseConflictComparator(
            input_dim=16, hidden=16, layers=2, dropout=0.0
        )
        diagnostic_model = PairwiseConflictComparator(
            input_dim=16, hidden=16, layers=2, dropout=0.0
        )
        diagnostic_model.load_state_dict(base_model.state_dict())
        base_optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
        diagnostic_optimizer = torch.optim.Adam(
            diagnostic_model.parameters(), lr=1e-3
        )
        common = dict(
            task_probs=task_prob,
            clip_probs=clip_prob,
            task_features=feat,
            num_classes=c,
            pre_prior_task_probs=task_prob,
            pre_prior_clip_probs=clip_prob,
            labels=labels,
            sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task,
            strong_clip_probs=strong_clip,
            strong_task_features=feat,
            strong_clip_features=clip_feat,
            cycle=2,
        )
        torch.manual_seed(99)
        base_result = run_context_refinement(
            **common,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator",
                TRAIN_STEPS_PER_CYCLE=20,
                EVAL_ONLY_LOGGING=True,
                AGREEMENT_AMBIGUITY_EVAL_ENABLED=False,
            ),
            comparator=base_model,
            comparator_optimizer=base_optimizer,
            log_fn=lambda _: None,
        )
        diagnostic_logs = []
        torch.manual_seed(99)
        diagnostic_result = run_context_refinement(
            **common,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator",
                TRAIN_STEPS_PER_CYCLE=20,
                EVAL_ONLY_LOGGING=True,
                AGREEMENT_AMBIGUITY_EVAL_ENABLED=True,
                AGREEMENT_COMPARATOR_PROBE_ENABLED=True,
                AGREEMENT_SYNTHETIC_FEASIBILITY_ENABLED=True,
                REAL_CONFLICT_GT_PROBE_ENABLED=True,
                REAL_CONFLICT_GT_PROBE_STEPS=10,
                REAL_CONFLICT_GT_PROBE_EXTENDED_20D_ENABLED=True,
            ),
            comparator=diagnostic_model,
            comparator_optimizer=diagnostic_optimizer,
            log_fn=diagnostic_logs.append,
        )
        for base_value, diagnostic_value in zip(
            base_model.state_dict().values(),
            diagnostic_model.state_dict().values(),
        ):
            self.assertTrue(torch.equal(base_value, diagnostic_value))
        for key in (
            "resolved_mask",
            "context_labels",
            "refined_targets",
            "anchor_mask",
            "strict_conflict_mask",
        ):
            self.assertTrue(torch.equal(base_result[key], diagnostic_result[key]))
        self.assertTrue(
            any("DUET agreement ambiguity summary eval-only" in line for line in diagnostic_logs)
        )
        self.assertTrue(
            any("DUET agreement candidate probe summary eval-only" in line for line in diagnostic_logs)
        )
        self.assertTrue(
            any("DUET agreement synthetic feasibility summary eval-only" in line for line in diagnostic_logs)
        )
        self.assertTrue(
            any("DUET real-conflict GT feature probe" in line for line in diagnostic_logs)
        )
        self.assertTrue(
            any(
                "DUET real-conflict GT feature probe 20D summary eval-only"
                in line
                for line in diagnostic_logs
            )
        )
        self.assertTrue(
            any(
                "DUET real-conflict 20D extra-feature distribution eval-only"
                in line
                and "construction_uses_gt=False" in line
                for line in diagnostic_logs
            )
        )
        self.assertTrue(
            any(
                "DUET real-conflict GT feature probe paired summary eval-only"
                in line
                for line in diagnostic_logs
            )
        )

    def test_comparator_pipeline_end_to_end(self):
        """整条 comparator 管线：resolved 标签只可能是 A 或 B，
        refined target 的 argmax == chosen，weak 全部保持原样。"""
        torch.manual_seed(2)
        n, c, d = 512, 8, 32
        feat = torch.randn(n, d)
        proto = torch.randn(c, d)
        sim = feat @ proto.t()
        task_prob = torch.softmax(sim * 3.0 + torch.randn(n, c) * 0.2, dim=1)
        clip_prob = torch.softmax(sim * 2.8 + torch.randn(n, c) * 0.4, dim=1)
        clip_feat = torch.randn(n, d)
        true_label = sim.argmax(dim=1)
        strong_task = torch.softmax(sim * 1.5 + torch.randn(n, c) * 0.7, dim=1)
        strong_clip = torch.softmax(sim * 1.3 + torch.randn(n, c) * 0.8, dim=1)
        model = PairwiseConflictComparator(input_dim=16, hidden=32, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logs = []
        result = run_context_refinement(
            task_prob, clip_prob, feat, num_classes=c,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator", TRAIN_STEPS_PER_CYCLE=30,
                COMPARATOR_GATE=0.15,
                REAL_MULTIVIEW_ENABLED=True,
                REAL_MULTIVIEW_TRAIN_FRACTION=0.60,
                REAL_MULTIVIEW_FINETUNE_STEPS=5,
                CONFLICT_MEMORY_ENABLED=True,
                CONFLICT_MEMORY_COVERAGE_FRACTION=0.80,
            ),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=true_label, sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task, strong_clip_probs=strong_clip,
            strong_task_features=feat, strong_clip_features=clip_feat,
            comparator=model, comparator_optimizer=optimizer,
            conflict_belief_memory=PersistentConflictBeliefMemory(),
            cycle=2, log_fn=logs.append,
        )
        resolved = result["resolved_mask"]
        self.assertEqual(int(result["weak_rejected_mask"].sum().item()), 0)
        if int(resolved.sum().item()) > 0:
            chosen = result["context_labels"][resolved]
            task_side = task_prob[resolved].argmax(dim=1)
            clip_side = clip_prob[resolved].argmax(dim=1)
            self.assertTrue(
                ((chosen == task_side) | (chosen == clip_side)).all()
            )
            self.assertTrue(
                (
                    result["refined_targets"][resolved].argmax(dim=1) == chosen
                ).all()
            )
        self.assertTrue(any("DUET comparator synthetic conflicts" in line for line in logs))
        multiview_log = next(
            line
            for line in logs
            if "DUET comparator real-multiview training" in line
        )
        self.assertIn("train_coverage=", multiview_log)
        self.assertIn("construction_uses_gt=False", multiview_log)
        conflict_total = int(result["strict_conflict_mask"].sum().item())
        memory_active = int(result["conflict_memory"]["active"].sum().item())
        self.assertEqual(memory_active, round(0.80 * conflict_total))
        self.assertTrue(
            any("DUET persistent conflict memory" in line for line in logs)
        )

    def test_residual_soft_pipeline_uses_fallback_vs_challenger_without_hard_delta(self):
        torch.manual_seed(23)
        n, c, d = 512, 8, 32
        feat = torch.randn(n, d)
        proto = torch.randn(c, d)
        sim = feat @ proto.t()
        task_prob = torch.softmax(sim * 3.0 + torch.randn(n, c) * 0.3, dim=1)
        clip_prob = torch.softmax(sim * 2.7 + torch.randn(n, c) * 0.5, dim=1)
        clip_feat = torch.randn(n, d)
        strong_task = torch.softmax(sim * 1.4 + torch.randn(n, c) * 0.8, dim=1)
        strong_clip = torch.softmax(sim * 1.2 + torch.randn(n, c) * 0.9, dim=1)
        model = PairwiseConflictComparator(input_dim=16, hidden=32, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logs = []
        result = run_context_refinement(
            task_prob,
            clip_prob,
            feat,
            num_classes=c,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator",
                TRAIN_STEPS_PER_CYCLE=0,
                COMPARATOR_COVERAGE_FRACTION=0.50,
                REAL_MULTIVIEW_ENABLED=True,
                REAL_MULTIVIEW_RESIDUAL_FALLBACK=True,
                REAL_MULTIVIEW_TRAIN_FRACTION=0.60,
                REAL_MULTIVIEW_FINETUNE_STEPS=0,
                REAL_MULTIVIEW_SYNTHETIC_MIX_FRACTION=0.0,
                SOFT_ONLY_ADMISSION=True,
            ),
            pre_prior_task_probs=task_prob,
            pre_prior_clip_probs=clip_prob,
            labels=sim.argmax(dim=1),
            sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task,
            strong_clip_probs=strong_clip,
            strong_task_features=feat,
            strong_clip_features=clip_feat,
            comparator=model,
            comparator_optimizer=optimizer,
            cycle=2,
            log_fn=logs.append,
        )
        resolved = result["resolved_mask"]
        fallback = (task_prob + clip_prob).argmax(dim=1)
        task_top1 = task_prob.argmax(dim=1)
        clip_top1 = clip_prob.argmax(dim=1)
        rows = torch.arange(n)
        mixed = 0.5 * (task_prob + clip_prob)
        stronger = torch.where(
            mixed[rows, task_top1] >= mixed[rows, clip_top1],
            task_top1,
            clip_top1,
        )
        challenger = torch.where(
            fallback == task_top1,
            clip_top1,
            torch.where(fallback == clip_top1, task_top1, stronger),
        )
        chosen = result["context_labels"][resolved]
        self.assertTrue(
            ((chosen == fallback[resolved]) | (chosen == challenger[resolved])).all()
        )
        kept = resolved & (result["context_labels"] == fallback)
        if int(kept.sum().item()) > 0:
            self.assertTrue(
                torch.equal(result["refined_targets"][kept], clip_prob[kept])
            )
        self.assertEqual(result["stats"]["admitted_delta"], 0)
        self.assertEqual(result["stats"]["optimizer_steps_this_cycle"], 0)
        self.assertTrue(
            any("candidate_a=duet_fallback" in line for line in logs)
        )
        self.assertTrue(
            any("residual_fallback=True" in line for line in logs)
        )
        self.assertTrue(
            any("router=direct_strong_neighborhood_evidence" in line for line in logs)
        )
        self.assertTrue(
            any("DUET comparator real-conflict distribution" in line for line in logs)
        )
        self.assertTrue(
            any("DUET comparator synthetic distribution" in line for line in logs)
        )
        self.assertTrue(
            any("DUET comparator real-margin distribution" in line for line in logs)
        )
        self.assertTrue(
            any("DUET comparator real-margin thresholds" in line for line in logs)
        )

    def test_dist_match_integration(self):
        """distribution matching 开启后：日志出现、kept 不超过 total、
        管线照常跑通（可能全部被滤掉也不报错）。"""
        torch.manual_seed(2)
        n, c, d = 512, 8, 32
        feat = torch.randn(n, d)
        proto = torch.randn(c, d)
        sim = feat @ proto.t()
        task_prob = torch.softmax(sim * 3.0 + torch.randn(n, c) * 0.2, dim=1)
        clip_prob = torch.softmax(sim * 2.8 + torch.randn(n, c) * 0.4, dim=1)
        clip_feat = torch.randn(n, d)
        true_label = sim.argmax(dim=1)
        strong_task = torch.softmax(sim * 1.5 + torch.randn(n, c) * 0.7, dim=1)
        strong_clip = torch.softmax(sim * 1.3 + torch.randn(n, c) * 0.8, dim=1)
        model = PairwiseConflictComparator(input_dim=16, hidden=32, layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logs = []
        run_context_refinement(
            task_prob, clip_prob, feat, num_classes=c,
            context_cfg=make_context_cfg(
                REFINER_TYPE="comparator", TRAIN_STEPS_PER_CYCLE=30,
                COMPARATOR_GATE=0.15, DIST_MATCH_SYNTHETIC=True,
            ),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=true_label, sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task, strong_clip_probs=strong_clip,
            strong_task_features=feat, strong_clip_features=clip_feat,
            comparator=model, comparator_optimizer=optimizer,
            cycle=2, log_fn=logs.append,
        )
        match_lines = [line for line in logs if "DUET comparator dist-match" in line]
        self.assertEqual(len(match_lines), 1)
        self.assertIn("synthetic_total=", match_lines[0])
        self.assertIn("kept=", match_lines[0])
        self.assertIn("before_trust_task=", match_lines[0])
        self.assertIn("before_trust_clip=", match_lines[0])
        self.assertIn("kept_trust_task=", match_lines[0])
        self.assertIn("kept_trust_clip=", match_lines[0])
        self.assertIn("ground_truth_affects_training=False", match_lines[0])

    def test_controls_shapes_and_no_nan(self):
        feat, task_prob, clip_prob, true_label = make_separable(n=256, num_classes=6, feature_dim=24)
        bank = ClassBalancedAnchorBank(6, 4, 24)
        task_conf, task_top1 = task_prob.max(dim=1)
        anchor_mask = task_top1 == clip_prob.argmax(dim=1)
        bank.update(feat[anchor_mask], task_top1[anchor_mask], torch.rand(anchor_mask.sum()))
        af, al, _, _, av = bank.flatten()
        for refiner in (cosine_knn_refine, prototype_refine):
            probs = refiner(feat[:10], af, al, av, 6, k=3) if refiner is cosine_knn_refine \
                else refiner(feat[:10], af, al, av, 6)
            self.assertEqual(probs.shape, (10, 6))
            self.assertTrue(torch.isfinite(probs).all())


class MethodFileContractTest(unittest.TestCase):
    """AST-level contract checks for the training-loop integration."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(
            Path("src/methods/oh/duet_first_cycle_prior_context_transformer.py").read_text()
        )

    def _function(self, name):
        return next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_train_target_defaults(self):
        fn = self._function("train_target")
        defaults = dict(
            zip(
                (arg.arg for arg in fn.args.kwonlyargs),
                fn.args.kw_defaults,
            )
        )
        self.assertTrue(defaults["first_cycle_prior"].value)
        self.assertTrue(defaults["context_conflict_transformer"].value)

    def test_context_requires_first_cycle_prior(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn(
            "duet_first_cycle_prior_context_transformer requires ",
            source,
        )
        self.assertIn("first_cycle_prior=True", source)

    def test_other_candidates_removed_from_clean_file(self):
        """train_target / obtain_label 中不得再出现其他候选方法 / swap /
        Gate D / strong-feature 收集相关代码。"""
        tree = ast.parse(
            Path(
                "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
            ).read_text()
        )
        source = "\n".join(
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("train_target", "obtain_label")
        )
        for removed in (
            "boundary_router",
            "attribute_reliability_kl",
            "support_conditioned_clip",
            "clip_confidence_delay",
            "pcgrad",
            "topk_conflict_probe",
            "swap_conflict_selection",
            "swap_audit",
            "gate_D",
            "GATE_D",
            "collect_strong",
        ):
            self.assertNotIn(removed, source, "should have removed: " + removed)

    def test_prior_runs_before_transformer(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertLess(
            source.index("apply_first_cycle_prior("),
            source.index("run_context_refinement("),
        )

    def test_resolved_updates_label_mask_kl_soft_mix(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn("label_mask = label_mask | resolved_mask", source)
        self.assertIn("kl_soft_output[resolved_mask]", source)
        self.assertIn('all_mix_output[context_payload["resolved_mask"]]', source)
        self.assertIn("mem_label==kl_soft_argmax", source)

    def test_weak_rejected_admission_false(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn(
            'admission_matching[context_payload["weak_rejected_mask"]] = False', source
        )

    def test_soft_only_admission_gate(self):
        """soft-only 消融：resolved 只做 KL soft target，不进 label_mask。"""
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn("SOFT_ONLY_ADMISSION", source)
        self.assertIn("DUET context soft-only", source)
        self.assertIn("hard_admission=0", source)

    def test_transition_auxiliary_uses_clip_kl_residual(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn('payload.get("residual_pairwise", False)', source)
        self.assertIn('payload["baseline_q"]', source)
        self.assertIn('payload["baseline_pair_mass"]', source)
        self.assertIn("q - baseline_q", source)
        self.assertIn("float(logits.size(0))", source)
        self.assertIn("clip_kl_residual", source)

    def test_transition_residual_matches_interpolated_clip_teacher_gradient(self):
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        function_source = ast.get_source_segment(
            source, self._function("conflict_memory_pairwise_loss")
        )
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(function_source, namespace)
        residual_loss = namespace["conflict_memory_pairwise_loss"]

        clip_target = torch.tensor(
            [[0.60, 0.30, 0.10], [0.20, 0.50, 0.30], [0.10, 0.30, 0.60]]
        )
        candidate_a = torch.tensor([0, 0, 1])
        candidate_b = torch.tensor([1, 1, 2])
        rows = torch.arange(3)
        pair_mass = (
            clip_target[rows, candidate_a] + clip_target[rows, candidate_b]
        )
        baseline_q = clip_target[rows, candidate_a] / pair_mass
        q = torch.tensor([0.85, 0.40, 0.25])
        weights = torch.tensor([0.50, 0.00, 0.80])
        active = torch.tensor([True, False, True])
        payload = {
            "active": active,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "q": q,
            "baseline_q": baseline_q,
            "baseline_pair_mass": pair_mass,
            "weight": weights,
            "residual_pairwise": True,
        }

        logits = torch.randn(3, 3, requires_grad=True)
        original_kl = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits, dim=1),
            clip_target,
            reduction="batchmean",
        )
        combined = original_kl + residual_loss(
            logits, torch.arange(3), payload
        )
        combined_grad = torch.autograd.grad(combined, logits)[0]

        corrected_target = clip_target.clone()
        delta = weights * pair_mass * (q - baseline_q)
        corrected_target[rows, candidate_a] += delta
        corrected_target[rows, candidate_b] -= delta
        expected_logits = logits.detach().clone().requires_grad_(True)
        expected = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(expected_logits, dim=1),
            corrected_target,
            reduction="batchmean",
        )
        expected_grad = torch.autograd.grad(expected, expected_logits)[0]
        self.assertTrue(torch.allclose(combined_grad, expected_grad, atol=1e-7))

    def test_strong_feature_collection_guarded_by_comparator_mode(self):
        """Regression：旧版 collect_strong 未定义就引用 strong_feas 的
        UnboundLocalError 已不存在；comparator 模式会在 comparator_mode
        保护下收集 strong 视图的 Task/CLIP feature（同 view synthetic 需要）。"""
        fn = self._function("obtain_label")
        body = ast.unparse(fn)
        self.assertNotIn("all_strong_features", body)
        self.assertNotIn("collect_strong", body)
        self.assertIn("all_strong_task_features", body)
        self.assertIn("all_strong_clip_features", body)
        self.assertIn("strong_task_feature", body)

    def test_no_modification_of_original_files(self):
        for name in ("plmatch.py", "plmatch_clean.py", "duet_first_cycle_prior.py"):
            self.assertNotIn(
                "context_conflict_transformer",
                Path("src/methods/oh", name).read_text(),
            )

    def test_entrypoint_registers_candidate_before_fcp_prefix(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            "import src.methods.oh.duet_first_cycle_prior_context_transformer",
            entrypoint,
        )
        context_branch = entrypoint.index(
            '"duet_first_cycle_prior_context_transformer_"'
        )
        fcp_branch = entrypoint.index(
            'cfg.MODEL.METHOD.startswith("duet_first_cycle_prior_")'
        )
        self.assertLess(context_branch, fcp_branch)

    def test_cfg_files_use_power_08_and_enabled(self):
        import yaml

        for dataset in ("office-home", "visda"):
            with open(
                Path(
                    "cfgs/{}/duet_first_cycle_prior_context_transformer.yaml".format(
                        dataset
                    )
                )
            ) as handle:
                data = yaml.safe_load(handle)
            self.assertEqual(data["DUET_FCP"]["POWER"], 0.8)
            self.assertTrue(data["DUET_CONTEXT"]["ENABLED"])
            self.assertEqual(data["DUET_CONTEXT"]["USE_WEAK_AGREEMENT"], False)
            self.assertEqual(data["DUET_CONTEXT"]["REFINER_TYPE"], "comparator")
            if dataset == "office-home":
                self.assertEqual(
                    data["DUET_CONTEXT"]["COMPARATOR_COVERAGE_FRACTION"],
                    0.20,
                )
            active_cycles = data["DUET_CONTEXT"]["ACTIVE_CYCLES"]
            self.assertIsInstance(active_cycles, list)
            self.assertTrue(all(isinstance(v, int) for v in active_cycles))
            self.assertNotIn(0, active_cycles)  # 第一轮保持纯 FCP

    def test_first_cycle_stays_pure_fcp(self):
        """第一轮（cycle index 0）不运行 Transformer：默认 ACTIVE_CYCLES
        从 index 1（第 2 个 cycle）开始。"""
        conf = Path("conf.py").read_text()
        self.assertIn("_C.DUET_CONTEXT.ACTIVE_CYCLES = [1]", conf)
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn("第一轮（cycle index 0）保持纯 DUET-FCP", source)

    def test_inactive_cycle_correction_line(self):
        """未激活的 cycle 也输出一行 corrections=0，便于逐轮对比。"""
        fn = self._function("obtain_label")
        body = ast.unparse(fn)
        self.assertIn("DUET context correction: cycle={}; active=False", body)
        self.assertIn("corrections=0", body)


if __name__ == "__main__":
    unittest.main()
