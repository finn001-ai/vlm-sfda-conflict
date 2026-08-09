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

from src.utils.duet_context import (
    ClassBalancedAnchorBank,
    ComparatorReplayMemory,
    DuetContextConflictTransformer,
    PairwiseConflictComparator,
    apply_pairwise_decision,
    apply_decision_rules,
    apply_weak_verification,
    build_comparator_features,
    build_synthetic_conflicts,
    cosine_knn_refine,
    prototype_refine,
    run_context_refinement,
    train_pairwise_comparator,
    train_context_transformer,
    _exclude_query_anchors,
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
        REPLAY_PER_DIRECTION=64,
        REPLAY_MIX_FRACTION=0.25,
        SOFT_ONLY_ADMISSION=False,
        DIST_MATCH_SYNTHETIC=False,
        DIST_MATCH_Z_MAX=1.5,
        DIST_MATCH_DIMS=[4, 5, 6, 7],
        MIN_DIST_MATCH_KEPT=16,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class AnchorBankTest(unittest.TestCase):
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
        self.assertIn("ground_truth_affects_training=False", joined)


class ComparatorTest(unittest.TestCase):
    """Pairwise conflict-resolution（REFINER_TYPE=comparator）测试。"""

    def _small_bank_pair(self, num_classes=6, dim=8):
        task_bank = ClassBalancedAnchorBank(num_classes, 2, dim)
        clip_bank = ClassBalancedAnchorBank(num_classes, 2, dim)
        return task_bank, clip_bank

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

    def test_decision_margin_gate(self):
        logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.05, -0.05]])
        task_top1 = torch.tensor([0, 0, 0])
        clip_top1 = torch.tensor([1, 1, 1])
        decision = apply_pairwise_decision(logits, task_top1, clip_top1, gate=0.2)
        self.assertEqual(decision["resolved"].tolist(), [True, True, False])
        self.assertEqual(decision["chosen"][0].item(), 0)  # trust Task
        self.assertEqual(decision["chosen"][1].item(), 1)  # trust CLIP

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
            ),
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=true_label, sample_indices=torch.arange(n),
            clip_features=clip_feat,
            strong_task_probs=strong_task, strong_clip_probs=strong_clip,
            strong_task_features=feat, strong_clip_features=clip_feat,
            comparator=model, comparator_optimizer=optimizer,
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
        self.assertTrue(
            any("DUET comparator real-conflict distribution" in line for line in logs)
        )
        self.assertTrue(
            any("DUET comparator synthetic distribution" in line for line in logs)
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
