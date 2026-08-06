"""Unit tests for duet_first_cycle_prior_context_transformer core modules.

Covers the 18 checks requested in section 18 of the task.  All tests run on
CPU with synthetic tensors; they never import the heavy training loop (which
needs cv2 / yacs / a GPU), only ``src.utils.duet_context`` and AST contracts
of the method entry file.
"""

import ast
import unittest
from pathlib import Path

import torch

from src.utils.duet_context import (
    ClassBalancedAnchorBank,
    DuetContextConflictTransformer,
    apply_decision_rules,
    apply_weak_verification,
    cosine_knn_refine,
    prototype_refine,
    run_context_refinement,
    train_context_transformer,
    _exclude_query_anchors,
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
            pre_prior_task_probs=task_prob,
            pre_prior_clip_probs=clip_prob,
            labels=true_label,
            sample_indices=torch.arange(feat.size(0)),
            anchors_per_class=8,
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None,
            train_steps_per_cycle=0,
            train_batch_size=32,
            seed=2020,
            cycle=1,
            eval_only_logging=False,
        )
        kwargs.update(overrides)
        return run_context_refinement(**kwargs)

    def test_03_ground_truth_never_affects_decisions(self):
        feat, task_prob, clip_prob, true_label = make_separable()
        base = dict(
            task_probs=task_prob, clip_probs=clip_prob, task_features=feat,
            num_classes=8, pre_prior_task_probs=task_prob,
            pre_prior_clip_probs=clip_prob, anchors_per_class=8,
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None, train_steps_per_cycle=0, seed=2020,
            eval_only_logging=False,
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
            pre_prior_task_probs=task_prob, pre_prior_clip_probs=clip_prob,
            labels=torch.arange(feat.size(0)) % 8,
            sample_indices=torch.arange(feat.size(0)),
            anchors_per_class=8,
            transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            optimizer=None, train_steps_per_cycle=0, seed=2020,
            eval_only_logging=False,
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
        result = self._run(abstain_when_uncertain=False)
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
            labels=true_label, sample_indices=torch.arange(feat.size(0)),
            use_strict_conflict=False, use_weak_agreement=False,
            anchors_per_class=8, transformer=DuetContextConflictTransformer(
                feature_dim=32, num_classes=8, model_dim=32, num_heads=4, ffn_dim=64
            ),
            eval_only_logging=False,
        )
        self.assertEqual(int(result["resolved_mask"].sum().item()), 0)
        self.assertEqual(int(result["weak_rejected_mask"].sum().item()), 0)
        self.assertTrue(torch.allclose(result["refined_targets"], clip_prob, atol=1e-6))

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

    def test_strong_feature_collection_removed(self):
        """Regression: 之前 obtain_label 里 collect_features=True 而
        collect_strong=False 会 UnboundLocalError；清理后 obtain_label 不再
        收集 strong 特征。注意：内部训练循环的 strong_x 一致性仍在。"""
        fn = self._function("obtain_label")
        body = ast.unparse(fn)
        self.assertNotIn("all_strong_features", body)
        self.assertNotIn("strong_feas", body)
        self.assertNotIn("collect_strong", body)

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
            self.assertEqual(data["DUET_CONTEXT"]["ACTIVE_CYCLES"], [1])

    def test_first_cycle_stays_pure_fcp(self):
        """第一轮（cycle index 0）不运行 Transformer：默认 ACTIVE_CYCLES
        从 index 1（第 2 个 cycle）开始。"""
        conf = Path("conf.py").read_text()
        self.assertIn("_C.DUET_CONTEXT.ACTIVE_CYCLES = [1]", conf)
        source = Path(
            "src/methods/oh/duet_first_cycle_prior_context_transformer.py"
        ).read_text()
        self.assertIn("第一轮（cycle index 0）保持纯 DUET-FCP", source)


if __name__ == "__main__":
    unittest.main()
