import unittest
from pathlib import Path

import torch

from src.utils.duet_delayed_credit import (
    initialize_delayed_credit,
    normalized_js_divergence,
    update_delayed_credit,
)


class DuetDelayedCreditTest(unittest.TestCase):
    def test_normalized_js_is_symmetric_and_bounded(self):
        left = torch.tensor([[0.9, 0.1], [0.4, 0.6]])
        right = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
        forward = normalized_js_divergence(left, right)
        reverse = normalized_js_divergence(right, left)
        self.assertTrue(torch.allclose(forward, reverse, atol=1e-7))
        self.assertGreaterEqual(float(forward.min().item()), 0.0)
        self.assertLessEqual(float(forward.max().item()), 1.0)
        self.assertAlmostEqual(float(forward[1].item()), 0.0, places=7)

    def test_identical_experts_remain_equal_fixed_point(self):
        probability = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        state = initialize_delayed_credit(probability, probability)
        updated, diagnostics = update_delayed_credit(
            state,
            probability,
            probability,
        )
        self.assertTrue(
            torch.allclose(updated["memory"], probability, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(updated["task_weight"], torch.full((2,), 0.5))
        )
        self.assertTrue(
            torch.allclose(diagnostics["memory_shift_l1"], torch.zeros(2))
        )

    def test_future_outcome_rewards_better_previous_expert(self):
        previous_task = torch.tensor([[0.9, 0.1]])
        previous_clip = torch.tensor([[0.2, 0.8]])
        state = initialize_delayed_credit(previous_task, previous_clip)
        current_outcome = torch.tensor([[0.82, 0.18]])
        updated, diagnostics = update_delayed_credit(
            state,
            current_outcome,
            current_outcome,
            credit_eta=4.0,
        )
        self.assertLess(
            float(diagnostics["task_delayed_loss"].item()),
            float(diagnostics["clip_delayed_loss"].item()),
        )
        self.assertGreater(
            float(updated["task_weight"].item()),
            float(updated["clip_weight"].item()),
        )

    def test_every_sample_has_positive_continuous_memory_update_rate(self):
        task = torch.tensor([[0.999, 0.001], [0.6, 0.4]])
        clip = torch.tensor([[0.001, 0.999], [0.4, 0.6]])
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        updated, diagnostics = update_delayed_credit(
            state,
            task,
            clip,
            memory_update_rate=0.5,
        )
        self.assertTrue((diagnostics["update_rate"] >= 0.25).all())
        self.assertTrue((diagnostics["update_rate"] <= 0.5).all())
        self.assertTrue((diagnostics["memory_shift_l1"] > 0.0).all())
        self.assertTrue(
            torch.allclose(
                updated["memory"].sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_method_contract_excludes_cosmo_teacher_primitives(self):
        source = Path(
            "src/methods/oh/duet_delayed_agreement_credit.py"
        ).read_text()
        self.assertNotIn("entropy_weighted_poe", source)
        self.assertNotIn("consensus_shift_factors", source)
        self.assertNotIn("modulate_anchored_consensus", source)
        self.assertNotIn("PairwiseConflictComparator", source)
        self.assertIn("soft_coverage=100.00%", source)
        self.assertIn("sample_self_history_only=True", source)

    def test_visda_entry_and_config_are_wired(self):
        entry = Path("image_target_of_oh_vs.py").read_text()
        config = Path(
            "cfgs/visda/duet_delayed_agreement_credit.yaml"
        ).read_text()
        script = Path(
            "tools/run_visda_duet_delayed_agreement_credit_proxy25.sh"
        ).read_text()
        self.assertIn("DUET_DELAYED_CREDIT.train_target(cfg)", entry)
        self.assertIn("METHOD: duet_delayed_agreement_credit", config)
        self.assertIn("--ratio 0.25", script)
        self.assertIn("full_evaluation_samples=${full_samples}", script)

    def test_full_handoff_reuses_dac_and_preserves_duet_budget(self):
        script = Path(
            "tools/run_visda_dac_duet_handoff_full.sh"
        ).read_text()
        self.assertIn("delayed_credit_state.pt", script)
        self.assertIn('"${dac_run_dir}/target_F.pt"', script)
        self.assertIn('"${handoff_source_dir}/source_F.pt"', script)
        self.assertIn('"${dac_run_dir}/target_C.pt"', script)
        self.assertIn("memory_rows=", script)
        self.assertIn("ACTIVE.CYCLE 4", script)
        self.assertIn("cfgs/visda/plmatch.yaml", script)
        self.assertIn("DUET_HANDOFF.FINAL_EXTRA_EPOCHS 1", script)
        self.assertIn("Total target passes: 32", script)
        plmatch = Path("src/methods/oh/plmatch.py").read_text()
        self.assertIn("cycle_max_iter = base_max_iter", plmatch)
        self.assertIn("DUET DAC handoff final-cycle budget:", plmatch)
        self.assertIn('osp.join(cfg.output_dir, "target_F.pt")', plmatch)

    def test_source_classifier_handoff_is_kept_as_an_ablation(self):
        script = Path(
            "tools/run_visda_dac_duet_handoff_fb_sourcec_full.sh"
        ).read_text()
        self.assertIn("DAC F/B + frozen source C", script)
        self.assertIn(
            'source/uda/VISDA-C/T/source_C.pt "${handoff_source_dir}/source_C.pt"',
            script,
        )
        self.assertNotIn('"${dac_run_dir}/target_C.pt"', script)
        self.assertIn("Total target passes: 31", script)

    def test_uniform5_visda_handoff_has_no_special_final_cycle(self):
        script = Path(
            "tools/run_visda_dac_duet_handoff_uniform5_full.sh"
        ).read_text()
        self.assertIn("TEST.MAX_EPOCH 5 TEST.INTERVAL 5", script)
        self.assertIn("DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0", script)
        self.assertIn("Uniform DUET schedule: 5/5/5/5", script)
        self.assertIn("handoff_target_passes=20", script)

    def test_office_home_uniform5_handoff_covers_all_tasks(self):
        config = Path(
            "cfgs/office-home/duet_delayed_agreement_credit.yaml"
        ).read_text()
        script = Path(
            "tools/run_office_home_dac_duet_handoff_uniform5_all.sh"
        ).read_text()
        self.assertIn("DATASET: office-home", config)
        self.assertIn("EPOCHS: 15", config)
        self.assertIn("for s in 0 1 2 3", script)
        self.assertIn("for t in 0 1 2 3", script)
        self.assertIn("TEST.MAX_EPOCH 5 TEST.INTERVAL 5", script)
        self.assertIn("DUET_HANDOFF.FINAL_EXTRA_EPOCHS 0", script)
        self.assertIn("handoff_target_passes=20", script)


if __name__ == "__main__":
    unittest.main()
