import math
import unittest
from pathlib import Path

import torch

from src.utils.dac_credit_preserving_refinement import (
    credit_preserving_refinement_step,
)
from src.utils.duet_delayed_credit import initialize_delayed_credit


class DacCreditPreservingRefinementTest(unittest.TestCase):
    def test_conflict_memory_is_not_rewritten(self):
        previous_task = torch.tensor([[0.90, 0.10], [0.80, 0.20]])
        previous_clip = torch.tensor([[0.10, 0.90], [0.20, 0.80]])
        state = initialize_delayed_credit(previous_task, previous_clip)
        state["memory"] = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        memory_before = state["memory"].clone()

        updated, payload = credit_preserving_refinement_step(
            state,
            task_probability=torch.tensor([[0.95, 0.05], [0.10, 0.90]]),
            clip_probability=torch.tensor([[0.05, 0.95], [0.90, 0.10]]),
            conflict_hard_fraction=1.0,
        )

        self.assertTrue(payload["conflict_mask"].all())
        self.assertTrue(torch.equal(updated["memory"], memory_before))
        self.assertTrue(
            torch.equal(payload["soft_target"], memory_before)
        )

    def test_agreement_can_refresh_memory(self):
        task = torch.tensor([[0.90, 0.10]])
        clip = torch.tensor([[0.80, 0.20]])
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor([[0.55, 0.45]])
        memory_before = state["memory"].clone()

        updated, payload = credit_preserving_refinement_step(
            state,
            task,
            clip,
        )

        self.assertTrue(payload["agreement_mask"].all())
        self.assertFalse(torch.equal(updated["memory"], memory_before))
        self.assertTrue(torch.equal(payload["soft_target"], clip))

    def test_hard_conflict_rank_has_fixed_coverage(self):
        task = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]]
        )
        clip = torch.flip(task, dims=(1,))
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor(
            [[0.99, 0.01], [0.90, 0.10], [0.70, 0.30], [0.55, 0.45], [0.51, 0.49]]
        )

        _, payload = credit_preserving_refinement_step(
            state,
            task,
            clip,
            conflict_hard_fraction=0.8,
        )

        expected = math.ceil(5 * 0.8)
        self.assertEqual(int(payload["hard_selected"].sum().item()), expected)
        self.assertTrue(payload["hard_selected"][:expected].all())
        self.assertFalse(bool(payload["hard_selected"][-1].item()))

    def test_task_supported_mode_replaces_only_task_direction(self):
        task = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]])
        clip = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]])
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor(
            [[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]]
        )

        _, payload = credit_preserving_refinement_step(
            state,
            task,
            clip,
            conflict_hard_fraction=0.0,
            soft_replacement_mode="task_supported",
        )

        self.assertTrue(
            torch.equal(
                payload["soft_replaced"],
                torch.tensor([True, False, True]),
            )
        )
        self.assertEqual(int(payload["hard_selected"].sum().item()), 0)
        self.assertTrue(
            torch.equal(payload["soft_target"][1], clip[1])
        )

    def test_cloud_entry_uses_dac_state_and_uniform_four_cycles(self):
        script = Path(
            "tools/run_visda_dac_credit_preserving_refinement_full.sh"
        ).read_text()
        self.assertIn("DUET_HANDOFF.CREDIT_PRESERVING True", script)
        self.assertIn('DUET_HANDOFF.STATE_PATH "$dac_state"', script)
        self.assertIn("DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.8", script)
        self.assertIn("DUET_HANDOFF.FREEZE_CLIP True", script)
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)
        self.assertIn("Total target passes: 31", script)

    def test_office_home_entry_reuses_one_task_dac_checkpoint(self):
        script = Path(
            "tools/run_office_home_dac_credit_preserving_refinement.sh"
        ).read_text()
        self.assertIn('task="${2:-AC}"', script)
        self.assertIn("AC) source_index=0; target_index=1", script)
        self.assertIn("duet_delayed_agreement_credit_office_home_full_seed", script)
        self.assertIn("DUET_HANDOFF.CREDIT_PRESERVING True", script)
        self.assertIn('DUET_HANDOFF.STATE_PATH "$dac_state"', script)
        self.assertIn("DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.8", script)
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)
        self.assertIn("Total target passes: 31", script)

    def test_office_home_residual_keeps_released_duet_curriculum(self):
        script = Path(
            "tools/run_office_home_dac_credit_residual_refinement.sh"
        ).read_text()
        self.assertIn("DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0", script)
        self.assertIn("DUET_HANDOFF.FREEZE_CLIP False", script)
        self.assertIn(
            "DUET_HANDOFF.SOFT_REPLACEMENT_MODE task_supported",
            script,
        )
        self.assertIn(
            "DUET_HANDOFF.CUMULATIVE_AGREEMENT_MASK True",
            script,
        )
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)
        self.assertIn("legacy_handoff_dir=", script)
        self.assertIn("preserved_uniform_handoff_copy", script)
        self.assertIn('cp -f "$dac_weight_f"', script)
        self.assertIn("DAC state missing; rebuilding full-data DAC", script)
        self.assertIn(
            "cfgs/office-home/duet_delayed_agreement_credit.yaml",
            script,
        )
        self.assertIn('MODEL.METHOD "$dac_method"', script)
        self.assertNotIn(
            'cp -f "${duet_run_dir}/target_F.pt"',
            script,
        )


if __name__ == "__main__":
    unittest.main()
