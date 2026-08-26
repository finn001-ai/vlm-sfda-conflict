import unittest
from pathlib import Path

import torch

from src.utils.dac_credit_preserving_refinement import (
    credit_preserving_refinement_step,
)
from src.utils.duet_delayed_credit import initialize_delayed_credit


class DcrSfdaAblationTest(unittest.TestCase):
    def test_writable_conflict_ablation_removes_memory_lock(self):
        task = torch.tensor([[0.95, 0.05]])
        clip = torch.tensor([[0.05, 0.95]])
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor([[0.80, 0.20]])
        memory_before = state["memory"].clone()

        updated, payload = credit_preserving_refinement_step(
            state,
            task,
            clip,
            conflict_hard_fraction=0.0,
            soft_replacement_mode="task_supported",
            memory_write_mode="writable",
        )

        self.assertFalse(torch.equal(updated["memory"], memory_before))
        self.assertFalse(payload["memory_preserved"].any())

    def test_no_residual_ablation_keeps_clip_soft_target(self):
        task = torch.tensor([[0.95, 0.05]])
        clip = torch.tensor([[0.05, 0.95]])
        state = initialize_delayed_credit(task, clip)
        state["memory"] = torch.tensor([[0.80, 0.20]])

        _, payload = credit_preserving_refinement_step(
            state,
            task,
            clip,
            conflict_hard_fraction=0.0,
            soft_replacement_mode="none",
        )

        self.assertFalse(payload["soft_replaced"].any())
        self.assertTrue(torch.equal(payload["soft_target"], clip))

    def test_runner_changes_one_named_module(self):
        script = Path(
            "tools/run_office_home_dcr_sfda_ablation.sh"
        ).read_text()
        self.assertIn("dcm_uniform)", script)
        self.assertIn('dcm_credit_mode="uniform"', script)
        self.assertIn("clm_writable)", script)
        self.assertIn('memory_write_mode="writable"', script)
        self.assertIn("arg_none)", script)
        self.assertIn('soft_replacement_mode="none"', script)
        self.assertIn(
            'DUET_HANDOFF.MEMORY_WRITE_MODE "$memory_write_mode"',
            script,
        )
        self.assertIn(
            'DUET_HANDOFF.CREDIT_MODE "$dcm_credit_mode"',
            script,
        )
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)

    def test_core_screen_is_predefined_domain_ring(self):
        script = Path(
            "tools/run_office_home_dcr_sfda_core_ablation.sh"
        ).read_text()
        self.assertIn("tasks=(AC CP PR RA)", script)
        self.assertIn(
            "variants=(dcm_uniform clm_writable arg_none)",
            script,
        )

    def test_full_ablation_runner_covers_all_tasks_and_resumes(self):
        script = Path(
            "tools/run_office_home_dcr_sfda_full_ablation.sh"
        ).read_text()
        self.assertIn("tasks=(AC AP AR CA CP CR PA PC PR RA RC RP)", script)
        self.assertIn("already complete; skipping", script)
        self.assertIn("summarize_office_home_dcr_sfda_ablation.py", script)


if __name__ == "__main__":
    unittest.main()
