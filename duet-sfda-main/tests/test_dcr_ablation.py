import inspect
import unittest
from pathlib import Path

import torch

from src.utils.dcr_refinement import (
    credit_preserving_refinement_step,
)
from src.utils.dcr_credit_memory import initialize_delayed_credit
from src.methods.oh import dcr


class DcrSfdaAblationTest(unittest.TestCase):
    def test_ablation_controls_reach_label_construction_signature(self):
        parameters = inspect.signature(dcr.obtain_label).parameters
        self.assertIn("credit_memory_write_mode", parameters)
        self.assertIn("credit_mode", parameters)
        self.assertIn("credit_feedback_mode", parameters)

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
            "tools/run_office_home_dcr_ablation.sh"
        ).read_text()
        self.assertIn("dcm_uniform)", script)
        self.assertIn('dcm_credit_mode="uniform"', script)
        self.assertIn("clm_writable)", script)
        self.assertIn('memory_write_mode="writable"', script)
        self.assertIn("arg_none)", script)
        self.assertIn('soft_replacement_mode="none"', script)
        self.assertIn("uniform_writable)", script)
        self.assertIn("uniform_writable_arg_none)", script)
        self.assertIn(
            'DCR.MEMORY_WRITE_MODE "$memory_write_mode"',
            script,
        )
        self.assertIn(
            'DCR.CREDIT_MODE "$dcm_credit_mode"',
            script,
        )
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)

    def test_core_screen_is_predefined_domain_ring(self):
        script = Path(
            "tools/run_office_home_dcr_core_ablation.sh"
        ).read_text()
        self.assertIn("tasks=(AC CP PR RA)", script)
        self.assertIn(
            "variants=(dcm_uniform clm_writable arg_none)",
            script,
        )

    def test_interaction_screen_tests_optimized_base_with_and_without_arg(self):
        script = Path(
            "tools/run_office_home_dcr_interaction_ablation.sh"
        ).read_text()
        self.assertIn("tasks=(AC CP PR RA)", script)
        self.assertIn(
            "variants=(uniform_writable uniform_writable_arg_none)",
            script,
        )
        self.assertIn("already complete; skipping", script)

    def test_full_ablation_runner_covers_all_tasks_and_resumes(self):
        script = Path(
            "tools/run_office_home_dcr_all_ablation.sh"
        ).read_text()
        self.assertIn("tasks=(AC AP AR CA CP CR PA PC PR RA RC RP)", script)
        self.assertIn("already complete; skipping", script)
        self.assertIn("summarize_office_home_dcr_ablation.py", script)


if __name__ == "__main__":
    unittest.main()
