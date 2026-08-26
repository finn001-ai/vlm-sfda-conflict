import math
import unittest
from pathlib import Path

import torch

from src.utils.dcr_credit_memory import initialize_delayed_credit
from src.utils.dcr_refinement import credit_preserving_refinement_step


class DcrRefinementTest(unittest.TestCase):
    def test_clm_locks_conflict_memory(self):
        task = torch.tensor([[0.90, 0.10], [0.80, 0.20]])
        vlm = torch.tensor([[0.10, 0.90], [0.20, 0.80]])
        state = initialize_delayed_credit(task, vlm)
        state["memory"] = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        memory_before = state["memory"].clone()

        updated, payload = credit_preserving_refinement_step(
            state,
            task_probability=task,
            clip_probability=vlm,
            conflict_hard_fraction=1.0,
        )

        self.assertTrue(payload["conflict_mask"].all())
        self.assertTrue(torch.equal(updated["memory"], memory_before))

    def test_agreement_can_refresh_memory(self):
        task = torch.tensor([[0.90, 0.10]])
        vlm = torch.tensor([[0.80, 0.20]])
        state = initialize_delayed_credit(task, vlm)
        state["memory"] = torch.tensor([[0.55, 0.45]])
        memory_before = state["memory"].clone()

        updated, payload = credit_preserving_refinement_step(state, task, vlm)

        self.assertTrue(payload["agreement_mask"].all())
        self.assertFalse(torch.equal(updated["memory"], memory_before))
        self.assertTrue(torch.equal(payload["soft_target"], vlm))

    def test_fixed_coverage_count_is_exact(self):
        task = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]]
        )
        vlm = torch.flip(task, dims=(1,))
        state = initialize_delayed_credit(task, vlm)
        _, payload = credit_preserving_refinement_step(
            state, task, vlm, conflict_hard_fraction=0.8
        )
        self.assertEqual(
            int(payload["hard_selected"].sum().item()), math.ceil(5 * 0.8)
        )

    def test_arg_replaces_only_task_supported_direction(self):
        task = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.9, 0.1]])
        vlm = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]])
        state = initialize_delayed_credit(task, vlm)
        state["memory"] = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
        _, payload = credit_preserving_refinement_step(
            state,
            task,
            vlm,
            conflict_hard_fraction=0.0,
            soft_replacement_mode="task_supported",
        )
        self.assertTrue(
            torch.equal(payload["soft_replaced"], torch.tensor([True, False, True]))
        )
        self.assertEqual(int(payload["hard_selected"].sum().item()), 0)

    def test_stable_runners_share_the_dcr_contract(self):
        expected = {
            "tools/run_office_home_dcr.sh": "ACTIVE.CYCLE 4",
            "tools/run_visda_dcr.sh": "ACTIVE.CYCLE 8",
            "tools/run_domainnet126_dcr.sh": "ACTIVE.CYCLE 4",
        }
        for path, schedule in expected.items():
            script = Path(path).read_text()
            self.assertIn("--cfg ", script)
            self.assertIn("/dcr.yaml", script)
            self.assertIn("DCR.CREDIT_PRESERVING True", script)
            self.assertIn("DCR.CONFLICT_HARD_FRACTION 0.0", script)
            self.assertIn("DCR.SOFT_REPLACEMENT_MODE task_supported", script)
            self.assertIn("DCR.MEMORY_WRITE_MODE locked", script)
            self.assertIn(schedule, script)


if __name__ == "__main__":
    unittest.main()
