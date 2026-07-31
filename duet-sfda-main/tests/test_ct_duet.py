import ast
import unittest
from pathlib import Path

import torch
import yaml

from src.utils.complementary_learning import complementary_conflict_loss


class ComplementaryConflictLossTest(unittest.TestCase):
    def test_uses_only_unselected_conflicts_and_excludes_both_candidates(self):
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.5, -0.5],
                [0.0, 2.0, 1.0, -1.0],
                [0.0, 0.0, 2.0, -1.0],
            ],
            requires_grad=True,
        )
        task_labels = torch.tensor([0, 1, 2])
        clip_labels = torch.tensor([1, 2, 2])
        selected_mask = torch.tensor([False, True, False])

        value, stats = complementary_conflict_loss(
            logits, task_labels, clip_labels, selected_mask
        )
        probs = logits[0].softmax(dim=0)
        conflict_loss = -torch.log1p(-probs[2]) - torch.log1p(-probs[3])
        expected = conflict_loss / logits.size(0)

        self.assertTrue(torch.allclose(value, expected))
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["mean_loss"], float(conflict_loss.item()))
        self.assertAlmostEqual(
            stats["outside_mass"], float((probs[2] + probs[3]).item())
        )

        value.backward()
        self.assertGreater(float(logits.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[2].abs().sum()), 0.0)

    def test_zero_conflict_result_remains_differentiable(self):
        logits = torch.randn(3, 4, requires_grad=True)
        value, stats = complementary_conflict_loss(
            logits,
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 2, 2]),
            torch.tensor([False, True, False]),
        )

        self.assertEqual(float(value.item()), 0.0)
        self.assertEqual(
            stats, {"count": 0, "mean_loss": 0.0, "outside_mass": 0.0}
        )
        value.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_rejects_binary_classification(self):
        with self.assertRaisesRegex(ValueError, "more than two classes"):
            complementary_conflict_loss(
                torch.randn(2, 2),
                torch.tensor([0, 0]),
                torch.tensor([1, 1]),
                torch.tensor([False, False]),
            )


class CTDuetContractTest(unittest.TestCase):
    def test_config_matches_first_cycle_prior_except_method(self):
        baseline = yaml.safe_load(
            Path("cfgs/visda/duet_first_cycle_prior.yaml").read_text()
        )
        candidate = yaml.safe_load(Path("cfgs/visda/ct_duet.yaml").read_text())
        candidate["MODEL"]["METHOD"] = baseline["MODEL"]["METHOD"]
        self.assertEqual(candidate, baseline)

    def test_wrapper_enables_only_prior_and_complementary_transition(self):
        wrapper = Path("src/methods/oh/ct_duet.py").read_text()
        self.assertIn("first_cycle_prior=True", wrapper)
        self.assertIn("complementary_transition=True", wrapper)
        for removed_component in (
            "target_head",
            "graph_teacher",
            "gtr",
            "dino",
        ):
            self.assertNotIn(removed_component, wrapper.lower())

    def test_original_duet_defaults_to_complementary_transition_disabled(self):
        tree = ast.parse(Path("src/methods/oh/plmatch.py").read_text())
        train_target = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_target"
        )
        obtain_label = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "obtain_label"
        )

        self.assertFalse(train_target.args.kw_defaults[-1].value)
        self.assertFalse(obtain_label.args.defaults[-1].value)

    def test_dispatch_and_cloud_script_are_dedicated(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn('cfg.MODEL.METHOD.startswith("ct_duet_")', entrypoint)
        self.assertIn("import src.methods.oh.ct_duet as CT_DUET", entrypoint)

        script = Path("tools/run_visda_ct_duet.sh").read_text()
        self.assertIn("--cfg cfgs/visda/ct_duet.yaml", script)
        self.assertIn('method="ct_duet_visda_full_seed${seed}"', script)
        self.assertIn('rm -rf -- "$run_dir"', script)
        self.assertIn('ACTIVE.CYCLE 8', script)
        self.assertIn('grep -c "CT-DUET cycle summary"', script)


if __name__ == "__main__":
    unittest.main()
