import ast
import unittest
from pathlib import Path

import yaml


class DuetFCPContractTest(unittest.TestCase):
    def load(self, path):
        return yaml.safe_load(Path(path).read_text())

    def assert_only_first_cycle_prior_differs(self, dataset):
        control = self.load(f"cfgs/{dataset}/plmatch.yaml")
        candidate = self.load(f"cfgs/{dataset}/duet_first_cycle_prior.yaml")
        power = candidate.pop("DUET_FCP")
        candidate["MODEL"]["METHOD"] = control["MODEL"]["METHOD"]

        self.assertEqual(power, {"POWER": 0.5})
        self.assertEqual(candidate, control)

    def test_office_home_matches_duet_except_prior(self):
        self.assert_only_first_cycle_prior_differs("office-home")

    def test_visda_matches_duet_except_prior(self):
        self.assert_only_first_cycle_prior_differs("visda")

    def test_original_duet_defaults_to_prior_disabled(self):
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

        train_defaults = dict(
            zip(
                (arg.arg for arg in train_target.args.kwonlyargs),
                train_target.args.kw_defaults,
            )
        )
        positional = [arg.arg for arg in obtain_label.args.args]
        obtain_defaults = dict(
            zip(positional[-len(obtain_label.args.defaults) :], obtain_label.args.defaults)
        )
        train_default = train_defaults["first_cycle_prior"]
        obtain_default = obtain_defaults["first_cycle_prior"]
        self.assertIsInstance(train_default, ast.Constant)
        self.assertFalse(train_default.value)
        self.assertIsInstance(obtain_default, ast.Constant)
        self.assertFalse(obtain_default.value)

    def test_wrapper_only_enables_first_cycle_prior(self):
        wrapper = Path(
            "src/methods/oh/duet_first_cycle_prior.py"
        ).read_text()
        self.assertIn(
            "plmatch.train_target(cfg, first_cycle_prior=True)",
            wrapper,
        )
        for removed_component in (
            "stable",
            "target_head",
            "graph_teacher",
            "gtr",
        ):
            self.assertNotIn(removed_component, wrapper.lower())


if __name__ == "__main__":
    unittest.main()
