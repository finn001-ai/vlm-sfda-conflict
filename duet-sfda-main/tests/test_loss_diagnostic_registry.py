import ast
import unittest
from pathlib import Path


class LossDiagnosticRegistryTest(unittest.TestCase):
    def test_every_literal_loss_record_has_a_registered_term(self):
        tree = ast.parse(Path("src/methods/oh/dccl.py").read_text())
        init_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "init_loss_diagnostics"
        )
        registered = set()
        for node in ast.walk(init_function):
            if (
                isinstance(node, ast.comprehension)
                and isinstance(node.iter, ast.Tuple)
            ):
                registered.update(
                    item.value
                    for item in node.iter.elts
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                )

        recorded = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "record_loss_diagnostic"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
            ):
                recorded.add(node.args[1].value)

        self.assertIn("boundary_flip", recorded)
        self.assertEqual(recorded - registered, set())


if __name__ == "__main__":
    unittest.main()
