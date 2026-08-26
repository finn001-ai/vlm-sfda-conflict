import unittest
from pathlib import Path


class MethodDispatchTest(unittest.TestCase):
    def test_dcr_has_two_explicit_stages(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn("import src.methods.oh.dcr as DCR", entrypoint)
        self.assertIn(
            "import src.methods.oh.dcr_memory as DCR_MEMORY", entrypoint
        )
        self.assertIn('cfg.MODEL.METHOD.startswith("dcr_memory_")', entrypoint)
        self.assertIn('cfg.MODEL.METHOD.startswith("dcr_")', entrypoint)
        self.assertIn("DCR_MEMORY.train_target(cfg)", entrypoint)
        self.assertIn("DCR.train_target(cfg)", entrypoint)

    def test_plmatch_uses_one_clean_implementation(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        source = Path("src/methods/oh/plmatch.py").read_text()
        self.assertIn("import src.methods.oh.plmatch as PLMATCH", entrypoint)
        self.assertIn("PLMATCH.train_target(cfg)", entrypoint)
        self.assertNotIn("first_cycle_prior", source)
        self.assertNotIn("swap_conflict_selection", source)
        self.assertEqual(
            [path.name for path in Path("src/methods/oh").glob("plmatch*.py")],
            ["plmatch.py"],
        )

    def test_retired_duet_methods_are_absent_from_runtime_tree(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertEqual(list(Path("src/methods/oh").glob("duet_*.py")), [])
        self.assertNotIn("src.methods.oh.duet_", entrypoint)
        self.assertIn("Unknown adaptation method", entrypoint)


if __name__ == "__main__":
    unittest.main()
