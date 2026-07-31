import unittest
from pathlib import Path


class MethodDispatchTest(unittest.TestCase):
    def test_duet_fcp_uses_dedicated_thin_wrapper(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()

        self.assertIn(
            'cfg.MODEL.METHOD.startswith("duet_first_cycle_prior_")',
            entrypoint,
        )
        self.assertIn(
            "import src.methods.oh.duet_first_cycle_prior as DUET_FCP",
            entrypoint,
        )
        self.assertNotIn("import src.methods.oh.dccl", entrypoint)
        self.assertNotIn('"reciprocal_boundary"', entrypoint)
        self.assertNotIn("import src.methods.oh.accd", entrypoint)
        self.assertIn("Unknown adaptation method", entrypoint)

    def test_plmatch_variants_use_original_plmatch_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            'cfg.MODEL.METHOD.startswith("plmatch_")',
            entrypoint,
        )
        script = Path("tools/run_visda_plmatch_proxy25_control.sh").read_text()
        self.assertIn('method="plmatch_visda_proxy25_seed2020"', script)
        self.assertIn("--cfg cfgs/visda/plmatch.yaml", script)
        self.assertIn('ACTIVE.ADAPTATION_LIST "$proxy_list"', script)
        self.assertIn('if [ "$checkpoint_count" -ne 16 ]', script)

    def test_boundary_flip_dispatch_is_removed(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertNotIn("boundary_flip", entrypoint.lower())


if __name__ == "__main__":
    unittest.main()
