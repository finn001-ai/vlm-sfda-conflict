import unittest
from pathlib import Path


class MethodDispatchTest(unittest.TestCase):
    def test_temporal_precision_variants_use_common_prefix_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()

        self.assertIn(
            'cfg.MODEL.METHOD.startswith("temporal_precision_head_")',
            entrypoint,
        )
        self.assertIn("Unknown adaptation method", entrypoint)

    def test_visda_mix_scripts_use_dispatchable_method_names(self):
        for name in (
            "run_visda_temporal_precision_head_mix040_preflight.sh",
            "run_visda_temporal_precision_head_mix040_seed2020.sh",
        ):
            script = Path("tools", name).read_text()
            method_line = next(
                line for line in script.splitlines() if line.startswith('method="')
            )
            method = method_line.removeprefix('method="').removesuffix('"')
            self.assertTrue(method.startswith("temporal_precision_head_seed"))
            self.assertIn("produced no VisDA-C accuracy records", script)

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


if __name__ == "__main__":
    unittest.main()
