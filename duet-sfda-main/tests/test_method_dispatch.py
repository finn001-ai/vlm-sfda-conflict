import unittest
from pathlib import Path


class MethodDispatchTest(unittest.TestCase):
    def test_only_stage14_control_alias_uses_dccl_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()

        self.assertIn(
            'cfg.MODEL.METHOD.startswith("temporal_precision_head_control_")',
            entrypoint,
        )
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

    def test_boundary_flip_variants_use_dedicated_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            'cfg.MODEL.METHOD.startswith("boundary_flip_duet_")',
            entrypoint,
        )
        script = Path(
            "tools/run_office_home_boundary_flip_duet_preflight.sh"
        ).read_text()
        self.assertIn(
            "--cfg cfgs/office-home/boundary_flip_duet.yaml", script
        )
        self.assertIn("analyze_boundary_flip_duet.py", script)
        visda_script = Path(
            "tools/run_visda_boundary_flip_duet.sh"
        ).read_text()
        self.assertIn("--cfg cfgs/visda/boundary_flip_duet.yaml", visda_script)
        self.assertIn("analyze_visda_boundary_flip_duet.py", visda_script)
        self.assertIn('rm -rf -- "$candidate_dir"', visda_script)
        self.assertNotIn("Move its output directory aside", visda_script)


if __name__ == "__main__":
    unittest.main()
