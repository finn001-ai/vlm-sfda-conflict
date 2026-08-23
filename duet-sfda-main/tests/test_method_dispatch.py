import unittest
from pathlib import Path


class MethodDispatchTest(unittest.TestCase):
    def test_anchored_consensus_has_separate_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            "import src.methods.oh.duet_anchored_consensus as "
            "DUET_ANCHORED_CONSENSUS",
            entrypoint,
        )
        self.assertIn(
            'cfg.MODEL.METHOD.startswith("duet_anchored_consensus_")',
            entrypoint,
        )

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

    def test_boundary_router_uses_dedicated_thin_wrapper(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            'cfg.MODEL.METHOD.startswith("duet_boundary_router_")',
            entrypoint,
        )
        self.assertIn(
            "import src.methods.oh.duet_boundary_router as DUET_BOUNDARY",
            entrypoint,
        )
        wrapper = Path("src/methods/oh/duet_boundary_router.py").read_text()
        self.assertIn("plmatch.train_target(cfg, boundary_router=True)", wrapper)

    def test_support_conditioned_clip_uses_dedicated_thin_wrapper(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        self.assertIn(
            'cfg.MODEL.METHOD.startswith("duet_support_conditioned_clip_")',
            entrypoint,
        )
        self.assertIn(
            "import src.methods.oh.duet_support_conditioned_clip as DUET_SUPPORT_CLIP",
            entrypoint,
        )
        wrapper = Path(
            "src/methods/oh/duet_support_conditioned_clip.py"
        ).read_text()
        self.assertIn(
            "plmatch.train_target(cfg, support_conditioned_clip=True)", wrapper
        )

    def test_support_conditioned_clip_memory_has_separate_dispatch(self):
        entrypoint = Path("image_target_of_oh_vs.py").read_text()
        memory_branch = entrypoint.index(
            'cfg.MODEL.METHOD.startswith("duet_support_conditioned_clip_memory_")'
        )
        first_cycle_branch = entrypoint.index(
            'cfg.MODEL.METHOD.startswith("duet_support_conditioned_clip_")'
        )
        self.assertLess(memory_branch, first_cycle_branch)
        self.assertIn(
            "import src.methods.oh.duet_support_conditioned_clip_memory as "
            "DUET_SUPPORT_CLIP_MEMORY",
            entrypoint,
        )
        wrapper = Path(
            "src/methods/oh/duet_support_conditioned_clip_memory.py"
        ).read_text()
        self.assertIn(
            "plmatch.train_target(cfg, support_conditioned_clip_memory=True)",
            wrapper,
        )


if __name__ == "__main__":
    unittest.main()
