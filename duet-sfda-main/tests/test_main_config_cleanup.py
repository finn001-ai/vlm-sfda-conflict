import unittest
from pathlib import Path

import yaml


STAGE14_DCCL_KEYS = {
    "CAND_PAR",
    "CALIB_MODE",
    "PL_MEMORY",
    "PL_STABLE_CYCLES",
    "PL_STABLE_MEMORY",
    "PL_MEMORY_WARMUP_CYCLES",
    "TARGET_HEAD_ADAPT",
    "TARGET_HEAD_MIX",
    "TARGET_HEAD_START_CYCLE",
    "PROMOTE_K",
    "TEMPORAL_DIAG",
    "GRAPH_TEACHER_FUSION",
    "GTF_APPLY_TO",
    "GTR_PAR",
}
BOUNDARY_DCCL_KEYS = STAGE14_DCCL_KEYS - {"CAND_PAR", "PROMOTE_K"} | {
    "LOSS_DIAG"
}
BOUNDARY_FLIP_KEYS = {
    "ENABLED",
    "START_CYCLE",
    "LOGIT_ALPHA",
    "MIN_ADJUSTED_CONFIDENCE",
    "MIN_MARGIN",
    "SEMANTIC_THRESHOLD",
    "STABLE_CYCLES",
    "MAX_SWITCHES",
    "MAX_PER_PAIR",
    "MIN_WEIGHT",
    "LOSS_PAR",
    "NEGATIVE_WEIGHT",
}


class MainConfigCleanupTest(unittest.TestCase):
    def load(self, relative_path):
        return yaml.safe_load(Path(relative_path).read_text())

    def test_office_home_main_configs_only_keep_active_dccl_keys(self):
        stage14 = self.load("cfgs/office-home/temporal_precision_head.yaml")
        boundary = self.load("cfgs/office-home/boundary_flip_duet.yaml")

        self.assertEqual(set(stage14["DCCL"]), STAGE14_DCCL_KEYS)
        self.assertEqual(set(boundary["DCCL"]), BOUNDARY_DCCL_KEYS)
        self.assertEqual(
            set(boundary["BOUNDARY_FLIP"]), BOUNDARY_FLIP_KEYS
        )
        self.assertTrue(boundary["BOUNDARY_FLIP"]["ENABLED"])
        self.assertNotIn("ACCD", boundary)

    def test_visda_main_configs_only_add_adaptation_list(self):
        stage14 = self.load("cfgs/visda/temporal_precision_head.yaml")
        boundary = self.load("cfgs/visda/boundary_flip_duet.yaml")

        self.assertEqual(
            set(stage14["DCCL"]), STAGE14_DCCL_KEYS | {"ADAPTATION_LIST"}
        )
        self.assertEqual(
            set(boundary["DCCL"]), BOUNDARY_DCCL_KEYS | {"ADAPTATION_LIST"}
        )
        self.assertEqual(
            set(boundary["BOUNDARY_FLIP"]), BOUNDARY_FLIP_KEYS
        )

    def test_boundary_flip_skips_legacy_dccl_conflict_path(self):
        training_host = Path("src/methods/oh/dccl.py").read_text()

        self.assertIn(
            "not cfg.ACCD.ENABLED and not cfg.BOUNDARY_FLIP.ENABLED",
            training_host,
        )
        self.assertIn("candidate_mass = None", training_host)
        self.assertIn(
            "candidate_mask = torch.zeros_like(label_mask)", training_host
        )


if __name__ == "__main__":
    unittest.main()
