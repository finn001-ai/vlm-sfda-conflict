import unittest
from pathlib import Path
import re

import yaml


STAGE14_DCCL_KEYS = {
    "PL_STABLE_CYCLES",
    "PL_MEMORY_WARMUP_CYCLES",
    "TARGET_HEAD_MIX",
    "TARGET_HEAD_START_CYCLE",
    "GTR_PAR",
}
BOUNDARY_DCCL_KEYS = STAGE14_DCCL_KEYS | {
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

    def test_dccl_host_has_no_legacy_experimental_branches(self):
        training_host = Path("src/methods/oh/dccl.py").read_text()
        self.assertLess(len(training_host.splitlines()), 1500)
        for legacy_name in (
            "TARGET_HEAD_VARIANT",
            "ClassPairFlowAdapter",
            "SourceAnchoredResidualClassifier",
            "PAIR_FEATURE_ADAPT",
            "COV_TRANSPORT_ADAPT",
            "THREE_VIEW_EM",
            "cfg.ACCD",
            "RECIPROCAL_BOUNDARY",
        ):
            self.assertNotIn(legacy_name, training_host)

    def test_every_registered_dccl_option_is_used_by_mainline(self):
        config_source = Path("conf.py").read_text()
        training_host = Path("src/methods/oh/dccl.py").read_text()
        registered = set(
            re.findall(r"_C\.DCCL\.([A-Z0-9_]+)", config_source)
        )
        referenced = set(
            re.findall(r"cfg\.DCCL\.([A-Z0-9_]+)", training_host)
        )
        self.assertEqual(registered, referenced)


if __name__ == "__main__":
    unittest.main()
