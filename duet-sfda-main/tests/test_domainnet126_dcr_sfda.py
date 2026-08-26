import tempfile
import unittest
from pathlib import Path

from src.utils.adaptation_lists import resolve_relative_image_rows


class DomainNet126DcrSfdaTest(unittest.TestCase):
    def test_relative_rows_are_rooted_without_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                "clipart/a/image1.jpg 3\n",
                "/absolute/image2.jpg 4\n",
            ]
            resolved = resolve_relative_image_rows(rows, directory)
            self.assertEqual(
                resolved[0],
                f"{Path(directory) / 'clipart/a/image1.jpg'} 3\n",
            )
            self.assertEqual(resolved[1], "/absolute/image2.jpg 4\n")

    def test_single_task_runner_uses_complete_dcr_contract(self):
        script = Path("tools/run_domainnet126_dcr_sfda.sh").read_text()
        self.assertIn('task="${2:-CP}"', script)
        self.assertIn("duet_delayed_agreement_credit_dcr_sfda_domainnet126", script)
        self.assertIn("DUET_HANDOFF.CREDIT_PRESERVING True", script)
        self.assertIn("DUET_HANDOFF.MEMORY_WRITE_MODE locked", script)
        self.assertIn("DUET_HANDOFF.SOFT_REPLACEMENT_MODE task_supported", script)
        self.assertIn("DUET_HANDOFF.CONFLICT_HARD_FRACTION 0.0", script)
        self.assertIn("TEST.MAX_EPOCH 4 TEST.INTERVAL 4", script)
        self.assertIn("ACTIVE.CYCLE 4", script)
        self.assertIn("passes=31", script)

    def test_all_task_runner_covers_twelve_directed_transfers(self):
        script = Path("tools/run_domainnet126_dcr_sfda_all.sh").read_text()
        self.assertIn("tasks=(CP CR CS PC PR PS RC RP RS SC SP SR)", script)
        self.assertIn("already complete; skipping", script)

    def test_domainnet_dcm_config_is_explicit(self):
        config = Path(
            "cfgs/domainnet126/duet_delayed_agreement_credit.yaml"
        ).read_text()
        self.assertIn("DATASET: domainnet126", config)
        self.assertIn("EPOCHS: 15", config)
        self.assertIn("ARCH: RN50", config)
        self.assertIn("CREDIT_MODE: delayed", config)
        self.assertIn("FEEDBACK_MODE: agreement_temporal", config)


if __name__ == "__main__":
    unittest.main()
