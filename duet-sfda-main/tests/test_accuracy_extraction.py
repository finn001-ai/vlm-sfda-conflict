import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AccuracyExtractionTest(unittest.TestCase):
    def test_extracts_final_peak_and_current_stage14_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            method_dir = Path(tmpdir) / "TV" / "boundary_flip_duet_test"
            method_dir.mkdir(parents=True)
            log_path = method_dir / "run.txt"
            log_path.write_text(
                """
DCCL:
  CALIB_POWER: 0.5
  PL_STABLE_CYCLES: 2
  PL_MEMORY_WARMUP_CYCLES: 1
  TARGET_HEAD_MIX: 0.3
  TARGET_HEAD_START_CYCLE: 1
  GTR_PAR: 0.05
BOUNDARY_FLIP:
  LOGIT_ALPHA: 0.15
  LOSS_PAR: 0.05
Task: TV, Iter:10/20; Cycle: 1/2; Accuracy = 70.20%; classifier_loss = 1.0
Task: TV, Iter:20/20; Cycle: 2/2; Accuracy = 69.80%; classifier_loss = 1.0
"""
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "tools/extract_final_accuracy.py",
                    "--glob",
                    str(log_path),
                    "--selection",
                    "peak",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = list(csv.DictReader(result.stdout.splitlines()))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "boundary_flip_duet_test")
        self.assertEqual(rows[0]["task"], "TV")
        self.assertEqual(rows[0]["accuracy"], "70.20")
        self.assertEqual(rows[0]["final_accuracy"], "69.80")
        self.assertEqual(rows[0]["peak_accuracy"], "70.20")
        self.assertEqual(rows[0]["target_head_mix"], "0.3")
        self.assertEqual(rows[0]["boundary_flip_logit_alpha"], "0.15")


if __name__ == "__main__":
    unittest.main()
