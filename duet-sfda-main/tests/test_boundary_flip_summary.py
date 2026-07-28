import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.analyze_boundary_flip_duet import read_final_accuracy, read_mechanism


CSV_HEADER = "method,task,final_accuracy\n"


class BoundaryFlipSummaryTest(unittest.TestCase):
    def test_reads_latest_accuracy_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "accuracy.csv")
            path.write_text(
                CSV_HEADER
                + "candidate,AC,73.10\n"
                + "candidate,AC,73.40\n"
                + "candidate,PC,73.20\n"
            )

            result = read_final_accuracy(path)

        self.assertEqual(result, {"AC": 73.4, "PC": 73.2})

    def test_mechanism_requires_active_samples_on_every_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("AC", "PC", "RC"):
                np.savez_compressed(
                    root / f"{task}_cycle03.npz",
                    task=np.array(task),
                    boundary_flip_candidate_mask=np.array([True, True]),
                    boundary_flip_stable_mask=np.array([True, False]),
                    boundary_flip_active_mask=np.array([True, False]),
                )

            result = read_mechanism(str(root / "*.npz"))

        self.assertTrue(result["all_tasks_active"])
        self.assertEqual(result["tasks"]["AC"]["active"], 1)


if __name__ == "__main__":
    unittest.main()
