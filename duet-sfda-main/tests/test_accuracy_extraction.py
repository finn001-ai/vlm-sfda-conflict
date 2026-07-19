import unittest

from tools.extract_final_accuracy import (
    TRAJECTORY_ACCURACY_PATTERN,
    select_final_and_peak,
    select_primary,
)


class AccuracyExtractionTest(unittest.TestCase):
    def test_keeps_final_as_primary_and_peak_as_diagnostic(self):
        log = """
Task: AC, Iter:10/40; Cycle: 1/4; Accuracy = 72.10%
Task: AC, Iter:20/40; Cycle: 2/4; Accuracy = 74.20%
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 73.00%
"""
        final, peak = select_final_and_peak(log)

        self.assertEqual(final[3:], ("4", "4", "73.00"))
        self.assertEqual(peak[3:], ("2", "4", "74.20"))

    def test_returns_empty_pair_without_task_accuracy(self):
        self.assertEqual(select_final_and_peak("no accuracy here"), (None, None))

    def test_peak_selection_populates_primary_accuracy(self):
        final = ("AC", "40", "40", "4", "4", "73.00")
        peak = ("AC", "20", "40", "2", "4", "74.20")

        self.assertEqual(select_primary(final, peak, "peak"), peak)
        self.assertEqual(select_primary(final, peak, "final"), final)

    def test_trajectory_records_are_extracted_separately(self):
        log = """
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 73.00%
Trajectory Ensemble Task: AC, Iter:30/40; Cycle: 4/4; Accuracy = 74.10%; Members=2
Trajectory Ensemble Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 73.90%; Members=3
"""
        final, peak = select_final_and_peak(log, TRAJECTORY_ACCURACY_PATTERN)
        standard_final, standard_peak = select_final_and_peak(log)

        self.assertEqual(final[5], "73.90")
        self.assertEqual(peak[5], "74.10")
        self.assertEqual(standard_final[5], "73.00")
        self.assertEqual(standard_peak[5], "73.00")


if __name__ == "__main__":
    unittest.main()
