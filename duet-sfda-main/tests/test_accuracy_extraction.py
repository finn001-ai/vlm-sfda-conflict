import unittest

from tools.extract_final_accuracy import select_final_and_peak


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


if __name__ == "__main__":
    unittest.main()
