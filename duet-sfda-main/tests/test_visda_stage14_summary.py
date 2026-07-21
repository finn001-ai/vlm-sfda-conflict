import unittest

from tools.summarize_visda_temporal_precision_head import (
    DEFAULT_CLASSES,
    parse_records,
    summarize,
)


class VisDAStage14SummaryTest(unittest.TestCase):
    def test_extracts_final_peak_and_per_class_accuracy(self):
        text = """
Task: TV, Iter:10/40; Cycle: 8/8; Accuracy = 80.00%; classifier_loss = 1
70 71 72 73 74 75 76 77 78 79 80 81
Task: TV, Iter:20/40; Cycle: 8/8; Accuracy = 81.00%; classifier_loss = 1
80 81 82 83 84 85 86 87 88 89 90 91
Task: TV, Iter:40/40; Cycle: 8/8; Accuracy = 80.50%; classifier_loss = 1
75 76 77 78 79 80 81 82 83 84 85 86
"""
        result = summarize(parse_records(text), DEFAULT_CLASSES)
        self.assertEqual(result["num_checkpoints"], 3)
        self.assertEqual(result["final"]["accuracy"], 80.5)
        self.assertEqual(result["oracle_peak"]["accuracy"], 81.0)
        self.assertEqual(result["classes"][0]["peak_checkpoint_accuracy"], 80.0)

    def test_rejects_missing_class_values(self):
        text = """
Task: TV, Iter:10/40; Cycle: 1/8; Accuracy = 80.00%; classifier_loss = 1
70 71
"""
        with self.assertRaises(ValueError):
            parse_records(text)


if __name__ == "__main__":
    unittest.main()
