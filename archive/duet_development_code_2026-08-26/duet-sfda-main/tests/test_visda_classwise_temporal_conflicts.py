import unittest

import numpy as np

from tools.analyze_visda_classwise_temporal_conflicts import analyze_cycles


def cycle(index, source, clip, teacher, target):
    return {
        "cycle": np.array(index),
        "source_label": np.array(source),
        "clip_label": np.array(clip),
        "teacher_label": np.array(teacher),
        "mix_label": np.array(teacher),
        "target_label": np.array(target),
        "label_mask": np.ones(len(target), dtype=bool),
    }


class VisDAClasswiseTemporalConflictsTest(unittest.TestCase):
    def test_detects_heterogeneous_predicted_class_routes(self):
        target = [0, 0, 1, 1, 2, 2]
        source = [1, 1, 0, 0, 1, 1]
        clip = [2, 2, 1, 1, 2, 2]
        teacher = [0, 0, 1, 1, 1, 1]
        cycles = [
            cycle(1, source, clip, teacher, target),
            cycle(2, source, clip, teacher, target),
        ]

        result = analyze_cycles(
            cycles,
            ["zero", "one", "two"],
            min_selected=1,
            min_gain_pp=1.0,
            min_win_rate=60.0,
        )

        self.assertEqual(result["decision"], "supports_class_conditional_routing")
        self.assertEqual(result["supported_predicted_classes"], ["zero"])
        self.assertIn("one", result["unsupported_predicted_classes"])

    def test_rejects_single_cycle(self):
        item = cycle(1, [0], [1], [0], [0])

        with self.assertRaises(ValueError):
            analyze_cycles([item], ["zero", "one"])


if __name__ == "__main__":
    unittest.main()
