import unittest

import numpy as np

from tools.analyze_visda_stage14_gap import analyze


def probabilities(target, mistakes=None):
    classes = int(target.max()) + 1
    result = np.full((target.size, classes), 0.02, dtype=np.float32)
    result[np.arange(target.size), target] = 0.94
    for row, prediction in mistakes or []:
        result[row] = 0.02
        result[row, prediction] = 0.94
    return result


def temporal(target, mix):
    source = target.copy()
    clip = (target + 1) % (int(target.max()) + 1)
    return {
        "target_label": target,
        "mix_label": mix,
        "label_mask": np.ones(target.size, dtype=bool),
        "source_label": source,
        "clip_label": clip,
    }


class VisDAStage14GapTest(unittest.TestCase):
    def test_finds_distributed_confusion_and_local_geometry_compression(self):
        target = np.repeat(np.arange(4), 20)
        duet_feature = np.eye(4, dtype=np.float32)[target]
        stage14_feature = duet_feature.copy()
        stage14_feature[target == 0] = np.array([1.0, 0.0, 0.0, 0.0])
        stage14_feature[target == 1] = np.array([0.99, 0.01, 0.0, 0.0])

        mistakes = []
        mistakes.extend((row, 1) for row in range(0, 8))
        mistakes.extend((row, 0) for row in range(20, 26))
        mistakes.extend((row, 3) for row in range(40, 45))
        duet_prob = probabilities(target)
        stage14_prob = probabilities(target, mistakes)
        duet_final = {
            "target_label": target,
            "task_feature": duet_feature,
            "base_task_prob": duet_prob,
            "task_prob": duet_prob,
        }
        stage14_final = {
            "target_label": target,
            "task_feature": stage14_feature,
            "base_task_prob": duet_prob,
            "task_prob": stage14_prob,
        }
        duet_temporal = temporal(target, target.copy())
        stage14_mix = target.copy()
        stage14_mix[target == 1] = 0
        stage14_temporal = temporal(target, stage14_mix)

        summary, tables = analyze(
            duet_final,
            stage14_final,
            duet_temporal,
            stage14_temporal,
            ["zero", "one", "two", "three"],
        )

        checks = summary["mentor_checks"]
        self.assertEqual(
            checks["confusion_scope"]["result"],
            "distributed_beyond_one_fixed_pair",
        )
        self.assertIn(
            "zero<->one",
            checks["feature_space"]["compressed_top_excess_pairs"],
        )
        self.assertEqual(
            checks["classifier_head"]["result"],
            "effective_head_harms_macro_accuracy",
        )
        self.assertEqual(len(tables["per_class"]), 4)
        self.assertEqual(len(tables["pair_confusion_geometry"]), 6)

    def test_rejects_unmatched_samples(self):
        target = np.array([0, 1])
        probability = probabilities(target)
        snapshot = {
            "target_label": target,
            "task_feature": np.eye(2, dtype=np.float32),
            "base_task_prob": probability,
            "task_prob": probability,
        }
        flow = temporal(target, target)
        unmatched = dict(snapshot)
        unmatched["target_label"] = target[::-1]
        with self.assertRaises(ValueError):
            analyze(snapshot, unmatched, flow, flow, ["zero", "one"])


if __name__ == "__main__":
    unittest.main()
