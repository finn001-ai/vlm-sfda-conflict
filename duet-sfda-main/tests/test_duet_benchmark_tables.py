import unittest

from tools.build_duet_benchmark_tables import (
    DEFAULT_CLASSES,
    OFFICE_TASKS,
    build_report,
    render_markdown,
)


def visda_record(cycle, accuracy):
    return {
        "iteration": 100,
        "max_iteration": 100,
        "cycle": cycle,
        "max_cycle": 8,
        "accuracy": accuracy,
        "class_accuracy": [accuracy] * 12,
    }


def office_row(task, final_accuracy, peak_accuracy):
    return {
        "task": task,
        "final_accuracy": final_accuracy,
        "final_cycle": 4,
        "final_iter": 100,
        "oracle_peak_accuracy": peak_accuracy,
        "oracle_peak_cycle": 4,
        "oracle_peak_iter": 100,
        "max_cycle": 4,
        "log": f"{task}.txt",
    }


class DuetBenchmarkTablesTest(unittest.TestCase):
    def setUp(self):
        self.visda = [
            visda_record(cycle, 80.0 + cycle)
            for cycle in range(1, 9)
        ]
        self.office = [
            office_row(task, 80.0 + index, 80.5 + index)
            for index, task in enumerate(OFFICE_TASKS)
        ]

    def test_builds_complete_summary_and_detail_tables(self):
        report = build_report(self.visda, self.office)

        self.assertEqual(report["summary"][0]["final_metric"], 88.0)
        self.assertAlmostEqual(report["summary"][1]["final_metric"], 85.5)
        self.assertEqual(len(report["visda_cycles"]), 8)
        self.assertEqual(len(report["visda_classes"]), len(DEFAULT_CLASSES))
        self.assertEqual(len(report["office_home_tasks"]), 12)

    def test_markdown_contains_all_sections(self):
        markdown = render_markdown(build_report(self.visda, self.office))

        self.assertIn("VisDA-C: final checkpoint of each cycle", markdown)
        self.assertIn("Office-Home: pure DUET, 12 tasks", markdown)
        self.assertIn("| RP |", markdown)

    def test_rejects_incomplete_visda_run(self):
        with self.assertRaisesRegex(ValueError, "finish 8 cycles"):
            build_report(self.visda[:-1], self.office)


if __name__ == "__main__":
    unittest.main()
