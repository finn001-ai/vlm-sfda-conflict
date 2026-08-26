import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path("tools/summarize_office_home_dcr_sfda_ablation.py")
SPEC = importlib.util.spec_from_file_location("dcr_ablation_summary", SCRIPT_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


class DcrAblationSummaryTest(unittest.TestCase):
    def test_extended_log_uses_fixed_four_cycle_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.txt"
            lines = []
            for cycle in range(1, 8):
                for interval in range(1, 5):
                    accuracy = cycle * 10 + interval
                    lines.append(
                        f"Task: AC, Iter:{interval}/4; Cycle: {cycle}/7; "
                        f"Accuracy = {accuracy:.2f}%; classifier_loss = 0.1"
                    )
            path.write_text("\n".join(lines))

            peak, final, used, total = SUMMARY.parse_log(path, "AC")

            self.assertEqual(peak, 44.0)
            self.assertEqual(final, 44.0)
            self.assertEqual(used, 16)
            self.assertEqual(total, 28)

    def test_non_task_accuracy_diagnostics_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.txt"
            lines = ["Comparator Accuracy = 99.99%"]
            for cycle in range(1, 5):
                for interval in range(1, 5):
                    lines.append(
                        f"Task: CP, Iter:{interval}/4; Cycle: {cycle}/4; "
                        f"Accuracy = {70 + cycle:.2f}%; classifier_loss = 0.1"
                    )
            path.write_text("\n".join(lines))

            peak, final, used, total = SUMMARY.parse_log(path, "CP")

            self.assertEqual(peak, 74.0)
            self.assertEqual(final, 74.0)
            self.assertEqual(used, 16)
            self.assertEqual(total, 16)


if __name__ == "__main__":
    unittest.main()
