import unittest
import csv
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.extract_final_accuracy import (
    TRAJECTORY_ACCURACY_PATTERN,
    main,
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

    def test_pair_flow_csv_columns_stay_aligned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            method_dir = Path(tmpdir) / "AC" / "pair_flow_method"
            method_dir.mkdir(parents=True)
            log_path = method_dir / "log.txt"
            log_path.write_text(
                """
TARGET_HEAD_VARIANT: pair_flow
PAIR_FLOW_RANK: 16
PAIR_FLOW_MIN_COUNT: 5
PAIR_FLOW_MIN_CYCLES: 2
PAIR_FLOW_MAX_GATE: 0.3
PAIR_FLOW_GATE_INIT: -2.0
PAIR_FEATURE_ADAPT: True
PAIR_FEATURE_START_CYCLE: 1
PAIR_FEATURE_LR_MULT: 1.0
PAIR_FEATURE_MAX_GATE: 0.05
PAIR_FEATURE_GATE_INIT: -2.0
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.10%; pair_flow_gate=0.041; pair_flow_active_rank=8
"""
            )
            stdout = io.StringIO()
            argv = [
                "extract_final_accuracy.py",
                "--glob",
                str(Path(tmpdir) / "*" / "*" / "*.txt"),
                "--selection",
                "peak",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                main()

        rows = list(csv.DictReader(io.StringIO(stdout.getvalue())))
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].get(None))
        self.assertEqual(rows[0]["target_head_variant"], "pair_flow")
        self.assertEqual(rows[0]["pair_flow_active_rank"], "8")
        self.assertEqual(rows[0]["pair_flow_gate_final"], "0.041")

    def test_pair_feature_metrics_are_extracted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            method_dir = Path(tmpdir) / "AC" / "pair_feature_method"
            method_dir.mkdir(parents=True)
            log_path = method_dir / "log.txt"
            log_path.write_text(
                """
TARGET_HEAD_VARIANT: blend
PAIR_FEATURE_ADAPT: True
PAIR_FEATURE_START_CYCLE: 1
PAIR_FEATURE_LR_MULT: 1.0
PAIR_FEATURE_MIN_ACTIVE_RANK: 8
PAIR_FEATURE_MAX_GATE: 0.05
PAIR_FEATURE_GATE_INIT: -2.0
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%; pair_feature_gate=0.0000; pair_feature_router_norm=0.0; pair_flow_active_rank=7; pair_feature_effective=False
"""
            )
            stdout = io.StringIO()
            argv = [
                "extract_final_accuracy.py",
                "--glob",
                str(Path(tmpdir) / "*" / "*" / "*.txt"),
                "--selection",
                "peak",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                main()

        rows = list(csv.DictReader(io.StringIO(stdout.getvalue())))
        self.assertEqual(rows[0]["pair_feature_adapt"], "True")
        self.assertEqual(rows[0]["pair_feature_min_active_rank"], "8")
        self.assertEqual(rows[0]["pair_feature_gate_final"], "0.0000")
        self.assertEqual(rows[0]["pair_feature_router_norm"], "0.0")
        self.assertEqual(rows[0]["pair_flow_active_rank"], "7")
        self.assertEqual(rows[0]["pair_feature_effective"], "False")

    def test_covariance_transport_metrics_are_extracted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            method_dir = Path(tmpdir) / "AC" / "covariance_method"
            method_dir.mkdir(parents=True)
            log_path = method_dir / "log.txt"
            log_path.write_text(
                """
TARGET_HEAD_VARIANT: blend
COV_TRANSPORT_ADAPT: True
COV_TRANSPORT_START_CYCLE: 1
COV_TRANSPORT_MIN_ANCHORS: 8
COV_TRANSPORT_RANK: 4
COV_TRANSPORT_MAX_GATE: 0.05
Task: AC, Iter:40/40; Cycle: 4/4; Accuracy = 74.20%; cov_transport_active_classes=42; cov_transport_coverage=0.71; cov_transport_mean_shift=0.04
"""
            )
            stdout = io.StringIO()
            argv = [
                "extract_final_accuracy.py",
                "--glob",
                str(Path(tmpdir) / "*" / "*" / "*.txt"),
                "--selection",
                "peak",
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(stdout):
                main()

        rows = list(csv.DictReader(io.StringIO(stdout.getvalue())))
        self.assertIsNone(rows[0].get(None))
        self.assertEqual(rows[0]["cov_transport_adapt"], "True")
        self.assertEqual(rows[0]["cov_transport_rank"], "4")
        self.assertEqual(rows[0]["cov_transport_active_classes"], "42")
        self.assertEqual(rows[0]["cov_transport_coverage"], "0.71")
        self.assertEqual(rows[0]["cov_transport_mean_shift"], "0.04")


if __name__ == "__main__":
    unittest.main()
