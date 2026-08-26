import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.utils.failure_audit import save_failure_audit_snapshot


class FailureAuditSnapshotTest(unittest.TestCase):
    def config(self, root):
        return SimpleNamespace(
            output_dir=str(root),
            FAILURE_AUDIT=SimpleNamespace(
                ENABLED=True,
                DIR="failure_audit",
                FEATURE_DTYPE="float16",
            ),
        )

    def test_writes_feature_as_configured_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_failure_audit_snapshot(
                self.config(directory),
                "snapshot.npz",
                target_label=np.array([0, 1]),
                task_feature=np.ones((2, 3), dtype=np.float32),
            )
            self.assertEqual(
                Path(path), Path(directory) / "failure_audit" / "snapshot.npz"
            )
            with np.load(path) as snapshot:
                self.assertEqual(snapshot["task_feature"].dtype, np.float16)

    def test_rejects_mismatched_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_failure_audit_snapshot(
                    self.config(directory),
                    "snapshot.npz",
                    target_label=np.array([0, 1]),
                    task_feature=np.ones((3, 3), dtype=np.float32),
                )


if __name__ == "__main__":
    unittest.main()
