import tempfile
import unittest
from pathlib import Path

from tools.prepare_visda_proxy_subset import read_rows, select_indices


class PrepareVisdaProxySubsetTest(unittest.TestCase):
    def test_selects_same_fraction_per_class_deterministically(self):
        rows = [
            (f"image_{label}_{index}.jpg {label}", label)
            for label in range(3)
            for index in range(8)
        ]

        selected_a, counts_a = select_indices(rows, ratio=0.25, seed=2020)
        selected_b, counts_b = select_indices(rows, ratio=0.25, seed=2020)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(counts_a, counts_b)
        self.assertEqual(dict(counts_a), {0: 2, 1: 2, 2: 2})

    def test_keeps_at_least_one_sample_per_class(self):
        rows = [("a.jpg 0", 0), ("b.jpg 1", 1), ("c.jpg 1", 1)]
        _selected, counts = select_indices(rows, ratio=0.01, seed=7)
        self.assertEqual(dict(counts), {0: 1, 1: 1})

    def test_reads_paths_containing_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "list.txt"
            path.write_text("folder with spaces/image.jpg 3\n")
            self.assertEqual(
                read_rows(path), [("folder with spaces/image.jpg 3", 3)]
            )


if __name__ == "__main__":
    unittest.main()
