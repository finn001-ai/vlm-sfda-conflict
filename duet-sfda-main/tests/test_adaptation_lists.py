import tempfile
import unittest
from pathlib import Path

from src.utils.adaptation_lists import load_adaptation_and_evaluation_rows


class AdaptationListsTest(unittest.TestCase):
    def test_plmatch_loader_keeps_proxy_indices_aligned(self):
        source = Path("src/methods/oh/plmatch.py").read_text()
        self.assertIn(
            'dsets["target"] = ImageList_idx(txt_tar',
            source,
        )
        self.assertIn(
            'dsets["test_aug"] = ImageList_idx(txt_tar',
            source,
        )
        self.assertIn(
            'dsets["test"] = ImageList_idx(txt_test',
            source,
        )

    def test_override_drives_adaptation_but_not_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            default_path = root / "default.txt"
            proxy_path = root / "proxy.txt"
            evaluation_path = root / "evaluation.txt"
            default_path.write_text("default_a 0\ndefault_b 1\n")
            proxy_path.write_text("proxy_b 1\n")
            evaluation_path.write_text("full_a 0\nfull_b 1\nfull_c 2\n")

            adaptation, evaluation, selected_path = (
                load_adaptation_and_evaluation_rows(
                    str(default_path),
                    str(evaluation_path),
                    str(proxy_path),
                )
            )

            self.assertEqual(adaptation, ["proxy_b 1\n"])
            self.assertEqual(
                evaluation,
                ["full_a 0\n", "full_b 1\n", "full_c 2\n"],
            )
            self.assertEqual(selected_path, proxy_path)

    def test_empty_override_preserves_original_target_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_path = root / "target.txt"
            evaluation_path = root / "evaluation.txt"
            target_path.write_text("target_a 0\n")
            evaluation_path.write_text("evaluation_a 0\n")

            adaptation, _evaluation, selected_path = (
                load_adaptation_and_evaluation_rows(
                    str(target_path),
                    str(evaluation_path),
                    "",
                )
            )

            self.assertEqual(adaptation, ["target_a 0\n"])
            self.assertEqual(selected_path, target_path)

    def test_missing_override_fails_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_path = root / "target.txt"
            evaluation_path = root / "evaluation.txt"
            target_path.write_text("target_a 0\n")
            evaluation_path.write_text("evaluation_a 0\n")

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Adaptation list override does not exist",
            ):
                load_adaptation_and_evaluation_rows(
                    str(target_path),
                    str(evaluation_path),
                    str(root / "missing.txt"),
                )


if __name__ == "__main__":
    unittest.main()
