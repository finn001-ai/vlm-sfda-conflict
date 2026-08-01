import unittest
from pathlib import Path

import yaml


class DuetBoundaryRouterContractTest(unittest.TestCase):
    def test_visda_candidate_differs_only_by_router(self):
        control = yaml.safe_load(Path("cfgs/visda/plmatch.yaml").read_text())
        candidate = yaml.safe_load(
            Path("cfgs/visda/duet_boundary_router.yaml").read_text()
        )
        boundary = candidate.pop("DUET_BOUNDARY")
        candidate["MODEL"]["METHOD"] = control["MODEL"]["METHOD"]

        self.assertEqual(boundary, {"TOP_FRACTION": 0.2})
        self.assertEqual(candidate, control)

    def test_control_keeps_released_arithmetic_fusion(self):
        code = Path("src/methods/oh/plmatch.py").read_text()
        control = yaml.safe_load(Path("cfgs/visda/plmatch.yaml").read_text())

        self.assertIn("all_mix_output = (all_output + clip_all_output) / 2.0", code)
        self.assertNotIn("FUSION", control["ACTIVE"])

    def test_proxy_runner_requires_matched_duet_contract_hash(self):
        control_runner = Path("tools/run_visda_plmatch_proxy25_control.sh").read_text()
        candidate_runner = Path(
            "tools/run_visda_duet_boundary_router_proxy25.sh"
        ).read_text()

        for path in (
            "conf.py",
            "cfgs/visda/plmatch.yaml",
            "src/methods/oh/plmatch.py",
            "src/utils/conflict_boundary.py",
        ):
            self.assertIn(path, control_runner)
            self.assertIn(path, candidate_runner)
        self.assertIn('cmp -s "$current_contract_hash" "$control_contract_hash"', candidate_runner)


if __name__ == "__main__":
    unittest.main()
