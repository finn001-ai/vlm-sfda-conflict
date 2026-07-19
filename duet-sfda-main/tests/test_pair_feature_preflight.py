import unittest

from tools.summarize_pair_feature_preflight import summarize_rows


def make_row(active_rank="8", router_norm="0.2"):
    return {
        "task": "AC",
        "accuracy": "73.8",
        "selection": "peak",
        "target_head_variant": "blend",
        "pair_feature_adapt": "True",
        "pair_flow_active_rank": active_rank,
        "pair_feature_gate_final": "0.006",
        "pair_feature_router_norm": router_norm,
    }


class PairFeaturePreflightTest(unittest.TestCase):
    def test_passes_only_after_basis_and_router_activate(self):
        summary = summarize_rows([make_row()])

        self.assertEqual(summary["decision"], "pass_mechanism_preflight")

    def test_fails_inactive_basis(self):
        summary = summarize_rows([make_row(active_rank="0")])

        self.assertEqual(summary["decision"], "fail_mechanism_preflight")


if __name__ == "__main__":
    unittest.main()
