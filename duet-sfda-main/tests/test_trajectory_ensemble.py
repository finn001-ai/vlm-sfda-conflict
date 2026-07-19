import unittest

import torch
import torch.nn as nn

from src.utils.trajectory_ensemble import (
    capture_trajectory_snapshot,
    load_trajectory_snapshot,
)


class TrajectoryEnsembleTest(unittest.TestCase):
    def test_snapshot_is_independent_and_restores_all_modules(self):
        netF = nn.Linear(2, 2)
        netB = nn.BatchNorm1d(2)
        target_head = nn.Linear(2, 2)
        expected_f = netF.weight.detach().clone()
        expected_b = netB.running_mean.detach().clone()
        expected_head = target_head.bias.detach().clone()

        snapshot = capture_trajectory_snapshot(
            netF, netB, target_head, checkpoint_index=2
        )
        netF.weight.data.add_(10.0)
        netB.running_mean.add_(5.0)
        target_head.bias.data.sub_(3.0)

        load_trajectory_snapshot(snapshot, netF, netB, target_head)

        self.assertEqual(snapshot["checkpoint_index"], 2)
        self.assertTrue(torch.equal(netF.weight, expected_f))
        self.assertTrue(torch.equal(netB.running_mean, expected_b))
        self.assertTrue(torch.equal(target_head.bias, expected_head))
        self.assertEqual(snapshot["netF"]["weight"].device.type, "cpu")

    def test_missing_target_head_state_is_rejected(self):
        netF = nn.Linear(2, 2)
        netB = nn.Linear(2, 2)
        target_head = nn.Linear(2, 2)
        snapshot = capture_trajectory_snapshot(netF, netB, None, 2)

        with self.assertRaises(ValueError):
            load_trajectory_snapshot(snapshot, netF, netB, target_head)

    def test_snapshot_supports_weight_norm_without_deepcopy(self):
        netF = nn.Linear(2, 2)
        netB = nn.Linear(2, 2)
        target_head = nn.utils.weight_norm(nn.Linear(2, 2))
        expected_weight_g = target_head.weight_g.detach().clone()
        snapshot = capture_trajectory_snapshot(netF, netB, target_head, 3)
        target_head.weight_g.data.add_(4.0)

        load_trajectory_snapshot(snapshot, netF, netB, target_head)

        self.assertTrue(torch.equal(target_head.weight_g, expected_weight_g))


if __name__ == "__main__":
    unittest.main()
