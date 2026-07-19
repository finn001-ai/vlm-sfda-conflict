"""State helpers for fixed training-trajectory ensembles."""

from __future__ import annotations


def clone_module_state_to_cpu(module):
    if module is None:
        return None
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def capture_trajectory_snapshot(netF, netB, target_head, checkpoint_index):
    return {
        "checkpoint_index": int(checkpoint_index),
        "netF": clone_module_state_to_cpu(netF),
        "netB": clone_module_state_to_cpu(netB),
        "target_head": clone_module_state_to_cpu(target_head),
    }


def load_trajectory_snapshot(snapshot, netF, netB, target_head):
    netF.load_state_dict(snapshot["netF"])
    netB.load_state_dict(snapshot["netB"])
    if target_head is not None:
        if snapshot["target_head"] is None:
            raise ValueError("Trajectory snapshot is missing target-head state")
        target_head.load_state_dict(snapshot["target_head"])
