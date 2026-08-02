"""Isolated DUET entry point for cycle-2 compatibility-controlled PCGrad."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Change only the cycle-2 unresolved-conflict gradient combination."""
    return plmatch.train_target(cfg, pcgrad_compatibility=True)
