"""Isolated DUET entry point for the attribute-reliability KL candidate."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Run DUET with only the first-cycle conflict KL target changed."""
    return plmatch.train_target(cfg, attribute_reliability_kl=True)
