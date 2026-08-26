"""Isolated DUET entry point for the support-conditioned CLIP candidate."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Change only the first-cycle unresolved-conflict CLIP KL target."""
    return plmatch.train_target(cfg, support_conditioned_clip=True)
