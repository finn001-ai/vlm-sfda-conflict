"""Isolated DUET entry point for unresolved conflict-memory conditioning."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Keep the locked support-conditioned CLIP target on unresolved memory."""
    return plmatch.train_target(cfg, support_conditioned_clip_memory=True)
