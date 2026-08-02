"""Isolated DUET entry point for first-cycle CLIP-confidence delay."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Delay only the locked low-CLIP-confidence cycle-1 agreements."""
    return plmatch.train_target(cfg, clip_confidence_delay=True)
