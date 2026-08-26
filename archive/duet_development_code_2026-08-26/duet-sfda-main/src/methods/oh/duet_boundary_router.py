"""DUET plus first-cycle gradient-normalized conflict soft routing."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Use the released DUET loop and enable only the boundary router."""
    return plmatch.train_target(cfg, boundary_router=True)
