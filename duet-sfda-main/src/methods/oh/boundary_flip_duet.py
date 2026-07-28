"""Boundary-Flip DUET entry point.

The training host is the validated temporal-precision DUET implementation.
Only the proposal generator and its explicitly weighted flip loss are new.
"""

from src.methods.oh import dccl


def validate_config(cfg) -> None:
    if not cfg.BOUNDARY_FLIP.ENABLED:
        raise ValueError("Boundary-Flip DUET requires BOUNDARY_FLIP.ENABLED=True")
    if not cfg.DCCL.TARGET_HEAD_ADAPT:
        raise ValueError("Boundary-Flip DUET requires the Stage14 target head")
    if cfg.DCCL.PL_MEMORY != "stable":
        raise ValueError("Boundary-Flip DUET requires stable pseudo-label memory")
    if cfg.ACCD.ENABLED or cfg.DCCL.RECIPROCAL_BOUNDARY:
        raise ValueError(
            "Boundary-Flip DUET cannot be combined with ACCD or reciprocal boundary"
        )
    if cfg.BOUNDARY_FLIP.LOSS_PAR <= 0:
        raise ValueError("BOUNDARY_FLIP.LOSS_PAR must be positive")


def train_target(cfg):
    validate_config(cfg)
    return dccl.train_target(cfg)
