"""Clean DUET first-cycle prior with an oracle-only top-k coverage probe."""

from src.methods.oh import plmatch


def train_target(cfg):
    """Run unchanged DUET-FCP while exporting detached per-cycle probe data."""
    return plmatch.train_target(cfg, first_cycle_prior=True, topk_conflict_probe=True)
