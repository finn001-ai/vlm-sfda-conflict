"""DUET-FCP with label-free swap-conflict hard-label selection.

Base is the original DUET plus the stage-14 first-cycle prior (DUET-FCP),
matching the run that produced the archived Top-k conflict probe data.  The
swap selection only adds hard pseudo labels for bidirectional_cross_support
(pure-swap) conflicts; the Top-k conflict probe itself stays an independent
oracle-diagnostic tool and is not enabled by this method.
"""

from src.methods.oh import plmatch


def train_target(cfg):
    """Run DUET-FCP with the swap-conflict selection rule enabled."""
    return plmatch.train_target(
        cfg,
        first_cycle_prior=True,
        swap_conflict_selection=True,
    )
