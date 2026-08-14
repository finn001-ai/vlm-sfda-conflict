import math
from types import SimpleNamespace

from src.utils.utils import adjust_learning_rate


class DummyOptimizer:
    def __init__(self, lr=1.0):
        self.param_groups = [{"lr": lr, "lr0": lr}]


def test_adacontrast_yacs_style_scheduler_config():
    optimizer = DummyOptimizer()
    cfg = SimpleNamespace(
        ADACONTRAST=SimpleNamespace(
            OPTIM_COS=True,
            OPTIM_EXP=False,
            FULL_PROGRESS=100,
            SCHEDULE=[10, 20],
            GAMMA=0.2,
        )
    )

    decay = adjust_learning_rate(optimizer, progress=50, args=cfg)

    assert math.isclose(decay, 0.5)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.5)


def test_adacontrast_legacy_scheduler_config_still_works():
    optimizer = DummyOptimizer()
    args = SimpleNamespace(
        optim_cos=False,
        optim_exp=False,
        full_progress=100,
        schedule=[10, 20],
        gamma=0.2,
    )

    decay = adjust_learning_rate(optimizer, progress=15, args=args)

    assert math.isclose(decay, 0.2)
    assert math.isclose(optimizer.param_groups[0]["lr"], 0.2)
