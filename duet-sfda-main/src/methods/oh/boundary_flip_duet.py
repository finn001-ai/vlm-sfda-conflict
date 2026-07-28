"""Boundary-Flip DUET 的独立入口。

中文阅读说明
------------
这个文件只负责两件事：

1. 检查当前配置确实是“Stage14 宿主 + Boundary-Flip”，避免与 ACCD、
   reciprocal-boundary 等历史实验同时开启；
2. 把训练交给 :mod:`src.methods.oh.dccl` 中已经验证过的 DUET/Stage14 主循环。

真正的新方法位于 ``src/utils/boundary_flip.py``，并在 ``dccl.train_target``
的伪标签刷新阶段生成候选、在 minibatch 阶段加入方向性 flip loss。
"""

from src.methods.oh import dccl


def validate_config(cfg) -> None:
    """在加载模型前阻止互相污染的实验组合。"""
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
    # 保持入口很薄：方法差异必须在显式配置和 dccl 主循环中可追踪，
    # 不在 wrapper 内静默改写超参数。
    validate_config(cfg)
    return dccl.train_target(cfg)
