"""Boundary-Flip DUET 的独立入口。

中文阅读说明
------------
这个文件只负责两件事：

1. 检查 Boundary-Flip 已显式开启；
2. 把训练交给 :mod:`src.methods.oh.dccl` 中已经验证过的 DUET/Stage14 主循环。

真正的新方法位于 ``src/utils/boundary_flip.py``，并在 ``dccl.train_target``
的伪标签刷新阶段生成候选、在 minibatch 阶段加入方向性 flip loss。

旧的 ACCD、reciprocal-boundary、residual/pair-flow target head 已从当前
主分支删除；完整历史可从标签
``archive/dccl-full-pre-prune-20260728`` 查阅。
"""

from src.methods.oh import dccl


def validate_config(cfg) -> None:
    """在加载模型前检查 Boundary-Flip 的最小运行契约。"""
    if not cfg.BOUNDARY_FLIP.ENABLED:
        raise ValueError("Boundary-Flip DUET requires BOUNDARY_FLIP.ENABLED=True")
    if cfg.BOUNDARY_FLIP.LOSS_PAR <= 0:
        raise ValueError("BOUNDARY_FLIP.LOSS_PAR must be positive")


def train_target(cfg):
    # 保持入口很薄：方法差异必须在显式配置和 dccl 主循环中可追踪，
    # 不在 wrapper 内静默改写超参数。
    validate_config(cfg)
    return dccl.train_target(cfg)
