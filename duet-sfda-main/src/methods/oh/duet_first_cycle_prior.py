"""DUET-FCP：发布版 DUET 加且仅加第一个 cycle 的 both-prior。"""

from src.methods.oh import plmatch


def train_target(cfg):
    """使用原始 DUET 主循环，并显式开启首轮 prior。"""
    return plmatch.train_target(cfg, first_cycle_prior=True)
