"""DCCL 伪标签记忆的监督工具。

这里的 ``weight`` 只控制样本对伪标签分类交叉熵（CE）的贡献。权重为 0
不表示样本不经过网络；它仍可参加一致性、KL 等其他损失，只是不会通过这项
hard-label CE 产生梯度。
"""

import torch
import torch.nn.functional as F


def dual_tier_supervision(
    current_mask,
    stable_mask,
    mix_conf,
    pending_scale,
    *,
    warmup,
):
    """构造 Stable/Pending/Conflict 三态掩码及逐样本 CE 权重。

    - Stable：当前 source/CLIP 一致且连续达到稳定轮数，CE 权重为 1。
    - Pending：当前一致但尚未稳定，CE 权重为
      ``pending_scale * mix_conf``。
    - Conflict：当前 source/CLIP 不一致，不在 ``selected_mask`` 中，
      因此该项 hard CE 权重保持为 0。

    注意：当前实现只“弱化 Pending、排除 Conflict”，尚未利用历史证据判断
    Conflict 中究竟是 source 还是 CLIP 更可靠。
    """
    if not 0.0 <= float(pending_scale) <= 1.0:
        raise ValueError("pending_scale must be in [0, 1]")
    if current_mask.shape != stable_mask.shape:
        raise ValueError("current_mask and stable_mask must have equal shapes")
    if current_mask.shape != mix_conf.shape:
        raise ValueError("mix_conf must have one value per pseudo label")
    if (stable_mask & ~current_mask).any():
        raise ValueError("stable samples must also be currently consistent")

    # Pending 仍是“当前一致”样本，只是同一标签尚未连续出现足够 cycles。
    pending_mask = current_mask & ~stable_mask
    selected_mask = stable_mask | pending_mask
    weights = torch.zeros_like(mix_conf)
    if warmup:
        # Cycle 1 没有足够历史，所有当前一致样本都按完整权重训练。
        weights[current_mask] = 1.0
    else:
        weights[stable_mask] = 1.0
        # 例如 pending_scale=0.5、mix_conf=0.8，则该样本权重为 0.4。
        # 这会按比例缩小它在加权 CE 分子中的损失与梯度贡献。
        weights[pending_mask] = float(pending_scale) * mix_conf[pending_mask]
    return selected_mask, pending_mask, weights


def weighted_cross_entropy(logits, labels, weights, epsilon=1e-6):
    """计算按有效监督权重归一化的 hard-label CE。

    单样本损失先乘 ``weights``，再除以当前 batch 的权重和。因此低权重样本
    仍参与训练，但相对于权重 1 的 Stable 样本，对最终梯度的影响更小。
    权重为 0 的 Conflict 在本项 CE 中损失和梯度贡献均为 0。
    """
    if logits.size(0) != labels.numel() or labels.numel() != weights.numel():
        raise ValueError("logits, labels, and weights must share batch size")
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    return (per_sample * weights).sum() / weights.sum().clamp_min(epsilon)
