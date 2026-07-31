"""CT-DUET 的互补标签损失；不依赖数据集或模型实现。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def complementary_conflict_loss(
    strong_logits: torch.Tensor,
    task_labels: torch.Tensor,
    clip_labels: torch.Tensor,
    selected_mask: torch.Tensor,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """用冲突候选对的补集构造负标签监督。

    task/CLIP 冲突不能可靠地回答两个 top-1 中谁正确，但二者共同排除的
    ``C-2`` 个类别仍然提供稠密负监督。这里只训练尚未被 DUET monotonic
    memory 准入的冲突样本；已准入样本继续完全使用原始 DUET 正标签 CE。

    返回的统计量只用于日志，不参与反向传播。
    """
    if strong_logits.ndim != 2:
        raise ValueError("strong_logits must have shape [batch, classes]")
    num_classes = strong_logits.size(1)
    if num_classes <= 2:
        raise ValueError("CT-DUET requires more than two classes")

    batch_size = strong_logits.size(0)
    for name, value in (
        ("task_labels", task_labels),
        ("clip_labels", clip_labels),
        ("selected_mask", selected_mask),
    ):
        if value.ndim != 1 or value.size(0) != batch_size:
            raise ValueError(f"{name} must have shape [batch]")

    task_labels = task_labels.to(device=strong_logits.device, dtype=torch.long)
    clip_labels = clip_labels.to(device=strong_logits.device, dtype=torch.long)
    selected_mask = selected_mask.to(
        device=strong_logits.device, dtype=torch.bool
    )
    conflict_mask = (~selected_mask) & task_labels.ne(clip_labels)
    conflict_count = int(conflict_mask.sum().item())
    if conflict_count == 0:
        zero = strong_logits.sum() * 0.0
        return zero, {"count": 0, "mean_loss": 0.0, "outside_mass": 0.0}

    conflict_probs = F.softmax(strong_logits[conflict_mask], dim=1)
    negative_mask = torch.ones_like(conflict_probs, dtype=torch.bool)
    negative_mask.scatter_(
        1, task_labels[conflict_mask].unsqueeze(1), False
    )
    negative_mask.scatter_(
        1, clip_labels[conflict_mask].unsqueeze(1), False
    )

    # 与 FullMatch 的 negative learning 一致，类别维求和、batch 维平均。
    # 这样后期只剩少量困难冲突时，损失会随冲突占比自然衰减，不会让一个
    # 残余冲突支配整个 mini-batch。
    safe_probs = conflict_probs.clamp(max=1.0 - float(epsilon))
    per_class_loss = -torch.log1p(-safe_probs)
    per_sample_loss = per_class_loss.masked_fill(~negative_mask, 0.0).sum(dim=1)
    outside_mass = (
        conflict_probs.masked_fill(~negative_mask, 0.0).sum(dim=1).mean()
    )
    training_loss = per_sample_loss.sum() / float(batch_size)
    return training_loss, {
        "count": conflict_count,
        "mean_loss": float(per_sample_loss.detach().mean().item()),
        "outside_mass": float(outside_mass.detach().item()),
    }
