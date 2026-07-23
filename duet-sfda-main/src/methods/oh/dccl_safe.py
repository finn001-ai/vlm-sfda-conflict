"""
Stage14 的安全精简实现（Temporal Precision Target Head）。

这份文件只保留 Stage14 真正生效的主干，目的是方便逐行阅读、审计和继续改进。
它不是 dccl.py 的全功能替代品，也不包含 Stage15 之后的实验分支。

保留的 Stage14 组件：
1. 源模型与 CLIP 的类别先验校准（both_prior）；
2. 两个 cycle 的可逆稳定伪标签记忆；
3. 冻结源分类头 + 可学习目标分类头的 0.7/0.3 logits 融合；
4. 弱/强增强一致性、稳定伪标签 CE、CLIP KL；
5. Stage10 继承下来的小权重图时序残差（GTR）；
6. CLIP visual encoder 的目标域自适应。

明确删除的分支：
prototype、pseudo-label expansion/class balance、candidate loss、promotion、
ACCD、target-head EMA/residual/pair-flow、trajectory ensemble、pair feature、
covariance transport、three-view EM、topology-prior 自动选择等。

推荐配置：cfgs/office-home/temporal_precision_head.yaml
原始完整实现：src/methods/oh/dccl.py
"""

from __future__ import annotations

import logging
import os
import os.path as osp
from dataclasses import dataclass

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datautils_domain import build_dataset
from src.data.data_list import GaussianBlur, ImageList_idx, NCropsTransform
from src.models import network
from src.utils import IID_losses, loss
from src.utils.conflict_diffusion import (
    adaptive_graph_teacher_fusion,
    dual_space_diffusion,
    graph_temporal_residual_weights,
    update_temporal_resolution,
)
from src.utils.consistency import prediction_consistency_kl
from src.utils.utils import image_test


LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Stage14 的固定语义
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage14Spec:
    """把 Stage14 的关键选择集中写在一处，避免散落在大量配置 if 中。"""

    calibration_mode: str = "both_prior"
    pseudo_label_memory: str = "stable"
    stable_cycles: int = 2
    stable_memory: str = "reversible"
    warmup_cycles: int = 1
    target_head_variant: str = "blend"
    target_head_mix: float = 0.3
    target_head_start_cycle: int = 1
    graph_teacher_apply_to: str = "none"


STAGE14 = Stage14Spec()


def validate_stage14_config(cfg) -> None:
    """
    启动前一次性检查配置，防止这份精简文件被误用于其他实验变体。

    这里的检查只负责“快速失败”，训练主循环不再为这些变体保留分支。
    """

    expected = {
        "DCCL.CALIB_MODE": (cfg.DCCL.CALIB_MODE, STAGE14.calibration_mode),
        "DCCL.PL_MEMORY": (cfg.DCCL.PL_MEMORY, STAGE14.pseudo_label_memory),
        "DCCL.PL_STABLE_CYCLES": (
            int(cfg.DCCL.PL_STABLE_CYCLES),
            STAGE14.stable_cycles,
        ),
        "DCCL.PL_STABLE_MEMORY": (
            cfg.DCCL.PL_STABLE_MEMORY,
            STAGE14.stable_memory,
        ),
        "DCCL.PL_MEMORY_WARMUP_CYCLES": (
            int(cfg.DCCL.PL_MEMORY_WARMUP_CYCLES),
            STAGE14.warmup_cycles,
        ),
        "DCCL.TARGET_HEAD_ADAPT": (bool(cfg.DCCL.TARGET_HEAD_ADAPT), True),
        "DCCL.TARGET_HEAD_VARIANT": (
            cfg.DCCL.TARGET_HEAD_VARIANT,
            STAGE14.target_head_variant,
        ),
        "DCCL.TARGET_HEAD_MIX": (
            float(cfg.DCCL.TARGET_HEAD_MIX),
            STAGE14.target_head_mix,
        ),
        "DCCL.TARGET_HEAD_START_CYCLE": (
            int(cfg.DCCL.TARGET_HEAD_START_CYCLE),
            STAGE14.target_head_start_cycle,
        ),
        "DCCL.GRAPH_TEACHER_FUSION": (
            bool(cfg.DCCL.GRAPH_TEACHER_FUSION),
            True,
        ),
        "DCCL.GTF_APPLY_TO": (
            cfg.DCCL.GTF_APPLY_TO,
            STAGE14.graph_teacher_apply_to,
        ),
        "DCCL.GTR_STABLE_CYCLES": (
            int(cfg.DCCL.GTR_STABLE_CYCLES),
            STAGE14.stable_cycles,
        ),
        "DCCL.GTR_MEMORY": (cfg.DCCL.GTR_MEMORY, STAGE14.stable_memory),
        "DCCL.TEMPORAL_DIAG": (bool(cfg.DCCL.TEMPORAL_DIAG), True),
        "DCCL.PROTO_ADAPT": (bool(cfg.DCCL.PROTO_ADAPT), False),
        "DCCL.PL_CLASS_BALANCE": (bool(cfg.DCCL.PL_CLASS_BALANCE), False),
        "DCCL.PL_EXPAND": (cfg.DCCL.PL_EXPAND, "none"),
        "DCCL.CAND_PAR": (float(cfg.DCCL.CAND_PAR), 0.0),
        "DCCL.KL_MODE": (cfg.DCCL.KL_MODE, "clip"),
        "DCCL.TARGET_HEAD_EMA": (bool(cfg.DCCL.TARGET_HEAD_EMA), False),
        "DCCL.TRAJECTORY_ENSEMBLE": (
            bool(cfg.DCCL.TRAJECTORY_ENSEMBLE),
            False,
        ),
        "DCCL.PAIR_FEATURE_ADAPT": (bool(cfg.DCCL.PAIR_FEATURE_ADAPT), False),
        "DCCL.COV_TRANSPORT_ADAPT": (
            bool(cfg.DCCL.COV_TRANSPORT_ADAPT),
            False,
        ),
        "DCCL.THREE_VIEW_EM": (bool(cfg.DCCL.THREE_VIEW_EM), False),
        "ACCD.ENABLED": (bool(cfg.ACCD.ENABLED), False),
    }
    mismatches = [
        f"{name}: 当前值={actual!r}, Stage14要求={required!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if mismatches:
        raise ValueError("dccl_safe.py 只支持 Stage14 配置：\n" + "\n".join(mismatches))
    if not cfg.MODEL.ARCH.startswith("res"):
        raise ValueError("Stage14 safe 当前只保留 ResNet 主干。")
    if float(cfg.DCCL.GTR_PAR) <= 0:
        raise ValueError("Stage14 需要启用小权重图时序残差 DCCL.GTR_PAR。")


# -----------------------------------------------------------------------------
# 2. 数据增强与数据加载
# -----------------------------------------------------------------------------


def build_target_transform() -> NCropsTransform:
    """
    每张训练图像产生三个视图：测试视图、弱增强、强增强。

    索引含义固定为：views[0]=确定性测试视图，views[1]=弱增强，views[2]=强增强。
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    test_view = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    weak_view = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    strong_view = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return NCropsTransform([test_view, weak_view, strong_view])


def data_load(cfg) -> dict[str, DataLoader]:
    """构造目标域训练、评估和整域伪标签推理所需的三个 DataLoader。"""

    adaptation_path = str(cfg.DCCL.ADAPTATION_LIST).strip()
    target_list_path = adaptation_path if adaptation_path else cfg.t_dset_path
    if adaptation_path and not osp.isfile(target_list_path):
        raise FileNotFoundError(
            f"DCCL.ADAPTATION_LIST does not exist: {target_list_path}"
        )
    with open(target_list_path) as handle:
        target_rows = handle.readlines()
    with open(cfg.test_dset_path) as handle:
        test_rows = handle.readlines()
    if adaptation_path:
        LOGGER.info(
            "Stage14代理列表: %s, 适配样本=%d, 完整评估样本=%d",
            target_list_path,
            len(target_rows),
            len(test_rows),
        )

    batch_size = cfg.TEST.BATCH_SIZE
    target_transform = build_target_transform()

    target_set = ImageList_idx(target_rows, transform=target_transform)
    test_set = ImageList_idx(test_rows, transform=image_test())
    # 伪标签索引必须与 target 子集一致；test 仍使用完整目标域列表。
    test_aug_set = ImageList_idx(target_rows, transform=target_transform)

    return {
        "target": DataLoader(
            target_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            drop_last=False,
        ),
        "test": DataLoader(
            test_set,
            batch_size=batch_size * 3,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            drop_last=False,
        ),
        "test_aug": DataLoader(
            test_aug_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            drop_last=False,
        ),
    }


# -----------------------------------------------------------------------------
# 3. 源分类头与目标分类头
# -----------------------------------------------------------------------------


def build_target_head(cfg, source_head: nn.Module) -> nn.Module:
    """创建与源分类头同结构的目标分类头，并从源权重初始化。"""

    target_head = network.feat_classifier(
        type=source_head.type,
        class_num=cfg.class_num,
        bottleneck_dim=cfg.bottleneck,
    ).cuda()
    target_head.load_state_dict(source_head.state_dict())
    return target_head


def blend_classifier_logits(
    cfg,
    features: torch.Tensor,
    source_logits: torch.Tensor,
    target_head: nn.Module,
    curr_cycle: int,
) -> torch.Tensor:
    """
    Stage14 的核心决策边界：先由源头锚定，再加入 30% 的目标域修正。

    cycle 0 只用源分类头；从 cycle 1（人类表述的第二轮）开始：
        logits = 0.7 * source_logits + 0.3 * target_logits
    """

    if curr_cycle < STAGE14.target_head_start_cycle:
        return source_logits
    target_logits = target_head(features)
    mix = float(cfg.DCCL.TARGET_HEAD_MIX)
    return (1.0 - mix) * source_logits + mix * target_logits


# -----------------------------------------------------------------------------
# 4. both_prior 校准与稳定伪标签记忆
# -----------------------------------------------------------------------------


def prior_calibrate(prob: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """
    按整域预测先验抑制高频类别，然后重新归一化。

    calibrated(c|x) ∝ prob(c|x) / mean_x(prob(c|x)) ** power
    """

    prior = prob.mean(dim=0).clamp_min(eps)
    calibrated = prob / prior.pow(power)
    return calibrated / calibrated.sum(dim=1, keepdim=True).clamp_min(eps)


def both_prior_calibration(
    cfg,
    source_prob: torch.Tensor,
    clip_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """分别校准任务模型和 CLIP，再以 1:1 得到混合软分布。"""

    source_cal = prior_calibrate(
        source_prob,
        float(cfg.DCCL.CALIB_POWER),
        float(cfg.DCCL.EPSILON),
    )
    clip_cal = prior_calibrate(
        clip_prob,
        float(cfg.DCCL.CALIB_POWER),
        float(cfg.DCCL.EPSILON),
    )
    mix_prob = (source_cal + clip_cal) / 2.0
    return source_cal, clip_cal, mix_prob


def init_pseudo_label_state(num_samples: int) -> dict[str, torch.Tensor]:
    """为每个目标样本保存当前候选类别、连续出现次数和稳定类别。"""

    return {
        "pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "pending_count": torch.zeros(num_samples, dtype=torch.long),
        "stable_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


def update_stable_pseudo_labels(
    cfg,
    agreement_mask: torch.Tensor,
    current_label: torch.Tensor,
    mix_confidence: torch.Tensor,
    state: dict[str, torch.Tensor] | None,
    curr_cycle: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    实现 Stage11/14 的两轮可逆记忆。

    - cycle 0 是 warmup：当前 source/CLIP 一致即可训练；
    - warmup 后：同一个一致标签连续出现两轮才进入 CE；
    - 一旦不再一致或标签改变，稳定标签立即撤销，因此错误不会永久累积。
    """

    confidence_mask = mix_confidence >= float(cfg.DCCL.PL_MEMORY_MIN_CONF)
    current_mask = agreement_mask & confidence_mask
    if state is None:
        state = init_pseudo_label_state(current_label.numel())

    same_as_pending = state["pending_label"] == current_label
    state["pending_count"] = torch.where(
        current_mask & same_as_pending,
        state["pending_count"] + 1,
        torch.where(
            current_mask,
            torch.ones_like(state["pending_count"]),
            torch.zeros_like(state["pending_count"]),
        ),
    )
    state["pending_label"] = torch.where(
        current_mask,
        current_label,
        torch.full_like(state["pending_label"], -1),
    )

    became_stable = current_mask & (
        state["pending_count"] >= STAGE14.stable_cycles
    )
    state["stable_label"] = torch.where(
        became_stable,
        current_label,
        torch.full_like(state["stable_label"], -1),
    )

    warmup = curr_cycle < STAGE14.warmup_cycles
    if warmup:
        selected_mask = current_mask
        memory_label = current_label
    else:
        selected_mask = state["stable_label"] >= 0
        memory_label = torch.where(
            selected_mask,
            state["stable_label"],
            current_label,
        )

    LOGGER.info(
        "Stage14伪标签记忆: cycle=%d, warmup=%s, 当前一致=%d, 稳定=%d, 用于CE=%d",
        curr_cycle + 1,
        warmup,
        int(current_mask.sum().item()),
        int((state["stable_label"] >= 0).sum().item()),
        int(selected_mask.sum().item()),
    )
    return selected_mask, memory_label, state


# -----------------------------------------------------------------------------
# 5. 整个目标域上的伪标签生成
# -----------------------------------------------------------------------------


@torch.no_grad()
def obtain_label(
    cfg,
    loader: DataLoader,
    netF: nn.Module,
    netB: nn.Module,
    source_head: nn.Module,
    target_head: nn.Module,
    text_inputs: torch.Tensor,
    clip_model: nn.Module,
    pseudo_label_state: dict[str, torch.Tensor] | None,
    curr_cycle: int,
):
    """
    每个 cycle 开始时扫描整个目标域，生成本轮固定使用的监督信息。

    真值 target_label 只用于日志/诊断，不参与伪标签选择和损失计算。
    """

    text_features = F.normalize(clip_model.encode_text(text_inputs), dim=1)
    clip_logit_scale = clip_model.logit_scale.exp()

    model_logits_all = []
    clip_logits_all = []
    task_features_all = []
    clip_features_all = []
    target_labels_all = []

    for views, labels, _sample_indices in loader:
        weak_x = views[1].cuda()

        # 任务分支：F -> B -> 冻结源头，并在第二轮后混入目标头。
        task_features = netB(netF(weak_x))
        source_logits = source_head(task_features)
        model_logits = blend_classifier_logits(
            cfg,
            task_features,
            source_logits,
            target_head,
            curr_cycle,
        )

        # CLIP 分支：归一化图像特征与固定文本特征做余弦分类。
        clip_features = F.normalize(clip_model.encode_image(weak_x), dim=1)
        clip_logits = clip_logit_scale * clip_features @ text_features.t()

        model_logits_all.append(model_logits.float().cpu())
        clip_logits_all.append(clip_logits.float().cpu())
        task_features_all.append(task_features.float().cpu())
        clip_features_all.append(clip_features.float().cpu())
        target_labels_all.append(labels.long().cpu())

    model_prob = F.softmax(torch.cat(model_logits_all, dim=0), dim=1)
    clip_prob = F.softmax(torch.cat(clip_logits_all, dim=0), dim=1)
    task_features = torch.cat(task_features_all, dim=0)
    clip_features = torch.cat(clip_features_all, dim=0)
    target_labels = torch.cat(target_labels_all, dim=0)

    # Stage3 的 both_prior：两个教师分别做类别频率校准。
    model_prob, clip_prob, mix_prob = both_prior_calibration(
        cfg,
        model_prob,
        clip_prob,
    )
    source_label = model_prob.argmax(dim=1)
    clip_label = clip_prob.argmax(dim=1)
    mix_confidence, mix_label = mix_prob.max(dim=1)

    # 只有两个分支 top-1 一致的样本才可能进入硬伪标签池。
    agreement_mask = source_label == clip_label
    label_mask, memory_label, pseudo_label_state = update_stable_pseudo_labels(
        cfg,
        agreement_mask,
        mix_label,
        mix_confidence,
        pseudo_label_state,
        curr_cycle,
    )

    valid_count = int(label_mask.sum().item())
    valid_accuracy = 0.0
    if valid_count > 0:
        valid_accuracy = float(
            (memory_label[label_mask] == target_labels[label_mask]).float().mean().item()
        )
    LOGGER.info(
        "Stage14伪标签: 有效=%d/%d, 诊断精度=%.2f%%, 全量mix诊断精度=%.2f%%",
        valid_count,
        target_labels.numel(),
        valid_accuracy * 100.0,
        float((mix_label == target_labels).float().mean().item()) * 100.0,
    )

    return {
        "memory_label": memory_label,
        "label_mask": label_mask,
        "mix_prob": mix_prob,
        "clip_prob": clip_prob,
        "model_prob": model_prob,
        "source_label": source_label,
        "clip_label": clip_label,
        "task_features": task_features,
        "clip_features": clip_features,
        "target_label": target_labels,
        "pseudo_label_state": pseudo_label_state,
        "image_records": loader.dataset.imgs,
    }


# -----------------------------------------------------------------------------
# 6. 图时序残差：只处理稳定且图结构支持的冲突样本
# -----------------------------------------------------------------------------


def init_gtr_state(num_samples: int) -> dict[str, torch.Tensor]:
    """GTR 与硬伪标签记忆分开维护，避免两种状态互相污染。"""

    return {
        "pending_label": torch.full((num_samples,), -1, dtype=torch.long),
        "pending_count": torch.zeros(num_samples, dtype=torch.long),
        "stable_label": torch.full((num_samples,), -1, dtype=torch.long),
    }


@torch.no_grad()
def build_graph_temporal_target(
    cfg,
    cycle_data: dict,
    gtr_state: dict[str, torch.Tensor] | None,
):
    """
    用任务特征图和 CLIP 特征图传播 agreement anchors，再生成低权重软监督。

    graph teacher 不替换主 CLIP teacher；它只进入额外的 GTR KL。
    """

    _task_graph, _clip_graph, graph_prob, anchors = dual_space_diffusion(
        cycle_data["task_features"],
        cycle_data["clip_features"],
        cycle_data["model_prob"],
        cycle_data["clip_prob"],
        cycle_data["source_label"],
        cycle_data["clip_label"],
        anchor_ratio=float(cfg.DCCL.GTF_ANCHOR_RATIO),
        anchor_min_per_class=int(cfg.DCCL.GTF_ANCHOR_MIN_PER_CLASS),
        k=int(cfg.DCCL.GTF_GRAPH_K),
        temperature=float(cfg.DCCL.GTF_TEMPERATURE),
        alpha=float(cfg.DCCL.GTF_ALPHA),
        steps=int(cfg.DCCL.GTF_STEPS),
        chunk_size=int(cfg.DCCL.GTF_CHUNK_SIZE),
    )

    base_teacher = (cycle_data["model_prob"] + cycle_data["clip_prob"]) / 2.0
    fused_teacher, graph_mix_weight = adaptive_graph_teacher_fusion(
        base_teacher,
        graph_prob,
        strength=float(cfg.DCCL.GTF_STRENGTH),
        eps=float(cfg.DCCL.EPSILON),
    )

    teacher_label = fused_teacher.argmax(dim=1)
    graph_label = graph_prob.argmax(dim=1)
    eligible = (
        (cycle_data["source_label"] != cycle_data["clip_label"])
        & (teacher_label == graph_label)
    )

    if gtr_state is None:
        gtr_state = init_gtr_state(teacher_label.numel())
    (
        gtr_state["pending_label"],
        gtr_state["pending_count"],
        gtr_state["stable_label"],
        newly_stable,
        stable_mask,
        demoted,
    ) = update_temporal_resolution(
        gtr_state["pending_label"],
        gtr_state["pending_count"],
        gtr_state["stable_label"],
        eligible,
        teacher_label,
        int(cfg.DCCL.GTR_STABLE_CYCLES),
        cfg.DCCL.GTR_MEMORY,
    )

    gtr_weight, graph_confidence, disagreement = graph_temporal_residual_weights(
        cycle_data["clip_prob"],
        graph_prob,
        teacher_label,
        cycle_data["source_label"],
        cycle_data["clip_label"],
        gtr_state["stable_label"],
        float(cfg.DCCL.GTR_MIN_GRAPH_CONF),
        float(cfg.DCCL.GTR_MIN_DISAGREEMENT),
        eps=float(cfg.DCCL.EPSILON),
    )

    active = gtr_weight > 0
    LOGGER.info(
        "Stage14 GTR: anchors=%d, eligible=%d, 新稳定=%d, 稳定=%d, 撤销=%d, loss有效=%d",
        int(anchors.sum().item()),
        int(eligible.sum().item()),
        int(newly_stable.sum().item()),
        int(stable_mask.sum().item()),
        int(demoted.sum().item()),
        int(active.sum().item()),
    )
    if active.any():
        LOGGER.info(
            "Stage14 GTR权重: mean=%.4f, graph_conf=%.4f, disagreement=%.4f, graph_mix=%.4f",
            float(gtr_weight[active].mean().item()),
            float(graph_confidence[active].mean().item()),
            float(disagreement[active].mean().item()),
            float(graph_mix_weight.mean().item()),
        )
    return fused_teacher, gtr_weight, gtr_state


# -----------------------------------------------------------------------------
# 7. CLIP visual encoder 自适应
# -----------------------------------------------------------------------------


def train_clip_visual(
    cfg,
    clip_model: nn.Module,
    image_records,
    mix_prob: torch.Tensor,
    text_inputs: torch.Tensor,
    clip_optimizer: optim.Optimizer,
    q_value: float,
):
    """
    用本轮 mix soft labels 更新 CLIP visual encoder；文本侧始终冻结。

    数据集中的 target 只用于打印诊断准确率，不进入 Tsallis MI 损失。
    """

    cfg.domain_name = cfg.domain[cfg.SETTING.T]
    clip_dataset = build_dataset(
        "sfuda",
        image_test(),
        image_records,
        mix_prob,
        cfg.DATA_DIR,
        cfg.domain_name,
        mode="test",
    )
    clip_loader = DataLoader(
        clip_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False,
    )

    total_correct = 0
    total_samples = 0
    for images, target, pseudo_label, _sample_index in clip_loader:
        images = images.cuda(int(cfg.GPU_ID), non_blocking=True)
        target = target.cuda(int(cfg.GPU_ID), non_blocking=True)
        pseudo_label = pseudo_label.cuda()

        logits, _ = clip_model(images, text_inputs)
        clip_prob = F.softmax(logits, dim=1)
        clip_loss, q_value = IID_losses.tsallis_mutual_info(
            clip_prob,
            pseudo_label,
            q_value,
            float(cfg.ACTIVE.BETA),
        )

        clip_optimizer.zero_grad()
        clip_loss.backward()
        clip_optimizer.step()

        total_correct += int((clip_prob.argmax(dim=1) == target).sum().item())
        total_samples += int(target.size(0))

    LOGGER.info(
        "CLIP visual 诊断准确率=%.2f%%",
        100.0 * total_correct / max(total_samples, 1),
    )
    return q_value


# -----------------------------------------------------------------------------
# 8. 损失、评估与诊断文件
# -----------------------------------------------------------------------------


def consistency_loss(
    weak_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    stop_gradient: bool = False,
):
    """要求同一图像的弱增强与强增强预测保持一致。"""

    return prediction_consistency_kl(
        weak_logits,
        strong_logits,
        stop_gradient=stop_gradient,
    )


@torch.no_grad()
def evaluate(
    cfg,
    loader: DataLoader,
    netF: nn.Module,
    netB: nn.Module,
    source_head: nn.Module,
    target_head: nn.Module,
    curr_cycle: int,
):
    """使用与训练完全相同的双头融合规则评估目标域。"""

    logits_all = []
    labels_all = []
    for images, labels, _sample_indices in loader:
        images = images.cuda()
        features = netB(netF(images))
        logits = blend_classifier_logits(
            cfg,
            features,
            source_head(features),
            target_head,
            curr_cycle,
        )
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.long().cpu())

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    prediction = logits.argmax(dim=1)
    instance_accuracy = float((prediction == labels).float().mean().item()) * 100.0
    mean_entropy = float(loss.Entropy(F.softmax(logits, dim=1)).mean().item())

    per_class_text = None
    if cfg.SETTING.DATASET == "VISDA-C":
        matrix = confusion_matrix(labels.numpy(), prediction.numpy())
        per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1) * 100.0
        instance_accuracy = float(per_class.mean())
        per_class_text = " ".join(str(np.round(value, 2)) for value in per_class)
    return instance_accuracy, mean_entropy, per_class_text


def save_temporal_diagnostics(cfg, curr_cycle: int, cycle_data: dict, teacher_prob):
    """
    保存论文分析所需的逐样本状态。

    target_label 仅随诊断文件保存，训练代码从不读取它来做选择。
    """

    output_dir = osp.join(cfg.output_dir, cfg.DCCL.TEMPORAL_DIAG_DIR)
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f"{cfg.name}_cycle{curr_cycle + 1:02d}.npz")
    np.savez_compressed(
        output_path,
        cycle=np.array(curr_cycle + 1, dtype=np.int64),
        task=np.array(cfg.name),
        mix_label=cycle_data["memory_label"].numpy().astype(np.int64),
        label_mask=cycle_data["label_mask"].numpy().astype(bool),
        source_label=cycle_data["source_label"].numpy().astype(np.int64),
        clip_label=cycle_data["clip_label"].numpy().astype(np.int64),
        task_prob=cycle_data["model_prob"].numpy().astype(np.float32),
        clip_prob=cycle_data["clip_prob"].numpy().astype(np.float32),
        teacher_label=teacher_prob.argmax(dim=1).numpy().astype(np.int64),
        teacher_prob=teacher_prob.numpy().astype(np.float32),
        target_label=cycle_data["target_label"].numpy().astype(np.int64),
    )
    LOGGER.info("Stage14时序诊断已保存: %s", output_path)


def copy_optimizer_initial_lr(optimizer: optim.Optimizer) -> optim.Optimizer:
    """保存每个参数组的初始学习率，供每个 cycle 内的余弦调度使用。"""

    for group in optimizer.param_groups:
        group["lr0"] = group["lr"]
    return optimizer


def cosine_scheduler(cfg, optimizer, iter_num: int, max_iter: int, lr_min=1e-6):
    """Stage14 原实现使用的逐 iteration 余弦学习率。"""

    for group in optimizer.param_groups:
        lr_max = group["lr0"]
        group["lr"] = lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + np.cos(np.pi * iter_num / max_iter)
        )
        group["weight_decay"] = cfg.OPTIM.WD
        group["momentum"] = cfg.OPTIM.MOMENTUM
        group["nesterov"] = cfg.OPTIM.NESTEROV
    return optimizer


# -----------------------------------------------------------------------------
# 9. Stage14 主训练循环
# -----------------------------------------------------------------------------


def train_target(cfg):
    """
    Stage14 主入口。

    每个 cycle 的顺序固定为：
      整域伪标签 -> 图时序软目标 -> CLIP更新 -> 任务模型更新 -> 分段评估。
    """

    validate_stage14_config(cfg)
    loaders = data_load(cfg)

    # 9.1 加载源模型。F/B 会适应，源分类头 C_s 始终冻结。
    netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    netB = network.feat_bottleneck(
        type="bn",
        feature_dim=netF.in_features,
        bottleneck_dim=cfg.bottleneck,
    ).cuda()
    source_head = network.feat_classifier(
        type="wn",
        class_num=cfg.class_num,
        bottleneck_dim=cfg.bottleneck,
    ).cuda()

    netF.load_state_dict(torch.load(osp.join(cfg.output_dir_src, "source_F.pt")))
    netB.load_state_dict(torch.load(osp.join(cfg.output_dir_src, "source_B.pt")))
    source_head.load_state_dict(
        torch.load(osp.join(cfg.output_dir_src, "source_C.pt"))
    )
    source_head.eval()
    source_head.requires_grad_(False)

    # 9.2 目标分类头从源分类头复制，但允许用目标伪标签更新。
    target_head = build_target_head(cfg, source_head)
    target_head.train()

    optimizer = optim.SGD(
        [
            {
                "params": netF.parameters(),
                "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1,
            },
            {
                "params": netB.parameters(),
                "lr": cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2,
            },
            {
                "params": target_head.parameters(),
                "lr": cfg.OPTIM.LR * cfg.DCCL.TARGET_HEAD_LR_MULT,
            },
        ]
    )
    optimizer = copy_optimizer_initial_lr(optimizer)

    # 9.3 CLIP 只更新视觉编码器；文本编码器和 prompt 表示保持冻结。
    clip_model, _preprocess, _tokenizer = clip.load(cfg.ACTIVE.ARCH)
    clip_model.float()
    text_inputs = build_text_prompts(cfg)
    clip_model.transformer.requires_grad_(False)
    clip_model.token_embedding.requires_grad_(False)
    clip_model.positional_embedding.requires_grad = False
    clip_model.ln_final.requires_grad_(False)
    clip_model.text_projection.requires_grad = False
    clip_optimizer = optim.Adam(
        [parameter for parameter in clip_model.visual.parameters() if parameter.requires_grad],
        lr=cfg.ACTIVE.FINE_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    max_iter = cfg.TEST.MAX_EPOCH * len(loaders["target"])
    interval_iter = max(1, max_iter // cfg.TEST.INTERVAL)
    pseudo_label_state = None
    gtr_state = None
    q_value = float(cfg.ACTIVE.Q_VALUE)

    for curr_cycle in range(cfg.ACTIVE.CYCLE):
        LOGGER.info("========== Stage14 cycle %d/%d ==========", curr_cycle + 1, cfg.ACTIVE.CYCLE)

        # 9.4 在当前模型上冻结一次整域伪标签，本 cycle 内不再改变。
        netF.eval()
        netB.eval()
        target_head.eval()
        cycle_data = obtain_label(
            cfg,
            loaders["test_aug"],
            netF,
            netB,
            source_head,
            target_head,
            text_inputs,
            clip_model,
            pseudo_label_state,
            curr_cycle,
        )
        pseudo_label_state = cycle_data["pseudo_label_state"]

        # 9.5 GTR 只补充冲突样本的弱软约束，不改变硬伪标签池和主 CLIP KL。
        gtr_target, gtr_weight, gtr_state = build_graph_temporal_target(
            cfg,
            cycle_data,
            gtr_state,
        )
        save_temporal_diagnostics(cfg, curr_cycle, cycle_data, gtr_target)

        # 9.6 先让 CLIP visual encoder 跟随本轮混合软标签适应一次。
        q_value = train_clip_visual(
            cfg,
            clip_model,
            cycle_data["image_records"],
            cycle_data["mix_prob"],
            text_inputs,
            clip_optimizer,
            q_value,
        )

        # 训练所需的整域缓存放到 GPU；样本索引把 batch 对回整域行号。
        memory_label = cycle_data["memory_label"].cuda()
        label_mask = cycle_data["label_mask"]
        clip_target = cycle_data["clip_prob"].cuda()
        gtr_target = gtr_target.cuda()
        gtr_weight = gtr_weight.cuda()

        netF.train()
        netB.train()
        target_head.train()
        target_iterator = iter(loaders["target"])

        iter_num = 0
        while iter_num < max_iter:
            try:
                views, _unused_label, sample_indices = next(target_iterator)
            except StopIteration:
                target_iterator = iter(loaders["target"])
                views, _unused_label, sample_indices = next(target_iterator)

            # BatchNorm 在 batch size=1 时不稳定，沿用原实现的跳过策略。
            if views[0].size(0) == 1:
                continue

            iter_num += 1
            optimizer = cosine_scheduler(cfg, optimizer, iter_num, max_iter)
            weak_x = views[1].cuda()
            strong_x = views[2].cuda()

            weak_features = netB(netF(weak_x))
            strong_features = netB(netF(strong_x))
            weak_logits = blend_classifier_logits(
                cfg,
                weak_features,
                source_head(weak_features),
                target_head,
                curr_cycle,
            )
            strong_logits = blend_classifier_logits(
                cfg,
                strong_features,
                source_head(strong_features),
                target_head,
                curr_cycle,
            )
            weak_prob = F.softmax(weak_logits, dim=1)

            # L_cons：所有样本都参与弱/强增强一致性。
            total_loss = (
                consistency_loss(
                    weak_logits,
                    strong_logits,
                    stop_gradient=bool(cfg.DCCL.CONSISTENCY_STOP_GRAD),
                )
                * float(cfg.ACTIVE.CON_PAR)
            )

            # L_pseudoCE：只有经过可逆两轮记忆筛选的样本参与硬监督。
            batch_label_mask = label_mask[sample_indices]
            if batch_label_mask.any():
                selected_global_indices = sample_indices[batch_label_mask]
                selected_logits = weak_logits[batch_label_mask.cuda()]
                selected_labels = memory_label[selected_global_indices]
                total_loss = total_loss + F.cross_entropy(
                    selected_logits,
                    selected_labels,
                ) * float(cfg.ACTIVE.CLS_PAR)

            # L_CLIP-KL：所有样本保持对校准后 CLIP 软分布的拟合。
            clip_kl = F.kl_div(
                weak_prob.log(),
                clip_target[sample_indices],
                reduction="batchmean",
            )
            total_loss = total_loss + clip_kl * float(cfg.ACTIVE.KL_PAR)

            # L_GTR：仅对稳定、图支持且 CLIP 支持不足的冲突样本生效。
            batch_gtr_weight = gtr_weight[sample_indices]
            if batch_gtr_weight.sum() > 0:
                per_sample_gtr = F.kl_div(
                    weak_prob.log(),
                    gtr_target[sample_indices],
                    reduction="none",
                ).sum(dim=1)
                gtr_loss = (
                    per_sample_gtr * batch_gtr_weight
                ).sum() / batch_gtr_weight.sum()
                total_loss = total_loss + gtr_loss * float(cfg.DCCL.GTR_PAR)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 每个 cycle 固定间隔评估，原 Stage14 的 peak 就来自这些日志点。
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                netF.eval()
                netB.eval()
                target_head.eval()
                accuracy, mean_entropy, per_class_text = evaluate(
                    cfg,
                    loaders["test"],
                    netF,
                    netB,
                    source_head,
                    target_head,
                    curr_cycle,
                )
                LOGGER.info(
                    "Task=%s, Iter=%d/%d, Cycle=%d/%d, Accuracy=%.2f%%, Entropy=%.4f, Loss=%.6f",
                    cfg.name,
                    iter_num,
                    max_iter,
                    curr_cycle + 1,
                    cfg.ACTIVE.CYCLE,
                    accuracy,
                    mean_entropy,
                    float(total_loss.detach().item()),
                )
                if per_class_text is not None:
                    LOGGER.info("VisDA-C per-class accuracy: %s", per_class_text)
                netF.train()
                netB.train()
                target_head.train()

    # 与原 dccl.py 不同，这里显式返回目标头，避免训练后的关键权重丢失。
    return netF, netB, source_head, target_head


# -----------------------------------------------------------------------------
# 10. CLIP 文本 prompt
# -----------------------------------------------------------------------------


def build_text_prompts(cfg) -> torch.Tensor:
    """按 Stage14 的固定模板构造每个类别对应的 CLIP 文本输入。"""

    with open(cfg.name_file) as handle:
        class_names = [token for line in handle for token in line.split()]
    class_names = [name.replace("_", " ") for name in class_names]
    cfg.classname = class_names
    prompt_prefix = cfg.ACTIVE.CTX_INIT.replace("_", " ")
    prompts = [f"{prompt_prefix} {name}." for name in class_names]
    return torch.cat([clip.tokenize(prompt) for prompt in prompts]).cuda()
