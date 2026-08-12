"""Class-balanced anchor bank + lightweight Context Transformer for DUET-FCP.

This module implements the core of the candidate method
``duet_first_cycle_prior_context_transformer``:

1.  Use high-confidence, Task/CLIP-consistent target samples as a
    class-balanced anchor bank with hard pseudo-labels.
2.  Treat Task/CLIP strict-conflict samples and low-confidence
    weak-agreement samples as image-level feature queries.
3.  A lightweight cross-attention Transformer (or a cosine-kNN / prototype
    control) re-classifies those queries against the anchor context, then the
    decision rules either admit them with a hard pseudo-label, keep the
    original agreement rule, or abstain.  The first training cycle
    (``ACTIVE_CYCLES`` default index 1, i.e. the second cycle) stays pure
    DUET-FCP so the anchor bank is built after at least one round of
    Task/CLIP target adaptation.

The module only depends on torch.  It never sees target ground-truth except
through the optional ``labels`` argument, which is used exclusively for
evaluation-only logging (``ground_truth_affects_training=False``).
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-9


def _softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits.float(), dim=1)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(_EPS))).sum(dim=1)


class ClassBalancedAnchorBank:
    """Per-class top-k anchor memory.

    Each class keeps at most ``anchors_per_class`` anchors, selected by the
    highest reliability score.  All tensors follow the spec:

        anchor_features: [C, K, D]
        anchor_labels:   [C, K]
        anchor_scores:   [C, K]
        anchor_indices:  [C, K]
        anchor_valid:    [C, K]

    Selection never uses target ground-truth.  Features are stored detached.
    """

    def __init__(
        self,
        num_classes: int,
        anchors_per_class: int,
        feature_dim: int,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_classes <= 0 or anchors_per_class <= 0 or feature_dim <= 0:
            raise ValueError("num_classes, anchors_per_class and feature_dim must be positive")
        self.num_classes = int(num_classes)
        self.anchors_per_class = int(anchors_per_class)
        self.feature_dim = int(feature_dim)
        self.seed = int(seed)
        self.device = device or torch.device("cpu")
        self.anchor_features = torch.zeros(
            self.num_classes, self.anchors_per_class, self.feature_dim,
            dtype=torch.float32, device=self.device,
        )
        self.anchor_labels = torch.zeros(
            self.num_classes, self.anchors_per_class,
            dtype=torch.long, device=self.device,
        )
        self.anchor_scores = torch.full(
            (self.num_classes, self.anchors_per_class),
            float("-inf"), dtype=torch.float32, device=self.device,
        )
        self.anchor_indices = torch.full(
            (self.num_classes, self.anchors_per_class),
            -1, dtype=torch.long, device=self.device,
        )
        self.anchor_valid = torch.zeros(
            self.num_classes, self.anchors_per_class,
            dtype=torch.bool, device=self.device,
        )

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
    ) -> "ClassBalancedAnchorBank":
        """Add samples and keep the highest-scoring K per class (detached)."""
        features = features.detach().float().to(self.device)
        labels = labels.detach().long().to(self.device)
        scores = scores.detach().float().to(self.device)
        if sample_indices is not None:
            sample_indices = sample_indices.detach().long().to(self.device)
        if features.dim() != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                "features must be [N, feature_dim] with feature_dim={}".format(
                    self.feature_dim
                )
            )
        if labels.shape != (features.size(0),):
            raise ValueError("labels must be [N]")
        if scores.shape != (features.size(0),):
            raise ValueError("scores must be [N]")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("labels must be in [0, num_classes)")

        for cls in range(self.num_classes):
            selected = torch.nonzero(labels == cls, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            cls_scores = scores[selected].float()
            if sample_indices is not None:
                cls_indices = sample_indices[selected].long()
            else:
                cls_indices = selected.long()
            # Deterministic ranking: primary = score (descending), secondary
            # = sample index (ascending).  Stable argsort keeps the original
            # ascending index order for ties.
            order = torch.argsort(cls_scores, descending=True, stable=True)
            keep = order[: self.anchors_per_class]
            count = keep.numel()
            self.anchor_features[cls, :count] = features[selected[keep]]
            self.anchor_labels[cls, :count] = labels[selected[keep]]
            self.anchor_scores[cls, :count] = scores[selected[keep]]
            self.anchor_indices[cls, :count] = cls_indices[keep]
            self.anchor_valid[cls, :count] = True
        return self

    def flatten(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return flat tensors: features, labels, scores, indices, valid."""
        valid = self.anchor_valid
        return (
            self.anchor_features[valid],
            self.anchor_labels[valid],
            self.anchor_scores[valid],
            self.anchor_indices[valid],
            self.anchor_valid[valid].clone(),
        )

    def per_class_counts(self) -> torch.Tensor:
        return self.anchor_valid.sum(dim=1)

    def num_classes_filled(self) -> int:
        return int((self.per_class_counts() > 0).sum().item())

    def summary(self) -> dict:
        counts = self.per_class_counts()
        return {
            "per_class_counts": counts.tolist(),
            "total": int(counts.sum().item()),
            "classes_filled": self.num_classes_filled(),
            "seed": self.seed,
        }


class DuetContextConflictTransformer(nn.Module):
    """Lightweight cross-attention Context Transformer.

    Query tokens are the image-level Task features of conflict /
    weak-agreement samples; key/value tokens are the anchor-bank Task
    features plus a class embedding of their hard pseudo-label.  Only this
    module is trained (all inputs are detached).

    Outputs:
        logits        [B_query, C]
        probabilities [B_query, C]
        attention     [B_query, 1, A] (or None when no queries)
        hidden        [B_query, MODEL_DIM]
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        model_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.query_projection = nn.Linear(feature_dim, model_dim)
        self.anchor_projection = nn.Linear(feature_dim, model_dim)
        self.class_embedding = nn.Embedding(num_classes, model_dim)
        self.cross_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(
        self,
        query_features: torch.Tensor,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_valid_mask: torch.Tensor,
        anchor_self_exclude: Optional[torch.Tensor] = None,
    ) -> dict:
        # query_features  [B, D]
        # anchor_features [A, D]
        # anchor_labels   [A]     (hard pseudo-label of each anchor)
        # anchor_valid    [A]     (bool)
        # anchor_self_exclude [B, A] bool: True = hide that anchor for this query
        if query_features.dim() != 2:
            raise ValueError("query_features must be [B, feature_dim]")
        batch = query_features.size(0)
        if batch == 0:
            empty = query_features.new_zeros(0, self.num_classes)
            return {
                "logits": empty,
                "probabilities": empty,
                "attention": None,
                "hidden": query_features.new_zeros(0, self.model_dim),
            }
        anchor_features = anchor_features.detach()
        anchor_labels = anchor_labels.detach().long()
        anchor_valid_mask = anchor_valid_mask.detach().bool()

        query_token = self.query_projection(query_features).unsqueeze(1)  # [B,1,H]
        anchor_token = self.anchor_projection(anchor_features).unsqueeze(0)  # [1,A,H]
        anchor_token = anchor_token + self.class_embedding(anchor_labels).unsqueeze(0)
        anchor_token = anchor_token.expand(batch, -1, -1)  # [B,A,H]

        key_padding_mask = ~anchor_valid_mask.unsqueeze(0).expand(batch, -1)
        if anchor_self_exclude is not None:
            key_padding_mask = key_padding_mask | anchor_self_exclude.bool()

        context, attention = self.cross_attention(
            query=query_token,
            key=anchor_token,
            value=anchor_token,
            key_padding_mask=key_padding_mask,
        )
        hidden = self.norm1(query_token + context)
        hidden = self.norm2(hidden + self.ffn(hidden))
        logits = self.classifier(hidden.squeeze(1))  # [B, C]
        if not torch.isfinite(logits).all():
            # Fully-masked anchor rows (or an empty class set) must not leak
            # NaN into the loss or the admission decision.
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "logits": logits,
            "probabilities": _softmax_probabilities(logits),
            "attention": attention,
            "hidden": hidden.squeeze(1),
        }


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor.float(), p=2, dim=1)


def cosine_knn_refine(
    query_features: torch.Tensor,
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    num_classes: int,
    k: int = 5,
) -> torch.Tensor:
    """Cosine kNN control: query vs anchor features -> [B, C] probabilities."""
    batch = query_features.size(0)
    if batch == 0:
        return query_features.new_zeros(0, num_classes)
    query_norm = _normalize_rows(query_features)
    anchor_norm = _normalize_rows(anchor_features)
    similarity = query_norm @ anchor_norm.t()  # [B, A]
    similarity = similarity.masked_fill(
        ~anchor_valid_mask.unsqueeze(0), float("-inf")
    )
    valid_count = int(anchor_valid_mask.sum().item())
    if valid_count == 0:
        return torch.full(
            (batch, num_classes), 1.0 / num_classes, dtype=query_features.dtype
        )
    top_k = min(k, valid_count)
    values, indices = similarity.topk(top_k, dim=1)
    values = values.masked_fill(~torch.isfinite(values), 0.0)
    labels_k = anchor_labels[indices]  # [B, k]
    scores = torch.zeros(
        batch, num_classes, dtype=query_features.dtype, device=query_features.device
    )
    scores.scatter_add_(1, labels_k, values)
    return _softmax_probabilities(scores.clamp_min(0.0))


def prototype_refine(
    query_features: torch.Tensor,
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Class-prototype control: cosine similarity to per-class anchor means."""
    batch = query_features.size(0)
    if batch == 0:
        return query_features.new_zeros(0, num_classes)
    device = query_features.device
    query_norm = _normalize_rows(query_features)
    anchor_norm = _normalize_rows(anchor_features)
    prototypes = torch.zeros(
        num_classes, anchor_norm.size(1), dtype=anchor_norm.dtype, device=device
    )
    class_present = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for cls in range(num_classes):
        mask = (anchor_labels == cls) & anchor_valid_mask
        if bool(mask.any().item()):
            prototypes[cls] = anchor_norm[mask].mean(dim=0)
            class_present[cls] = True
    if not bool(class_present.any().item()):
        return torch.full(
            (batch, num_classes), 1.0 / num_classes, dtype=query_features.dtype
        )
    prototype_norm = _normalize_rows(prototypes)
    similarity = query_norm @ prototype_norm.t()  # [B, C]
    similarity[:, ~class_present] = float("-inf")
    return _softmax_probabilities(similarity)


def _topk_mean_cosine_support(
    query_feature: torch.Tensor,
    class_features: torch.Tensor,
    class_valid: torch.Tensor,
    topk: int,
) -> tuple[float, float]:
    """query 与该类 anchor 的 top-k 余弦相似度均值 + 是否有 anchor。

    用 top-k 均值而不是 max，避免单个异常 anchor 撑高分。
    """
    # 统一设备：anchor 特征可能在 GPU（bank），query 特征可能还在 CPU
    # （obtain_label 里收集），先对齐到 query 的 device 再算。
    query_feature = query_feature.detach().float()
    class_features = class_features.detach().float().to(query_feature.device)
    class_valid = class_valid.to(query_feature.device)
    valid_features = class_features[class_valid]
    if valid_features.numel() == 0:
        return 0.0, 0.0
    query_norm = F.normalize(query_feature.unsqueeze(0), dim=1)
    anchor_norm = F.normalize(valid_features, dim=1)
    similarities = (query_norm @ anchor_norm.t()).squeeze(0)
    k = min(int(topk), similarities.numel())
    return float(similarities.topk(k).values.mean().item()), 1.0


def _log_pair_distribution(
    features: torch.Tensor,
    tag: str,
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    """输出一组 pair 特征（16 维）的均值分布，用于对比 synthetic vs real。"""
    if features.numel() == 0:
        return
    means = features.float().mean(dim=0)
    log_fn(
        "DUET comparator {} distribution: cycle={}; p_task_A={:.4f}; "
        "p_task_B={:.4f}; p_clip_A={:.4f}; p_clip_B={:.4f}; "
        "task_margin={:.4f}; clip_margin={:.4f}; task_entropy={:.4f}; "
        "clip_entropy={:.4f}; task_sim_A={:.4f}; task_sim_B={:.4f}; "
        "clip_sim_A={:.4f}; clip_sim_B={:.4f}; "
        "ground_truth_affects_training=False".format(
            tag,
            cycle,
            means[0].item(),
            means[1].item(),
            means[2].item(),
            means[3].item(),
            means[6].item(),
            means[7].item(),
            means[4].item(),
            means[5].item(),
            means[8].item(),
            means[9].item(),
            means[10].item(),
            means[11].item(),
        )
    )


def _log_real_comparator_margin_distribution(
    margins: torch.Tensor,
    cycle: int,
    gate: float,
    log_fn: Callable[[str], None],
) -> None:
    """Log GT-free margin diagnostics over all real strict conflicts."""
    margins = margins.detach().float().flatten().cpu()
    total = int(margins.numel())
    if total == 0:
        log_fn(
            "DUET comparator real-margin distribution: cycle={}; total=0; "
            "mean=nan; p50=nan; p75=nan; p90=nan; p95=nan; max=nan; "
            "gate={:.2f}; ground_truth_affects_training=False".format(
                cycle, gate
            )
        )
        log_fn(
            "DUET comparator real-margin thresholds: cycle={}; total=0; "
            "margin_ge_0.10=0/0 (0.00%); margin_ge_0.15=0/0 (0.00%); "
            "margin_ge_0.20=0/0 (0.00%); margin_ge_0.25=0/0 (0.00%); "
            "margin_ge_0.30=0/0 (0.00%); "
            "ground_truth_affects_training=False".format(cycle)
        )
        return

    quantiles = torch.quantile(
        margins, torch.tensor([0.50, 0.75, 0.90, 0.95])
    )
    log_fn(
        "DUET comparator real-margin distribution: cycle={}; total={}; "
        "mean={:.4f}; p50={:.4f}; p75={:.4f}; p90={:.4f}; p95={:.4f}; "
        "max={:.4f}; gate={:.2f}; "
        "ground_truth_affects_training=False".format(
            cycle,
            total,
            margins.mean().item(),
            quantiles[0].item(),
            quantiles[1].item(),
            quantiles[2].item(),
            quantiles[3].item(),
            margins.max().item(),
            gate,
        )
    )

    threshold_parts = []
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30):
        count = int((margins >= threshold).sum().item())
        rate = 100.0 * count / total
        threshold_parts.append(
            "margin_ge_{:.2f}={}/{} ({:.2f}%)".format(
                threshold, count, total, rate
            )
        )
    log_fn(
        "DUET comparator real-margin thresholds: cycle={}; total={}; {}; "
        "ground_truth_affects_training=False".format(
            cycle, total, "; ".join(threshold_parts)
        )
    )


@torch.no_grad()
def _log_fixed_conflict_trajectory(
    trajectory: list[dict],
    *,
    task_candidates: torch.Tensor,
    clip_candidates: torch.Tensor,
    labels: torch.Tensor,
    coverages: list[int],
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    """Eval-only trajectory on one fixed real-conflict cohort.

    The cohort is fixed before comparator optimization.  Every checkpoint is
    evaluated without the comparator gate, so changes in selection quality
    cannot be confused with changes in coverage.  GT is read only here,
    after training has completed; it never selects or restores a checkpoint.
    """
    task_candidates = task_candidates.detach().long().cpu()
    clip_candidates = clip_candidates.detach().long().cpu()
    labels = labels.detach().long().cpu()
    total = int(labels.numel())
    if total == 0:
        return
    if task_candidates.shape != labels.shape or clip_candidates.shape != labels.shape:
        raise ValueError("fixed-conflict candidates and labels must have equal shape")

    task_correct = task_candidates == labels
    clip_correct = clip_candidates == labels
    oracle_correct = task_correct | clip_correct
    oracle_correct_count = int(oracle_correct.sum().item())

    def pct(correct: torch.Tensor) -> float:
        return 100.0 * float(correct.float().mean().item())

    for checkpoint in trajectory:
        logits = checkpoint["fixed_logits"].detach().float().cpu()
        if logits.shape != (total, 2):
            raise ValueError("fixed-conflict checkpoint logits must be [N, 2]")
        probabilities = _softmax_probabilities(logits)
        trust_task = probabilities[:, 0]
        trust_clip = probabilities[:, 1]
        margins = (trust_task - trust_clip).abs()
        chosen = torch.where(
            trust_task >= trust_clip, task_candidates, clip_candidates
        )
        comparator_correct = chosen == labels
        comparator_correct_count = int(comparator_correct.sum().item())
        conditional = (
            100.0 * comparator_correct_count / oracle_correct_count
            if oracle_correct_count > 0
            else float("nan")
        )
        quantiles = torch.quantile(margins, torch.tensor([0.50, 0.90]))
        log_fn(
            "DUET comparator fixed-conflict trajectory eval-only: cycle={}; "
            "step={}; fixed_conflicts={}; synthetic_train_loss={:.6f}; "
            "task_acc={:.2f}%; clip_acc={:.2f}%; comparator_acc={:.2f}%; "
            "candidate_oracle_acc={:.2f}%; conditional_arbitration_acc={:.2f}%; "
            "trust_task={}; trust_clip={}; margin_mean={:.4f}; "
            "margin_p50={:.4f}; margin_p90={:.4f}; gate_used=False; "
            "checkpoint_selected_by_gt=False; ground_truth_affects_training=False".format(
                cycle,
                checkpoint["step"],
                total,
                checkpoint["synthetic_train_loss"],
                pct(task_correct),
                pct(clip_correct),
                pct(comparator_correct),
                pct(oracle_correct),
                conditional,
                int((trust_task >= trust_clip).sum().item()),
                int((trust_task < trust_clip).sum().item()),
                float(margins.mean().item()),
                float(quantiles[0].item()),
                float(quantiles[1].item()),
            )
        )

        order = torch.argsort(margins, descending=True, stable=True)
        coverage_parts = []
        for requested_coverage in coverages:
            selected_count = max(
                1,
                min(
                    total,
                    int(round(total * float(requested_coverage) / 100.0)),
                ),
            )
            selected = order[:selected_count]
            selected_oracle_count = int(oracle_correct[selected].sum().item())
            selected_comparator_count = int(
                comparator_correct[selected].sum().item()
            )
            selected_conditional = (
                100.0 * selected_comparator_count / selected_oracle_count
                if selected_oracle_count > 0
                else float("nan")
            )
            prefix = "coverage_{}".format(int(requested_coverage))
            coverage_parts.extend(
                [
                    "{}_n={}".format(prefix, selected_count),
                    "{}_task_acc={:.2f}%".format(
                        prefix, pct(task_correct[selected])
                    ),
                    "{}_clip_acc={:.2f}%".format(
                        prefix, pct(clip_correct[selected])
                    ),
                    "{}_comparator_acc={:.2f}%".format(
                        prefix, pct(comparator_correct[selected])
                    ),
                    "{}_candidate_oracle_acc={:.2f}%".format(
                        prefix, pct(oracle_correct[selected])
                    ),
                    "{}_conditional_arbitration_acc={:.2f}%".format(
                        prefix, selected_conditional
                    ),
                ]
            )
        log_fn(
            "DUET comparator fixed-coverage trajectory eval-only: cycle={}; "
            "step={}; {}; checkpoint_selected_by_gt=False; "
            "ground_truth_affects_training=False".format(
                cycle, checkpoint["step"], "; ".join(coverage_parts)
            )
        )


def _zscore_filter(
    features: torch.Tensor,
    reference: torch.Tensor,
    dims: list,
    z_max: float,
    min_kept: int = 16,
) -> tuple[torch.Tensor, bool]:
    """按 reference（真实 conflict）的分布，过滤 synthetic 特征池。

    对指定维度逐维算 z-score，要求所有匹配维度的 |z| <= z_max；
    维度方差为 0 时视为完全匹配（z=0）。
    命中数不足 ``min_kept`` 且池子够大时，退化为保留 mean|z| 最小的
    ``min_kept`` 个样本，保证始终有训练数据。
    """
    if features.numel() == 0 or reference.numel() == 0:
        return torch.zeros(features.size(0), dtype=torch.bool), False
    features = features.float()
    reference = reference.float()
    mean = reference.mean(dim=0)
    std = reference.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    z = (features - mean.unsqueeze(0)) / std.unsqueeze(0)
    z_sub = z[:, dims]
    keep = (z_sub.abs() <= float(z_max)).all(dim=1)
    mean_abs_z = z_sub.abs().mean(dim=1)
    if int(keep.sum().item()) < int(min_kept) and features.size(0) > 0:
        order = torch.argsort(mean_abs_z)
        n_keep = min(int(min_kept), features.size(0))
        keep = torch.zeros_like(keep)
        keep[order[:n_keep]] = True
        return keep, True
    return keep, False


class PairwiseConflictComparator(nn.Module):
    """Pairwise conflict-resolution comparator（二选一 + 边际 abstain）。

    输入是 class-agnostic 的相对证据（16 维），输出只有 2 个 logits：
    trust Task 侧候选 vs trust CLIP 侧候选。abstain 由边际门槛决定，
    不训练第三个类。
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("COMPARATOR_LAYERS must be >= 1")
        blocks = []
        current = int(input_dim)
        for _ in range(int(layers)):
            blocks.append(nn.Linear(current, int(hidden)))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
            current = int(hidden)
        blocks.append(nn.Linear(current, 2))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features.float())


def build_comparator_features(
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    task_features: torch.Tensor,
    clip_features: Optional[torch.Tensor],
    task_bank: ClassBalancedAnchorBank,
    clip_bank: Optional[ClassBalancedAnchorBank],
    class_a: torch.Tensor,
    class_b: torch.Tensor,
    sim_topk: int,
) -> torch.Tensor:
    """为每个 (A=Task 候选, B=CLIP 候选) 构造 16 维 class-agnostic 特征。

    维度：
        [p_task(A), p_task(B), p_clip(A), p_clip(B),
         task_entropy, clip_entropy, task_margin, clip_margin,
         task_sim_A, task_sim_B, clip_sim_A, clip_sim_B,
         task_A_avail, task_B_avail, clip_A_avail, clip_B_avail]
    """
    batch = class_a.numel()
    features = torch.zeros(batch, 16, dtype=torch.float32, device=task_probs.device)
    task_probs = task_probs.float()
    clip_probs = clip_probs.float()
    for i in range(batch):
        a = int(class_a[i].item())
        b = int(class_b[i].item())
        task_sorted = task_probs[i].sort(descending=True).values
        clip_sorted = clip_probs[i].sort(descending=True).values
        task_margin = float(task_sorted[0] - task_sorted[1])
        clip_margin = float(clip_sorted[0] - clip_sorted[1])
        task_entropy = float(_entropy(task_probs[i : i + 1])[0].item())
        clip_entropy = float(_entropy(clip_probs[i : i + 1])[0].item())
        task_sim_a, task_avail_a = _topk_mean_cosine_support(
            task_features[i], task_bank.anchor_features[a], task_bank.anchor_valid[a], sim_topk
        )
        task_sim_b, task_avail_b = _topk_mean_cosine_support(
            task_features[i], task_bank.anchor_features[b], task_bank.anchor_valid[b], sim_topk
        )
        if clip_features is not None and clip_bank is not None:
            clip_sim_a, clip_avail_a = _topk_mean_cosine_support(
                clip_features[i], clip_bank.anchor_features[a], clip_bank.anchor_valid[a], sim_topk
            )
            clip_sim_b, clip_avail_b = _topk_mean_cosine_support(
                clip_features[i], clip_bank.anchor_features[b], clip_bank.anchor_valid[b], sim_topk
            )
        else:
            clip_sim_a = clip_sim_b = 0.0
            clip_avail_a = clip_avail_b = 0.0
        features[i] = torch.tensor(
            [
                float(task_probs[i, a]),
                float(task_probs[i, b]),
                float(clip_probs[i, a]),
                float(clip_probs[i, b]),
                task_entropy,
                clip_entropy,
                task_margin,
                clip_margin,
                task_sim_a,
                task_sim_b,
                clip_sim_a,
                clip_sim_b,
                task_avail_a,
                task_avail_b,
                clip_avail_a,
                clip_avail_b,
            ],
            dtype=torch.float32,
            device=features.device,
        )
    return features


def build_synthetic_conflicts(
    pool_labels: torch.Tensor,
    pool_strong_task_probs: torch.Tensor,
    pool_strong_clip_probs: torch.Tensor,
    pool_strong_task_features: torch.Tensor,
    pool_strong_clip_features: torch.Tensor,
    task_bank: ClassBalancedAnchorBank,
    clip_bank: ClassBalancedAnchorBank,
    *,
    min_runner_prob: float,
    max_top1_margin: float,
    sim_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """同 view synthetic conflict：同一张 strong 增广图上 Task/CLIP 的 disagreement。

    高可信 weak consensus (Task=A, CLIP=A) 之后，只看同一 strong 视图：
      - Task(strong)=B, CLIP(strong)=A  → synthetic conflict，target=trust CLIP(1)
      - Task(strong)=A, CLIP(strong)=B  → synthetic conflict，target=trust Task(0)
      - 两边都保持 A                    → 没有 conflict，丢弃
      - 两边都离开 A（B≠C）             → 无法确定谁保持了原标签，丢弃
    16 维证据全部来自同一 strong 视图：概率 = strong probs，
    anchor similarity = strong feature vs anchor bank（bank 仍是 weak-view
    特征库，与推理时一致）。flip 需要过 hard-gate（MIN_RUNNER_PROB /
    MAX_TOP1_MARGIN，margin = p_B - p_A）。
    """
    if pool_strong_task_probs is None or pool_strong_clip_probs is None:
        raise ValueError("same-view synthetic conflicts require strong Task/CLIP probs")
    if pool_strong_task_features is None or pool_strong_clip_features is None:
        raise ValueError(
            "same-view synthetic conflicts require strong Task/CLIP features"
        )
    feature_rows = []
    target_rows = []
    task_side = 0
    clip_side = 0
    task_flip_only = 0
    clip_flip_only = 0
    both_flip = 0
    no_conflict = 0
    # 用全部 anchor 候选（不是 top-8 bank）造 synthetic 对：top-8 的 anchor
    # 太“容易”，augmentation 几乎不会把它们翻错，翻错/强 runner-up 更多
    # 出现在置信度稍低的候选里，那才是像真实 conflict 的 hard disagreement。
    for pool_id in range(pool_labels.numel()):
        anchor_label = int(pool_labels[pool_id].item())
        if anchor_label < 0:
            continue
        strong_task = pool_strong_task_probs[pool_id]
        strong_clip = pool_strong_clip_probs[pool_id]
        task_top1 = int(strong_task.argmax().item())
        clip_top1 = int(strong_clip.argmax().item())
        task_flipped = task_top1 != anchor_label
        clip_flipped = clip_top1 != anchor_label
        if task_flipped and not clip_flipped:
            # 同一 strong 视图：Task 翻到 B，CLIP 保持 A → trust CLIP
            task_flip_only += 1
            p_b = float(strong_task[task_top1].item())
            p_a = float(strong_task[anchor_label].item())
            margin = p_b - p_a
            if p_b < min_runner_prob or margin > max_top1_margin:
                continue
            features = build_comparator_features(
                strong_task.unsqueeze(0),
                strong_clip.unsqueeze(0),
                pool_strong_task_features[pool_id : pool_id + 1],
                pool_strong_clip_features[pool_id : pool_id + 1],
                task_bank,
                clip_bank,
                class_a=torch.tensor([task_top1]),
                class_b=torch.tensor([anchor_label]),
                sim_topk=sim_topk,
            )
            feature_rows.append(features[0])
            target_rows.append(1.0)  # trust CLIP
            task_side += 1
        elif clip_flipped and not task_flipped:
            # 同一 strong 视图：CLIP 翻到 B，Task 保持 A → trust Task
            clip_flip_only += 1
            p_b = float(strong_clip[clip_top1].item())
            p_a = float(strong_clip[anchor_label].item())
            margin = p_b - p_a
            if p_b < min_runner_prob or margin > max_top1_margin:
                continue
            features = build_comparator_features(
                strong_task.unsqueeze(0),
                strong_clip.unsqueeze(0),
                pool_strong_task_features[pool_id : pool_id + 1],
                pool_strong_clip_features[pool_id : pool_id + 1],
                task_bank,
                clip_bank,
                class_a=torch.tensor([anchor_label]),
                class_b=torch.tensor([clip_top1]),
                sim_topk=sim_topk,
            )
            feature_rows.append(features[0])
            target_rows.append(0.0)  # trust Task
            clip_side += 1
        elif task_flipped and clip_flipped:
            both_flip += 1
        else:
            no_conflict += 1
    counts = {
        "task_flip_only": task_flip_only,
        "clip_flip_only": clip_flip_only,
        "both_flip": both_flip,
        "no_conflict": no_conflict,
        "task_side": task_side,
        "clip_side": clip_side,
    }
    if not feature_rows:
        return (
            torch.zeros(0, 16, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
            counts,
        )
    return (
        torch.stack(feature_rows),
        torch.tensor(
            target_rows,
            dtype=torch.float32,
            device=feature_rows[0].device,
        ),
        counts,
    )


class ComparatorReplayMemory:
    """Persistent comparator 的历史 synthetic replay buffer。

    按信任方向（trust Task / trust CLIP）分别保存最多
    ``per_direction_capacity`` 个 matched synthetic 样本；
    训练时与当前 cycle 的 matched synthetic 按比例混合。
    """

    def __init__(
        self,
        per_direction_capacity: int = 64,
        feature_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.per_direction_capacity = int(per_direction_capacity)
        self.device = device or torch.device("cpu")
        self.feature_dim = int(feature_dim) if feature_dim is not None else None
        self.task_features = torch.zeros(
            0, self.feature_dim or 0, dtype=torch.float32, device=self.device
        )
        self.clip_features = torch.zeros(
            0, self.feature_dim or 0, dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def update(self, features: torch.Tensor, targets: torch.Tensor) -> "ComparatorReplayMemory":
        """按方向追加 matched synthetic，每个方向最多保留 capacity 个（去旧保新）。"""
        features = features.detach().float().to(self.device)
        targets = targets.detach().float().to(self.device)
        if features.dim() != 2:
            return self
        if self.feature_dim is None:
            self.feature_dim = features.size(1)
            self.task_features = torch.zeros(
                0, self.feature_dim, dtype=torch.float32, device=self.device
            )
            self.clip_features = torch.zeros(
                0, self.feature_dim, dtype=torch.float32, device=self.device
            )
        for direction, storage in ((0.0, "task_features"), (1.0, "clip_features")):
            mask = targets == direction
            if int(mask.sum().item()) == 0:
                continue
            current = getattr(self, storage)
            new_chunk = features[mask]
            merged = (
                torch.cat([current, new_chunk], dim=0)
                if current.numel() > 0
                else new_chunk
            )
            if merged.size(0) > self.per_direction_capacity:
                merged = merged[-self.per_direction_capacity:]
            setattr(self, storage, merged)
        return self

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """从 memory 中均衡采样 n 个样本（两个方向各一半）。"""
        n = max(1, int(n))
        pools = ((self.task_features, 0.0), (self.clip_features, 1.0))
        half = max(1, n // 2)
        feature_rows = []
        target_rows = []
        for pool, target_value in pools:
            if pool.size(0) == 0:
                continue
            indices = torch.randint(
                0,
                pool.size(0),
                (half,),
                generator=generator,
                device=self.device,
            )
            feature_rows.append(pool[indices])
            target_rows.append(
                torch.full((half,), target_value, dtype=torch.float32, device=self.device)
            )
        if not feature_rows:
            return (
                torch.zeros(0, self.feature_dim or 0, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        return torch.cat(feature_rows), torch.cat(target_rows)

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 memory 的全部 (features, targets)。"""
        parts_features = []
        parts_targets = []
        for pool, target_value in (
            (self.task_features, 0.0),
            (self.clip_features, 1.0),
        ):
            if pool.size(0) > 0:
                parts_features.append(pool)
                parts_targets.append(
                    torch.full(
                        (pool.size(0),),
                        target_value,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
        if not parts_features:
            return (
                torch.zeros(0, self.feature_dim or 0, device=self.device),
                torch.zeros(0, dtype=torch.float32, device=self.device),
            )
        return torch.cat(parts_features), torch.cat(parts_targets)

    def total(self) -> int:
        return int(self.task_features.size(0)) + int(self.clip_features.size(0))

    def clear(self) -> "ComparatorReplayMemory":
        self.task_features = torch.zeros(
            0, self.feature_dim or 0, dtype=torch.float32, device=self.device
        )
        self.clip_features = torch.zeros(
            0, self.feature_dim or 0, dtype=torch.float32, device=self.device
        )
        return self


def train_pairwise_comparator(
    comparator: PairwiseConflictComparator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    memory_features: Optional[torch.Tensor] = None,
    memory_targets: Optional[torch.Tensor] = None,
    memory_fraction: float = 0.25,
    trajectory_features: Optional[torch.Tensor] = None,
    trajectory_interval: int = 0,
    trajectory_sink: Optional[list[dict]] = None,
) -> Optional[float]:
    """在 synthetic conflict 对上训练 comparator（2-way CE）。

    与 Run 9/10 的训练方式完全一致：matched synthetic 已经 1:1 平衡，
    池子 <= batch_size 时每个 step 直接用全部样本，否则随机抽 batch。
    提供 ``memory_features/memory_targets`` 时启用 replay：每个 step 按
    ``memory_fraction`` 从历史 memory 采样、其余从当前 matched synthetic
    采样（persistent + replay 实验）。
    """
    if features.numel() == 0 or features.size(0) < 2:
        return None
    if trajectory_sink is not None and int(trajectory_interval) < 1:
        raise ValueError("trajectory_interval must be >= 1 when collecting trajectory")

    def capture_trajectory(step: int) -> None:
        """Capture predictions only; target GT is deliberately unavailable here."""
        if trajectory_sink is None:
            return
        comparator.eval()
        with torch.no_grad():
            current_logits = comparator(features.detach())
            current_loss = float(
                F.cross_entropy(
                    current_logits,
                    targets.detach().long().to(current_logits.device),
                ).item()
            )
            if trajectory_features is None:
                fixed_logits = torch.zeros(0, 2, dtype=torch.float32)
            else:
                fixed_logits = comparator(
                    trajectory_features.detach().to(current_logits.device)
                ).detach().float().cpu()
        trajectory_sink.append(
            {
                "step": int(step),
                "synthetic_train_loss": current_loss,
                "fixed_logits": fixed_logits,
            }
        )

    capture_trajectory(0)
    comparator.train()
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    use_memory = (
        memory_features is not None
        and memory_targets is not None
        and memory_features.size(0) >= 1
    )
    if use_memory:
        combined_features = torch.cat(
            [features, memory_features.detach().to(features.device)], dim=0
        )
        combined_targets = torch.cat(
            [targets, memory_targets.detach().to(features.device)], dim=0
        )
        n_memory = max(1, int(round(batch_size * memory_fraction)))
        n_current = max(1, batch_size - n_memory)
    total_loss = 0.0
    counted = 0
    total_steps = max(1, int(steps))
    for step in range(1, total_steps + 1):
        if use_memory:
            current_indices = torch.randint(
                0,
                features.size(0),
                (n_current,),
                generator=generator,
                device=features.device,
            )
            memory_indices = torch.randint(
                0,
                memory_features.size(0),
                (n_memory,),
                generator=generator,
                device=features.device,
            )
            indices = torch.cat(
                [current_indices, features.size(0) + memory_indices]
            )
            indices = indices[
                torch.randperm(indices.numel(), generator=generator, device=features.device)
            ]
            logits = comparator(combined_features[indices].detach())
            batch_targets = combined_targets[indices].detach().long()
        else:
            if features.size(0) <= batch_size:
                indices = torch.arange(features.size(0), device=features.device)
            else:
                indices = torch.randperm(
                    features.size(0), generator=generator, device=features.device
                )[:batch_size]
            logits = comparator(features[indices].detach())
            batch_targets = targets[indices].detach().long()
        batch_targets = batch_targets.to(logits.device)
        loss = F.cross_entropy(logits, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        counted += 1
        if trajectory_sink is not None and (
            step % int(trajectory_interval) == 0 or step == total_steps
        ):
            capture_trajectory(step)
            comparator.train()
    comparator.eval()
    return total_loss / counted if counted else None


def _stratified_binary_train_val_split(
    targets: torch.Tensor,
    *,
    val_fraction: float,
    min_val_per_direction: int,
    seed: int,
) -> Optional[dict]:
    """Deterministically split both comparator directions into train/val.

    The same number of validation rows is taken from trust-Task (0) and
    trust-CLIP (1), so the validation CE cannot be dominated by one direction.
    Returns ``None`` when either direction has fewer than two rows.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError("val_fraction must satisfy 0 < value < 1")
    if int(min_val_per_direction) < 1:
        raise ValueError("min_val_per_direction must be >= 1")
    targets = targets.detach().long()
    direction_indices = [
        torch.nonzero(targets == direction, as_tuple=False).flatten()
        for direction in (0, 1)
    ]
    if any(indices.numel() < 2 for indices in direction_indices):
        return None
    min_direction_count = min(indices.numel() for indices in direction_indices)
    val_per_direction = min(
        min_direction_count - 1,
        max(
            int(min_val_per_direction),
            int(round(min_direction_count * float(val_fraction))),
        ),
    )
    generator = torch.Generator(device=targets.device)
    generator.manual_seed(int(seed))
    train_parts = []
    val_parts = []
    for indices in direction_indices:
        permutation = torch.randperm(
            indices.numel(), generator=generator, device=targets.device
        )
        shuffled = indices[permutation]
        val_parts.append(shuffled[:val_per_direction])
        train_parts.append(shuffled[val_per_direction:])
    train_indices = torch.cat(train_parts)
    val_indices = torch.cat(val_parts)
    train_indices = train_indices[
        torch.randperm(
            train_indices.numel(), generator=generator, device=targets.device
        )
    ]
    val_indices = val_indices[
        torch.randperm(
            val_indices.numel(), generator=generator, device=targets.device
        )
    ]
    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "val_per_direction": int(val_per_direction),
    }


@torch.no_grad()
def _comparator_validation_loss(
    comparator: PairwiseConflictComparator,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    comparator.eval()
    logits = comparator(features.detach())
    return float(
        F.cross_entropy(
            logits, targets.detach().long().to(logits.device)
        ).item()
    )


@torch.no_grad()
def _real_margin_checkpoint_values(
    comparator: PairwiseConflictComparator,
    real_features: Optional[torch.Tensor],
    gate: float,
) -> tuple[float, float, float, float]:
    if real_features is None or real_features.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    comparator.eval()
    probabilities = _softmax_probabilities(comparator(real_features.detach()))
    margins = (probabilities[:, 0] - probabilities[:, 1]).abs().float().cpu()
    quantiles = torch.quantile(margins, torch.tensor([0.50, 0.90]))
    coverage = float((margins >= float(gate)).float().mean().item())
    return (
        float(margins.mean().item()),
        float(quantiles[0].item()),
        float(quantiles[1].item()),
        coverage,
    )


def train_pairwise_comparator_early_stopping(
    comparator: PairwiseConflictComparator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_steps: int,
    batch_size: int,
    seed: int,
    val_fraction: float,
    min_val_per_direction: int,
    check_interval: int,
    patience: int,
    memory_features: Optional[torch.Tensor] = None,
    memory_targets: Optional[torch.Tensor] = None,
    memory_fraction: float = 0.25,
    real_features: Optional[torch.Tensor] = None,
    gate: float = 0.20,
    cycle: int = 1,
    log_fn: Callable[[str], None] = logging.info,
) -> dict:
    """GT-free max-update training with synthetic-validation early stopping.

    Validation CE is the only stopping signal. Real-conflict margins are
    logged at checkpoints for diagnostics and never affect model selection.
    The best model, optimizer and RNG states are restored because the
    comparator and Adam optimizer persist across cycles.
    """
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")
    if int(check_interval) < 1:
        raise ValueError("check_interval must be >= 1")
    if int(patience) < 1:
        raise ValueError("patience must be >= 1")
    if not 0.0 <= float(memory_fraction) < 1.0:
        raise ValueError("memory_fraction must satisfy 0 <= value < 1")
    split = _stratified_binary_train_val_split(
        targets,
        val_fraction=val_fraction,
        min_val_per_direction=min_val_per_direction,
        seed=seed,
    )
    if split is None:
        comparator.eval()
        log_fn(
            "DUET comparator early-stop skipped: cycle={}; reason="
            "insufficient_samples_for_stratified_split; current_samples={}; "
            "ground_truth_affects_training=False".format(
                cycle, features.size(0)
            )
        )
        return {
            "train_loss": None,
            "best_val_loss": None,
            "best_step": 0,
            "optimizer_steps": 0,
            "stopped_early": False,
            "train_samples": 0,
            "val_samples": 0,
            "val_per_direction": 0,
            "memory_samples_per_step": 0,
        }

    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    train_features = features[train_indices].detach()
    train_targets = targets[train_indices].detach()
    val_features = features[val_indices].detach()
    val_targets = targets[val_indices].detach()
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed) + 100003)
    use_memory = (
        memory_fraction > 0.0
        and memory_features is not None
        and memory_targets is not None
        and memory_features.size(0) >= 1
    )
    if use_memory:
        memory_features = memory_features.detach().to(features.device)
        memory_targets = memory_targets.detach().to(features.device)
        n_memory = max(1, int(round(batch_size * memory_fraction)))
        n_current = max(1, int(batch_size) - n_memory)
    else:
        n_memory = 0
        n_current = int(batch_size)

    best_val_loss = _comparator_validation_loss(
        comparator, val_features, val_targets
    )
    best_step = 0
    best_model_state = {
        name: value.detach().clone()
        for name, value in comparator.state_dict().items()
    }
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_cpu_rng_state = torch.get_rng_state().clone()
    best_cuda_rng_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    stale_checks = 0
    total_train_loss = 0.0
    interval_train_loss = 0.0
    interval_steps = 0
    optimizer_steps = 0
    stopped_early = False

    def log_checkpoint(step: int, train_loss: Optional[float]) -> None:
        real_mean, real_p50, real_p90, real_coverage = (
            _real_margin_checkpoint_values(
                comparator, real_features, gate
            )
        )
        train_loss_str = (
            "none" if train_loss is None else "{:.6f}".format(train_loss)
        )
        log_fn(
            "DUET comparator early-stop checkpoint: cycle={}; step={}; "
            "train_loss={}; val_loss={:.6f}; best_val_loss={:.6f}; "
            "best_step={}; stale_checks={}; real_margin_mean={:.4f}; "
            "real_margin_p50={:.4f}; real_margin_p90={:.4f}; "
            "coverage_at_gate={:.2f}%; gate={:.2f}; "
            "ground_truth_affects_training=False".format(
                cycle,
                step,
                train_loss_str,
                current_val_loss,
                best_val_loss,
                best_step,
                stale_checks,
                real_mean,
                real_p50,
                real_p90,
                100.0 * real_coverage,
                gate,
            )
        )

    current_val_loss = best_val_loss
    log_checkpoint(0, None)
    comparator.train()
    for step in range(1, int(max_steps) + 1):
        if use_memory:
            current_indices = torch.randint(
                0,
                train_features.size(0),
                (n_current,),
                generator=generator,
                device=features.device,
            )
            memory_indices = torch.randint(
                0,
                memory_features.size(0),
                (n_memory,),
                generator=generator,
                device=features.device,
            )
            batch_features = torch.cat(
                [train_features[current_indices], memory_features[memory_indices]],
                dim=0,
            )
            batch_targets = torch.cat(
                [train_targets[current_indices], memory_targets[memory_indices]],
                dim=0,
            )
            batch_permutation = torch.randperm(
                batch_features.size(0),
                generator=generator,
                device=features.device,
            )
            batch_features = batch_features[batch_permutation]
            batch_targets = batch_targets[batch_permutation]
        else:
            if train_features.size(0) <= batch_size:
                current_indices = torch.arange(
                    train_features.size(0), device=features.device
                )
            else:
                current_indices = torch.randperm(
                    train_features.size(0),
                    generator=generator,
                    device=features.device,
                )[:batch_size]
            batch_features = train_features[current_indices]
            batch_targets = train_targets[current_indices]
        comparator.train()
        logits = comparator(batch_features.detach())
        loss = F.cross_entropy(
            logits, batch_targets.detach().long().to(logits.device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        total_train_loss += loss_value
        interval_train_loss += loss_value
        interval_steps += 1
        optimizer_steps = step

        should_check = (
            step % int(check_interval) == 0 or step == int(max_steps)
        )
        if not should_check:
            continue
        current_val_loss = _comparator_validation_loss(
            comparator, val_features, val_targets
        )
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_step = step
            best_model_state = {
                name: value.detach().clone()
                for name, value in comparator.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_cpu_rng_state = torch.get_rng_state().clone()
            best_cuda_rng_states = (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            )
            stale_checks = 0
        else:
            stale_checks += 1
        mean_interval_loss = interval_train_loss / max(interval_steps, 1)
        log_checkpoint(step, mean_interval_loss)
        interval_train_loss = 0.0
        interval_steps = 0
        if stale_checks >= int(patience):
            stopped_early = step < int(max_steps)
            break
        comparator.train()

    comparator.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    torch.set_rng_state(best_cpu_rng_state)
    if best_cuda_rng_states is not None:
        torch.cuda.set_rng_state_all(best_cuda_rng_states)
    comparator.eval()
    return {
        "train_loss": (
            total_train_loss / optimizer_steps if optimizer_steps else None
        ),
        "best_val_loss": best_val_loss,
        "best_step": int(best_step),
        "optimizer_steps": int(optimizer_steps),
        "stopped_early": bool(stopped_early),
        "train_samples": int(train_indices.numel()),
        "val_samples": int(val_indices.numel()),
        "val_per_direction": int(split["val_per_direction"]),
        "memory_samples_per_step": int(n_memory),
    }


def train_pairwise_comparator_epochs(
    comparator: PairwiseConflictComparator,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    memory_features: Optional[torch.Tensor] = None,
    memory_targets: Optional[torch.Tensor] = None,
    memory_fraction: float = 0.25,
) -> Optional[float]:
    """Full-batch epoch training for the pairwise comparator.

    Each epoch contains every current matched-synthetic sample exactly once,
    plus replay samples drawn to match ``memory_fraction``, and performs one
    optimizer update.  Consequently ``epochs`` always means exactly
    ``epochs`` optimizer updates, independent of the current sample count.

    ``batch_size`` is retained for call-site compatibility but intentionally
    does not split the current samples into mini-batches.
    """
    if not 0.0 <= float(memory_fraction) < 1.0:
        raise ValueError("memory_fraction must satisfy 0 <= value < 1")
    if features.numel() == 0 or features.size(0) < 2:
        return None
    comparator.train()
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    device = features.device
    use_memory = (
        memory_fraction > 0.0
        and memory_features is not None
        and memory_targets is not None
        and memory_features.size(0) >= 1
    )
    n_memory_per_epoch = (
        max(
            1,
            int(
                round(
                    features.size(0)
                    * memory_fraction
                    / (1.0 - memory_fraction)
                )
            ),
        )
        if use_memory
        else 0
    )
    total_loss = 0.0
    counted = 0
    for _ in range(max(1, int(epochs))):
        permutation = torch.randperm(
            features.size(0), generator=generator, device=device
        )
        current_features = features[permutation].detach()
        current_targets = targets[permutation].detach()
        if use_memory:
            memory_indices = torch.randint(
                0,
                memory_features.size(0),
                (n_memory_per_epoch,),
                generator=generator,
                device=device,
            )
            batch_features = torch.cat(
                [current_features, memory_features[memory_indices].detach()],
                dim=0,
            )
            batch_targets = torch.cat(
                [current_targets, memory_targets[memory_indices].detach()],
                dim=0,
            )
        else:
            batch_features = current_features
            batch_targets = current_targets
        logits = comparator(batch_features)
        batch_targets = batch_targets.to(logits.device)
        loss = F.cross_entropy(logits, batch_targets.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        counted += 1
    comparator.eval()
    return total_loss / counted if counted else None


def apply_pairwise_decision(
    logits: torch.Tensor,
    task_top1: torch.Tensor,
    clip_top1: torch.Tensor,
    *,
    gate: float,
    coverage_fraction: float = 0.0,
) -> dict:
    """2-way decision with an absolute-margin or rank-coverage gate.

    ``coverage_fraction > 0`` selects exactly the highest-margin fraction of
    the current fixed conflict cohort and ignores the absolute gate.  This is
    label-free and avoids coverage drift when comparator logits become more
    confident across cycles.
    """
    if not 0.0 <= float(coverage_fraction) <= 1.0:
        raise ValueError("coverage_fraction must satisfy 0 <= value <= 1")
    probabilities = _softmax_probabilities(logits)
    trust_task = probabilities[:, 0]
    trust_clip = probabilities[:, 1]
    margin = (trust_task - trust_clip).abs()
    if float(coverage_fraction) > 0.0 and margin.numel() > 0:
        selected_count = max(
            1,
            min(
                margin.numel(),
                int(round(margin.numel() * float(coverage_fraction))),
            ),
        )
        order = torch.argsort(margin, descending=True, stable=True)
        resolved = torch.zeros_like(margin, dtype=torch.bool)
        resolved[order[:selected_count]] = True
        selection_mode = "rank_coverage"
    else:
        resolved = margin >= gate
        selected_count = int(resolved.sum().item())
        selection_mode = "absolute_margin"
    chosen = torch.where(trust_task >= trust_clip, task_top1, clip_top1)
    return {
        "resolved": resolved,
        "chosen": chosen,
        "trust_task": trust_task,
        "trust_clip": trust_clip,
        "confidence": torch.maximum(trust_task, trust_clip),
        "margin": margin,
        "probabilities": probabilities,
        "selection_mode": selection_mode,
        "selected_count": int(selected_count),
    }


def apply_decision_rules(
    context_probs: torch.Tensor,
    task_top1: torch.Tensor,
    clip_top1: torch.Tensor,
    *,
    accept_conf: float,
    accept_margin: float,
    allow_third_class: bool,
    abstain_when_uncertain: bool,
    third_class_conf: float,
    third_class_margin: float,
) -> dict:
    """Strict-conflict decision: support Task / support CLIP / third / abstain."""
    probabilities = context_probs.float()
    context_conf, context_top1 = probabilities.max(dim=1)
    sorted_probs, _ = probabilities.sort(dim=1, descending=True)
    context_margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    is_task = context_top1 == task_top1
    is_clip = context_top1 == clip_top1
    is_third = ~(is_task | is_clip)
    if abstain_when_uncertain:
        resolved = (is_task | is_clip) & (
            context_conf >= accept_conf
        ) & (
            context_margin >= accept_margin
        )
        third_resolved = (
            is_third
            & allow_third_class
            & (context_conf >= third_class_conf)
            & (context_margin >= third_class_margin)
        )
    else:
        resolved = (is_task | is_clip) | (is_third & allow_third_class)
        third_resolved = is_third & allow_third_class
    resolved = resolved | third_resolved
    return {
        "resolved": resolved,
        "context_top1": context_top1,
        "context_conf": context_conf,
        "context_margin": context_margin,
        "is_task": is_task,
        "is_clip": is_clip,
        "is_third": is_third,
        "third_resolved": third_resolved,
    }


def apply_weak_verification(
    context_probs: torch.Tensor,
    common_label: torch.Tensor,
    *,
    weak_accept_conf: float,
    weak_accept_margin: float,
    abstain_when_uncertain: bool,
) -> dict:
    """Weak-agreement verification: keep the common Top-1 or defer."""
    probabilities = context_probs.float()
    context_conf, context_top1 = probabilities.max(dim=1)
    sorted_probs, _ = probabilities.sort(dim=1, descending=True)
    context_margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    agrees_with_common = context_top1 == common_label
    if abstain_when_uncertain:
        passed = agrees_with_common & (
            context_conf >= weak_accept_conf
        ) & (
            context_margin >= weak_accept_margin
        )
    else:
        passed = agrees_with_common
    return {
        "passed": passed,
        "context_top1": context_top1,
        "context_conf": context_conf,
        "context_margin": context_margin,
        "agrees_with_common": agrees_with_common,
    }


def train_context_transformer(
    transformer: DuetContextConflictTransformer,
    optimizer: torch.optim.Optimizer,
    anchor_features: torch.Tensor,
    anchor_labels: torch.Tensor,
    anchor_valid_mask: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> Optional[float]:
    """Leave-one-out anchor classification.

    Each pseudo-query is one anchor; its own anchor token is excluded from the
    context.  Labels are only high-confidence hard pseudo-labels.  Gradients
    never reach netF/netB/netC/CLIP because all inputs are detached.
    """
    valid_positions = torch.nonzero(anchor_valid_mask, as_tuple=False).flatten()
    if valid_positions.numel() < 2:
        return None
    transformer.train()
    generator = torch.Generator(device=anchor_features.device)
    generator.manual_seed(int(seed))
    total_loss = 0.0
    counted = 0
    for _ in range(max(1, int(steps))):
        if valid_positions.numel() <= batch_size:
            query_positions = valid_positions
        else:
            perm = torch.randperm(
                valid_positions.numel(), generator=generator, device=anchor_features.device
            )[:batch_size]
            query_positions = valid_positions[perm]
        query_features = anchor_features[query_positions].detach()
        pseudo_targets = anchor_labels[query_positions].detach().long()
        exclude = torch.zeros(
            query_positions.numel(),
            anchor_features.size(0),
            dtype=torch.bool,
            device=anchor_features.device,
        )
        exclude[
            torch.arange(query_positions.numel(), device=anchor_features.device),
            query_positions,
        ] = True
        output = transformer(
            query_features=query_features,
            anchor_features=anchor_features,
            anchor_labels=anchor_labels,
            anchor_valid_mask=anchor_valid_mask,
            anchor_self_exclude=exclude,
        )
        loss = F.cross_entropy(output["logits"], pseudo_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
        counted += 1
    transformer.eval()
    if counted == 0:
        return None
    return total_loss / counted


def _exclude_query_anchors(
    query_sample_indices: Optional[torch.Tensor],
    anchor_sample_indices: Optional[torch.Tensor],
    batch: int,
    num_anchors: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build [B, A] mask that hides a query's own image from its context."""
    if query_sample_indices is None or anchor_sample_indices is None:
        return None
    query_ids = query_sample_indices.long().to(device)
    anchor_ids = anchor_sample_indices.long().to(device)
    exclude = torch.zeros(batch, num_anchors, dtype=torch.bool, device=device)
    for row in range(batch):
        exclude[row] = anchor_ids == query_ids[row]
    return exclude


def run_context_refinement(
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    task_features: torch.Tensor,
    *,
    num_classes: int,
    context_cfg: object,
    pre_prior_task_probs: Optional[torch.Tensor] = None,
    pre_prior_clip_probs: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    sample_indices: Optional[torch.Tensor] = None,
    transformer: Optional[DuetContextConflictTransformer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    clip_features: Optional[torch.Tensor] = None,
    strong_task_probs: Optional[torch.Tensor] = None,
    strong_clip_probs: Optional[torch.Tensor] = None,
    strong_task_features: Optional[torch.Tensor] = None,
    strong_clip_features: Optional[torch.Tensor] = None,
    comparator: Optional[PairwiseConflictComparator] = None,
    comparator_optimizer: Optional[torch.optim.Optimizer] = None,
    replay_memory: Optional[ComparatorReplayMemory] = None,
    cycle: int = 1,
    log_fn: Callable[[str], None] = logging.info,
) -> dict:
    """Run the whole context pipeline and return per-sample decisions + stats.

    All returned masks have shape [N] (or [N, C] for refined targets) and are
    aligned with the input ``task_probs`` rows.

    ``context_cfg`` is the DUET_CONTEXT config block (or any object exposing
    the same attributes); all thresholds / switches / training hyper-params
    are read from it so the call site stays short.
    """
    if str(context_cfg.REFINER_TYPE) == "comparator":
        # pairwise conflict-resolution 是独立管线，走专用入口。
        return run_comparator_refinement(
            task_probs=task_probs,
            clip_probs=clip_probs,
            task_features=task_features,
            num_classes=num_classes,
            context_cfg=context_cfg,
            clip_features=clip_features,
            strong_task_probs=strong_task_probs,
            strong_clip_probs=strong_clip_probs,
            strong_task_features=strong_task_features,
            strong_clip_features=strong_clip_features,
            pre_prior_task_probs=pre_prior_task_probs,
            pre_prior_clip_probs=pre_prior_clip_probs,
            labels=labels,
            sample_indices=sample_indices,
            comparator=comparator,
            comparator_optimizer=comparator_optimizer,
            replay_memory=replay_memory,
            cycle=cycle,
            log_fn=log_fn,
        )
    use_strict_conflict = bool(context_cfg.USE_STRICT_CONFLICT)
    use_weak_agreement = bool(context_cfg.USE_WEAK_AGREEMENT)
    anchors_per_class = int(context_cfg.ANCHORS_PER_CLASS)
    anchor_task_conf = float(context_cfg.ANCHOR_TASK_CONF)
    anchor_clip_conf = float(context_cfg.ANCHOR_CLIP_CONF)
    anchor_task_entropy = float(context_cfg.ANCHOR_TASK_ENTROPY)
    anchor_clip_entropy = float(context_cfg.ANCHOR_CLIP_ENTROPY)
    entropy_weight = float(context_cfg.ENTROPY_WEIGHT)
    require_pre_post_prior_agreement = bool(
        context_cfg.REQUIRE_PRE_POST_PRIOR_AGREEMENT
    )
    weak_conf_threshold = float(context_cfg.WEAK_CONF_THRESHOLD)
    weak_entropy_threshold = float(context_cfg.WEAK_ENTROPY_THRESHOLD)
    accept_conf = float(context_cfg.ACCEPT_CONF)
    accept_margin = float(context_cfg.ACCEPT_MARGIN)
    weak_accept_conf = float(context_cfg.WEAK_ACCEPT_CONF)
    weak_accept_margin = float(context_cfg.WEAK_ACCEPT_MARGIN)
    third_class_conf = float(context_cfg.THIRD_CLASS_CONF)
    third_class_margin = float(context_cfg.THIRD_CLASS_MARGIN)
    allow_third_class = bool(context_cfg.ALLOW_THIRD_CLASS)
    abstain_when_uncertain = bool(context_cfg.ABSTAIN_WHEN_UNCERTAIN)
    refiner_type = str(context_cfg.REFINER_TYPE)
    train_steps_per_cycle = int(context_cfg.TRAIN_STEPS_PER_CYCLE)
    train_batch_size = int(context_cfg.TRAIN_BATCH_SIZE)
    knn_k = int(getattr(context_cfg, "KNN_K", 5))
    seed = int(context_cfg.SEED) + int(cycle - 1)
    eval_only_logging = bool(context_cfg.EVAL_ONLY_LOGGING)
    task_probs = task_probs.float()
    clip_probs = clip_probs.float()
    num_samples = task_probs.size(0)
    # 训练时 features 在 CPU 收集；Transformer / 对照基线在 GPU 上执行。
    device = task_features.device
    if transformer is not None:
        device = next(transformer.parameters()).device
    task_conf, task_top1 = task_probs.max(dim=1)
    clip_conf, clip_top1 = clip_probs.max(dim=1)
    task_entropy = _entropy(task_probs)
    clip_entropy = _entropy(clip_probs)

    strict_conflict_mask = task_top1 != clip_top1
    # 弱一致性样本
    weak_agreement_mask = (
        (task_top1 == clip_top1)
        & (
            (task_conf < weak_conf_threshold)
            | (clip_conf < weak_conf_threshold)
            | (task_entropy > weak_entropy_threshold)
            | (clip_entropy > weak_entropy_threshold)
        )
    )
    # 高一致性样本
    anchor_mask = (
        (task_top1 == clip_top1)
        & (task_conf >= anchor_task_conf)
        & (clip_conf >= anchor_clip_conf)
        & (task_entropy <= anchor_task_entropy)
        & (clip_entropy <= anchor_clip_entropy)
    ) # 高一致性样本
    # if (
    #     require_pre_post_prior_agreement
    #     and pre_prior_task_probs is not None
    #     and pre_prior_clip_probs is not None
    # ):
    #     pre_task_top1 = pre_prior_task_probs.float().argmax(dim=1)
    #     pre_clip_top1 = pre_prior_clip_probs.float().argmax(dim=1)
    #     pre_agree = pre_task_top1 == pre_clip_top1
    #     same_common = pre_task_top1 == task_top1
    #     anchor_mask = anchor_mask & pre_agree & same_common

    query_mask = torch.zeros(num_samples, dtype=torch.bool)
    if use_strict_conflict:
        query_mask = query_mask | strict_conflict_mask
    if use_weak_agreement:
        query_mask = query_mask | weak_agreement_mask

    resolved_mask = torch.zeros(num_samples, dtype=torch.bool)
    weak_rejected_mask = torch.zeros(num_samples, dtype=torch.bool)
    context_labels = torch.full((num_samples,), -1, dtype=torch.long)
    refined_targets = clip_probs.detach().clone()

    query_count = int(query_mask.sum().item())
    anchor_count = int(anchor_mask.sum().item())
    stats = {
        "post_prior_agreement": int((task_top1 == clip_top1).sum().item()),
        "strict_conflicts": int(strict_conflict_mask.sum().item()),
        "weak_agreement": int(weak_agreement_mask.sum().item()),
        "query_count": query_count,
        "anchor_count": anchor_count,
        "anchor_bank_total": 0,
        "anchor_per_class_counts": [],
        "anchor_mean_task_conf": float("nan"),
        "anchor_mean_clip_conf": float("nan"),
        "train_loss": None,
        "resolved_strict": 0,
        "support_task": 0,
        "support_clip": 0,
        "third_class": 0,
        "abstain": 0,
        "weak_passed": 0,
        "weak_deferred": (
            int(weak_agreement_mask.sum().item()) if use_weak_agreement else 0
        ),
        "context_mean_conf": float("nan"),
        "context_mean_margin": float("nan"),
    }

    if query_count > 0 and anchor_count >= 2:
        anchor_bank = ClassBalancedAnchorBank(
            num_classes=num_classes,
            anchors_per_class=anchors_per_class,
            feature_dim=task_features.size(1),
            seed=seed,
            device=device,
        )
        reliability = (
            task_conf[anchor_mask]
            + clip_conf[anchor_mask]
            - entropy_weight * (task_entropy[anchor_mask] + clip_entropy[anchor_mask])
        )
        anchor_indices_for_bank = (
            sample_indices[anchor_mask] if sample_indices is not None else None
        )
        anchor_bank.update(
            features=task_features[anchor_mask].detach(),
            labels=task_top1[anchor_mask].detach().long(),
            scores=reliability.detach(),
            sample_indices=anchor_indices_for_bank,
        )
        (
            anchor_features,
            anchor_labels,
            _anchor_scores,
            anchor_sample_indices,
            anchor_valid,
        ) = anchor_bank.flatten()
        anchor_features = anchor_features.detach().to(device)
        anchor_labels = anchor_labels.detach().long().to(device)
        anchor_valid = anchor_valid.detach().bool().to(device)
        if anchor_sample_indices is not None:
            anchor_sample_indices = anchor_sample_indices.detach().long().to(device)
        stats["anchor_per_class_counts"] = anchor_bank.per_class_counts().tolist()
        stats["anchor_bank_total"] = int(
            anchor_bank.per_class_counts().sum().item()
        )
        stats["anchor_mean_task_conf"] = float(task_conf[anchor_mask].mean().item())
        stats["anchor_mean_clip_conf"] = float(clip_conf[anchor_mask].mean().item())

        if refiner_type == "transformer":
            if transformer is None:
                raise ValueError("refiner_type=transformer requires a transformer module")
            if train_steps_per_cycle > 0 and optimizer is not None:
                stats["train_loss"] = train_context_transformer(
                    transformer,
                    optimizer,
                    anchor_features,
                    anchor_labels,
                    anchor_valid,
                    steps=train_steps_per_cycle,
                    batch_size=train_batch_size,
                    seed=seed,
                )

        query_features = task_features[query_mask].detach().to(device)
        query_sample_indices = (
            sample_indices[query_mask] if sample_indices is not None else None
        )
        if refiner_type == "transformer":
            transformer.eval()
            with torch.no_grad():
                self_exclude = _exclude_query_anchors(
                    query_sample_indices,
                    anchor_sample_indices,
                    query_features.size(0),
                    anchor_features.size(0),
                    device,
                )
                context_probs = transformer(
                    query_features=query_features,
                    anchor_features=anchor_features,
                    anchor_labels=anchor_labels,
                    anchor_valid_mask=anchor_valid,
                    anchor_self_exclude=self_exclude,
                )["probabilities"] # type: ignore
        elif refiner_type == "cosine_knn":
            context_probs = cosine_knn_refine(
                query_features,
                anchor_features,
                anchor_labels,
                anchor_valid,
                num_classes,
                k=knn_k,
            )
        elif refiner_type == "prototype":
            context_probs = prototype_refine(
                query_features,
                anchor_features,
                anchor_labels,
                anchor_valid,
                num_classes,
            )
        else:
            raise ValueError(
                "unknown DUET_CONTEXT.REFINER_TYPE: {}".format(refiner_type)
            )
        context_probs = context_probs.float().cpu()

        query_positions = torch.nonzero(query_mask, as_tuple=False).flatten()
        strict_positions = torch.nonzero(
            strict_conflict_mask & query_mask, as_tuple=False
        ).flatten()
        weak_positions = torch.nonzero(
            weak_agreement_mask & query_mask, as_tuple=False
        ).flatten()
        position_to_query = torch.full((num_samples,), -1, dtype=torch.long)
        position_to_query[query_positions] = torch.arange(query_positions.numel())

        strict_conf = None
        strict_margin = None
        if strict_positions.numel() > 0 and use_strict_conflict:
            strict_query_rows = position_to_query[strict_positions]
            decision = apply_decision_rules(
                context_probs[strict_query_rows],
                task_top1[strict_positions],
                clip_top1[strict_positions],
                accept_conf=accept_conf,
                accept_margin=accept_margin,
                allow_third_class=allow_third_class,
                abstain_when_uncertain=abstain_when_uncertain,
                third_class_conf=third_class_conf,
                third_class_margin=third_class_margin,
            )
            resolved_strict = strict_positions[decision["resolved"]]
            resolved_mask[resolved_strict] = True
            context_labels[resolved_strict] = decision["context_top1"][
                decision["resolved"]
            ].long()
            refined_targets[resolved_strict] = context_probs[strict_query_rows][
                decision["resolved"]
            ].detach()
            stats["resolved_strict"] = int(decision["resolved"].sum().item())
            stats["support_task"] = int(
                (decision["resolved"] & decision["is_task"]).sum().item()
            )
            stats["support_clip"] = int(
                (decision["resolved"] & decision["is_clip"]).sum().item()
            )
            stats["third_class"] = int(decision["third_resolved"].sum().item())
            stats["abstain"] = int((~decision["resolved"]).sum().item())
            strict_conf = decision["context_conf"]
            strict_margin = decision["context_margin"]

        weak_conf = None
        weak_margin = None
        if weak_positions.numel() > 0 and use_weak_agreement:
            weak_query_rows = position_to_query[weak_positions]
            verification = apply_weak_verification(
                context_probs[weak_query_rows],
                task_top1[weak_positions],  # common label (task == clip)
                weak_accept_conf=weak_accept_conf,
                weak_accept_margin=weak_accept_margin,
                abstain_when_uncertain=abstain_when_uncertain,
            )
            deferred = weak_positions[~verification["passed"]]
            weak_rejected_mask[deferred] = True
            stats["weak_passed"] = int(verification["passed"].sum().item())
            stats["weak_deferred"] = int((~verification["passed"]).sum().item())
            weak_conf = verification["context_conf"]
            weak_margin = verification["context_margin"]

        all_conf_parts = []
        all_margin_parts = []
        if strict_conf is not None:
            all_conf_parts.append(strict_conf)
            all_margin_parts.append(strict_margin)
        if weak_conf is not None:
            all_conf_parts.append(weak_conf)
            all_margin_parts.append(weak_margin)
        if all_conf_parts:
            combined_conf = torch.cat(all_conf_parts)
            combined_margin = torch.cat(all_margin_parts)
            stats["context_mean_conf"] = float(combined_conf.mean().item())
            stats["context_mean_margin"] = float(combined_margin.mean().item())
    else:
        stats["abstain"] = (
            int(strict_conflict_mask.sum().item()) if use_strict_conflict else 0
        )
        stats["weak_deferred"] = (
            int(weak_agreement_mask.sum().item()) if use_weak_agreement else 0
        )

    # ---- 修正统计（不依赖 target ground-truth）----
    strict_total = (
        int(strict_conflict_mask.sum().item()) if use_strict_conflict else 0
    )
    weak_total = (
        int(weak_agreement_mask.sum().item()) if use_weak_agreement else 0
    )
    stats["resolved_rate_pct"] = (
        100.0 * stats["resolved_strict"] / strict_total if strict_total > 0 else 0.0
    )
    stats["weak_defer_rate_pct"] = (
        100.0 * stats["weak_deferred"] / weak_total if weak_total > 0 else 0.0
    )
    stats["final_admitted"] = (
        stats["post_prior_agreement"]
        - stats["weak_deferred"]
        + stats["resolved_strict"]
    )
    stats["admitted_delta"] = (
        stats["resolved_strict"] - stats["weak_deferred"]
    )
    _log_correction_stats(stats, cycle, log_fn)

    _log_context_stats(stats, refiner_type, cycle, log_fn)
    if eval_only_logging and labels is not None:
        _log_eval_only_metrics(
            stats,
            resolved_mask=resolved_mask,
            weak_rejected_mask=weak_rejected_mask,
            context_labels=context_labels,
            task_top1=task_top1,
            clip_top1=clip_top1,
            all_label=labels,
            anchor_mask=anchor_mask,
            weak_agreement_mask=weak_agreement_mask,
            strict_conflict_mask=strict_conflict_mask,
            cycle=cycle,
            log_fn=log_fn,
        )
    return {
        "strict_conflict_mask": strict_conflict_mask,
        "weak_agreement_mask": weak_agreement_mask,
        "query_mask": query_mask,
        "anchor_mask": anchor_mask,
        "resolved_mask": resolved_mask,
        "weak_rejected_mask": weak_rejected_mask,
        "context_labels": context_labels,
        "refined_targets": refined_targets,
        "stats": stats,
    }


def run_comparator_refinement(
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    task_features: torch.Tensor,
    *,
    num_classes: int,
    context_cfg: object,
    clip_features: Optional[torch.Tensor],
    pre_prior_task_probs: Optional[torch.Tensor] = None,
    pre_prior_clip_probs: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    sample_indices: Optional[torch.Tensor] = None,
    strong_task_probs: Optional[torch.Tensor] = None,
    strong_clip_probs: Optional[torch.Tensor] = None,
    strong_task_features: Optional[torch.Tensor] = None,
    strong_clip_features: Optional[torch.Tensor] = None,
    comparator: Optional[PairwiseConflictComparator] = None,
    comparator_optimizer: Optional[torch.optim.Optimizer] = None,
    replay_memory: Optional[ComparatorReplayMemory] = None,
    cycle: int = 1,
    log_fn: Callable[[str], None] = logging.info,
) -> dict:
    """Pairwise conflict-resolution 专用管线（REFINER_TYPE=comparator）。

    只处理 strict conflict：对每个冲突 (Task=A, CLIP=B) 构造 class-agnostic
    的相对证据特征，用 2-way comparator 输出 trust Task / trust CLIP，
    再按 |trust_T - trust_C| >= gate 决定 resolved / abstain。
    训练用 anchor 池造两类 synthetic conflict（Task 侧错 / CLIP 侧错），
    对手类 B 优先取 strong 增广真实 flip 的类别，fallback 到该分支 runner-up，
    且都要求 hard-gate（MIN_RUNNER_PROB / MAX_TOP1_MARGIN）。
    """
    use_strict_conflict = bool(context_cfg.USE_STRICT_CONFLICT)
    anchors_per_class = int(context_cfg.ANCHORS_PER_CLASS)
    anchor_task_conf = float(context_cfg.ANCHOR_TASK_CONF)
    anchor_clip_conf = float(context_cfg.ANCHOR_CLIP_CONF)
    anchor_task_entropy = float(context_cfg.ANCHOR_TASK_ENTROPY)
    anchor_clip_entropy = float(context_cfg.ANCHOR_CLIP_ENTROPY)
    entropy_weight = float(context_cfg.ENTROPY_WEIGHT)
    require_pre_post_prior_agreement = bool(
        context_cfg.REQUIRE_PRE_POST_PRIOR_AGREEMENT
    )
    sim_topk = int(getattr(context_cfg, "SIM_TOPK", 3))
    min_runner_prob = float(getattr(context_cfg, "MIN_RUNNER_PROB", 0.10))
    max_top1_margin = float(getattr(context_cfg, "MAX_TOP1_MARGIN", 0.60))
    gate = float(getattr(context_cfg, "COMPARATOR_GATE", 0.20))
    coverage_fraction = float(
        getattr(context_cfg, "COMPARATOR_COVERAGE_FRACTION", 0.0)
    )
    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError(
            "COMPARATOR_COVERAGE_FRACTION must satisfy 0 <= value <= 1"
        )
    dist_match_synthetic = bool(
        getattr(context_cfg, "DIST_MATCH_SYNTHETIC", False)
    )
    dist_match_z_max = float(getattr(context_cfg, "DIST_MATCH_Z_MAX", 1.5))
    dist_match_dims = [
        int(v) for v in getattr(context_cfg, "DIST_MATCH_DIMS", [0, 1, 2, 3, 4, 5, 6, 7])
    ]
    min_dist_match_kept = int(getattr(context_cfg, "MIN_DIST_MATCH_KEPT", 16))
    train_steps_per_cycle = int(context_cfg.TRAIN_STEPS_PER_CYCLE)
    train_batch_size = int(context_cfg.TRAIN_BATCH_SIZE)
    replay_mix_fraction = float(
        getattr(context_cfg, "REPLAY_MIX_FRACTION", 0.25)
    )
    if not 0.0 <= replay_mix_fraction < 1.0:
        raise ValueError("REPLAY_MIX_FRACTION must satisfy 0 <= value < 1")
    comparator_epochs = int(getattr(context_cfg, "COMPARATOR_EPOCHS", 0))
    early_stop_enabled = bool(
        getattr(context_cfg, "EARLY_STOP_ENABLED", False)
    )
    early_stop_val_fraction = float(
        getattr(context_cfg, "EARLY_STOP_VAL_FRACTION", 0.20)
    )
    early_stop_min_val_per_direction = int(
        getattr(context_cfg, "EARLY_STOP_MIN_VAL_PER_DIRECTION", 6)
    )
    early_stop_check_interval = int(
        getattr(context_cfg, "EARLY_STOP_CHECK_INTERVAL", 10)
    )
    early_stop_patience = int(
        getattr(context_cfg, "EARLY_STOP_PATIENCE", 3)
    )
    trajectory_enabled = bool(
        getattr(context_cfg, "EVAL_TRAJECTORY_ENABLED", False)
    )
    trajectory_interval = int(
        getattr(context_cfg, "EVAL_TRAJECTORY_INTERVAL", 10)
    )
    trajectory_coverages = [
        int(value)
        for value in getattr(
            context_cfg, "EVAL_TRAJECTORY_COVERAGES", [10, 20, 40, 60, 80]
        )
    ]
    if trajectory_enabled and trajectory_interval < 1:
        raise ValueError("EVAL_TRAJECTORY_INTERVAL must be >= 1")
    if any(value <= 0 or value > 100 for value in trajectory_coverages):
        raise ValueError("EVAL_TRAJECTORY_COVERAGES values must be in [1, 100]")
    if early_stop_enabled and comparator_epochs > 0:
        raise ValueError(
            "EARLY_STOP_ENABLED and COMPARATOR_EPOCHS cannot both be active"
        )
    seed = int(context_cfg.SEED) + int(cycle - 1)
    eval_only_logging = bool(context_cfg.EVAL_ONLY_LOGGING)

    task_probs = task_probs.float()
    clip_probs = clip_probs.float()
    num_samples = task_probs.size(0)
    device = task_features.device
    if comparator is not None:
        device = next(comparator.parameters()).device
    task_conf, task_top1 = task_probs.max(dim=1)
    clip_conf, clip_top1 = clip_probs.max(dim=1)
    task_entropy = _entropy(task_probs)
    clip_entropy = _entropy(clip_probs)

    strict_conflict_mask = task_top1 != clip_top1
    weak_agreement_mask = torch.zeros(num_samples, dtype=torch.bool)
    anchor_mask = (
        (task_top1 == clip_top1)
        & (task_conf >= anchor_task_conf)
        & (clip_conf >= anchor_clip_conf)
        & (task_entropy <= anchor_task_entropy)
        & (clip_entropy <= anchor_clip_entropy)
    )
    if (
        require_pre_post_prior_agreement
        and pre_prior_task_probs is not None
        and pre_prior_clip_probs is not None
    ):
        pre_task_top1 = pre_prior_task_probs.float().argmax(dim=1)
        pre_clip_top1 = pre_prior_clip_probs.float().argmax(dim=1)
        pre_agree = pre_task_top1 == pre_clip_top1
        same_common = pre_task_top1 == task_top1
        anchor_mask = anchor_mask & pre_agree & same_common

    query_mask = strict_conflict_mask.clone() if use_strict_conflict else torch.zeros(
        num_samples, dtype=torch.bool
    )
    resolved_mask = torch.zeros(num_samples, dtype=torch.bool)
    weak_rejected_mask = torch.zeros(num_samples, dtype=torch.bool)
    context_labels = torch.full((num_samples,), -1, dtype=torch.long)
    refined_targets = clip_probs.detach().clone()

    strict_total = int(strict_conflict_mask.sum().item())
    anchor_count = int(anchor_mask.sum().item())
    stats = {
        "post_prior_agreement": int((task_top1 == clip_top1).sum().item()),
        "strict_conflicts": strict_total,
        "weak_agreement": 0,
        "query_count": int(query_mask.sum().item()),
        "anchor_count": anchor_count,
        "anchor_bank_total": 0,
        "anchor_per_class_counts": [],
        "anchor_mean_task_conf": float("nan"),
        "anchor_mean_clip_conf": float("nan"),
        "train_loss": None,
        "train_current_samples": 0,
        "train_memory_samples": 0,
        "optimizer_steps_this_cycle": 0,
        "final_current_loss": None,
        "early_stop_best_val_loss": None,
        "early_stop_best_step": None,
        "early_stop_stopped": False,
        "resolved_strict": 0,
        "support_task": 0,
        "support_clip": 0,
        "third_class": 0,
        "abstain": strict_total,
        "weak_passed": 0,
        "weak_deferred": 0,
        "context_mean_conf": float("nan"),
        "context_mean_margin": float("nan"),
    }

    if clip_features is None:
        raise ValueError("refiner_type=comparator requires clip_features")
    if strict_total > 0 and anchor_count >= 2:
        reliability = (
            task_conf[anchor_mask]
            + clip_conf[anchor_mask]
            - entropy_weight * (task_entropy[anchor_mask] + clip_entropy[anchor_mask])
        )
        # pool 数组按候选位置索引，bank 也存候选位置（不是全局 sample id），
        # 保证 build_synthetic_conflicts 用 anchor_indices 能正确取回 probs/特征。
        pool_ids = torch.arange(anchor_count, device=device)
        pool_task_probs = task_probs[anchor_mask].detach().to(device)
        pool_clip_probs = clip_probs[anchor_mask].detach().to(device)
        pool_task_features = task_features[anchor_mask].detach().to(device)
        pool_clip_features = clip_features[anchor_mask].detach().to(device)
        pool_labels = task_top1[anchor_mask].detach().long().to(device)
        pool_strong_task = (
            strong_task_probs[anchor_mask].detach().to(device)
            if strong_task_probs is not None
            else None
        )
        pool_strong_clip = (
            strong_clip_probs[anchor_mask].detach().to(device)
            if strong_clip_probs is not None
            else None
        )
        pool_strong_task_features = (
            strong_task_features[anchor_mask].detach().to(device)
            if strong_task_features is not None
            else None
        )
        pool_strong_clip_features = (
            strong_clip_features[anchor_mask].detach().to(device)
            if strong_clip_features is not None
            else None
        )

        task_bank = ClassBalancedAnchorBank(
            num_classes=num_classes,
            anchors_per_class=anchors_per_class,
            feature_dim=task_features.size(1),
            seed=seed,
            device=device,
        )
        task_bank.update(
            features=pool_task_features,
            labels=pool_labels,
            scores=reliability.detach().to(device),
            sample_indices=pool_ids,
        )
        clip_bank = ClassBalancedAnchorBank(
            num_classes=num_classes,
            anchors_per_class=anchors_per_class,
            feature_dim=clip_features.size(1),
            seed=seed,
            device=device,
        )
        clip_bank.update(
            features=pool_clip_features,
            labels=pool_labels,
            scores=reliability.detach().to(device),
            sample_indices=pool_ids,
        )
        stats["anchor_bank_total"] = int(task_bank.per_class_counts().sum().item())
        stats["anchor_per_class_counts"] = task_bank.per_class_counts().tolist()
        stats["anchor_mean_task_conf"] = float(task_conf[anchor_mask].mean().item())
        stats["anchor_mean_clip_conf"] = float(clip_conf[anchor_mask].mean().item())

        synthetic_features, synthetic_targets, synthetic_counts = build_synthetic_conflicts(
            pool_labels,
            pool_strong_task_probs=pool_strong_task,
            pool_strong_clip_probs=pool_strong_clip,
            pool_strong_task_features=pool_strong_task_features,
            pool_strong_clip_features=pool_strong_clip_features,
            task_bank=task_bank,
            clip_bank=clip_bank,
            min_runner_prob=min_runner_prob,
            max_top1_margin=max_top1_margin,
            sim_topk=sim_topk,
        )
        log_fn(
            "DUET comparator strong conflict counts: cycle={}; "
            "task_flip_only={}; clip_flip_only={}; both_flip={}; "
            "no_conflict={}; ground_truth_affects_training=False".format(
                cycle,
                synthetic_counts["task_flip_only"],
                synthetic_counts["clip_flip_only"],
                synthetic_counts["both_flip"],
                synthetic_counts["no_conflict"],
            )
        )
        log_fn(
            "DUET comparator synthetic conflicts: cycle={}; task_side={}; "
            "clip_side={}; total={}; "
            "ground_truth_affects_training=False".format(
                cycle,
                synthetic_counts["task_side"],
                synthetic_counts["clip_side"],
                synthetic_features.size(0),
            )
        )
        _log_pair_distribution(
            synthetic_features, "synthetic", cycle, log_fn
        )
        strict_positions = torch.nonzero(
            strict_conflict_mask & query_mask, as_tuple=False
        ).flatten()
        real_features = torch.zeros(
            0, 16, dtype=torch.float32
        )
        if strict_positions.numel() > 0:
            real_features = build_comparator_features(
                task_probs[strict_positions],
                clip_probs[strict_positions],
                task_features[strict_positions],
                clip_features[strict_positions],
                task_bank,
                clip_bank,
                class_a=task_top1[strict_positions],
                class_b=clip_top1[strict_positions],
                sim_topk=sim_topk,
            ).to(device)
            _log_pair_distribution(
                real_features.cpu(), "real-conflict", cycle, log_fn
            )
        # distribution matching：只保留“长得像真实 conflict”的 synthetic 对
        if (
            dist_match_synthetic
            and synthetic_features.size(0) > 0
            and real_features.numel() > 0
        ):
            # 按“信任方向”分别过滤再合并，避免 matching 把某一方向筛没，
            # 导致 comparator 出现 trust Task / trust CLIP 数量严重失衡。
            match_device = synthetic_features.device
            trust_task_mask = (synthetic_targets == 0.0).to(match_device)  # CLIP-error -> trust Task
            trust_clip_mask = (synthetic_targets == 1.0).to(match_device)  # Task-error -> trust CLIP
            before_trust_task = int(trust_task_mask.sum().item())
            before_trust_clip = int(trust_clip_mask.sum().item())
            keep_task, fallback_task = _zscore_filter(
                synthetic_features[trust_task_mask].cpu(),
                real_features.cpu(),
                dist_match_dims,
                dist_match_z_max,
                min_kept=min_dist_match_kept,
            )
            keep_task = keep_task.to(match_device)
            keep_clip, fallback_clip = _zscore_filter(
                synthetic_features[trust_clip_mask].cpu(),
                real_features.cpu(),
                dist_match_dims,
                dist_match_z_max,
                min_kept=min_dist_match_kept,
            )
            keep_clip = keep_clip.to(match_device)
            keep = torch.zeros(
                synthetic_features.size(0),
                dtype=torch.bool,
                device=match_device,
            )
            keep[trust_task_mask] = keep_task
            keep[trust_clip_mask] = keep_clip
            # 训练前强制两侧数量平衡：下采样多的一侧到 min 数，
            # 避免 trust Task / trust CLIP 比例失衡导致方向坍缩。
            task_kept_positions = torch.nonzero(
                keep & trust_task_mask, as_tuple=False
            ).flatten()
            clip_kept_positions = torch.nonzero(
                keep & trust_clip_mask, as_tuple=False
            ).flatten()
            balance_n = min(
                task_kept_positions.numel(), clip_kept_positions.numel()
            )
            balanced_keep = torch.zeros_like(keep)
            if balance_n > 0:
                # 固定 seed 的随机下采样，避免顺序/类别偏差
                balance_generator = torch.Generator(device=match_device)
                balance_generator.manual_seed(int(seed))
                task_perm = torch.randperm(
                    task_kept_positions.numel(),
                    generator=balance_generator,
                    device=match_device,
                )
                clip_perm = torch.randperm(
                    clip_kept_positions.numel(),
                    generator=balance_generator,
                    device=match_device,
                )
                balanced_keep[task_kept_positions[task_perm[:balance_n]]] = True
                balanced_keep[clip_kept_positions[clip_perm[:balance_n]]] = True
            keep = balanced_keep
            kept = int(keep.sum().item())
            kept_trust_task = int((keep & trust_task_mask).sum().item())
            kept_trust_clip = int((keep & trust_clip_mask).sum().item())
            log_fn(
                "DUET comparator dist-match: cycle={}; synthetic_total={}; "
                "kept={}; kept_rate={:.2f}%; z_max={:.2f}; dims={}; "
                "before_trust_task={}; before_trust_clip={}; "
                "kept_trust_task={}; kept_trust_clip={}; balanced=True; "
                "mode_task={}; mode_clip={}; min_kept={}; "
                "ground_truth_affects_training=False".format(
                    cycle,
                    synthetic_features.size(0),
                    kept,
                    100.0 * kept / max(synthetic_features.size(0), 1),
                    dist_match_z_max,
                    dist_match_dims,
                    before_trust_task,
                    before_trust_clip,
                    kept_trust_task,
                    kept_trust_clip,
                    "fallback" if fallback_task else "zscore",
                    "fallback" if fallback_clip else "zscore",
                    min_dist_match_kept,
                )
            )
            synthetic_features = synthetic_features[keep]
            synthetic_targets = synthetic_targets[keep]
            _log_pair_distribution(
                synthetic_features, "synthetic-matched", cycle, log_fn
            )
        if comparator is None:
            raise ValueError("refiner_type=comparator requires a comparator module")
        if (
            synthetic_features.size(0) >= 2
            and comparator_optimizer is not None
            and (comparator_epochs > 0 or train_steps_per_cycle > 0)
        ):
            memory_features = None
            memory_targets = None
            if replay_memory is not None and replay_memory.total() >= 1:
                memory_features, memory_targets = replay_memory.as_tensors()
                log_fn(
                    "DUET comparator replay: cycle={}; memory_total={}; "
                    "mix_fraction={:.2f}; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        replay_memory.total(),
                        replay_mix_fraction,
                    )
                )
            if early_stop_enabled:
                current_samples = int(synthetic_features.size(0))
                early_stop_result = train_pairwise_comparator_early_stopping(
                    comparator,
                    comparator_optimizer,
                    synthetic_features,
                    synthetic_targets,
                    max_steps=train_steps_per_cycle,
                    batch_size=train_batch_size,
                    seed=seed,
                    val_fraction=early_stop_val_fraction,
                    min_val_per_direction=early_stop_min_val_per_direction,
                    check_interval=early_stop_check_interval,
                    patience=early_stop_patience,
                    memory_features=memory_features,
                    memory_targets=memory_targets,
                    memory_fraction=replay_mix_fraction,
                    real_features=real_features,
                    gate=gate,
                    cycle=cycle,
                    log_fn=log_fn,
                )
                stats["train_loss"] = early_stop_result["train_loss"]
                stats["train_current_samples"] = current_samples
                stats["train_memory_samples"] = early_stop_result[
                    "memory_samples_per_step"
                ]
                stats["optimizer_steps_this_cycle"] = early_stop_result[
                    "optimizer_steps"
                ]
                stats["early_stop_best_val_loss"] = early_stop_result[
                    "best_val_loss"
                ]
                stats["early_stop_best_step"] = early_stop_result["best_step"]
                stats["early_stop_stopped"] = early_stop_result[
                    "stopped_early"
                ]
                comparator.eval()
                with torch.no_grad():
                    final_logits = comparator(synthetic_features.detach())
                    stats["final_current_loss"] = float(
                        F.cross_entropy(
                            final_logits,
                            synthetic_targets.detach().long().to(final_logits.device),
                        ).item()
                    )
                best_val_loss = stats["early_stop_best_val_loss"]
                best_val_loss_str = (
                    "none"
                    if best_val_loss is None
                    else "{:.6f}".format(best_val_loss)
                )
                log_fn(
                    "DUET comparator training: cycle={}; "
                    "mode=synthetic_val_early_stop; max_updates={}; "
                    "optimizer_steps_this_cycle={}; best_step={}; "
                    "stopped_early={}; current_samples={}; train_samples={}; "
                    "val_samples={}; val_per_direction={}; "
                    "memory_bank_samples={}; memory_samples_per_step={}; "
                    "best_val_loss={}; final_current_loss={:.6f}; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        train_steps_per_cycle,
                        stats["optimizer_steps_this_cycle"],
                        stats["early_stop_best_step"],
                        stats["early_stop_stopped"],
                        current_samples,
                        early_stop_result["train_samples"],
                        early_stop_result["val_samples"],
                        early_stop_result["val_per_direction"],
                        memory_features.size(0)
                        if memory_features is not None
                        else 0,
                        stats["train_memory_samples"],
                        best_val_loss_str,
                        stats["final_current_loss"],
                    )
                )
            elif comparator_epochs > 0:
                current_samples = int(synthetic_features.size(0))
                use_replay = (
                    memory_features is not None
                    and memory_targets is not None
                    and memory_features.size(0) >= 1
                    and replay_mix_fraction > 0.0
                )
                memory_samples = (
                    max(
                        1,
                        int(
                            round(
                                current_samples
                                * replay_mix_fraction
                                / (1.0 - replay_mix_fraction)
                            )
                        ),
                    )
                    if use_replay
                    else 0
                )
                stats["train_loss"] = train_pairwise_comparator_epochs(
                    comparator,
                    comparator_optimizer,
                    synthetic_features,
                    synthetic_targets,
                    epochs=comparator_epochs,
                    batch_size=train_batch_size,
                    seed=seed,
                    memory_features=memory_features,
                    memory_targets=memory_targets,
                    memory_fraction=replay_mix_fraction,
                )
                stats["train_current_samples"] = current_samples
                stats["train_memory_samples"] = memory_samples
                stats["optimizer_steps_this_cycle"] = max(
                    1, comparator_epochs
                )
                comparator.eval()
                with torch.no_grad():
                    final_logits = comparator(synthetic_features.detach())
                    stats["final_current_loss"] = float(
                        F.cross_entropy(
                            final_logits,
                            synthetic_targets.detach().long().to(final_logits.device),
                        ).item()
                    )
                log_fn(
                    "DUET comparator training: cycle={}; mode=full_batch_epochs; "
                    "epochs={}; current_samples={}; memory_bank_samples={}; "
                    "memory_samples={}; effective_batch_size={}; "
                    "optimizer_steps_this_cycle={}; final_current_loss={:.6f}; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        comparator_epochs,
                        current_samples,
                        memory_features.size(0)
                        if memory_features is not None
                        else 0,
                        memory_samples,
                        current_samples + memory_samples,
                        stats["optimizer_steps_this_cycle"],
                        stats["final_current_loss"],
                    )
                )
            else:
                trajectory = [] if (
                    trajectory_enabled
                    and eval_only_logging
                    and labels is not None
                    and strict_positions.numel() > 0
                ) else None
                stats["train_loss"] = train_pairwise_comparator(
                    comparator,
                    comparator_optimizer,
                    synthetic_features,
                    synthetic_targets,
                    steps=train_steps_per_cycle,
                    batch_size=train_batch_size,
                    seed=seed,
                    memory_features=memory_features,
                    memory_targets=memory_targets,
                    memory_fraction=replay_mix_fraction,
                    trajectory_features=(
                        real_features if trajectory is not None else None
                    ),
                    trajectory_interval=trajectory_interval,
                    trajectory_sink=trajectory,
                )
                stats["train_current_samples"] = int(
                    synthetic_features.size(0)
                )
                stats["train_memory_samples"] = (
                    max(1, int(round(train_batch_size * replay_mix_fraction)))
                    if memory_features is not None and replay_mix_fraction > 0.0
                    else 0
                )
                stats["optimizer_steps_this_cycle"] = max(
                    1, train_steps_per_cycle
                )
                comparator.eval()
                with torch.no_grad():
                    final_logits = comparator(synthetic_features.detach())
                    stats["final_current_loss"] = float(
                        F.cross_entropy(
                            final_logits,
                            synthetic_targets.detach().long().to(final_logits.device),
                        ).item()
                    )
                log_fn(
                    "DUET comparator training: cycle={}; mode=fixed_steps; "
                    "optimizer_steps_this_cycle={}; current_samples={}; "
                    "memory_bank_samples={}; memory_samples_per_step={}; "
                    "final_current_loss={:.6f}; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        stats["optimizer_steps_this_cycle"],
                        stats["train_current_samples"],
                        memory_features.size(0)
                        if memory_features is not None
                        else 0,
                        stats["train_memory_samples"],
                        stats["final_current_loss"],
                    )
                )
                if trajectory is not None:
                    _log_fixed_conflict_trajectory(
                        trajectory,
                        task_candidates=task_top1[strict_positions],
                        clip_candidates=clip_top1[strict_positions],
                        labels=labels[strict_positions],
                        coverages=trajectory_coverages,
                        cycle=cycle,
                        log_fn=log_fn,
                    )
            if replay_memory is not None:
                # 训练后把当前 cycle 的 matched synthetic 写入历史 memory
                replay_memory.update(synthetic_features, synthetic_targets)

        if strict_positions.numel() > 0:
            comparator.eval()
            with torch.no_grad():
                logits = comparator(real_features).cpu()
            decision = apply_pairwise_decision(
                logits,
                task_top1[strict_positions],
                clip_top1[strict_positions],
                gate=gate,
                coverage_fraction=coverage_fraction,
            )
            # Diagnostic only: summarize every real strict conflict, including
            # abstained rows.  No target labels or random operations are used.
            _log_real_comparator_margin_distribution(
                decision["margin"], cycle, gate, log_fn
            )
            selected_count = int(decision["resolved"].sum().item())
            achieved_coverage = (
                100.0 * selected_count / int(strict_positions.numel())
            )
            log_fn(
                "DUET comparator selection: cycle={}; mode={}; selected={}; "
                "total={}; achieved_coverage={:.2f}%; "
                "requested_coverage={:.2f}%; absolute_gate={:.2f}; "
                "absolute_gate_ignored={}; ground_truth_affects_training=False".format(
                    cycle,
                    decision["selection_mode"],
                    selected_count,
                    int(strict_positions.numel()),
                    achieved_coverage,
                    100.0 * coverage_fraction,
                    gate,
                    decision["selection_mode"] == "rank_coverage",
                )
            )
            resolved_rows = decision["resolved"]
            resolved_strict = strict_positions[resolved_rows]
            resolved_mask[resolved_strict] = True
            context_labels[resolved_strict] = decision["chosen"][
                resolved_rows
            ].long()
            winning_distribution = torch.where(
                (
                    decision["trust_task"].unsqueeze(1)
                    >= decision["trust_clip"].unsqueeze(1)
                ).cpu(),
                task_probs[strict_positions],
                clip_probs[strict_positions],
            )
            refined_targets[resolved_strict] = winning_distribution[
                resolved_rows
            ].detach()
            stats["resolved_strict"] = int(resolved_rows.sum().item())
            stats["support_task"] = int(
                (
                    resolved_rows
                    & (decision["trust_task"] >= decision["trust_clip"])
                )
                .sum()
                .item()
            )
            stats["support_clip"] = int(
                (
                    resolved_rows
                    & (decision["trust_task"] < decision["trust_clip"])
                )
                .sum()
                .item()
            )
            stats["abstain"] = int((~resolved_rows).sum().item())
            stats["context_mean_conf"] = float(
                decision["confidence"][resolved_rows].mean().item()
            ) if int(resolved_rows.sum().item()) > 0 else float("nan")
            stats["context_mean_margin"] = float(
                decision["margin"][resolved_rows].mean().item()
            ) if int(resolved_rows.sum().item()) > 0 else float("nan")

    strict_total_used = strict_total if use_strict_conflict else 0
    stats["resolved_rate_pct"] = (
        100.0 * stats["resolved_strict"] / strict_total_used
        if strict_total_used > 0
        else 0.0
    )
    stats["weak_defer_rate_pct"] = 0.0
    stats["final_admitted"] = (
        stats["post_prior_agreement"] - stats["weak_deferred"] + stats["resolved_strict"]
    )
    stats["admitted_delta"] = stats["resolved_strict"] - stats["weak_deferred"]
    _log_correction_stats(stats, cycle, log_fn)
    _log_context_stats(stats, "comparator", cycle, log_fn)
    if eval_only_logging and labels is not None:
        _log_eval_only_metrics(
            stats,
            resolved_mask=resolved_mask,
            weak_rejected_mask=weak_rejected_mask,
            context_labels=context_labels,
            task_top1=task_top1,
            clip_top1=clip_top1,
            all_label=labels,
            anchor_mask=anchor_mask,
            weak_agreement_mask=weak_agreement_mask,
            strict_conflict_mask=strict_conflict_mask,
            cycle=cycle,
            log_fn=log_fn,
        )
    return {
        "strict_conflict_mask": strict_conflict_mask,
        "weak_agreement_mask": weak_agreement_mask,
        "query_mask": query_mask,
        "anchor_mask": anchor_mask,
        "resolved_mask": resolved_mask,
        "weak_rejected_mask": weak_rejected_mask,
        "context_labels": context_labels,
        "refined_targets": refined_targets,
        "stats": stats,
    }


def _fmt_pct(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    return "{:.2f}%".format(value * 100.0)


def _log_context_stats(
    stats: dict,
    refiner_type: str,
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    train_loss = stats.get("train_loss")
    loss_str = "{:.6f}".format(train_loss) if train_loss is not None else "none"
    log_fn(
        "DUET context refinement: cycle={}; active=True; refiner={}; "
        "post_prior_agreement={}; strict_conflicts={}; weak_agreement={}; "
        "anchor_candidates={}; anchors_total={}; anchors_per_class=[{}]; "
        "anchor_task_conf={:.4f}; "
        "anchor_clip_conf={:.4f}; train_loss={}; resolved_strict={}; "
        "support_task={}; support_clip={}; third_class={}; abstain={}; "
        "weak_passed={}; weak_deferred={}; context_conf={:.4f}; "
        "context_margin={:.4f}; ground_truth_affects_training=False".format(
            cycle,
            refiner_type,
            stats["post_prior_agreement"],
            stats["strict_conflicts"],
            stats["weak_agreement"],
            stats["anchor_count"],
            stats["anchor_bank_total"],
            ",".join(str(v) for v in stats["anchor_per_class_counts"]),
            stats["anchor_mean_task_conf"],
            stats["anchor_mean_clip_conf"],
            loss_str,
            stats["resolved_strict"],
            stats["support_task"],
            stats["support_clip"],
            stats["third_class"],
            stats["abstain"],
            stats["weak_passed"],
            stats["weak_deferred"],
            stats["context_mean_conf"],
            stats["context_mean_margin"],
        )
    )


def _log_correction_stats(
    stats: dict,
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    """每轮修正统计（不使用 target ground-truth）。

    - resolved：原本无标签的 strict conflict 被赋予 hard label 的数量；
    - weak_deferred：原本会被准入的 weak agreement 被暂缓的数量；
    - final_admitted = 原始 agreement - weak_deferred + resolved。
    """
    log_fn(
        "DUET context correction: cycle={}; original_agreement={}; "
        "strict_conflicts={}; resolved={}; resolved_rate={:.2f}%; "
        "support_task={}; support_clip={}; third_class={}; "
        "weak_agreement={}; weak_passed={}; weak_deferred={}; "
        "weak_defer_rate={:.2f}%; final_admitted={}; admitted_delta={}; "
        "ground_truth_affects_training=False".format(
            cycle,
            stats["post_prior_agreement"],
            stats["strict_conflicts"],
            stats["resolved_strict"],
            stats["resolved_rate_pct"],
            stats["support_task"],
            stats["support_clip"],
            stats["third_class"],
            stats["weak_agreement"],
            stats["weak_passed"],
            stats["weak_deferred"],
            stats["weak_defer_rate_pct"],
            stats["final_admitted"],
            stats["admitted_delta"],
        )
    )


def _log_eval_only_metrics(
    stats: dict,
    *,
    resolved_mask: torch.Tensor,
    weak_rejected_mask: torch.Tensor,
    context_labels: torch.Tensor,
    task_top1: torch.Tensor,
    clip_top1: torch.Tensor,
    all_label: torch.Tensor,
    anchor_mask: torch.Tensor,
    weak_agreement_mask: torch.Tensor,
    strict_conflict_mask: torch.Tensor,
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    """Target labels are read here only; they never affect training."""
    all_label = all_label.long()

    def acc(pred: torch.Tensor, mask: torch.Tensor) -> str:
        if int(mask.sum().item()) == 0:
            return "nan"
        return _fmt_pct(float((pred[mask] == all_label[mask]).float().mean().item()))

    anchor_precision = acc(task_top1, anchor_mask)
    strict_task_acc = acc(task_top1, strict_conflict_mask)
    strict_clip_acc = acc(clip_top1, strict_conflict_mask)
    # "Original mixed" on conflicts is ambiguous (task != clip); report the
    # task-side proxy, consistent with the unresolved DUET path.
    strict_mix_acc = acc(task_top1, strict_conflict_mask)
    # Same-subset diagnostic: all four accuracies below are evaluated on the
    # exact same rows selected by the comparator (resolved_mask).  The older
    # strict_task_acc / strict_clip_acc use the full strict-conflict set and
    # therefore must not be compared directly with resolved_acc.
    resolved_subset_task_acc = acc(task_top1, resolved_mask)
    resolved_subset_clip_acc = acc(clip_top1, resolved_mask)
    resolved_comparator_acc = acc(context_labels, resolved_mask)
    if int(resolved_mask.sum().item()) == 0:
        resolved_candidate_oracle_acc = "nan"
        conditional_arbitration_acc = "nan"
    else:
        candidate_oracle_correct = (
            (task_top1[resolved_mask] == all_label[resolved_mask])
            | (clip_top1[resolved_mask] == all_label[resolved_mask])
        )
        candidate_oracle_correct_count = int(
            candidate_oracle_correct.sum().item()
        )
        resolved_candidate_oracle_acc = _fmt_pct(
            float(candidate_oracle_correct.float().mean().item())
        )
        if candidate_oracle_correct_count == 0:
            conditional_arbitration_acc = "nan"
        else:
            resolved_correct_count = int(
                (
                    context_labels[resolved_mask]
                    == all_label[resolved_mask]
                ).sum().item()
            )
            conditional_arbitration_acc = _fmt_pct(
                resolved_correct_count / candidate_oracle_correct_count
            )
    # Backward-compatible alias retained for existing log parsers.
    resolved_acc = resolved_comparator_acc

    support_task_sub = (resolved_mask & (context_labels == task_top1)).sum().item() > 0
    support_task_acc = (
        _fmt_pct(
            float(
                (
                    all_label[resolved_mask & (context_labels == task_top1)]
                    == task_top1[resolved_mask & (context_labels == task_top1)]
                )
                .float()
                .mean()
                .item()
            )
        )
        if support_task_sub
        else "nan"
    )
    support_clip_sub = (resolved_mask & (context_labels == clip_top1)).sum().item() > 0
    support_clip_acc = (
        _fmt_pct(
            float(
                (
                    all_label[resolved_mask & (context_labels == clip_top1)]
                    == clip_top1[resolved_mask & (context_labels == clip_top1)]
                )
                .float()
                .mean()
                .item()
            )
        )
        if support_clip_sub
        else "nan"
    )
    third_mask = (
        resolved_mask & (context_labels != task_top1) & (context_labels != clip_top1)
    )
    third_sub = third_mask.sum().item() > 0
    third_acc = (
        _fmt_pct(float((all_label[third_mask] == context_labels[third_mask]).float().mean().item()))
        if third_sub
        else "nan"
    )
    abstain_mask = strict_conflict_mask & ~resolved_mask
    abstain_sub = abstain_mask.sum().item() > 0
    abstain_orig_acc = (
        _fmt_pct(float((all_label[abstain_mask] == task_top1[abstain_mask]).float().mean().item()))
        if abstain_sub
        else "nan"
    )
    weak_orig_precision = acc(task_top1, weak_agreement_mask)
    weak_passed_mask = weak_agreement_mask & ~weak_rejected_mask
    weak_passed_sub = weak_passed_mask.sum().item() > 0
    weak_passed_precision = (
        _fmt_pct(float((all_label[weak_passed_mask] == task_top1[weak_passed_mask]).float().mean().item()))
        if weak_passed_sub
        else "nan"
    )
    weak_deferred_sub = weak_rejected_mask.sum().item() > 0
    weak_deferred_precision = (
        _fmt_pct(float((all_label[weak_rejected_mask] == task_top1[weak_rejected_mask]).float().mean().item()))
        if weak_deferred_sub
        else "nan"
    )
    log_fn(
        "DUET context eval-only: cycle={}; anchor_precision={}; "
        "strict_task_acc={}; strict_clip_acc={}; strict_mix_acc={}; "
        "resolved_acc={}; resolved_subset_task_acc={}; "
        "resolved_subset_clip_acc={}; resolved_comparator_acc={}; "
        "resolved_candidate_oracle_acc={}; conditional_arbitration_acc={}; "
        "support_task_acc={}; "
        "support_clip_acc={}; "
        "third_acc={}; abstain_orig_acc={}; weak_orig_precision={}; "
        "weak_passed_precision={}; weak_deferred_precision={}; "
        "ground_truth_affects_training=False".format(
            cycle,
            anchor_precision,
            strict_task_acc,
            strict_clip_acc,
            strict_mix_acc,
            resolved_acc,
            resolved_subset_task_acc,
            resolved_subset_clip_acc,
            resolved_comparator_acc,
            resolved_candidate_oracle_acc,
            conditional_arbitration_acc,
            support_task_acc,
            support_clip_acc,
            third_acc,
            abstain_orig_acc,
            weak_orig_precision,
            weak_passed_precision,
            weak_deferred_precision,
        )
    )
    # ---- 修正正确 / 修正错误（仅 eval，真标签只用于此日志）----
    resolved_count = int(resolved_mask.sum().item())
    resolved_correct = (
        int((context_labels[resolved_mask] == all_label[resolved_mask]).sum().item())
        if resolved_count
        else 0
    )
    resolved_error = resolved_count - resolved_correct
    resolved_acc_pct = (
        100.0 * resolved_correct / resolved_count if resolved_count else float("nan")
    )

    deferred_mask = weak_agreement_mask & weak_rejected_mask
    deferred_count = int(deferred_mask.sum().item())
    # 暂缓正确：共同伪标签本来就是错的（common != gt），推迟是对的
    deferred_right = (
        int((task_top1[deferred_mask] != all_label[deferred_mask]).sum().item())
        if deferred_count
        else 0
    )
    deferred_wrong = deferred_count - deferred_right

    passed_mask = weak_agreement_mask & ~weak_rejected_mask
    passed_count = int(passed_mask.sum().item())
    passed_right = (
        int((task_top1[passed_mask] == all_label[passed_mask]).sum().item())
        if passed_count
        else 0
    )
    passed_wrong = passed_count - passed_right

    intervention_total = resolved_count + deferred_count
    intervention_error = resolved_error + deferred_wrong
    intervention_error_rate_pct = (
        100.0 * intervention_error / intervention_total
        if intervention_total
        else float("nan")
    )
    log_fn(
        "DUET context correction eval-only: cycle={}; resolved_correct={}; "
        "resolved_error={}; resolved_acc={:.2f}%; weak_deferred_right={}; "
        "weak_deferred_wrong={}; weak_passed_right={}; weak_passed_wrong={}; "
        "intervention_total={}; intervention_error={}; "
        "intervention_error_rate={:.2f}%; "
        "ground_truth_affects_training=False".format(
            cycle,
            resolved_correct,
            resolved_error,
            resolved_acc_pct,
            deferred_right,
            deferred_wrong,
            passed_right,
            passed_wrong,
            intervention_total,
            intervention_error,
            intervention_error_rate_pct,
        )
    )
