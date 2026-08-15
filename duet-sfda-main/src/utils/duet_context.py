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

The module only depends on torch.  The formal DUET method never sees target
ground-truth.  The optional ``labels`` argument is used for evaluation-only
logging and, when explicitly enabled, isolated offline feature probes whose
models and outputs never enter formal training or admission.
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


def _stratified_probe_fold_ids(
    strata: torch.Tensor,
    folds: int,
    seed: int,
) -> torch.Tensor:
    """Assign deterministic round-robin folds within each diagnostic stratum."""
    strata = strata.detach().long().cpu().flatten()
    if folds < 2:
        raise ValueError("GT feature probe requires at least 2 folds")
    fold_ids = torch.full_like(strata, -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for value in torch.unique(strata, sorted=True).tolist():
        positions = torch.nonzero(strata == int(value), as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        order = torch.randperm(positions.numel(), generator=generator)
        shuffled = positions[order]
        fold_ids[shuffled] = torch.arange(shuffled.numel()) % int(folds)
    if bool((fold_ids < 0).any().item()):
        raise RuntimeError("failed to assign every GT feature-probe row to a fold")
    return fold_ids


class _RealConflictFeatureProbe(nn.Module):
    """Small local model used only by the offline GT feature-capacity probe."""

    def __init__(self, input_dim: int, kind: str, hidden: int) -> None:
        super().__init__()
        if kind == "logistic":
            self.network = nn.Linear(int(input_dim), 2)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden)),
                nn.GELU(),
                nn.Linear(int(hidden), int(hidden)),
                nn.GELU(),
                nn.Linear(int(hidden), 2),
            )
        else:
            raise ValueError("GT feature probe kind must be logistic or mlp")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float())


def _fit_real_conflict_feature_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    test_features: torch.Tensor,
    *,
    kind: str,
    hidden: int,
    steps: int,
    lr: float,
    seed: int,
) -> torch.Tensor:
    """Fit one fixed-budget CPU probe and return test logits."""
    if steps < 1:
        raise ValueError("GT feature probe steps must be >= 1")
    if lr <= 0.0:
        raise ValueError("GT feature probe learning rate must be positive")
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_standardized = (train_features - mean) / std
    test_standardized = (test_features - mean) / std
    # Restore the caller's CPU RNG state after local model initialization and
    # optimization so enabling this eval-only branch cannot alter later DUET
    # dropout, sampling, or optimizer behavior.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = _RealConflictFeatureProbe(
            input_dim=train_features.size(1), kind=kind, hidden=hidden
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(lr), weight_decay=1e-4
        )
        model.train()
        with torch.enable_grad():
            for _ in range(int(steps)):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(
                    model(train_standardized), train_targets.long()
                )
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            return model(test_standardized).detach().cpu()


def _build_extended_real_conflict_probe_features(
    base_features: torch.Tensor,
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    task_profile_reference: torch.Tensor,
    clip_profile_reference: torch.Tensor,
    *,
    ranking_chunk_size: int = 512,
) -> torch.Tensor:
    """Append four GT-free distribution/reliability signals to the formal 16D.

    The formal Comparator remains 16-dimensional.  These extra columns exist
    only for the offline capacity probe:

      16 task profile drift to the confident-agreement reference
      17 CLIP profile drift to the confident-agreement reference
      18 Task/CLIP Jensen-Shannon divergence
      19 normalized all-class-pair ranking disagreement

    Profile sorting follows the class-identity-free reliability proxy used by
    MG-MTTA.  The reference is supplied by the caller so the diagnostic's
    exact GT-free reference population is explicit and testable.
    """
    base = base_features.detach().float()
    task = task_probs.detach().float().to(base.device)
    clip = clip_probs.detach().float().to(base.device)
    task_reference = task_profile_reference.detach().float().to(base.device).flatten()
    clip_reference = clip_profile_reference.detach().float().to(base.device).flatten()
    if base.dim() != 2 or base.size(1) != 16:
        raise ValueError("extended GT probe expects base_features with shape [N, 16]")
    if task.dim() != 2 or clip.shape != task.shape or task.size(0) != base.size(0):
        raise ValueError("extended GT probe Task/CLIP probabilities must have shape [N, C]")
    if task.size(1) < 2:
        raise ValueError("extended GT probe requires at least two classes")
    if task_reference.shape != (task.size(1),) or clip_reference.shape != (
        task.size(1),
    ):
        raise ValueError("extended GT probe profile references must have shape [C]")
    if ranking_chunk_size < 1:
        raise ValueError("ranking_chunk_size must be >= 1")
    tensors = (base, task, clip, task_reference, clip_reference)
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("extended GT probe inputs must be finite")

    task_sorted = task.sort(dim=1, descending=True).values
    clip_sorted = clip.sort(dim=1, descending=True).values
    task_reference_sorted = task_reference.sort(descending=True).values.unsqueeze(0)
    clip_reference_sorted = clip_reference.sort(descending=True).values.unsqueeze(0)
    task_profile_drift = (task_sorted - task_reference_sorted).abs().sum(dim=1)
    clip_profile_drift = (clip_sorted - clip_reference_sorted).abs().sum(dim=1)

    eps = 1e-12
    midpoint = 0.5 * (task + clip)
    task_safe = task.clamp_min(eps)
    clip_safe = clip.clamp_min(eps)
    midpoint_safe = midpoint.clamp_min(eps)
    js_divergence = 0.5 * (
        (task_safe * (task_safe.log() - midpoint_safe.log())).sum(dim=1)
        + (clip_safe * (clip_safe.log() - midpoint_safe.log())).sum(dim=1)
    )

    num_samples, num_classes = task.shape
    pair_mask = torch.triu(
        torch.ones(
            num_classes,
            num_classes,
            dtype=torch.bool,
            device=base.device,
        ),
        diagonal=1,
    )
    pair_count = float(num_classes * (num_classes - 1) // 2)
    ranking_disagreement = torch.empty(
        num_samples, dtype=torch.float32, device=base.device
    )
    # Chunking avoids materializing [N, C, C] differences for all VisDA or
    # Office-Home conflicts at once. Exact probability ties are not counted as
    # discordant in either direction.
    for start in range(0, num_samples, int(ranking_chunk_size)):
        stop = min(start + int(ranking_chunk_size), num_samples)
        task_chunk = task[start:stop]
        clip_chunk = clip[start:stop]
        task_diff = task_chunk.unsqueeze(2) - task_chunk.unsqueeze(1)
        clip_diff = clip_chunk.unsqueeze(2) - clip_chunk.unsqueeze(1)
        discordant = ((task_diff * clip_diff) < 0.0) & pair_mask.unsqueeze(0)
        ranking_disagreement[start:stop] = (
            discordant.sum(dim=(1, 2)).float() / pair_count
        )

    extended = torch.cat(
        [
            base,
            task_profile_drift.unsqueeze(1),
            clip_profile_drift.unsqueeze(1),
            js_divergence.unsqueeze(1),
            ranking_disagreement.unsqueeze(1),
        ],
        dim=1,
    )
    if extended.shape != (base.size(0), 20):
        raise RuntimeError("extended GT probe failed to construct [N, 20] features")
    return extended


def _log_real_conflict_gt_feature_probe_eval_only(
    features: torch.Tensor,
    task_candidates: torch.Tensor,
    clip_candidates: torch.Tensor,
    labels: torch.Tensor,
    current_comparator_logits: torch.Tensor,
    *,
    folds: int,
    steps: int,
    hidden: int,
    lr: float,
    seed: int,
    cycle: int,
    log_fn: Callable[[str], None],
    feature_label: str = "16D_current_comparator_features",
    log_variant: str = "",
) -> dict:
    """Measure a GT-supervised ceiling of real-conflict feature evidence.

    This is a deliberately non-methodological offline probe.  GT determines
    its binary training targets, but only inside held-out cross-validation.
    Formal Comparator weights, pseudo-labels, admission, and all returned
    training state remain untouched.
    """
    log_stem = "DUET real-conflict GT feature probe"
    if log_variant:
        log_stem = "{} {}".format(log_stem, log_variant)
    x = features.detach().float().cpu()
    task_candidates = task_candidates.detach().long().cpu().flatten()
    clip_candidates = clip_candidates.detach().long().cpu().flatten()
    labels = labels.detach().long().cpu().flatten()
    current_logits = current_comparator_logits.detach().float().cpu()
    total = int(x.size(0))
    if x.dim() != 2:
        raise ValueError("GT feature probe features must have shape [N, D]")
    if any(tensor.shape != (total,) for tensor in (task_candidates, clip_candidates, labels)):
        raise ValueError("GT feature probe candidates and labels must have shape [N]")
    if current_logits.shape != (total, 2):
        raise ValueError("current_comparator_logits must have shape [N, 2]")
    if total == 0:
        raise ValueError("GT feature probe requires at least one real conflict")
    if bool((task_candidates == clip_candidates).any().item()):
        raise ValueError("GT feature probe expects strict Task/CLIP conflicts")

    task_correct = labels == task_candidates
    clip_correct = labels == clip_candidates
    oracle = task_correct | clip_correct
    neither = ~oracle
    task_target_count = int(task_correct.sum().item())
    clip_target_count = int(clip_correct.sum().item())
    oracle_count = int(oracle.sum().item())
    neither_count = int(neither.sum().item())
    max_folds = min(task_target_count, clip_target_count)
    effective_folds = min(int(folds), max_folds)
    if effective_folds < 2:
        log_fn(
            "{} eval-only: cycle={}; "
            "status=skipped_insufficient_binary_targets; total={}; "
            "task_correct_count={}; clip_correct_count={}; neither_count={}; "
            "requested_folds={}; effective_folds={}; probe_uses_gt=True; "
            "formal_method_affected=False".format(
                log_stem,
                cycle,
                total,
                task_target_count,
                clip_target_count,
                neither_count,
                folds,
                effective_folds,
            )
        )
        return {
            "status": "skipped_insufficient_binary_targets",
            "total": total,
            "oracle_count": oracle_count,
            "neither_count": neither_count,
        }

    binary_targets = torch.full((total,), -1, dtype=torch.long)
    binary_targets[task_correct] = 0
    binary_targets[clip_correct] = 1
    # Three strata keep Task-correct, CLIP-correct, and neither rows spread
    # across held-out folds.  Neither rows are evaluated but never trained on.
    strata = torch.full((total,), 2, dtype=torch.long)
    strata[task_correct] = 0
    strata[clip_correct] = 1
    fold_ids = _stratified_probe_fold_ids(strata, effective_folds, seed)

    predictions = {
        "logistic": torch.full((total,), -1, dtype=torch.long),
        "mlp": torch.full((total,), -1, dtype=torch.long),
    }
    fold_rows: list[dict] = []
    for fold in range(effective_folds):
        test_mask = fold_ids == fold
        train_mask = (fold_ids != fold) & oracle
        train_targets = binary_targets[train_mask]
        if int(torch.unique(train_targets).numel()) != 2:
            raise RuntimeError("each GT feature-probe training fold must contain both directions")
        row = {
            "fold": fold + 1,
            "test_count": int(test_mask.sum().item()),
            "train_count": int(train_mask.sum().item()),
        }
        for kind_offset, kind in enumerate(("logistic", "mlp")):
            fold_logits = _fit_real_conflict_feature_probe(
                x[train_mask],
                train_targets,
                x[test_mask],
                kind=kind,
                hidden=hidden,
                steps=steps,
                lr=lr,
                seed=int(seed) + 1000 * fold + 100 * kind_offset,
            )
            fold_prediction = fold_logits.argmax(dim=1)
            predictions[kind][test_mask] = fold_prediction
            fold_correct = torch.where(
                fold_prediction == 0,
                task_correct[test_mask],
                clip_correct[test_mask],
            )
            row["{}_acc".format(kind)] = 100.0 * float(
                fold_correct.float().mean().item()
            )
        fold_rows.append(row)
        log_fn(
            "{} fold eval-only: cycle={}; "
            "fold={}/{}; train_oracle_n={}; test_all_n={}; "
            "logistic_acc={:.2f}%; mlp_acc={:.2f}%; probe_uses_gt=True; "
            "formal_method_affected=False".format(
                log_stem,
                cycle,
                fold + 1,
                effective_folds,
                row["train_count"],
                row["test_count"],
                row["logistic_acc"],
                row["mlp_acc"],
            )
        )

    if any(bool((prediction < 0).any().item()) for prediction in predictions.values()):
        raise RuntimeError("GT feature probe did not produce out-of-fold predictions for all rows")

    current_prediction = current_logits.argmax(dim=1)

    def metrics_for_prediction(prediction: torch.Tensor) -> tuple[float, float]:
        correct = torch.where(prediction == 0, task_correct, clip_correct)
        all_acc = 100.0 * float(correct.float().mean().item())
        conditional = 100.0 * float(correct[oracle].float().mean().item())
        return all_acc, conditional

    current_acc, current_conditional = metrics_for_prediction(current_prediction)
    logistic_acc, logistic_conditional = metrics_for_prediction(predictions["logistic"])
    mlp_acc, mlp_conditional = metrics_for_prediction(predictions["mlp"])
    task_acc = 100.0 * task_target_count / total
    clip_acc = 100.0 * clip_target_count / total
    oracle_acc = 100.0 * oracle_count / total
    result = {
        "status": "ok",
        "feature_label": feature_label,
        "total": total,
        "oracle_count": oracle_count,
        "neither_count": neither_count,
        "task_acc": task_acc,
        "clip_acc": clip_acc,
        "current_comparator_acc": current_acc,
        "current_conditional_arbitration_acc": current_conditional,
        "logistic_probe_acc": logistic_acc,
        "logistic_conditional_arbitration_acc": logistic_conditional,
        "mlp_probe_acc": mlp_acc,
        "mlp_conditional_arbitration_acc": mlp_conditional,
        "candidate_oracle_acc": oracle_acc,
        "folds": fold_rows,
    }
    log_fn(
        "{} summary eval-only: cycle={}; "
        "features={}; total={}; folds={}; "
        "task_correct_count={}; clip_correct_count={}; neither_count={}; "
        "task_acc={:.2f}%; clip_acc={:.2f}%; "
        "synthetic_comparator_acc={:.2f}%; "
        "logistic_probe_acc={:.2f}%; mlp_probe_acc={:.2f}%; "
        "candidate_oracle_acc={:.2f}%; "
        "synthetic_conditional_arbitration_acc={:.2f}%; "
        "logistic_conditional_arbitration_acc={:.2f}%; "
        "mlp_conditional_arbitration_acc={:.2f}%; "
        "probe_steps={}; probe_hidden={}; probe_lr={:.6f}; "
        "cv_preprocessing=train_fold_only; checkpoint_selection=none_fixed_steps; "
        "probe_uses_gt=True; formal_comparator_uses_gt=False; "
        "formal_method_affected=False".format(
            log_stem,
            cycle,
            feature_label,
            total,
            effective_folds,
            task_target_count,
            clip_target_count,
            neither_count,
            task_acc,
            clip_acc,
            current_acc,
            logistic_acc,
            mlp_acc,
            oracle_acc,
            current_conditional,
            logistic_conditional,
            mlp_conditional,
            steps,
            hidden,
            lr,
        )
    )
    return result


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
    duet_fallback_candidates: Optional[torch.Tensor] = None,
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
    if duet_fallback_candidates is None:
        duet_fallback_candidates = task_candidates
    duet_fallback_candidates = duet_fallback_candidates.detach().long().cpu()
    labels = labels.detach().long().cpu()
    total = int(labels.numel())
    if total == 0:
        return
    if (
        task_candidates.shape != labels.shape
        or clip_candidates.shape != labels.shape
        or duet_fallback_candidates.shape != labels.shape
    ):
        raise ValueError("fixed-conflict candidates and labels must have equal shape")

    task_correct = task_candidates == labels
    clip_correct = clip_candidates == labels
    duet_fallback_correct = duet_fallback_candidates == labels
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
            "task_acc={:.2f}%; clip_acc={:.2f}%; duet_fallback_acc={:.2f}%; "
            "comparator_acc={:.2f}%; "
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
                pct(duet_fallback_correct),
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
            comparator_acc = pct(comparator_correct[selected])
            fallback_acc = pct(duet_fallback_correct[selected])
            gain = comparator_acc - fallback_acc
            achieved_fraction = float(selected_count) / float(total)
            coverage_parts.extend(
                [
                    "{}_n={}".format(prefix, selected_count),
                    "{}_task_acc={:.2f}%".format(
                        prefix, pct(task_correct[selected])
                    ),
                    "{}_clip_acc={:.2f}%".format(
                        prefix, pct(clip_correct[selected])
                    ),
                    "{}_duet_fallback_acc={:.2f}%".format(
                        prefix, fallback_acc
                    ),
                    "{}_comparator_acc={:.2f}%".format(
                        prefix, comparator_acc
                    ),
                    "{}_gain_over_duet_fallback={:+.2f}pp".format(
                        prefix, gain
                    ),
                    "{}_coverage_weighted_gain={:+.3f}pp".format(
                        prefix, achieved_fraction * gain
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


@torch.no_grad()
def _log_agreement_ambiguity_eval_only(
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    fractions: list[int],
    cycle: int,
    log_fn: Callable[[str], None],
) -> dict:
    """Evaluate shared-Top2 recovery in the ambiguous agreement tail.

    This function is deliberately isolated from candidate construction,
    comparator training and admission.  It reads GT only to emit diagnostics
    after the Task/CLIP probabilities already exist, and its return value is
    never consumed by the training path.
    """
    task_probs = task_probs.detach().float().cpu()
    clip_probs = clip_probs.detach().float().cpu()
    labels = labels.detach().long().cpu()
    if task_probs.dim() != 2 or clip_probs.shape != task_probs.shape:
        raise ValueError("task_probs and clip_probs must have equal [N, C] shape")
    if labels.shape != (task_probs.size(0),):
        raise ValueError("labels must have shape [N]")
    if task_probs.size(1) < 2:
        raise ValueError("agreement ambiguity diagnostic requires >= 2 classes")
    if not fractions or any(int(value) <= 0 or int(value) > 100 for value in fractions):
        raise ValueError("agreement ambiguity fractions must be in [1, 100]")

    _, task_indices = torch.topk(task_probs, k=2, dim=1)
    _, clip_indices = torch.topk(clip_probs, k=2, dim=1)
    task_top1 = task_indices[:, 0]
    clip_top1 = clip_indices[:, 0]
    agreement_mask = task_top1 == clip_top1
    agreement_total = int(agreement_mask.sum().item())
    shared_top2_mask = agreement_mask & (
        task_indices[:, 1] == clip_indices[:, 1]
    )
    shared_count = int(shared_top2_mask.sum().item())
    different_top2_count = agreement_total - shared_count
    shared_rate = (
        100.0 * shared_count / agreement_total
        if agreement_total > 0
        else float("nan")
    )

    rows: list[dict] = []
    if shared_count > 0:
        a = task_top1[shared_top2_mask]
        b = task_indices[shared_top2_mask, 1]
        gt = labels[shared_top2_mask]
        row_ids = torch.arange(shared_count)
        task_a = task_probs[shared_top2_mask][row_ids, a]
        task_b = task_probs[shared_top2_mask][row_ids, b]
        clip_a = clip_probs[shared_top2_mask][row_ids, a]
        clip_b = clip_probs[shared_top2_mask][row_ids, b]
        task_gap = task_a - task_b
        clip_gap = clip_a - clip_b
        ambiguity_gap = torch.maximum(task_gap, clip_gap)
        # Stable sorting makes equal-gap selection deterministic without GT.
        order = torch.argsort(ambiguity_gap, descending=False, stable=True)

        for fraction in fractions:
            fraction = int(fraction)
            selected_count = max(
                1,
                min(
                    shared_count,
                    int(round(shared_count * fraction / 100.0)),
                ),
            )
            selected = order[:selected_count]
            selected_gt = gt[selected]
            selected_a = a[selected]
            selected_b = b[selected]
            gt_is_top1 = selected_gt == selected_a
            gt_is_top2 = selected_gt == selected_b
            gt_neither = ~(gt_is_top1 | gt_is_top2)
            top1_count = int(gt_is_top1.sum().item())
            top2_count = int(gt_is_top2.sum().item())
            neither_count = int(gt_neither.sum().item())
            row = {
                "fraction": fraction,
                "count": selected_count,
                "gt_is_top1_count": top1_count,
                "gt_is_top1_rate": 100.0 * top1_count / selected_count,
                "gt_is_top2_count": top2_count,
                "gt_is_top2_rate": 100.0 * top2_count / selected_count,
                "gt_neither_count": neither_count,
                "gt_neither_rate": 100.0 * neither_count / selected_count,
                "candidate_oracle_acc": (
                    100.0 * (top1_count + top2_count) / selected_count
                ),
                "task_gap_mean": float(task_gap[selected].mean().item()),
                "clip_gap_mean": float(clip_gap[selected].mean().item()),
                "ambiguity_gap_mean": float(
                    ambiguity_gap[selected].mean().item()
                ),
                "task_a_prob_mean": float(task_a[selected].mean().item()),
                "task_b_prob_mean": float(task_b[selected].mean().item()),
                "clip_a_prob_mean": float(clip_a[selected].mean().item()),
                "clip_b_prob_mean": float(clip_b[selected].mean().item()),
            }
            rows.append(row)
            log_fn(
                "DUET agreement ambiguity fraction eval-only: cycle={}; "
                "fraction={}%; n={}; gt_is_top1_count={}; "
                "gt_is_top1_rate={:.2f}%; top1_acc={:.2f}%; "
                "gt_is_top2_count={}; gt_is_top2_rate={:.2f}%; "
                "top2_recovery_rate={:.2f}%; gt_neither_count={}; "
                "gt_neither_rate={:.2f}%; neither_rate={:.2f}%; "
                "candidate_oracle_acc={:.2f}%; task_gap_mean={:.6f}; "
                "clip_gap_mean={:.6f}; ambiguity_gap_mean={:.6f}; "
                "Task_A_prob_mean={:.6f}; Task_B_prob_mean={:.6f}; "
                "CLIP_A_prob_mean={:.6f}; CLIP_B_prob_mean={:.6f}; "
                "selection_uses_gt=False; ground_truth_affects_training=False".format(
                    cycle,
                    fraction,
                    selected_count,
                    top1_count,
                    row["gt_is_top1_rate"],
                    row["gt_is_top1_rate"],
                    top2_count,
                    row["gt_is_top2_rate"],
                    row["gt_is_top2_rate"],
                    neither_count,
                    row["gt_neither_rate"],
                    row["gt_neither_rate"],
                    row["candidate_oracle_acc"],
                    row["task_gap_mean"],
                    row["clip_gap_mean"],
                    row["ambiguity_gap_mean"],
                    row["task_a_prob_mean"],
                    row["task_b_prob_mean"],
                    row["clip_a_prob_mean"],
                    row["clip_b_prob_mean"],
                )
            )

    row_25 = next((row for row in rows if row["fraction"] == 25), None)
    if row_25 is None:
        summary_25 = (
            "ambiguous_25_count=0; gt_is_top1_count=0; gt_is_top1_rate=nan; "
            "gt_is_top2_count=0; gt_is_top2_rate=nan; gt_neither_count=0; "
            "gt_neither_rate=nan; candidate_oracle_acc=nan"
        )
    else:
        summary_25 = (
            "ambiguous_25_count={}; gt_is_top1_count={}; "
            "gt_is_top1_rate={:.2f}%; gt_is_top2_count={}; "
            "gt_is_top2_rate={:.2f}%; gt_neither_count={}; "
            "gt_neither_rate={:.2f}%; candidate_oracle_acc={:.2f}%"
        ).format(
            row_25["count"],
            row_25["gt_is_top1_count"],
            row_25["gt_is_top1_rate"],
            row_25["gt_is_top2_count"],
            row_25["gt_is_top2_rate"],
            row_25["gt_neither_count"],
            row_25["gt_neither_rate"],
            row_25["candidate_oracle_acc"],
        )
    agreement_reference_rate = "100.00%" if agreement_total > 0 else "nan"
    log_fn(
        "DUET agreement ambiguity summary eval-only: cycle={}; "
        "agreement_total={}; agreement_reference_rate={}; "
        "shared_top2_agreement_count={}; shared_top2_agreement_rate={:.2f}%; "
        "different_top2_count={}; {}; selection_uses_gt=False; "
        "ground_truth_affects_training=False".format(
            cycle,
            agreement_total,
            agreement_reference_rate,
            shared_count,
            shared_rate,
            different_top2_count,
            summary_25,
        )
    )
    return {
        "agreement_total": agreement_total,
        "shared_top2_agreement_count": shared_count,
        "different_top2_count": different_top2_count,
        "fractions": rows,
    }


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


def _align_historical_conflict_snapshot(
    snapshot: dict,
    current_sample_indices: torch.Tensor,
) -> dict:
    """Align a pre-adaptation snapshot to the current loader row order."""
    if not snapshot or "sample_indices" not in snapshot:
        raise ValueError("transition supervision requires a historical snapshot")
    historical_indices = snapshot["sample_indices"].detach().long().cpu()
    current_indices = current_sample_indices.detach().long().cpu()
    if historical_indices.numel() != current_indices.numel():
        raise ValueError("historical/current transition snapshots have different sizes")
    if torch.unique(historical_indices).numel() != historical_indices.numel():
        raise ValueError("historical transition sample indices must be unique")
    order = torch.argsort(historical_indices, stable=True)
    sorted_indices = historical_indices[order]
    positions = torch.searchsorted(sorted_indices, current_indices)
    if bool((positions >= sorted_indices.numel()).any().item()) or not torch.equal(
        sorted_indices[positions], current_indices
    ):
        raise ValueError("historical/current transition sample identities differ")
    aligned_rows = order[positions]
    aligned = {"sample_indices": current_indices}
    required = (
        "task_probs",
        "clip_probs",
        "pre_prior_task_probs",
        "pre_prior_clip_probs",
        "task_features",
        "clip_features",
    )
    for key in required:
        if key not in snapshot:
            raise ValueError("historical transition snapshot missing {}".format(key))
        value = snapshot[key].detach().cpu()
        if value.size(0) != historical_indices.numel():
            raise ValueError("historical transition tensor {} has wrong N".format(key))
        aligned[key] = value[aligned_rows]
    return aligned


@torch.no_grad()
def build_delayed_transition_supervision(
    historical_snapshot: dict,
    current_task_probs: torch.Tensor,
    current_clip_probs: torch.Tensor,
    current_task_view_probs: torch.Tensor,
    current_clip_view_probs: torch.Tensor,
    *,
    num_classes: int,
    anchors_per_class: int,
    anchor_task_conf: float,
    anchor_clip_conf: float,
    anchor_task_entropy: float,
    anchor_clip_entropy: float,
    entropy_weight: float,
    require_pre_post_prior_agreement: bool,
    sim_topk: int,
    min_view_agreement: float,
    min_per_direction: int,
    seed: int,
) -> dict:
    """Turn pre-Cycle-1 real conflicts into delayed A/B supervision.

    A historical conflict is labeled only when the post-Cycle-1 weak Task and
    CLIP predictions agree on one of its original candidates and each branch's
    stochastic views support that same candidate often enough.  No target GT
    enters construction, ranking, balancing, or training.
    """
    if not 0.5 <= float(min_view_agreement) <= 1.0:
        raise ValueError("min_view_agreement must be in [0.5, 1]")
    if int(min_per_direction) < 1:
        raise ValueError("min_per_direction must be >= 1")
    historical_task = historical_snapshot["task_probs"].detach().float().cpu()
    historical_clip = historical_snapshot["clip_probs"].detach().float().cpu()
    historical_task_feature = (
        historical_snapshot["task_features"].detach().float().cpu()
    )
    historical_clip_feature = (
        historical_snapshot["clip_features"].detach().float().cpu()
    )
    current_task = current_task_probs.detach().float().cpu()
    current_clip = current_clip_probs.detach().float().cpu()
    task_views = current_task_view_probs.detach().float().cpu()
    clip_views = current_clip_view_probs.detach().float().cpu()
    if task_views.dim() == 2:
        task_views = task_views.unsqueeze(0)
        clip_views = clip_views.unsqueeze(0)
    if historical_task.shape != historical_clip.shape:
        raise ValueError("historical Task/CLIP probability shapes must match")
    if current_task.shape != historical_task.shape or current_clip.shape != historical_task.shape:
        raise ValueError("historical/current probability shapes must match")
    if task_views.shape[1:] != current_task.shape or clip_views.shape != task_views.shape:
        raise ValueError("transition view probabilities must be [V,N,C]")

    historical_task_conf, candidate_a = historical_task.max(dim=1)
    historical_clip_conf, candidate_b = historical_clip.max(dim=1)
    historical_task_entropy = _entropy(historical_task)
    historical_clip_entropy = _entropy(historical_clip)
    historical_conflict = candidate_a != candidate_b

    anchor_mask = (
        (candidate_a == candidate_b)
        & (historical_task_conf >= float(anchor_task_conf))
        & (historical_clip_conf >= float(anchor_clip_conf))
        & (historical_task_entropy <= float(anchor_task_entropy))
        & (historical_clip_entropy <= float(anchor_clip_entropy))
    )
    if require_pre_post_prior_agreement:
        pre_task = historical_snapshot["pre_prior_task_probs"].argmax(dim=1)
        pre_clip = historical_snapshot["pre_prior_clip_probs"].argmax(dim=1)
        anchor_mask &= (pre_task == pre_clip) & (pre_task == candidate_a)

    anchor_count = int(anchor_mask.sum().item())
    empty = torch.zeros(0, 16, dtype=torch.float32)
    empty_target = torch.zeros(0, dtype=torch.long)
    if anchor_count < 2 or int(historical_conflict.sum().item()) == 0:
        return {
            "features": empty,
            "targets": empty_target,
            "selected_mask": torch.zeros(historical_task.size(0), dtype=torch.bool),
            "matured_label": torch.full((historical_task.size(0),), -1, dtype=torch.long),
            "historical_conflicts": int(historical_conflict.sum().item()),
            "matured": 0,
            "raw_choose_a": 0,
            "raw_choose_b": 0,
            "balanced_per_direction": 0,
            "anchor_count": anchor_count,
            "ready": False,
        }

    reliability = (
        historical_task_conf[anchor_mask]
        + historical_clip_conf[anchor_mask]
        - float(entropy_weight)
        * (
            historical_task_entropy[anchor_mask]
            + historical_clip_entropy[anchor_mask]
        )
    )
    anchor_labels = candidate_a[anchor_mask].long()
    pool_ids = torch.arange(anchor_count)
    task_bank = ClassBalancedAnchorBank(
        num_classes=num_classes,
        anchors_per_class=anchors_per_class,
        feature_dim=historical_task_feature.size(1),
        seed=seed,
        device=torch.device("cpu"),
    )
    clip_bank = ClassBalancedAnchorBank(
        num_classes=num_classes,
        anchors_per_class=anchors_per_class,
        feature_dim=historical_clip_feature.size(1),
        seed=seed,
        device=torch.device("cpu"),
    )
    task_bank.update(
        historical_task_feature[anchor_mask], anchor_labels, reliability, pool_ids
    )
    clip_bank.update(
        historical_clip_feature[anchor_mask], anchor_labels, reliability, pool_ids
    )

    current_task_top1 = current_task.argmax(dim=1)
    current_clip_top1 = current_clip.argmax(dim=1)
    matured_label = current_task_top1.clone()
    weak_agreement = current_task_top1 == current_clip_top1
    in_original_pair = (matured_label == candidate_a) | (matured_label == candidate_b)
    task_view_support = (
        task_views.argmax(dim=2) == matured_label.unsqueeze(0)
    ).float().mean(dim=0)
    clip_view_support = (
        clip_views.argmax(dim=2) == matured_label.unsqueeze(0)
    ).float().mean(dim=0)
    selected_mask = (
        historical_conflict
        & weak_agreement
        & in_original_pair
        & (task_view_support >= float(min_view_agreement))
        & (clip_view_support >= float(min_view_agreement))
    )
    selected_positions = torch.nonzero(selected_mask, as_tuple=False).flatten()
    targets = (matured_label[selected_positions] == candidate_b[selected_positions]).long()
    raw_choose_a = int((targets == 0).sum().item())
    raw_choose_b = int((targets == 1).sum().item())
    balance_n = min(raw_choose_a, raw_choose_b)
    ready = balance_n >= int(min_per_direction)
    # Keep every matured real conflict. Direction balancing belongs in the
    # minibatch sampler; downsampling here would discard most high-precision
    # samples whenever one transition direction is naturally more common.
    training_positions = (
        selected_positions if ready else torch.zeros(0, dtype=torch.long)
    )
    if training_positions.numel() > 0:
        features = build_comparator_features(
            historical_task[training_positions],
            historical_clip[training_positions],
            historical_task_feature[training_positions],
            historical_clip_feature[training_positions],
            task_bank,
            clip_bank,
            class_a=candidate_a[training_positions],
            class_b=candidate_b[training_positions],
            sim_topk=sim_topk,
        ).cpu()
        balanced_targets = (
            matured_label[training_positions] == candidate_b[training_positions]
        ).long()
    else:
        features = empty
        balanced_targets = empty_target
    return {
        "features": features,
        "targets": balanced_targets,
        "selected_mask": selected_mask,
        "matured_label": matured_label,
        "historical_candidate_a": candidate_a,
        "historical_candidate_b": candidate_b,
        "historical_conflicts": int(historical_conflict.sum().item()),
        "matured": int(selected_mask.sum().item()),
        "raw_choose_a": raw_choose_a,
        "raw_choose_b": raw_choose_b,
        "balanced_per_direction": int(balance_n if ready else 0),
        "anchor_count": anchor_count,
        "task_view_support": task_view_support,
        "clip_view_support": clip_view_support,
        "ready": bool(ready),
    }


@torch.no_grad()
def fuse_transition_comparator_vote(
    committee: dict,
    comparator_prob_a: torch.Tensor,
    weak_task_probs: torch.Tensor,
    weak_clip_probs: torch.Tensor,
    *,
    comparator_weight: float,
    coverage_fraction: float,
) -> dict:
    """Fuse historical real-conflict learning with the current committee."""
    if not 0.0 <= float(comparator_weight) <= 1.0:
        raise ValueError("comparator_weight must be in [0, 1]")
    total = int(committee["q"].numel())
    if comparator_prob_a.numel() != total:
        raise ValueError("transition comparator vote must match committee rows")
    committee_q = committee["q"].detach().float().cpu()
    comparator_q = comparator_prob_a.detach().float().cpu()
    q = (
        (1.0 - float(comparator_weight)) * committee_q
        + float(comparator_weight) * comparator_q
    ).clamp(_EPS, 1.0 - _EPS)
    direction_agreement = (
        (committee_q - 0.5) * (comparator_q - 0.5) >= 0.0
    ).float()
    source_agreement = committee["source_agreement"].detach().float().cpu()
    reliability = (
        2.0 * (q - 0.5).abs()
        * source_agreement
        * (0.5 + 0.5 * direction_agreement)
    ).clamp(0.0, 1.0)
    selected_count = max(
        1, min(total, int(round(total * float(coverage_fraction))))
    )
    order = torch.argsort(reliability, descending=True, stable=True)
    active = torch.zeros(total, dtype=torch.bool)
    active[order[:selected_count]] = True
    weight = torch.zeros(total, dtype=torch.float32)
    weight[active] = 0.25 + 0.75 * reliability[active]
    task = weak_task_probs.detach().float().cpu()
    clip = weak_clip_probs.detach().float().cpu()
    candidate_a = committee["candidate_a"].detach().long().cpu()
    candidate_b = committee["candidate_b"].detach().long().cpu()
    rows = torch.arange(total)
    target = 0.5 * (task + clip)
    pair_mass = target[rows, candidate_a] + target[rows, candidate_b]
    target[rows, candidate_a] = pair_mass * q
    target[rows, candidate_b] = pair_mass * (1.0 - q)
    fused = dict(committee)
    fused.update(
        {
            "active": active,
            "target": target,
            "q": q,
            "weight": weight,
            "reliability": reliability,
            "transition_q": comparator_q,
            "transition_committee_agreement": direction_agreement,
        }
    )
    return fused


def _rowwise_js(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.float().clamp_min(_EPS)
    second = second.float().clamp_min(_EPS)
    midpoint = 0.5 * (first + second)
    return 0.5 * (
        (first * (first.log() - midpoint.log())).sum(dim=1)
        + (second * (second.log() - midpoint.log())).sum(dim=1)
    )


@torch.no_grad()
def _anchor_neighbor_distribution(
    query_features: torch.Tensor,
    bank: ClassBalancedAnchorBank,
    *,
    num_classes: int,
    neighbors: int,
) -> torch.Tensor:
    anchor_features, anchor_labels, _, _, _ = bank.flatten()
    if anchor_features.size(0) == 0:
        return torch.full(
            (query_features.size(0), num_classes),
            1.0 / num_classes,
            dtype=torch.float32,
            device=query_features.device,
        )
    query = F.normalize(query_features.detach().float(), dim=1)
    anchors = F.normalize(anchor_features.detach().float(), dim=1).to(query.device)
    labels = anchor_labels.detach().long().to(query.device)
    k = max(1, min(int(neighbors), anchors.size(0)))
    similarities, positions = (query @ anchors.t()).topk(k, dim=1)
    weights = F.softmax(similarities / 0.07, dim=1)
    result = torch.zeros(
        query.size(0), num_classes, dtype=torch.float32, device=query.device
    )
    result.scatter_add_(1, labels[positions], weights)
    return result.clamp_min(_EPS)


@torch.no_grad()
def build_reliability_gated_fusion(
    weak_task_probs: torch.Tensor,
    weak_clip_probs: torch.Tensor,
    task_view_probs: torch.Tensor,
    clip_view_probs: torch.Tensor,
    weak_task_features: torch.Tensor,
    weak_clip_features: torch.Tensor,
    task_view_features: torch.Tensor,
    clip_view_features: torch.Tensor,
    task_bank: ClassBalancedAnchorBank,
    clip_bank: ClassBalancedAnchorBank,
    *,
    num_classes: int,
    neighbors: int,
    temperature: float,
    coverage_fraction: float,
) -> dict:
    """Candidate-level committee over real Task/CLIP conflicts.

    ``A`` and ``B`` are the weak Task and CLIP top-1 classes.  Weak and
    stochastic-view posteriors vote on A/B independently, as do Task/CLIP
    anchor neighborhoods.  This keeps class identity (unlike a single
    branch-reliability scalar) and lets several independent observations
    overturn the equal-fusion fallback.  Target labels are never used.
    """
    if float(temperature) <= 0.0:
        raise ValueError("RELIABILITY_GATE_TEMPERATURE must be positive")
    if not 0.0 < float(coverage_fraction) <= 1.0:
        raise ValueError(
            "RELIABILITY_GATE_COVERAGE_FRACTION must satisfy 0 < value <= 1"
        )
    if weak_task_probs.shape != weak_clip_probs.shape:
        raise ValueError("weak Task/CLIP probabilities must have equal shape")
    if task_view_probs.dim() == 2:
        task_view_probs = task_view_probs.unsqueeze(0)
        clip_view_probs = clip_view_probs.unsqueeze(0)
        task_view_features = task_view_features.unsqueeze(0)
        clip_view_features = clip_view_features.unsqueeze(0)
    expected_view_shape = (
        task_view_probs.size(0),
        weak_task_probs.size(0),
        weak_task_probs.size(1),
    )
    if task_view_probs.shape != expected_view_shape or clip_view_probs.shape != expected_view_shape:
        raise ValueError("reliability-gate view probabilities must be [V,N,C]")
    if task_view_features.dim() != 3 or clip_view_features.dim() != 3:
        raise ValueError("reliability-gate view features must be [V,N,D]")
    if task_view_features.shape[:2] != expected_view_shape[:2] or clip_view_features.shape[:2] != expected_view_shape[:2]:
        raise ValueError("reliability-gate view features must match [V,N]")
    total = int(weak_task_probs.size(0))
    if total == 0:
        return {
            "active": torch.zeros(0, dtype=torch.bool),
            "target": weak_task_probs.new_zeros(0, num_classes),
            "candidate_a": torch.zeros(0, dtype=torch.long),
            "candidate_b": torch.zeros(0, dtype=torch.long),
            "q": torch.zeros(0),
            "baseline_q": torch.zeros(0),
            "baseline_pair_mass": torch.zeros(0),
            "weight": torch.zeros(0),
            "reliability": torch.zeros(0),
            "source_agreement": torch.zeros(0),
        }

    device = weak_task_probs.device
    weak_task = weak_task_probs.detach().float().to(device)
    weak_clip = weak_clip_probs.detach().float().to(device)
    task_views = task_view_probs.detach().float().to(device)
    clip_views = clip_view_probs.detach().float().to(device)
    task_neighbor_weak = _anchor_neighbor_distribution(
        weak_task_features.to(device), task_bank,
        num_classes=num_classes, neighbors=neighbors,
    )
    clip_neighbor_weak = _anchor_neighbor_distribution(
        weak_clip_features.to(device), clip_bank,
        num_classes=num_classes, neighbors=neighbors,
    )
    task_neighbors = [task_neighbor_weak]
    clip_neighbors = [clip_neighbor_weak]
    for view in range(task_views.size(0)):
        task_neighbors.append(
            _anchor_neighbor_distribution(
                task_view_features[view].to(device), task_bank,
                num_classes=num_classes, neighbors=neighbors,
            )
        )
        clip_neighbors.append(
            _anchor_neighbor_distribution(
                clip_view_features[view].to(device), clip_bank,
                num_classes=num_classes, neighbors=neighbors,
            )
        )

    candidate_a = weak_task.argmax(dim=1)
    candidate_b = weak_clip.argmax(dim=1)
    if bool((candidate_a == candidate_b).any().item()):
        raise ValueError(
            "candidate-evidence committee expects strict Task/CLIP conflicts"
        )
    rows = torch.arange(total, device=device)

    def pair_probability(distribution: torch.Tensor) -> torch.Tensor:
        prob_a = distribution[rows, candidate_a]
        prob_b = distribution[rows, candidate_b]
        return (prob_a + _EPS) / (prob_a + prob_b + 2.0 * _EPS)

    # This is the A-vs-B target already supplied by DUET's original CLIP KL.
    # The auxiliary loss can therefore inject only q - baseline_q instead of
    # training the same conflict a second time with a complete soft CE.
    baseline_q = pair_probability(weak_clip)
    baseline_pair_mass = (
        weak_clip[rows, candidate_a] + weak_clip[rows, candidate_b]
    )

    # Keep the two networks balanced: each contributes one vote per view.
    posterior_sources = [weak_task, weak_clip]
    posterior_sources.extend(task_views[view] for view in range(task_views.size(0)))
    posterior_sources.extend(clip_views[view] for view in range(clip_views.size(0)))
    neighbor_sources = task_neighbors + clip_neighbors
    posterior_votes = torch.stack(
        [pair_probability(source) for source in posterior_sources]
    )
    neighbor_votes = torch.stack(
        [pair_probability(source) for source in neighbor_sources]
    )
    posterior_q = posterior_votes.mean(dim=0)
    neighbor_q = neighbor_votes.mean(dim=0)
    raw_q = 0.5 * (posterior_q + neighbor_q)

    all_votes = torch.cat([posterior_votes, neighbor_votes], dim=0)
    vote_a_fraction = (all_votes >= 0.5).float().mean(dim=0)
    source_agreement = torch.maximum(vote_a_fraction, 1.0 - vote_a_fraction)
    # A low temperature sharpens the committee belief for the auxiliary loss
    # without changing which candidate wins or which samples are selected.
    q = torch.sigmoid(
        torch.logit(raw_q.clamp(_EPS, 1.0 - _EPS)) / float(temperature)
    )
    reliability = (
        (2.0 * (raw_q - 0.5).abs())
        * source_agreement
        * (1.0 - 0.5 * (posterior_q - neighbor_q).abs())
    ).clamp(0.0, 1.0)

    # Preserve every non-candidate class from the exact DUET fallback and
    # reallocate only the A/B probability mass according to the committee.
    fallback = 0.5 * (weak_task + weak_clip)
    target = fallback.clone()
    pair_mass = fallback[rows, candidate_a] + fallback[rows, candidate_b]
    target[rows, candidate_a] = pair_mass * q
    target[rows, candidate_b] = pair_mass * (1.0 - q)
    selected_count = max(
        1, min(total, int(round(total * float(coverage_fraction))))
    )
    order = torch.argsort(reliability, descending=True, stable=True)
    active = torch.zeros(total, dtype=torch.bool, device=device)
    active[order[:selected_count]] = True
    # Selected samples always retain a small training weight; reliability then
    # scales genuinely decisive committee observations up to full strength.
    weight = torch.zeros(total, dtype=torch.float32, device=device)
    weight[active] = 0.25 + 0.75 * reliability[active]
    return {
        "active": active.cpu(),
        "target": target.cpu(),
        "candidate_a": candidate_a.cpu(),
        "candidate_b": candidate_b.cpu(),
        "q": q.cpu(),
        "baseline_q": baseline_q.cpu(),
        "baseline_pair_mass": baseline_pair_mass.cpu(),
        "weight": weight.cpu(),
        "reliability": reliability.cpu(),
        "source_agreement": source_agreement.cpu(),
    }


@torch.no_grad()
def _log_agreement_synthetic_feasibility_eval_only(
    pool_labels: torch.Tensor,
    pool_strong_task_probs: torch.Tensor,
    pool_strong_clip_probs: torch.Tensor,
    pool_strong_task_features: torch.Tensor,
    pool_strong_clip_features: torch.Tensor,
    *,
    task_bank: ClassBalancedAnchorBank,
    clip_bank: ClassBalancedAnchorBank,
    sim_topk: int,
    fractions: list[int],
    cycle: int,
    log_fn: Callable[[str], None],
    pool_gt_labels: Optional[torch.Tensor] = None,
) -> dict:
    """Check whether strong views provide candidate-semantic A/B supervision.

    ``pool_labels`` is the high-confidence weak-view consensus pseudo label Y.
    For rows where both strong branches share Top1=A and Top2=B, Y=A gives a
    choose-A target and Y=B gives a choose-B target.  Construction and ranking
    never read GT; optional GT is consulted only after selection to report the
    pseudo-target precision.  Nothing returned here is consumed by training.
    """
    if not fractions or any(int(value) <= 0 or int(value) > 100 for value in fractions):
        raise ValueError(
            "agreement synthetic feasibility fractions must be in [1, 100]"
        )

    labels_cpu = pool_labels.detach().long().cpu()
    task_probs = pool_strong_task_probs.detach().float().cpu()
    clip_probs = pool_strong_clip_probs.detach().float().cpu()
    task_features = pool_strong_task_features.detach().float().cpu()
    clip_features = pool_strong_clip_features.detach().float().cpu()
    pool_size = int(labels_cpu.numel())
    if task_probs.dim() != 2 or clip_probs.shape != task_probs.shape:
        raise ValueError("strong Task/CLIP probabilities must have equal [N, C] shape")
    if task_probs.size(0) != pool_size or task_probs.size(1) < 2:
        raise ValueError("strong probabilities must match pool labels and have >= 2 classes")
    if task_features.size(0) != pool_size or clip_features.size(0) != pool_size:
        raise ValueError("strong features and pool labels must have equal N")
    gt_cpu = None
    if pool_gt_labels is not None:
        gt_cpu = pool_gt_labels.detach().long().cpu()
        if gt_cpu.shape != labels_cpu.shape:
            raise ValueError("pool_gt_labels must have the same shape as pool_labels")

    _, task_indices = torch.topk(task_probs, k=2, dim=1)
    _, clip_indices = torch.topk(clip_probs, k=2, dim=1)
    shared_mask = (
        (task_indices[:, 0] == clip_indices[:, 0])
        & (task_indices[:, 1] == clip_indices[:, 1])
    )
    shared_positions = torch.nonzero(shared_mask, as_tuple=False).flatten()
    shared_count = int(shared_positions.numel())
    shared_rate = 100.0 * shared_count / pool_size if pool_size > 0 else float("nan")
    if shared_count == 0:
        log_fn(
            "DUET agreement synthetic feasibility summary eval-only: cycle={}; "
            "anchor_pool_total={}; strong_shared_pair_count=0; "
            "strong_shared_pair_rate={:.2f}%; pseudo_Y_is_A_count=0; "
            "pseudo_Y_is_B_count=0; pseudo_Y_neither_count=0; usable_count=0; "
            "choose_B_share=nan; pseudo_target_precision=nan; "
            "ambiguous_25_count=0; construction_uses_gt=False; "
            "training_changed=False; ground_truth_affects_training=False".format(
                cycle, pool_size, shared_rate
            )
        )
        return {
            "anchor_pool_total": pool_size,
            "strong_shared_pair_count": 0,
            "pseudo_y_is_a_count": 0,
            "pseudo_y_is_b_count": 0,
            "pseudo_y_neither_count": 0,
            "usable_count": 0,
            "fractions": [],
        }

    a = task_indices[shared_positions, 0]
    b = task_indices[shared_positions, 1]
    pseudo_y = labels_cpu[shared_positions]
    choose_a = pseudo_y == a
    choose_b = pseudo_y == b
    usable = choose_a | choose_b
    choose_a_count = int(choose_a.sum().item())
    choose_b_count = int(choose_b.sum().item())
    neither_count = shared_count - choose_a_count - choose_b_count
    usable_count = choose_a_count + choose_b_count

    row_ids = torch.arange(shared_count)
    task_a = task_probs[shared_positions][row_ids, a]
    task_b = task_probs[shared_positions][row_ids, b]
    clip_a = clip_probs[shared_positions][row_ids, a]
    clip_b = clip_probs[shared_positions][row_ids, b]
    task_gap = task_a - task_b
    clip_gap = clip_a - clip_b
    ambiguity_gap = torch.maximum(task_gap, clip_gap)
    order = torch.argsort(ambiguity_gap, descending=False, stable=True)

    usable_local = torch.nonzero(usable, as_tuple=False).flatten()
    if usable_count > 0:
        usable_features = build_comparator_features(
            task_probs[shared_positions[usable_local]],
            clip_probs[shared_positions[usable_local]],
            task_features[shared_positions[usable_local]],
            clip_features[shared_positions[usable_local]],
            task_bank,
            clip_bank,
            class_a=a[usable_local],
            class_b=b[usable_local],
            sim_topk=sim_topk,
        )
        usable_choose_a = choose_a[usable_local]
        usable_choose_b = choose_b[usable_local]
        _log_pair_distribution(
            usable_features[usable_choose_a],
            "agreement-synthetic-choose-A",
            cycle,
            log_fn,
        )
        _log_pair_distribution(
            usable_features[usable_choose_b],
            "agreement-synthetic-choose-B",
            cycle,
            log_fn,
        )

    rows: list[dict] = []
    selected_25_local = torch.zeros(0, dtype=torch.long)
    for fraction in fractions:
        fraction = int(fraction)
        selected_count = max(
            1,
            min(shared_count, int(round(shared_count * fraction / 100.0))),
        )
        selected = order[:selected_count]
        selected_choose_a = choose_a[selected]
        selected_choose_b = choose_b[selected]
        selected_usable = selected_choose_a | selected_choose_b
        selected_a_count = int(selected_choose_a.sum().item())
        selected_b_count = int(selected_choose_b.sum().item())
        selected_usable_count = selected_a_count + selected_b_count
        selected_neither = selected_count - selected_usable_count
        if fraction == 25:
            selected_25_local = selected[selected_usable]

        target_precision = float("nan")
        choose_a_precision = float("nan")
        choose_b_precision = float("nan")
        if gt_cpu is not None and selected_usable_count > 0:
            selected_gt = gt_cpu[shared_positions[selected]]
            selected_y = pseudo_y[selected]
            pseudo_correct = selected_gt == selected_y
            target_precision = 100.0 * float(
                pseudo_correct[selected_usable].float().mean().item()
            )
            if selected_a_count > 0:
                choose_a_precision = 100.0 * float(
                    pseudo_correct[selected_choose_a].float().mean().item()
                )
            if selected_b_count > 0:
                choose_b_precision = 100.0 * float(
                    pseudo_correct[selected_choose_b].float().mean().item()
                )

        row = {
            "fraction": fraction,
            "count": selected_count,
            "choose_a_count": selected_a_count,
            "choose_b_count": selected_b_count,
            "neither_count": selected_neither,
            "usable_count": selected_usable_count,
            "usable_rate": 100.0 * selected_usable_count / selected_count,
            "choose_b_share": (
                100.0 * selected_b_count / selected_usable_count
                if selected_usable_count > 0
                else float("nan")
            ),
            "pseudo_target_precision": target_precision,
            "choose_a_pseudo_precision": choose_a_precision,
            "choose_b_pseudo_precision": choose_b_precision,
            "task_gap_mean": float(task_gap[selected].mean().item()),
            "clip_gap_mean": float(clip_gap[selected].mean().item()),
            "ambiguity_gap_mean": float(ambiguity_gap[selected].mean().item()),
            "task_a_prob_mean": float(task_a[selected].mean().item()),
            "task_b_prob_mean": float(task_b[selected].mean().item()),
            "clip_a_prob_mean": float(clip_a[selected].mean().item()),
            "clip_b_prob_mean": float(clip_b[selected].mean().item()),
        }
        rows.append(row)
        log_fn(
            "DUET agreement synthetic feasibility fraction eval-only: cycle={}; "
            "fraction={}%; n={}; pseudo_Y_is_A_count={}; "
            "pseudo_Y_is_B_count={}; pseudo_Y_neither_count={}; usable_count={}; "
            "usable_rate={:.2f}%; choose_B_share={:.2f}%; "
            "pseudo_target_precision={:.2f}%; choose_A_pseudo_precision={:.2f}%; "
            "choose_B_pseudo_precision={:.2f}%; task_gap_mean={:.6f}; "
            "clip_gap_mean={:.6f}; ambiguity_gap_mean={:.6f}; "
            "Task_A_prob_mean={:.6f}; Task_B_prob_mean={:.6f}; "
            "CLIP_A_prob_mean={:.6f}; CLIP_B_prob_mean={:.6f}; "
            "construction_uses_gt=False; training_changed=False; "
            "ground_truth_affects_training=False".format(
                cycle,
                fraction,
                selected_count,
                selected_a_count,
                selected_b_count,
                selected_neither,
                selected_usable_count,
                row["usable_rate"],
                row["choose_b_share"],
                row["pseudo_target_precision"],
                row["choose_a_pseudo_precision"],
                row["choose_b_pseudo_precision"],
                row["task_gap_mean"],
                row["clip_gap_mean"],
                row["ambiguity_gap_mean"],
                row["task_a_prob_mean"],
                row["task_b_prob_mean"],
                row["clip_a_prob_mean"],
                row["clip_b_prob_mean"],
            )
        )

    if selected_25_local.numel() > 0:
        tail_features = build_comparator_features(
            task_probs[shared_positions[selected_25_local]],
            clip_probs[shared_positions[selected_25_local]],
            task_features[shared_positions[selected_25_local]],
            clip_features[shared_positions[selected_25_local]],
            task_bank,
            clip_bank,
            class_a=a[selected_25_local],
            class_b=b[selected_25_local],
            sim_topk=sim_topk,
        )
        _log_pair_distribution(
            tail_features,
            "agreement-synthetic-ambiguous-25",
            cycle,
            log_fn,
        )

    row_25 = next((row for row in rows if row["fraction"] == 25), None)
    summary_25 = (
        "ambiguous_25_count=0; usable_25_count=0; choose_B_25_count=0; "
        "choose_B_25_share=nan; pseudo_target_25_precision=nan"
        if row_25 is None
        else (
            "ambiguous_25_count={}; usable_25_count={}; choose_B_25_count={}; "
            "choose_B_25_share={:.2f}%; pseudo_target_25_precision={:.2f}%"
        ).format(
            row_25["count"],
            row_25["usable_count"],
            row_25["choose_b_count"],
            row_25["choose_b_share"],
            row_25["pseudo_target_precision"],
        )
    )
    overall_precision = float("nan")
    if gt_cpu is not None and usable_count > 0:
        overall_precision = 100.0 * float(
            (gt_cpu[shared_positions][usable] == pseudo_y[usable]).float().mean().item()
        )
    log_fn(
        "DUET agreement synthetic feasibility summary eval-only: cycle={}; "
        "anchor_pool_total={}; strong_shared_pair_count={}; "
        "strong_shared_pair_rate={:.2f}%; pseudo_Y_is_A_count={}; "
        "pseudo_Y_is_B_count={}; pseudo_Y_neither_count={}; usable_count={}; "
        "usable_rate={:.2f}%; choose_B_share={:.2f}%; "
        "pseudo_target_precision={:.2f}%; {}; construction_uses_gt=False; "
        "training_changed=False; ground_truth_affects_training=False".format(
            cycle,
            pool_size,
            shared_count,
            shared_rate,
            choose_a_count,
            choose_b_count,
            neither_count,
            usable_count,
            100.0 * usable_count / shared_count,
            100.0 * choose_b_count / usable_count if usable_count > 0 else float("nan"),
            overall_precision,
            summary_25,
        )
    )
    return {
        "anchor_pool_total": pool_size,
        "strong_shared_pair_count": shared_count,
        "pseudo_y_is_a_count": choose_a_count,
        "pseudo_y_is_b_count": choose_b_count,
        "pseudo_y_neither_count": neither_count,
        "usable_count": usable_count,
        "fractions": rows,
    }


@torch.no_grad()
def _log_agreement_candidate_probe_eval_only(
    task_probs: torch.Tensor,
    clip_probs: torch.Tensor,
    task_features: torch.Tensor,
    clip_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    comparator: PairwiseConflictComparator,
    task_bank: ClassBalancedAnchorBank,
    clip_bank: ClassBalancedAnchorBank,
    sim_topk: int,
    fractions: list[int],
    cycle: int,
    log_fn: Callable[[str], None],
) -> dict:
    """Probe the trained strict-conflict comparator on agreement A/B pairs.

    A is the shared Top1 and B is the shared Top2.  Comparator output position
    0 is interpreted as choose A and position 1 as choose B only inside this
    diagnostic.  No gate, pseudo-label update, optimizer step or admission is
    performed, and GT is used only after the GT-free pair ranking/prediction.
    """
    if not fractions or any(int(value) <= 0 or int(value) > 100 for value in fractions):
        raise ValueError("agreement candidate probe fractions must be in [1, 100]")
    task_probs_cpu = task_probs.detach().float().cpu()
    clip_probs_cpu = clip_probs.detach().float().cpu()
    task_features_cpu = task_features.detach().float().cpu()
    clip_features_cpu = clip_features.detach().float().cpu()
    labels_cpu = labels.detach().long().cpu()
    if task_probs_cpu.dim() != 2 or clip_probs_cpu.shape != task_probs_cpu.shape:
        raise ValueError("task_probs and clip_probs must have equal [N, C] shape")
    if task_probs_cpu.size(1) < 2:
        raise ValueError("agreement candidate probe requires >= 2 classes")
    if task_features_cpu.size(0) != task_probs_cpu.size(0):
        raise ValueError("task_features and probabilities must have equal N")
    if clip_features_cpu.size(0) != task_probs_cpu.size(0):
        raise ValueError("clip_features and probabilities must have equal N")
    if labels_cpu.shape != (task_probs_cpu.size(0),):
        raise ValueError("labels must have shape [N]")

    _, task_indices = torch.topk(task_probs_cpu, k=2, dim=1)
    _, clip_indices = torch.topk(clip_probs_cpu, k=2, dim=1)
    shared_mask = (
        (task_indices[:, 0] == clip_indices[:, 0])
        & (task_indices[:, 1] == clip_indices[:, 1])
    )
    shared_positions = torch.nonzero(shared_mask, as_tuple=False).flatten()
    shared_count = int(shared_positions.numel())
    if shared_count == 0:
        log_fn(
            "DUET agreement candidate probe summary eval-only: cycle={}; "
            "shared_top2_count=0; ambiguous_25_count=0; comparator_acc=nan; "
            "comparator_gain_over_top1=nan; recovered_top1_errors=0; "
            "overridden_correct_top1=0; net_corrections=0; "
            "output_semantics=0_choose_A_1_choose_B; gate_used=False; "
            "admission_changed=False; selection_uses_gt=False; "
            "ground_truth_affects_training=False".format(cycle)
        )
        return {"shared_top2_count": 0, "fractions": []}

    a = task_indices[shared_positions, 0]
    b = task_indices[shared_positions, 1]
    row_ids = torch.arange(shared_count)
    task_a = task_probs_cpu[shared_positions][row_ids, a]
    task_b = task_probs_cpu[shared_positions][row_ids, b]
    clip_a = clip_probs_cpu[shared_positions][row_ids, a]
    clip_b = clip_probs_cpu[shared_positions][row_ids, b]
    task_gap = task_a - task_b
    clip_gap = clip_a - clip_b
    ambiguity_gap = torch.maximum(task_gap, clip_gap)
    order = torch.argsort(ambiguity_gap, descending=False, stable=True)

    pair_features = build_comparator_features(
        task_probs_cpu[shared_positions],
        clip_probs_cpu[shared_positions],
        task_features_cpu[shared_positions],
        clip_features_cpu[shared_positions],
        task_bank,
        clip_bank,
        class_a=a,
        class_b=b,
        sim_topk=sim_topk,
    )
    _log_pair_distribution(
        pair_features, "agreement-candidate-probe", cycle, log_fn
    )
    selected_25_count = max(
        1, min(shared_count, int(round(shared_count * 25 / 100.0)))
    )
    _log_pair_distribution(
        pair_features[order[:selected_25_count]],
        "agreement-candidate-probe-ambiguous-25",
        cycle,
        log_fn,
    )
    comparator_device = next(comparator.parameters()).device
    comparator.eval()
    logits = comparator(pair_features.to(comparator_device)).detach().float().cpu()
    probabilities = _softmax_probabilities(logits)
    choose_b = probabilities[:, 1] > probabilities[:, 0]
    margins = (probabilities[:, 0] - probabilities[:, 1]).abs()
    gt = labels_cpu[shared_positions]

    rows: list[dict] = []
    for fraction in fractions:
        fraction = int(fraction)
        selected_count = max(
            1,
            min(
                shared_count,
                int(round(shared_count * fraction / 100.0)),
            ),
        )
        selected = order[:selected_count]
        selected_a = a[selected]
        selected_b = b[selected]
        selected_gt = gt[selected]
        selected_choose_b = choose_b[selected]
        chosen = torch.where(
            selected_choose_b, selected_b, selected_a
        )
        top1_correct = selected_gt == selected_a
        top2_correct = selected_gt == selected_b
        oracle_correct = top1_correct | top2_correct
        comparator_correct = chosen == selected_gt
        top1_count = int(top1_correct.sum().item())
        top2_count = int(top2_correct.sum().item())
        oracle_count = int(oracle_correct.sum().item())
        comparator_count = int(comparator_correct.sum().item())
        choose_b_count = int(selected_choose_b.sum().item())
        recovered = int((selected_choose_b & top2_correct).sum().item())
        overridden = int((selected_choose_b & top1_correct).sum().item())
        neither_count = selected_count - oracle_count
        row = {
            "fraction": fraction,
            "count": selected_count,
            "top1_acc": 100.0 * top1_count / selected_count,
            "top2_acc": 100.0 * top2_count / selected_count,
            "comparator_acc": 100.0 * comparator_count / selected_count,
            "candidate_oracle_acc": 100.0 * oracle_count / selected_count,
            "conditional_arbitration_acc": (
                100.0 * comparator_count / oracle_count
                if oracle_count > 0
                else float("nan")
            ),
            "neither_rate": 100.0 * neither_count / selected_count,
            "choose_a_count": selected_count - choose_b_count,
            "choose_b_count": choose_b_count,
            "choose_b_precision": (
                100.0 * recovered / choose_b_count
                if choose_b_count > 0
                else float("nan")
            ),
            "recovered_top1_errors": recovered,
            "overridden_correct_top1": overridden,
            "net_corrections": recovered - overridden,
            "comparator_gain_over_top1": (
                100.0 * (comparator_count - top1_count) / selected_count
            ),
            "margin_mean": float(margins[selected].mean().item()),
        }
        rows.append(row)
        log_fn(
            "DUET agreement candidate probe fraction eval-only: cycle={}; "
            "fraction={}%; n={}; top1_acc={:.2f}%; top2_acc={:.2f}%; "
            "comparator_acc={:.2f}%; comparator_gain_over_top1={:.2f}pp; "
            "candidate_oracle_acc={:.2f}%; conditional_arbitration_acc={:.2f}%; "
            "neither_rate={:.2f}%; choose_A_count={}; choose_B_count={}; "
            "choose_B_precision={:.2f}%; recovered_top1_errors={}; "
            "overridden_correct_top1={}; net_corrections={}; "
            "comparator_margin_mean={:.6f}; "
            "output_semantics=0_choose_A_1_choose_B; gate_used=False; "
            "admission_changed=False; selection_uses_gt=False; "
            "ground_truth_affects_training=False".format(
                cycle,
                fraction,
                selected_count,
                row["top1_acc"],
                row["top2_acc"],
                row["comparator_acc"],
                row["comparator_gain_over_top1"],
                row["candidate_oracle_acc"],
                row["conditional_arbitration_acc"],
                row["neither_rate"],
                row["choose_a_count"],
                row["choose_b_count"],
                row["choose_b_precision"],
                row["recovered_top1_errors"],
                row["overridden_correct_top1"],
                row["net_corrections"],
                row["margin_mean"],
            )
        )

    row_25 = next((row for row in rows if row["fraction"] == 25), None)
    if row_25 is None:
        summary = (
            "ambiguous_25_count=0; comparator_acc=nan; "
            "comparator_gain_over_top1=nan; recovered_top1_errors=0; "
            "overridden_correct_top1=0; net_corrections=0"
        )
    else:
        summary = (
            "ambiguous_25_count={}; top1_acc={:.2f}%; top2_acc={:.2f}%; "
            "comparator_acc={:.2f}%; comparator_gain_over_top1={:.2f}pp; "
            "candidate_oracle_acc={:.2f}%; conditional_arbitration_acc={:.2f}%; "
            "recovered_top1_errors={}; overridden_correct_top1={}; "
            "net_corrections={}"
        ).format(
            row_25["count"],
            row_25["top1_acc"],
            row_25["top2_acc"],
            row_25["comparator_acc"],
            row_25["comparator_gain_over_top1"],
            row_25["candidate_oracle_acc"],
            row_25["conditional_arbitration_acc"],
            row_25["recovered_top1_errors"],
            row_25["overridden_correct_top1"],
            row_25["net_corrections"],
        )
    log_fn(
        "DUET agreement candidate probe summary eval-only: cycle={}; "
        "shared_top2_count={}; {}; output_semantics=0_choose_A_1_choose_B; "
        "gate_used=False; admission_changed=False; selection_uses_gt=False; "
        "ground_truth_affects_training=False".format(
            cycle, shared_count, summary
        )
    )
    return {"shared_top2_count": shared_count, "fractions": rows}


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


class PersistentConflictBeliefMemory:
    """Per-image GT-free soft belief for Task-vs-CLIP candidates.

    History is accumulated only while the ordered pair remains exactly
    ``A=Task Top1, B=CLIP Top1``. A changed pair resets that image's record,
    so probabilities with different candidate semantics are never averaged.
    """

    def __init__(self) -> None:
        self.records = {}

    def state_dict(self) -> dict:
        return copy.deepcopy(self.records)

    def load_state_dict(self, state: Optional[dict]) -> None:
        self.records = copy.deepcopy(state or {})

    @torch.no_grad()
    def update(
        self,
        sample_indices: torch.Tensor,
        candidate_a: torch.Tensor,
        candidate_b: torch.Tensor,
        current_q: torch.Tensor,
        view_reliability: torch.Tensor,
        *,
        cycle: int,
        coverage_fraction: float,
    ) -> dict:
        """Update memory and select a fixed reliability-ranked conflict pool."""
        total = int(sample_indices.numel())
        tensors = (candidate_a, candidate_b, current_q, view_reliability)
        if any(tensor.shape != (total,) for tensor in tensors):
            raise ValueError("conflict-memory inputs must all have shape [N]")
        if not 0.0 < float(coverage_fraction) <= 1.0:
            raise ValueError(
                "CONFLICT_MEMORY_COVERAGE_FRACTION must satisfy 0 < value <= 1"
            )
        sample_indices = sample_indices.detach().long().cpu()
        if sample_indices.unique().numel() != total:
            raise ValueError("conflict-memory sample_indices must be unique")
        candidate_a = candidate_a.detach().long().cpu()
        candidate_b = candidate_b.detach().long().cpu()
        current_q = current_q.detach().float().cpu().clamp(0.0, 1.0)
        view_reliability = (
            view_reliability.detach().float().cpu().clamp(0.0, 1.0)
        )

        memory_q = torch.zeros(total, dtype=torch.float32)
        weights = torch.zeros(total, dtype=torch.float32)
        observations = torch.zeros(total, dtype=torch.long)
        temporal_reliability = torch.ones(total, dtype=torch.float32)
        reset_count = 0
        for row in range(total):
            sample_id = int(sample_indices[row].item())
            a = int(candidate_a[row].item())
            b = int(candidate_b[row].item())
            q_now = float(current_q[row].item())
            view_now = float(view_reliability[row].item())
            direction_now = int(q_now < 0.5)
            record = self.records.get(sample_id)
            same_pair = bool(
                record is not None
                and int(record["candidate_a"]) == a
                and int(record["candidate_b"]) == b
                and int(record["last_cycle"]) == int(cycle) - 1
            )
            if not same_pair:
                if record is not None:
                    reset_count += 1
                record = {
                    "candidate_a": a,
                    "candidate_b": b,
                    "q_mean": q_now,
                    "view_mean": view_now,
                    "observations": 1,
                    "direction_matches": 0,
                    "last_direction": direction_now,
                    "last_cycle": int(cycle),
                }
            else:
                old_n = int(record["observations"])
                record["q_mean"] = (
                    old_n * float(record["q_mean"]) + q_now
                ) / float(old_n + 1)
                record["view_mean"] = (
                    old_n * float(record["view_mean"]) + view_now
                ) / float(old_n + 1)
                record["direction_matches"] = int(
                    record["direction_matches"]
                ) + int(int(record["last_direction"]) == direction_now)
                record["observations"] = old_n + 1
                record["last_direction"] = direction_now
                record["last_cycle"] = int(cycle)
            self.records[sample_id] = record

            n_observations = int(record["observations"])
            temporal = (
                1.0
                if n_observations == 1
                else float(record["direction_matches"])
                / float(n_observations - 1)
            )
            q_value = float(record["q_mean"])
            confidence = 2.0 * abs(q_value - 0.5)
            memory_q[row] = q_value
            observations[row] = n_observations
            temporal_reliability[row] = temporal
            weights[row] = confidence * float(record["view_mean"]) * temporal

        selected_count = (
            max(1, min(total, int(round(total * float(coverage_fraction)))))
            if total > 0
            else 0
        )
        active = torch.zeros(total, dtype=torch.bool)
        if selected_count > 0:
            order = torch.argsort(weights, descending=True, stable=True)
            active[order[:selected_count]] = True
        return {
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "q": memory_q,
            "weight": weights,
            "active": active,
            "observations": observations,
            "temporal_reliability": temporal_reliability,
            "pair_resets": int(reset_count),
        }


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
    balance_current_directions: bool = False,
    trajectory_features: Optional[torch.Tensor] = None,
    trajectory_interval: int = 0,
    trajectory_sink: Optional[list[dict]] = None,
) -> Optional[float]:
    """在 synthetic conflict 对上训练 comparator（2-way CE）。

    与 Run 9/10 的训练方式完全一致：matched synthetic 已经 1:1 平衡，
    池子 <= batch_size 时每个 step 直接用全部样本，否则随机抽 batch。
    提供 ``memory_features/memory_targets`` 时启用 replay：每个 step 按
    ``memory_fraction`` 从历史 memory 采样、其余从当前 matched synthetic
    采样（persistent + replay 实验）。``balance_current_directions=True``
    时每个 batch 等量抽取两个方向，但不删除 majority 样本。
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
    direction_zero = torch.nonzero(
        targets.detach().long().to(features.device) == 0,
        as_tuple=False,
    ).flatten()
    direction_one = torch.nonzero(
        targets.detach().long().to(features.device) == 1,
        as_tuple=False,
    ).flatten()
    if balance_current_directions and (
        direction_zero.numel() == 0 or direction_one.numel() == 0
    ):
        raise ValueError(
            "balanced comparator sampling requires both target directions"
        )
    direction_states = {
        0: {"pool": direction_zero, "order": None, "cursor": 0},
        1: {"pool": direction_one, "order": None, "cursor": 0},
    }

    def sample_direction(direction: int, count: int) -> torch.Tensor:
        """Cycle through shuffled direction pools before reusing a row."""
        state = direction_states[direction]
        pieces = []
        remaining = int(count)
        while remaining > 0:
            if (
                state["order"] is None
                or state["cursor"] >= state["order"].numel()
            ):
                state["order"] = state["pool"][
                    torch.randperm(
                        state["pool"].numel(),
                        generator=generator,
                        device=features.device,
                    )
                ]
                state["cursor"] = 0
            available = state["order"].numel() - state["cursor"]
            take = min(remaining, available)
            pieces.append(
                state["order"][state["cursor"] : state["cursor"] + take]
            )
            state["cursor"] += take
            remaining -= take
        return torch.cat(pieces)
    use_memory = (
        memory_features is not None
        and memory_targets is not None
        and memory_features.size(0) >= 1
    )
    minimum_current = 2 if balance_current_directions else 1
    n_current = max(minimum_current, int(batch_size))
    if use_memory:
        combined_features = torch.cat(
            [features, memory_features.detach().to(features.device)], dim=0
        )
        combined_targets = torch.cat(
            [targets, memory_targets.detach().to(features.device)], dim=0
        )
        n_memory = max(1, int(round(batch_size * memory_fraction)))
        n_current = max(minimum_current, batch_size - n_memory)
    total_loss = 0.0
    counted = 0
    total_steps = max(1, int(steps))
    for step in range(1, total_steps + 1):
        if use_memory:
            if balance_current_directions:
                n_zero = n_current // 2
                n_one = n_current - n_zero
                current_indices = torch.cat(
                    [
                        sample_direction(0, n_zero),
                        sample_direction(1, n_one),
                    ]
                )
            else:
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
            if balance_current_directions:
                n_zero = n_current // 2
                n_one = n_current - n_zero
                indices = torch.cat(
                    [
                        sample_direction(0, n_zero),
                        sample_direction(1, n_one),
                    ]
                )
                indices = indices[
                    torch.randperm(
                        indices.numel(),
                        generator=generator,
                        device=features.device,
                    )
                ]
            elif features.size(0) <= batch_size:
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


@torch.no_grad()
def build_real_conflict_multiview_supervision(
    weak_task_probs: torch.Tensor,
    weak_clip_probs: torch.Tensor,
    strong_task_probs: torch.Tensor,
    strong_clip_probs: torch.Tensor,
    class_a: torch.Tensor,
    class_b: torch.Tensor,
    *,
    train_fraction: float,
    temperature: float,
    weak_pair_features: Optional[torch.Tensor] = None,
    strong_pair_features: Optional[torch.Tensor] = None,
    residual_from_fallback: bool = False,
) -> dict:
    """Build label-free soft A/B targets from weak/strong conflict stability.

    ``A`` is the weak Task top-1 and ``B`` the weak CLIP top-1.  Each branch
    votes with its A-vs-B normalized margin in both views.  A branch receives
    less weight when its weak and strong margins disagree.  Only a fixed top
    fraction by the resulting confidence is used for real-conflict fine-tuning;
    neither construction nor ranking receives target labels.
    """
    probability_tensors = (
        weak_task_probs,
        weak_clip_probs,
        strong_task_probs,
        strong_clip_probs,
    )
    if any(tensor.dim() != 2 for tensor in probability_tensors):
        raise ValueError("all multiview probabilities must have shape [N, C]")
    if any(tensor.shape != weak_task_probs.shape for tensor in probability_tensors):
        raise ValueError("all multiview probabilities must have equal shape")
    total = int(weak_task_probs.size(0))
    if class_a.shape != (total,) or class_b.shape != (total,):
        raise ValueError("class_a and class_b must have shape [N]")
    if not 0.0 < float(train_fraction) <= 1.0:
        raise ValueError("train_fraction must satisfy 0 < value <= 1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if total == 0:
        return {
            "selected": torch.zeros(0, dtype=torch.bool),
            "soft_targets": torch.zeros(0, 2, dtype=torch.float32),
            "weights": torch.zeros(0, dtype=torch.float32),
            "score": torch.zeros(0, dtype=torch.float32),
            "confidence": torch.zeros(0, dtype=torch.float32),
            "task_reliability": torch.zeros(0, dtype=torch.float32),
            "clip_reliability": torch.zeros(0, dtype=torch.float32),
        }

    device = weak_task_probs.device
    class_a = class_a.detach().long().to(device)
    class_b = class_b.detach().long().to(device)
    rows = torch.arange(total, device=device)

    def normalized_pair_margin(probabilities: torch.Tensor) -> torch.Tensor:
        probabilities = probabilities.detach().float().to(device)
        prob_a = probabilities[rows, class_a]
        prob_b = probabilities[rows, class_b]
        return (prob_a - prob_b) / (prob_a + prob_b).clamp_min(1e-8)

    task_weak_margin = normalized_pair_margin(weak_task_probs)
    task_strong_margin = normalized_pair_margin(strong_task_probs)
    clip_weak_margin = normalized_pair_margin(weak_clip_probs)
    clip_strong_margin = normalized_pair_margin(strong_clip_probs)
    task_reliability = 1.0 - 0.5 * (
        task_weak_margin - task_strong_margin
    ).abs()
    clip_reliability = 1.0 - 0.5 * (
        clip_weak_margin - clip_strong_margin
    ).abs()
    if residual_from_fallback:
        if weak_pair_features is None or strong_pair_features is None:
            raise ValueError(
                "residual multiview supervision requires weak/strong pair features"
            )
        if weak_pair_features.shape != strong_pair_features.shape:
            raise ValueError("weak/strong pair features must have equal shape")
        if weak_pair_features.shape != (total, 16):
            raise ValueError("weak/strong pair features must have shape [N, 16]")

        # Candidate A is defined by the weak DUET fallback, so using the weak
        # probability margin as supervision would merely teach identity. The
        # residual target instead uses the independently augmented strong view
        # and neighborhood support from both weak and strong representations.
        strong_probability_evidence = 0.5 * (
            task_strong_margin + clip_strong_margin
        )

        def anchor_evidence(features: torch.Tensor) -> torch.Tensor:
            features = features.detach().float().to(device)
            differences = torch.stack(
                [
                    0.5 * (features[:, 8] - features[:, 9]),
                    0.5 * (features[:, 10] - features[:, 11]),
                ],
                dim=1,
            )
            available = torch.stack(
                [
                    features[:, 12] * features[:, 13],
                    features[:, 14] * features[:, 15],
                ],
                dim=1,
            )
            return (differences * available).sum(dim=1) / available.sum(
                dim=1
            ).clamp_min(1.0)

        neighborhood_evidence = 0.5 * (
            anchor_evidence(weak_pair_features)
            + anchor_evidence(strong_pair_features)
        )
        score = strong_probability_evidence + neighborhood_evidence
        # Reliability now measures agreement between the independent strong
        # probability vote and the cross-view neighborhood vote.
        agreement = (
            strong_probability_evidence * neighborhood_evidence >= 0.0
        ).float()
        task_reliability = 0.5 + 0.5 * agreement
        clip_reliability = task_reliability.clone()
    else:
        task_evidence = 0.5 * (task_weak_margin + task_strong_margin)
        clip_evidence = 0.5 * (clip_weak_margin + clip_strong_margin)
        # Positive score means choose A/Task; negative means choose B/CLIP.
        score = task_reliability * task_evidence + clip_reliability * clip_evidence
    confidence = (0.5 * score.abs()).clamp(0.0, 1.0)
    probability_a = torch.sigmoid(score / float(temperature))
    soft_targets = torch.stack([probability_a, 1.0 - probability_a], dim=1)

    selected_count = max(1, min(total, int(round(total * train_fraction))))
    order = torch.argsort(confidence, descending=True, stable=True)
    selected = torch.zeros(total, dtype=torch.bool, device=device)
    selected[order[:selected_count]] = True
    # Confidence weighting plus inverse-frequency direction balancing prevents
    # a mostly-one-sided pseudo pool from collapsing the comparator.
    selected_confidence = confidence[selected]
    selected_direction = score[selected] < 0.0  # False=A/Task, True=B/CLIP
    direction_weights = torch.ones_like(selected_confidence)
    direction_counts = torch.stack(
        [(selected_direction == direction).sum() for direction in (False, True)]
    ).float()
    if bool((direction_counts > 0).all().item()):
        for direction in (False, True):
            direction_weights[selected_direction == direction] = (
                float(selected_count) / (2.0 * direction_counts[int(direction)])
            )
    weights = selected_confidence * direction_weights
    weights = weights / weights.mean().clamp_min(1e-6)
    return {
        "selected": selected,
        "soft_targets": soft_targets[selected],
        "weights": weights,
        "score": score,
        "confidence": confidence,
        "task_reliability": task_reliability,
        "clip_reliability": clip_reliability,
    }


def train_pairwise_comparator_real_multiview(
    comparator: PairwiseConflictComparator,
    optimizer: torch.optim.Optimizer,
    real_features: torch.Tensor,
    real_soft_targets: torch.Tensor,
    real_weights: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    synthetic_features: Optional[torch.Tensor] = None,
    synthetic_targets: Optional[torch.Tensor] = None,
    synthetic_mix_fraction: float = 0.25,
) -> Optional[float]:
    """Fine-tune on label-free real conflicts, retaining synthetic replay."""
    if real_features.numel() == 0 or int(steps) <= 0:
        return None
    if real_soft_targets.shape != (real_features.size(0), 2):
        raise ValueError("real_soft_targets must have shape [N, 2]")
    if real_weights.shape != (real_features.size(0),):
        raise ValueError("real_weights must have shape [N]")
    if not 0.0 <= float(synthetic_mix_fraction) < 1.0:
        raise ValueError("synthetic_mix_fraction must satisfy 0 <= value < 1")

    # ``real_features`` is normally built on the comparator GPU, while the
    # weak/strong probabilities (and therefore soft targets/weights) are
    # collected on CPU. Align every training tensor before creating indices.
    device = next(comparator.parameters()).device
    real_features = real_features.detach().to(device)
    real_soft_targets = real_soft_targets.detach().to(device)
    real_weights = real_weights.detach().to(device)
    if synthetic_features is not None:
        synthetic_features = synthetic_features.detach().to(device)
    if synthetic_targets is not None:
        synthetic_targets = synthetic_targets.detach().to(device)

    use_synthetic = (
        synthetic_features is not None
        and synthetic_targets is not None
        and synthetic_features.size(0) > 0
        and synthetic_mix_fraction > 0.0
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total_loss = 0.0
    comparator.train()
    for _ in range(int(steps)):
        if real_features.size(0) <= batch_size:
            real_indices = torch.arange(
                real_features.size(0), device=real_features.device
            )
        else:
            real_indices = torch.randperm(
                real_features.size(0),
                generator=generator,
                device=real_features.device,
            )[:batch_size]
        logits = comparator(real_features[real_indices].detach())
        targets = real_soft_targets[real_indices].detach().to(logits.device)
        weights = real_weights[real_indices].detach().to(logits.device)
        real_row_loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
        real_loss = (real_row_loss * weights).sum() / weights.sum().clamp_min(1e-8)
        loss = real_loss
        if use_synthetic:
            synthetic_batch = max(
                1,
                int(round(real_indices.numel() * synthetic_mix_fraction)),
            )
            synthetic_indices = torch.randint(
                0,
                synthetic_features.size(0),
                (synthetic_batch,),
                generator=generator,
                device=real_features.device,
            )
            synthetic_logits = comparator(
                synthetic_features[synthetic_indices].detach()
            )
            synthetic_loss = F.cross_entropy(
                synthetic_logits,
                synthetic_targets[synthetic_indices].detach().long().to(
                    synthetic_logits.device
                ),
            )
            loss = (
                (1.0 - synthetic_mix_fraction) * real_loss
                + synthetic_mix_fraction * synthetic_loss
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item())
    comparator.eval()
    return total_loss / int(steps)


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
    conflict_belief_memory: Optional[PersistentConflictBeliefMemory] = None,
    historical_conflict_snapshot: Optional[dict] = None,
    reliability_task_view_probs: Optional[torch.Tensor] = None,
    reliability_clip_view_probs: Optional[torch.Tensor] = None,
    reliability_task_view_features: Optional[torch.Tensor] = None,
    reliability_clip_view_features: Optional[torch.Tensor] = None,
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
            conflict_belief_memory=conflict_belief_memory,
            historical_conflict_snapshot=historical_conflict_snapshot,
            reliability_task_view_probs=reliability_task_view_probs,
            reliability_clip_view_probs=reliability_clip_view_probs,
            reliability_task_view_features=reliability_task_view_features,
            reliability_clip_view_features=reliability_clip_view_features,
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
            duet_fallback_top1=(task_probs + clip_probs).argmax(dim=1),
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
    conflict_belief_memory: Optional[PersistentConflictBeliefMemory] = None,
    historical_conflict_snapshot: Optional[dict] = None,
    reliability_task_view_probs: Optional[torch.Tensor] = None,
    reliability_clip_view_probs: Optional[torch.Tensor] = None,
    reliability_task_view_features: Optional[torch.Tensor] = None,
    reliability_clip_view_features: Optional[torch.Tensor] = None,
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
    real_conflict_gt_probe_enabled = bool(
        getattr(context_cfg, "REAL_CONFLICT_GT_PROBE_ENABLED", False)
    )
    real_conflict_gt_probe_folds = int(
        getattr(context_cfg, "REAL_CONFLICT_GT_PROBE_FOLDS", 5)
    )
    real_conflict_gt_probe_steps = int(
        getattr(context_cfg, "REAL_CONFLICT_GT_PROBE_STEPS", 300)
    )
    real_conflict_gt_probe_hidden = int(
        getattr(context_cfg, "REAL_CONFLICT_GT_PROBE_HIDDEN", 64)
    )
    real_conflict_gt_probe_lr = float(
        getattr(context_cfg, "REAL_CONFLICT_GT_PROBE_LR", 0.01)
    )
    real_conflict_gt_probe_extended_20d_enabled = bool(
        getattr(
            context_cfg,
            "REAL_CONFLICT_GT_PROBE_EXTENDED_20D_ENABLED",
            False,
        )
    )
    real_multiview_enabled = bool(
        getattr(context_cfg, "REAL_MULTIVIEW_ENABLED", False)
    )
    real_multiview_residual_fallback = bool(
        getattr(
            context_cfg,
            "REAL_MULTIVIEW_RESIDUAL_FALLBACK",
            False,
        )
    )
    real_multiview_train_fraction = float(
        getattr(context_cfg, "REAL_MULTIVIEW_TRAIN_FRACTION", 0.60)
    )
    real_multiview_finetune_steps = int(
        getattr(context_cfg, "REAL_MULTIVIEW_FINETUNE_STEPS", 100)
    )
    real_multiview_temperature = float(
        getattr(context_cfg, "REAL_MULTIVIEW_TEMPERATURE", 0.50)
    )
    real_multiview_synthetic_mix_fraction = float(
        getattr(
            context_cfg,
            "REAL_MULTIVIEW_SYNTHETIC_MIX_FRACTION",
            0.25,
        )
    )
    conflict_memory_enabled = bool(
        getattr(context_cfg, "CONFLICT_MEMORY_ENABLED", False)
    )
    conflict_memory_coverage = float(
        getattr(context_cfg, "CONFLICT_MEMORY_COVERAGE_FRACTION", 0.80)
    )
    conflict_memory_temperature = float(
        getattr(context_cfg, "CONFLICT_MEMORY_TEMPERATURE", 0.50)
    )
    reliability_gate_enabled = bool(
        getattr(context_cfg, "RELIABILITY_GATE_ENABLED", False)
    )
    reliability_gate_coverage = float(
        getattr(context_cfg, "RELIABILITY_GATE_COVERAGE_FRACTION", 0.80)
    )
    reliability_gate_temperature = float(
        getattr(context_cfg, "RELIABILITY_GATE_TEMPERATURE", 0.25)
    )
    reliability_gate_neighbors = int(
        getattr(context_cfg, "RELIABILITY_GATE_NEIGHBORS", 5)
    )
    reliability_gate_num_views = int(
        getattr(context_cfg, "RELIABILITY_GATE_NUM_VIEWS", 1)
    )
    transition_supervision_enabled = bool(
        getattr(context_cfg, "TRANSITION_SUPERVISION_ENABLED", False)
    )
    transition_min_view_agreement = float(
        getattr(context_cfg, "TRANSITION_MIN_VIEW_AGREEMENT", 0.75)
    )
    transition_min_per_direction = int(
        getattr(context_cfg, "TRANSITION_MIN_PER_DIRECTION", 16)
    )
    transition_train_steps = int(
        getattr(context_cfg, "TRANSITION_TRAIN_STEPS", 400)
    )
    transition_synthetic_mix_fraction = float(
        getattr(
            context_cfg,
            "TRANSITION_SYNTHETIC_MIX_FRACTION",
            0.25,
        )
    )
    transition_comparator_weight = float(
        getattr(context_cfg, "TRANSITION_COMPARATOR_WEIGHT", 0.50)
    )
    if reliability_gate_enabled and conflict_memory_enabled:
        raise ValueError(
            "RELIABILITY_GATE_ENABLED and CONFLICT_MEMORY_ENABLED are exclusive"
        )
    if reliability_gate_enabled and not 0.0 < reliability_gate_coverage <= 1.0:
        raise ValueError(
            "RELIABILITY_GATE_COVERAGE_FRACTION must satisfy 0 < value <= 1"
        )
    if reliability_gate_enabled and reliability_gate_temperature <= 0.0:
        raise ValueError("RELIABILITY_GATE_TEMPERATURE must be positive")
    if reliability_gate_enabled and reliability_gate_neighbors < 1:
        raise ValueError("RELIABILITY_GATE_NEIGHBORS must be >= 1")
    if reliability_gate_enabled and reliability_gate_num_views < 1:
        raise ValueError("RELIABILITY_GATE_NUM_VIEWS must be >= 1")
    if transition_supervision_enabled and not reliability_gate_enabled:
        raise ValueError(
            "TRANSITION_SUPERVISION_ENABLED requires RELIABILITY_GATE_ENABLED"
        )
    if transition_supervision_enabled and historical_conflict_snapshot is None:
        raise ValueError(
            "TRANSITION_SUPERVISION_ENABLED requires a historical snapshot"
        )
    if transition_supervision_enabled and int(cycle) != 2:
        raise ValueError(
            "transition supervision currently supports Cycle 2 only"
        )
    if transition_supervision_enabled and sample_indices is None:
        raise ValueError(
            "TRANSITION_SUPERVISION_ENABLED requires stable sample indices"
        )
    if transition_supervision_enabled and not (
        0.5 <= transition_min_view_agreement <= 1.0
    ):
        raise ValueError("TRANSITION_MIN_VIEW_AGREEMENT must be in [0.5, 1]")
    if transition_supervision_enabled and transition_min_per_direction < 1:
        raise ValueError("TRANSITION_MIN_PER_DIRECTION must be >= 1")
    if transition_supervision_enabled and transition_train_steps < 1:
        raise ValueError("TRANSITION_TRAIN_STEPS must be >= 1")
    if transition_supervision_enabled and not (
        0.0 <= transition_synthetic_mix_fraction < 1.0
    ):
        raise ValueError(
            "TRANSITION_SYNTHETIC_MIX_FRACTION must satisfy 0 <= value < 1"
        )
    if transition_supervision_enabled and not (
        0.0 <= transition_comparator_weight <= 1.0
    ):
        raise ValueError("TRANSITION_COMPARATOR_WEIGHT must be in [0, 1]")
    if conflict_memory_enabled and not 0.0 < conflict_memory_coverage <= 1.0:
        raise ValueError(
            "CONFLICT_MEMORY_COVERAGE_FRACTION must satisfy 0 < value <= 1"
        )
    if conflict_memory_enabled and conflict_memory_temperature <= 0.0:
        raise ValueError("CONFLICT_MEMORY_TEMPERATURE must be positive")
    if conflict_memory_enabled and conflict_belief_memory is None:
        raise ValueError(
            "CONFLICT_MEMORY_ENABLED requires PersistentConflictBeliefMemory"
        )
    if conflict_memory_enabled and real_multiview_residual_fallback:
        raise ValueError(
            "CONFLICT_MEMORY_ENABLED requires Task-vs-CLIP candidate semantics"
        )
    if real_multiview_enabled and not 0.0 < real_multiview_train_fraction <= 1.0:
        raise ValueError(
            "REAL_MULTIVIEW_TRAIN_FRACTION must satisfy 0 < value <= 1"
        )
    if (
        real_multiview_enabled
        and not real_multiview_residual_fallback
        and real_multiview_finetune_steps < 1
    ):
        raise ValueError("REAL_MULTIVIEW_FINETUNE_STEPS must be >= 1")
    if real_multiview_enabled and real_multiview_finetune_steps < 0:
        raise ValueError("REAL_MULTIVIEW_FINETUNE_STEPS must be >= 0")
    if real_multiview_enabled and real_multiview_temperature <= 0.0:
        raise ValueError("REAL_MULTIVIEW_TEMPERATURE must be positive")
    if real_multiview_enabled and not (
        0.0 <= real_multiview_synthetic_mix_fraction < 1.0
    ):
        raise ValueError(
            "REAL_MULTIVIEW_SYNTHETIC_MIX_FRACTION must satisfy 0 <= value < 1"
        )
    if real_conflict_gt_probe_enabled and real_conflict_gt_probe_folds < 2:
        raise ValueError("REAL_CONFLICT_GT_PROBE_FOLDS must be >= 2")
    if real_conflict_gt_probe_enabled and real_conflict_gt_probe_steps < 1:
        raise ValueError("REAL_CONFLICT_GT_PROBE_STEPS must be >= 1")
    if real_conflict_gt_probe_enabled and real_conflict_gt_probe_hidden < 1:
        raise ValueError("REAL_CONFLICT_GT_PROBE_HIDDEN must be >= 1")
    if real_conflict_gt_probe_enabled and real_conflict_gt_probe_lr <= 0.0:
        raise ValueError("REAL_CONFLICT_GT_PROBE_LR must be positive")
    agreement_ambiguity_eval_enabled = bool(
        getattr(context_cfg, "AGREEMENT_AMBIGUITY_EVAL_ENABLED", False)
    )
    agreement_candidate_probe_enabled = bool(
        getattr(context_cfg, "AGREEMENT_COMPARATOR_PROBE_ENABLED", False)
    )
    agreement_synthetic_feasibility_enabled = bool(
        getattr(
            context_cfg,
            "AGREEMENT_SYNTHETIC_FEASIBILITY_ENABLED",
            False,
        )
    )
    agreement_ambiguity_fractions = [
        int(value)
        for value in getattr(
            context_cfg,
            "AGREEMENT_AMBIGUITY_FRACTIONS",
            [10, 25, 50, 100],
        )
    ]
    if (
        agreement_ambiguity_eval_enabled
        or agreement_candidate_probe_enabled
        or agreement_synthetic_feasibility_enabled
    ) and (
        not agreement_ambiguity_fractions
        or any(
            value <= 0 or value > 100
            for value in agreement_ambiguity_fractions
        )
        or 25 not in agreement_ambiguity_fractions
    ):
        raise ValueError(
            "AGREEMENT_AMBIGUITY_FRACTIONS must contain 25 and use values in [1, 100]"
        )
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
    aligned_historical_snapshot = None
    if transition_supervision_enabled:
        aligned_historical_snapshot = _align_historical_conflict_snapshot(
            historical_conflict_snapshot,
            sample_indices,
        )

    # Evaluation-only branch.  It runs before strict-conflict construction,
    # anchor banks, comparator updates and admission, and its output is ignored.
    if (
        agreement_ambiguity_eval_enabled
        and eval_only_logging
        and labels is not None
    ):
        _log_agreement_ambiguity_eval_only(
            task_probs,
            clip_probs,
            labels,
            fractions=agreement_ambiguity_fractions,
            cycle=cycle,
            log_fn=log_fn,
        )

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
    conflict_memory_payload = {
        "candidate_a": torch.full((num_samples,), -1, dtype=torch.long),
        "candidate_b": torch.full((num_samples,), -1, dtype=torch.long),
        "q": torch.full((num_samples,), 0.5, dtype=torch.float32),
        "weight": torch.zeros(num_samples, dtype=torch.float32),
        "active": torch.zeros(num_samples, dtype=torch.bool),
        "observations": torch.zeros(num_samples, dtype=torch.long),
    }
    reliability_gate_payload = {
        "active": torch.zeros(num_samples, dtype=torch.bool),
        "switch": torch.zeros(num_samples, dtype=torch.bool),
        "target": clip_probs.detach().clone(),
        "candidate_a": torch.full((num_samples,), -1, dtype=torch.long),
        "candidate_b": torch.full((num_samples,), -1, dtype=torch.long),
        "q": torch.full((num_samples,), 0.5, dtype=torch.float32),
        "baseline_q": torch.full((num_samples,), 0.5, dtype=torch.float32),
        "baseline_pair_mass": torch.zeros(num_samples, dtype=torch.float32),
        "weight": torch.zeros(num_samples, dtype=torch.float32),
        "reliability": torch.zeros(num_samples, dtype=torch.float32),
        "source_agreement": torch.zeros(num_samples, dtype=torch.float32),
        # Delayed transition supervision is used to change the original
        # DUET CLIP teacher, so train only that change in the Task branch.
        "residual_pairwise": bool(transition_supervision_enabled),
    }

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

        # Eval-only feasibility test for a candidate-semantic comparator.
        # Weak consensus supplies pseudo Y; strong shared Top1/Top2 supplies
        # A/B.  Its output is deliberately ignored by all training decisions.
        if (
            agreement_synthetic_feasibility_enabled
            and eval_only_logging
            and pool_strong_task is not None
            and pool_strong_clip is not None
            and pool_strong_task_features is not None
            and pool_strong_clip_features is not None
        ):
            pool_gt_labels = None
            if labels is not None:
                pool_gt_labels = labels.detach().long().cpu()[anchor_mask.cpu()]
            _log_agreement_synthetic_feasibility_eval_only(
                pool_labels,
                pool_strong_task,
                pool_strong_clip,
                pool_strong_task_features,
                pool_strong_clip_features,
                task_bank=task_bank,
                clip_bank=clip_bank,
                sim_topk=sim_topk,
                fractions=agreement_ambiguity_fractions,
                cycle=cycle,
                log_fn=log_fn,
                pool_gt_labels=pool_gt_labels,
            )

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
        strong_real_features = torch.zeros(
            0, 16, dtype=torch.float32
        )
        residual_router_logits = None
        transition_result = None
        transition_comparator_trained = False
        real_candidate_a = task_top1[strict_positions]
        real_candidate_b = clip_top1[strict_positions]
        if strict_positions.numel() > 0:
            if real_multiview_residual_fallback:
                duet_mix = 0.5 * (
                    task_probs[strict_positions]
                    + clip_probs[strict_positions]
                )
                real_candidate_a = duet_mix.argmax(dim=1)
                strict_task_candidate = task_top1[strict_positions]
                strict_clip_candidate = clip_top1[strict_positions]
                rows = torch.arange(strict_positions.numel())
                task_mix_score = duet_mix[rows, strict_task_candidate]
                clip_mix_score = duet_mix[rows, strict_clip_candidate]
                stronger_branch_candidate = torch.where(
                    task_mix_score >= clip_mix_score,
                    strict_task_candidate,
                    strict_clip_candidate,
                )
                real_candidate_b = torch.where(
                    real_candidate_a == strict_task_candidate,
                    strict_clip_candidate,
                    torch.where(
                        real_candidate_a == strict_clip_candidate,
                        strict_task_candidate,
                        stronger_branch_candidate,
                    ),
                )
            real_features = build_comparator_features(
                task_probs[strict_positions],
                clip_probs[strict_positions],
                task_features[strict_positions],
                clip_features[strict_positions],
                task_bank,
                clip_bank,
                class_a=real_candidate_a,
                class_b=real_candidate_b,
                sim_topk=sim_topk,
            ).to(device)
            _log_pair_distribution(
                real_features.cpu(), "real-conflict", cycle, log_fn
            )
            if real_multiview_residual_fallback:
                if (
                    strong_task_probs is None
                    or strong_clip_probs is None
                    or strong_task_features is None
                    or strong_clip_features is None
                ):
                    raise ValueError(
                        "residual fallback mode requires strong probabilities and features"
                    )
                strong_real_features = build_comparator_features(
                    strong_task_probs[strict_positions],
                    strong_clip_probs[strict_positions],
                    strong_task_features[strict_positions],
                    strong_clip_features[strict_positions],
                    task_bank,
                    clip_bank,
                    class_a=real_candidate_a,
                    class_b=real_candidate_b,
                    sim_topk=sim_topk,
                ).to(device)
                log_fn(
                    "DUET residual candidate construction: cycle={}; "
                    "candidate_a=duet_fallback; candidate_b=task_clip_challenger; "
                    "strong_view=True; neighborhood_views=weak,strong; "
                    "construction_uses_gt=False; ground_truth_affects_training=False".format(
                        cycle
                    )
                )
            if transition_supervision_enabled:
                if comparator is None or comparator_optimizer is None:
                    raise ValueError(
                        "transition supervision requires comparator and optimizer"
                    )
                task_transition_views = (
                    reliability_task_view_probs
                    if reliability_task_view_probs is not None
                    else strong_task_probs.unsqueeze(0)
                )
                clip_transition_views = (
                    reliability_clip_view_probs
                    if reliability_clip_view_probs is not None
                    else strong_clip_probs.unsqueeze(0)
                )
                transition_result = build_delayed_transition_supervision(
                    aligned_historical_snapshot,
                    task_probs,
                    clip_probs,
                    task_transition_views,
                    clip_transition_views,
                    num_classes=num_classes,
                    anchors_per_class=anchors_per_class,
                    anchor_task_conf=anchor_task_conf,
                    anchor_clip_conf=anchor_clip_conf,
                    anchor_task_entropy=anchor_task_entropy,
                    anchor_clip_entropy=anchor_clip_entropy,
                    entropy_weight=entropy_weight,
                    require_pre_post_prior_agreement=(
                        require_pre_post_prior_agreement
                    ),
                    sim_topk=sim_topk,
                    min_view_agreement=transition_min_view_agreement,
                    min_per_direction=transition_min_per_direction,
                    seed=seed,
                )
                log_fn(
                    "DUET delayed real-conflict supervision: cycle={}; "
                    "historical_conflicts={}; matured={}; matured_rate={:.2f}%; "
                    "raw_choose_a={}; raw_choose_b={}; "
                    "minority_direction_samples={}; train_samples={}; "
                    "historical_anchor_count={}; min_view_agreement={:.2f}; "
                    "construction=pre_cycle1_conflict_to_post_cycle1_stable_agreement; "
                    "uses_current_conflict_gt=False; ground_truth_affects_training=False".format(
                        cycle,
                        transition_result["historical_conflicts"],
                        transition_result["matured"],
                        100.0
                        * transition_result["matured"]
                        / max(transition_result["historical_conflicts"], 1),
                        transition_result["raw_choose_a"],
                        transition_result["raw_choose_b"],
                        transition_result["balanced_per_direction"],
                        transition_result["features"].size(0),
                        transition_result["anchor_count"],
                        transition_min_view_agreement,
                    )
                )
                if eval_only_logging and labels is not None and transition_result["matured"] > 0:
                    matured_mask = transition_result["selected_mask"]
                    matured_label = transition_result["matured_label"][matured_mask]
                    matured_gt = labels.detach().long().cpu()[matured_mask]
                    pseudo_precision = float(
                        (matured_label == matured_gt).float().mean().item()
                    )
                    log_fn(
                        "DUET delayed real-conflict supervision eval-only: "
                        "cycle={}; matured_n={}; pseudo_label_precision={:.2f}%; "
                        "selection_uses_gt=False; ground_truth_affects_training=False".format(
                            cycle,
                            transition_result["matured"],
                            100.0 * pseudo_precision,
                        )
                    )
                if transition_result["ready"]:
                    transition_features = transition_result["features"].to(device)
                    transition_targets = transition_result["targets"].to(device)
                    use_synthetic_regularizer = (
                        transition_synthetic_mix_fraction > 0.0
                        and synthetic_features.size(0) >= 2
                    )
                    transition_loss = train_pairwise_comparator(
                        comparator,
                        comparator_optimizer,
                        transition_features,
                        transition_targets,
                        steps=transition_train_steps,
                        batch_size=train_batch_size,
                        seed=seed + 104729,
                        memory_features=(
                            synthetic_features.to(device)
                            if use_synthetic_regularizer
                            else None
                        ),
                        memory_targets=(
                            synthetic_targets.to(device)
                            if use_synthetic_regularizer
                            else None
                        ),
                        memory_fraction=transition_synthetic_mix_fraction,
                        balance_current_directions=True,
                    )
                    transition_comparator_trained = True
                    stats["train_loss"] = transition_loss
                    stats["train_current_samples"] = int(
                        transition_features.size(0)
                    )
                    stats["train_memory_samples"] = (
                        int(synthetic_features.size(0))
                        if use_synthetic_regularizer
                        else 0
                    )
                    stats["optimizer_steps_this_cycle"] = transition_train_steps
                    log_fn(
                        "DUET transition comparator training: cycle={}; "
                        "optimizer_steps={}; real_transition_samples={}; "
                        "synthetic_regularizer_samples={}; synthetic_mix_fraction={:.2f}; "
                        "train_loss={}; supervision_distribution=historical_real_conflicts; "
                        "sampling=direction_balanced_without_downsampling; "
                        "ground_truth_affects_training=False".format(
                            cycle,
                            transition_train_steps,
                            transition_features.size(0),
                            synthetic_features.size(0)
                            if use_synthetic_regularizer
                            else 0,
                            transition_synthetic_mix_fraction
                            if use_synthetic_regularizer
                            else 0.0,
                            "none"
                            if transition_loss is None
                            else "{:.6f}".format(transition_loss),
                        )
                    )
                else:
                    log_fn(
                        "DUET transition comparator training skipped: cycle={}; "
                        "reason=insufficient_balanced_matured_conflicts; "
                        "required_per_direction={}; committee_fallback=True; "
                        "ground_truth_affects_training=False".format(
                            cycle,
                            transition_min_per_direction,
                        )
                    )
            if reliability_gate_enabled:
                if (
                    strong_task_probs is None
                    or strong_clip_probs is None
                    or strong_task_features is None
                    or strong_clip_features is None
                ):
                    raise ValueError(
                        "RELIABILITY_GATE_ENABLED requires strong probabilities and features"
                    )
                gate_local = build_reliability_gated_fusion(
                    task_probs[strict_positions],
                    clip_probs[strict_positions],
                    (
                        reliability_task_view_probs[:, strict_positions]
                        if reliability_task_view_probs is not None
                        else strong_task_probs[strict_positions]
                    ),
                    (
                        reliability_clip_view_probs[:, strict_positions]
                        if reliability_clip_view_probs is not None
                        else strong_clip_probs[strict_positions]
                    ),
                    task_features[strict_positions],
                    clip_features[strict_positions],
                    (
                        reliability_task_view_features[:, strict_positions]
                        if reliability_task_view_features is not None
                        else strong_task_features[strict_positions]
                    ),
                    (
                        reliability_clip_view_features[:, strict_positions]
                        if reliability_clip_view_features is not None
                        else strong_clip_features[strict_positions]
                    ),
                    task_bank,
                    clip_bank,
                    num_classes=num_classes,
                    neighbors=reliability_gate_neighbors,
                    temperature=reliability_gate_temperature,
                    coverage_fraction=reliability_gate_coverage,
                )
                gate_method = "candidate_evidence_committee"
                if transition_comparator_trained:
                    comparator.eval()
                    with torch.no_grad():
                        transition_prob_a = F.softmax(
                            comparator(real_features.detach()), dim=1
                        )[:, 0].cpu()
                    gate_local = fuse_transition_comparator_vote(
                        gate_local,
                        transition_prob_a,
                        task_probs[strict_positions],
                        clip_probs[strict_positions],
                        comparator_weight=transition_comparator_weight,
                        coverage_fraction=reliability_gate_coverage,
                    )
                    gate_method = (
                        "candidate_committee_plus_delayed_real_conflict_comparator"
                    )
                    transition_agreement = float(
                        gate_local["transition_committee_agreement"].mean().item()
                    )
                    log_fn(
                        "DUET transition-comparator fusion: cycle={}; "
                        "current_conflicts={}; comparator_weight={:.2f}; "
                        "committee_weight={:.2f}; direction_agreement={:.2f}%; "
                        "coverage={:.2f}%; labels_used_for_fusion=False; "
                        "ground_truth_affects_training=False".format(
                            cycle,
                            int(strict_positions.numel()),
                            transition_comparator_weight,
                            1.0 - transition_comparator_weight,
                            100.0 * transition_agreement,
                            100.0 * reliability_gate_coverage,
                        )
                    )
                for key in (
                    "active", "target", "candidate_a", "candidate_b", "q",
                    "baseline_q", "baseline_pair_mass", "weight",
                    "reliability", "source_agreement",
                ):
                    reliability_gate_payload[key][strict_positions] = gate_local[key]
                gate_active = gate_local["active"]
                gate_count = int(gate_active.sum().item())
                baseline_local = 0.5 * (
                    task_probs[strict_positions] + clip_probs[strict_positions]
                )
                gated_local = gate_local["target"]
                baseline_label = baseline_local.argmax(dim=1)
                gated_label = gated_local.argmax(dim=1)
                switches = gate_active & (gated_label != baseline_label)
                reliability_gate_payload["switch"][strict_positions] = switches
                log_fn(
                    "DUET reliability-gated comparator: cycle={}; conflicts={}; "
                    "views={}; active={}; coverage={:.2f}%; candidate_q_mean={:.4f}; "
                    "candidate_q_p10={:.4f}; candidate_q_p90={:.4f}; "
                    "reliability_mean={:.4f}; source_agreement_mean={:.4f}; "
                    "effective_sample_equivalent={:.2f}; "
                    "switches_from_duet_fallback={}; "
                    "method={}; "
                    "candidate_space=task_top1_vs_clip_top1; "
                    "hard_admission_changed=False; original_clip_kl_changed=False; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        int(strict_positions.numel()),
                        reliability_gate_num_views,
                        gate_count,
                        100.0 * gate_count / int(strict_positions.numel()),
                        float(gate_local["q"][gate_active].mean().item()),
                        float(torch.quantile(gate_local["q"][gate_active], 0.10).item()),
                        float(torch.quantile(gate_local["q"][gate_active], 0.90).item()),
                        float(
                            gate_local["reliability"][gate_active].mean().item()
                        ),
                        float(
                            gate_local["source_agreement"][gate_active].mean().item()
                        ),
                        float(gate_local["weight"][gate_active].sum().item()),
                        int(switches.sum().item()),
                        gate_method,
                    )
                )
                if eval_only_logging and labels is not None and gate_count > 0:
                    active_labels = labels[strict_positions][gate_active].long()
                    baseline_acc = float(
                        (baseline_label[gate_active] == active_labels)
                        .float().mean().item()
                    )
                    gated_acc = float(
                        (gated_label[gate_active] == active_labels)
                        .float().mean().item()
                    )
                    switch_count = int(switches.sum().item())
                    switch_precision = (
                        float(
                            (gated_label[switches] == labels[strict_positions][switches])
                            .float().mean().item()
                        )
                        if switch_count > 0
                        else float("nan")
                    )
                    beneficial_switches = int(
                        (gated_label[switches] == labels[strict_positions][switches])
                        .sum().item()
                    )
                    harmful_switches = int(
                        (baseline_label[switches] == labels[strict_positions][switches])
                        .sum().item()
                    )
                    net_corrected = beneficial_switches - harmful_switches
                    log_fn(
                        "DUET reliability-gated comparator eval-only: cycle={}; "
                        "same_subset_n={}; duet_fallback_acc={:.2f}%; "
                        "gated_comparator_acc={:.2f}%; gain={:+.2f}pp; "
                        "switches={}; switch_precision={:.2f}%; "
                        "beneficial_switches={}; harmful_switches={}; "
                        "net_corrected={:+d}; full_target_equivalent={:+.2f}pp; "
                        "ground_truth_affects_training=False".format(
                            cycle,
                            gate_count,
                            100.0 * baseline_acc,
                            100.0 * gated_acc,
                            100.0 * (gated_acc - baseline_acc),
                            switch_count,
                            100.0 * switch_precision,
                            beneficial_switches,
                            harmful_switches,
                            net_corrected,
                            100.0 * net_corrected / num_samples,
                        )
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
            not transition_supervision_enabled
            and synthetic_features.size(0) >= 2
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
                        duet_fallback_candidates=(
                            task_probs[strict_positions]
                            + clip_probs[strict_positions]
                        ).argmax(dim=1),
                        labels=labels[strict_positions],
                        coverages=trajectory_coverages,
                        cycle=cycle,
                        log_fn=log_fn,
                    )
            if replay_memory is not None:
                # 训练后把当前 cycle 的 matched synthetic 写入历史 memory
                replay_memory.update(synthetic_features, synthetic_targets)

        # Formal GT-free adaptation on the real strict-conflict distribution.
        # Pseudo targets come only from weak/strong A-vs-B stability. Synthetic
        # conflicts remain in the loss as an anti-collapse regularizer.
        if real_multiview_enabled and strict_positions.numel() > 0:
            if comparator_optimizer is None:
                raise ValueError(
                    "REAL_MULTIVIEW_ENABLED requires a comparator optimizer"
                )
            if strong_task_probs is None or strong_clip_probs is None:
                raise ValueError(
                    "REAL_MULTIVIEW_ENABLED requires strong Task/CLIP probabilities"
                )
            multiview = build_real_conflict_multiview_supervision(
                task_probs[strict_positions],
                clip_probs[strict_positions],
                strong_task_probs[strict_positions],
                strong_clip_probs[strict_positions],
                real_candidate_a,
                real_candidate_b,
                train_fraction=real_multiview_train_fraction,
                temperature=real_multiview_temperature,
                weak_pair_features=real_features,
                strong_pair_features=strong_real_features,
                residual_from_fallback=real_multiview_residual_fallback,
            )
            multiview_selected = multiview["selected"]
            selected_count = int(multiview_selected.sum().item())
            if real_multiview_residual_fallback:
                # The evidence already exists on every real conflict. Using it
                # directly avoids fitting a weak-view MLP to strong-view pseudo
                # targets (the previous run stayed at ln(2) and made zero
                # switches). This path has no optimizer or RNG side effects.
                router_score = multiview["score"].detach().float().cpu()
                residual_router_logits = torch.stack(
                    [router_score, -router_score], dim=1
                ) / real_multiview_temperature
                real_multiview_loss = None
                stats["train_loss"] = None
                stats["train_current_samples"] = selected_count
                stats["optimizer_steps_this_cycle"] = 0
            else:
                real_multiview_loss = train_pairwise_comparator_real_multiview(
                    comparator,
                    comparator_optimizer,
                    real_features[multiview_selected.to(real_features.device)],
                    multiview["soft_targets"],
                    multiview["weights"],
                    steps=real_multiview_finetune_steps,
                    batch_size=train_batch_size,
                    seed=seed + 7919,
                    synthetic_features=synthetic_features,
                    synthetic_targets=synthetic_targets,
                    synthetic_mix_fraction=real_multiview_synthetic_mix_fraction,
                )
                stats["train_loss"] = real_multiview_loss
                stats["train_current_samples"] = selected_count
                stats["optimizer_steps_this_cycle"] = real_multiview_finetune_steps
            selected_soft_targets = multiview["soft_targets"]
            pseudo_choose_task = int(
                (selected_soft_targets[:, 0] >= selected_soft_targets[:, 1])
                .sum()
                .item()
            )
            log_fn(
                "DUET comparator real-multiview training: cycle={}; "
                "views=weak_plus_strong; residual_fallback={}; "
                "real_conflicts={}; selected={}; train_coverage={:.2f}%; "
                "pseudo_choose_a_keep={}; pseudo_choose_b_switch={}; "
                "confidence_mean={:.4f}; "
                "task_stability_mean={:.4f}; clip_stability_mean={:.4f}; "
                "temperature={:.3f}; router={}; finetune_steps={}; "
                "synthetic_mix_fraction={:.2f}; train_loss={}; "
                "construction_uses_gt=False; ground_truth_affects_training=False".format(
                    cycle,
                    real_multiview_residual_fallback,
                    int(strict_positions.numel()),
                    selected_count,
                    100.0 * selected_count / int(strict_positions.numel()),
                    pseudo_choose_task,
                    selected_count - pseudo_choose_task,
                    float(multiview["confidence"][multiview_selected].mean().item()),
                    float(
                        multiview["task_reliability"][multiview_selected]
                        .mean()
                        .item()
                    ),
                    float(
                        multiview["clip_reliability"][multiview_selected]
                        .mean()
                        .item()
                    ),
                    real_multiview_temperature,
                    (
                        "direct_strong_neighborhood_evidence"
                        if real_multiview_residual_fallback
                        else "learned_mlp"
                    ),
                    real_multiview_finetune_steps,
                    real_multiview_synthetic_mix_fraction,
                    (
                        "none"
                        if real_multiview_loss is None
                        else "{:.6f}".format(float(real_multiview_loss))
                    ),
                )
            )

        # Zero-intervention probe: reuse the trained strict-conflict comparator
        # on A=shared Top1 / B=shared Top2 pairs.  Results are logging-only.
        if (
            agreement_candidate_probe_enabled
            and eval_only_logging
            and labels is not None
            and comparator is not None
        ):
            _log_agreement_candidate_probe_eval_only(
                task_probs,
                clip_probs,
                task_features,
                clip_features,
                labels,
                comparator=comparator,
                task_bank=task_bank,
                clip_bank=clip_bank,
                sim_topk=sim_topk,
                fractions=agreement_ambiguity_fractions,
                cycle=cycle,
                log_fn=log_fn,
            )

        if strict_positions.numel() > 0:
            if residual_router_logits is not None:
                logits = residual_router_logits
            else:
                comparator.eval()
                with torch.no_grad():
                    logits = comparator(real_features).cpu()
            if conflict_memory_enabled:
                if strong_task_probs is None or strong_clip_probs is None:
                    raise ValueError(
                        "CONFLICT_MEMORY_ENABLED requires strong Task/CLIP probabilities"
                    )
                # Two independent voters produce the current observation:
                # the synthetic-trained comparator and weak/strong A-vs-B
                # stability. Disagreement pulls q toward 0.5 instead of
                # creating a confident pseudo label.
                memory_evidence = build_real_conflict_multiview_supervision(
                    task_probs[strict_positions],
                    clip_probs[strict_positions],
                    strong_task_probs[strict_positions],
                    strong_clip_probs[strict_positions],
                    task_top1[strict_positions],
                    clip_top1[strict_positions],
                    train_fraction=1.0,
                    temperature=conflict_memory_temperature,
                )
                comparator_prob = _softmax_probabilities(logits).cpu()
                comparator_signed = comparator_prob[:, 0] - comparator_prob[:, 1]
                multiview_signed = torch.tanh(
                    memory_evidence["score"].detach().float().cpu()
                    / conflict_memory_temperature
                )
                combined_signed = 0.5 * (
                    comparator_signed + multiview_signed
                )
                current_q = (0.5 + 0.5 * combined_signed).clamp(0.0, 1.0)
                voter_agreement = (
                    comparator_signed * multiview_signed >= 0.0
                ).float()
                view_reliability = 0.5 * (
                    memory_evidence["task_reliability"].float().cpu()
                    + memory_evidence["clip_reliability"].float().cpu()
                )
                view_reliability = view_reliability * (
                    0.5 + 0.5 * voter_agreement
                )
                conflict_sample_indices = (
                    sample_indices[strict_positions]
                    if sample_indices is not None
                    else strict_positions
                )
                memory_local = conflict_belief_memory.update(
                    conflict_sample_indices,
                    task_top1[strict_positions],
                    clip_top1[strict_positions],
                    current_q,
                    view_reliability,
                    cycle=cycle,
                    coverage_fraction=conflict_memory_coverage,
                )
                for key in (
                    "candidate_a",
                    "candidate_b",
                    "q",
                    "weight",
                    "active",
                    "observations",
                ):
                    conflict_memory_payload[key][strict_positions] = memory_local[key]
                active_local = memory_local["active"]
                active_count = int(active_local.sum().item())
                effective_sample_equivalent = float(
                    memory_local["weight"][active_local].sum().item()
                )
                effective_coverage = effective_sample_equivalent / max(
                    int(strict_positions.numel()), 1
                )
                log_fn(
                    "DUET persistent conflict memory: cycle={}; conflicts={}; "
                    "conflict_fraction_of_target={:.2f}%; active={}; "
                    "raw_coverage={:.2f}%; active_fraction_of_target={:.2f}%; "
                    "effective_sample_equivalent={:.2f}; "
                    "effective_coverage={:.2f}%; "
                    "effective_fraction_of_target={:.2f}%; "
                    "q_mean={:.4f}; weight_mean={:.4f}; observations_mean={:.2f}; "
                    "pair_resets={}; candidates=task_top1_vs_clip_top1; "
                    "hard_admission_changed=False; kl_target_changed=False; "
                    "ground_truth_affects_training=False".format(
                        cycle,
                        int(strict_positions.numel()),
                        100.0 * int(strict_positions.numel()) / num_samples,
                        active_count,
                        100.0 * active_count / int(strict_positions.numel()),
                        100.0 * active_count / num_samples,
                        effective_sample_equivalent,
                        100.0 * effective_coverage,
                        100.0 * effective_sample_equivalent / num_samples,
                        float(memory_local["q"][active_local].mean().item()),
                        float(memory_local["weight"][active_local].mean().item()),
                        float(
                            memory_local["observations"][active_local]
                            .float()
                            .mean()
                            .item()
                        ),
                        memory_local["pair_resets"],
                    )
                )
                if eval_only_logging and labels is not None and active_count > 0:
                    chosen = torch.where(
                        memory_local["q"] >= 0.5,
                        task_top1[strict_positions],
                        clip_top1[strict_positions],
                    )
                    active_labels = labels[strict_positions][active_local].long()
                    task_active = task_top1[strict_positions][active_local]
                    clip_active = clip_top1[strict_positions][active_local]
                    fallback_active = (
                        task_probs[strict_positions]
                        + clip_probs[strict_positions]
                    ).argmax(dim=1)[active_local]
                    memory_acc = float(
                        (chosen[active_local] == active_labels).float().mean().item()
                    )
                    task_acc = float(
                        (task_active == active_labels).float().mean().item()
                    )
                    clip_acc = float(
                        (clip_active == active_labels).float().mean().item()
                    )
                    fallback_acc = float(
                        (fallback_active == active_labels).float().mean().item()
                    )
                    oracle_acc = float(
                        (
                            (task_active == active_labels)
                            | (clip_active == active_labels)
                        ).float().mean().item()
                    )
                    log_fn(
                        "DUET persistent conflict memory eval-only: cycle={}; "
                        "same_subset_n={}; task_acc={:.2f}%; clip_acc={:.2f}%; "
                        "duet_fallback_acc={:.2f}%; memory_acc={:.2f}%; "
                        "candidate_oracle_acc={:.2f}%; "
                        "memory_gain_over_duet_fallback={:+.2f}pp; "
                        "ground_truth_affects_training=False".format(
                            cycle,
                            active_count,
                            100.0 * task_acc,
                            100.0 * clip_acc,
                            100.0 * fallback_acc,
                            100.0 * memory_acc,
                            100.0 * oracle_acc,
                            100.0 * (memory_acc - fallback_acc),
                        )
                    )
            if (
                real_conflict_gt_probe_enabled
                and eval_only_logging
                and labels is not None
            ):
                probe_16d = _log_real_conflict_gt_feature_probe_eval_only(
                    real_features,
                    task_top1[strict_positions],
                    clip_top1[strict_positions],
                    labels[strict_positions],
                    logits,
                    folds=real_conflict_gt_probe_folds,
                    steps=real_conflict_gt_probe_steps,
                    hidden=real_conflict_gt_probe_hidden,
                    lr=real_conflict_gt_probe_lr,
                    seed=seed,
                    cycle=cycle,
                    log_fn=log_fn,
                )
                if real_conflict_gt_probe_extended_20d_enabled:
                    extended_features = _build_extended_real_conflict_probe_features(
                        real_features,
                        task_probs[strict_positions],
                        clip_probs[strict_positions],
                        pool_task_probs.mean(dim=0),
                        pool_clip_probs.mean(dim=0),
                    )
                    extra = extended_features[:, 16:20].detach().float().cpu()
                    extra_std = extra.std(dim=0, unbiased=False)
                    log_fn(
                        "DUET real-conflict 20D extra-feature distribution "
                        "eval-only: cycle={}; reference=cycle_high_confidence_"
                        "agreement_mean_posterior; reference_n={}; "
                        "task_profile_drift_mean={:.6f}; "
                        "task_profile_drift_std={:.6f}; "
                        "clip_profile_drift_mean={:.6f}; "
                        "clip_profile_drift_std={:.6f}; "
                        "task_clip_js_mean={:.6f}; task_clip_js_std={:.6f}; "
                        "ranking_disagreement_mean={:.6f}; "
                        "ranking_disagreement_std={:.6f}; "
                        "construction_uses_gt=False; formal_method_affected=False".format(
                            cycle,
                            anchor_count,
                            extra[:, 0].mean().item(),
                            extra_std[0].item(),
                            extra[:, 1].mean().item(),
                            extra_std[1].item(),
                            extra[:, 2].mean().item(),
                            extra_std[2].item(),
                            extra[:, 3].mean().item(),
                            extra_std[3].item(),
                        )
                    )
                    probe_20d = _log_real_conflict_gt_feature_probe_eval_only(
                        extended_features,
                        task_top1[strict_positions],
                        clip_top1[strict_positions],
                        labels[strict_positions],
                        logits,
                        folds=real_conflict_gt_probe_folds,
                        steps=real_conflict_gt_probe_steps,
                        hidden=real_conflict_gt_probe_hidden,
                        lr=real_conflict_gt_probe_lr,
                        seed=seed,
                        cycle=cycle,
                        log_fn=log_fn,
                        feature_label=(
                            "20D_16D_plus_task_profile_drift_"
                            "clip_profile_drift_js_ranking_disagreement"
                        ),
                        log_variant="20D",
                    )
                    if (
                        probe_16d.get("status") == "ok"
                        and probe_20d.get("status") == "ok"
                    ):
                        log_fn(
                            "DUET real-conflict GT feature probe paired summary "
                            "eval-only: cycle={}; reference=cycle_high_confidence_"
                            "agreement_mean_posterior; reference_n={}; "
                            "fold_assignment_identical=True; "
                            "logistic_16d_acc={:.2f}%; logistic_20d_acc={:.2f}%; "
                            "logistic_delta={:+.2f}pp; "
                            "mlp_16d_acc={:.2f}%; mlp_20d_acc={:.2f}%; "
                            "mlp_delta={:+.2f}pp; "
                            "logistic_16d_conditional={:.2f}%; "
                            "logistic_20d_conditional={:.2f}%; "
                            "candidate_oracle_acc={:.2f}%; "
                            "extra_features=task_profile_drift,clip_profile_drift,"
                            "task_clip_js,all_pair_ranking_disagreement; "
                            "profile_sort_removes_class_identity=True; "
                            "formal_comparator_dim=16; formal_method_affected=False; "
                            "ground_truth_affects_training=False".format(
                                cycle,
                                anchor_count,
                                probe_16d["logistic_probe_acc"],
                                probe_20d["logistic_probe_acc"],
                                probe_20d["logistic_probe_acc"]
                                - probe_16d["logistic_probe_acc"],
                                probe_16d["mlp_probe_acc"],
                                probe_20d["mlp_probe_acc"],
                                probe_20d["mlp_probe_acc"]
                                - probe_16d["mlp_probe_acc"],
                                probe_16d[
                                    "logistic_conditional_arbitration_acc"
                                ],
                                probe_20d[
                                    "logistic_conditional_arbitration_acc"
                                ],
                                probe_16d["candidate_oracle_acc"],
                            )
                        )
            decision = apply_pairwise_decision(
                logits,
                real_candidate_a,
                real_candidate_b,
                gate=gate,
                coverage_fraction=coverage_fraction,
            )
            # Diagnostic only: summarize every real strict conflict, including
            # abstained rows.  No target labels or random operations are used.
            _log_real_comparator_margin_distribution(
                decision["margin"], cycle, gate, log_fn
            )
            selected_count = int(decision["resolved"].sum().item())
            switch_count = int(
                (
                    decision["resolved"]
                    & (decision["chosen"] != real_candidate_a.cpu())
                )
                .sum()
                .item()
            )
            achieved_coverage = (
                100.0 * selected_count / int(strict_positions.numel())
            )
            log_fn(
                "DUET comparator selection: cycle={}; mode={}; selected={}; "
                "total={}; achieved_coverage={:.2f}%; "
                "requested_coverage={:.2f}%; absolute_gate={:.2f}; "
                "absolute_gate_ignored={}; candidate_semantics={}; "
                "switches_from_fallback={}; "
                "ground_truth_affects_training=False".format(
                    cycle,
                    decision["selection_mode"],
                    selected_count,
                    int(strict_positions.numel()),
                    achieved_coverage,
                    100.0 * coverage_fraction,
                    gate,
                    decision["selection_mode"] == "rank_coverage",
                    (
                        "duet_fallback_vs_challenger"
                        if real_multiview_residual_fallback
                        else "task_vs_clip"
                    ),
                    switch_count,
                )
            )
            resolved_rows = decision["resolved"]
            resolved_strict = strict_positions[resolved_rows]
            resolved_mask[resolved_strict] = True
            context_labels[resolved_strict] = decision["chosen"][
                resolved_rows
            ].long()
            if real_multiview_residual_fallback:
                # Safe residual semantics: keeping fallback must be exactly a
                # no-op relative to original DUET's CLIP KL target. Only rows
                # whose router actually chooses the challenger are modified.
                winning_distribution = clip_probs[strict_positions].clone()
                residual_distribution = 0.5 * (
                    task_probs[strict_positions]
                    + clip_probs[strict_positions]
                )
                switch_rows = decision["chosen"] != real_candidate_a.cpu()
                rows = torch.nonzero(switch_rows, as_tuple=False).flatten()
                pair_mass = (
                    residual_distribution[rows, real_candidate_a[rows]]
                    + residual_distribution[rows, real_candidate_b[rows]]
                )
                if rows.numel() > 0:
                    winning_distribution[rows] = residual_distribution[rows]
                    winning_distribution[rows, real_candidate_a[rows]] = (
                        decision["trust_task"][rows].cpu() * pair_mass
                    )
                    winning_distribution[rows, real_candidate_b[rows]] = (
                        decision["trust_clip"][rows].cpu() * pair_mass
                    )
            else:
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
            if real_multiview_residual_fallback:
                chosen = decision["chosen"]
                strict_task_candidate = task_top1[strict_positions]
                strict_clip_candidate = clip_top1[strict_positions]
                stats["support_task"] = int(
                    (resolved_rows & (chosen == strict_task_candidate)).sum().item()
                )
                stats["support_clip"] = int(
                    (resolved_rows & (chosen == strict_clip_candidate)).sum().item()
                )
                stats["third_class"] = int(
                    (
                        resolved_rows
                        & (chosen != strict_task_candidate)
                        & (chosen != strict_clip_candidate)
                    )
                    .sum()
                    .item()
                )
            else:
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
    soft_only_admission = bool(
        getattr(context_cfg, "SOFT_ONLY_ADMISSION", False)
    )
    if soft_only_admission:
        stats["final_admitted"] = stats["post_prior_agreement"]
        stats["admitted_delta"] = 0
    else:
        stats["final_admitted"] = (
            stats["post_prior_agreement"]
            - stats["weak_deferred"]
            + stats["resolved_strict"]
        )
        stats["admitted_delta"] = (
            stats["resolved_strict"] - stats["weak_deferred"]
        )
    _log_correction_stats(stats, cycle, log_fn)
    _log_context_stats(stats, "comparator", cycle, log_fn)
    # Reliability-gate runs have their own same-subset fallback-vs-gated
    # evaluation above. The generic raw-comparator metric describes a
    # different decision path and is misleading for the method actually used.
    if (
        eval_only_logging
        and labels is not None
        and not reliability_gate_enabled
    ):
        _log_eval_only_metrics(
            stats,
            resolved_mask=resolved_mask,
            weak_rejected_mask=weak_rejected_mask,
            context_labels=context_labels,
            task_top1=task_top1,
            clip_top1=clip_top1,
            duet_fallback_top1=(task_probs + clip_probs).argmax(dim=1),
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
        "conflict_memory": conflict_memory_payload,
        "reliability_gate": reliability_gate_payload,
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
    duet_fallback_top1: Optional[torch.Tensor] = None,
    all_label: torch.Tensor,
    anchor_mask: torch.Tensor,
    weak_agreement_mask: torch.Tensor,
    strict_conflict_mask: torch.Tensor,
    cycle: int,
    log_fn: Callable[[str], None],
) -> None:
    """Target labels are read here only; they never affect training."""
    all_label = all_label.long()
    if duet_fallback_top1 is None:
        # Backward-compatible fallback for direct diagnostic callers. The
        # production pipeline always supplies the actual DUET mixed output.
        duet_fallback_top1 = task_top1

    def acc(pred: torch.Tensor, mask: torch.Tensor) -> str:
        if int(mask.sum().item()) == 0:
            return "nan"
        return _fmt_pct(float((pred[mask] == all_label[mask]).float().mean().item()))

    anchor_precision = acc(task_top1, anchor_mask)
    strict_task_acc = acc(task_top1, strict_conflict_mask)
    strict_clip_acc = acc(clip_top1, strict_conflict_mask)
    # Actual original DUET fallback before context intervention.
    strict_mix_acc = acc(duet_fallback_top1, strict_conflict_mask)
    # Same-subset diagnostic: all four accuracies below are evaluated on the
    # exact same rows selected by the comparator (resolved_mask).  The older
    # strict_task_acc / strict_clip_acc use the full strict-conflict set and
    # therefore must not be compared directly with resolved_acc.
    resolved_subset_task_acc = acc(task_top1, resolved_mask)
    resolved_subset_clip_acc = acc(clip_top1, resolved_mask)
    resolved_subset_duet_fallback_acc = acc(duet_fallback_top1, resolved_mask)
    resolved_comparator_acc = acc(context_labels, resolved_mask)
    resolved_count_for_gain = int(resolved_mask.sum().item())
    if resolved_count_for_gain == 0:
        resolved_gain_over_duet_fallback = "nan"
        coverage_weighted_gain_over_duet_fallback = "nan"
    else:
        comparator_value = float(
            (context_labels[resolved_mask] == all_label[resolved_mask])
            .float()
            .mean()
            .item()
        )
        fallback_value = float(
            (duet_fallback_top1[resolved_mask] == all_label[resolved_mask])
            .float()
            .mean()
            .item()
        )
        gain_pp = 100.0 * (comparator_value - fallback_value)
        strict_count = max(1, int(strict_conflict_mask.sum().item()))
        resolved_gain_over_duet_fallback = "{:+.2f}pp".format(gain_pp)
        coverage_weighted_gain_over_duet_fallback = "{:+.3f}pp".format(
            gain_pp * resolved_count_for_gain / strict_count
        )
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
        _fmt_pct(
            float(
                (
                    all_label[abstain_mask]
                    == duet_fallback_top1[abstain_mask]
                )
                .float()
                .mean()
                .item()
            )
        )
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
        "resolved_subset_clip_acc={}; resolved_subset_duet_fallback_acc={}; "
        "resolved_comparator_acc={}; resolved_gain_over_duet_fallback={}; "
        "coverage_weighted_gain_over_duet_fallback={}; "
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
            resolved_subset_duet_fallback_acc,
            resolved_comparator_acc,
            resolved_gain_over_duet_fallback,
            coverage_weighted_gain_over_duet_fallback,
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
