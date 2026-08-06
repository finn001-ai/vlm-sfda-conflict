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
    pre_prior_task_probs: Optional[torch.Tensor] = None,
    pre_prior_clip_probs: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    sample_indices: Optional[torch.Tensor] = None,
    use_strict_conflict: bool = True,
    use_weak_agreement: bool = True,
    anchors_per_class: int = 8,
    anchor_task_conf: float = 0.90,
    anchor_clip_conf: float = 0.90,
    anchor_task_entropy: float = 0.40,
    anchor_clip_entropy: float = 0.40,
    entropy_weight: float = 1.0,
    require_pre_post_prior_agreement: bool = True,
    weak_conf_threshold: float = 0.70,
    weak_entropy_threshold: float = 1.00,
    accept_conf: float = 0.75,
    accept_margin: float = 0.20,
    weak_accept_conf: float = 0.75,
    weak_accept_margin: float = 0.20,
    third_class_conf: float = 0.85,
    third_class_margin: float = 0.30,
    allow_third_class: bool = True,
    abstain_when_uncertain: bool = True,
    refiner_type: str = "transformer",
    transformer: Optional[DuetContextConflictTransformer] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train_steps_per_cycle: int = 100,
    train_batch_size: int = 64,
    seed: int = 2020,
    cycle: int = 1,
    eval_only_logging: bool = True,
    log_fn: Callable[[str], None] = logging.info,
) -> dict:
    """Run the whole context pipeline and return per-sample decisions + stats.

    All returned masks have shape [N] (or [N, C] for refined targets) and are
    aligned with the input ``task_probs`` rows.
    """
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
    weak_agreement_mask = (
        (task_top1 == clip_top1)
        & (
            (task_conf < weak_conf_threshold)
            | (clip_conf < weak_conf_threshold)
            | (task_entropy > weak_entropy_threshold)
            | (clip_entropy > weak_entropy_threshold)
        )
    )
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
                )["probabilities"]
        elif refiner_type == "cosine_knn":
            context_probs = cosine_knn_refine(
                query_features,
                anchor_features,
                anchor_labels,
                anchor_valid,
                num_classes,
                k=5,
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
        "anchors_total={}; anchors_per_class=[{}]; anchor_task_conf={:.4f}; "
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
    resolved_acc = acc(context_labels, resolved_mask)

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
        "resolved_acc={}; support_task_acc={}; support_clip_acc={}; "
        "third_acc={}; abstain_orig_acc={}; weak_orig_precision={}; "
        "weak_passed_precision={}; weak_deferred_precision={}; "
        "ground_truth_affects_training=False".format(
            cycle,
            anchor_precision,
            strict_task_acc,
            strict_clip_acc,
            strict_mix_acc,
            resolved_acc,
            support_task_acc,
            support_clip_acc,
            third_acc,
            abstain_orig_acc,
            weak_orig_precision,
            weak_passed_precision,
            weak_deferred_precision,
        )
    )
