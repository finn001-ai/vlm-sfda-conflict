"""Closed-form feature transport from source/CLIP agreement geometry."""

from __future__ import annotations

import torch
import torch.nn as nn


class AgreementCovarianceTransport(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        rank: int,
        min_anchors: int,
        max_gate: float,
        epsilon: float,
    ):
        super().__init__()
        if num_classes <= 1 or feature_dim <= 0:
            raise ValueError("Covariance transport dimensions must be positive")
        if rank <= 0 or rank >= feature_dim:
            raise ValueError("Covariance transport rank must be in [1, feature_dim)")
        if min_anchors <= rank:
            raise ValueError("Minimum anchors must exceed covariance rank")
        if not 0.0 < max_gate <= 1.0:
            raise ValueError("Covariance transport max gate must be in (0, 1]")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.min_anchors = int(min_anchors)
        self.max_gate = float(max_gate)
        self.epsilon = float(epsilon)

        self.register_buffer("means", torch.zeros(num_classes, feature_dim))
        self.register_buffer(
            "bases", torch.zeros(num_classes, rank, feature_dim)
        )
        self.register_buffer("residual_variance", torch.ones(num_classes))
        self.register_buffer(
            "anchor_counts", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "active_classes", torch.zeros(num_classes, dtype=torch.bool)
        )
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("fixed_source", torch.empty(0, dtype=torch.long))
        self.register_buffer("fixed_clip", torch.empty(0, dtype=torch.long))
        self.register_buffer("candidate_probability", torch.empty(0, 2))
        self._diagnostics = {
            "anchors": 0,
            "active_classes": 0,
            "fixed_conflicts": 0,
            "eligible_conflicts": 0,
            "eligible_coverage": 0.0,
            "mean_relative_shift": 0.0,
        }

    @torch.no_grad()
    def fit(self, features, source_label, clip_label, mix_probability):
        if bool(self.fitted.item()):
            raise ValueError("Agreement covariance geometry is already frozen")
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError("Features must be sample-by-feature_dim")
        if mix_probability.shape != (features.size(0), self.num_classes):
            raise ValueError("Mix probability shape does not match features")
        if (
            source_label.numel() != features.size(0)
            or clip_label.numel() != features.size(0)
        ):
            raise ValueError("Candidate labels and features have different samples")

        features = features.detach().float().cpu()
        source_label = source_label.detach().long().cpu()
        clip_label = clip_label.detach().long().cpu()
        mix_probability = mix_probability.detach().float().cpu()
        if source_label.min() < 0 or source_label.max() >= self.num_classes:
            raise ValueError("Source candidate is outside the class range")
        if clip_label.min() < 0 or clip_label.max() >= self.num_classes:
            raise ValueError("CLIP candidate is outside the class range")

        agreement = source_label == clip_label
        means = torch.zeros_like(self.means, device="cpu")
        bases = torch.zeros_like(self.bases, device="cpu")
        residual_variance = torch.ones_like(
            self.residual_variance, device="cpu"
        )
        anchor_counts = torch.zeros_like(self.anchor_counts, device="cpu")
        active_classes = torch.zeros_like(self.active_classes, device="cpu")

        for class_index in range(self.num_classes):
            rows = agreement & (source_label == class_index)
            count = int(rows.sum().item())
            anchor_counts[class_index] = count
            if count < self.min_anchors:
                continue
            class_features = features[rows]
            class_mean = class_features.mean(dim=0)
            centered = class_features - class_mean
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            class_basis = vh[: self.rank]
            projection = (centered @ class_basis.t()) @ class_basis
            residual = centered - projection

            means[class_index] = class_mean
            bases[class_index] = class_basis
            residual_variance[class_index] = residual.square().mean().clamp_min(
                self.epsilon
            )
            active_classes[class_index] = True

        device = self.means.device
        self.means.copy_(means.to(device))
        self.bases.copy_(bases.to(device))
        self.residual_variance.copy_(residual_variance.to(device))
        self.anchor_counts.copy_(anchor_counts.to(device))
        self.active_classes.copy_(active_classes.to(device))
        self.fixed_source = source_label.to(device)
        self.fixed_clip = clip_label.to(device)
        rows = torch.arange(features.size(0))
        candidate_probability = torch.stack(
            (
                mix_probability[rows, source_label],
                mix_probability[rows, clip_label],
            ),
            dim=1,
        )
        self.candidate_probability = candidate_probability.to(device)
        self.fitted.fill_(True)

        diagnostics = self.measure(
            features.to(device),
            torch.arange(features.size(0), device=device),
        )
        self._diagnostics = diagnostics
        return dict(diagnostics)

    def _project(self, features, labels):
        means = self.means[labels].to(features.dtype)
        bases = self.bases[labels].to(features.dtype)
        centered = features - means
        coordinate = torch.einsum("bd,brd->br", centered, bases)
        projection = means + torch.einsum("br,brd->bd", coordinate, bases)
        residual_mse = (features - projection).float().square().mean(dim=1)
        energy = residual_mse / self.residual_variance[labels].clamp_min(
            self.epsilon
        )
        return projection, energy

    def _transport_delta(self, features, sample_indices):
        sample_indices = sample_indices.to(
            device=features.device, dtype=torch.long
        )
        if sample_indices.numel() != features.size(0):
            raise ValueError("Covariance transport needs one sample index per feature")
        if (
            sample_indices.min() < 0
            or sample_indices.max() >= self.fixed_source.numel()
        ):
            raise ValueError("Covariance transport sample index is out of range")

        source = self.fixed_source[sample_indices]
        clip = self.fixed_clip[sample_indices]
        eligible = (
            (source != clip)
            & self.active_classes[source]
            & self.active_classes[clip]
        )
        source_projection, source_energy = self._project(features, source)
        clip_projection, clip_energy = self._project(features, clip)

        probability = self.candidate_probability[sample_indices].float()
        candidate_mass = probability.sum(dim=1).clamp(max=1.0)
        geometry_log_weight = torch.stack(
            (
                probability[:, 0].clamp_min(self.epsilon).log()
                - 0.5 * source_energy,
                probability[:, 1].clamp_min(self.epsilon).log()
                - 0.5 * clip_energy,
            ),
            dim=1,
        )
        geometry_weight = torch.softmax(geometry_log_weight, dim=1).to(
            features.dtype
        )
        target = (
            geometry_weight[:, :1] * source_projection
            + geometry_weight[:, 1:] * clip_projection
        )
        raw_delta = candidate_mass.to(features.dtype).unsqueeze(1) * (
            target - features
        )
        raw_delta = raw_delta * eligible.to(features.dtype).unsqueeze(1)

        raw_norm = raw_delta.float().norm(dim=1, keepdim=True)
        feature_norm = features.detach().float().norm(
            dim=1, keepdim=True
        ).clamp_min(self.epsilon)
        max_norm = self.max_gate * feature_norm
        scale = torch.minimum(
            torch.ones_like(raw_norm), max_norm / raw_norm.clamp_min(self.epsilon)
        )
        return raw_delta * scale.to(raw_delta.dtype), eligible

    def forward(self, features, sample_indices):
        if not bool(self.fitted.item()):
            return features
        delta, _ = self._transport_delta(features, sample_indices)
        return features + delta

    @torch.no_grad()
    def measure(self, features, sample_indices):
        if not bool(self.fitted.item()):
            return dict(self._diagnostics)
        features = features.to(device=self.means.device, dtype=self.means.dtype)
        delta, eligible = self._transport_delta(features, sample_indices)
        conflict = self.fixed_source != self.fixed_clip
        fixed_conflicts = int(conflict.sum().item())
        eligible_conflicts = int(eligible.sum().item())
        if eligible.any():
            relative_shift = delta[eligible].norm(dim=1) / features[
                eligible
            ].norm(dim=1).clamp_min(self.epsilon)
            mean_relative_shift = float(relative_shift.mean().item())
        else:
            mean_relative_shift = 0.0
        return {
            "anchors": int((self.fixed_source == self.fixed_clip).sum().item()),
            "active_classes": int(self.active_classes.sum().item()),
            "fixed_conflicts": fixed_conflicts,
            "eligible_conflicts": eligible_conflicts,
            "eligible_coverage": (
                eligible_conflicts / fixed_conflicts if fixed_conflicts else 0.0
            ),
            "mean_relative_shift": mean_relative_shift,
        }

    def diagnostics(self):
        return dict(self._diagnostics)
