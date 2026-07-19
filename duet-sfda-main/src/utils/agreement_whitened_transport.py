"""Global feature transport selected by held-out source/CLIP agreements."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shrunk_covariance(values, shrinkage: float, epsilon: float):
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / max(values.size(0) - 1, 1)
    isotropic_scale = covariance.trace() / covariance.size(0)
    identity = torch.eye(
        covariance.size(0), device=covariance.device, dtype=covariance.dtype
    )
    return (
        (1.0 - shrinkage) * covariance
        + shrinkage * isotropic_scale * identity
        + epsilon * identity
    )


def _symmetric_power(matrix, exponent: float, epsilon: float):
    matrix = 0.5 * (matrix + matrix.t())
    eigenvalue, eigenvector = torch.linalg.eigh(matrix)
    powered = eigenvalue.clamp_min(epsilon).pow(exponent)
    return (eigenvector * powered.unsqueeze(0)) @ eigenvector.t()


class AgreementWhitenedTransport(nn.Module):
    """Whiten, align, and recolor target features toward classifier geometry."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        min_anchors: int,
        shrinkage: float,
        holdout_ratio: float,
        max_gate: float,
        min_improvement: float,
        epsilon: float,
    ):
        super().__init__()
        if num_classes <= 1 or feature_dim <= 0:
            raise ValueError("Whitened transport dimensions must be positive")
        if min_anchors < 4:
            raise ValueError("Whitened transport needs at least four anchors")
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError("Whitened transport shrinkage must be in (0, 1]")
        if not 0.0 < holdout_ratio < 0.5:
            raise ValueError("Whitened transport holdout ratio must be in (0, 0.5)")
        if not 0.0 < max_gate <= 1.0:
            raise ValueError("Whitened transport max gate must be in (0, 1]")
        if min_improvement < 0.0:
            raise ValueError("Whitened transport improvement must be nonnegative")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.min_anchors = int(min_anchors)
        self.shrinkage = float(shrinkage)
        self.holdout_ratio = float(holdout_ratio)
        self.max_gate = float(max_gate)
        self.min_improvement = float(min_improvement)
        self.epsilon = float(epsilon)

        self.register_buffer("input_mean", torch.zeros(feature_dim))
        self.register_buffer("reference_mean", torch.zeros(feature_dim))
        self.register_buffer("transport_matrix", torch.eye(feature_dim))
        self.register_buffer("selected_strength", torch.tensor(0.0))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))
        self._diagnostics = self._empty_diagnostics()

    def _empty_diagnostics(self):
        return {
            "anchors": 0,
            "train_anchors": 0,
            "heldout_anchors": 0,
            "active_classes": 0,
            "selected_strength": 0.0,
            "heldout_baseline_loss": 0.0,
            "heldout_selected_loss": 0.0,
            "heldout_loss_improvement": 0.0,
            "heldout_baseline_accuracy": 0.0,
            "heldout_selected_accuracy": 0.0,
            "heldout_accuracy_delta": 0.0,
            "mean_relative_shift": 0.0,
        }

    def _stratified_holdout(self, labels):
        holdout = torch.zeros(labels.numel(), dtype=torch.bool)
        for class_index in labels.unique(sorted=True):
            rows = torch.nonzero(labels == class_index, as_tuple=False).squeeze(1)
            if rows.numel() < 2:
                continue
            holdout_count = max(
                1, int(round(float(rows.numel()) * self.holdout_ratio))
            )
            holdout_count = min(holdout_count, int(rows.numel()) - 1)
            score = (rows * 1103515245 + 12345).remainder(2147483647)
            selected = rows[torch.argsort(score)[:holdout_count]]
            holdout[selected] = True
        return holdout

    def _fit_map(self, features, reference):
        input_mean = features.mean(dim=0)
        reference_mean = reference.mean(dim=0)
        input_centered = features - input_mean
        reference_centered = reference - reference_mean
        input_covariance = _shrunk_covariance(
            features, self.shrinkage, self.epsilon
        )
        reference_covariance = _shrunk_covariance(
            reference, self.shrinkage, self.epsilon
        )
        input_whitener = _symmetric_power(
            input_covariance, -0.5, self.epsilon
        )
        reference_whitener = _symmetric_power(
            reference_covariance, -0.5, self.epsilon
        )
        reference_colorer = _symmetric_power(
            reference_covariance, 0.5, self.epsilon
        )
        input_white = input_centered @ input_whitener
        reference_white = reference_centered @ reference_whitener
        u, _, vh = torch.linalg.svd(
            input_white.t() @ reference_white, full_matrices=False
        )
        rotation = u @ vh
        matrix = input_whitener @ rotation @ reference_colorer
        return input_mean, reference_mean, matrix

    def _apply_strength(self, features, strength: float):
        if strength <= 0.0:
            return features
        float_features = features.float()
        feature_norm = float_features.norm(dim=1, keepdim=True).clamp_min(
            self.epsilon
        )
        unit_features = float_features / feature_norm
        mapped = (
            (unit_features - self.input_mean.float())
            @ self.transport_matrix.float()
            + self.reference_mean.float()
        )
        mapped = F.normalize(mapped, dim=1, eps=self.epsilon) * feature_norm
        raw_delta = mapped - float_features
        raw_norm = raw_delta.norm(dim=1, keepdim=True)
        max_norm = float(strength) * feature_norm
        scale = torch.minimum(
            torch.ones_like(raw_norm),
            max_norm / raw_norm.clamp_min(self.epsilon),
        )
        return features + (raw_delta * scale).to(features.dtype)

    def _balanced_metrics(self, logits, labels):
        losses = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(dim=1) == labels
        class_losses = []
        class_accuracy = []
        for class_index in labels.unique(sorted=True):
            rows = labels == class_index
            class_losses.append(losses[rows].mean())
            class_accuracy.append(correct[rows].float().mean())
        return (
            float(torch.stack(class_losses).mean().item()),
            float(torch.stack(class_accuracy).mean().item()),
        )

    @torch.no_grad()
    def fit(
        self,
        features,
        source_label,
        clip_label,
        classifier_weight,
        classifier_bias=None,
    ):
        if bool(self.fitted.item()):
            raise ValueError("Agreement-whitened transport is already frozen")
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError("Features must be sample-by-feature_dim")
        if classifier_weight.shape != (self.num_classes, self.feature_dim):
            raise ValueError("Classifier weight shape does not match transport")
        if (
            source_label.numel() != features.size(0)
            or clip_label.numel() != features.size(0)
        ):
            raise ValueError("Agreement labels and features have different samples")

        features = features.detach().float().cpu()
        source_label = source_label.detach().long().cpu()
        clip_label = clip_label.detach().long().cpu()
        classifier_weight = classifier_weight.detach().float().cpu()
        classifier_bias = (
            classifier_bias.detach().float().cpu()
            if classifier_bias is not None
            else None
        )
        agreement_rows = torch.nonzero(
            source_label == clip_label, as_tuple=False
        ).squeeze(1)
        agreement_labels = source_label[agreement_rows]
        diagnostics = self._empty_diagnostics()
        diagnostics["anchors"] = int(agreement_rows.numel())

        if agreement_rows.numel() < self.min_anchors:
            self.fitted.fill_(True)
            self._diagnostics = diagnostics
            return dict(diagnostics)

        holdout_local = self._stratified_holdout(agreement_labels)
        train_local = ~holdout_local
        train_rows = agreement_rows[train_local]
        heldout_rows = agreement_rows[holdout_local]
        train_labels = agreement_labels[train_local]
        heldout_labels = agreement_labels[holdout_local]
        diagnostics["train_anchors"] = int(train_rows.numel())
        diagnostics["heldout_anchors"] = int(heldout_rows.numel())
        train_classes = set(train_labels.tolist())
        heldout_classes = set(heldout_labels.tolist())
        diagnostics["active_classes"] = len(train_classes & heldout_classes)
        if train_rows.numel() < 2 or heldout_rows.numel() < 1:
            self.fitted.fill_(True)
            self._diagnostics = diagnostics
            return dict(diagnostics)

        unit_features = F.normalize(features, dim=1, eps=self.epsilon)
        unit_classifier = F.normalize(
            classifier_weight, dim=1, eps=self.epsilon
        )
        input_mean, reference_mean, matrix = self._fit_map(
            unit_features[train_rows].double(),
            unit_classifier[train_labels].double(),
        )
        device = self.input_mean.device
        self.input_mean.copy_(input_mean.float().to(device))
        self.reference_mean.copy_(reference_mean.float().to(device))
        self.transport_matrix.copy_(matrix.float().to(device))

        heldout_features = features[heldout_rows].to(device)
        heldout_labels = heldout_labels.to(device)
        weight = classifier_weight.to(device)
        bias = classifier_bias.to(device) if classifier_bias is not None else None
        baseline_logits = F.linear(heldout_features, weight, bias)
        baseline_loss, baseline_accuracy = self._balanced_metrics(
            baseline_logits, heldout_labels
        )
        selected_strength = 0.0
        selected_loss = baseline_loss
        selected_accuracy = baseline_accuracy
        strengths = [
            0.1 * self.max_gate,
            0.25 * self.max_gate,
            0.5 * self.max_gate,
            0.75 * self.max_gate,
            self.max_gate,
        ]
        for strength in strengths:
            candidate = self._apply_strength(heldout_features, strength)
            candidate_loss, candidate_accuracy = self._balanced_metrics(
                F.linear(candidate, weight, bias), heldout_labels
            )
            improvement = baseline_loss - candidate_loss
            accuracy_preserved = candidate_accuracy + 1e-12 >= baseline_accuracy
            if (
                accuracy_preserved
                and improvement >= self.min_improvement
            ):
                selected_strength = float(strength)
                selected_loss = candidate_loss
                selected_accuracy = candidate_accuracy
                break

        self.selected_strength.fill_(selected_strength)
        self.fitted.fill_(True)
        transported = self._apply_strength(features.to(device), selected_strength)
        relative_shift = (transported - features.to(device)).float().norm(
            dim=1
        ) / features.to(device).float().norm(dim=1).clamp_min(self.epsilon)
        diagnostics.update(
            {
                "selected_strength": selected_strength,
                "heldout_baseline_loss": baseline_loss,
                "heldout_selected_loss": selected_loss,
                "heldout_loss_improvement": baseline_loss - selected_loss,
                "heldout_baseline_accuracy": baseline_accuracy,
                "heldout_selected_accuracy": selected_accuracy,
                "heldout_accuracy_delta": selected_accuracy - baseline_accuracy,
                "mean_relative_shift": float(relative_shift.mean().item()),
            }
        )
        self._diagnostics = diagnostics
        return dict(diagnostics)

    def forward(self, features, sample_indices=None):
        del sample_indices
        if not bool(self.fitted.item()):
            return features
        return self._apply_strength(features, float(self.selected_strength.item()))

    def diagnostics(self):
        return dict(self._diagnostics)
