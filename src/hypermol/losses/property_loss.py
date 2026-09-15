"""Masked losses for classification and regression molecular properties."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_label_matrix(
    labels: Any,
    mask: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    labels = torch.as_tensor(labels)
    if labels.numel() == 0:
        raise ValueError("Training labels must not be empty.")
    if labels.ndim == 1:
        labels = labels[:, None]
        single_task = True
    elif labels.ndim == 2:
        single_task = labels.shape[1] == 1
    else:
        raise ValueError(f"Binary property labels must have shape [N] or [N, T]; got {tuple(labels.shape)}.")
    labels = labels.to(dtype=torch.float32)

    if mask is None:
        observed = torch.isfinite(labels)
    else:
        observed = torch.as_tensor(mask, dtype=torch.bool)
        if observed.ndim == 1:
            observed = observed[:, None]
        if observed.shape != labels.shape:
            raise ValueError(
                f"label_mask must match labels; got {tuple(observed.shape)} and {tuple(labels.shape)}."
            )
    if not observed.any():
        raise ValueError("Property labels contain no observed targets.")
    observed_values = labels[observed]
    if not torch.isfinite(observed_values).all() or not torch.all(
        (observed_values == 0) | (observed_values == 1)
    ):
        raise ValueError("Observed binary property labels must be finite and encoded as 0/1.")
    labels = torch.where(observed, labels, torch.zeros_like(labels))
    return labels, observed, single_task


def compute_pos_weight_from_labels(
    labels: Any,
    *,
    mask: Any | None = None,
    max_pos_weight: float | None = None,
) -> float | torch.Tensor:
    """Return training-only ``n_negative/n_positive`` weights per task.

    Missing multi-task labels are ignored. A multi-task endpoint with only one
    observed class receives a neutral weight of one and will be excluded from
    macro ROC-AUC whenever its evaluation split is also single-class. The
    original single-task contract remains fail-closed when either class is
    absent.
    """

    labels, observed, single_task = _as_label_matrix(labels, mask=mask)
    if max_pos_weight is not None:
        max_pos_weight = float(max_pos_weight)
        if not math.isfinite(max_pos_weight) or max_pos_weight <= 0:
            raise ValueError("max_pos_weight must be a finite positive number.")

    weights = []
    eligible_tasks = 0
    for task_index in range(labels.shape[1]):
        values = labels[:, task_index][observed[:, task_index]]
        positive_count = int((values == 1).sum().item())
        negative_count = int((values == 0).sum().item())
        if positive_count == 0 or negative_count == 0:
            if single_task:
                raise ValueError(
                    "pos_weight requires both classes in the training split; "
                    f"got n_negative={negative_count}, n_positive={positive_count}."
                )
            weight = 1.0
        else:
            eligible_tasks += 1
            weight = negative_count / positive_count
        if max_pos_weight is not None:
            weight = min(weight, max_pos_weight)
        weights.append(float(weight))
    if not single_task and eligible_tasks == 0:
        raise ValueError("No multi-task endpoint has both classes in the training split.")
    if single_task:
        return weights[0]
    return torch.tensor(weights, dtype=torch.float32)


class BinaryPropertyLoss(nn.Module):
    """Masked weighted BCE normalized by the effective observed-label weight."""

    def __init__(self, pos_weight: float | torch.Tensor | None = None):
        super().__init__()
        if pos_weight is None:
            pos_weight = 1.0
        value = torch.as_tensor(pos_weight, dtype=torch.float32)
        if value.ndim > 1 or value.numel() == 0:
            raise ValueError("pos_weight must be a scalar or one-dimensional task vector.")
        if not torch.isfinite(value).all() or not torch.all(value > 0):
            raise ValueError("pos_weight values must be finite and positive.")
        self.register_buffer("pos_weight", value)

    @classmethod
    def from_training_labels(
        cls,
        labels: Any,
        *,
        mask: Any | None = None,
        max_pos_weight: float | None = None,
    ) -> "BinaryPropertyLoss":
        return cls(
            compute_pos_weight_from_labels(
                labels,
                mask=mask,
                max_pos_weight=max_pos_weight,
            )
        )

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor] | torch.Tensor,
        batch: Mapping[str, Any] | torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del is_train

        logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs
        targets = batch["labels"] if isinstance(batch, Mapping) else batch
        label_mask = batch.get("label_mask") if isinstance(batch, Mapping) else None
        if not isinstance(logits, torch.Tensor):
            raise TypeError("outputs['logits'] must be a torch.Tensor.")
        if logits.ndim == 1:
            logits_matrix = logits[:, None]
        elif logits.ndim == 2:
            logits_matrix = logits
        else:
            raise ValueError(f"Binary property logits must have shape [N] or [N, T]; got {tuple(logits.shape)}.")
        target_matrix, observed, _single_task = _as_label_matrix(targets, mask=label_mask)
        target_matrix = target_matrix.to(device=logits.device, dtype=logits.dtype)
        observed = observed.to(device=logits.device)
        if logits_matrix.shape != target_matrix.shape:
            raise ValueError(
                "logits and labels must have identical [N, T] shapes; "
                f"got {tuple(logits_matrix.shape)} and {tuple(target_matrix.shape)}."
            )
        if not torch.isfinite(logits_matrix).all():
            raise ValueError("Binary property logits must be finite.")

        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        if pos_weight.numel() not in (1, logits_matrix.shape[1]):
            raise ValueError(
                f"pos_weight has {pos_weight.numel()} values for {logits_matrix.shape[1]} tasks."
            )
        per_label_loss = F.binary_cross_entropy_with_logits(
            logits_matrix,
            target_matrix,
            reduction="none",
            pos_weight=pos_weight,
        )
        sample_weights = torch.where(target_matrix > 0.5, pos_weight, torch.ones_like(target_matrix))
        total_loss_sum = per_label_loss[observed].sum()
        weighted_sample_sum = sample_weights[observed].sum()
        loss = total_loss_sum / weighted_sample_sum.clamp_min(torch.finfo(logits.dtype).tiny)

        detached_logits = logits_matrix.detach().cpu()
        detached_targets = target_matrix.detach().cpu()
        detached_mask = observed.detach().cpu()
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "total_loss_sum": float(total_loss_sum.detach().cpu()),
            "sample_size": int(observed.sum().item()),
            "molecule_size": int(logits_matrix.shape[0]),
            "weighted_sample_sum": float(weighted_sample_sum.detach().cpu()),
            "logits": detached_logits,
            "probs": torch.sigmoid(detached_logits),
            "targets": detached_targets,
            "label_mask": detached_mask,
        }


def compute_regression_target_stats(
    labels: Any,
    *,
    mask: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute training-only per-task mean and population standard deviation."""

    values = torch.as_tensor(labels, dtype=torch.float32)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(f"Regression labels must have shape [N] or [N, T]; got {tuple(values.shape)}.")
    observed = torch.isfinite(values) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
    if observed.ndim == 1:
        observed = observed[:, None]
    if observed.shape != values.shape:
        raise ValueError("Regression label_mask must match labels.")
    means = []
    standard_deviations = []
    for task_index in range(values.shape[1]):
        task_values = values[:, task_index][observed[:, task_index]]
        if task_values.numel() < 2 or not torch.isfinite(task_values).all():
            raise ValueError(f"Regression task {task_index} needs at least two finite training labels.")
        mean = task_values.mean()
        std = task_values.std(unbiased=False)
        if not torch.isfinite(std) or float(std) <= 0:
            raise ValueError(f"Regression task {task_index} has zero/non-finite training standard deviation.")
        means.append(mean)
        standard_deviations.append(std)
    return torch.stack(means), torch.stack(standard_deviations)


class RegressionPropertyLoss(nn.Module):
    """Masked MSE on training-standardized targets.

    The model predicts standardized targets. ``denormalize`` converts those
    outputs back to the original MoleculeNet units for metrics and exports.
    """

    def __init__(self, target_mean: Any, target_std: Any):
        super().__init__()
        mean = torch.as_tensor(target_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(target_std, dtype=torch.float32).reshape(-1)
        if mean.numel() == 0 or mean.shape != std.shape:
            raise ValueError("target_mean and target_std must be matching non-empty task vectors.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not torch.all(std > 0):
            raise ValueError("Regression normalization statistics must be finite with positive std.")
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)

    def _prediction_matrix(self, outputs: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        predictions = outputs["logits"] if isinstance(outputs, Mapping) else outputs
        if not isinstance(predictions, torch.Tensor):
            raise TypeError("Regression model outputs must be a torch.Tensor.")
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        elif predictions.ndim != 2:
            raise ValueError(
                f"Regression predictions must have shape [N] or [N, T]; got {tuple(predictions.shape)}."
            )
        if predictions.shape[1] != self.target_mean.numel():
            raise ValueError(
                f"Regression output has {predictions.shape[1]} tasks but normalization has "
                f"{self.target_mean.numel()}."
            )
        if not torch.isfinite(predictions).all():
            raise ValueError("Regression predictions must be finite.")
        return predictions

    def denormalize(self, outputs: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        predictions = self._prediction_matrix(outputs)
        mean = self.target_mean.to(device=predictions.device, dtype=predictions.dtype)
        std = self.target_std.to(device=predictions.device, dtype=predictions.dtype)
        return predictions * std + mean

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor] | torch.Tensor,
        batch: Mapping[str, Any] | torch.Tensor,
        is_train: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del is_train
        predictions = self._prediction_matrix(outputs)
        targets = batch["labels"] if isinstance(batch, Mapping) else batch
        label_mask = batch.get("label_mask") if isinstance(batch, Mapping) else None
        targets = torch.as_tensor(targets, device=predictions.device, dtype=predictions.dtype)
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.shape != predictions.shape:
            raise ValueError(
                f"Regression predictions and labels must match; got {predictions.shape} and {targets.shape}."
            )
        observed = torch.isfinite(targets) if label_mask is None else torch.as_tensor(
            label_mask, device=predictions.device, dtype=torch.bool
        )
        if observed.ndim == 1:
            observed = observed[:, None]
        if observed.shape != targets.shape or not observed.any():
            raise ValueError("Regression label mask is invalid or contains no observed targets.")
        if not torch.isfinite(targets[observed]).all():
            raise ValueError("Observed regression targets must be finite.")
        mean = self.target_mean.to(device=predictions.device, dtype=predictions.dtype)
        std = self.target_std.to(device=predictions.device, dtype=predictions.dtype)
        normalized_targets = (torch.where(observed, targets, mean) - mean) / std
        squared_errors = (predictions - normalized_targets).square()
        total_loss_sum = squared_errors[observed].sum()
        sample_size = int(observed.sum().item())
        loss = total_loss_sum / sample_size
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "total_loss_sum": float(total_loss_sum.detach().cpu()),
            "sample_size": sample_size,
            "molecule_size": int(predictions.shape[0]),
            "weighted_sample_sum": float(sample_size),
            "predictions": self.denormalize(predictions).detach().cpu(),
            "targets": targets.detach().cpu(),
            "label_mask": observed.detach().cpu(),
        }


__all__ = [
    "BinaryPropertyLoss",
    "RegressionPropertyLoss",
    "compute_pos_weight_from_labels",
    "compute_regression_target_stats",
]
