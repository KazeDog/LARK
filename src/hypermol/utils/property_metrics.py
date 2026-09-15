"""Metrics for masked classification and regression molecular properties.

The helpers in this module deliberately reject undefined evaluations instead
of silently returning ``0`` or ``nan``.  In particular, ROC-AUC is undefined
when an evaluation split contains only one class, so such a split is treated
as a protocol error.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _as_1d_numpy(values: Any, *, name: str) -> np.ndarray:
    """Convert a tensor/array-like single-task vector to a one-dimensional array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must describe one binary task with shape [N] or [N, 1]; "
            f"got shape {np.asarray(values).shape}."
        )
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable NumPy sigmoid."""

    logits = logits.astype(np.float64, copy=False)
    probabilities = np.empty_like(logits, dtype=np.float64)
    nonnegative = logits >= 0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probabilities[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return probabilities


def compute_binary_property_metrics(
    labels: Any,
    probabilities: Any | None = None,
    logits: Any | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute publication-facing metrics for a binary property task.

    Exactly one of ``probabilities`` or ``logits`` must be supplied.  Labels
    must be fully observed and encoded as 0/1.  A single-class label vector is
    rejected because ROC-AUC (the model-selection metric) is then undefined.

    Args:
        labels: Binary targets with shape ``[N]`` or ``[N, 1]``.
        probabilities: Positive-class probabilities, mutually exclusive with
            ``logits``.
        logits: Positive-class logits, mutually exclusive with
            ``probabilities``.
        threshold: Probability threshold used for discrete metrics.

    Returns:
        ROC-AUC, average precision (AP), accuracy, balanced accuracy, F1, and
        Matthews correlation coefficient.

    Raises:
        ValueError: If inputs are malformed, ambiguous, non-finite, or labels
            contain fewer than two classes.
    """

    if (probabilities is None) == (logits is None):
        raise ValueError("Provide exactly one of probabilities or logits.")
    if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be finite and in [0, 1], got {threshold!r}.")

    label_array = _as_1d_numpy(labels, name="labels")
    try:
        finite_labels = np.isfinite(label_array)
    except TypeError as exc:
        raise ValueError("labels must be numeric binary values encoded as 0/1.") from exc
    if not np.all(finite_labels) or not np.all(np.isin(label_array, (0, 1))):
        raise ValueError("labels must contain only finite binary values encoded as 0/1.")
    label_array = label_array.astype(np.int64, copy=False)
    if np.unique(label_array).size != 2:
        raise ValueError(
            "Binary property metrics require both negative and positive labels; "
            "ROC-AUC is undefined for a single-class split."
        )

    if probabilities is not None:
        probability_array = _as_1d_numpy(probabilities, name="probabilities")
        try:
            probability_array = probability_array.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("probabilities must be numeric.") from exc
        if not np.all(np.isfinite(probability_array)):
            raise ValueError("probabilities must contain only finite values.")
        if np.any((probability_array < 0.0) | (probability_array > 1.0)):
            raise ValueError("probabilities must lie in [0, 1].")
    else:
        logit_array = _as_1d_numpy(logits, name="logits")
        try:
            logit_array = logit_array.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("logits must be numeric.") from exc
        if not np.all(np.isfinite(logit_array)):
            raise ValueError("logits must contain only finite values.")
        probability_array = _sigmoid(logit_array)

    if probability_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            "labels and predictions must contain the same number of samples; "
            f"got {label_array.shape[0]} and {probability_array.shape[0]}."
        )

    predictions = (probability_array >= float(threshold)).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(label_array, probability_array)),
        "average_precision": float(average_precision_score(label_array, probability_array)),
        "accuracy": float(accuracy_score(label_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(label_array, predictions)),
        "f1": float(f1_score(label_array, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(label_array, predictions)),
    }


def _as_2d_numpy(values: Any, *, name: str) -> tuple[np.ndarray, bool]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim == 1:
        return array[:, None], True
    if array.ndim == 2:
        return array, array.shape[1] == 1
    raise ValueError(f"{name} must have shape [N] or [N, T]; got {array.shape}.")


def compute_multitask_property_metrics(
    labels: Any,
    probabilities: Any | None = None,
    logits: Any | None = None,
    *,
    mask: Any | None = None,
    task_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-task metrics and MoleculeNet-style macro averages.

    Missing labels are ignored. A task contributes to every macro metric only
    when its evaluated observations contain both classes, matching the
    requirement for a defined ROC-AUC. Top-level ``roc_auc`` and
    ``average_precision`` are therefore macro averages over eligible tasks.
    """

    if (probabilities is None) == (logits is None):
        raise ValueError("Provide exactly one of probabilities or logits.")
    label_matrix, _single = _as_2d_numpy(labels, name="labels")
    prediction_values = probabilities if probabilities is not None else logits
    prediction_matrix, _ = _as_2d_numpy(prediction_values, name="predictions")
    if label_matrix.shape != prediction_matrix.shape:
        raise ValueError(
            f"labels and predictions must have the same [N, T] shape; "
            f"got {label_matrix.shape} and {prediction_matrix.shape}."
        )
    if mask is None:
        observed = np.isfinite(label_matrix)
    else:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        observed, _ = _as_2d_numpy(mask, name="mask")
        observed = observed.astype(bool, copy=False)
        if observed.shape != label_matrix.shape:
            raise ValueError(
                f"mask must match labels; got {observed.shape} and {label_matrix.shape}."
            )
    names = [str(value) for value in (task_names or [f"task_{index}" for index in range(label_matrix.shape[1])])]
    if len(names) != label_matrix.shape[1] or len(names) != len(set(names)):
        raise ValueError("task_names must be unique and match the task dimension.")

    metric_names = ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "mcc")
    task_metrics: dict[str, dict[str, Any]] = {}
    eligible_values = {metric: [] for metric in metric_names}
    observed_label_count = 0
    for task_index, task_name in enumerate(names):
        task_mask = observed[:, task_index]
        task_labels = label_matrix[task_mask, task_index]
        task_predictions = prediction_matrix[task_mask, task_index]
        observed_label_count += int(task_mask.sum())
        if task_labels.size == 0:
            task_metrics[task_name] = {
                "eligible": False,
                "observed": 0,
                "negative": 0,
                "positive": 0,
                "reason": "no_observed_labels",
            }
            continue
        try:
            numeric_labels = task_labels.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Task {task_name} labels must be numeric.") from exc
        if not np.all(np.isfinite(numeric_labels)) or not np.all(np.isin(numeric_labels, (0, 1))):
            raise ValueError(f"Task {task_name} observed labels must be finite 0/1 values.")
        negative = int((numeric_labels == 0).sum())
        positive = int((numeric_labels == 1).sum())
        if negative == 0 or positive == 0:
            task_metrics[task_name] = {
                "eligible": False,
                "observed": int(task_labels.size),
                "negative": negative,
                "positive": positive,
                "reason": "single_class",
            }
            continue
        metrics = compute_binary_property_metrics(
            numeric_labels,
            probabilities=task_predictions if probabilities is not None else None,
            logits=task_predictions if logits is not None else None,
            threshold=threshold,
        )
        task_metrics[task_name] = {
            "eligible": True,
            "observed": int(task_labels.size),
            "negative": negative,
            "positive": positive,
            **metrics,
        }
        for metric in metric_names:
            eligible_values[metric].append(float(metrics[metric]))

    eligible_count = len(eligible_values["roc_auc"])
    if eligible_count == 0:
        raise ValueError("No property task contains both negative and positive labels; macro ROC-AUC is undefined.")
    result: dict[str, Any] = {
        metric: float(np.mean(values)) for metric, values in eligible_values.items()
    }
    result.update(
        {
            "aggregation": "macro_over_tasks_with_both_classes",
            "num_tasks": int(label_matrix.shape[1]),
            "eligible_tasks": int(eligible_count),
            "skipped_tasks": int(label_matrix.shape[1] - eligible_count),
            "observed_labels": int(observed_label_count),
            "task_metrics": task_metrics,
        }
    )
    return result


def compute_multitask_regression_metrics(
    labels: Any,
    predictions: Any,
    *,
    mask: Any | None = None,
    task_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-task and macro RMSE/MAE/Spearman in original target units."""

    label_matrix, _single = _as_2d_numpy(labels, name="labels")
    prediction_matrix, _ = _as_2d_numpy(predictions, name="predictions")
    if label_matrix.shape != prediction_matrix.shape:
        raise ValueError(
            f"Regression labels and predictions must have the same [N, T] shape; "
            f"got {label_matrix.shape} and {prediction_matrix.shape}."
        )
    if mask is None:
        observed = np.isfinite(label_matrix)
    else:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        observed, _ = _as_2d_numpy(mask, name="mask")
        observed = observed.astype(bool, copy=False)
        if observed.shape != label_matrix.shape:
            raise ValueError("Regression mask must match labels.")
    names = [str(value) for value in (task_names or [f"task_{index}" for index in range(label_matrix.shape[1])])]
    if len(names) != label_matrix.shape[1] or len(names) != len(set(names)):
        raise ValueError("task_names must be unique and match the task dimension.")

    task_metrics: dict[str, dict[str, Any]] = {}
    rmse_values: list[float] = []
    mae_values: list[float] = []
    spearman_values: list[float] = []
    observed_label_count = 0
    for task_index, task_name in enumerate(names):
        task_mask = observed[:, task_index]
        targets = label_matrix[task_mask, task_index].astype(np.float64, copy=False)
        estimates = prediction_matrix[task_mask, task_index].astype(np.float64, copy=False)
        observed_label_count += int(task_mask.sum())
        if targets.size == 0:
            task_metrics[task_name] = {"eligible": False, "observed": 0, "reason": "no_observed_labels"}
            continue
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(estimates)):
            raise ValueError(f"Regression task {task_name} contains non-finite observed values.")
        errors = estimates - targets
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        mae = float(np.mean(np.abs(errors)))
        # TDC's ``spearman`` evaluator is scipy.stats.spearmanr(y_true,
        # y_pred)[0]. Keep the same implementation here so benchmark scores
        # do not differ because of ranking or tie-handling conventions.
        spearman = float(spearmanr(targets, estimates)[0])
        if not np.isfinite(spearman):
            raise ValueError(
                f"Regression task {task_name} has undefined Spearman correlation; "
                "at least two non-constant target and prediction ranks are required."
            )
        task_metrics[task_name] = {
            "eligible": True,
            "observed": int(targets.size),
            "rmse": rmse,
            "mae": mae,
            "spearman": spearman,
        }
        rmse_values.append(rmse)
        mae_values.append(mae)
        spearman_values.append(spearman)
    if not rmse_values:
        raise ValueError("No regression task contains an observed label.")
    return {
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "spearman": float(np.mean(spearman_values)),
        "aggregation": "macro_over_tasks_with_observed_targets",
        "num_tasks": int(label_matrix.shape[1]),
        "eligible_tasks": int(len(rmse_values)),
        "skipped_tasks": int(label_matrix.shape[1] - len(rmse_values)),
        "observed_labels": int(observed_label_count),
        "task_metrics": task_metrics,
    }


__all__ = [
    "compute_binary_property_metrics",
    "compute_multitask_property_metrics",
    "compute_multitask_regression_metrics",
]
