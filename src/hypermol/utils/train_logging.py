"""Training console/file logging helpers."""

from __future__ import annotations

import csv
import json
import math
import numbers
import os
import pprint
import sys
from typing import Any, Mapping

from tqdm import tqdm

try:
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - exercised only when loguru is unavailable.
    _loguru_logger = None


class _FallbackLogger:
    def _format(self, message: str, *args: Any, **kwargs: Any) -> str:
        if args or kwargs:
            try:
                return str(message).format(*args, **kwargs)
            except Exception:
                return " ".join([str(message), *(str(arg) for arg in args)])
        return str(message)

    def _write(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        text = self._format(message, *args, **kwargs)
        print(f"{level} | {text}")
        sys.stdout.flush()

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._write("INFO", message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._write("WARNING", message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._write("ERROR", message, *args, **kwargs)


logger = _loguru_logger if _loguru_logger is not None else _FallbackLogger()


def setup_training_logger(save_dir: str, filename: str = "training_log.txt") -> str:
    """Configure a kg_mol_old-style logger that writes to console and file."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)
    if _loguru_logger is None:
        return log_path

    logger.remove()
    logger.configure(extra={"run_ctx": ""})
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<cyan>{name}</cyan> | "
        "<level>{level}</level> | "
        "<level>{message}</level>",
        colorize=True,
    )
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}",
        enqueue=True,
        encoding="utf-8",
    )
    return log_path


def log_config(config: Mapping[str, Any], title: str = "Configuration") -> None:
    try:
        formatted = pprint.pformat(dict(config), indent=4, width=100, sort_dicts=True)
    except TypeError:
        formatted = pprint.pformat(dict(config), indent=4, width=100)
    title_line = f"| {title:^60} |"
    border = "+" + "-" * (len(title_line) - 2) + "+"
    logger.info("\n{}\n{}\n{}\n{}\n{}", border, title_line, border, formatted, border)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def round_metric_values(value: Any, ndigits: int = 4) -> Any:
    """Convert metric payloads to JSON-safe values and round float-like values."""
    value = _json_safe(value)
    if isinstance(value, Mapping):
        return {str(k): round_metric_values(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_metric_values(v, ndigits=ndigits) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return round(numeric, ndigits) if math.isfinite(numeric) else numeric
    return value


def format_metric_values(value: Any, ndigits: int = 4) -> Any:
    """Format metric payloads for human-readable logs with fixed decimal places."""
    value = _json_safe(value)
    if isinstance(value, Mapping):
        return {str(k): format_metric_values(v, ndigits=ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [format_metric_values(v, ndigits=ndigits) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return f"{numeric:.{ndigits}f}" if math.isfinite(numeric) else str(numeric)
    return value


def format_metric_payload(payload: Any, ndigits: int = 4) -> str:
    return json.dumps(format_metric_values(payload, ndigits=ndigits), ensure_ascii=False, sort_keys=True)


def metrics_to_string(metrics: Mapping[str, Any], skip_keys: set[str] | None = None) -> str:
    skip = skip_keys or {"preds", "targets", "slot_preds", "slot_targets", "topk_preds", "topk_targets"}
    parts = []
    for key in sorted(metrics):
        if key in skip:
            continue
        value = _json_safe(metrics[key])
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any] | None = None,
) -> None:
    message = (
        f"Epoch {epoch}/{total_epochs} | "
        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
    )
    train_text = metrics_to_string(train_metrics)
    val_text = metrics_to_string(val_metrics)
    if train_text:
        message += f" | train: {train_text}"
    if val_text:
        message += f" | val: {val_text}"
    if test_metrics:
        test_text = metrics_to_string(test_metrics)
        if test_text:
            message += f" | test: {test_text}"
    logger.info(message)


def log_best_model(
    epoch: int,
    val_loss: float,
    val_metrics: Mapping[str, Any],
    test_loss: float,
    test_metrics: Mapping[str, Any],
) -> None:
    logger.error(
        "NEW BEST MODEL | VAL  | epoch={} | loss={:.4f} | metrics={}",
        epoch,
        float(val_loss),
        format_metric_payload(val_metrics),
    )
    logger.error(
        "NEW BEST MODEL | TEST | epoch={} | loss={:.4f} | metrics={}",
        epoch,
        float(test_loss),
        format_metric_payload(test_metrics),
    )


def format_lrs(optimizer: Any) -> str:
    values = []
    for group in getattr(optimizer, "param_groups", []):
        lr = group.get("lr")
        if lr is not None:
            values.append(f"{float(lr):.6g}")
    return ", ".join(values) if values else "n/a"


def append_epoch_metrics(
    csv_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_loss: float,
    is_best: bool,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    test_loss: float | None = None,
    test_metrics: Mapping[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row = {
        "epoch": epoch,
        "train_loss": f"{float(train_loss):.4f}",
        "val_loss": f"{float(val_loss):.4f}",
        "best_loss": f"{float(best_loss):.4f}",
        "is_best": int(is_best),
        "train_metrics": format_metric_payload(train_metrics),
        "val_metrics": format_metric_payload(val_metrics),
        "test_loss": "" if test_loss is None else f"{float(test_loss):.4f}",
        "test_metrics": "" if test_metrics is None else format_metric_payload(test_metrics),
    }
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def write_metrics_json(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(format_metric_values(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
