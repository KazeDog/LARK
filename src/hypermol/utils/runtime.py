"""Runtime helpers shared by training scripts."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, device: torch.device | str | int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(int(seed))
    if device is not None and torch.cuda.is_available():
        cuda_device = torch.device("cuda", int(device)) if isinstance(device, int) else torch.device(device)
        if cuda_device.type == "cuda":
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def get_device(gpu: int = -1) -> torch.device:
    if torch.cuda.is_available() and gpu >= 0:
        gpu = int(gpu)
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"Requested cuda:{gpu}, but only {torch.cuda.device_count()} visible CUDA device(s) are available."
            )
        torch.cuda.set_device(gpu)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


class NullSummaryWriter:
    """Fallback writer when TensorBoard is not installed."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def build_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return NullSummaryWriter()
