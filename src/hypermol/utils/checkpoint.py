"""Checkpoint helpers."""

from __future__ import annotations

import os
import random
import hashlib
from typing import Any, Dict, Optional

import numpy as np
import torch


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def capture_rng_state() -> Dict[str, Any]:
    """Capture all RNG streams used by deterministic single-process training."""

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG streams saved by :func:`capture_rng_state`."""

    if not isinstance(state, dict):
        raise TypeError("Checkpoint RNG state must be a dictionary.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda_all")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG state count does not match the visible CUDA device count: "
                f"checkpoint={len(cuda_states)}, visible={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    # Checkpoints are local, trusted training artifacts and include optimizer
    # plus Python/NumPy RNG state.  PyTorch 2.6 defaults to weights_only=True,
    # which cannot deserialize those non-tensor fields.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return torch.load(path, map_location=map_location)


def load_checkpoint_if_available(path: str, map_location: str = "cpu") -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        return load_checkpoint(path, map_location=map_location)
    except FileNotFoundError:
        return None
