"""Utilities for selecting pretraining tasks from config."""

from __future__ import annotations

from collections.abc import Iterable


DEFAULT_PRETRAIN_TASKS: tuple[str, ...] = (
    "mam",
    "angle",
    "torsion",
    "fingerprint",
    "r_matrix",
    "electron_conservation",
)

PRETRAIN_TASK_ALIASES = {
    "atom": "mam",
    "atom_mask": "mam",
    "masked_atom": "mam",
    "masked_atom_modeling": "mam",
    "molecular_fingerprint": "fingerprint",
    "fp": "fingerprint",
    "rmat": "r_matrix",
    "r_matrix_prediction": "r_matrix",
    "delta_be": "r_matrix",
    "delta_be_matrix": "r_matrix",
    "be_delta": "r_matrix",
    "electron": "electron_conservation",
    "electron_conservation_loss": "electron_conservation",
}


def _split_task_string(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def normalize_pretrain_tasks(value: object = None) -> tuple[str, ...]:
    """Normalize config task selectors.

    Accepted examples:
      - missing/None/"all" -> all default tasks
      - ["mam", "r_matrix"]
      - "mam,r_matrix,fingerprint"
      - {"enabled": ["mam", "r_matrix"]}
      - {"mam": true, "angle": false, "r_matrix": true}
    """

    if value is None:
        return DEFAULT_PRETRAIN_TASKS

    if isinstance(value, dict):
        if "enabled" in value:
            return normalize_pretrain_tasks(value.get("enabled"))
        value = [name for name, enabled in value.items() if bool(enabled)]

    if isinstance(value, str):
        if value.strip().lower() in {"", "all", "*"}:
            return DEFAULT_PRETRAIN_TASKS
        value = _split_task_string(value)

    if not isinstance(value, Iterable):
        raise ValueError(f"Invalid pretraining task selector: {value!r}")

    tasks: list[str] = []
    for raw_task in value:
        task = str(raw_task).strip().lower().replace("-", "_")
        if not task:
            continue
        if task in {"all", "*"}:
            return DEFAULT_PRETRAIN_TASKS
        task = PRETRAIN_TASK_ALIASES.get(task, task)
        if task not in DEFAULT_PRETRAIN_TASKS:
            valid = ", ".join(DEFAULT_PRETRAIN_TASKS)
            raise ValueError(f"Unknown pretraining task '{raw_task}'. Valid tasks: {valid}.")
        if task not in tasks:
            tasks.append(task)

    if not tasks:
        raise ValueError("At least one pretraining task must be enabled.")
    return tuple(tasks)


def get_pretrain_tasks_from_config(cfg: dict) -> tuple[str, ...]:
    if "tasks" in cfg:
        return normalize_pretrain_tasks(cfg["tasks"])
    return normalize_pretrain_tasks(cfg.get("pretrain_tasks"))


def get_model_required_tasks(tasks: object = None) -> tuple[str, ...]:
    """Return task outputs required by the model.

    Electron conservation is a loss on the predicted R/Delta-BE matrix, so it
    needs the R-matrix head even when the direct R-matrix loss is disabled.
    """

    normalized = list(normalize_pretrain_tasks(tasks))
    if "electron_conservation" in normalized and "r_matrix" not in normalized:
        normalized.append("r_matrix")
    return tuple(normalized)
