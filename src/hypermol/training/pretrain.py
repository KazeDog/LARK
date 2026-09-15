"""Pretraining entrypoint for mapped ORDerly reaction data."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pretrain.yaml"

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from hypermol.data.collator import ReactionCollator
from hypermol.data.audit_be import audit_reaction_dataset_from_paths
from hypermol.data.reaction_dataset import ReactionDataset, clean_text
from hypermol.data.reaction_identity import (
    REACTION_IDENTITY_FALLBACK,
    REACTION_IDENTITY_METHOD,
    canonical_main_component_identity as _canonical_main_component_identity,
    canonical_side_identity as _canonical_side_identity,
)
from hypermol.losses.pretrain_loss import PretrainLoss
from hypermol.models.pretrain_model import PretrainModel
from hypermol.utils.checkpoint import load_checkpoint_if_available, save_checkpoint
from hypermol.utils.config import load_yaml_config
from hypermol.utils.metrics import reduce_metrics
from hypermol.utils.training_budget import (
    resolve_training_budget,
    validate_resume_training_budget,
)
from hypermol.utils.pretrain_tasks import get_pretrain_tasks_from_config
from hypermol.utils.runtime import build_summary_writer, get_device, move_to_device, set_seed
from hypermol.utils.train_logging import (
    append_epoch_metrics,
    format_lrs,
    format_metric_payload,
    log_best_model,
    log_config,
    log_epoch_summary,
    logger,
    setup_training_logger,
    write_json,
    write_metrics_json,
)


def resolve_split_file(cfg: dict, data_root: str, split: str) -> str:
    for key in (f"{split}_file", f"{split}_path", f"{split}_csv"):
        if cfg.get(key):
            value = cfg[key]
            return value if os.path.isabs(value) else os.path.join(data_root, value)
    parquet_path = os.path.join(data_root, f"{split}.parquet")
    if os.path.exists(parquet_path):
        return parquet_path
    return os.path.join(data_root, f"{split}.csv")


VALIDATION_MODES = {"auto", "explicit", "split_from_train"}
INPUT_GROUP_SCOPES = {"objective_input", "both_sides_connected"}
PERIODIC_CHECKPOINT_DIRECTORY = "periodic"


def resolve_checkpointing_config(cfg: dict) -> dict:
    """Normalize checkpoint retention without changing model-selection policy."""

    raw = cfg.get("checkpointing", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("checkpointing must be a mapping when configured.")
    value = raw.get("periodic_every_epochs", 0)
    if isinstance(value, bool):
        raise TypeError("checkpointing.periodic_every_epochs must be an integer.")
    try:
        interval = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "checkpointing.periodic_every_epochs must be an integer."
        ) from exc
    if interval < 0:
        raise ValueError("checkpointing.periodic_every_epochs must be >= 0.")
    return {
        "periodic_every_epochs": interval,
        "periodic_directory": PERIODIC_CHECKPOINT_DIRECTORY,
    }


def periodic_checkpoint_path(cfg: dict, save_dir: str, epoch: int) -> str | None:
    """Return the numbered snapshot path when ``epoch`` hits the configured interval."""

    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("Checkpoint epochs must be positive.")
    checkpointing = resolve_checkpointing_config(cfg)
    interval = checkpointing["periodic_every_epochs"]
    if interval <= 0 or epoch % interval:
        return None
    return os.path.join(
        save_dir,
        checkpointing["periodic_directory"],
        f"epoch_{epoch:04d}.pt",
    )


def maybe_save_periodic_checkpoint(
    cfg: dict,
    save_dir: str,
    epoch: int,
    state: dict,
) -> str | None:
    """Atomically save a full, resumable numbered checkpoint when scheduled."""

    path = periodic_checkpoint_path(cfg, save_dir, epoch)
    if path is None:
        return None
    save_checkpoint(path, state)
    return path


def get_validation_config(cfg: dict) -> dict:
    """Return the normalized validation-source configuration.

    ``auto`` preserves the historical behavior: use a resolved valid file when
    it exists, otherwise split the training source.  New experiment configs
    should prefer an explicit mode so adding a file to ``data_root`` cannot
    silently change the split protocol.
    """

    raw = cfg.get("validation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("validation must be a mapping when configured.")
    mode = str(raw.get("mode", "auto")).strip().lower()
    if mode not in VALIDATION_MODES:
        raise ValueError(
            "validation.mode must be one of "
            f"{sorted(VALIDATION_MODES)}, got {mode!r}."
        )
    fraction = raw.get(
        "fraction",
        cfg.get("valid_fraction", cfg.get("valid_ratio", 0.05)),
    )
    return {"mode": mode, "fraction": float(fraction)}


def _validation_fraction(cfg: dict) -> float:
    valid_fraction = float(get_validation_config(cfg)["fraction"])
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("validation.fraction must be strictly between 0 and 1.")
    return valid_fraction


def _split_seed(cfg: dict) -> int:
    """Return the data-partition seed independently from the training seed."""

    integrity_cfg = cfg.get("split_integrity", {}) or {}
    validation_cfg = cfg.get("validation", {}) or {}
    return int(
        integrity_cfg.get(
            "seed",
            validation_cfg.get("seed", cfg.get("seed", 42)),
        )
    )


def _input_group_scope(integrity_cfg: dict) -> str:
    value = str(integrity_cfg.get("group_scope", "objective_input")).strip().lower()
    if value not in INPUT_GROUP_SCOPES:
        raise ValueError(
            "split_integrity.group_scope must be one of "
            f"{sorted(INPUT_GROUP_SCOPES)}, got {value!r}."
        )
    return value


def _effective_input_group_side(cfg: dict, integrity_cfg: dict) -> str | None:
    objective_side = _directional_input_side(cfg)
    if objective_side is not None and _input_group_scope(integrity_cfg) == "both_sides_connected":
        return "both"
    return objective_side


def get_reaction_objective_config(cfg: dict) -> dict:
    value = cfg.get("reaction_objective", {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("reaction_objective must be a mapping when configured.")
    return dict(value)


def build_protocol_metadata(cfg: dict) -> dict:
    reaction_objective = get_reaction_objective_config(cfg)
    objective_mode = str(reaction_objective.get("mode", "reaction_reconstruction")).lower()
    dual_view = objective_mode in {"bidirectional", "bidirectional_single_side"}
    directional = objective_mode in {
        "product_to_reactant",
        "reactant_to_product",
        "bidirectional",
        "bidirectional_single_side",
    }
    protocol_name = str(
        cfg.get(
            "protocol_version",
            cfg.get(
                "pretrain_protocol",
                "dual_view_v1" if dual_view else ("directional_v2" if directional else "legacy_v1"),
            ),
        )
    )
    protocol_family = protocol_name.lower()
    if protocol_family.startswith("dual_view") and not dual_view:
        raise ValueError(
            f"protocol_version={protocol_name!r} requires reaction_objective.mode="
            "'bidirectional_single_side'."
        )
    if protocol_family.startswith("directional") and (not directional or dual_view):
        raise ValueError(
            f"protocol_version={protocol_name!r} requires reaction_objective.mode to be "
            "'product_to_reactant' or 'reactant_to_product'."
        )
    if protocol_family.startswith("legacy") and directional:
        raise ValueError(
            f"protocol_version={protocol_name!r} is incompatible with directional objective {objective_mode!r}."
        )
    metadata = {
        "name": protocol_name,
        "directional": directional,
        "dual_view": dual_view,
        "test_evaluation": "best_checkpoint_once" if directional else "every_epoch",
        "reaction_objective": reaction_objective,
    }
    if directional:
        condition_cfg = cfg.get("condition", {}) or {}
        context_cfg = cfg.get("context", {}) or {}
        consistency_cfg = cfg.get("consistency", {}) or {}
        loss_cfg = cfg.get("loss", {}) or {}
        model_cfg = cfg.get("model", {}) or {}
        be_cfg = cfg.get("be_matrix", {}) or {}
        metadata["corruption_parameters"] = {
            "mask_prob": float(cfg.get("mask_prob", 0.15)),
            "neighbor_mask_prob": float(cfg.get("neighbor_mask_prob", 0.3)),
            "leave_unmasked_prob": float(cfg.get("leave_unmasked_prob", 0.1)),
            "random_token_prob": float(cfg.get("random_token_prob", 0.1)),
        }
        metadata["shortcut_controls"] = {
            "condition_enabled": bool(condition_cfg.get("enabled", False)),
            "context_enabled": bool(context_cfg.get("enabled", False)),
            "r_matrix_use_be_pair_feature": bool(model_cfg.get("r_matrix_use_be_pair_feature", False)),
            "aromatic_mode": str(be_cfg.get("aromatic_mode", cfg.get("aromatic_mode", "aromatic_1p5"))),
        }
        if dual_view:
            metadata["context"] = {
                "enabled": bool(context_cfg.get("enabled", False)),
                "column": str(context_cfg.get("column", "unmapped_components")),
                "molecule_db": context_cfg.get("molecule_db"),
                "max_components": int(context_cfg.get("max_components", 0)),
                "missing_policy": str(context_cfg.get("missing_policy", "error")),
                "mask_prob": float(context_cfg.get("mask_prob", cfg.get("mask_prob", 0.15))),
            }
            metadata["consistency"] = {
                "enabled": bool(consistency_cfg.get("enabled", True)),
                "projection_dim": int(consistency_cfg.get("projection_dim", 128)),
                "loss": "paired_cosine_projection",
            }
            metadata["loss_composition"] = {
                "formula": "L_core + lambda_ctx * L_context + lambda_cons * L_consistency",
                "lambda_ctx": float(loss_cfg.get("lambda_ctx", 0.0)),
                "lambda_cons": float(loss_cfg.get("lambda_cons", 0.0)),
            }
    return metadata


def uses_directional_protocol(cfg: dict) -> bool:
    return bool(build_protocol_metadata(cfg)["directional"])


def checkpoint_protocol_metadata(checkpoint: dict) -> dict:
    """Resolve protocol metadata from new or historical checkpoints."""

    metadata = checkpoint.get("protocol_metadata") if isinstance(checkpoint, dict) else None
    if isinstance(metadata, dict):
        return dict(metadata)
    checkpoint_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    return build_protocol_metadata(checkpoint_cfg if isinstance(checkpoint_cfg, dict) else {})


def validate_resume_protocol(cfg: dict, checkpoint: dict) -> None:
    """Reject an exact resume when checkpoint and data contracts differ."""

    expected = build_protocol_metadata(cfg)
    observed = checkpoint_protocol_metadata(checkpoint)
    if expected != observed:
        raise ValueError(
            "Cannot resume optimizer/epoch state across pretraining protocols: "
            f"config={expected!r}, checkpoint={observed!r}. "
            "Use --init_checkpoint for a warm start with reset optimizer and training state."
        )


def validate_resume_data_contract(cfg: dict, checkpoint: dict) -> None:
    """Reject exact resume when split construction or data sources changed."""

    saved_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved_cfg, dict):
        saved_cfg = {}
    keys = (
        "data_root",
        "molecule_db",
        "train_file",
        "valid_file",
        "test_file",
        "seed",
        "validation",
        "valid_fraction",
        "valid_ratio",
        "split_integrity",
        "selection_metric",
        "selection_mode",
        "tasks",
        "model",
        "loss",
        "condition",
        "context",
        "consistency",
        "lr",
        "weight_decay",
        "batch_size",
        "mask_prob",
        "neighbor_mask_prob",
        "leave_unmasked_prob",
        "random_token_prob",
        "deterministic",
        "early_stop_patience",
        "early_stop_min_delta",
    )
    differences = {
        key: {"config": cfg.get(key), "checkpoint": saved_cfg.get(key)}
        for key in keys
        if cfg.get(key) != saved_cfg.get(key)
    }
    current_runtime = cfg.get("split_integrity_runtime")
    saved_runtime = saved_cfg.get("split_integrity_runtime")
    if current_runtime != saved_runtime:
        differences["split_integrity_runtime"] = {
            "config": current_runtime,
            "checkpoint": saved_runtime,
        }
    if differences:
        raise ValueError(
            "Cannot exactly resume with a different data/split contract: "
            f"{differences!r}. Use --init_checkpoint for a warm start."
        )


def load_compatible_initial_weights(model: torch.nn.Module, state_dict: dict) -> dict:
    """Warm-start shape-compatible weights and reinitialize an incompatible R head."""

    normalized = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    current = model.state_dict()
    r_head_incompatible = any(
        key.startswith("r_matrix_head.")
        and key in current
        and tuple(value.shape) != tuple(current[key].shape)
        for key, value in normalized.items()
    )
    compatible = {}
    skipped = []
    for key, value in normalized.items():
        if key not in current or tuple(value.shape) != tuple(current[key].shape):
            skipped.append(key)
            continue
        if r_head_incompatible and key.startswith("r_matrix_head."):
            skipped.append(key)
            continue
        compatible[key] = value
    load_result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded_parameter_tensors": len(compatible),
        "skipped_parameter_tensors": len(skipped),
        "r_matrix_head_reinitialized": bool(r_head_incompatible),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }


def _reaction_objective_dataset_kwargs(cfg: dict, split: str) -> dict:
    objective_cfg = get_reaction_objective_config(cfg)
    kwargs = {"split": split}
    if not objective_cfg:
        # The legacy loader sampled validation/test corruption stochastically.
        # Keep that behavior even though split is now passed explicitly.
        kwargs["deterministic_eval"] = False
        return kwargs

    mapping = {
        "mode": "objective",
        "corruption": "corruption_strategy",
        "train_pair_mask": "train_pair_mask_strategy",
        "eval_pair_mask": "eval_pair_mask_strategy",
        "pair_prior": "pair_prior_mode",
        "augmentation_seed": "augmentation_seed",
        "deterministic_eval": "deterministic_eval",
        "local_canvas": "local_canvas",
    }
    for config_key, dataset_key in mapping.items():
        if config_key in objective_cfg:
            kwargs[dataset_key] = objective_cfg[config_key]
    if split.lower() in {"valid", "val", "validation", "test", "eval"} and "eval_corruption" in objective_cfg:
        kwargs["corruption_strategy"] = objective_cfg["eval_corruption"]
    if "augmentation_seed" not in kwargs:
        kwargs["augmentation_seed"] = int(cfg.get("seed", 42))
    return kwargs


def _build_pretrain_dataset(cfg: dict, split_csv: str, molecule_db: str, split: str) -> ReactionDataset:
    be_cfg = cfg.get("be_matrix", {})
    audit_cfg = be_cfg.get("audit", {})
    condition_cfg = cfg.get("condition", {})
    context_cfg = cfg.get("context", {}) or {}
    context_db = context_cfg.get("molecule_db")
    if context_db and not os.path.isabs(str(context_db)):
        context_db = os.path.join(str(cfg.get("data_root", "")), str(context_db))
    return ReactionDataset(
        split_csv=split_csv,
        molecule_db=molecule_db,
        mode="pretrain",
        task="pretrain",
        mask_prob=cfg.get("mask_prob", 0.15),
        leave_unmasked_prob=cfg.get("leave_unmasked_prob", 0.1),
        random_token_prob=cfg.get("random_token_prob", 0.1),
        neighbor_mask_prob=cfg.get("neighbor_mask_prob", 0.3),
        aromatic_mode=be_cfg.get("aromatic_mode", cfg.get("aromatic_mode", "aromatic_1p5")),
        be_audit_tolerance=audit_cfg.get("tolerance", 1e-4),
        condition_enabled=condition_cfg.get("enabled", False),
        condition_column=condition_cfg.get("column", "condition_vector"),
        condition_dim=condition_cfg.get("dim", 514),
        unmapped_components_column=condition_cfg.get("unmapped_components_column", "unmapped_components"),
        context_enabled=context_cfg.get("enabled", False),
        context_column=context_cfg.get("column", "unmapped_components"),
        context_molecule_db=context_db,
        context_max_components=context_cfg.get("max_components", 0),
        context_missing_policy=context_cfg.get("missing_policy", "error"),
        context_mask_prob=context_cfg.get("mask_prob", cfg.get("mask_prob", 0.15)),
        load_all_columns=False,
        **_reaction_objective_dataset_kwargs(cfg, split),
    )


def _split_train_valid(train_dataset: ReactionDataset, cfg: dict):
    valid_fraction = _validation_fraction(cfg)
    total = len(train_dataset)
    if total < 2:
        raise ValueError("At least 2 training samples are required to split a validation subset.")

    valid_size = int(round(total * valid_fraction))
    valid_size = min(max(1, valid_size), total - 1)
    train_size = total - valid_size
    generator = torch.Generator().manual_seed(_split_seed(cfg))
    logger.info(
        "Split validation from training source | validation_mode={} | train={} | "
        "valid={} | valid_fraction={:.4f}",
        get_validation_config(cfg)["mode"],
        train_size,
        valid_size,
        valid_fraction,
    )
    return random_split(train_dataset, [train_size, valid_size], generator=generator)


def _split_train_valid_views(train_dataset: ReactionDataset, valid_dataset: ReactionDataset, cfg: dict):
    """Split identical rows into independent train and deterministic validation views."""

    if len(train_dataset) != len(valid_dataset):
        raise ValueError("Train and validation views must contain the same source rows.")
    valid_fraction = _validation_fraction(cfg)
    total = len(train_dataset)
    if total < 2:
        raise ValueError("At least 2 training samples are required to split a validation subset.")

    valid_size = min(max(1, int(round(total * valid_fraction))), total - 1)
    train_size = total - valid_size
    generator = torch.Generator().manual_seed(_split_seed(cfg))
    index_splits = random_split(range(total), [train_size, valid_size], generator=generator)
    train_indices = list(index_splits[0].indices)
    valid_indices = list(index_splits[1].indices)
    logger.info(
        "Split independent train/valid views from training source | "
        "validation_mode={} | train={} | valid={} | valid_fraction={:.4f}",
        get_validation_config(cfg)["mode"],
        train_size,
        valid_size,
        valid_fraction,
    )
    return Subset(train_dataset, train_indices), Subset(valid_dataset, valid_indices)


def _reaction_pair_keys(
    dataset: ReactionDataset,
    side_cache: dict[str, bytes] | None = None,
) -> list[tuple[bytes, bytes]]:
    frame = dataset.df

    def key(value: object) -> bytes:
        raw = clean_text(value)
        if side_cache is not None and raw in side_cache:
            return side_cache[raw]
        digest = hashlib.sha256(_canonical_side_identity(raw).encode("utf-8")).digest()
        if side_cache is not None:
            side_cache[raw] = digest
        return digest

    return [
        (key(reactant), key(product))
        for reactant, product in zip(frame["reactant_smiles"], frame["prod_smiles"])
    ]


def _directional_input_side(cfg: dict) -> str | None:
    mode = str(get_reaction_objective_config(cfg).get("mode", "reaction_reconstruction")).lower()
    if mode == "product_to_reactant":
        return "product"
    if mode == "reactant_to_product":
        return "reactant"
    if mode in {"bidirectional", "bidirectional_single_side"}:
        return "both"
    return None


def _input_keys(reaction_keys: list[tuple[bytes, bytes]], input_side: str) -> list[bytes]:
    if input_side not in {"reactant", "product"}:
        raise ValueError(f"_input_keys requires one side, got {input_side!r}.")
    position = 1 if input_side == "product" else 0
    return [key[position] for key in reaction_keys]


def _bidirectional_component_group_keys(
    *reaction_key_sets: list[tuple[bytes, bytes]] | None,
) -> list[list[bytes] | None]:
    """Group reactions by connected components over either possible input side.

    If one canonical side occurs anywhere else as a reactant or product, all
    connected reactions receive one group key.  This is the strict split
    contract required when both directions are exposed during pretraining.
    """

    parent: dict[bytes, bytes] = {}

    def find(key: bytes) -> bytes:
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(left: bytes, right: bytes) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for keys in reaction_key_sets:
        if keys is None:
            continue
        for reactant_key, product_key in keys:
            union(reactant_key, product_key)

    outputs: list[list[bytes] | None] = []
    for keys in reaction_key_sets:
        if keys is None:
            outputs.append(None)
            continue
        outputs.append([find(reactant_key) for reactant_key, _ in keys])
    return outputs


def _input_isolation_method(cfg: dict, integrity_cfg: dict | None = None) -> str:
    integrity_cfg = integrity_cfg or (cfg.get("split_integrity", {}) or {})
    effective_side = _effective_input_group_side(cfg, integrity_cfg)
    if _input_group_scope(integrity_cfg) == "both_sides_connected":
        return "both_side_component_isolation_v1"
    return (
        "bidirectional_input_component_isolation_v1"
        if effective_side == "both"
        else "directional_input_group_isolation_v1"
    )


INPUT_GROUP_IDENTITIES = {"full_side", "main_product"}
INPUT_GROUP_SPLIT_ALGORITHMS = {"torch_randperm_v1", "sha256_ordered_prefix_v1"}


def _input_group_identity(integrity_cfg: dict) -> str:
    value = str(integrity_cfg.get("input_group_identity", "full_side")).strip().lower()
    if value not in INPUT_GROUP_IDENTITIES:
        raise ValueError(
            "split_integrity.input_group_identity must be one of "
            f"{sorted(INPUT_GROUP_IDENTITIES)}, got {value!r}."
        )
    return value


def _input_group_split_algorithm(integrity_cfg: dict) -> str:
    value = str(
        integrity_cfg.get("group_split_algorithm", "torch_randperm_v1")
    ).strip().lower()
    if value not in INPUT_GROUP_SPLIT_ALGORITHMS:
        raise ValueError(
            "split_integrity.group_split_algorithm must be one of "
            f"{sorted(INPUT_GROUP_SPLIT_ALGORITHMS)}, got {value!r}."
        )
    return value


def _directional_input_group_keys(
    dataset: ReactionDataset,
    reaction_keys: list[tuple[bytes, bytes]],
    *,
    input_side: str,
    identity_mode: str,
) -> list[bytes]:
    if identity_mode == "full_side":
        return _input_keys(reaction_keys, input_side)
    if identity_mode != "main_product" or input_side != "product":
        raise ValueError(
            "split_integrity.input_group_identity='main_product' is supported only "
            "for a product_to_reactant directional objective."
        )
    return [
        hashlib.sha256(
            _canonical_main_component_identity(value).encode("utf-8")
        ).digest()
        for value in dataset.df["prod_smiles"]
    ]


def _ordered_unique_indices(
    keys: list[tuple[bytes, bytes]],
    forbidden: set[tuple[bytes, bytes]] | None = None,
    deduplicate: bool = True,
) -> tuple[list[int], int, int]:
    forbidden = forbidden or set()
    seen: set[tuple[bytes, bytes]] = set()
    indices = []
    duplicate_count = 0
    overlap_count = 0
    for idx, key in enumerate(keys):
        if key in forbidden:
            overlap_count += 1
            continue
        if deduplicate and key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        indices.append(idx)
    return indices, duplicate_count, overlap_count


def _reaction_key_hash(keys: list[tuple[bytes, bytes]], indices: list[int]) -> str:
    digest = hashlib.sha256()
    for idx in indices:
        reactant, product = keys[idx]
        digest.update(reactant)
        digest.update(b"\x1e")
        digest.update(product)
        digest.update(b"\n")
    return digest.hexdigest()


def _input_key_hash(keys: list[bytes], indices: list[int]) -> str:
    digest = hashlib.sha256()
    for idx in indices:
        digest.update(keys[idx])
        digest.update(b"\n")
    return digest.hexdigest()


def _input_overlap_counts(
    train_keys: set[bytes],
    valid_keys: set[bytes],
    test_keys: set[bytes],
) -> dict[str, int]:
    return {
        "train_valid": len(train_keys & valid_keys),
        "train_test": len(train_keys & test_keys),
        "valid_test": len(valid_keys & test_keys),
    }


def _integrity_subset(
    dataset: ReactionDataset,
    keys: list[tuple[bytes, bytes]],
    forbidden: set[tuple[bytes, bytes]] | None,
    deduplicate: bool,
    input_keys: list[bytes] | None = None,
) -> tuple[Subset, list[int], dict]:
    indices, duplicates, overlaps = _ordered_unique_indices(
        keys,
        forbidden=forbidden,
        deduplicate=deduplicate,
    )
    kept_keys = {keys[idx] for idx in indices}
    summary = {
        "source_rows": len(keys),
        "kept_rows": len(indices),
        "removed_duplicate_rows": duplicates,
        "removed_overlap_rows": overlaps,
        "unique_reaction_pairs": len(kept_keys),
        "ordered_reaction_sha256": _reaction_key_hash(keys, indices),
    }
    if input_keys is not None:
        summary.update(
            {
                "unique_input_groups": len({input_keys[idx] for idx in indices}),
                "ordered_input_sha256": _input_key_hash(input_keys, indices),
            }
        )
    return Subset(dataset, indices), indices, summary


def _filter_by_input_groups(
    reaction_keys: list[tuple[bytes, bytes]],
    input_keys: list[bytes],
    *,
    forbidden_inputs: set[bytes],
    deduplicate: bool,
) -> tuple[list[int], dict]:
    seen_reactions: set[tuple[bytes, bytes]] = set()
    indices = []
    duplicates = 0
    input_overlaps = 0
    for idx, (reaction_key, input_key) in enumerate(zip(reaction_keys, input_keys)):
        if input_key in forbidden_inputs:
            input_overlaps += 1
            continue
        if deduplicate and reaction_key in seen_reactions:
            duplicates += 1
            continue
        seen_reactions.add(reaction_key)
        indices.append(idx)
    return indices, {
        "source_rows": len(reaction_keys),
        "kept_rows": len(indices),
        "removed_duplicate_rows": duplicates,
        "removed_input_overlap_rows": input_overlaps,
        "unique_reaction_pairs": len({reaction_keys[idx] for idx in indices}),
        "unique_input_groups": len({input_keys[idx] for idx in indices}),
        "ordered_reaction_sha256": _reaction_key_hash(reaction_keys, indices),
        "ordered_input_sha256": _input_key_hash(input_keys, indices),
    }


def _deterministic_group_split(
    indices: list[int],
    input_keys: list[bytes],
    cfg: dict,
) -> tuple[list[int], list[int]]:
    """Split source-order indices without placing one input group on both sides."""

    valid_fraction = _validation_fraction(cfg)
    groups: dict[bytes, list[int]] = {}
    for idx in indices:
        groups.setdefault(input_keys[idx], []).append(idx)
    if len(groups) < 2:
        raise ValueError(
            "At least 2 distinct directional input groups are required to split validation."
        )

    integrity_cfg = cfg.get("split_integrity", {}) or {}
    algorithm = _input_group_split_algorithm(integrity_cfg)
    if algorithm == "sha256_ordered_prefix_v1":
        seed_prefix = str(_split_seed(cfg)).encode("ascii") + b"\x1f"
        shuffled_groups = sorted(
            groups,
            key=lambda key: (hashlib.sha256(seed_prefix + key).digest(), key),
        )
    else:
        ordered_groups = list(groups)
        generator = torch.Generator().manual_seed(_split_seed(cfg))
        permutation = torch.randperm(len(ordered_groups), generator=generator).tolist()
        shuffled_groups = [ordered_groups[position] for position in permutation]
    target_rows = len(indices) * valid_fraction
    cumulative = 0
    cut_candidates = []
    for cut, group_key in enumerate(shuffled_groups[:-1], start=1):
        cumulative += len(groups[group_key])
        cut_candidates.append((abs(cumulative - target_rows), cut))
    valid_cut = min(cut_candidates)[1]
    valid_groups = set(shuffled_groups[:valid_cut])
    train_indices = [idx for idx in indices if input_keys[idx] not in valid_groups]
    valid_indices = [idx for idx in indices if input_keys[idx] in valid_groups]
    logger.info(
        "Split validation from training source by canonical {} input groups | "
        "validation_mode={} | "
        "train_rows={} | valid_rows={} | train_groups={} | valid_groups={} | valid_fraction={:.4f}",
        _effective_input_group_side(cfg, integrity_cfg),
        get_validation_config(cfg)["mode"],
        len(train_indices),
        len(valid_indices),
        len(groups) - len(valid_groups),
        len(valid_groups),
        valid_fraction,
    )
    return train_indices, valid_indices


def _subset_summary(
    reaction_keys: list[tuple[bytes, bytes]],
    input_keys: list[bytes],
    indices: list[int],
    **extra,
) -> dict:
    summary = {
        "source_rows": len(reaction_keys),
        "kept_rows": len(indices),
        "unique_reaction_pairs": len({reaction_keys[idx] for idx in indices}),
        "unique_input_groups": len({input_keys[idx] for idx in indices}),
        "ordered_reaction_sha256": _reaction_key_hash(reaction_keys, indices),
        "ordered_input_sha256": _input_key_hash(input_keys, indices),
    }
    summary.update(extra)
    return summary


SOURCE_INDEX_CONTRACT_VERSION = 1


def _split_source_signature(path: str | None) -> dict:
    if not path:
        return {"path": None, "exists": False, "size": None, "mtime_ns": None}
    absolute = os.path.abspath(path)
    try:
        stat = os.stat(absolute)
    except FileNotFoundError:
        return {"path": absolute, "exists": False, "size": None, "mtime_ns": None}
    return {
        "path": absolute,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _split_manifest_location(
    cfg: dict,
    integrity_cfg: dict,
    *,
    train_path: str,
    valid_path: str | None,
    test_path: str,
) -> tuple[str | None, str | None]:
    cache_dir = integrity_cfg.get("cache_dir")
    if not cache_dir:
        return None, None
    semantic_integrity = {
        key: value
        for key, value in integrity_cfg.items()
        if key not in {"cache_dir"}
    }
    objective_mode = str(
        get_reaction_objective_config(cfg).get("mode", "reaction_reconstruction")
    ).lower()
    validation_cfg = get_validation_config(cfg)
    group_scope = _input_group_scope(integrity_cfg)
    payload = {
        "contract_version": SOURCE_INDEX_CONTRACT_VERSION,
        "sources": {
            "train": _split_source_signature(train_path),
            "valid": _split_source_signature(valid_path),
            "test": _split_source_signature(test_path),
        },
        "reaction_identity_method": REACTION_IDENTITY_METHOD,
        "objective_mode": objective_mode,
        "input_side": _effective_input_group_side(cfg, integrity_cfg),
        "objective_input_side": _directional_input_side(cfg),
        "group_scope": group_scope,
        "seed": _split_seed(cfg),
        "validation": validation_cfg,
        "split_integrity": semantic_integrity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return os.path.join(os.path.abspath(str(cache_dir)), f"split_manifest_{cache_key}.json"), cache_key


def _validate_source_index_contract(
    contract: dict,
    *,
    train_base: ReactionDataset,
    valid_base: ReactionDataset | None,
    test_base: ReactionDataset,
    cfg: dict,
) -> None:
    if not isinstance(contract, dict) or contract.get("version") != SOURCE_INDEX_CONTRACT_VERSION:
        raise ValueError("Invalid split source-index contract version.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Split source-index contract is missing runtime metadata.")
    integrity_cfg = cfg.get("split_integrity", {}) or {}
    expected_input_side = _effective_input_group_side(cfg, integrity_cfg)
    if runtime.get("method") != _input_isolation_method(cfg, integrity_cfg):
        raise ValueError("Split source-index contract has the wrong integrity method.")
    if runtime.get("input_side") != expected_input_side:
        raise ValueError("Split source-index contract has the wrong directional input side.")
    if runtime.get("group_scope", "objective_input") != _input_group_scope(integrity_cfg):
        raise ValueError("Split source-index contract has the wrong input-group scope.")

    expected_rows = {
        "train": len(train_base),
        "valid": len(valid_base) if valid_base is not None else None,
        "test": len(test_base),
    }
    if contract.get("source_rows") != expected_rows:
        raise ValueError(
            "Split source-index contract source row counts changed: "
            f"contract={contract.get('source_rows')!r}, current={expected_rows!r}."
        )
    split_specs = contract.get("splits")
    if not isinstance(split_specs, dict) or set(split_specs) != {"train", "valid", "test"}:
        raise ValueError("Split source-index contract must define train/valid/test indices.")
    expected_sources = {
        "train": "train",
        "valid": "valid" if valid_base is not None else "train",
        "test": "test",
    }
    source_lengths = {
        "train": len(train_base),
        "valid": len(valid_base) if valid_base is not None else 0,
        "test": len(test_base),
    }
    for split, expected_source in expected_sources.items():
        spec = split_specs[split]
        if not isinstance(spec, dict) or spec.get("source") != expected_source:
            raise ValueError(
                f"Split source-index contract uses the wrong source for {split}: {spec!r}."
            )
        indices = spec.get("indices")
        if not isinstance(indices, list):
            raise ValueError(f"Split source-index contract indices for {split} must be a list.")
        source_length = source_lengths[expected_source]
        for idx in indices:
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < source_length:
                raise ValueError(
                    f"Split source-index contract has an out-of-range {split} index: {idx!r}."
                )
        runtime_rows = runtime.get(split, {}).get("kept_rows")
        if runtime_rows != len(indices):
            raise ValueError(
                f"Split source-index contract row-count mismatch for {split}: "
                f"runtime={runtime_rows!r}, indices={len(indices)}."
            )


def _datasets_from_source_index_contract(
    contract: dict,
    *,
    train_base: ReactionDataset,
    valid_base: ReactionDataset | None,
    test_base: ReactionDataset,
    cfg: dict,
) -> tuple[ReactionDataset, ReactionDataset, ReactionDataset]:
    _validate_source_index_contract(
        contract,
        train_base=train_base,
        valid_base=valid_base,
        test_base=test_base,
        cfg=cfg,
    )
    roots = {"train": train_base, "test": test_base}
    if valid_base is not None:
        roots["valid"] = valid_base
    else:
        valid_view = train_base.split_view("valid")
        valid_view.corruption_strategy = _reaction_objective_dataset_kwargs(cfg, "valid").get(
            "corruption_strategy", valid_view.corruption_strategy
        )
        roots["valid"] = valid_view
    datasets = []
    for split in ("train", "valid", "test"):
        spec = contract["splits"][split]
        source_name = spec["source"]
        root_name = "valid" if split == "valid" else source_name
        datasets.append(Subset(roots[root_name], list(spec["indices"])))
    cfg["split_integrity_runtime"] = copy.deepcopy(contract["runtime"])
    return tuple(datasets)


def _make_source_index_contract(
    *,
    train_base: ReactionDataset,
    valid_base: ReactionDataset | None,
    test_base: ReactionDataset,
    train_indices: list[int],
    valid_indices: list[int],
    test_indices: list[int],
    runtime: dict,
) -> dict:
    return {
        "version": SOURCE_INDEX_CONTRACT_VERSION,
        "source_rows": {
            "train": len(train_base),
            "valid": len(valid_base) if valid_base is not None else None,
            "test": len(test_base),
        },
        "splits": {
            "train": {"source": "train", "indices": list(train_indices)},
            "valid": {
                "source": "valid" if valid_base is not None else "train",
                "indices": list(valid_indices),
            },
            "test": {"source": "test", "indices": list(test_indices)},
        },
        "runtime": copy.deepcopy(runtime),
    }


def _load_split_manifest(path: str, cache_key: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("cache_key") != cache_key:
            raise ValueError("cache key mismatch")
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            raise ValueError("missing contract")
        return contract
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignore invalid split manifest | path={} | error={}", path, exc)
        return None


def _write_split_manifest(path: str, cache_key: str, contract: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                {"cache_key": cache_key, "contract": contract},
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _return_datasets(
    datasets: tuple,
    contract: dict | None,
    return_source_index_contract: bool,
):
    if return_source_index_contract:
        return (*datasets, contract)
    return datasets


def build_pretrain_datasets(
    cfg: dict,
    *,
    source_index_contract: dict | None = None,
    return_source_index_contract: bool = False,
):
    """Build train/valid/test datasets with optional exact-reaction isolation."""

    data_root = cfg["data_root"]
    db_name = cfg.get("molecule_db", "smiles.lmdb")
    molecule_db = db_name if os.path.isabs(db_name) else os.path.join(data_root, db_name)
    train_csv = resolve_split_file(cfg, data_root, "train")
    valid_csv = resolve_split_file(cfg, data_root, "valid")
    test_csv = resolve_split_file(cfg, data_root, "test")

    validation_cfg = get_validation_config(cfg)
    validation_mode = validation_cfg["mode"]
    resolved_valid_exists = os.path.isfile(valid_csv)
    if validation_mode == "explicit" and not resolved_valid_exists:
        raise FileNotFoundError(
            "validation.mode='explicit' requires an existing validation file: "
            f"{valid_csv}"
        )
    valid_exists = (
        resolved_valid_exists
        if validation_mode in {"auto", "explicit"}
        else False
    )
    resolved_validation_mode = "explicit" if valid_exists else "split_from_train"

    train_base = _build_pretrain_dataset(cfg, train_csv, molecule_db, split="train")
    valid_base = _build_pretrain_dataset(cfg, valid_csv, molecule_db, split="valid") if valid_exists else None
    test_base = _build_pretrain_dataset(cfg, test_csv, molecule_db, split="test")

    integrity_cfg = cfg.get("split_integrity", {}) or {}
    integrity_enabled = bool(integrity_cfg.get("enabled", False))
    if not integrity_enabled:
        if valid_base is not None:
            return _return_datasets(
                (train_base, valid_base, test_base),
                None,
                return_source_index_contract,
            )
        if uses_directional_protocol(cfg):
            valid_view = train_base.split_view("valid")
            valid_view.corruption_strategy = _reaction_objective_dataset_kwargs(cfg, "valid").get(
                "corruption_strategy", valid_view.corruption_strategy
            )
            train_dataset, valid_dataset = _split_train_valid_views(train_base, valid_view, cfg)
        else:
            train_dataset, valid_dataset = _split_train_valid(train_base, cfg)
        return _return_datasets(
            (train_dataset, valid_dataset, test_base),
            None,
            return_source_index_contract,
        )

    deduplicate = bool(integrity_cfg.get("deduplicate_within_splits", True))
    exclude_overlap = bool(integrity_cfg.get("exclude_cross_split_overlap", True))
    group_by_input_side = bool(integrity_cfg.get("group_by_input_side", False))
    objective_input_side = _directional_input_side(cfg)
    group_scope = _input_group_scope(integrity_cfg)
    input_side = _effective_input_group_side(cfg, integrity_cfg)
    input_group_identity = _input_group_identity(integrity_cfg)
    group_split_algorithm = _input_group_split_algorithm(integrity_cfg)
    if group_by_input_side and objective_input_side is None:
        raise ValueError(
            "split_integrity.group_by_input_side requires a directional "
            "product_to_reactant, reactant_to_product, or bidirectional_single_side objective."
        )
    if group_by_input_side and not exclude_overlap:
        raise ValueError(
            "split_integrity.group_by_input_side=true requires "
            "exclude_cross_split_overlap=true."
        )
    if group_by_input_side and input_group_identity == "main_product" and input_side != "product":
        raise ValueError(
            "split_integrity.input_group_identity='main_product' requires a "
            "product_to_reactant objective."
        )

    manifest_path, manifest_key = _split_manifest_location(
        cfg,
        integrity_cfg,
        train_path=train_csv,
        valid_path=valid_csv if valid_exists else None,
        test_path=test_csv,
    )
    if source_index_contract is not None:
        datasets = _datasets_from_source_index_contract(
            source_index_contract,
            train_base=train_base,
            valid_base=valid_base,
            test_base=test_base,
            cfg=cfg,
        )
        return _return_datasets(
            datasets,
            source_index_contract,
            return_source_index_contract,
        )
    if group_by_input_side and manifest_path is not None:
        cached_contract = _load_split_manifest(manifest_path, manifest_key)
        if cached_contract is not None:
            try:
                datasets = _datasets_from_source_index_contract(
                    cached_contract,
                    train_base=train_base,
                    valid_base=valid_base,
                    test_base=test_base,
                    cfg=cfg,
                )
            except ValueError as exc:
                logger.warning(
                    "Ignore stale split manifest | path={} | error={}",
                    manifest_path,
                    exc,
                )
            else:
                logger.info("Loaded split manifest: {}", manifest_path)
                return _return_datasets(
                    datasets,
                    cached_contract,
                    return_source_index_contract,
                )

    # Do not retain a raw-SMILES -> canonical-SMILES dictionary here.  ORDerly
    # reaction sides are mostly unique, so such a cache costs hundreds of MB.
    # Fixed-size SHA-256 keys keep identity comparisons memory bounded.
    train_keys = _reaction_pair_keys(train_base)
    valid_keys = _reaction_pair_keys(valid_base) if valid_base is not None else None
    test_keys = _reaction_pair_keys(test_base)
    if input_side == "both":
        if input_group_identity != "full_side":
            raise ValueError(
                "Bidirectional input isolation requires split_integrity.input_group_identity='full_side'."
            )
        train_input_keys, valid_input_keys, test_input_keys = _bidirectional_component_group_keys(
            train_keys,
            valid_keys,
            test_keys,
        )
    else:
        train_input_keys = (
            _directional_input_group_keys(
                train_base,
                train_keys,
                input_side=input_side,
                identity_mode=input_group_identity,
            )
            if input_side is not None
            else None
        )
        valid_input_keys = (
            _directional_input_group_keys(
                valid_base,
                valid_keys,
                input_side=input_side,
                identity_mode=input_group_identity,
            )
            if input_side is not None and valid_keys is not None
            else None
        )
        test_input_keys = (
            _directional_input_group_keys(
                test_base,
                test_keys,
                input_side=input_side,
                identity_mode=input_group_identity,
            )
            if input_side is not None
            else None
        )

    if group_by_input_side:
        # The official test set has precedence.  It is never discarded merely
        # because a training-source row has the same directional input.
        test_dataset, test_indices, test_summary = _integrity_subset(
            test_base,
            test_keys,
            forbidden=None,
            # Preserve every official test row.  Deduplication is appropriate
            # for the training corpus, but changing test multiplicities would
            # silently change the benchmark distribution.
            deduplicate=False,
            input_keys=test_input_keys,
        )
        test_summary["removed_input_overlap_rows"] = 0
        test_input_set = {test_input_keys[idx] for idx in test_indices}

        if valid_base is not None:
            valid_indices, valid_summary = _filter_by_input_groups(
                valid_keys,
                valid_input_keys,
                forbidden_inputs=test_input_set,
                deduplicate=deduplicate,
            )
            valid_summary["removed_overlap_rows"] = valid_summary[
                "removed_input_overlap_rows"
            ]
            valid_dataset = Subset(valid_base, valid_indices)
            valid_input_set = {valid_input_keys[idx] for idx in valid_indices}
            train_indices, train_summary = _filter_by_input_groups(
                train_keys,
                train_input_keys,
                forbidden_inputs=test_input_set | valid_input_set,
                deduplicate=deduplicate,
            )
            train_summary["removed_overlap_rows"] = train_summary[
                "removed_input_overlap_rows"
            ]
            train_dataset = Subset(train_base, train_indices)
            train_source_filter = dict(train_summary)
        else:
            filtered_indices, train_source_filter = _filter_by_input_groups(
                train_keys,
                train_input_keys,
                forbidden_inputs=test_input_set,
                deduplicate=deduplicate,
            )
            train_indices, valid_indices = _deterministic_group_split(
                filtered_indices,
                train_input_keys,
                cfg,
            )
            valid_view = train_base.split_view("valid")
            valid_view.corruption_strategy = _reaction_objective_dataset_kwargs(cfg, "valid").get(
                "corruption_strategy", valid_view.corruption_strategy
            )
            train_dataset = Subset(train_base, train_indices)
            valid_dataset = Subset(valid_view, valid_indices)
            split_extra = {
                "source": (
                    "deterministic canonical directional-input group split after "
                    "official-test exclusion and exact-reaction deduplication"
                ),
                "removed_duplicate_rows_before_split": train_source_filter[
                    "removed_duplicate_rows"
                ],
                "removed_input_overlap_rows_before_split": train_source_filter[
                    "removed_input_overlap_rows"
                ],
                "removed_overlap_rows": 0,
            }
            train_summary = _subset_summary(
                train_keys,
                train_input_keys,
                train_indices,
                **split_extra,
            )
            valid_summary = _subset_summary(
                train_keys,
                train_input_keys,
                valid_indices,
                **split_extra,
            )

        final_train_key_set = {train_keys[idx] for idx in train_indices}
        if valid_base is not None:
            final_valid_key_set = {valid_keys[idx] for idx in valid_indices}
            final_valid_input_set = {valid_input_keys[idx] for idx in valid_indices}
        else:
            final_valid_key_set = {train_keys[idx] for idx in valid_indices}
            final_valid_input_set = {train_input_keys[idx] for idx in valid_indices}
        final_test_key_set = {test_keys[idx] for idx in test_indices}
        final_train_input_set = {train_input_keys[idx] for idx in train_indices}
        final_test_input_set = {test_input_keys[idx] for idx in test_indices}
        reaction_overlap_counts = {
            "train_valid": len(final_train_key_set & final_valid_key_set),
            "train_test": len(final_train_key_set & final_test_key_set),
            "valid_test": len(final_valid_key_set & final_test_key_set),
        }
        input_overlap_counts = _input_overlap_counts(
            final_train_input_set,
            final_valid_input_set,
            final_test_input_set,
        )
        if any(reaction_overlap_counts.values()) or any(input_overlap_counts.values()):
            raise RuntimeError(
                "Directional input-group isolation failed: "
                f"reaction={reaction_overlap_counts}, input={input_overlap_counts}"
            )
        runtime_summary = {
            "method": _input_isolation_method(cfg, integrity_cfg),
            "validation": {
                "configured_mode": validation_mode,
                "resolved_mode": resolved_validation_mode,
                "fraction": validation_cfg["fraction"],
            },
            "reaction_identity_method": REACTION_IDENTITY_METHOD,
            "invalid_smiles_fallback": REACTION_IDENTITY_FALLBACK,
            "component_order_normalized": True,
            "atom_map_invariant": True,
            "group_by_input_side": True,
            "input_side": input_side,
            "objective_input_side": objective_input_side,
            "group_scope": group_scope,
            "split_seed": _split_seed(cfg),
            "input_group_identity": input_group_identity,
            "group_split_algorithm": group_split_algorithm,
            "official_test_precedence": True,
            "official_test_rows_preserved": True,
            "deduplicate_within_splits": deduplicate,
            "exclude_cross_split_overlap": exclude_overlap,
            "train_source_filter": train_source_filter,
            "train": train_summary,
            "valid": valid_summary,
            "test": test_summary,
            "post_filter_exact_overlap": reaction_overlap_counts,
            "post_filter_input_overlap": input_overlap_counts,
        }
        cfg["split_integrity_runtime"] = runtime_summary
        contract = _make_source_index_contract(
            train_base=train_base,
            valid_base=valid_base,
            test_base=test_base,
            train_indices=train_indices,
            valid_indices=valid_indices,
            test_indices=test_indices,
            runtime=runtime_summary,
        )
        if manifest_path is not None:
            _write_split_manifest(manifest_path, manifest_key, contract)
            logger.info("Wrote split manifest: {}", manifest_path)
        logger.info("Split integrity: {}", runtime_summary)
        return _return_datasets(
            (train_dataset, valid_dataset, test_dataset),
            contract,
            return_source_index_contract,
        )

    train_subset, train_indices, train_summary = _integrity_subset(
        train_base,
        train_keys,
        forbidden=None,
        deduplicate=deduplicate,
        input_keys=train_input_keys,
    )
    train_key_set = {train_keys[idx] for idx in train_indices}

    if valid_base is not None:
        valid_subset, valid_indices, valid_summary = _integrity_subset(
            valid_base,
            valid_keys,
            forbidden=train_key_set if exclude_overlap else None,
            deduplicate=deduplicate,
            input_keys=valid_input_keys,
        )
        valid_key_set = {valid_keys[idx] for idx in valid_indices}
        train_dataset, valid_dataset = train_subset, valid_subset
        final_train_key_set = train_key_set
        final_valid_key_set = valid_key_set
    else:
        valid_view = train_base.split_view("valid")
        valid_view.corruption_strategy = _reaction_objective_dataset_kwargs(cfg, "valid").get(
            "corruption_strategy", valid_view.corruption_strategy
        )
        filtered_valid_source = Subset(valid_view, train_indices)
        train_dataset, valid_dataset = _split_train_valid_views(train_subset, filtered_valid_source, cfg)
        train_original_indices = [train_indices[int(position)] for position in train_dataset.indices]
        valid_original_indices = [train_indices[int(position)] for position in valid_dataset.indices]
        corpus_summary = train_summary
        train_summary = {
            "source_rows": len(train_keys),
            "kept_rows": len(train_original_indices),
            "removed_duplicate_rows_before_split": corpus_summary["removed_duplicate_rows"],
            "removed_overlap_rows": 0,
            "unique_reaction_pairs": len(train_original_indices),
            "ordered_reaction_sha256": _reaction_key_hash(train_keys, train_original_indices),
        }
        if train_input_keys is not None:
            train_summary.update(
                {
                    "unique_input_groups": len(
                        {train_input_keys[idx] for idx in train_original_indices}
                    ),
                    "ordered_input_sha256": _input_key_hash(
                        train_input_keys, train_original_indices
                    ),
                }
            )
        valid_summary = {
            "source": "group-isolated random split after exact-reaction deduplication",
            "source_rows": len(train_keys),
            "kept_rows": len(valid_original_indices),
            "removed_duplicate_rows_before_split": corpus_summary["removed_duplicate_rows"],
            "removed_overlap_rows": 0,
            "unique_reaction_pairs": len(valid_original_indices),
            "ordered_reaction_sha256": _reaction_key_hash(train_keys, valid_original_indices),
        }
        if train_input_keys is not None:
            valid_summary.update(
                {
                    "unique_input_groups": len(
                        {train_input_keys[idx] for idx in valid_original_indices}
                    ),
                    "ordered_input_sha256": _input_key_hash(
                        train_input_keys, valid_original_indices
                    ),
                }
            )
        final_train_key_set = {train_keys[idx] for idx in train_original_indices}
        final_valid_key_set = {train_keys[idx] for idx in valid_original_indices}
        valid_key_set = set()

    forbidden_test = train_key_set | valid_key_set if exclude_overlap else None
    test_dataset, test_indices, test_summary = _integrity_subset(
        test_base,
        test_keys,
        forbidden=forbidden_test,
        deduplicate=deduplicate,
        input_keys=test_input_keys,
    )
    final_test_key_set = {test_keys[idx] for idx in test_indices}
    overlap_counts = {
        "train_valid": len(final_train_key_set & final_valid_key_set),
        "train_test": len(final_train_key_set & final_test_key_set),
        "valid_test": len(final_valid_key_set & final_test_key_set),
    }
    if exclude_overlap and any(overlap_counts.values()):
        raise RuntimeError(f"Split-integrity filtering failed: {overlap_counts}")
    if input_side is not None:
        final_train_input_set = {
            train_input_keys[idx]
            for idx in (
                train_original_indices if valid_base is None else train_indices
            )
        }
        final_valid_input_set = {
            (train_input_keys if valid_base is None else valid_input_keys)[idx]
            for idx in (
                valid_original_indices if valid_base is None else valid_indices
            )
        }
        final_test_input_set = {test_input_keys[idx] for idx in test_indices}
        input_overlap_counts = _input_overlap_counts(
            final_train_input_set,
            final_valid_input_set,
            final_test_input_set,
        )
    else:
        input_overlap_counts = None
    runtime_summary = {
        "method": "canonical_exact_reactant_product_pair_v2",
        "validation": {
            "configured_mode": validation_mode,
            "resolved_mode": resolved_validation_mode,
            "fraction": validation_cfg["fraction"],
        },
        "reaction_identity_method": REACTION_IDENTITY_METHOD,
        "invalid_smiles_fallback": REACTION_IDENTITY_FALLBACK,
        "component_order_normalized": True,
        "atom_map_invariant": True,
        "group_by_input_side": False,
        "input_side": input_side,
        "objective_input_side": objective_input_side,
        "group_scope": group_scope,
        "split_seed": _split_seed(cfg),
        "input_group_identity": input_group_identity,
        "group_split_algorithm": group_split_algorithm,
        "deduplicate_within_splits": deduplicate,
        "exclude_cross_split_overlap": exclude_overlap,
        "train": train_summary,
        "valid": valid_summary,
        "test": test_summary,
        "post_filter_exact_overlap": overlap_counts,
        "post_filter_input_overlap": input_overlap_counts,
    }
    cfg["split_integrity_runtime"] = runtime_summary
    logger.info("Split integrity: {}", runtime_summary)
    return _return_datasets(
        (train_dataset, valid_dataset, test_dataset),
        None,
        return_source_index_contract,
    )


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset, valid_dataset, test_dataset = build_pretrain_datasets(cfg)

    collator = ReactionCollator(mode="pretrain", task="pretrain")
    budget_cfg = cfg.get("training_budget", {}) or {}
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
    }
    if common["num_workers"] > 0:
        common["prefetch_factor"] = cfg.get("prefetch_factor", 4)
        common["persistent_workers"] = True
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=bool(budget_cfg.get("require_full_train_batches", False)),
            **common,
        ),
        DataLoader(valid_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )


def build_model(cfg: dict) -> PretrainModel:
    model_cfg = cfg.get("model", {})
    condition_cfg = cfg.get("condition", {})
    context_cfg = cfg.get("context", {}) or {}
    consistency_cfg = cfg.get("consistency", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    active_tasks = get_pretrain_tasks_from_config(cfg)
    return PretrainModel(
        mol_embed_dim=model_cfg.get("mol_embed_dim", 256),
        mol_num_kernel=model_cfg.get("mol_num_kernel", 256),
        mol_num_heads=model_cfg.get("mol_num_heads", 16),
        mol_num_layers=model_cfg.get("mol_num_layers", 6),
        mol_hidden_size=model_cfg.get("mol_hidden_size", 256),
        hg_embed_dim=model_cfg.get("hg_embed_dim", model_cfg.get("mol_embed_dim", 256)),
        hg_num_heads=model_cfg.get("hg_num_heads", 16),
        hg_layers=model_cfg.get("hg_layers", 6),
        dropout=model_cfg.get("dropout", 0.1),
        r_matrix_use_be_pair_feature=model_cfg.get("r_matrix_use_be_pair_feature", False),
        be_rbf_bins=model_cfg.get("be_rbf_bins", 81),
        be_rbf_min=model_cfg.get("be_rbf_min", 0.0),
        be_rbf_max=model_cfg.get("be_rbf_max", 8.0),
        condition_enabled=condition_cfg.get("enabled", False),
        condition_dim=condition_cfg.get("dim", 514),
        condition_dropout_prob=condition_cfg.get("dropout_prob", 0.0),
        context_role_enabled=context_cfg.get("enabled", False),
        consistency_enabled=consistency_cfg.get(
            "enabled",
            float(loss_cfg.get("lambda_cons", 0.0)) > 0.0,
        ),
        consistency_projection_dim=consistency_cfg.get("projection_dim", 128),
        active_tasks=active_tasks,
    )


def build_pretrain_loss(cfg: dict) -> PretrainLoss:
    loss_cfg = dict(cfg.get("loss", {}))
    if float(loss_cfg.get("lambda_ctx", 0.0)) > 0.0 and not bool(
        (cfg.get("context", {}) or {}).get("enabled", False)
    ):
        raise ValueError("loss.lambda_ctx > 0 requires context.enabled=true.")
    if float(loss_cfg.get("lambda_cons", 0.0)) > 0.0:
        if not build_protocol_metadata(cfg).get("dual_view", False):
            raise ValueError("loss.lambda_cons > 0 requires the dual-view protocol.")
        if not bool((cfg.get("consistency", {}) or {}).get("enabled", True)):
            raise ValueError("loss.lambda_cons > 0 requires consistency.enabled=true.")
    loss_cfg["active_tasks"] = get_pretrain_tasks_from_config(cfg)
    return PretrainLoss(**loss_cfg)


def get_early_stop_settings(cfg: dict) -> tuple[int, float]:
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    if patience < 0:
        raise ValueError("early_stop_patience must be >= 0.")
    if min_delta < 0:
        raise ValueError("early_stop_min_delta must be >= 0.")
    return patience, min_delta


def is_val_loss_improved(val_loss: float, best_loss: float, min_delta: float = 0.0) -> bool:
    return float(val_loss) < float(best_loss) - float(min_delta)


def get_selection_metric(cfg: dict) -> str:
    return str(cfg.get("selection_metric", "avg_total_loss"))


def get_selection_mode(cfg: dict) -> str:
    configured = cfg.get("selection_mode")
    if configured is not None:
        mode = str(configured).lower()
    else:
        metric = get_selection_metric(cfg).lower()
        mode = "max" if any(token in metric for token in ("f1", "accuracy", "recall", "precision", "auc")) else "min"
    if mode not in {"min", "max"}:
        raise ValueError("selection_mode must be 'min' or 'max'.")
    return mode


def get_selection_value(cfg: dict, val_loss: float, val_metrics: dict) -> float:
    metric = get_selection_metric(cfg)
    if metric == "avg_total_loss":
        return float(val_loss)
    if metric not in val_metrics:
        raise KeyError(
            f"selection_metric={metric!r} is absent from validation metrics: {sorted(val_metrics)}"
        )
    return float(val_metrics[metric])


def is_selection_improved(value: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "max":
        return float(value) > float(best) + float(min_delta)
    return float(value) < float(best) - float(min_delta)


def maybe_run_be_audit(cfg: dict, save_dir: str) -> None:
    be_cfg = cfg.get("be_matrix", {})
    audit_cfg = be_cfg.get("audit", {})
    if not audit_cfg.get("enabled", False):
        return
    data_root = cfg["data_root"]
    db_name = cfg.get("molecule_db", "smiles.lmdb")
    molecule_db = db_name if os.path.isabs(db_name) else os.path.join(data_root, db_name)
    splits = audit_cfg.get("splits", ["train", "valid", "test"])
    max_samples = int(audit_cfg.get("max_samples", 0) or 0)
    tolerance = float(audit_cfg.get("tolerance", 1e-4))
    result = {}
    for split in splits:
        split_csv = resolve_split_file(cfg, data_root, split)
        if not os.path.exists(split_csv):
            logger.warning("Skip BE matrix audit | split={} | missing_csv={}", split, split_csv)
            continue
        logger.info("Running BE matrix audit | split={} | max_samples={}", split, max_samples or "all")
        result[split] = audit_reaction_dataset_from_paths(
            split_csv=split_csv,
            molecule_db=molecule_db,
            aromatic_mode=be_cfg.get("aromatic_mode", cfg.get("aromatic_mode", "aromatic_1p5")),
            max_samples=max_samples,
            tolerance=tolerance,
        )
        summary = result[split]["summary"]
        logger.info(
            "BE audit {} | samples={} | bad={} | non_conserved={} | map_mismatch={} | negative_diagonal={}",
            split,
            summary.get("num_samples", 0),
            summary.get("num_bad_samples", 0),
            summary.get("num_non_conserved", 0),
            summary.get("num_map_mismatch", 0),
            summary.get("num_negative_diagonal", 0),
        )
    write_json(os.path.join(save_dir, "be_audit.json"), result)


def _batch_graph_view_count(batch: dict) -> int:
    canvas = batch.get("reaction_canvas_mask")
    if torch.is_tensor(canvas):
        return int(canvas.shape[0])
    pair_index = batch.get("view_pair_index")
    if torch.is_tensor(pair_index):
        return int(pair_index.numel())
    raise KeyError("Pretraining batch is missing reaction_canvas_mask.")


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_steps: int = 0,
    desc: str | None = None,
    expected_local_graph_views: int = 0,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    logs = []
    processed_batches = 0
    optimizer_steps = 0
    local_graph_views = 0
    iterator = tqdm(loader, desc=desc or ("Train" if is_train else "Eval"), leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(iterator, start=1):
            if not batch:
                continue
            batch_graph_views = _batch_graph_view_count(batch)
            if is_train and expected_local_graph_views > 0 and batch_graph_views != expected_local_graph_views:
                raise RuntimeError(
                    "Formal ablation budget requires a full, fixed-size graph-view batch: "
                    f"expected={expected_local_graph_views}, observed={batch_graph_views}. "
                    "Audit invalid dataset rows instead of silently changing compute."
                )
            batch = move_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, log = criterion(outputs, batch, is_train=is_train)
            # Private differentiable components are consumed only by the DDP
            # runner and must not retain computation graphs in epoch logs.
            log.pop("_rmat_loss_tensor", None)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                local_graph_views += batch_graph_views
            logs.append(log)
            processed_batches += 1
            iterator.set_postfix({"loss": f"{log.get('total_loss', 0.0):.4f}"})
            if max_steps > 0 and processed_batches >= max_steps:
                break
    if not logs:
        raise RuntimeError(f"{desc or ('Train' if is_train else 'Eval')} produced no valid batches.")
    metrics = reduce_metrics(logs)
    metrics["processed_batches"] = processed_batches
    metrics["optimizer_steps"] = optimizer_steps
    metrics["local_graph_views"] = local_graph_views
    return metrics.get("avg_total_loss", 0.0), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume_path", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Warm-start compatible model weights only; optimizer, epoch, and best-loss state are reset.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override config gpu (use 0 after masking one GPU).")
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()
    if args.resume_path and args.init_checkpoint:
        parser.error("--resume_path and --init_checkpoint are mutually exclusive.")

    cfg = load_yaml_config(args.config).raw
    if args.gpu is not None:
        cfg["gpu"] = int(args.gpu)
    active_tasks = get_pretrain_tasks_from_config(cfg)
    cfg["active_pretrain_tasks"] = list(active_tasks)
    protocol_metadata = build_protocol_metadata(cfg)
    cfg["protocol_metadata"] = protocol_metadata
    directional_protocol = bool(protocol_metadata["directional"])
    training_budget_runtime = resolve_training_budget(cfg, world_size=1)
    cfg["training_budget_runtime"] = training_budget_runtime
    if args.max_steps > 0 and training_budget_runtime["enabled"]:
        parser.error(
            "--max_steps is a one-epoch smoke limit and cannot be combined with "
            "config.training_budget; edit max_optimizer_steps instead."
        )
    ckpt = load_checkpoint_if_available(args.resume_path, map_location="cpu")
    init_ckpt = load_checkpoint_if_available(args.init_checkpoint, map_location="cpu")
    if args.resume_path and ckpt is None:
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_path}")
    if args.init_checkpoint and init_ckpt is None:
        raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
    if ckpt:
        validate_resume_protocol(cfg, ckpt)
        training_budget_runtime = validate_resume_training_budget(
            cfg,
            ckpt,
            world_size=1,
        )
        cfg["training_budget_runtime"] = training_budget_runtime
    if args.resume_path:
        save_dir = os.path.dirname(os.path.abspath(args.resume_path))
        run_name = os.path.basename(save_dir)
    else:
        run_name = cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(cfg["save_dir"], run_name)
    # Build and validate the complete split contract before touching an
    # existing run directory during exact resume.
    train_loader, val_loader, test_loader = build_loaders(cfg)
    if ckpt:
        validate_resume_data_contract(cfg, ckpt)
    log_path = setup_training_logger(save_dir)
    log_config(cfg, title="Pretrain Configuration")
    write_json(os.path.join(save_dir, "config_resolved.json"), cfg)
    logger.info("Run directory: {}", save_dir)
    logger.info("Training log: {}", log_path)
    device = get_device(cfg.get("gpu", -1))
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False), device=device)
    logger.info("Device: {}", device)
    logger.info("Active pretraining tasks: {}", ", ".join(active_tasks))
    logger.info("Pretraining protocol: {}", protocol_metadata)
    logger.info("Training budget: {}", training_budget_runtime)

    maybe_run_be_audit(cfg, save_dir)
    model = build_model(cfg).to(device)
    criterion = build_pretrain_loss(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    early_stop_patience, early_stop_min_delta = get_early_stop_settings(cfg)
    selection_metric = get_selection_metric(cfg)
    selection_mode = get_selection_mode(cfg)
    if early_stop_patience > 0:
        logger.info(
            "Early stopping enabled | monitor={} | mode={} | patience={} | min_delta={:.6g}",
            selection_metric,
            selection_mode,
            early_stop_patience,
            early_stop_min_delta,
        )
    else:
        logger.info("Early stopping disabled | set early_stop_patience > 0 to enable.")

    initialization_metadata = None
    start_epoch = 1
    best_loss = float("-inf") if selection_mode == "max" else float("inf")
    early_stop_bad_epochs = 0
    global_optimizer_step = 0
    cumulative_graph_views = 0
    if ckpt:
        initialization_metadata = ckpt.get("initialization_metadata")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        saved_selection_metric = ckpt.get("best_selection_metric", "avg_total_loss")
        saved_selection_mode = ckpt.get("best_selection_mode", "min")
        if saved_selection_metric != selection_metric or saved_selection_mode != selection_mode:
            raise ValueError(
                f"Cannot resume selector {(selection_metric, selection_mode)!r} from checkpoint selector "
                f"{(saved_selection_metric, saved_selection_mode)!r}; use --init_checkpoint."
            )
        best_loss = ckpt.get("best_selection_value", ckpt.get("best_loss", best_loss))
        early_stop_bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0) or 0)
        global_optimizer_step = int(ckpt.get("global_optimizer_step", 0) or 0)
        cumulative_graph_views = int(ckpt.get("cumulative_graph_views", 0) or 0)
        logger.info(
            "Resumed from {} | start_epoch={} | best_loss={:.4f} | "
            "early_stop_bad_epochs={} | global_optimizer_step={} | graph_views={}",
            args.resume_path,
            start_epoch,
            best_loss,
            early_stop_bad_epochs,
            global_optimizer_step,
            cumulative_graph_views,
        )
    elif init_ckpt:
        init_state = init_ckpt.get("model_state_dict", init_ckpt)
        load_report = load_compatible_initial_weights(model, init_state)
        initialization_metadata = {
            "mode": "warm_start",
            "source": os.path.abspath(args.init_checkpoint),
            "source_protocol": checkpoint_protocol_metadata(init_ckpt),
            "load_report": load_report,
        }
        logger.info("Warm-started model weights: {}", initialization_metadata)

    writer = build_summary_writer(os.path.join(save_dir, "tb"))
    last_test_metrics = {}
    last_test_loss = 0.0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_step_limit = int(args.max_steps)
        if training_budget_runtime["enabled"]:
            remaining_steps = (
                training_budget_runtime["max_optimizer_steps"] - global_optimizer_step
            )
            if remaining_steps <= 0:
                break
            train_step_limit = remaining_steps
        logger.info("[Epoch {}] current lrs: {}", epoch, format_lrs(optimizer))
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            max_steps=train_step_limit,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Train]",
            expected_local_graph_views=(
                training_budget_runtime["local_graph_views_per_step"]
                if training_budget_runtime["enforce_runtime_graph_views_per_step"]
                else 0
            ),
        )
        epoch_optimizer_steps = int(train_metrics.get("optimizer_steps", 0))
        if epoch_optimizer_steps <= 0:
            raise RuntimeError("Training epoch completed without an optimizer step.")
        global_optimizer_step += epoch_optimizer_steps
        cumulative_graph_views += int(train_metrics.get("local_graph_views", 0))
        train_metrics["global_optimizer_step"] = global_optimizer_step
        train_metrics["cumulative_graph_views"] = cumulative_graph_views
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Valid]",
        )
        selection_value = get_selection_value(cfg, val_loss, val_metrics)
        if directional_protocol:
            test_loss, test_metrics = None, None
        else:
            test_loss, test_metrics = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                optimizer=None,
                max_steps=args.max_steps,
                desc=f"Epoch {epoch}/{cfg['epochs']} [Test]",
            )
            last_test_metrics = test_metrics
            last_test_loss = test_loss
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", val_loss, epoch)
        if not directional_protocol:
            writer.add_scalar("loss/test", test_loss, epoch)
        is_best = is_selection_improved(selection_value, best_loss, selection_mode, early_stop_min_delta)
        if is_best:
            best_loss = selection_value
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1
        state = {
            "epoch": epoch,
            "global_optimizer_step": global_optimizer_step,
            "cumulative_graph_views": cumulative_graph_views,
            "training_budget_runtime": training_budget_runtime,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_selection_metric": selection_metric,
            "best_selection_mode": selection_mode,
            "best_selection_value": best_loss,
            "early_stop_bad_epochs": early_stop_bad_epochs,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_monitor": selection_metric,
            "config": cfg,
            "protocol_metadata": protocol_metadata,
            "initialization_metadata": initialization_metadata,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_loss": test_loss,
            "test_evaluated": not directional_protocol,
        }
        save_checkpoint(os.path.join(save_dir, "latest.pt"), state)
        reached_training_budget = bool(
            training_budget_runtime["enabled"]
            and global_optimizer_step >= training_budget_runtime["max_optimizer_steps"]
        )
        if reached_training_budget:
            save_checkpoint(os.path.join(save_dir, "budget_final.pt"), state)
        if is_best:
            if directional_protocol:
                logger.error(
                    "NEW BEST MODEL | VAL | epoch={} | {}={:.4f} | test=deferred_until_training_complete",
                    epoch,
                    selection_metric,
                    float(selection_value),
                )
            else:
                log_best_model(epoch, val_loss, val_metrics, test_loss, test_metrics)
            save_checkpoint(os.path.join(save_dir, "best.pt"), state)
        periodic_path = maybe_save_periodic_checkpoint(cfg, save_dir, epoch, state)
        if periodic_path is not None:
            logger.info(
                "PERIODIC CHECKPOINT | epoch={} | path={}",
                epoch,
                periodic_path,
            )
        log_epoch_summary(epoch, cfg["epochs"], train_loss, val_loss, train_metrics, val_metrics, test_metrics)
        append_epoch_metrics(
            os.path.join(save_dir, "epoch_metrics.csv"),
            epoch,
            train_loss,
            val_loss,
            best_loss,
            is_best,
            train_metrics,
            val_metrics,
            test_loss,
            test_metrics,
        )
        if early_stop_patience > 0:
            logger.info(
                "Early stopping status | monitor={} | bad_epochs={}/{} | best_value={:.4f}",
                selection_metric,
                early_stop_bad_epochs,
                early_stop_patience,
                best_loss,
            )
            if early_stop_bad_epochs >= early_stop_patience:
                logger.info(
                    "EARLY STOP | monitor={} | epoch={} | best_value={:.4f} | patience={}",
                    selection_metric,
                    epoch,
                    best_loss,
                    early_stop_patience,
                )
                break
        if reached_training_budget:
            logger.info(
                "TRAINING BUDGET REACHED | optimizer_steps={} | graph_views={}",
                global_optimizer_step,
                cumulative_graph_views,
            )
            break
        if args.max_steps > 0:
            break

    if training_budget_runtime["enabled"] and training_budget_runtime["require_exact_budget"]:
        expected_steps = training_budget_runtime["max_optimizer_steps"]
        expected_views = training_budget_runtime["max_graph_views"]
        if global_optimizer_step != expected_steps or cumulative_graph_views != expected_views:
            raise RuntimeError(
                "Formal pretraining budget was not completed exactly: "
                f"steps={global_optimizer_step}/{expected_steps}, "
                f"graph_views={cumulative_graph_views}/{expected_views}."
            )

    if directional_protocol:
        evaluation_name = (
            "budget_final.pt"
            if training_budget_runtime.get("checkpoint_selection") == "final_budget"
            else "best.pt"
        )
        evaluation_path = os.path.join(save_dir, evaluation_name)
        evaluation_ckpt = load_checkpoint_if_available(evaluation_path, map_location="cpu")
        if evaluation_ckpt is None:
            raise RuntimeError(
                "Directional protocol did not produce its selected checkpoint: "
                f"{evaluation_path}"
            )
        model.load_state_dict(evaluation_ckpt["model_state_dict"])
        selected_epoch = int(evaluation_ckpt.get("epoch", 0))
        last_test_loss, last_test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"Selected epoch {selected_epoch} [Test]",
        )
        evaluation_ckpt["test_metrics"] = last_test_metrics
        evaluation_ckpt["test_loss"] = last_test_loss
        evaluation_ckpt["test_evaluated"] = True
        evaluation_ckpt["test_evaluation_checkpoint"] = evaluation_name
        evaluation_ckpt["protocol_metadata"] = protocol_metadata
        save_checkpoint(evaluation_path, evaluation_ckpt)
        writer.add_scalar("loss/test", last_test_loss, selected_epoch)
        logger.info(
            "Directional protocol final test | checkpoint={} | epoch={} | loss={:.4f} | metrics={}",
            evaluation_path,
            selected_epoch,
            last_test_loss,
            format_metric_payload(last_test_metrics),
        )

    test_metrics = last_test_metrics
    writer.flush()
    writer.close()
    result = {
        "save_dir": save_dir,
        "best_loss": best_loss,
        "best_selection_metric": selection_metric,
        "best_selection_mode": selection_mode,
        "best_selection_value": best_loss,
        "test_loss": last_test_loss,
        "test_metrics": test_metrics,
        "test_evaluation_checkpoint": (
            evaluation_name if directional_protocol else "last_epoch"
        ),
        "protocol_metadata": protocol_metadata,
        "training_budget_runtime": training_budget_runtime,
        "global_optimizer_step": global_optimizer_step,
        "cumulative_graph_views": cumulative_graph_views,
        "initialization_metadata": initialization_metadata,
    }
    write_metrics_json(os.path.join(save_dir, "final_metrics.json"), result)
    logger.info("Final test metrics: {}", format_metric_payload(test_metrics))
    print(format_metric_payload(result))


if __name__ == "__main__":
    main()
