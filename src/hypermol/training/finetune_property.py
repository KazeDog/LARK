"""SMILES-only masked classification/regression property finetuning.

This entrypoint always selects checkpoints using a task-appropriate validation
metric.  By default it touches the test loader exactly once, after reloading
``best.pt``.  An explicit ``test_every_epoch`` config enables diagnostic test
tracking after every epoch while keeping checkpoint selection validation-only.
The three supported transfer variants are intentionally explicit so that a
scratch run cannot silently load pretraining and a frozen run cannot silently
train the encoder.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, Mapping


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "finetune_esol.yaml"


import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hypermol.data.property import MolecularPropertyCollator, MolecularPropertyDataset
from hypermol.losses.property_loss import (
    BinaryPropertyLoss,
    RegressionPropertyLoss,
    compute_pos_weight_from_labels,
    compute_regression_target_stats,
)
from hypermol.models.property_model import MolecularPropertyModel
from hypermol.utils.checkpoint import checkpoint_sha256, load_checkpoint, save_checkpoint
from hypermol.utils.config import load_yaml_config
from hypermol.utils.pretrained import load_molecular_encoder_weights, parameter_trainability_summary
from hypermol.utils.property_metrics import (
    compute_multitask_property_metrics,
    compute_multitask_regression_metrics,
)
from hypermol.utils.runtime import build_summary_writer, get_device, move_to_device, set_seed
from hypermol.utils.train_logging import (
    append_epoch_metrics,
    format_lrs,
    format_metric_payload,
    log_config,
    log_epoch_summary,
    logger,
    setup_training_logger,
    write_json,
    write_metrics_json,
)


VARIANTS = (
    "pretrained_finetune",
    "pretrained_frozen",
    "scratch_finetune",
)
SPLITS = ("train", "valid", "test")
FINGERPRINT_VERSION = "moleculenet_property_v2"
TASK_TYPES = ("classification", "regression")
TEST_EVALUATION_BEST_ONCE = "best_checkpoint_once"
TEST_EVALUATION_EPOCHWISE = "every_epoch_plus_best_checkpoint_once"
SELECTION_ALIASES = {
    "roc_auc": "roc_auc",
    "auroc": "roc_auc",
    "average_precision": "average_precision",
    "ap": "average_precision",
    "pr_auc": "average_precision",
    "prc_auc": "average_precision",
    "rmse": "rmse",
    "mae": "mae",
    "spearman": "spearman",
    "spearmanr": "spearman",
    "spearman_r": "spearman",
}
SWEEP_CRITICAL_CONFIG_KEYS = (
    "batch_size",
    "num_workers",
    "lr",
    "weight_decay",
    "epochs",
    "early_stop_patience",
    "early_stop_min_delta",
    "task_type",
    "selection_metric",
    "selection_mode",
    "official_metric",
    "model",
    "loss",
)


def resolve_property_protocol(cfg: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return task type, metric key, optimization mode, and config metric name."""

    task_type = str(cfg.get("task_type") or "classification").strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Property task_type must be one of {TASK_TYPES}; got {task_type!r}.")
    default_metric = "roc_auc" if task_type == "classification" else "rmse"
    raw_metric = str(cfg.get("selection_metric") or default_metric).strip().lower().replace("-", "_")
    if raw_metric.startswith("validation_"):
        raw_metric = raw_metric[len("validation_") :]
    try:
        metric = SELECTION_ALIASES[raw_metric]
    except KeyError as exc:
        raise ValueError(f"Unsupported property selection metric: {raw_metric!r}.") from exc
    allowed = (
        {"roc_auc", "average_precision"}
        if task_type == "classification"
        else {"rmse", "mae", "spearman"}
    )
    if metric not in allowed:
        raise ValueError(f"Selection metric {metric!r} is incompatible with {task_type}.")
    mode = "min" if metric in {"rmse", "mae"} else "max"
    return task_type, metric, mode, f"validation_{metric}"


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one immutable experiment input or output."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    # Match the formal sweep runner byte-for-byte, including escaped Unicode.
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_configured_fingerprint_binding(
    configured_payload: Mapping[str, Any],
    trainer_payload: Mapping[str, Any],
) -> None:
    """Bind a sweep-runner fingerprint to the inputs observed by the trainer."""

    configured_identity = configured_payload.get("identity") or {}
    trainer_identity = trainer_payload["identity"]
    observed_identity = {
        "dataset": str(configured_identity.get("dataset") or "").strip().lower(),
        "variant": str(configured_identity.get("variant") or "").strip(),
        "seed": int(configured_identity.get("seed", -1)),
    }
    expected_identity = {
        "dataset": str(trainer_identity["dataset"]).strip().lower(),
        "variant": str(trainer_identity["variant"]),
        "seed": int(trainer_identity["seed"]),
    }
    if observed_identity != expected_identity:
        raise ValueError(
            "Configured run fingerprint identity does not match actual trainer identity: "
            f"configured={observed_identity}, actual={expected_identity}"
        )

    configured_data = configured_payload.get("data") or {}
    configured_split_hashes = configured_data.get("split_hashes")
    if isinstance(configured_split_hashes, Mapping):
        observed_split_hashes = {split: str(configured_split_hashes.get(split) or "") for split in SPLITS}
    else:
        observed_split_hashes = {}
        for split in SPLITS:
            key = f"{split}_csv"
            entry = configured_data.get(key) or {}
            observed_split_hashes[split] = str(entry.get("sha256") or "") if isinstance(entry, Mapping) else ""
    expected_split_hashes = {
        split: str(trainer_payload["data"]["split_hashes"][split]) for split in SPLITS
    }
    if observed_split_hashes != expected_split_hashes:
        raise ValueError(
            "Configured run fingerprint split hashes do not match actual files: "
            f"configured={observed_split_hashes}, actual={expected_split_hashes}"
        )

    configured_pretraining = configured_payload.get("pretraining") or {}
    observed_pretrain_hash = configured_pretraining.get("checkpoint_sha256")
    expected_pretrain_hash = trainer_payload["pretraining"]["checkpoint_sha256"]
    if observed_pretrain_hash != expected_pretrain_hash:
        raise ValueError(
            "Configured run fingerprint pretraining checkpoint hash does not match the actual checkpoint: "
            f"configured={observed_pretrain_hash}, actual={expected_pretrain_hash}"
        )

    configured_protocol = configured_payload.get("protocol")
    if not isinstance(configured_protocol, Mapping):
        raise ValueError("Configured run fingerprint has no protocol mapping to validate.")
    actual_protocol = trainer_payload["configured_protocol"]
    observed_protocol = dict(configured_protocol)
    expected_protocol = {key: actual_protocol.get(key) for key in observed_protocol}
    if observed_protocol != expected_protocol:
        raise ValueError(
            "Configured run fingerprint property protocol does not match the trainer: "
            f"configured={observed_protocol}, actual={expected_protocol}"
        )

    configured_base = configured_payload.get("base_config")
    if isinstance(configured_base, Mapping):
        configured_critical = configured_base.get("critical")
        if not isinstance(configured_critical, Mapping):
            raise ValueError("Configured run fingerprint base_config has no critical mapping.")
        actual_critical = trainer_payload["configured_critical"]
        observed_critical = dict(configured_critical)
        expected_critical = {key: actual_critical.get(key) for key in observed_critical}
        if observed_critical != expected_critical:
            raise ValueError(
                "Configured run fingerprint critical config does not match the trainer: "
                f"configured={observed_critical}, actual={expected_critical}"
            )


def _split_csv(data_cfg: Mapping[str, Any], split: str) -> str:
    key = f"{split}_csv"
    if data_cfg.get(key):
        return os.path.abspath(os.path.expanduser(str(data_cfg[key])))
    if not data_cfg.get("data_root"):
        raise ValueError(f"Property config requires data.{key} or data.data_root.")
    return os.path.abspath(os.path.join(str(data_cfg["data_root"]), f"{split}.csv"))


def resolve_data_paths(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("Property config requires a data mapping.")
    split_paths = {split: _split_csv(data_cfg, split) for split in SPLITS}
    data_root = os.path.abspath(
        os.path.expanduser(str(data_cfg.get("data_root") or os.path.dirname(split_paths["train"])))
    )
    molecule_store = (
        data_cfg.get("molecule_store")
        or data_cfg.get("molecule_db")
        or os.path.join(data_root, "molecule_store.mdb")
    )
    manifest = data_cfg.get("manifest") or data_cfg.get("manifest_path") or os.path.join(data_root, "manifest.json")
    paths = {
        "data_root": data_root,
        "split_paths": split_paths,
        "molecule_store": os.path.abspath(os.path.expanduser(str(molecule_store))),
        "manifest": os.path.abspath(os.path.expanduser(str(manifest))),
    }
    for name, path in {
        **{f"{split}_csv": value for split, value in split_paths.items()},
        "molecule_store": paths["molecule_store"],
        "manifest": paths["manifest"],
    }.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Property {name} does not exist: {path}")
    return paths


def validate_smiles_only_manifest(
    manifest_path: str,
    split_hashes: Mapping[str, str],
    molecule_store_sha256: str,
) -> Dict[str, Any]:
    """Fail closed if the prepared dataset does not document SMILES-only inputs."""

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Property manifest is not a JSON object: {manifest_path}")
    descriptor_columns = manifest.get("descriptor_columns_consumed")
    if descriptor_columns != []:
        raise ValueError(
            "Property manifest must explicitly report descriptor_columns_consumed=[]; "
            f"observed {descriptor_columns!r}."
        )
    source_columns = manifest.get("source_columns_consumed")
    num_tasks = int(manifest.get("num_tasks", max(0, len(source_columns or []) - 1)))
    if not isinstance(source_columns, list) or num_tasks <= 0 or len(source_columns) != num_tasks + 1:
        raise ValueError(
            "Property manifest must document exactly one SMILES column and all task-label columns; "
            f"observed {source_columns!r}."
        )
    task_names = manifest.get("task_names", source_columns[1:])
    label_columns = manifest.get(
        "processed_label_columns",
        ["label"] if num_tasks == 1 else [f"label_{index:03d}" for index in range(num_tasks)],
    )
    if (
        not isinstance(task_names, list)
        or not isinstance(label_columns, list)
        or len(task_names) != num_tasks
        or len(label_columns) != num_tasks
        or len(set(map(str, task_names))) != num_tasks
    ):
        raise ValueError("Property manifest contains inconsistent task metadata.")
    manifest["num_tasks"] = num_tasks
    manifest["task_names"] = [str(value) for value in task_names]
    manifest["processed_label_columns"] = [str(value) for value in label_columns]
    task_type = str(manifest.get("task_type") or "classification").strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Property manifest has unsupported task_type: {task_type!r}.")
    manifest["task_type"] = task_type
    manifest_splits = manifest.get("splits", {})
    for split, observed_hash in split_hashes.items():
        expected_hash = (manifest_splits.get(split) or {}).get("csv_sha256")
        if not expected_hash:
            raise ValueError(f"Property manifest has no CSV hash for split '{split}'.")
        if str(expected_hash) != str(observed_hash):
            raise ValueError(
                f"Property split hash mismatch for {split}: manifest={expected_hash}, observed={observed_hash}"
            )
    store_metadata = manifest.get("molecule_store") or {}
    if store_metadata.get("fingerprint_fields_present") != []:
        raise ValueError(
            "Property molecule store must explicitly report fingerprint_fields_present=[]; "
            f"observed {store_metadata.get('fingerprint_fields_present')!r}."
        )
    expected_store_hash = str(store_metadata.get("file_sha256") or "")
    if not expected_store_hash:
        raise ValueError("Property manifest has no molecule-store SHA-256.")
    if expected_store_hash != str(molecule_store_sha256):
        raise ValueError(
            "Property molecule-store hash mismatch: "
            f"manifest={expected_store_hash}, observed={molecule_store_sha256}"
        )
    disjointness = manifest.get("disjointness") or {}
    if disjointness.get("passed") is not True:
        raise ValueError("Property manifest does not contain a passing scaffold-disjointness audit.")
    return manifest


def resolve_variant(cfg: Mapping[str, Any], cli_variant: str | None = None) -> str:
    variant = str(cli_variant or cfg.get("variant") or "").strip()
    if variant not in VARIANTS:
        raise ValueError(f"Property variant must be exactly one of {VARIANTS}; observed {variant!r}.")
    return variant


def resolve_property_test_protocol(cfg: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether test is tracked per epoch and its provenance label."""

    test_every_epoch = bool(cfg.get("test_every_epoch", False))
    test_evaluation = (
        TEST_EVALUATION_EPOCHWISE if test_every_epoch else TEST_EVALUATION_BEST_ONCE
    )
    return test_every_epoch, test_evaluation


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    identity_overridden = args.seed is not None or args.variant is not None
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.gpu is not None:
        cfg["gpu"] = int(args.gpu)
    if args.run_name:
        cfg["run_name"] = str(args.run_name)
    elif identity_overridden:
        # A base name such as ``bace_pretrained_finetune_seed42`` must never
        # survive a manual ``--variant scratch_finetune --seed 43`` override.
        cfg["run_name"] = ""
    cfg["variant"] = resolve_variant(cfg, args.variant)
    cfg["max_steps"] = int(args.max_steps)
    cli_test_every_epoch = getattr(args, "test_every_epoch", None)
    if cli_test_every_epoch is not None:
        cfg["test_every_epoch"] = bool(cli_test_every_epoch)
    test_every_epoch, test_evaluation = resolve_property_test_protocol(cfg)
    cfg["test_every_epoch"] = test_every_epoch
    cfg["test_evaluation"] = test_evaluation
    task_type, selection_metric, selection_mode, selection_name = resolve_property_protocol(cfg)
    cfg["task_type"] = task_type
    cfg["selection_metric"] = selection_name
    cfg["selection_mode"] = selection_mode

    if cfg["variant"] == "scratch_finetune":
        cfg["pretrain_path"] = ""
        cfg["freeze_pretrained"] = False
    else:
        pretrain_path = str(cfg.get("pretrain_path") or "").strip()
        if not pretrain_path:
            raise ValueError(f"Variant {cfg['variant']} requires a non-empty pretrain_path.")
        pretrain_path = os.path.abspath(os.path.expanduser(pretrain_path))
        if not os.path.isfile(pretrain_path):
            raise FileNotFoundError(f"Pretraining checkpoint does not exist: {pretrain_path}")
        cfg["pretrain_path"] = pretrain_path
        cfg["freeze_pretrained"] = cfg["variant"] == "pretrained_frozen"

    if int(cfg.get("seed", 42)) < 0:
        raise ValueError("seed must be non-negative.")
    if int(cfg.get("epochs", 0)) <= 0:
        raise ValueError("epochs must be positive.")
    if int(cfg.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be positive.")
    if float(cfg.get("lr", 0.0)) <= 0:
        raise ValueError("lr must be positive.")
    if task_type == "classification":
        threshold = float(cfg.get("classification_threshold", 0.5))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("classification_threshold must be finite and in [0, 1].")
    gradient_clip_norm = float(cfg.get("gradient_clip_norm", 1.0))
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be finite and non-negative.")
    if int(args.max_steps) < 0:
        raise ValueError("max_steps must be non-negative.")
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    if patience < 0 or min_delta < 0:
        raise ValueError("early_stop_patience and early_stop_min_delta must be non-negative.")
    if selection_metric not in {"roc_auc", "average_precision", "rmse", "mae", "spearman"}:
        raise AssertionError("Internal property selection metric resolution failed.")
    return cfg


def build_run_provenance(
    cfg: Mapping[str, Any],
    config_path: str,
    data_paths: Mapping[str, Any],
    max_steps: int,
) -> Dict[str, Any]:
    split_hashes = {split: sha256_file(path) for split, path in data_paths["split_paths"].items()}
    molecule_store_hash = sha256_file(data_paths["molecule_store"])
    manifest = validate_smiles_only_manifest(
        data_paths["manifest"],
        split_hashes,
        molecule_store_hash,
    )
    pretrain_path = str(cfg.get("pretrain_path") or "")
    pretrain_hash = sha256_file(pretrain_path) if pretrain_path else None
    dataset_name = str(cfg.get("dataset") or cfg.get("data", {}).get("dataset") or manifest.get("dataset") or "")
    manifest_dataset = str(manifest.get("dataset") or "").strip().lower()
    if not dataset_name or not manifest_dataset or dataset_name.strip().lower() != manifest_dataset:
        raise ValueError(
            "Property config/manifest dataset mismatch: "
            f"config={dataset_name!r}, manifest={manifest.get('dataset')!r}"
        )
    task_type, selection_metric, selection_mode, selection_name = resolve_property_protocol(cfg)
    if task_type != manifest["task_type"]:
        raise ValueError(
            f"Property config/manifest task_type mismatch: config={task_type}, manifest={manifest['task_type']}"
        )
    manifest_selection = str(manifest.get("selection_metric") or selection_metric).strip().lower()
    if manifest_selection != selection_metric:
        raise ValueError(
            "Property config/manifest selection metric mismatch: "
            f"config={selection_metric}, manifest={manifest_selection}"
        )
    test_every_epoch, test_evaluation = resolve_property_test_protocol(cfg)
    trainer_payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "identity": {
            "dataset": dataset_name,
            "variant": cfg["variant"],
            "seed": int(cfg.get("seed", 42)),
        },
        "data": {
            "split_paths": dict(data_paths["split_paths"]),
            "split_hashes": split_hashes,
            "manifest_path": data_paths["manifest"],
            "manifest_sha256": sha256_file(data_paths["manifest"]),
            "molecule_store_path": data_paths["molecule_store"],
            "molecule_store_sha256": molecule_store_hash,
            "descriptor_columns_consumed": manifest["descriptor_columns_consumed"],
            "num_tasks": int(manifest["num_tasks"]),
            "task_names": list(manifest["task_names"]),
            "processed_label_columns": list(manifest["processed_label_columns"]),
            "task_type": task_type,
        },
        "pretraining": {
            "checkpoint_path": pretrain_path or None,
            "checkpoint_sha256": pretrain_hash,
            "encoder_frozen": cfg["variant"] == "pretrained_frozen",
        },
        "training_protocol": {
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "weight_decay": float(cfg.get("weight_decay", 0.0)),
            "class_balance": bool(cfg.get("class_balance", False)),
            "classification_threshold": float(cfg.get("classification_threshold", 0.5)),
            "gradient_clip_norm": float(cfg.get("gradient_clip_norm", 1.0)),
            "early_stop_patience": int(cfg.get("early_stop_patience", 0) or 0),
            "early_stop_min_delta": float(cfg.get("early_stop_min_delta", 0.0) or 0.0),
            "deterministic": bool(cfg.get("deterministic", True)),
            "max_steps": int(max_steps),
            "task_type": task_type,
            "selection_metric": selection_name,
            "selection_mode": selection_mode,
            "test_every_epoch": test_every_epoch,
            "test_evaluation": test_evaluation,
            "model": cfg.get("model", {}),
            "loss": cfg.get("loss", {}),
        },
        # These two views mirror the formal sweep payload.  They make a
        # generated job config fail closed if someone changes a critical
        # hyperparameter or protocol flag after its fingerprint was minted.
        "configured_protocol": {
            "formal_run": bool(cfg.get("formal_run", False)),
            "max_steps": int(max_steps),
            "task_type": task_type,
            "selection_metric": selection_name,
            "selection_mode": selection_mode,
            "official_metric": cfg.get("official_metric"),
            "test_every_epoch": test_every_epoch,
            "test_evaluation": test_evaluation,
            "deterministic": bool(cfg.get("deterministic", True)),
        },
        "configured_critical": {
            key: cfg.get(key) for key in SWEEP_CRITICAL_CONFIG_KEYS
        },
        "source_config": {
            "path": os.path.abspath(config_path),
            "sha256": sha256_file(config_path),
        },
    }
    trainer_fingerprint = canonical_fingerprint(trainer_payload)
    configured_fingerprint = str(cfg.get("run_fingerprint") or "").strip()
    property_metadata = cfg.get("moleculenet_property") or {}
    if configured_fingerprint:
        if not isinstance(property_metadata, Mapping):
            raise ValueError("A configured run_fingerprint requires moleculenet_property metadata.")
        metadata_fingerprint = str(property_metadata.get("run_fingerprint") or "")
        configured_payload = property_metadata.get("run_fingerprint_payload")
        if metadata_fingerprint != configured_fingerprint:
            raise ValueError("Top-level and moleculenet_property run fingerprints disagree.")
        if not isinstance(configured_payload, Mapping):
            raise ValueError("Configured run_fingerprint has no canonical payload to validate.")
        if canonical_fingerprint(configured_payload) != configured_fingerprint:
            raise ValueError("Configured MoleculeNet property run fingerprint payload is inconsistent.")
        validate_configured_fingerprint_binding(configured_payload, trainer_payload)
        run_fingerprint = configured_fingerprint
        run_fingerprint_payload = dict(configured_payload)
    else:
        run_fingerprint = trainer_fingerprint
        run_fingerprint_payload = trainer_payload
    return {
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": run_fingerprint_payload,
        "trainer_provenance_fingerprint": trainer_fingerprint,
        "trainer_provenance_payload": trainer_payload,
        "split_hashes": split_hashes,
        "manifest_sha256": trainer_payload["data"]["manifest_sha256"],
        "molecule_store_sha256": trainer_payload["data"]["molecule_store_sha256"],
        "pretrain_checkpoint_sha256": pretrain_hash,
        "num_tasks": int(manifest["num_tasks"]),
        "task_names": list(manifest["task_names"]),
        "processed_label_columns": list(manifest["processed_label_columns"]),
        "task_type": task_type,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
    }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(cfg: Mapping[str, Any], data_paths: Mapping[str, Any]):
    with open(data_paths["manifest"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    source_columns = list(manifest.get("source_columns_consumed") or [])
    num_tasks = int(manifest.get("num_tasks", max(0, len(source_columns) - 1)))
    task_names = list(manifest.get("task_names") or source_columns[1:])
    label_columns = list(
        manifest.get("processed_label_columns")
        or (["label"] if num_tasks == 1 else [f"label_{index:03d}" for index in range(num_tasks)])
    )
    task_type = str(manifest.get("task_type") or cfg.get("task_type") or "classification").strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Invalid property task_type in {data_paths['manifest']}: {task_type!r}")
    if str(cfg.get("task_type") or task_type).strip().lower() != task_type:
        raise ValueError("Property loader config/manifest task_type mismatch.")
    if num_tasks <= 0 or len(task_names) != num_tasks or len(label_columns) != num_tasks:
        raise ValueError(f"Invalid property task metadata in {data_paths['manifest']}")
    datasets = {
        split: MolecularPropertyDataset(
            split_csv_path=data_paths["split_paths"][split],
            molecule_store_path=data_paths["molecule_store"],
            label_columns=label_columns,
            task_names=task_names,
            task_type=task_type,
        )
        for split in SPLITS
    }
    seed = int(cfg.get("seed", 42))
    common = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 0)),
        "collate_fn": MolecularPropertyCollator(task_names=task_names),
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    if common["num_workers"] > 0:
        common["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
        common["persistent_workers"] = True
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=train_generator, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, datasets, {
        "num_tasks": num_tasks,
        "task_names": task_names,
        "processed_label_columns": label_columns,
        "task_type": task_type,
    }


def build_model(cfg: Mapping[str, Any], *, num_tasks: int = 1) -> MolecularPropertyModel:
    model_cfg = dict(cfg.get("model", {}))
    return MolecularPropertyModel(
        model_cfg=model_cfg,
        hidden_dim=model_cfg.get("head_hidden_dim"),
        dropout=model_cfg.get("dropout", 0.1),
        freeze_encoder=False,
        num_tasks=int(num_tasks),
    )


def initialize_variant(model: MolecularPropertyModel, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply one and only one transfer policy, with strict encoder loading."""

    variant = resolve_variant(cfg)
    load_result: Dict[str, Any] = {
        "loaded": False,
        "loaded_keys": 0,
        "missing": [],
        "unexpected": [],
    }
    if variant in {"pretrained_finetune", "pretrained_frozen"}:
        pretrain_path = str(cfg.get("pretrain_path") or "")
        if not pretrain_path or not os.path.isfile(pretrain_path):
            raise FileNotFoundError(f"Variant {variant} requires an existing pretraining checkpoint: {pretrain_path}")
        load_result = load_molecular_encoder_weights(
            model,
            pretrain_path,
            map_location="cpu",
            strict=True,
        )
        if not load_result.get("loaded") or int(load_result.get("loaded_keys", 0)) <= 0:
            raise RuntimeError(f"No molecular encoder parameters were loaded for variant {variant}.")
        if load_result.get("missing") or load_result.get("unexpected"):
            raise RuntimeError(f"Strict pretrained encoder load was incomplete: {load_result}")
        model.set_encoder_frozen(variant == "pretrained_frozen")
    else:
        if cfg.get("pretrain_path"):
            raise ValueError("scratch_finetune must have an empty pretrain_path after variant resolution.")
        model.set_encoder_frozen(False)

    encoder_parameters = list(model.backbone.encoder.parameters())
    any_trainable = any(parameter.requires_grad for parameter in encoder_parameters)
    if variant == "pretrained_frozen":
        if any_trainable:
            raise RuntimeError("pretrained_frozen left trainable encoder parameters.")
        if model.backbone.encoder.training:
            raise RuntimeError("pretrained_frozen encoder must be in eval mode.")
    elif not any_trainable:
        raise RuntimeError(f"Variant {variant} unexpectedly froze the molecular encoder.")
    return load_result


def resolve_pos_weight(cfg: Mapping[str, Any], train_labels: Any):
    loss_cfg = cfg.get("loss") or {}
    value = loss_cfg.get("pos_weight")
    if "pos_weight" not in loss_cfg and bool(cfg.get("class_balance", False)):
        value = "auto"
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError("loss.pos_weight string value must be 'auto'.")
        return compute_pos_weight_from_labels(
            train_labels,
            max_pos_weight=loss_cfg.get("max_pos_weight"),
        )
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 1 or not torch.isfinite(tensor).all() or not torch.all(tensor > 0):
            raise ValueError("loss.pos_weight task vector must contain finite positive values.")
        return tensor
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("loss.pos_weight must be positive and finite, null, 'auto', or a task vector.")
    return value


def get_property_selection_score(val_metrics: Mapping[str, Any], metric: str = "roc_auc") -> float:
    if metric not in val_metrics:
        raise ValueError(f"Property validation metrics do not contain {metric}.")
    score = float(val_metrics[metric])
    if not math.isfinite(score):
        raise ValueError(f"Property validation {metric} is not finite.")
    return score


def is_property_metric_improved(
    score: float,
    best_score: float,
    *,
    mode: str = "max",
    min_delta: float = 0.0,
) -> bool:
    if mode == "max":
        return float(score) > float(best_score) + float(min_delta)
    if mode == "min":
        return float(score) < float(best_score) - float(min_delta)
    raise ValueError(f"Property selection mode must be 'min' or 'max', got {mode!r}.")


def is_property_roc_auc_improved(score: float, best_score: float, min_delta: float = 0.0) -> bool:
    """Backward-compatible helper retained for existing tests/callers."""

    return is_property_metric_improved(score, best_score, mode="max", min_delta=min_delta)


def assert_frozen_encoder_eval(model: MolecularPropertyModel) -> None:
    if any(parameter.requires_grad for parameter in model.backbone.encoder.parameters()):
        raise RuntimeError("Frozen property encoder gained trainable parameters.")
    if model.backbone.encoder.training:
        raise RuntimeError("Frozen property encoder entered train mode.")


def run_property_epoch(
    model: MolecularPropertyModel,
    loader: DataLoader,
    criterion: BinaryPropertyLoss | RegressionPropertyLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int = 0,
    desc: str = "",
    collect_predictions: bool = False,
    threshold: float = 0.5,
    gradient_clip_norm: float = 1.0,
    task_type: str = "classification",
) -> tuple[float, Dict[str, Any], list[Dict[str, Any]]]:
    task_type = str(task_type).strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported property task_type: {task_type!r}")
    is_train = optimizer is not None
    model.train(is_train)
    if getattr(model, "encoder_frozen", False):
        assert_frozen_encoder_eval(model)

    total_loss_sum = 0.0
    weighted_sample_sum = 0.0
    sample_size = 0
    labels_all = []
    label_masks_all = []
    predictions_all = []
    molecule_size = 0
    task_names: list[str] | None = None
    prediction_rows: list[Dict[str, Any]] = []
    disable_progress = os.environ.get("HYPERMOL_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
    iterator = tqdm(loader, desc=desc or ("Train" if is_train else "Eval"), leave=False, disable=disable_progress)
    for step, batch in enumerate(iterator, start=1):
        if not batch:
            continue
        batch = move_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(batch)
            criterion_result = criterion(outputs, batch, is_train=is_train)
            if not isinstance(criterion_result, tuple) or len(criterion_result) != 2:
                raise TypeError("Property loss must return (loss, logging_output).")
            loss, loss_log = criterion_result
            if is_train:
                loss.backward()
                if float(gradient_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        float(gradient_clip_norm),
                    )
                optimizer.step()
                if getattr(model, "encoder_frozen", False):
                    assert_frozen_encoder_eval(model)

        labels = batch["labels"].detach()
        logits = outputs["logits"].detach()
        label_mask = batch.get("label_mask")
        if labels.ndim == 1:
            labels = labels[:, None]
        if logits.ndim == 1:
            logits = logits[:, None]
        if label_mask is None:
            label_mask = torch.isfinite(labels)
        elif label_mask.ndim == 1:
            label_mask = label_mask[:, None]
        if labels.shape != logits.shape or labels.shape != label_mask.shape:
            raise RuntimeError(
                f"Property label/logit/mask shapes diverged: {labels.shape}, {logits.shape}, {label_mask.shape}"
            )
        predictions = (
            torch.sigmoid(logits)
            if task_type == "classification"
            else criterion.denormalize(logits)
        )
        batch_size = int(label_mask.sum().item())
        batch_molecules = int(labels.shape[0])
        logged_sum = loss_log.get("total_loss_sum") if isinstance(loss_log, Mapping) else None
        logged_weight = loss_log.get("weighted_sample_sum") if isinstance(loss_log, Mapping) else None
        total_loss_sum += float(logged_sum) if logged_sum is not None else float(loss.detach().cpu()) * batch_size
        weighted_sample_sum += float(logged_weight) if logged_weight is not None else batch_size
        sample_size += batch_size
        molecule_size += batch_molecules
        labels_all.append(labels.cpu())
        label_masks_all.append(label_mask.cpu())
        predictions_all.append(predictions.cpu())
        batch_task_names = batch.get("task_names")
        if batch_task_names is not None:
            current_names = [str(value) for value in batch_task_names]
            if task_names is None:
                task_names = current_names
            elif task_names != current_names:
                raise RuntimeError("Property task names changed between batches.")

        if collect_predictions:
            source_indices = batch["source_index"].detach().view(-1).cpu().tolist()
            smiles = list(batch["smiles"])
            label_values = labels.cpu().numpy()
            mask_values = label_mask.cpu().numpy().astype(bool)
            prediction_values = predictions.cpu().numpy()
            if not (len(source_indices) == len(smiles) == len(label_values) == len(prediction_values)):
                raise RuntimeError("Property prediction metadata and tensor lengths diverged.")
            names = task_names or [f"task_{index}" for index in range(labels.shape[1])]
            for row_index, (source_index, molecule) in enumerate(zip(source_indices, smiles)):
                for task_index, task_name in enumerate(names):
                    if not mask_values[row_index, task_index]:
                        continue
                    row = {
                        "source_index": int(source_index),
                        "canonical_smiles": str(molecule),
                        "label": (
                            int(label_values[row_index, task_index])
                            if task_type == "classification"
                            else float(label_values[row_index, task_index])
                        ),
                    }
                    predicted_value = float(prediction_values[row_index, task_index])
                    if task_type == "classification":
                        row.update(
                            {
                                "probability": predicted_value,
                                "prediction": int(predicted_value >= float(threshold)),
                            }
                        )
                    else:
                        row["predicted_value"] = predicted_value
                    if labels.shape[1] > 1:
                        row.update({"task_index": int(task_index), "task_name": str(task_name)})
                    prediction_rows.append(row)
        iterator.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})
        # Classification smoke runs extend to a prefix containing both classes;
        # regression smoke runs stop at the requested batch count.
        if max_steps > 0 and step >= max_steps:
            if task_type == "regression":
                break
            prefix_labels = torch.cat(labels_all).numpy()
            prefix_mask = torch.cat(label_masks_all).numpy().astype(bool)
            if any(
                set(prefix_labels[prefix_mask[:, task_index], task_index].astype(int).tolist()) == {0, 1}
                for task_index in range(prefix_labels.shape[1])
            ):
                break

    if sample_size <= 0:
        raise RuntimeError(f"Property epoch produced no samples: {desc or 'unnamed epoch'}")
    labels_numpy = torch.cat(labels_all).numpy()
    label_masks_numpy = torch.cat(label_masks_all).numpy().astype(bool)
    predictions_numpy = torch.cat(predictions_all).numpy()
    if task_type == "classification":
        metrics = compute_multitask_property_metrics(
            labels_numpy,
            probabilities=predictions_numpy,
            mask=label_masks_numpy,
            task_names=task_names,
            threshold=float(threshold),
        )
    else:
        metrics = compute_multitask_regression_metrics(
            labels_numpy,
            predictions_numpy,
            mask=label_masks_numpy,
            task_names=task_names,
        )
    if not collect_predictions:
        metrics.pop("task_metrics", None)
    if weighted_sample_sum <= 0:
        raise RuntimeError(f"Property epoch has non-positive effective sample weight: {weighted_sample_sum}")
    average_loss = total_loss_sum / weighted_sample_sum
    metrics = dict(metrics)
    metrics["avg_total_loss"] = float(average_loss)
    metrics["sample_size"] = int(sample_size)
    metrics["molecule_size"] = int(molecule_size)
    return float(average_loss), metrics, prediction_rows


def write_prediction_csv(path: str, rows: Iterable[Mapping[str, Any]], selected_epoch: int) -> None:
    rows = [dict(row, selected_epoch=int(selected_epoch)) for row in rows]
    if not rows:
        raise ValueError("Refusing to write an empty property prediction CSV.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    fieldnames = ["source_index", "canonical_smiles"]
    if "task_index" in rows[0]:
        fieldnames.extend(["task_index", "task_name"])
    fieldnames.append("label")
    if "predicted_value" in rows[0]:
        fieldnames.append("predicted_value")
    else:
        fieldnames.extend(["probability", "prediction"])
    fieldnames.append("selected_epoch")
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--variant", choices=VARIANTS, default=None, help="Strict transfer-learning variant.")
    parser.add_argument("--run_name", default="", help="Override config run name.")
    parser.add_argument("--gpu", type=int, default=None, help="Override config GPU; use -1 for CPU.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Smoke-run batch limit; a split extends only as needed to include both classes.",
    )
    parser.add_argument(
        "--test-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override config test_every_epoch; test metrics never select checkpoints.",
    )
    parser.add_argument(
        "--resume_path",
        default="",
        help="Resume this exact run in place from a property-training checkpoint.",
    )
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_yaml_config(args.config).raw, args)
    torch_threads = int(os.environ.get("HYPERMOL_TORCH_THREADS", "0") or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(max(1, min(torch_threads, 4)))
        except RuntimeError:
            pass

    data_paths = resolve_data_paths(cfg)
    provenance = build_run_provenance(cfg, args.config, data_paths, args.max_steps)
    cfg.update(provenance)
    cfg["num_tasks"] = int(provenance["num_tasks"])
    cfg["task_names"] = list(provenance["task_names"])
    cfg["processed_label_columns"] = list(provenance["processed_label_columns"])
    cfg["task_type"] = str(provenance["task_type"])
    cfg["data"] = dict(cfg["data"])
    cfg["data"].update(
        {
            "data_root": data_paths["data_root"],
            "molecule_store": data_paths["molecule_store"],
            "manifest": data_paths["manifest"],
            **{f"{split}_csv": path for split, path in data_paths["split_paths"].items()},
        }
    )
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["deterministic"] = bool(cfg.get("deterministic", True))

    dataset_name = str(
        cfg.get("dataset")
        or cfg.get("data", {}).get("dataset")
        or provenance["run_fingerprint_payload"]["identity"]["dataset"]
        or "property"
    )
    default_run_name = (
        f"{dataset_name}_{cfg['variant']}_seed{cfg['seed']}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_name = str(cfg.get("run_name") or default_run_name)
    cfg["run_name"] = run_name
    save_root = str(cfg.get("save_dir") or "").strip()
    if not save_root:
        raise ValueError("Property config requires save_dir.")
    resume_path = (
        os.path.abspath(os.path.expanduser(str(args.resume_path)))
        if args.resume_path
        else ""
    )
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Property resume checkpoint does not exist: {resume_path}")
        save_dir = os.path.dirname(resume_path)
        expected_save_dir = os.path.abspath(os.path.join(save_root, run_name))
        if save_dir != expected_save_dir:
            raise ValueError(
                "Property --resume_path must point inside the configured exact run directory: "
                f"checkpoint_dir={save_dir}, configured_dir={expected_save_dir}"
            )
    else:
        save_dir = os.path.abspath(os.path.join(save_root, run_name))
    if os.path.isfile(os.path.join(save_dir, "final_metrics.json")):
        raise FileExistsError(f"Refusing to overwrite a completed property run: {save_dir}")

    log_path = setup_training_logger(save_dir)
    log_config(cfg, title="SMILES-only Molecular Property Configuration")
    write_json(os.path.join(save_dir, "config_resolved.json"), cfg)
    write_json(os.path.join(save_dir, "run_fingerprint.json"), provenance)
    logger.info("Run directory: {}", save_dir)
    logger.info("Training log: {}", log_path)
    logger.info("Run fingerprint: {}", provenance["run_fingerprint"])
    logger.info("Split hashes: {}", format_metric_payload(provenance["split_hashes"]))

    device = get_device(int(cfg.get("gpu", -1)))
    set_seed(cfg["seed"], deterministic=cfg["deterministic"], device=device)
    logger.info("Device: {}", device)

    loaders, datasets, task_spec = build_loaders(cfg, data_paths)
    logger.info(
        "Dataset={} | task_type={} | tasks={} | train={} | valid={} | test={} | descriptor_columns_consumed=[]",
        dataset_name,
        task_spec["task_type"],
        task_spec["num_tasks"],
        len(datasets["train"]),
        len(datasets["valid"]),
        len(datasets["test"]),
    )
    model = build_model(cfg, num_tasks=task_spec["num_tasks"])
    load_result = initialize_variant(model, cfg)
    model = model.to(device)
    trainability = parameter_trainability_summary(model)
    logger.info(
        "Variant={} | pretrained_loaded={} | loaded_keys={} | trainable_params={} | frozen_params={}",
        cfg["variant"],
        bool(load_result.get("loaded")),
        int(load_result.get("loaded_keys", 0)),
        trainability["trainable_params"],
        trainability["frozen_params"],
    )

    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not optimizer_parameters:
        raise RuntimeError("Property model has no trainable parameters.")
    optimizer = AdamW(
        optimizer_parameters,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    task_type, selection_metric, selection_mode, selection_name = resolve_property_protocol(cfg)
    if task_type == "classification":
        pos_weight = resolve_pos_weight(cfg, datasets["train"].labels_matrix)
        criterion: BinaryPropertyLoss | RegressionPropertyLoss = BinaryPropertyLoss(pos_weight=pos_weight).to(device)
        if isinstance(pos_weight, torch.Tensor):
            pos_weight_log: Any = {
                "num_tasks": int(pos_weight.numel()),
                "min": float(pos_weight.min()),
                "median": float(pos_weight.median()),
                "max": float(pos_weight.max()),
            }
        else:
            pos_weight_log = pos_weight
        logger.info("Masked binary property loss | pos_weight={}", pos_weight_log)
    else:
        target_mean, target_std = compute_regression_target_stats(datasets["train"].labels_matrix)
        criterion = RegressionPropertyLoss(target_mean=target_mean, target_std=target_std).to(device)
        logger.info(
            "Masked standardized regression MSE | target_mean={} | target_std={}",
            target_mean.tolist(),
            target_std.tolist(),
        )

    writer = build_summary_writer(os.path.join(save_dir, "tb"))
    test_every_epoch, test_evaluation = resolve_property_test_protocol(cfg)
    test_evaluation_count = 0
    epochwise_test_evaluation_count = 0
    epochs_completed = 0
    best_score = float("inf") if selection_mode == "min" else float("-inf")
    best_loss = float("inf")
    best_epoch = 0
    best_val_metrics: Dict[str, Any] = {}
    bad_epochs = 0
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    logger.info(
        "Selection protocol | monitor=val_{} | mode={} | patience={} | min_delta={} | "
        "test={} | test_used_for_selection=false",
        selection_metric,
        selection_mode,
        patience,
        min_delta,
        test_evaluation,
    )
    threshold = float(cfg.get("classification_threshold", 0.5))
    gradient_clip_norm = float(cfg.get("gradient_clip_norm", 1.0))

    start_epoch = 1
    if resume_path:
        resume_checkpoint = load_checkpoint(resume_path, map_location="cpu")
        checkpoint_fingerprint = str(resume_checkpoint.get("run_fingerprint") or "")
        if checkpoint_fingerprint != str(provenance["run_fingerprint"]):
            raise ValueError(
                "Property resume checkpoint fingerprint mismatch: "
                f"checkpoint={checkpoint_fingerprint}, config={provenance['run_fingerprint']}"
            )
        if str(resume_checkpoint.get("variant")) != str(cfg["variant"]):
            raise ValueError("Property resume checkpoint variant does not match the config.")
        if int(resume_checkpoint.get("seed", -1)) != int(cfg["seed"]):
            raise ValueError("Property resume checkpoint seed does not match the config.")
        if int(resume_checkpoint.get("max_steps", -1)) != int(args.max_steps):
            raise ValueError("Property resume checkpoint max_steps does not match the config.")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        epochs_completed = int(resume_checkpoint["epoch"])
        best_score = float(resume_checkpoint["best_score"])
        best_loss = float(resume_checkpoint["best_loss"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        best_val_metrics = dict(resume_checkpoint.get("best_val_metrics") or {})
        bad_epochs = int(resume_checkpoint.get("early_stop_bad_epochs", 0))
        test_evaluation_count = int(resume_checkpoint.get("test_evaluation_count", epochs_completed if test_every_epoch else 0))
        epochwise_test_evaluation_count = int(
            resume_checkpoint.get(
                "epochwise_test_evaluation_count",
                epochs_completed if test_every_epoch else 0,
            )
        )
        logger.info(
            "RESUME | checkpoint={} | completed_epoch={} | start_epoch={} | best_epoch={} | "
            "best_val_{}={:.6f} | bad_epochs={}",
            resume_path,
            epochs_completed,
            start_epoch,
            best_epoch,
            selection_metric,
            best_score,
            bad_epochs,
        )
        if start_epoch > int(cfg["epochs"]):
            raise ValueError(
                f"Property resume checkpoint already reached epoch {epochs_completed}, "
                f"but config epochs={cfg['epochs']}."
            )

    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        logger.info("[Epoch {}] current lrs: {}", epoch, format_lrs(optimizer))
        train_loss, train_metrics, _ = run_property_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_steps=args.max_steps,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Train]",
            threshold=threshold,
            gradient_clip_norm=gradient_clip_norm,
            task_type=task_type,
        )
        val_loss, val_metrics, _ = run_property_epoch(
            model,
            loaders["valid"],
            criterion,
            device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Valid]",
            threshold=threshold,
            gradient_clip_norm=gradient_clip_norm,
            task_type=task_type,
        )
        epoch_test_loss: float | None = None
        epoch_test_metrics: Dict[str, Any] = {}
        if test_every_epoch:
            epoch_test_loss, epoch_test_metrics, _ = run_property_epoch(
                model,
                loaders["test"],
                criterion,
                device,
                optimizer=None,
                max_steps=args.max_steps,
                desc=f"Epoch {epoch}/{cfg['epochs']} [Test]",
                threshold=threshold,
                gradient_clip_norm=gradient_clip_norm,
                task_type=task_type,
            )
            test_evaluation_count += 1
            epochwise_test_evaluation_count += 1
            write_json(
                os.path.join(save_dir, f"test_epoch_{epoch:04d}_metrics.json"),
                {
                    "epoch": int(epoch),
                    "test_loss": float(epoch_test_loss),
                    "test_metrics": epoch_test_metrics,
                    "diagnostic_only": True,
                    "used_for_checkpoint_selection": False,
                    "run_fingerprint": provenance["run_fingerprint"],
                },
            )
        val_score = get_property_selection_score(val_metrics, selection_metric)
        is_best = is_property_metric_improved(
            val_score,
            best_score,
            mode=selection_mode,
            min_delta=min_delta,
        )
        if is_best:
            best_score = val_score
            best_loss = val_loss
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1

        epochs_completed = epoch
        state = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "variant": cfg["variant"],
            "seed": cfg["seed"],
            "best_metric": f"val_{selection_metric}",
            "best_score": best_score,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "early_stop_bad_epochs": bad_epochs,
            "early_stop_patience": patience,
            "early_stop_min_delta": min_delta,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": epoch_test_metrics,
            "test_evaluation": test_evaluation,
            "test_evaluation_count": test_evaluation_count,
            "epochwise_test_evaluation_count": epochwise_test_evaluation_count,
            "test_used_for_checkpoint_selection": False,
            "max_steps": int(args.max_steps),
            "smoke_run": bool(args.max_steps > 0),
            "run_fingerprint": provenance["run_fingerprint"],
            "run_fingerprint_payload": provenance["run_fingerprint_payload"],
            "split_hashes": provenance["split_hashes"],
            "pretrain_checkpoint_sha256": provenance["pretrain_checkpoint_sha256"],
            "config": cfg,
        }
        save_checkpoint(os.path.join(save_dir, "latest.pt"), state)
        if is_best:
            save_checkpoint(os.path.join(save_dir, "best.pt"), state)
            logger.error(
                "NEW BEST MODEL | monitor=val_{} | epoch={} | score={:.6f} | val_loss={:.6f}",
                selection_metric,
                epoch,
                best_score,
                best_loss,
            )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", val_loss, epoch)
        writer.add_scalar(f"metric/valid_{selection_metric}", val_score, epoch)
        if epoch_test_loss is not None:
            writer.add_scalar("loss/test_epochwise", epoch_test_loss, epoch)
            if selection_metric in epoch_test_metrics:
                writer.add_scalar(
                    f"metric/test_{selection_metric}_epochwise",
                    float(epoch_test_metrics[selection_metric]),
                    epoch,
                )
        log_epoch_summary(
            epoch,
            int(cfg["epochs"]),
            train_loss,
            val_loss,
            train_metrics,
            val_metrics,
            epoch_test_metrics,
        )
        append_epoch_metrics(
            os.path.join(save_dir, "epoch_metrics.csv"),
            epoch,
            train_loss,
            val_loss,
            best_loss,
            is_best,
            train_metrics,
            val_metrics,
            epoch_test_loss,
            epoch_test_metrics,
        )
        if patience > 0 and bad_epochs >= patience:
            logger.info(
                "EARLY STOP | epoch={} | best_epoch={} | best_val_{}={:.6f} | patience={}",
                epoch,
                best_epoch,
                selection_metric,
                best_score,
                patience,
            )
            break
        if args.max_steps > 0:
            break

    best_path = os.path.join(save_dir, "best.pt")
    if best_epoch <= 0 or not os.path.isfile(best_path):
        raise RuntimeError("Validation selection did not produce best.pt; test evaluation is forbidden.")
    try:
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        selected = torch.load(best_path, map_location="cpu")
    if selected.get("run_fingerprint") != provenance["run_fingerprint"]:
        raise ValueError("Selected checkpoint run fingerprint does not match this run.")
    if selected.get("split_hashes") != provenance["split_hashes"]:
        raise ValueError("Selected checkpoint split hashes do not match this run.")
    model.load_state_dict(selected["model_state_dict"], strict=True)
    if cfg["variant"] == "pretrained_frozen":
        model.set_encoder_frozen(True)
        assert_frozen_encoder_eval(model)

    # Reloading the validation-selected checkpoint keeps the reported result
    # independent of the diagnostic per-epoch test trajectory.
    test_loss, test_metrics, test_predictions = run_property_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        max_steps=args.max_steps,
        desc=f"Selected epoch {best_epoch} [Test]",
        collect_predictions=True,
        threshold=threshold,
        gradient_clip_norm=gradient_clip_norm,
        task_type=task_type,
    )
    test_evaluation_count += 1
    expected_test_evaluations = epochs_completed + 1 if test_every_epoch else 1
    if test_evaluation_count != expected_test_evaluations:
        raise RuntimeError(
            "Property test evaluation count mismatch: "
            f"expected {expected_test_evaluations}, got {test_evaluation_count}."
        )
    predictions_path = os.path.join(save_dir, "test_predictions.csv")
    write_prediction_csv(predictions_path, test_predictions, selected_epoch=best_epoch)

    writer.add_scalar("loss/test_once", test_loss, best_epoch)
    if selection_metric in test_metrics:
        writer.add_scalar(
            f"metric/test_{selection_metric}_once",
            float(test_metrics[selection_metric]),
            best_epoch,
        )
    writer.flush()
    writer.close()

    best_checkpoint_hash = checkpoint_sha256(best_path)
    result = {
        "dataset": dataset_name,
        "save_dir": save_dir,
        "variant": cfg["variant"],
        "seed": cfg["seed"],
        "task_type": task_type,
        "best_metric": f"val_{selection_metric}",
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_val_metrics": best_val_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_evaluation": test_evaluation,
        "test_evaluation_checkpoint": "best.pt",
        "test_evaluation_count": test_evaluation_count,
        "test_every_epoch": test_every_epoch,
        "epochwise_test_evaluation_count": epochwise_test_evaluation_count,
        "epochs_completed": epochs_completed,
        "test_used_for_checkpoint_selection": False,
        "predictions_csv": predictions_path,
        "max_steps": int(args.max_steps),
        "smoke_run": bool(args.max_steps > 0),
        "run_fingerprint": provenance["run_fingerprint"],
        "split_hashes": provenance["split_hashes"],
        "checkpoint_hashes": {
            "pretrain": provenance["pretrain_checkpoint_sha256"],
            "best": best_checkpoint_hash,
        },
        "pretrain_checkpoint_sha256": provenance["pretrain_checkpoint_sha256"],
        "best_checkpoint_sha256": best_checkpoint_hash,
        "manifest_sha256": provenance["manifest_sha256"],
        "molecule_store_sha256": provenance["molecule_store_sha256"],
        "descriptor_columns_consumed": [],
        "num_tasks": int(task_spec["num_tasks"]),
        "task_names": list(task_spec["task_names"]),
        "metric_aggregation": str(test_metrics.get("aggregation", "")),
    }
    write_metrics_json(os.path.join(save_dir, "final_metrics.json"), result)
    logger.info(
        "Final validation-selected checkpoint test | protocol={} | selected_epoch={} | "
        "loss={:.6f} | metrics={} | predictions={}",
        test_evaluation,
        best_epoch,
        test_loss,
        format_metric_payload(test_metrics),
        predictions_path,
    )
    print(format_metric_payload(result))


if __name__ == "__main__":
    main()
