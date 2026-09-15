"""Reaction classification finetuning entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "finetune_schneider.yaml"

import torch
from torch.optim import AdamW
from torch.utils.data import Subset

from hypermol.data.downstream import ReactionClassificationDataset
from hypermol.losses.downstream_loss import ReactionClassificationLoss
from hypermol.models.reaction_class_model import ReactionClassModel
from hypermol.training.downstream_common import build_pair_loader, run_downstream_epoch
from hypermol.utils.checkpoint import (
    capture_rng_state,
    checkpoint_sha256,
    load_checkpoint_if_available,
    restore_rng_state,
    save_checkpoint,
)
from hypermol.utils.config import load_yaml_config
from hypermol.utils.pretrained import (
    freeze_module_parameters,
    load_fusion_backbone_weights,
    load_molecular_encoder_weights,
    parameter_trainability_summary,
)
from hypermol.utils.runtime import build_summary_writer, get_device, set_seed
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


def _split_csv(data_cfg: dict, split: str) -> str:
    key = f"{split}_csv"
    if key in data_cfg:
        return data_cfg[key]
    filename = "valid.csv" if split == "valid" else f"{split}.csv"
    return os.path.join(data_cfg["data_root"], filename)


def sample_few_shot_dataset(dataset: ReactionClassificationDataset, samples_per_class: int, seed: int):
    """Return a train-only few-shot subset with up to N samples per reaction class."""
    samples_per_class = int(samples_per_class)
    if samples_per_class <= 0:
        logger.info("Reaction class train sampling: full dataset | train_size={}", len(dataset))
        return dataset

    rng = random.Random(seed)
    class_to_indices = {}
    for idx, raw_label in enumerate(dataset.df[dataset.label_column].tolist()):
        label = int(float(raw_label))
        class_to_indices.setdefault(label, []).append(idx)

    selected = []
    per_class_counts = {}
    for label in sorted(class_to_indices):
        indices = class_to_indices[label]
        chosen = rng.sample(indices, samples_per_class) if len(indices) > samples_per_class else list(indices)
        selected.extend(chosen)
        per_class_counts[label] = len(chosen)
    rng.shuffle(selected)

    logger.info(
        "Reaction class train sampling: few-shot | samples_per_class={} | train_size {} -> {} | classes={}",
        samples_per_class,
        len(dataset),
        len(selected),
        len(class_to_indices),
    )
    logger.info(
        "Few-shot class count range: min={} | max={}",
        min(per_class_counts.values()) if per_class_counts else 0,
        max(per_class_counts.values()) if per_class_counts else 0,
    )
    return Subset(dataset, selected)


def build_loaders(cfg: dict):
    data_cfg = cfg["data"]
    molecule_db = data_cfg.get("molecule_db", os.path.join(data_cfg["data_root"], "smiles.mdb"))
    datasets = {
        split: ReactionClassificationDataset(
            split_csv_path=_split_csv(data_cfg, split),
            molecule_lmdb_path=molecule_db,
            label_column=data_cfg.get("label_column", ""),
        )
        for split in ("train", "valid", "test")
    }
    num_classes = int(cfg.get("num_classes") or max(dataset.num_classes for dataset in datasets.values()))
    datasets["train"] = sample_few_shot_dataset(
        datasets["train"],
        samples_per_class=cfg.get("samples_per_class", -1),
        seed=cfg.get("seed", 42),
    )
    loaders = {
        "train": build_pair_loader(datasets["train"], cfg, task="reaction_class", shuffle=True),
        "valid": build_pair_loader(datasets["valid"], cfg, task="reaction_class", shuffle=False),
        "test": build_pair_loader(datasets["test"], cfg, task="reaction_class", shuffle=False),
    }
    return loaders, num_classes


def build_model(cfg: dict, num_classes: int) -> ReactionClassModel:
    model_cfg = cfg.get("model", {})
    return ReactionClassModel(
        num_classes=num_classes,
        model_cfg=model_cfg,
        reaction_repr_mode=cfg.get("reaction_repr_mode", "center"),
        center_source=cfg.get("center_source", "auto"),
        hidden_dim=model_cfg.get("head_hidden_dim"),
        dropout=model_cfg.get("dropout", 0.1),
    )


def get_reaction_class_early_stop_settings(cfg: dict) -> tuple[int, float]:
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    min_delta = float(cfg.get("early_stop_min_delta", 0.0) or 0.0)
    if patience < 0:
        raise ValueError("early_stop_patience must be >= 0.")
    if min_delta < 0:
        raise ValueError("early_stop_min_delta must be >= 0.")
    return patience, min_delta


def get_reaction_class_selection_score(val_metrics: dict) -> float:
    if "macro_f1" not in val_metrics:
        raise ValueError("Reaction-class validation metrics do not contain macro_f1; cannot select best model by val_macro_f1.")
    return float(val_metrics["macro_f1"])


def is_reaction_class_f1_improved(val_macro_f1: float, best_val_macro_f1: float, min_delta: float = 0.0) -> bool:
    return float(val_macro_f1) > float(best_val_macro_f1) + float(min_delta)


def get_reaction_class_test_every_epoch(cfg: dict) -> bool:
    """Return the required Schneider epoch-wise test policy.

    ``cfg`` is retained for call-site compatibility, but Schneider always
    evaluates the fixed test split after every completed epoch.  Test metrics
    are diagnostic only: validation macro-F1 remains the sole checkpoint
    selection signal.
    """

    del cfg
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume_path", default="")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="Override config samples_per_class. Few-shot keeps N samples per class; -1 uses the full training set.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config).raw
    # Schneider has a fixed epoch-wise test contract.  Normalize the resolved
    # config so old configs containing ``test_every_epoch: false`` cannot make
    # the recorded configuration disagree with the behavior of this runner.
    cfg["test_every_epoch"] = get_reaction_class_test_every_epoch(cfg)
    if args.samples_per_class is not None:
        cfg["samples_per_class"] = int(args.samples_per_class)
    run_name = cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg["save_dir"], run_name)
    log_path = setup_training_logger(save_dir)
    log_config(cfg, title="Reaction Class Configuration")
    write_json(os.path.join(save_dir, "config_resolved.json"), cfg)
    logger.info("Run directory: {}", save_dir)
    logger.info("Training log: {}", log_path)
    device = get_device(cfg.get("gpu", -1))
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False), device=device)
    logger.info("Device: {}", device)

    loaders, num_classes = build_loaders(cfg)
    logger.info("Auto inferred num_classes={}", num_classes)
    model = build_model(cfg, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    criterion = ReactionClassificationLoss().to(device)

    ckpt = load_checkpoint_if_available(args.resume_path, map_location="cpu")
    start_epoch = 1
    best_loss = float("inf")
    best_score = float("-inf")
    best_epoch = 0
    best_val_metrics = {}
    best_test_metrics = {}
    early_stop_bad_epochs = 0
    early_stop_patience, early_stop_min_delta = get_reaction_class_early_stop_settings(cfg)
    test_every_epoch = get_reaction_class_test_every_epoch(cfg)
    if ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", best_loss)
        best_score = float(ckpt.get("best_score", ckpt.get("best_val_macro_f1", best_score)))
        if best_score == float("-inf"):
            metric_source = ckpt.get("best_val_metrics") or ckpt.get("val_metrics") or {}
            if "macro_f1" in metric_source:
                best_score = float(metric_source["macro_f1"])
        best_epoch = int(ckpt.get("best_epoch", 0) or 0)
        best_val_metrics = ckpt.get("best_val_metrics", {}) or {}
        best_test_metrics = ckpt.get("best_test_metrics", {}) or {}
        early_stop_bad_epochs = int(ckpt.get("early_stop_bad_epochs", 0) or 0)
        if ckpt.get("rng_state"):
            restore_rng_state(ckpt["rng_state"])
        logger.info(
            "Resumed from {} | start_epoch={} | best_val_macro_f1={:.4f} | best_epoch={} | early_stop_bad_epochs={}",
            args.resume_path,
            start_epoch,
            best_score,
            best_epoch,
            early_stop_bad_epochs,
        )
    elif cfg.get("pretrain_path"):
        if getattr(model, "backbone_type", "") == "hypergraph":
            result = load_fusion_backbone_weights(model, cfg["pretrain_path"], map_location="cpu", strict=False)
            logger.info(
                "Loaded pretrained FusionBackbone from {} | loaded_keys={} | missing={} | unexpected={}",
                cfg["pretrain_path"],
                result["loaded_keys"],
                len(result["missing"]),
                len(result["unexpected"]),
            )
            freeze_module = model.backbone
            freeze_name = "FusionBackbone"
        else:
            result = load_molecular_encoder_weights(model, cfg["pretrain_path"], map_location="cpu", strict=False)
            logger.info(
                "Loaded pretrained MolecularEncoder from {} | loaded_keys={} | missing={} | unexpected={}",
                cfg["pretrain_path"],
                result["loaded_keys"],
                len(result["missing"]),
                len(result["unexpected"]),
            )
            freeze_module = model.backbone.encoder
            freeze_name = "MolecularEncoder"
        if cfg.get("freeze_pretrained", False):
            freeze_stats = freeze_module_parameters(freeze_module)
            train_stats = parameter_trainability_summary(model)
            logger.info(
                "Frozen pretrained {} | frozen_params={} | trainable_params={} | total_params={}",
                freeze_name,
                freeze_stats["frozen_params"],
                train_stats["trainable_params"],
                train_stats["total_params"],
            )
        else:
            logger.info("Pretrained {} remains trainable.", freeze_name)

    if early_stop_patience > 0:
        logger.info(
            "Early stopping enabled | monitor=val_macro_f1 | mode=max | patience={} | min_delta={:.6g}",
            early_stop_patience,
            early_stop_min_delta,
        )
    else:
        logger.info("Early stopping disabled | set early_stop_patience > 0 to enable.")

    writer = build_summary_writer(os.path.join(save_dir, "tb"))
    last_test_metrics = {}

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        logger.info("[Epoch {}] current lrs: {}", epoch, format_lrs(optimizer))
        train_loss, train_metrics = run_downstream_epoch(
            "reaction_class",
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            max_steps=args.max_steps,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Train]",
        )
        val_loss, val_metrics = run_downstream_epoch(
            "reaction_class",
            model,
            loaders["valid"],
            criterion,
            device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"Epoch {epoch}/{cfg['epochs']} [Valid]",
        )
        test_loss = None
        test_metrics = {}
        if test_every_epoch:
            test_loss, test_metrics = run_downstream_epoch(
                "reaction_class",
                model,
                loaders["test"],
                criterion,
                device,
                optimizer=None,
                max_steps=args.max_steps,
                desc=f"Epoch {epoch}/{cfg['epochs']} [Test]",
            )
            last_test_metrics = test_metrics
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/valid", val_loss, epoch)
        if test_loss is not None:
            writer.add_scalar("loss/test", test_loss, epoch)
        if "macro_f1" in val_metrics:
            writer.add_scalar("metric/valid_macro_f1", float(val_metrics["macro_f1"]), epoch)
        if "macro_f1" in test_metrics:
            writer.add_scalar("metric/test_macro_f1", float(test_metrics["macro_f1"]), epoch)
        if "accuracy" in val_metrics:
            writer.add_scalar("metric/valid_accuracy", float(val_metrics["accuracy"]), epoch)
        if "accuracy" in test_metrics:
            writer.add_scalar("metric/test_accuracy", float(test_metrics["accuracy"]), epoch)
        val_macro_f1 = get_reaction_class_selection_score(val_metrics)
        is_best = is_reaction_class_f1_improved(val_macro_f1, best_score, early_stop_min_delta)
        if is_best:
            best_score = val_macro_f1
            best_loss = val_loss
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics if test_every_epoch else {}
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_score": best_score,
            "best_metric": "val_macro_f1",
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "best_test_metrics": best_test_metrics,
            "early_stop_bad_epochs": early_stop_bad_epochs,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stop_monitor": "val_macro_f1",
            "config": cfg,
            "num_classes": num_classes,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "test_loss": test_loss,
            "rng_state": capture_rng_state(),
        }
        save_checkpoint(os.path.join(save_dir, "latest.pt"), state)
        if is_best:
            logger.error(
                "NEW BEST MODEL | monitor=val_macro_f1 | mode=max | epoch={} | score={:.4f} | val_loss={:.4f}",
                epoch,
                best_score,
                float(val_loss),
            )
            if test_every_epoch:
                log_best_model(epoch, val_loss, val_metrics, test_loss, test_metrics)
            save_checkpoint(os.path.join(save_dir, "best.pt"), state)
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
                "Early stopping status | monitor=val_macro_f1 | bad_epochs={}/{} | best_val_macro_f1={:.4f} | best_epoch={}",
                early_stop_bad_epochs,
                early_stop_patience,
                best_score,
                best_epoch,
            )
            if early_stop_bad_epochs >= early_stop_patience:
                logger.info(
                    "EARLY STOP | monitor=val_macro_f1 | epoch={} | best_epoch={} | best_val_macro_f1={:.4f} | patience={}",
                    epoch,
                    best_epoch,
                    best_score,
                    early_stop_patience,
                )
                break
        if args.max_steps > 0:
            break

    if not test_every_epoch:
        best_path = os.path.join(save_dir, "best.pt")
        selected = load_checkpoint_if_available(best_path, map_location="cpu")
        if not selected:
            raise RuntimeError(f"Validation-only selection did not produce a checkpoint: {best_path}")
        model.load_state_dict(selected["model_state_dict"])
        test_loss, best_test_metrics = run_downstream_epoch(
            "reaction_class",
            model,
            loaders["test"],
            criterion,
            device,
            optimizer=None,
            max_steps=args.max_steps,
            desc=f"Selected epoch {best_epoch} [Test]",
        )
        last_test_metrics = best_test_metrics
        logger.info(
            "Final test | checkpoint={} | epoch={} | loss={:.4f} | metrics={}",
            best_path,
            best_epoch,
            float(test_loss),
            format_metric_payload(best_test_metrics),
        )

    writer.flush()
    writer.close()
    result = {
        "save_dir": save_dir,
        "best_metric": "val_macro_f1",
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "last_test_metrics": last_test_metrics,
        "test_metrics": best_test_metrics,
        "test_evaluation": "every_epoch_legacy" if test_every_epoch else "best_checkpoint_once",
        "test_evaluation_checkpoint": "best.pt",
        "test_every_epoch": test_every_epoch,
        "checkpoint_selection_source": "validation_macro_f1_only",
        "test_metrics_used_for_checkpoint_selection": False,
        "max_steps": int(args.max_steps),
        "smoke_run": bool(args.max_steps > 0),
        "best_checkpoint_sha256": checkpoint_sha256(os.path.join(save_dir, "best.pt")),
    }
    write_metrics_json(os.path.join(save_dir, "final_metrics.json"), result)
    logger.info("Final best test metrics: {}", format_metric_payload(best_test_metrics))
    print(format_metric_payload(result))


if __name__ == "__main__":
    main()
