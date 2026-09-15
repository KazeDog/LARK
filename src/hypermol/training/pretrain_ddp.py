"""Distributed pretraining entrypoint.

Launch with torchrun, for example:

    PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
      -m hypermol.training.pretrain_ddp

This file intentionally lives next to the single-GPU ``pretrain.py`` entrypoint
instead of replacing it.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from collections.abc import Iterator, Sized
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pretrain.yaml"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from hypermol.data.collator import ReactionCollator
from hypermol.training.pretrain import (
    _batch_graph_view_count,
    build_protocol_metadata,
    build_pretrain_datasets,
    build_model,
    build_pretrain_loss,
    checkpoint_protocol_metadata,
    get_early_stop_settings,
    get_pretrain_tasks_from_config,
    get_selection_metric,
    get_selection_mode,
    get_selection_value,
    is_selection_improved,
    load_compatible_initial_weights,
    maybe_save_periodic_checkpoint,
    maybe_run_be_audit,
    validate_resume_data_contract,
    validate_resume_protocol,
)
from hypermol.utils.checkpoint import load_checkpoint_if_available, save_checkpoint
from hypermol.utils.config import load_yaml_config
from hypermol.utils.metrics import reduce_metrics
from hypermol.utils.training_budget import (
    resolve_training_budget,
    validate_resume_training_budget,
)
from hypermol.utils.runtime import build_summary_writer, move_to_device, set_seed
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


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else 1


def is_rank0() -> bool:
    return get_rank() == 0


def setup_distributed(preferred_gpu: int = -1) -> tuple[int, int, int, torch.device]:
    """Initialize torch.distributed from torchrun environment variables."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP pretraining requires CUDA when WORLD_SIZE > 1.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available() and preferred_gpu >= 0:
            torch.cuda.set_device(preferred_gpu)
            device = torch.device("cuda", preferred_gpu)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_dist_initialized():
        return value
    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_gather_objects(value: Any) -> list[Any]:
    if not is_dist_initialized():
        return [value]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def strip_large_metric_arrays(log: dict, collect_fp_auc: bool = False) -> dict:
    """Avoid all-gathering large fingerprint probability arrays by default."""

    if collect_fp_auc:
        return log
    out = dict(log)
    out.pop("fp_probs", None)
    out.pop("fp_targets", None)
    return out


def flatten_gathered_logs(gathered: list[Any]) -> list[dict]:
    logs: list[dict] = []
    for item in gathered:
        if not item:
            continue
        logs.extend(item)
    return logs


def replace_pair_mean_component(
    total_loss: torch.Tensor,
    local_pair_mean: torch.Tensor | None,
    local_pair_weight: float,
    global_pair_weight: float | torch.Tensor,
    world_size: int,
    component_weight: float,
) -> torch.Tensor:
    """Replace one local mean by its globally weighted DDP contribution."""

    if local_pair_mean is None:
        return total_loss
    local_weight = local_pair_mean.new_tensor(float(local_pair_weight))
    global_weight = torch.as_tensor(
        global_pair_weight,
        device=local_pair_mean.device,
        dtype=local_pair_mean.dtype,
    ).detach()
    if float(global_weight.item()) <= 0.0:
        return total_loss
    scale = int(world_size) * local_weight / global_weight
    return total_loss + float(component_weight) * (scale - 1.0) * local_pair_mean


def replace_local_pair_mean_with_global_ddp_mean(
    total_loss: torch.Tensor,
    local_pair_mean: torch.Tensor | None,
    local_pair_weight: float,
    component_weight: float,
) -> torch.Tensor:
    """Make a pair-normalized component match one global effective batch.

    PyTorch DDP averages gradients from all ranks.  Backpropagating each
    rank's ``numerator / local_denominator`` therefore gives every rank equal
    influence even when reaction canvases have very different pair counts.
    Replacing the local component with

    ``world_size * local_numerator / global_denominator``

    makes the DDP-averaged gradient exactly
    ``sum(local_numerator) / sum(local_denominator)``.  The denominator is a
    detached count and intentionally does not participate in autograd.
    """

    if local_pair_mean is None:
        return total_loss
    local_weight = local_pair_mean.new_tensor(float(local_pair_weight))
    global_weight = local_weight.detach().clone()
    if is_dist_initialized():
        dist.all_reduce(global_weight, op=dist.ReduceOp.SUM)
    return replace_pair_mean_component(
        total_loss=total_loss,
        local_pair_mean=local_pair_mean,
        local_pair_weight=float(local_pair_weight),
        global_pair_weight=global_weight,
        world_size=get_world_size(),
        component_weight=component_weight,
    )


class ExactDistributedSampler(Sampler[int]):
    """Shard evaluation rows without DistributedSampler padding duplicates."""

    def __init__(self, dataset: Sized, num_replicas: int | None = None, rank: int | None = None) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas if num_replicas is not None else get_world_size())
        self.rank = int(rank if rank is not None else get_rank())
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)


def build_rank_synchronized_pretrain_datasets(cfg: dict):
    """Build a directional split once on rank 0 and replay source indices elsewhere."""

    integrity_cfg = cfg.get("split_integrity", {}) or {}
    group_by_input_side = bool(
        integrity_cfg.get("enabled", False)
        and integrity_cfg.get("group_by_input_side", False)
    )
    if not group_by_input_side:
        return build_pretrain_datasets(cfg)

    datasets = None
    contract = None
    if is_rank0():
        train_dataset, valid_dataset, test_dataset, contract = build_pretrain_datasets(
            cfg,
            return_source_index_contract=True,
        )
        datasets = (train_dataset, valid_dataset, test_dataset)
        if contract is None:
            raise RuntimeError("Rank 0 did not produce a directional split source-index contract.")
    contract = broadcast_object(contract, src=0)
    if not is_rank0():
        datasets = build_pretrain_datasets(cfg, source_index_contract=contract)
    return datasets


def build_distributed_loaders(cfg: dict):
    train_dataset, valid_dataset, test_dataset = build_rank_synchronized_pretrain_datasets(cfg)
    budget_cfg = cfg.get("training_budget", {}) or {}
    require_full_train_batches = bool(
        budget_cfg.get("require_full_train_batches", False)
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=require_full_train_batches,
        seed=int(cfg.get("seed", 42)),
    )
    valid_sampler = ExactDistributedSampler(
        valid_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )
    test_sampler = ExactDistributedSampler(
        test_dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )

    collator = ReactionCollator(mode="pretrain", task="pretrain")
    common = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 0),
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
    }
    if common["num_workers"] > 0:
        common["prefetch_factor"] = cfg.get("prefetch_factor", 4)
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=require_full_train_batches,
        **common,
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, **common)
    test_loader = DataLoader(test_dataset, sampler=test_sampler, **common)
    return train_loader, valid_loader, test_loader, train_sampler


def load_model_state_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    """Load checkpoints saved from plain modules or DDP-wrapped modules."""

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass
    stripped = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def gather_epoch_metrics(local_logs: list[dict]) -> dict:
    gathered = all_gather_objects(local_logs)
    flattened = flatten_gathered_logs(gathered)
    if not flattened:
        raise RuntimeError("Distributed epoch produced no valid batches on any rank.")
    if not is_rank0():
        return {}
    return reduce_metrics(flattened)


def run_epoch_ddp(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_steps: int = 0,
    desc: str | None = None,
    collect_fp_auc: bool = False,
    expected_local_graph_views: int = 0,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)
    forward_model = model if is_train or not isinstance(model, DDP) else model.module
    if not is_train and is_dist_initialized():
        # Exact evaluation shards may have unequal iteration counts, so eval
        # bypasses DDP.forward. Synchronize BatchNorm/other buffers once first.
        for buffer in forward_model.buffers():
            dist.broadcast(buffer, src=0)
    logs = []
    processed_batches = 0
    optimizer_steps = 0
    local_graph_views = 0
    iterator = loader
    if is_rank0():
        iterator = tqdm(loader, desc=desc or ("Train" if is_train else "Eval"), leave=False)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(iterator, start=1):
            if is_train and is_dist_initialized():
                valid_flag = torch.tensor(
                    1 if batch else 0,
                    device=device,
                    dtype=torch.int32,
                )
                dist.all_reduce(valid_flag, op=dist.ReduceOp.MIN)
                if int(valid_flag.item()) == 0:
                    raise RuntimeError(
                        "At least one DDP rank received an empty training batch; all ranks abort synchronously."
                    )
            if not batch:
                if is_train:
                    raise RuntimeError(
                        "DDP received an empty training batch. Clean or filter the dataset before distributed training."
                    )
                continue
            batch_graph_views = _batch_graph_view_count(batch)
            if is_train and expected_local_graph_views > 0:
                minimum_views = torch.tensor(batch_graph_views, device=device, dtype=torch.int64)
                maximum_views = minimum_views.clone()
                if is_dist_initialized():
                    dist.all_reduce(minimum_views, op=dist.ReduceOp.MIN)
                    dist.all_reduce(maximum_views, op=dist.ReduceOp.MAX)
                if (
                    int(minimum_views.item()) != expected_local_graph_views
                    or int(maximum_views.item()) != expected_local_graph_views
                ):
                    raise RuntimeError(
                        "Formal DDP ablation budget requires a full, fixed-size local "
                        f"graph-view batch on every rank: expected={expected_local_graph_views}, "
                        f"observed_rank_range=[{int(minimum_views.item())}, "
                        f"{int(maximum_views.item())}]."
                    )
            batch = move_to_device(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(batch)
            loss, log = criterion(outputs, batch, is_train=is_train)
            local_rmat_mean = log.pop("_rmat_loss_tensor", None)
            if is_train:
                loss = replace_local_pair_mean_with_global_ddp_mean(
                    total_loss=loss,
                    local_pair_mean=local_rmat_mean,
                    local_pair_weight=float(log.get("rmat_loss_weight", 0.0)),
                    component_weight=float(criterion.rmat_weight),
                )
                # Report the same effective objective whose gradients DDP will
                # average, rather than the pre-rescaling rank-local mean.
                effective_loss = loss.detach().clone()
                if is_dist_initialized():
                    dist.all_reduce(effective_loss, op=dist.ReduceOp.SUM)
                    effective_loss /= get_world_size()
                effective_loss_value = float(effective_loss.item())
                log["total_loss"] = effective_loss_value
                log["total_loss_sum"] = effective_loss_value * int(log.get("sample_size", 1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                local_graph_views += batch_graph_views
            logs.append(strip_large_metric_arrays(log, collect_fp_auc=collect_fp_auc))
            processed_batches += 1
            if is_rank0() and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"loss": f"{log.get('total_loss', 0.0):.4f}"})
            if max_steps > 0 and processed_batches >= max_steps:
                break

    metrics = gather_epoch_metrics(logs)
    metrics["processed_batches"] = processed_batches
    metrics["optimizer_steps"] = optimizer_steps
    metrics["local_graph_views"] = local_graph_views
    metrics["global_graph_views"] = local_graph_views * get_world_size()
    loss_value = metrics.get("avg_total_loss", 0.0) if is_rank0() else 0.0
    return loss_value, metrics


def state_dict_for_checkpoint(model: torch.nn.Module) -> dict:
    module = model.module if isinstance(model, DDP) else model
    return module.state_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume_path", default="")
    parser.add_argument(
        "--init_checkpoint",
        default="",
        help="Warm-start compatible model weights only; optimizer, epoch, and best-loss state are reset.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Override preferred gpu for single-process fallback.")
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()
    if args.resume_path and args.init_checkpoint:
        parser.error("--resume_path and --init_checkpoint are mutually exclusive.")

    cfg = load_yaml_config(args.config).raw
    if args.max_steps > 0 and cfg.get("training_budget"):
        parser.error(
            "--max_steps is a one-epoch smoke limit and cannot be combined with "
            "config.training_budget; edit max_optimizer_steps instead."
        )
    if args.gpu is not None:
        cfg["gpu"] = int(args.gpu)
    active_tasks = get_pretrain_tasks_from_config(cfg)
    cfg["active_pretrain_tasks"] = list(active_tasks)
    protocol_metadata = build_protocol_metadata(cfg)
    cfg["protocol_metadata"] = protocol_metadata
    directional_protocol = bool(protocol_metadata["directional"])
    ckpt = load_checkpoint_if_available(args.resume_path, map_location="cpu")
    init_ckpt = load_checkpoint_if_available(args.init_checkpoint, map_location="cpu")
    if args.resume_path and ckpt is None:
        raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_path}")
    if args.init_checkpoint and init_ckpt is None:
        raise FileNotFoundError(f"Initialization checkpoint not found: {args.init_checkpoint}")
    if ckpt:
        validate_resume_protocol(cfg, ckpt)
    rank, world_size, local_rank, device = setup_distributed(preferred_gpu=int(cfg.get("gpu", -1)))
    training_budget_runtime = resolve_training_budget(cfg, world_size=world_size)
    cfg["training_budget_runtime"] = training_budget_runtime
    if ckpt:
        training_budget_runtime = validate_resume_training_budget(
            cfg,
            ckpt,
            world_size=world_size,
        )
        cfg["training_budget_runtime"] = training_budget_runtime
    if not is_rank0() and hasattr(logger, "remove"):
        logger.remove()

    try:
        save_dir = None
        if is_rank0():
            if args.resume_path:
                save_dir = os.path.dirname(os.path.abspath(args.resume_path))
            else:
                run_name = cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join(cfg["save_dir"], run_name)
        save_dir = broadcast_object(save_dir, src=0)
        run_name = os.path.basename(save_dir)

        set_seed(int(cfg.get("seed", 42)) + rank, deterministic=cfg.get("deterministic", False), device=device)

        train_loader, val_loader, test_loader, train_sampler = build_distributed_loaders(cfg)
        if ckpt:
            validate_resume_data_contract(cfg, ckpt)
        if is_rank0():
            log_path = setup_training_logger(save_dir)
            log_config(cfg, title="DDP Pretrain Configuration")
            write_json(os.path.join(save_dir, "config_resolved.json"), cfg)
            logger.info("Run directory: {}", save_dir)
            logger.info("Training log: {}", log_path)
            logger.info(
                "DDP world_size={} | local_rank={} | device={} | per_gpu_batch_size={} | effective_batch_size={}",
                world_size,
                local_rank,
                device,
                cfg["batch_size"],
                cfg["batch_size"] * world_size,
            )
            logger.info("Active pretraining tasks: {}", ", ".join(active_tasks))
            logger.info("Pretraining protocol: {}", protocol_metadata)
            logger.info("Training budget: {}", training_budget_runtime)
            maybe_run_be_audit(cfg, save_dir)
        if is_dist_initialized():
            dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
        model = build_model(cfg).to(device)
        criterion = build_pretrain_loss(cfg).to(device)
        early_stop_patience, early_stop_min_delta = get_early_stop_settings(cfg)
        selection_metric = get_selection_metric(cfg)
        selection_mode = get_selection_mode(cfg)
        if is_rank0():
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
            load_model_state_flexible(model, ckpt["model_state_dict"])
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
        elif init_ckpt:
            init_state = init_ckpt.get("model_state_dict", init_ckpt)
            load_report = load_compatible_initial_weights(model, init_state)
            initialization_metadata = {
                "mode": "warm_start",
                "source": os.path.abspath(args.init_checkpoint),
                "source_protocol": checkpoint_protocol_metadata(init_ckpt),
                "load_report": load_report,
            }
            if is_rank0():
                logger.info("Warm-started model weights: {}", initialization_metadata)

        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=bool(cfg.get("ddp_find_unused_parameters", True)),
            )

        optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        if ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
            if is_rank0():
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

        writer = build_summary_writer(os.path.join(save_dir, "tb")) if is_rank0() else None
        last_test_metrics = {}
        last_test_loss = 0.0
        collect_fp_auc = bool(cfg.get("ddp_collect_fp_auc", False))

        for epoch in range(start_epoch, cfg["epochs"] + 1):
            train_step_limit = int(args.max_steps)
            if training_budget_runtime["enabled"]:
                remaining_steps = (
                    training_budget_runtime["max_optimizer_steps"] - global_optimizer_step
                )
                if remaining_steps <= 0:
                    break
                train_step_limit = remaining_steps
            train_sampler.set_epoch(epoch)
            if is_rank0():
                logger.info("[Epoch {}] current lrs: {}", epoch, format_lrs(optimizer))

            train_loss, train_metrics = run_epoch_ddp(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                max_steps=train_step_limit,
                desc=f"Epoch {epoch}/{cfg['epochs']} [Train-DDP]",
                collect_fp_auc=collect_fp_auc,
                expected_local_graph_views=(
                    training_budget_runtime["local_graph_views_per_step"]
                    if training_budget_runtime["enforce_runtime_graph_views_per_step"]
                    else 0
                ),
            )
            epoch_optimizer_steps = int(train_metrics.get("optimizer_steps", 0))
            if epoch_optimizer_steps <= 0:
                raise RuntimeError("DDP training epoch completed without an optimizer step.")
            global_optimizer_step += epoch_optimizer_steps
            cumulative_graph_views += int(train_metrics.get("global_graph_views", 0))
            train_metrics["global_optimizer_step"] = global_optimizer_step
            train_metrics["cumulative_graph_views"] = cumulative_graph_views
            val_loss, val_metrics = run_epoch_ddp(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                max_steps=args.max_steps,
                desc=f"Epoch {epoch}/{cfg['epochs']} [Valid-DDP]",
                collect_fp_auc=collect_fp_auc,
            )
            if directional_protocol:
                test_loss, test_metrics = None, None
            else:
                test_loss, test_metrics = run_epoch_ddp(
                    model,
                    test_loader,
                    criterion,
                    device,
                    optimizer=None,
                    max_steps=args.max_steps,
                    desc=f"Epoch {epoch}/{cfg['epochs']} [Test-DDP]",
                    collect_fp_auc=collect_fp_auc,
                )

            if is_rank0():
                selection_value = get_selection_value(cfg, val_loss, val_metrics)
                if not directional_protocol:
                    last_test_metrics = test_metrics
                    last_test_loss = test_loss
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/valid", val_loss, epoch)
                if not directional_protocol:
                    writer.add_scalar("loss/test", test_loss, epoch)
                is_best = is_selection_improved(
                    selection_value, best_loss, selection_mode, early_stop_min_delta
                )
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
                    "model_state_dict": state_dict_for_checkpoint(model),
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
                    "world_size": world_size,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "test_loss": test_loss,
                    "test_evaluated": not directional_protocol,
                }
                save_checkpoint(os.path.join(save_dir, "latest.pt"), state)
                reached_training_budget = bool(
                    training_budget_runtime["enabled"]
                    and global_optimizer_step
                    >= training_budget_runtime["max_optimizer_steps"]
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
                periodic_path = maybe_save_periodic_checkpoint(
                    cfg,
                    save_dir,
                    epoch,
                    state,
                )
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
                should_stop = False
                if early_stop_patience > 0:
                    logger.info(
                        "Early stopping status | monitor={} | bad_epochs={}/{} | best_value={:.4f}",
                        selection_metric,
                        early_stop_bad_epochs,
                        early_stop_patience,
                        best_loss,
                    )
                    if early_stop_bad_epochs >= early_stop_patience:
                        should_stop = True
                        logger.info(
                            "EARLY STOP | monitor={} | epoch={} | best_value={:.4f} | patience={}",
                            selection_metric,
                            epoch,
                            best_loss,
                            early_stop_patience,
                        )
            else:
                should_stop = False
                reached_training_budget = bool(
                    training_budget_runtime["enabled"]
                    and global_optimizer_step
                    >= training_budget_runtime["max_optimizer_steps"]
                )

            if reached_training_budget and is_rank0():
                logger.info(
                    "TRAINING BUDGET REACHED | optimizer_steps={} | graph_views={}",
                    global_optimizer_step,
                    cumulative_graph_views,
                )
            should_stop = bool(should_stop or reached_training_budget)

            should_stop = broadcast_object(should_stop, src=0)
            if should_stop:
                break
            if args.max_steps > 0:
                break

        if training_budget_runtime["enabled"] and training_budget_runtime["require_exact_budget"]:
            expected_steps = training_budget_runtime["max_optimizer_steps"]
            expected_views = training_budget_runtime["max_graph_views"]
            if global_optimizer_step != expected_steps or cumulative_graph_views != expected_views:
                raise RuntimeError(
                    "Formal DDP pretraining budget was not completed exactly: "
                    f"steps={global_optimizer_step}/{expected_steps}, "
                    f"graph_views={cumulative_graph_views}/{expected_views}."
                )

        if directional_protocol:
            if is_dist_initialized():
                dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)
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
            target_model = model.module if isinstance(model, DDP) else model
            load_model_state_flexible(target_model, evaluation_ckpt["model_state_dict"])
            selected_epoch = int(evaluation_ckpt.get("epoch", 0))
            last_test_loss, last_test_metrics = run_epoch_ddp(
                model,
                test_loader,
                criterion,
                device,
                optimizer=None,
                max_steps=args.max_steps,
                desc=f"Selected epoch {selected_epoch} [Test-DDP]",
                collect_fp_auc=collect_fp_auc,
            )
            if is_rank0():
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

        if is_rank0():
            writer.flush()
            writer.close()
            result = {
                "save_dir": save_dir,
                "best_loss": best_loss,
                "best_selection_metric": selection_metric,
                "best_selection_mode": selection_mode,
                "best_selection_value": best_loss,
                "test_loss": last_test_loss,
                "test_metrics": last_test_metrics,
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
            logger.info("Final test metrics: {}", format_metric_payload(last_test_metrics))
            print(format_metric_payload(result))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
