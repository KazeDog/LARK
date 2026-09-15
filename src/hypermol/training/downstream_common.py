from __future__ import annotations

import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hypermol.data.downstream import CONDITION_FIELDS
from hypermol.data.downstream import ReactionPairCollator
from hypermol.utils.metrics import (
    compute_condition_metrics,
    compute_reaction_class_metrics,
    compute_yield_metrics,
)
from hypermol.utils.runtime import move_to_device


def build_pair_loader(dataset, cfg: Dict, task: str, shuffle: bool) -> DataLoader:
    model_cfg = cfg.get("model", {}) or {}
    common = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 0)),
        "collate_fn": ReactionPairCollator(
            task=task,
            aux_input_mode=str(model_cfg.get("aux_input_mode", "edge")),
        ),
        "pin_memory": torch.cuda.is_available(),
    }
    if common["num_workers"] > 0:
        common["prefetch_factor"] = int(cfg.get("prefetch_factor", 4))
        common["persistent_workers"] = True
    return DataLoader(dataset, shuffle=shuffle, **common)


def run_downstream_epoch(
    task: str,
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_steps: int = 0,
    teacher_forcing: bool = False,
    condition_beams=(1, 3, 1, 5, 1),
    desc: str | None = None,
) -> tuple[float, Dict]:
    is_train = optimizer is not None
    model.train(is_train)
    logs = []
    disable_progress = os.environ.get("HYPERMOL_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
    iterator = tqdm(loader, desc=desc or ("Train" if is_train else "Eval"), leave=False, disable=disable_progress)
    for step, batch in enumerate(iterator, start=1):
        if not batch:
            continue
        batch = move_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        if task == "condition":
            outputs = model(batch, teacher_forcing=is_train and teacher_forcing)
        else:
            outputs = model(batch)
        loss, log = criterion(outputs, batch, is_train=is_train)
        if task == "condition" and not is_train:
            if "fused" in outputs and hasattr(model, "inference_from_fused"):
                topk_predictions = model.inference_from_fused(outputs["fused"].detach(), beams=condition_beams)
            else:
                topk_predictions = model.inference(batch, beams=condition_beams)
            log["topk_preds"] = topk_predictions.detach().cpu()
            log["topk_targets"] = torch.stack([batch["labels"][slot] for slot in CONDITION_FIELDS], dim=-1).detach().cpu()
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        logs.append(log)
        iterator.set_postfix({"loss": f"{log.get('total_loss', 0.0):.4f}"})
        if max_steps > 0 and step >= max_steps:
            break

    if not logs:
        return 0.0, {}
    loss = sum(log.get("total_loss", 0.0) for log in logs) / len(logs)
    metrics = {"avg_total_loss": loss}
    if not is_train:
        metrics.update(_aggregate_eval_metrics(task, logs))
    return loss, metrics


def _aggregate_eval_metrics(task: str, logs: list[Dict]) -> Dict:
    if task == "condition":
        slot_preds = {}
        slot_targets = {}
        first = logs[0]
        for slot in first.get("slot_preds", {}):
            slot_preds[slot] = torch.cat([log["slot_preds"][slot] for log in logs]).numpy()
            slot_targets[slot] = torch.cat([log["slot_targets"][slot] for log in logs]).numpy()
        out = compute_condition_metrics(slot_preds, slot_targets)
        if any("topk_preds" in log for log in logs):
            topk_preds = torch.cat([log["topk_preds"] for log in logs if "topk_preds" in log], dim=0)
            topk_targets = torch.cat([log["topk_targets"] for log in logs if "topk_targets" in log], dim=0)
            valid = torch.all(topk_targets >= 0, dim=-1)
            if torch.any(valid):
                hits = torch.any(torch.all(topk_preds[valid] == topk_targets[valid].unsqueeze(1), dim=-1), dim=-1)
                out["topk_exact_acc"] = float(hits.float().mean().item())
                out["topk_candidates"] = int(topk_preds.shape[1])
        return out

    if task in {"yield", "selectivity"}:
        preds = torch.cat([log["preds"] for log in logs]).numpy()
        targets = torch.cat([log["targets"] for log in logs]).numpy()
        return compute_yield_metrics(preds, targets)

    preds = torch.cat([log["preds"] for log in logs]).numpy()
    targets = torch.cat([log["targets"] for log in logs]).numpy()
    probs = torch.cat([log["probs"] for log in logs]).numpy() if any("probs" in log for log in logs) else None
    return compute_reaction_class_metrics(preds, targets, probs)
