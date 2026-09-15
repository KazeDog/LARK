from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypermol.data.downstream import CONDITION_FIELDS


class ReactionClassificationLoss(nn.Module):
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict, is_train: bool = True) -> Tuple[torch.Tensor, Dict]:
        labels = batch["labels"]
        loss = F.cross_entropy(outputs["logits"], labels)
        preds = torch.argmax(outputs["logits"], dim=-1)
        probs = torch.softmax(outputs["logits"], dim=-1)
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "preds": preds.detach().cpu(),
            "probs": probs.detach().cpu(),
            "targets": labels.detach().cpu(),
        }


class ConditionPredictionLoss(nn.Module):
    def __init__(
        self,
        class_weight: float = 1.0,
        reg_weight: float = 0.0,
        label_smoothing: float = 0.0,
        regression_targets=(),
        slot_class_weights: Dict[str, torch.Tensor] | None = None,
        slot_loss_weights: Dict[str, float] | None = None,
        expose_ddp_components: bool = False,
    ):
        super().__init__()
        self.class_weight = float(class_weight)
        self.reg_weight = float(reg_weight)
        self.label_smoothing = float(label_smoothing)
        self.expose_ddp_components = bool(expose_ddp_components)
        self.regression_targets = tuple(str(key) for key in regression_targets)
        if len(set(self.regression_targets)) != len(self.regression_targets):
            raise ValueError(f"Condition regression targets must be unique: {self.regression_targets}")
        unsupported_regression = set(self.regression_targets) - {"temperature", "time"}
        if unsupported_regression:
            raise ValueError(f"Unsupported condition regression targets: {sorted(unsupported_regression)}")
        self.slot_loss_weights = {
            slot: float((slot_loss_weights or {}).get(slot, 1.0)) for slot in CONDITION_FIELDS
        }
        if any(value <= 0 for value in self.slot_loss_weights.values()):
            raise ValueError("Condition slot loss weights must be positive.")
        self._slot_class_weight_buffers: Dict[str, str] = {}
        for slot, values in (slot_class_weights or {}).items():
            if slot not in CONDITION_FIELDS:
                raise ValueError(f"Unsupported condition slot class weight: {slot}")
            tensor = torch.as_tensor(values, dtype=torch.float32)
            if tensor.ndim != 1 or not torch.isfinite(tensor).all() or torch.any(tensor <= 0):
                raise ValueError(f"Class weights for {slot} must be a finite positive vector.")
            buffer_name = f"_class_weights_{slot}"
            self.register_buffer(buffer_name, tensor)
            self._slot_class_weight_buffers[slot] = buffer_name

    def _slot_loss(
        self,
        slot: str,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target >= 0
        if not torch.any(valid):
            return logits.sum() * 0.0, logits.new_tensor(0.0)
        weight = None
        if slot in self._slot_class_weight_buffers:
            weight = getattr(self, self._slot_class_weight_buffers[slot])
            if int(weight.numel()) != int(logits.shape[-1]):
                raise ValueError(
                    f"Class weight size for {slot} does not match logits: {weight.numel()} != {logits.shape[-1]}"
                )
        loss = F.cross_entropy(
            logits[valid],
            target[valid],
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        if weight is None:
            denominator = valid.sum().to(device=logits.device, dtype=logits.dtype)
        else:
            denominator = weight.index_select(0, target[valid]).sum().to(dtype=logits.dtype)
        return loss, denominator.detach()

    def forward(self, outputs: Dict, batch: Dict, is_train: bool = True) -> Tuple[torch.Tensor, Dict]:
        labels = batch["labels"]
        class_results = {
            slot: self._slot_loss(slot, outputs["class_logits"][slot], labels[slot])
            for slot in CONDITION_FIELDS
        }
        class_losses = {slot: values[0] for slot, values in class_results.items()}
        class_denominators = {slot: values[1] for slot, values in class_results.items()}
        slot_weight_sum = sum(self.slot_loss_weights.values())
        class_loss = sum(
            class_losses[slot] * self.slot_loss_weights[slot] for slot in CONDITION_FIELDS
        ) / slot_weight_sum

        reg_loss = class_loss.new_tensor(0.0)
        reg_terms = 0
        regression_losses = {}
        regression_denominators = {}
        for key in self.regression_targets:
            if key not in outputs.get("regression", {}):
                raise KeyError(f"Condition model did not return enabled regression target {key!r}.")
            mask = batch["reg_masks"][key]
            regression_denominators[key] = mask.sum().to(
                device=outputs["regression"][key].device,
                dtype=outputs["regression"][key].dtype,
            ).detach()
            if torch.any(mask):
                regression_losses[key] = F.mse_loss(
                    outputs["regression"][key][mask],
                    batch["reg_targets"][key][mask],
                )
                reg_loss = reg_loss + regression_losses[key]
                reg_terms += 1
            else:
                regression_losses[key] = outputs["regression"][key].sum() * 0.0
        if reg_terms > 0:
            reg_loss = reg_loss / reg_terms

        loss = self.class_weight * class_loss + self.reg_weight * reg_loss
        log = {
            "total_loss": float(loss.detach().cpu()),
            "class_loss": float(class_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
            "slot_losses": {slot: float(value.detach().cpu()) for slot, value in class_losses.items()},
            "slot_preds": {slot: torch.argmax(outputs["class_logits"][slot], dim=-1).detach().cpu() for slot in CONDITION_FIELDS},
            "slot_targets": {slot: labels[slot].detach().cpu() for slot in CONDITION_FIELDS},
        }
        if self.expose_ddp_components:
            log["_ddp_components"] = {
                "class_means": class_losses,
                "class_denominators": class_denominators,
                "regression_means": regression_losses,
                "regression_denominators": regression_denominators,
            }
        return loss, log


class YieldRegressionLoss(nn.Module):
    def __init__(
        self,
        uncertainty: bool = True,
        target_key: str = "yield",
        loss_type: str | None = None,
        huber_delta: float = 10.0,
        standardize_targets: bool = False,
        target_mean: float | None = None,
        target_std: float | None = None,
    ):
        super().__init__()
        self.target_key = str(target_key)
        self.standardize_targets = bool(standardize_targets)
        if self.standardize_targets:
            if target_mean is None or target_std is None:
                raise ValueError(
                    "standardize_targets=true requires training-only target_mean and target_std."
                )
            if not torch.isfinite(torch.tensor(float(target_mean))) or not torch.isfinite(
                torch.tensor(float(target_std))
            ):
                raise ValueError("Yield target normalization statistics must be finite.")
            if float(target_std) <= 0:
                raise ValueError("Yield target_std must be positive.")
        resolved_mean = float(target_mean or 0.0)
        resolved_std = float(target_std or 1.0)
        self.register_buffer("target_mean", torch.tensor(resolved_mean, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(resolved_std, dtype=torch.float32))
        if loss_type is None:
            loss_type = "uncertainty" if uncertainty else "mse"
        self.loss_type = str(loss_type).lower()
        if self.loss_type in {"nll", "gaussian_nll"}:
            self.loss_type = "uncertainty"
        if self.loss_type in {"smooth_l1", "smoothl1"}:
            self.loss_type = "huber"
        if self.loss_type not in {"uncertainty", "mse", "huber"}:
            raise ValueError("YieldRegressionLoss loss_type must be one of: uncertainty, mse, huber.")
        self.huber_delta = float(huber_delta)
        if self.loss_type == "huber" and self.huber_delta <= 0:
            raise ValueError("YieldRegressionLoss huber_delta must be positive.")

    def _huber_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        error = pred - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return (0.5 * quadratic.pow(2) + delta * linear).mean()

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict, is_train: bool = True) -> Tuple[torch.Tensor, Dict]:
        target = batch[self.target_key]
        raw_mean = outputs["mean"]
        target_mean = self.target_mean.to(device=target.device, dtype=target.dtype)
        target_std = self.target_std.to(device=target.device, dtype=target.dtype)
        if self.standardize_targets:
            loss_target = (target - target_mean) / target_std
            loss_mean = raw_mean
            metric_mean = raw_mean * target_std + target_mean
            huber_delta = raw_mean.new_tensor(self.huber_delta) / target_std.to(raw_mean.dtype)
        else:
            loss_target = target
            loss_mean = raw_mean
            metric_mean = raw_mean
            huber_delta = raw_mean.new_tensor(self.huber_delta)
        if self.loss_type == "uncertainty":
            logvar = outputs["logvar"].clamp(min=-10.0, max=10.0)
            loss = 0.5 * (torch.exp(-logvar) * (loss_target - loss_mean) ** 2 + logvar)
            loss = loss.mean()
        elif self.loss_type == "mse":
            loss = F.mse_loss(loss_mean, loss_target)
        else:
            loss = self._huber_loss(loss_mean, loss_target, delta=huber_delta)
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "preds": metric_mean.detach().cpu(),
            "targets": target.detach().cpu(),
        }


class YieldPairwiseRankingLoss(nn.Module):
    """Pairwise loss for ranking two condition candidates of the same reaction.

    The ranking collator flattens each pair as [better, worse], so the desired
    score margin is score_better - score_worse > 0.
    """

    def __init__(
        self,
        loss_type: str = "bce",
        margin: float = 1.0,
        temperature: float = 1.0,
        weight_by_diff: bool = False,
    ):
        super().__init__()
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"bce", "margin"}:
            raise ValueError("YieldPairwiseRankingLoss loss_type must be 'bce' or 'margin'.")
        self.margin = float(margin)
        self.temperature = max(float(temperature), 1e-6)
        self.weight_by_diff = bool(weight_by_diff)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict, is_train: bool = True) -> Tuple[torch.Tensor, Dict]:
        pair_count = int(batch.get("pair_count", 0))
        scores = outputs["mean"]
        if pair_count <= 0:
            pair_count = int(scores.numel() // 2)
        if scores.numel() != pair_count * 2:
            raise ValueError(f"Ranking batch expected {pair_count * 2} scores, got {scores.numel()}.")
        pair_scores = scores.view(pair_count, 2)
        score_diff = pair_scores[:, 0] - pair_scores[:, 1]
        if self.loss_type == "margin":
            per_pair_loss = F.relu(self.margin - score_diff)
        else:
            per_pair_loss = F.softplus(-score_diff / self.temperature)

        yield_diff = batch.get("yield_diff")
        if self.weight_by_diff and yield_diff is not None and yield_diff.numel() == per_pair_loss.numel():
            weights = yield_diff.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
            weights = weights / weights.mean().clamp_min(1e-6)
            per_pair_loss = per_pair_loss * weights
        loss = per_pair_loss.mean()
        correct = score_diff > 0
        return loss, {
            "total_loss": float(loss.detach().cpu()),
            "pair_correct": int(correct.detach().cpu().sum().item()),
            "pair_total": int(correct.numel()),
            "score_diff": score_diff.detach().cpu(),
            "yield_diff": yield_diff.detach().cpu() if isinstance(yield_diff, torch.Tensor) else torch.empty(0),
        }
