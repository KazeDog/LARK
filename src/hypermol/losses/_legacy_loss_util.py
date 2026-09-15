from typing import Callable

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor, tensor
from hypermol.utils.pretrain_tasks import normalize_pretrain_tasks

class PretrainLoss(nn.Module):
    """
    一个统一的损失和指标计算模块。
    - 在训练模式下 (is_train=True)，只高效地计算总损失。
    - 在评估模式下 (is_train=False)，计算总损失并收集所有评价指标的原始数据。
    """

    def __init__(
        self,
        mam_weight=1.0,
        angle_weight=0.5,
        torsion_weight=0.5,
        fp_weight=1.0,
        rmat_weigh=10.0,
        rmat_weight=None,
        rmat_positive_weight=1.0,
        rmat_negative_weight=1.0,
        rmat_positive_threshold=1e-4,
        rmat_metric_threshold=0.5,
        rmat_upper_triangle: bool = False,
        electron_conservation_weight=0.0,
        lambda_ctx: float = 0.0,
        lambda_cons: float = 0.0,
        context_mam_weight: float = 1.0,
        context_fp_weight: float = 1.0,
        active_tasks=None,
    ):
        super().__init__()
        self.mam_weight = mam_weight
        self.angle_weight = angle_weight
        self.torsion_weight = torsion_weight
        self.fp_weight = fp_weight
        self.rmat_weight = rmat_weigh if rmat_weight is None else rmat_weight
        self.rmat_positive_weight = float(rmat_positive_weight)
        self.rmat_negative_weight = float(rmat_negative_weight)
        self.rmat_positive_threshold = float(rmat_positive_threshold)
        self.rmat_metric_threshold = float(rmat_metric_threshold)
        self.rmat_upper_triangle = bool(rmat_upper_triangle)
        self.electron_conservation_weight = electron_conservation_weight
        self.lambda_ctx = float(lambda_ctx)
        self.lambda_cons = float(lambda_cons)
        self.context_mam_weight = float(context_mam_weight)
        self.context_fp_weight = float(context_fp_weight)
        if self.lambda_ctx < 0.0 or self.lambda_cons < 0.0:
            raise ValueError("lambda_ctx and lambda_cons must be non-negative.")
        if self.context_mam_weight < 0.0 or self.context_fp_weight < 0.0:
            raise ValueError("Context task weights must be non-negative.")
        self.active_tasks = set(normalize_pretrain_tasks(active_tasks))

    def is_task_enabled(self, task: str) -> bool:
        return task in self.active_tasks

    @staticmethod
    def _first_tensor(mapping: dict, names: tuple[str, ...]):
        for name in names:
            value = mapping.get(name)
            if isinstance(value, torch.Tensor):
                return value
        return None

    @staticmethod
    def _pair_mask(mask: torch.Tensor | None, reference: torch.Tensor, name: str) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.to(device=reference.device, dtype=torch.bool)
        if mask.shape != reference.shape:
            raise ValueError(
                f"{name} shape {tuple(mask.shape)} does not match Delta-BE shape {tuple(reference.shape)}"
            )
        return mask

    def _delta_be_masks(
        self,
        targets: dict,
        reference: torch.Tensor,
        is_train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve strict and legacy Delta-BE masks.

        New collators may provide separate train/eval masks plus a full-canvas
        validity mask.  Historical batches expose only ``r_matrix_mask`` and a
        one-dimensional ``reaction_canvas_mask``.  The precedence below keeps
        that old behavior while allowing strict evaluation on the full valid
        canvas.
        """

        valid = self._first_tensor(
            targets,
            (
                "delta_be_valid_mask",
                "r_matrix_valid_mask",
                "valid_pair_mask",
                "pair_valid_mask",
            ),
        )
        valid = self._pair_mask(valid, reference, "valid pair mask")
        if valid is None:
            canvas = self._first_tensor(targets, ("reaction_canvas_mask", "canvas_atom_mask"))
            if canvas is not None:
                canvas = canvas.to(device=reference.device, dtype=torch.bool)
                if canvas.ndim != 2 or canvas.shape[0] != reference.shape[0] or canvas.shape[1] != reference.shape[1]:
                    raise ValueError(
                        f"reaction_canvas_mask shape {tuple(canvas.shape)} is incompatible with "
                        f"Delta-BE shape {tuple(reference.shape)}"
                    )
                valid = canvas.unsqueeze(2) & canvas.unsqueeze(1)

        legacy = self._pair_mask(
            self._first_tensor(targets, ("r_matrix_mask", "delta_be_mask")),
            reference,
            "legacy Delta-BE mask",
        )
        train_mask = self._pair_mask(
            self._first_tensor(
                targets,
                (
                    "delta_be_train_mask",
                    "r_matrix_train_mask",
                    "train_pair_mask",
                    "pair_train_mask",
                    "delta_be_loss_mask",
                    "r_matrix_loss_mask",
                    "train_mask",
                ),
            ),
            reference,
            "Delta-BE train mask",
        )
        eval_mask = self._pair_mask(
            self._first_tensor(
                targets,
                (
                    "delta_be_eval_mask",
                    "r_matrix_eval_mask",
                    "eval_pair_mask",
                    "pair_eval_mask",
                    "eval_mask",
                ),
            ),
            reference,
            "Delta-BE eval mask",
        )

        if is_train:
            loss_mask = train_mask if train_mask is not None else legacy
            if loss_mask is None:
                loss_mask = valid
        else:
            loss_mask = eval_mask if eval_mask is not None else valid
            if loss_mask is None:
                loss_mask = legacy

        # Evaluation metrics prefer an explicit eval mask, then the strict full
        # valid canvas, then the historical local supervision mask.
        metric_mask = eval_mask if eval_mask is not None else valid
        if metric_mask is None:
            metric_mask = legacy if legacy is not None else loss_mask

        if loss_mask is None:
            loss_mask = torch.ones_like(reference, dtype=torch.bool)
        if metric_mask is None:
            metric_mask = loss_mask
        if valid is not None:
            loss_mask = loss_mask & valid
            metric_mask = metric_mask & valid
        if self.rmat_upper_triangle:
            upper = torch.triu(
                torch.ones(reference.shape[-2:], device=reference.device, dtype=torch.bool),
                diagonal=0,
            )
            loss_mask = loss_mask & upper.unsqueeze(0)
            metric_mask = metric_mask & upper.unsqueeze(0)
        return loss_mask, metric_mask

    @staticmethod
    def _context_flags_for_valid_entries(
        targets: dict,
        valid_mask: torch.Tensor,
        expected_entries: int,
        name: str,
    ) -> torch.Tensor:
        context_mask = targets.get("context_molecule_mask")
        if context_mask is None:
            return torch.zeros(expected_entries, dtype=torch.bool, device=valid_mask.device)
        context_mask = context_mask.to(device=valid_mask.device, dtype=torch.bool).view(-1)
        if valid_mask.ndim < 1 or valid_mask.shape[0] != context_mask.numel():
            raise ValueError(
                f"context_molecule_mask does not align with {name}: "
                f"{tuple(context_mask.shape)} vs {tuple(valid_mask.shape)}"
            )
        view_shape = (context_mask.numel(),) + (1,) * (valid_mask.ndim - 1)
        expanded = context_mask.view(view_shape).expand_as(valid_mask)
        flags = expanded[valid_mask]
        if flags.numel() != expected_entries:
            raise ValueError(
                f"{name} logits/targets contain {expected_entries} entries but role projection contains "
                f"{flags.numel()}."
            )
        return flags

    @staticmethod
    def _paired_consistency_loss(model_output: dict, targets: dict) -> torch.Tensor | None:
        embeddings = model_output.get("consistency_embeddings")
        pair_index = targets.get("view_pair_index")
        direction = targets.get("view_direction_id")
        if embeddings is None or pair_index is None or direction is None:
            return None
        pair_index = pair_index.to(device=embeddings.device, dtype=torch.long).view(-1)
        direction = direction.to(device=embeddings.device, dtype=torch.long).view(-1)
        if embeddings.ndim != 2 or embeddings.shape[0] != pair_index.numel() or direction.numel() != pair_index.numel():
            raise ValueError("Consistency embeddings and paired-view metadata must share the same batch dimension.")

        first, second = [], []
        for pair_id in torch.unique(pair_index, sorted=True):
            p2r = torch.nonzero((pair_index == pair_id) & (direction == 0), as_tuple=False).flatten()
            r2p = torch.nonzero((pair_index == pair_id) & (direction == 1), as_tuple=False).flatten()
            if p2r.numel() != 1 or r2p.numel() != 1:
                raise ValueError(
                    f"View pair {int(pair_id.item())} must contain exactly one P->R and one R->P embedding."
                )
            first.append(embeddings[p2r[0]])
            second.append(embeddings[r2p[0]])
        if not first:
            return embeddings.sum() * 0.0
        first_tensor = torch.stack(first, dim=0)
        second_tensor = torch.stack(second, dim=0)
        return (1.0 - F.cosine_similarity(first_tensor, second_tensor, dim=-1)).mean()

    def forward(self, model_output: dict, targets: dict, is_train: bool = True):
        """
        计算损失，并根据模式决定是否收集指标数据。

        Args:
            model_output (dict): 模型 forward 方法的输出。
            targets (dict): collator 返回的标签和掩码。
            is_train (bool): 是否为训练模式。True 表示只计算损失，False 表示计算损失和指标。

        Returns:
            total_loss (torch.Tensor): 用于反向传播的加权总损失。
            logging_output (dict): 如果 is_train=True，只包含损失；否则包含损失和指标计数。
        """
        # --- 初始化 ---
        reference_tensor = next((t for t in model_output.values() if t is not None and isinstance(t, torch.Tensor)),
                                next((t for t in targets.values() if t is not None and isinstance(t, torch.Tensor))))

        if reference_tensor is None:
            # 如果批次为空或无效，返回零损失
            return torch.tensor(0.0), {}

        core_loss = torch.tensor(0.0, device=reference_tensor.device, dtype=torch.float32)
        context_loss = torch.tensor(0.0, device=reference_tensor.device, dtype=torch.float32)
        consistency_loss = torch.tensor(0.0, device=reference_tensor.device, dtype=torch.float32)
        logging_output = {}

        # --- 1. 原子类型掩码 (MAM) 任务 ---
        if self.is_task_enabled("mam") and (self.mam_weight > 0 or self.context_mam_weight > 0) and "mam_logits" in model_output and model_output[
            "mam_logits"] is not None and "mam_targets" in targets:
            logits = model_output["mam_logits"]
            target_vals = targets["mam_targets"]
            mask = targets["mam_loss_mask"]
            masked_targets = target_vals[mask]

            if masked_targets.numel() > 0:
                context_flags = self._context_flags_for_valid_entries(
                    targets,
                    mask,
                    masked_targets.numel(),
                    "MAM",
                )
                core_flags = ~context_flags
                if core_flags.any():
                    loss = F.cross_entropy(logits[core_flags], masked_targets[core_flags])
                    core_loss += self.mam_weight * loss
                    logging_output["mam_loss"] = loss.item()

                    if not is_train:
                        with torch.no_grad():
                            preds = logits[core_flags].argmax(dim=-1)
                            logging_output["mam_hit"] = (preds == masked_targets[core_flags]).sum().item()
                            logging_output["mam_total"] = int(core_flags.sum().item())
                if context_flags.any() and self.context_mam_weight > 0.0:
                    loss = F.cross_entropy(logits[context_flags], masked_targets[context_flags])
                    context_loss += self.context_mam_weight * loss
                    logging_output["context_mam_loss"] = loss.item()
                    if not is_train:
                        with torch.no_grad():
                            preds = logits[context_flags].argmax(dim=-1)
                            logging_output["context_mam_hit"] = (
                                preds == masked_targets[context_flags]
                            ).sum().item()
                            logging_output["context_mam_total"] = int(context_flags.sum().item())

        # --- 3. 键角区间预测任务 (核心新增部分) ---
        if (self.is_task_enabled("angle") and
                self.angle_weight > 0 and
                "angle_logits" in model_output and
                "bond_angles_bin" in targets):

            logits = model_output["angle_logits"]  # 预测值, 形状: [B, N_angles, Num_bins]
            target_vals = targets["bond_angles_bin"]  # 标签, 形状: [B, N_angles]

            # 关键：找到所有非填充的、真实存在的键角位置
            # 根据描述，padding值为5
            padding_value = -5
            valid_mask = (target_vals != padding_value)

            # 仅选择有效位置的logits和targets
            valid_logits = logits
            valid_targets = target_vals[valid_mask]

            if valid_targets.numel() > 0:
                if logits.shape[0] != valid_targets.shape[0]:
                    raise ValueError(
                        f"Angle logits and targets disagree: {logits.shape[0]} vs {valid_targets.shape[0]}."
                    )
                context_flags = self._context_flags_for_valid_entries(
                    targets,
                    valid_mask,
                    valid_targets.numel(),
                    "angle",
                )
                core_flags = ~context_flags
                # 计算交叉熵损失
                # PyTorch的CrossEntropyLoss期望logits的形状是 [Num_samples, Num_classes]
                # 和targets的形状是 [Num_samples]
                if core_flags.any():
                    loss = F.cross_entropy(valid_logits[core_flags], valid_targets[core_flags])
                    core_loss += self.angle_weight * loss
                    logging_output["angle_loss"] = loss.item()

                    if not is_train:
                        with torch.no_grad():
                            # 在评估模式下，计算分类准确率
                            preds = valid_logits[core_flags].argmax(dim=-1)
                            logging_output["angle_hit"] = (preds == valid_targets[core_flags]).sum().item()
                            logging_output["angle_total"] = int(core_flags.sum().item())

        # --- 4. 扭转角区间预测任务 ---
        if (self.is_task_enabled("torsion") and
                self.torsion_weight > 0 and
                "torsion_logits" in model_output and model_output["torsion_logits"] is not None and
                "torsion_angles_bin" in targets):

            # 1. 获取输入
            # a. 获取模型输出的、已经是“扁平化”的logits
            #    形状: [total_real_torsions, num_torsion_bins]
            logits = model_output["torsion_logits"]

            # b. 获取collator提供的、经过填充的批处理标签
            #    形状: [batch_size, max_torsions]
            target_vals = targets["torsion_angles_bin"]

            # 2. 准备标签 (Targets)
            # a. 确定用于填充的特殊值 (根据您的collator定义)
            padding_value = -5

            # b. 创建一个布尔掩码，找到所有非填充且拓扑有效的扭转角位置
            #    terminal bond 等无法构成两侧局部平面的边会由 collator 标为 False
            valid_mask = targets.get("torsion_valid_mask", target_vals != padding_value)
            valid_mask = valid_mask.bool() & (target_vals != padding_value)

            # c. 使用布尔掩码索引，从批处理的target_vals中筛选出所有有效标签
            #    这个操作会自动将提取出的值展平 (flatten)
            #    valid_targets 形状: [total_real_torsions]
            valid_targets = target_vals[valid_mask]

            # 3. 计算损失和指标
            # 确保至少有一个有效的目标可以用于计算
            if valid_targets.numel() > 0 and logits.numel() > 0:

                # 作为一个健壮性检查，确保模型的输出数量与标签数量一致
                if logits.shape[0] != valid_targets.shape[0]:
                    # 如果不匹配，说明模型或collator在计算“真实扭转角数量”时存在不一致
                    raise ValueError(
                        f"扭转角Logits和targets在masking后维度不匹配。 "
                        f"Logits的数量: {logits.shape[0]}, Targets的数量: {valid_targets.shape[0]}"
                    )

                context_flags = self._context_flags_for_valid_entries(
                    targets,
                    valid_mask,
                    valid_targets.numel(),
                    "torsion",
                )
                core_flags = ~context_flags

                # a. 计算交叉熵损失
                # PyTorch的CrossEntropyLoss期望logits: [N, C] 和 targets: [N]
                # N = total_real_torsions, C = num_torsion_bins
                if core_flags.any():
                    loss = F.cross_entropy(logits[core_flags], valid_targets[core_flags])

                    # b. 将此任务的损失加权后计入总损失
                    core_loss += self.torsion_weight * loss

                    # c. 记录该任务的损失值（用于日志打印）
                    logging_output["torsion_loss"] = loss.item()

                    # d. 如果是评估模式 (is_train=False)，则额外计算和收集评估指标
                    if not is_train:
                        with torch.no_grad():  # 在不计算梯度的模式下进行，以节省资源
                            # i. 计算准确率
                            preds = logits[core_flags].argmax(dim=-1)
                            logging_output["torsion_hit"] = (preds == valid_targets[core_flags]).sum().item()
                            logging_output["torsion_total"] = int(core_flags.sum().item())

                        # # ii. 收集概率和标签，用于在epoch结束后计算AUC
                        # probs = F.softmax(logits, dim=-1)
                        # logging_output["torsion_probs"] = probs.cpu().numpy()
                        # logging_output["torsion_targets"] = valid_targets.cpu().numpy()

        # --- 4. 分子指纹预测任务 ---
        if self.is_task_enabled("fingerprint") and (self.fp_weight > 0 or self.context_fp_weight > 0) and "fingerprint_logits" in model_output and "maccs_fp" in targets:
            logits = model_output["fingerprint_logits"]
            target_vals = torch.cat([targets["morgan2048_fp"], targets["maccs_fp"], targets["rdkit_fp"]], dim=-1)
            context_flags = targets.get("context_molecule_mask")
            if context_flags is None:
                context_flags = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
            else:
                context_flags = context_flags.to(device=logits.device, dtype=torch.bool).view(-1)
            if context_flags.numel() != logits.shape[0] or target_vals.shape[0] != logits.shape[0]:
                raise ValueError("Fingerprint predictions, targets, and molecule roles must align.")
            core_flags = ~context_flags
            if core_flags.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[core_flags],
                    target_vals[core_flags].float(),
                )
                core_loss += self.fp_weight * loss
                logging_output["fp_loss"] = loss.item()

                if not is_train:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits[core_flags])
                        preds = (probs > 0.5).long()
                        logging_output["fp_correct"] = (preds == target_vals[core_flags]).sum().item()
                        logging_output["fp_total"] = target_vals[core_flags].numel()
                        # 存储原始概率和标签以计算 ROC-AUC
                        logging_output["fp_probs"] = probs.cpu().numpy()
                        logging_output["fp_targets"] = target_vals[core_flags].cpu().numpy()
            if context_flags.any() and self.context_fp_weight > 0.0:
                loss = F.binary_cross_entropy_with_logits(
                    logits[context_flags],
                    target_vals[context_flags].float(),
                )
                context_loss += self.context_fp_weight * loss
                logging_output["context_fp_loss"] = loss.item()
                if not is_train:
                    with torch.no_grad():
                        probs = torch.sigmoid(logits[context_flags])
                        preds = (probs > 0.5).long()
                        logging_output["context_fp_correct"] = (
                            preds == target_vals[context_flags]
                        ).sum().item()
                        logging_output["context_fp_total"] = target_vals[context_flags].numel()

        # --- 2. Delta-BE / legacy R-matrix prediction task ---
        delta_be_preds = self._first_tensor(model_output, ("delta_be_preds", "r_matrix_preds"))
        delta_be_targets = self._first_tensor(targets, ("delta_be_targets", "r_matrix_targets"))
        if (
            self.is_task_enabled("r_matrix")
            and self.rmat_weight > 0
            and delta_be_preds is not None
            and delta_be_targets is not None
        ):
            target_vals = delta_be_targets.to(device=delta_be_preds.device, dtype=delta_be_preds.dtype)
            if target_vals.shape != delta_be_preds.shape:
                raise ValueError(
                    f"Delta-BE prediction shape {tuple(delta_be_preds.shape)} does not match "
                    f"target shape {tuple(target_vals.shape)}"
                )
            loss_mask, metric_mask = self._delta_be_masks(targets, target_vals, is_train=is_train)
            preds_to_compare = delta_be_preds[loss_mask]
            targets_to_compare = target_vals[loss_mask]

            # Every DDP rank must expose the component, including a rare rank
            # with zero valid pairs, so all ranks enter the same collective.
            if is_train:
                logging_output["_rmat_loss_tensor"] = delta_be_preds.sum() * 0.0

            if targets_to_compare.numel() > 0:
                element_loss = F.smooth_l1_loss(
                    preds_to_compare,
                    targets_to_compare,
                    beta=1.0,
                    reduction="none",
                )
                positive = targets_to_compare.abs() > self.rmat_positive_threshold
                element_weights = torch.where(
                    positive,
                    element_loss.new_full((), self.rmat_positive_weight),
                    element_loss.new_full((), self.rmat_negative_weight),
                )
                weight_sum = element_weights.sum()
                if float(weight_sum.item()) > 0.0:
                    weighted_loss_sum = (element_loss * element_weights).sum()
                    loss = weighted_loss_sum / weight_sum
                else:
                    weighted_loss_sum = element_loss.sum() * 0.0
                    loss = element_loss.sum() * 0.0
                core_loss += self.rmat_weight * loss
                # Keep the differentiable local mean available to the DDP
                # training loop.  DDP otherwise averages rank-local means,
                # which overweights ranks whose molecular canvases contain
                # fewer valid pairs.  The loop consumes (and removes) this
                # private entry before metric aggregation.
                if is_train:
                    logging_output["_rmat_loss_tensor"] = loss
                logging_output["rmat_loss"] = loss.item()
                logging_output["rmat_loss_sum"] = weighted_loss_sum.item()
                logging_output["rmat_loss_weight"] = weight_sum.item()
                logging_output["delta_be_loss"] = loss.item()
                logging_output["rmat_positive_count"] = int(positive.sum().item())
                logging_output["rmat_negative_count"] = int((~positive).sum().item())
                if positive.any():
                    logging_output["rmat_positive_loss"] = element_loss[positive].mean().item()
                if (~positive).any():
                    logging_output["rmat_negative_loss"] = element_loss[~positive].mean().item()

                if not is_train:
                    with torch.no_grad():
                        metric_preds = delta_be_preds[metric_mask]
                        metric_targets = target_vals[metric_mask]
                        absolute_error = (metric_preds - metric_targets).abs()
                        metric_positive = metric_targets.abs() > self.rmat_positive_threshold
                        metric_negative = ~metric_positive

                        logging_output["rmat_mae_sum"] = absolute_error.sum().item()
                        logging_output["rmat_total_preds"] = metric_targets.numel()
                        if metric_positive.any():
                            positive_error_sum = absolute_error[metric_positive].sum().item()
                            positive_total = int(metric_positive.sum().item())
                            logging_output["rmat_positive_mae_sum"] = positive_error_sum
                            logging_output["rmat_positive_total"] = positive_total
                            logging_output["delta_be_positive_mae_sum"] = positive_error_sum
                            logging_output["delta_be_positive_total"] = positive_total
                        if metric_negative.any():
                            negative_error_sum = absolute_error[metric_negative].sum().item()
                            negative_total = int(metric_negative.sum().item())
                            logging_output["rmat_negative_mae_sum"] = negative_error_sum
                            logging_output["rmat_negative_total"] = negative_total
                            logging_output["delta_be_negative_mae_sum"] = negative_error_sum
                            logging_output["delta_be_negative_total"] = negative_total

                        true_centers = metric_targets.abs() >= self.rmat_metric_threshold
                        pred_centers = metric_preds.abs() >= self.rmat_metric_threshold
                        logging_output["rmat_tp"] = (true_centers & pred_centers).sum().item()
                        logging_output["rmat_fp"] = (~true_centers & pred_centers).sum().item()
                        logging_output["rmat_fn"] = (true_centers & ~pred_centers).sum().item()

        conservation_pair_mask = self._first_tensor(
            targets,
            ("delta_be_valid_mask", "r_matrix_valid_mask", "valid_pair_mask", "pair_valid_mask"),
        )
        conservation_canvas_mask = self._first_tensor(
            targets,
            ("reaction_canvas_mask", "canvas_atom_mask"),
        )
        if (
            self.is_task_enabled("electron_conservation")
            and
            self.electron_conservation_weight > 0
            and delta_be_preds is not None
            and (conservation_pair_mask is not None or conservation_canvas_mask is not None)
        ):
            preds = delta_be_preds
            if conservation_pair_mask is not None:
                pair_mask = self._pair_mask(
                    conservation_pair_mask,
                    preds,
                    "electron-conservation valid pair mask",
                )
            else:
                canvas_mask = conservation_canvas_mask.to(device=preds.device, dtype=torch.bool)
                pair_mask = canvas_mask.unsqueeze(2) & canvas_mask.unsqueeze(1)
            if pair_mask.any():
                pair_mask_float = pair_mask.to(preds.dtype)
                pred_delta_sum = (preds * pair_mask_float).sum(dim=(1, 2))
                pair_count = pair_mask_float.sum(dim=(1, 2)).clamp_min(1.0)
                normalized_abs_sum = pred_delta_sum.abs() / pair_count
                loss = normalized_abs_sum.mean()
                core_loss += self.electron_conservation_weight * loss
                logging_output["electron_conservation_loss"] = loss.item()
                if not is_train:
                    logging_output["electron_conservation_abs_sum"] = pred_delta_sum.abs().sum().item()
                    logging_output["electron_conservation_norm_abs_sum"] = normalized_abs_sum.sum().item()
                    logging_output["electron_conservation_total"] = pred_delta_sum.numel()

        paired_loss = self._paired_consistency_loss(model_output, targets)
        if paired_loss is not None:
            consistency_loss = paired_loss
            logging_output["consistency_loss"] = consistency_loss.item()
        elif self.lambda_cons > 0.0:
            raise ValueError(
                "lambda_cons > 0 requires consistency_embeddings plus view_pair_index/view_direction_id."
            )

        total_loss = (
            core_loss
            + self.lambda_ctx * context_loss
            + self.lambda_cons * consistency_loss
        )
        logging_output["core_loss"] = core_loss.item()
        logging_output["context_loss"] = context_loss.item()
        logging_output.setdefault("consistency_loss", consistency_loss.item())

        batch_reference = targets.get("reaction_canvas_mask")
        if not isinstance(batch_reference, torch.Tensor):
            batch_reference = delta_be_targets
        sample_size = int(batch_reference.shape[0]) if isinstance(batch_reference, torch.Tensor) and batch_reference.ndim > 0 else 1
        logging_output["total_loss"] = total_loss.item()
        logging_output["total_loss_sum"] = total_loss.item() * sample_size
        logging_output["sample_size"] = sample_size

        return total_loss, logging_output


class RetroLoss(nn.Module):
    """
    为逆合成微调任务计算多任务损失，并收集用于评估的原始数据。
    """

    def __init__(
        self,
        rmat_weight=1.0,
        atom_id_weight=1.0,
        rmat_loss_type='smooth_l1',
        rmat_positive_weight=1.0,
        rmat_negative_weight=1.0,
        rmat_positive_threshold=1e-4,
        charge_delta_weight=0.0,
        total_h_delta_weight=0.0,
        atom_num_delta_weight=0.0,
        chiral_weight=0.0,
        aromatic_weight=0.0,
        completion_weight=0.0,
        attachment_site_weight=0.0,
        attachment_bond_weight=0.0,
        attachment_positive_weight=1.0,
        lgm_weight=0.0,
        lgc_pair_weight=0.0,
        lgc_bond_weight=0.0,
        lgc_positive_weight=1.0,
        lg_contrastive_weight=0.0,
        lg_contrastive_temperature=0.1,
    ):
        super().__init__()
        self.rmat_weight = rmat_weight
        self.atom_id_weight = atom_id_weight
        self.rmat_loss_type = str(rmat_loss_type).lower()
        self.rmat_positive_weight = float(rmat_positive_weight)
        self.rmat_negative_weight = float(rmat_negative_weight)
        self.rmat_positive_threshold = float(rmat_positive_threshold)
        self.charge_delta_weight = float(charge_delta_weight)
        self.total_h_delta_weight = float(total_h_delta_weight)
        self.atom_num_delta_weight = float(atom_num_delta_weight)
        self.chiral_weight = float(chiral_weight)
        self.aromatic_weight = float(aromatic_weight)
        self.completion_weight = float(completion_weight)
        self.attachment_site_weight = float(attachment_site_weight)
        self.attachment_bond_weight = float(attachment_bond_weight)
        self.attachment_positive_weight = float(attachment_positive_weight)
        self.lgm_weight = float(lgm_weight)
        self.lgc_pair_weight = float(lgc_pair_weight)
        self.lgc_bond_weight = float(lgc_bond_weight)
        self.lgc_positive_weight = float(lgc_positive_weight)
        self.lg_contrastive_weight = float(lg_contrastive_weight)
        self.lg_contrastive_temperature = max(float(lg_contrastive_temperature), 1e-4)

        self.atom_id_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.full_ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.lgm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _rmat_elementwise_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.rmat_loss_type == 'smooth_l1':
            return F.smooth_l1_loss(preds, targets, beta=1.0, reduction='none')
        return F.mse_loss(preds, targets, reduction='none')

    @staticmethod
    def _masked_smooth_l1(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        mask = mask.bool()
        if not mask.any():
            return None
        return F.smooth_l1_loss(preds[mask], targets[mask].to(preds.dtype), beta=1.0)

    @staticmethod
    def _binary_loss_with_optional_pos_weight(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        positive_weight: float = 1.0,
    ) -> torch.Tensor | None:
        logits = logits.float()
        targets = targets.to(logits.device).float()
        if mask is not None:
            mask = mask.to(logits.device).bool()
            if not mask.any():
                return None
            logits = logits[mask]
            targets = targets[mask]
        pos_weight = None
        if float(positive_weight) != 1.0:
            pos_weight = torch.tensor(float(positive_weight), device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def forward(self, model_output: dict, targets: dict, is_train: bool = True):
        total_loss = torch.tensor(0.0, device=next(iter(model_output.values())).device)
        logging_output = {}

        # --- 1. 计算 R矩阵 损失 ---
        if self.rmat_weight > 0 and "r_matrix_preds" in model_output:
            preds = model_output["r_matrix_preds"]
            target_vals = targets["padded_r_matrix_targets"]
            mask = targets["padded_r_matrix_mask"]

            preds_to_compare = preds[mask]
            targets_to_compare = target_vals[mask]

            if targets_to_compare.numel() > 0:
                element_loss = self._rmat_elementwise_loss(preds_to_compare, targets_to_compare)
                positive_mask = targets_to_compare.abs() > self.rmat_positive_threshold
                if self.rmat_positive_weight != 1.0 or self.rmat_negative_weight != 1.0:
                    weights = torch.where(
                        positive_mask,
                        torch.full_like(element_loss, self.rmat_positive_weight),
                        torch.full_like(element_loss, self.rmat_negative_weight),
                    )
                    loss = (element_loss * weights).sum() / weights.sum().clamp_min(1e-8)
                else:
                    loss = element_loss.mean()
                total_loss += self.rmat_weight * loss
                logging_output["rmat_loss"] = loss.item()
                logging_output["rmat_positive_count"] = int(positive_mask.sum().item())
                logging_output["rmat_negative_count"] = int((~positive_mask).sum().item())
                if not is_train:
                    # 收集用于计算MAE的数据
                    logging_output["rmat_preds"] = preds_to_compare.detach()
                    logging_output["rmat_targets"] = targets_to_compare.detach()

        # --- 2. 计算 原子身份 损失 ---
        if self.atom_id_weight > 0 and "atom_identity_logits" in model_output:
            logits = model_output["atom_identity_logits"]
            target_vals = targets["padded_atom_identity_targets"]

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_vals.view(-1)

            loss = self.atom_id_loss_fn(logits_flat, targets_flat)
            total_loss += self.atom_id_weight * loss
            logging_output["atom_id_loss"] = loss.item()
            if not is_train:
                # 找到有效的目标进行准确率计算
                valid_mask = (targets_flat != -100)
                # 收集用于计算准确率的数据
                logging_output["atom_id_preds"] = logits_flat.argmax(-1)[valid_mask].detach()
                logging_output["atom_id_targets"] = targets_flat[valid_mask].detach()

        # --- 3. Full USPTO50K-aligned atom-state auxiliary targets ---
        if "atom_state_mask" in targets:
            atom_state_mask = targets["atom_state_mask"].bool()
            regression_specs = [
                ("charge_delta", "charge_delta_preds", self.charge_delta_weight, "charge_delta_loss"),
                ("total_h_delta", "total_h_delta_preds", self.total_h_delta_weight, "total_h_delta_loss"),
                ("atom_num_delta", "atom_num_delta_preds", self.atom_num_delta_weight, "atom_num_delta_loss"),
            ]
            for target_key, pred_key, weight, log_key in regression_specs:
                if weight <= 0 or pred_key not in model_output or target_key not in targets:
                    continue
                target_vals = targets[target_key].to(model_output[pred_key].device)
                valid_mask = atom_state_mask.to(target_vals.device) & (target_vals != -100)
                loss = self._masked_smooth_l1(model_output[pred_key], target_vals, valid_mask)
                if loss is None:
                    continue
                total_loss += float(weight) * loss
                logging_output[log_key] = loss.item()

            if self.chiral_weight > 0 and "reactant_chiral_logits" in model_output and "reactant_chiral_targets" in targets:
                logits = model_output["reactant_chiral_logits"]
                target_vals = targets["reactant_chiral_targets"].to(logits.device)
                loss = self.full_ce_loss_fn(logits.view(-1, logits.size(-1)), target_vals.view(-1))
                total_loss += self.chiral_weight * loss
                logging_output["chiral_loss"] = loss.item()

            if self.aromatic_weight > 0 and "reactant_aromatic_logits" in model_output and "reactant_aromatic_targets" in targets:
                logits = model_output["reactant_aromatic_logits"]
                target_vals = targets["reactant_aromatic_targets"].to(logits.device)
                loss = self.full_ce_loss_fn(logits.view(-1, logits.size(-1)), target_vals.view(-1))
                total_loss += self.aromatic_weight * loss
                logging_output["aromatic_loss"] = loss.item()

        # --- 4. Full USPTO50K-aligned fragment/completion and attachment targets ---
        if self.completion_weight > 0 and "completion_logits" in model_output and "requires_completion" in targets:
            completion_logits = model_output["completion_logits"]
            completion_targets = targets["requires_completion"].to(completion_logits.device)
            loss = self._binary_loss_with_optional_pos_weight(
                completion_logits,
                completion_targets,
            )
            if loss is not None:
                total_loss += self.completion_weight * loss
                logging_output["completion_loss"] = loss.item()
                if not is_train:
                    logging_output["completion_preds"] = (
                        torch.sigmoid(completion_logits.detach()) >= 0.5
                    )
                    logging_output["completion_targets"] = completion_targets.detach().bool()

        if self.lgm_weight > 0 and "lgm_fragment_logits" in model_output and "lgm_fragment_targets" in targets:
            logits = model_output["lgm_fragment_logits"]
            target_vals = targets["lgm_fragment_targets"].to(logits.device).long()
            valid_mask = target_vals != -100
            if valid_mask.any():
                loss = self.lgm_loss_fn(logits, target_vals)
                total_loss += self.lgm_weight * loss
                logging_output["lgm_loss"] = loss.item()
                if not is_train:
                    logging_output["lgm_preds"] = logits.argmax(dim=-1).detach()[valid_mask]
                    logging_output["lgm_targets"] = target_vals.detach()[valid_mask]

        if (
            self.attachment_site_weight > 0
            and "attachment_site_logits" in model_output
            and "attachment_site_targets" in targets
        ):
            logits = model_output["attachment_site_logits"]
            target_vals = targets["attachment_site_targets"].to(logits.device)
            mask = targets.get("reaction_canvas_mask")
            if mask is not None:
                mask = mask.to(logits.device).bool()
            loss = self._binary_loss_with_optional_pos_weight(
                logits,
                target_vals,
                mask=mask,
                positive_weight=self.attachment_positive_weight,
            )
            if loss is not None:
                total_loss += self.attachment_site_weight * loss
                logging_output["attachment_site_loss"] = loss.item()
                if not is_train:
                    eval_mask = mask if mask is not None else torch.ones_like(target_vals, dtype=torch.bool)
                    logging_output["attachment_site_preds"] = (
                        torch.sigmoid(logits.detach()[eval_mask]) >= 0.5
                    )
                    logging_output["attachment_site_targets_eval"] = target_vals.detach()[eval_mask].bool()

        if (
            self.attachment_bond_weight > 0
            and "attachment_bond_order_preds" in model_output
            and "attachment_bond_order_targets" in targets
            and "attachment_site_targets" in targets
        ):
            preds = model_output["attachment_bond_order_preds"]
            target_vals = targets["attachment_bond_order_targets"].to(preds.device)
            mask = targets["attachment_site_targets"].to(preds.device).bool()
            loss = self._masked_smooth_l1(preds, target_vals, mask)
            if loss is not None:
                total_loss += self.attachment_bond_weight * loss
                logging_output["attachment_bond_loss"] = loss.item()

        # --- 5. Selected-LG-conditioned product x gate connection targets ---
        if (
            self.lgc_pair_weight > 0
            and "lgc_pair_logits" in model_output
            and "lgc_pair_targets" in targets
            and "lgc_pair_mask" in targets
        ):
            logits = model_output["lgc_pair_logits"]
            target_vals = targets["lgc_pair_targets"].to(logits.device)
            mask = targets["lgc_pair_mask"].to(logits.device).bool()
            loss = self._binary_loss_with_optional_pos_weight(
                logits,
                target_vals,
                mask=mask,
                positive_weight=self.lgc_positive_weight,
            )
            if loss is not None:
                total_loss += self.lgc_pair_weight * loss
                logging_output["lgc_pair_loss"] = loss.item()
                if not is_train:
                    logging_output["lgc_pair_preds"] = (
                        torch.sigmoid(logits.detach()[mask]) >= 0.5
                    )
                    logging_output["lgc_pair_targets_eval"] = target_vals.detach()[mask].bool()

        if (
            self.lgc_bond_weight > 0
            and "lgc_bond_order_logits" in model_output
            and "lgc_bond_order_targets" in targets
        ):
            logits = model_output["lgc_bond_order_logits"]
            target_vals = targets["lgc_bond_order_targets"].to(logits.device).long()
            valid_mask = target_vals != -100
            if valid_mask.any():
                loss = F.cross_entropy(logits[valid_mask], target_vals[valid_mask])
                total_loss += self.lgc_bond_weight * loss
                logging_output["lgc_bond_loss"] = loss.item()
                if not is_train:
                    logging_output["lgc_bond_preds"] = logits.detach()[valid_mask].argmax(dim=-1)
                    logging_output["lgc_bond_targets_eval"] = target_vals.detach()[valid_mask]

        if (
            self.lg_contrastive_weight > 0
            and "lg_contrastive_product" in model_output
            and "lg_contrastive_fragment" in model_output
            and "lgm_fragment_targets" in targets
        ):
            labels = targets["lgm_fragment_targets"].to(model_output["lg_contrastive_product"].device).long()
            valid = (labels > 1)
            if "lg_atom_mask" in targets:
                valid = valid & targets["lg_atom_mask"].to(labels.device).bool().any(dim=-1)
            if int(valid.sum().item()) > 1:
                product_repr = model_output["lg_contrastive_product"][valid]
                fragment_repr = model_output["lg_contrastive_fragment"][valid]
                valid_labels = labels[valid]
                logits = product_repr @ fragment_repr.transpose(0, 1)
                logits = logits / self.lg_contrastive_temperature
                positive_mask = valid_labels.unsqueeze(1) == valid_labels.unsqueeze(0)
                positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
                loss = -(
                    torch.logsumexp(positive_logits, dim=1)
                    - torch.logsumexp(logits, dim=1)
                ).mean()
                total_loss += self.lg_contrastive_weight * loss
                logging_output["lg_contrastive_loss"] = loss.item()

        logging_output["total_loss"] = total_loss.item()
        return total_loss, logging_output

# noinspection SpellCheckingInspection,PyPep8Naming
class bce_loss(nn.Module):
    def __init__(self, weights=None):
        super(bce_loss, self).__init__()
        self.weights = weights

    def forward(self, pred, label):
        if self.weights is not None:
            fore_weights = torch.as_tensor(self.weights[0], device=label.device, dtype=label.dtype)
            back_weights = torch.as_tensor(self.weights[1], device=label.device, dtype=label.dtype)
            weights = label * back_weights + (1.0 - label) * fore_weights
        else:
            weights = torch.ones(label.shape, device=label.device)

        loss = F.binary_cross_entropy_with_logits(pred, label, weights, reduction='none')
        return loss


# noinspection SpellCheckingInspection,PyPep8Naming
class NTXentLoss_atom(nn.Module):
    def __init__(self, t=0.1):
        super(NTXentLoss_atom, self).__init__()
        self.T = t
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, out, out_mask, labels):
        out = nn.functional.normalize(out, dim=-1)
        out_mask = nn.functional.normalize(out_mask, dim=-1)

        logits = torch.matmul(out_mask, out.permute(0, 2, 1))
        logits /= self.T

        softmaxs = self.softmax(logits)
        loss = self.criterion(softmaxs.transpose(1, 2), labels)

        return loss, logits


# noinspection SpellCheckingInspection
class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask

    @staticmethod
    def _dot_similarity(x: Tensor, y: Tensor):
        v: Tensor = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_similarity(self, x: Tensor, y: Tensor):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v: Tensor = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        mask = self.mask_samples_from_same_repr.to(device=similarity_matrix.device)
        negatives = similarity_matrix[mask].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size, device=logits.device, dtype=torch.long)
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)



class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, np.ndarray):
            self.alpha = torch.tensor(alpha, device=device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        one_hot_key[:, 0] = 0  # ignore 0 index.
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()


def get_focal_loss(pred, label, alpha, device):
    loss = FocalLoss(alpha=alpha, device=device)
    return loss(pred, label)

class UncertaintyLoss(nn.Module):
    """
    Computes the weighted loss based on task uncertainty.

    This module learns a homoscedastic uncertainty for each task, which is then used
    to weigh the individual task losses. The total loss is a sum of the weighted
    task losses and a regularization term for the uncertainty parameters.

    Ref: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (https://arxiv.org/abs/1705.07115)

    Usage:
        loss_func = UncertaintyLoss(num_tasks=4)
        total_loss = loss_func(loss_map, loss_acd, loss_bcd, loss_mfp)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    """

    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks (int): The number of tasks (and losses) to weigh.
        """
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("Number of tasks must be positive.")

        # Initialize log_vars, which correspond to log(sigma^2) for each task.
        # Initializing them to 0.0 means each task initially has a weight of 1.0.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        self.num_tasks = num_tasks

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total multi-task loss.

        Args:
            *losses: A variable number of tensors, each representing the loss for a single task.
                     The number of losses must match `self.num_tasks`.

        Returns:
            A single scalar tensor representing the total combined loss.
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, but got {len(losses)}.")

        total_loss = 0
        for i, loss in enumerate(losses):
            # Calculate the precision (1/sigma^2) from the learnable parameter
            precision = torch.exp(-self.log_vars[i])

            # The first part of the loss for this task
            task_loss_term = precision * loss

            # The second part (regularization) for this task
            regularization_term = self.log_vars[i]

            # Note: The original paper has a factor of 0.5 for the regularization term,
            # which comes from the Gaussian likelihood derivation. It can be omitted
            # as it's just a constant scaling factor, but for completeness:
            # total_loss += task_loss_term + 0.5 * regularization_term
            # In practice, omitting the 0.5 is common and works well.
            total_loss += task_loss_term + regularization_term

        return total_loss

