import loguru
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score, accuracy_score)
from torch import Tensor
import torch.nn.functional as F

def reduce_metrics(logging_outputs: list) -> dict:
    """
    聚合一个 epoch 的所有 logging_outputs，计算最终的平均损失和评价指标。
    """
    if not logging_outputs:
        return {}

    # --- 聚合所有可以简单求和的数值 ---
    # 使用 .get(key, 0) 来安全地处理在训练阶段可能不存在的键
    agg_log = {
        key: sum(log.get(key, 0) for log in logging_outputs)
        for key in [
            "total_loss", "total_loss_sum", "sample_size",
            "core_loss", "context_loss", "consistency_loss",
            "mam_loss", "mam_hit", "mam_total",
            "context_mam_loss", "context_mam_hit", "context_mam_total",
            "angle_loss", "angle_hit", "angle_total",
            "torsion_loss", "torsion_hit", "torsion_total",
            "fp_loss", "fp_correct", "fp_total",
            "context_fp_loss", "context_fp_correct", "context_fp_total",
            "rmat_loss", "rmat_loss_sum", "rmat_loss_weight", "rmat_mae_sum", "rmat_total_preds",
            "rmat_positive_count", "rmat_negative_count",
            "rmat_positive_mae_sum", "rmat_positive_total",
            "rmat_negative_mae_sum", "rmat_negative_total",
            "rmat_tp", "rmat_fp", "rmat_fn",
            "electron_conservation_loss",
            "electron_conservation_abs_sum",
            "electron_conservation_norm_abs_sum",
            "electron_conservation_total",
        ]
    }

    # --- 单独收集用于 ROC-AUC 计算的数组 ---
    # 只有在验证阶段的 log 中才有这些键
    all_fp_probs = np.concatenate(
        [log["fp_probs"] for log in logging_outputs if "fp_probs" in log and log["fp_probs"].size > 0]) if any(
        "fp_probs" in log for log in logging_outputs) else np.array([])
    all_fp_targets = np.concatenate(
        [log["fp_targets"] for log in logging_outputs if "fp_targets" in log and log["fp_targets"].size > 0]) if any(
        "fp_targets" in log for log in logging_outputs) else np.array([])

    num_batches = len(logging_outputs)
    if num_batches == 0: return {}

    # --- 计算最终的、可读的指标 ---
    final_metrics = {}

    # 平均损失 (除以批次数)
    if agg_log["sample_size"] > 0 and agg_log["total_loss_sum"] != 0:
        final_metrics["avg_total_loss"] = agg_log["total_loss_sum"] / agg_log["sample_size"]
    else:
        # Historical/third-party logs may not yet provide weighted sums.
        final_metrics["avg_total_loss"] = agg_log["total_loss"] / num_batches
    for name in ("core_loss", "context_loss", "consistency_loss"):
        if any(name in log for log in logging_outputs):
            final_metrics[f"avg_{name}"] = agg_log[name] / num_batches
    if agg_log.get("mam_loss", 0) > 0: final_metrics["avg_mam_loss"] = agg_log["mam_loss"] / num_batches
    if agg_log.get("context_mam_loss", 0) > 0:
        final_metrics["avg_context_mam_loss"] = agg_log["context_mam_loss"] / num_batches
    if agg_log.get("angle_loss", 0) > 0: final_metrics["avg_angle_loss"] = agg_log["angle_loss"] / num_batches
    if agg_log.get("torsion_loss", 0) > 0: final_metrics["avg_torsion_loss"] = agg_log["torsion_loss"] / num_batches
    if agg_log.get("fp_loss", 0) > 0: final_metrics["avg_fp_loss"] = agg_log["fp_loss"] / num_batches
    if agg_log.get("context_fp_loss", 0) > 0:
        final_metrics["avg_context_fp_loss"] = agg_log["context_fp_loss"] / num_batches
    if agg_log.get("rmat_loss_weight", 0) > 0:
        final_metrics["avg_rmat_loss"] = agg_log["rmat_loss_sum"] / agg_log["rmat_loss_weight"]
    elif agg_log.get("rmat_loss", 0) > 0:
        final_metrics["avg_rmat_loss"] = agg_log["rmat_loss"] / num_batches
    if agg_log.get("electron_conservation_loss", 0) > 0:
        final_metrics["avg_electron_conservation_loss"] = agg_log["electron_conservation_loss"] / num_batches


    # MAM 准确率
    if agg_log["mam_total"] > 0:
        final_metrics["mam_accuracy"] = agg_log["mam_hit"] / agg_log["mam_total"]
    if agg_log["context_mam_total"] > 0:
        final_metrics["context_mam_accuracy"] = (
            agg_log["context_mam_hit"] / agg_log["context_mam_total"]
        )

    # --- 新增: 计算键角任务的最终指标 ---
    if agg_log.get("angle_total", 0) > 0:
        final_metrics["angle_accuracy"] = agg_log["angle_hit"] / agg_log["angle_total"]

    if agg_log.get("torsion_total", 0) > 0:
        final_metrics["torsion_accuracy"] = agg_log["torsion_hit"] / agg_log["torsion_total"]

    # 指纹指标
    if agg_log["fp_total"] > 0:
        final_metrics["fp_accuracy"] = agg_log["fp_correct"] / agg_log["fp_total"]
        if all_fp_targets.size > 0 and all_fp_probs.size > 0:
            try:
                # 确保标签和预测有相同的形状
                if all_fp_targets.shape == all_fp_probs.shape:
                    final_metrics["fp_roc_auc"] = roc_auc_score(all_fp_targets.flatten(), all_fp_probs.flatten())
                else:  # 如果形状不匹配，可能是一个批次只有一个样本导致的问题
                    final_metrics["fp_roc_auc"] = 0.0
            except ValueError:
                final_metrics["fp_roc_auc"] = 0.0  # e.g. if only one class present in all batches
    if agg_log["context_fp_total"] > 0:
        final_metrics["context_fp_accuracy"] = (
            agg_log["context_fp_correct"] / agg_log["context_fp_total"]
        )

    # a. 计算 R矩阵 MAE (平均绝对误差)
    if agg_log["rmat_total_preds"] > 0:
        final_metrics["rmat_mae"] = agg_log["rmat_mae_sum"] / agg_log["rmat_total_preds"]
    if agg_log["rmat_positive_total"] > 0:
        final_metrics["rmat_positive_mae"] = (
            agg_log["rmat_positive_mae_sum"] / agg_log["rmat_positive_total"]
        )
        final_metrics["delta_be_positive_mae"] = final_metrics["rmat_positive_mae"]
    if agg_log["rmat_negative_total"] > 0:
        final_metrics["rmat_negative_mae"] = (
            agg_log["rmat_negative_mae_sum"] / agg_log["rmat_negative_total"]
        )
        final_metrics["delta_be_negative_mae"] = final_metrics["rmat_negative_mae"]
    if agg_log["rmat_positive_count"] or agg_log["rmat_negative_count"]:
        final_metrics["rmat_positive_count"] = agg_log["rmat_positive_count"]
        final_metrics["rmat_negative_count"] = agg_log["rmat_negative_count"]
    if agg_log.get("electron_conservation_total", 0) > 0:
        final_metrics["electron_conservation_abs_sum"] = (
            agg_log["electron_conservation_abs_sum"] / agg_log["electron_conservation_total"]
        )
        final_metrics["electron_conservation_norm_abs_sum"] = (
            agg_log["electron_conservation_norm_abs_sum"] / agg_log["electron_conservation_total"]
        )

    # b. 计算 R矩阵 F1-Score (精确率、召回率、F1)
    rmat_tp = agg_log["rmat_tp"]
    rmat_fp = agg_log["rmat_fp"]
    rmat_fn = agg_log["rmat_fn"]

    # 计算精确率，并处理分母为0的情况
    precision = rmat_tp / (rmat_tp + rmat_fp) if (rmat_tp + rmat_fp) > 0 else 0.0
    # 计算召回率，并处理分母为0的情况
    recall = rmat_tp / (rmat_tp + rmat_fn) if (rmat_tp + rmat_fn) > 0 else 0.0
    # 计算F1分数，并处理分母为0的情况
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 只有在至少有一个真实的正类时，这些指标才有意义
    if (rmat_tp + rmat_fn) > 0:
        final_metrics["rmat_precision"] = precision
        final_metrics["rmat_recall"] = recall
        final_metrics["rmat_f1_score"] = f1_score

    return final_metrics

def do_compute_metrics_retro(logging_outputs_agg: dict):
    """
    一个新的指标计算函数，专门用于逆合成任务。
    """
    metrics = {}

    # 1. 计算 R矩阵 MAE
    if "rmat_preds_all" in logging_outputs_agg and logging_outputs_agg["rmat_preds_all"].numel() > 0:
        mae = F.l1_loss(logging_outputs_agg["rmat_preds_all"], logging_outputs_agg["rmat_targets_all"])
        metrics["rmat_mae"] = mae.item()

    # 2. 计算 原子身份 准确率
    if "atom_id_preds_all" in logging_outputs_agg and logging_outputs_agg["atom_id_preds_all"].numel() > 0:
        acc = accuracy_score(
            logging_outputs_agg["atom_id_targets_all"].cpu().numpy(),
            logging_outputs_agg["atom_id_preds_all"].cpu().numpy()
        )
        metrics["atom_id_acc"] = acc

    return metrics

def do_compute_metrics(score, pred, target):
    acc_ = compute_accuracy(target, pred)
    # auroc_ = compute_auc(target, score)
    f1_score_ = compute_f1_score(target, pred)
    precision_ =compute_precision(target, pred)
    recall_ = compute_recall(target, pred)

    return acc_, None, f1_score_, precision_, recall_

def compute_accuracy(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy

def compute_precision(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    return precision

def compute_recall(y_true, y_pred):
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    return recall

def compute_f1_score(y_true, y_pred):
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return f1

def compute_auc(y_true, y_score):
    auc = metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
    # fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    # auc = metrics.auc(fpr, tpr)
    return auc


def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def compute_cls_metric(y_true, y_pred):
    # print('y_pred=', y_pred)
    y_pred = np.array(y_pred)
    # print('y_true=', y_true)
    # y_true = y_true[:, 1::2]
    # y_true = [item[0] for item in y_true]
    y_true = np.array(y_true).reshape(y_pred.shape)
    is_valid = y_true >= 0
    roc_list = []
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        label = (label[valid] + 0.0)
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred[valid]))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_cls_metric_tensor(y_true: Tensor, y_pred: Tensor):
    y_true = y_true.view(y_pred.shape)
    is_valid = y_true >= 0
    roc_list = torch.zeros(y_true.shape[1], device='cpu')
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        label = (label[valid] + 0.0)
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(torch.unique(label)) == 2:
            # roc_list[i] = roc_auc_score(label, pred[valid])
            roc_list[i] = roc_auc_score(label.detach().cpu().numpy(), pred[valid].detach().cpu().numpy())
        else:
            loguru.logger.warning(f"No positively labeled pretrain_data available for label {i}. Cannot compute ROC-AUC.")

    roc_auc: Tensor = torch.mean(roc_list)
    if torch.isnan(roc_auc):
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc.item()

def compute_reg_metric(y_true: Tensor, y_pred: Tensor):
    y_true = y_true.view(y_pred.shape)
    mae_list = []
    rmse_list = []
    for i in range(y_true.shape[1]):
        label, pred = y_true[:, i], y_pred[:, i]
        # mae = mean_absolute_error(label, pred)
        mae = torch.mean(torch.abs(label - pred))
        # rmse = torch.sqrt(mean_squared_error(label, pred, squared=True))
        rmse = torch.sqrt(torch.mean((label - pred) ** 2))
        mae_list.append(mae)
        rmse_list.append(rmse)

    mae, rmse = torch.mean(torch.tensor(mae_list)), torch.mean(torch.tensor(rmse_list))
    # return mae, rmse
    return mae.item(), rmse.item()


def _round4(value):
    return round(float(value), 4)


def compute_reaction_class_metrics(preds, targets, probs=None) -> dict:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.size == 0:
        return {}
    out = {"accuracy": _round4(metrics.accuracy_score(targets, preds))}
    try:
        out["macro_f1"] = _round4(metrics.f1_score(targets, preds, average="macro"))
    except Exception:
        out["macro_f1"] = 0.0
    if probs is not None:
        probs = np.asarray(probs)
        try:
            if probs.ndim == 2 and probs.shape[0] == targets.shape[0]:
                out["auc"] = _round4(metrics.roc_auc_score(targets, probs, average="macro", multi_class="ovr"))
            else:
                out["auc"] = 0.0
        except Exception:
            out["auc"] = 0.0
    return out


def compute_condition_metrics(slot_preds: dict, slot_targets: dict) -> dict:
    out = {}
    exact_mask = None
    exact_hits = None
    for slot, preds in slot_preds.items():
        pred_arr = np.asarray(preds)
        target_arr = np.asarray(slot_targets[slot])
        valid = target_arr >= 0
        if valid.any():
            out[f"{slot}_acc"] = float((pred_arr[valid] == target_arr[valid]).mean())
        if exact_mask is None:
            exact_mask = valid.copy()
            exact_hits = pred_arr == target_arr
        else:
            exact_mask = exact_mask & valid
            exact_hits = exact_hits & (pred_arr == target_arr)
    if exact_mask is not None and exact_mask.any():
        out["exact_match_acc"] = float(exact_hits[exact_mask].mean())
    return out


def compute_yield_metrics(preds, targets) -> dict:
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.size == 0:
        return {}
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    out = {"mae": mae, "rmse": rmse}
    try:
        out["r2"] = float(metrics.r2_score(targets, preds))
    except Exception:
        out["r2"] = 0.0
    return out


# noinspection SpellCheckingInspection
def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred).reshape(-1)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


# noinspection SpellCheckingInspection
def calc_cliff_rmse(y_test_pred, y_test, cliff_mols_test=None, smiles_test=None,
                    y_train=None, smiles_train=None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred).reshape(-1) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        # noinspection PyUnresolvedReferences
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)
