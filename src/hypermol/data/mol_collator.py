
from typing import List

import torch
import lmdb
import pickle
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from hypermol.utils.debug import profile

from hypermol.data.preprocess import CompoundKit

atom_id_names = set(list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names)
bond_id_names = set(list(CompoundKit.bond_vocab_dict.keys()))

none_int_names = {
    'van_der_waals_radis', 'mass', 'atom_pos', 'atom_distances_2d', 'pair_angles',
    'triple_angles', 'bond_distances', 'bond_angles', 'edge_distances', 'atom_bond_distances',
    'bond_distances', 'label'
}
non_numpy_names = {
    'smiles',  # str
    'angles_atom_index',  # list
    'angles_bond_index'  # list
}

fingerprint_keys = {
    'morgan_fp', 'morgan2048_fp', 'maccs_fp', 'rdkit_fp',
    'torsion_fp', 'atom_pair_fp', 'layered_fp'
}


def _torsion_topology_mask(edges) -> torch.Tensor:
    """Mark edges whose two endpoints both have a non-edge neighbor."""
    edges_np = np.asarray(edges, dtype=np.int64)
    if edges_np.size == 0:
        return torch.zeros(0, dtype=torch.bool)
    edges_np = edges_np.reshape(-1, 2)
    adjacency = {}
    for i, j in edges_np:
        if i < 0 or j < 0:
            continue
        adjacency.setdefault(int(i), set()).add(int(j))
        adjacency.setdefault(int(j), set()).add(int(i))

    valid = []
    for i, j in edges_np:
        if i < 0 or j < 0:
            valid.append(False)
            continue
        i_neighbors = adjacency.get(int(i), set()) - {int(j)}
        j_neighbors = adjacency.get(int(j), set()) - {int(i)}
        valid.append(bool(i_neighbors) and bool(j_neighbors))
    return torch.tensor(valid, dtype=torch.bool)


def _batch_torsion_topology_mask(items, max_torsions: int) -> torch.Tensor:
    mask = torch.zeros(len(items), max_torsions, dtype=torch.bool)
    for row, item in enumerate(items):
        if 'edges' not in item or item['edges'] is None:
            continue
        item_mask = _torsion_topology_mask(item['edges'])
        width = min(max_torsions, int(item_mask.numel()))
        if width > 0:
            mask[row, :width] = item_mask[:width]
    return mask


# @profile
def collator_pretrain(items):
    """
    一个高性能、健壮且无冗余的 collator 函数，用于预训练。
    此版本完整实现了所有功能，并返回一个包含所有字段的扁平字典。
    """
    # --- 1. 确定批次的最大尺寸 & 初始化 ---
    max_len = max([len(item['atomic_num']) for item in items])
    has_edges = 'edges' in items[0] and items[0]['edges'] is not None and len(items[0]['edges']) > 0
    max_edges = max([len(item['edges']) for item in items]) if has_edges else 0
    max_angels = max([len(item['angles_atom_index']) for item in items])
    batch_size = len(items)

    batch = {}

    # --- 2. 定义白名单和处理规则 ---
    keys_to_process = (
            atom_id_names | bond_id_names | fingerprint_keys |
            {
                'atomic_num', 'atom_pos', 'edges',
                'mam_targets', 'angles_atom_index',
                'atom_distances_2d', 'bond_angles_bin', 'torsion_angles_bin',
                'atom_map_num',
            }
    )

    padding_rules = {
        **{name: (-1, True, torch.long) for name in atom_id_names},
        **{name: (-1, True, torch.long) for name in bond_id_names},
        'edge_types': (-1, True, torch.long),
        'edges': (-1, False, torch.long),
        'augmented_atoms': (-1, True, torch.long),
        'atom_pos': (0.0, False, torch.float),
        'mam_targets': (-1, True, torch.long),
        'coord_targets': (0.0, False, torch.float),
        'bond_pos': (0.0, False, torch.float),
        # 'atom_distances_2d': (-1.0, False, torch.float),
        # 'bond_distances_2d': (-1.0, False, torch.float),
        'atom_distances_2d': (-1.0, False, torch.long),
        'bond_angles_bin': (-5.0, False, torch.long),
        'torsion_angles_bin': (-5.0, False, torch.long),
        'angles_atom_index': (-1, False, torch.long),
        'atom_map_num': (0, False, torch.long),
    }

    # --- 3. 提取、转换和填充所有来自 Dataset 的数据 ---
    tensor_lists = {key: [] for key in keys_to_process if key in items[0]}

    for item in items:
        for key in tensor_lists.keys():
            if type(item[key]) == float:
                tensor_lists[key].append(torch.tensor(item[key], dtype=torch.float))
            else:
                # print(key)
                tensor_lists[key].append(torch.from_numpy(np.array(item[key])))

    for name, tensors in tensor_lists.items():
        if not tensors: continue

        if name in fingerprint_keys:
            batch[name] = torch.stack(tensors).float()
            continue

        if name in padding_rules:
            pad_val, plus_one, dtype = padding_rules[name]

            if name.endswith('_distances_2d'):
                target_len = max_len if 'atom' in name else max_edges
                padded_list = [F.pad(t, (0, target_len - t.shape[1], 0, target_len - t.shape[0]), value=pad_val) for t
                               in tensors]
                batch[name] = torch.stack(padded_list).to(dtype)
            elif name in bond_id_names:
                padded_list = []
                for t in tensors:
                    # 确认这是一个3D张量
                    if t.ndim != 3:
                        raise ValueError(
                            f"Expected a 3D tensor for bond path feature '{key}', but got {t.ndim} dimensions.")

                    num_atoms = t.shape[0]
                    # (右, 左, 下, 上) 的填充顺序
                    padding = (0, 0, 0, max_len - num_atoms, 0, max_len - num_atoms)
                    # 使用 0 进行填充，因为 0 通常是 embedding 的 padding_idx
                    padded_tensor = F.pad(t, padding, value=-1) + 1
                    padded_list.append(padded_tensor)

                batch[name] = torch.stack(padded_list)
            else:
                padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=pad_val)
                if plus_one:
                    padded_tensor += 1
                batch[name] = padded_tensor.to(dtype)
        else:
            # print(name)
            batch[name] = torch.stack(tensors).to(torch.float)

    # --- 4. 一次性准备所有会被复用的派生张量 ---
    atom_lengths = torch.tensor([len(item['atomic_num']) for item in items], dtype=torch.long)
    batch['atom_length'] = atom_lengths

    # # --- 5. 集中进行所有在线计算 (模型输入) ---
    # atom_pos_padded = batch['atom_pos']
    #
    # delta_coords = atom_pos_padded.unsqueeze(2) - atom_pos_padded.unsqueeze(1)
    # batch['atom_distances_3d'] = torch.sqrt(torch.sum(delta_coords ** 2, dim=-1) + 1e-6)



    # --- 6. 集中创建所有掩码 (Attention 和 Loss) ---
    if max_len > 0:
        atom_mask = (batch['atomic_num'] != 0)
        batch["atom_mask"] = atom_mask

        atom_range = torch.arange(max_len + 1, device=atom_lengths.device)
        effective_atom_lengths = atom_lengths + 1
        atom_padding_mask_1d = (atom_range.unsqueeze(0) < effective_atom_lengths.unsqueeze(1))
        batch["atom_attention_mask"] = (atom_padding_mask_1d.unsqueeze(2) * atom_padding_mask_1d.unsqueeze(1)).long()

        # b) Loss Masks for loss function
        pad_idx = 0
        mask_idx_final = 121  # 示例

        # b-1) MAM Loss Mask
        batch['mam_loss_mask'] = (batch['mam_targets'] != pad_idx)



    # --- 7. 重命名与整理 (保持不变) ---
    if 'bond_angles_bin' in batch:
        # 形状: [batch_size, max_angles]
        batch['angle_valid_mask'] = (batch['bond_angles_bin'] != -5)

    if 'torsion_angles_bin' in batch:
        # 形状: [batch_size, max_torsions]
        topology_mask = _batch_torsion_topology_mask(items, batch['torsion_angles_bin'].shape[1])
        batch['torsion_valid_mask'] = (batch['torsion_angles_bin'] != -5) & topology_mask

    # --- 8. 直接返回扁平字典 ---
    return batch

def collator_finetune(items):
    """
    一个高性能、健壮且无冗余的 collator 函数，用于预训练。
    此版本完整实现了所有功能，并返回一个包含所有字段的扁平字典。
    """
    # --- 1. 确定批次的最大尺寸 & 初始化 ---
    max_len = max([len(item['atomic_num']) for item in items])
    has_edges = 'edges' in items[0] and items[0]['edges'] is not None and len(items[0]['edges']) > 0
    max_edges = max([len(item['edges']) for item in items]) if has_edges else 0
    max_angels = max([len(item['angles_atom_index']) for item in items])
    batch_size = len(items)

    batch = {}

    # --- 2. 定义白名单和处理规则 ---
    keys_to_process = (
            atom_id_names | bond_id_names | fingerprint_keys |
            {
                'atomic_num', 'atom_pos', 'edges',
                'mam_targets', 'angles_atom_index',
                'atom_distances_2d', 'label',
                'atom_map_num',
            }
    )

    padding_rules = {
        **{name: (-1, True, torch.long) for name in atom_id_names},
        **{name: (-1, True, torch.long) for name in bond_id_names},
        'edge_types': (-1, True, torch.long),
        'edges': (-1, False, torch.long),
        'atom_pos': (0.0, False, torch.float),
        'mam_targets': (-1, True, torch.long),
        # 'coord_targets': (0.0, False, torch.float),
        # 'bond_pos': (0.0, False, torch.float),
        'atom_distances_2d': (-1.0, False, torch.long),
        # 'bond_angles_bin': (-5.0, False, torch.long),
        # 'torsion_angles_bin': (-5.0, False, torch.long),
        'angles_atom_index': (-1, False, torch.long),
        'atom_map_num': (0, False, torch.long),
    }

    # --- 3. 提取、转换和填充所有来自 Dataset 的数据 ---
    tensor_lists = {key: [] for key in keys_to_process if key in items[0]}

    for item in items:
        for key in tensor_lists.keys():
            if type(item[key]) == float:
                tensor_lists[key].append(torch.tensor(item[key], dtype=torch.float))
            else:
                # print(key)
                tensor_lists[key].append(torch.from_numpy(np.array(item[key])))

    for name, tensors in tensor_lists.items():
        if not tensors: continue

        if name in fingerprint_keys:
            batch[name] = torch.stack(tensors).float()
            continue

        if name in padding_rules:
            pad_val, plus_one, dtype = padding_rules[name]

            if name.endswith('_distances_2d'):
                target_len = max_len if 'atom' in name else max_edges
                padded_list = [F.pad(t, (0, target_len - t.shape[1], 0, target_len - t.shape[0]), value=pad_val) for t
                               in tensors]
                batch[name] = torch.stack(padded_list).to(dtype)
            elif name in bond_id_names:
                padded_list = []
                for t in tensors:
                    # 确认这是一个3D张量
                    if t.ndim != 3:
                        raise ValueError(
                            f"Expected a 3D tensor for bond path feature '{key}', but got {t.ndim} dimensions.")

                    num_atoms = t.shape[0]
                    # (右, 左, 下, 上) 的填充顺序
                    padding = (0, 0, 0, max_len - num_atoms, 0, max_len - num_atoms)
                    # 使用 0 进行填充，因为 0 通常是 embedding 的 padding_idx
                    padded_tensor = F.pad(t, padding, value=-1) + 1
                    padded_list.append(padded_tensor)

                batch[name] = torch.stack(padded_list)
            else:
                padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=pad_val)
                if plus_one:
                    padded_tensor += 1
                batch[name] = padded_tensor.to(dtype)
        else:
            # print(name)
            batch[name] = torch.stack(tensors).to(torch.float)

    # --- 4. 一次性准备所有会被复用的派生张量 ---
    atom_lengths = torch.tensor([len(item['atomic_num']) for item in items], dtype=torch.long)
    batch['atom_length'] = atom_lengths

    # # --- 5. 集中进行所有在线计算 (模型输入) ---
    # atom_pos_padded = batch['atom_pos']
    #
    # delta_coords = atom_pos_padded.unsqueeze(2) - atom_pos_padded.unsqueeze(1)
    # batch['atom_distances_3d'] = torch.sqrt(torch.sum(delta_coords ** 2, dim=-1) + 1e-6)



    # --- 6. 集中创建所有掩码 (Attention 和 Loss) ---
    if max_len > 0:
        atom_mask = (batch['atomic_num'] != 0)
        batch["atom_mask"] = atom_mask

        atom_range = torch.arange(max_len + 1, device=atom_lengths.device)
        effective_atom_lengths = atom_lengths + 1
        atom_padding_mask_1d = (atom_range.unsqueeze(0) < effective_atom_lengths.unsqueeze(1))
        batch["atom_attention_mask"] = (atom_padding_mask_1d.unsqueeze(2) * atom_padding_mask_1d.unsqueeze(1)).long()

    # --- 8. 直接返回扁平字典 ---
    return batch


class MoleculeCollator:
    """
    底层组件：只负责将一批【任何形式】的分子特征字典进行填充（padding）。
    它能智能地处理预训练任务中新增的标签。
    """

    def __call__(self, molecule_feature_dicts, mode):
        num_molecules = len(molecule_feature_dicts)
        if num_molecules == 0:
            return {}

        # # 检查是否为预训练模式（通过是否存在mam_targets键）
        # is_pretraining = 'mam_targets' in molecule_feature_dicts[0]
        #
        # # 找到批次中最大的分子（原子数最多）
        # max_nodes = max(d['atom_features'].shape[0] for d in molecule_feature_dicts)
        # atom_feature_dim = molecule_feature_dicts[0]['atom_features'].shape[1]


        # 构建返回的字典
        if mode == 'pretrain':
            batch = collator_pretrain(molecule_feature_dicts)
        else:
            batch = collator_finetune(molecule_feature_dicts)

        return batch


# class MoleculeCollator:
#     """
#     底层组件：只负责将一批【任何形式】的分子特征字典进行填充（padding）。
#     它能智能地处理预训练任务中新增的标签。
#     """
#
#     def __call__(self, molecule_feature_dicts):
#         num_molecules = len(molecule_feature_dicts)
#         if num_molecules == 0:
#             return {}
#
#         # 检查是否为预训练模式（通过是否存在mam_targets键）
#         is_pretraining = 'mam_targets' in molecule_feature_dicts[0]
#
#         # 找到批次中最大的分子（原子数最多）
#         max_nodes = max(d['atom_features'].shape[0] for d in molecule_feature_dicts)
#         atom_feature_dim = molecule_feature_dicts[0]['atom_features'].shape[1]
#
#         # 初始化所有需要的张量
#         padded_atom_features = torch.zeros(num_molecules, max_nodes, atom_feature_dim, dtype=torch.float)
#         padded_adj_matrix = torch.zeros(num_molecules, max_nodes, max_nodes, dtype=torch.float)
#         mask = torch.zeros(num_molecules, max_nodes, dtype=torch.bool)
#
#         padded_mam_targets = torch.full((num_molecules, max_nodes), -100, dtype=torch.long) if is_pretraining else None
#
#         # 循环填充每个分子
#         for i, d in enumerate(molecule_feature_dicts):
#             num_nodes = d['atom_features'].shape[0]
#             padded_atom_features[i, :num_nodes] = d['atom_features']
#             mask[i, :num_nodes] = True
#
#             edge_index = d.get('edge_index')
#             if edge_index is not None and edge_index.numel() > 0:
#                 padded_adj_matrix[i, edge_index[0], edge_index[1]] = 1
#
#             if is_pretraining:
#                 padded_mam_targets[i, :num_nodes] = d['mam_targets']
#
#         # 构建返回的字典
#         batch = {
#             "padded_atom_features": padded_atom_features,
#             "padded_adj_matrix": padded_adj_matrix,
#             "mask": mask
#         }
#         if is_pretraining:
#             batch["padded_mam_targets"] = padded_mam_targets
#
#         return batch
