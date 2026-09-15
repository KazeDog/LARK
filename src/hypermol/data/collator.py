from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from hypermol.utils.debug import profile

from hypermol.data.mol_collator import MoleculeCollator


def _molecule_map_list(molecule: dict) -> dict[int, int]:
    """Return ``map_number -> local_atom_index`` for old and compact stores.

    Historical pretraining stores materialized ``map_list`` while the compact
    USPTO condition store keeps only the atom-aligned ``atom_map_num`` array.
    Treating the latter as unmapped silently produced a zero-width reaction
    canvas in downstream tasks.  Reconstructing the dictionary here keeps the
    collator contract identical for both store formats.
    """

    existing = molecule.get("map_list")
    if existing:
        return {int(map_num): int(local_idx) for map_num, local_idx in existing.items()}

    atom_map_numbers = molecule.get("atom_map_num")
    if atom_map_numbers is None:
        return {}
    if torch.is_tensor(atom_map_numbers):
        atom_map_numbers = atom_map_numbers.detach().cpu().tolist()
    else:
        atom_map_numbers = np.asarray(atom_map_numbers).reshape(-1).tolist()

    reconstructed: dict[int, int] = {}
    for local_idx, value in enumerate(atom_map_numbers):
        map_num = int(value)
        if map_num > 0:
            reconstructed.setdefault(map_num, int(local_idx))
    return reconstructed

@profile
class HypergraphCollator:
    """
    顶层Collator：负责完整的批处理流程。
    - 它接收一批由Dataset增强过的反应样本。
    - 它构建超图结构（关联矩阵）。
    - 它调用底层的MoleculeCollator来处理所有分子层面的填充。
    - 它自己负责反应级数据（如R矩阵）的填充。
    - 最后，它将所有部分组装成一个最终的、可直接输入模型的batch字典。
    """

    def __init__(self, mode):
        """初始化时，创建一个可重用的MoleculeCollator实例。"""
        self.molecule_collator = MoleculeCollator()
        self.mode = mode

    def __call__(self, augmented_reaction_batch):
        """
        将一批反应样本转换为一个批处理张量字典。
        Args:
            augmented_reaction_batch (list[dict]): 一个列表，其中每个字典代表一个由ReactionDataset.__getitem__处理过的反应样本。
        Returns:
            dict: 一个包含所有模型所需张量的最终batch字典。
        """
        # 1. 收集所有已经由Dataset增强过的分子特征字典
        molecule_feature_dicts = []
        labels = []
        num_reactants_per_reaction = [len(r['reactants_features']) for r in augmented_reaction_batch]
        num_products_per_reaction = [len(r['products_features']) for r in augmented_reaction_batch]

        for reaction in augmented_reaction_batch:
            molecule_feature_dicts.extend(reaction['reactants_features'])
            molecule_feature_dicts.extend(reaction['products_features'])
            if self.mode == 'finetune':
                labels.append(reaction['label'])

        if self.mode == 'finetune':
            labels = torch.tensor(np.array(labels, dtype=int), dtype=torch.long)
        map_lists = [_molecule_map_list(d) for d in molecule_feature_dicts]

        num_molecules_per_reaction = [len(r['reactants_features']) + len(r['products_features']) for r in
                                      augmented_reaction_batch]
        # `batch_vec` 标记每个分子属于哪个反应 (0, 0, 1, 1, 1, ...)
        batch_vec = torch.repeat_interleave(
            torch.arange(len(augmented_reaction_batch)),
            torch.tensor(num_molecules_per_reaction)
        ).long()

        # 检查是否为预训练模式（通过样本中是否存在R矩阵目标键）
        # is_pretraining = 'r_matrix_targets' in augmented_reaction_batch[0]
        # if is_pretraining:
        #     self.mode = 'pretrain'

        # 2. 调用MoleculeCollator来处理所有分子层面的填充
        molecule_batch = self.molecule_collator(molecule_feature_dicts, self.mode)

        # 3. 填充反应级的R矩阵标签和掩码 (如果存在)
        if self.mode == 'pretrain':
            # a. 找到批次中最大的R矩阵维度以确定填充尺寸
            max_r_dim = max(
                r['r_matrix_targets'].shape[0] for r in augmented_reaction_batch) if augmented_reaction_batch else 0

            # b. 创建用于填充的空张量
            padded_r_matrix_targets = torch.zeros(len(augmented_reaction_batch), max_r_dim, max_r_dim,
                                                  dtype=torch.float)
            padded_r_matrix_mask = torch.zeros(len(augmented_reaction_batch), max_r_dim, max_r_dim, dtype=torch.bool)

            # c. 循环填充每个反应的R矩阵数据
            for i, r in enumerate(augmented_reaction_batch):
                dim = r['r_matrix_targets'].shape[0]
                padded_r_matrix_targets[i, :dim, :dim] = r['r_matrix_targets']
                padded_r_matrix_mask[i, :dim, :dim] = r['r_matrix_mask']

        else:
            # 如果不是预训练，我们也需要计算一个 max_r_dim 以创建画布掩码
            # 这可以通过检查所有 map_list 来实现
            all_map_nums = [mn for ml in map_lists for mn in ml.keys()]
            max_r_dim = max(all_map_nums) if all_map_nums else 0

        # --- 核心新增: 创建反应画布掩码 ---
        batch_size = len(augmented_reaction_batch)
        # 形状: [batch_size, max_r_dim]
        reaction_canvas_mask = torch.zeros(batch_size, max_r_dim, dtype=torch.bool)

        # 遍历每个分子来填充这个掩码
        for mol_idx, mol_map_list in enumerate(map_lists):
            reaction_idx = batch_vec[mol_idx].item()
            for map_num in mol_map_list.keys():
                if (map_num - 1) < max_r_dim:
                    reaction_canvas_mask[reaction_idx, map_num - 1] = True

        # 4. 构建超图结构（关联矩阵 H_in, H_out）和监督任务的标签
        num_total_nodes = len(molecule_feature_dicts)
        num_edges = len(augmented_reaction_batch)
        H_in_rows, H_in_cols, H_out_rows, H_out_cols = [], [], [], []
        current_node_idx = 0
        for i, (num_r, num_p) in enumerate(zip(num_reactants_per_reaction, num_products_per_reaction)):
            reactants_indices = range(current_node_idx, current_node_idx + num_r)
            H_in_rows.extend(reactants_indices);
            H_in_cols.extend([i] * num_r)
            current_node_idx += num_r
            products_indices = range(current_node_idx, current_node_idx + num_p)
            H_out_rows.extend(products_indices);
            H_out_cols.extend([i] * num_p)
            current_node_idx += num_p

        # 确定非零元素的数量
        nnz_in = len(H_in_rows)
        nnz_out = len(H_out_rows)

        # 创建正确大小的 `values` 张量
        values_in = torch.ones(nnz_in)
        values_out = torch.ones(nnz_out)

        # 确保索引是 LongTensor
        indices_in = torch.tensor([H_in_rows, H_in_cols], dtype=torch.long)
        indices_out = torch.tensor([H_out_rows, H_out_cols], dtype=torch.long)

        # 使用新的 `values` 张量创建稀疏张量
        H_in = torch.sparse_coo_tensor(indices_in, values_in, (num_total_nodes, num_edges))
        H_out = torch.sparse_coo_tensor(indices_out, values_out, (num_total_nodes, num_edges))

        # 为监督学习任务构建标签（0代表反应物，1代表产物）
        supervised_labels = torch.zeros(num_total_nodes, dtype=torch.float)
        current_node_idx = 0
        for i, (num_r, num_p) in enumerate(zip(num_reactants_per_reaction, num_products_per_reaction)):
            product_start_idx = current_node_idx + num_r
            supervised_labels[product_start_idx: product_start_idx + num_p] = 1.0
            current_node_idx += num_r + num_p

        # 5. 整合所有结果到一个最终的batch字典中
        final_batch = {
            "H_in": H_in,
            "H_out": H_out,
            "supervised_labels": supervised_labels,
            'map_lists': map_lists,
            'batch_vec': batch_vec,
            'reaction_canvas_mask': reaction_canvas_mask,
            'labels': labels,
        }
        # 将来自MoleculeCollator的结果合并进来
        final_batch.update(molecule_batch)

        # 如果是预训练模式，再将R矩阵相关的结果合并进来
        if self.mode == 'pretrain':
            final_batch["r_matrix_targets"] = padded_r_matrix_targets
            final_batch["r_matrix_mask"] = padded_r_matrix_mask

        return final_batch

class HypergraphCollatorForGT:
    """
    一个独立的顶层Collator，专门为基于【稠密序列Transformer】的模型（如HyperGT）准备数据。
    它负责将由Dataset处理过的反应样本，转换成模型所需的完整批处理字典。
    """

    def __init__(self, mode: str):
        """
        初始化。
        Args:
            mode (str): 'pretrain' 或 'finetune'/'supervised'。
        """
        self.mode = mode
        # 它内部包含一个MoleculeCollator实例来处理原子级数据
        self.molecule_collator = MoleculeCollator()

    def _combine_be_matrices(self, dicts_list, max_dim):
        """一个内部辅助函数，用于合并一组分子的BE矩阵。"""
        combined = np.zeros((max_dim, max_dim), dtype=np.float32)
        for d in dicts_list:
            if 'be_matrix' in d:
                mat = d['be_matrix']
                # 确保mat是numpy数组
                if torch.is_tensor(mat):
                    mat = mat.numpy()
                combined[:mat.shape[0], :mat.shape[1]] += mat
        return combined

    # @profile
    def __call__(self, augmented_reaction_batch: list[dict]):
        """
        将一批反应样本转换为一个批处理张量字典。
        """
        # A dual-view dataset returns one wrapper per chemical reaction.  The
        # model still consumes a flat batch, while these local pair ids retain
        # the exact P->R / R->P correspondence for the consistency objective.
        expanded_batch = []
        next_pair_index = 0
        for sample in augmented_reaction_batch:
            if not sample:
                continue
            views = sample.get("pretrain_views")
            if views is None:
                expanded_batch.append(sample)
                continue
            if len(views) != 2 or any(not view for view in views):
                continue
            direction_ids = {int(view.get("view_direction_id", -1)) for view in views}
            if direction_ids != {0, 1}:
                raise ValueError("Each dual-view sample must contain direction ids 0 and 1 exactly once.")
            for view in views:
                flat_view = dict(view)
                flat_view["_view_pair_index"] = next_pair_index
                expanded_batch.append(flat_view)
            next_pair_index += 1
        augmented_reaction_batch = expanded_batch

        # 过滤掉可能由Dataset返回的空/无效样本
        augmented_reaction_batch = [
            r
            for r in augmented_reaction_batch
            if r and (r.get("model_input_features") or r.get('reactants_features'))
        ]
        if not augmented_reaction_batch:
            return {}

        paired_flags = ["_view_pair_index" in sample for sample in augmented_reaction_batch]
        if any(paired_flags) and not all(paired_flags):
            raise ValueError("A pretraining batch cannot mix dual-view and single-view samples.")
        is_dual_view_batch = bool(paired_flags and all(paired_flags))

        batch_size = len(augmented_reaction_batch)
        is_pretraining = self.mode == 'pretrain'
        is_finetuning = self.mode == 'finetune'

        # 1. 收集所有分子特征字典和反应级元信息
        molecule_feature_dicts = []
        explicit_objective_contract = all("model_input_features" in r for r in augmented_reaction_batch)
        model_input_roles = []
        model_input_role_ids = []
        if explicit_objective_contract:
            num_nodes_per_reaction = [len(r["model_input_features"]) for r in augmented_reaction_batch]
            for reaction in augmented_reaction_batch:
                features = reaction["model_input_features"]
                roles = list(reaction.get("model_input_roles", []))
                if len(roles) != len(features):
                    raise ValueError("model_input_roles must align with model_input_features.")
                role_ids = [int(value) for value in reaction.get("model_input_role_ids", [0] * len(features))]
                if len(role_ids) != len(features):
                    raise ValueError("model_input_role_ids must align with model_input_features.")
                if any(value < 0 for value in role_ids):
                    raise ValueError("model_input_role_ids must be non-negative.")
                molecule_feature_dicts.extend(features)
                model_input_roles.append(roles)
                model_input_role_ids.append(role_ids)
            num_reactants_per_reaction = [sum(role == "reactant" for role in roles) for roles in model_input_roles]
            num_products_per_reaction = [sum(role == "product" for role in roles) for roles in model_input_roles]
        else:
            num_reactants_per_reaction = [len(r['reactants_features']) for r in augmented_reaction_batch]
            num_products_per_reaction = [len(r['products_features']) for r in augmented_reaction_batch]
            num_context_per_reaction = [len(r.get("context_features", [])) for r in augmented_reaction_batch]
            num_nodes_per_reaction = [
                r + p + c
                for r, p, c in zip(
                    num_reactants_per_reaction,
                    num_products_per_reaction,
                    num_context_per_reaction,
                )
            ]
            for reaction in augmented_reaction_batch:
                context_features = list(reaction.get("context_features", []))
                context_role_ids = [
                    int(value) for value in reaction.get("context_role_ids", [])
                ]
                if len(context_role_ids) < len(context_features):
                    context_role_ids.extend([0] * (len(context_features) - len(context_role_ids)))
                if len(context_role_ids) > len(context_features):
                    raise ValueError("context_role_ids must align with context_features.")
                if any(value < 0 for value in context_role_ids):
                    raise ValueError("context_role_ids must be non-negative.")
                molecule_feature_dicts.extend(reaction['reactants_features'])
                molecule_feature_dicts.extend(reaction['products_features'])
                molecule_feature_dicts.extend(context_features)
                model_input_roles.append(
                    ["reactant"] * len(reaction['reactants_features'])
                    + ["product"] * len(reaction['products_features'])
                    + ["context"] * len(context_features)
                )
                model_input_role_ids.append(
                    [0] * (
                        len(reaction["reactants_features"])
                        + len(reaction["products_features"])
                    )
                    + context_role_ids
                )

        # 2. 委托 MoleculeCollator 处理所有原子级的批处理
        # molecule_batch 包含了 'padded_atom_features', 'atom_mask' 等所有原子级信息
        molecule_batch = self.molecule_collator(molecule_feature_dicts, self.mode)

        # 3. 构建Transformer序列长度和掩码
        max_nodes_in_reaction = max(num_nodes_per_reaction) if num_nodes_per_reaction else 0
        sequence_length = max_nodes_in_reaction + 1
        src_key_padding_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
        for i in range(batch_size):
            num_nodes = num_nodes_per_reaction[i]
            src_key_padding_mask[i, :num_nodes] = False  # 标记真实分子节点
            src_key_padding_mask[i, max_nodes_in_reaction] = False  # 标记真实超边节点

        # 4. 构建批处理的 H_in, H_out (用于位置编码)
        padded_H_in = torch.zeros(batch_size, max_nodes_in_reaction, 1, dtype=torch.float)
        padded_H_out = torch.zeros(batch_size, max_nodes_in_reaction, 1, dtype=torch.float)
        padded_H_context = torch.zeros(batch_size, max_nodes_in_reaction, 1, dtype=torch.float)
        padded_context_role_ids = torch.zeros(
            batch_size,
            max_nodes_in_reaction,
            dtype=torch.long,
        )
        for i, (roles, role_ids) in enumerate(zip(model_input_roles, model_input_role_ids)):
            for node_idx, (role, role_id) in enumerate(zip(roles, role_ids)):
                if role == "reactant":
                    padded_H_in[i, node_idx, 0] = 1.0
                elif role == "product":
                    padded_H_out[i, node_idx, 0] = 1.0
                elif role == "context":
                    padded_H_context[i, node_idx, 0] = 1.0
                    padded_context_role_ids[i, node_idx] = int(role_id)
                else:
                    raise ValueError(f"Unsupported model input role: {role}")

        # 5. 构建RMatrixHead等高级预测头所需的映射信息
        map_lists = [_molecule_map_list(d) for d in molecule_feature_dicts]
        batch_vec = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(num_nodes_per_reaction)).long()

        canvas_index_lists = []
        canvas_map_ids = []
        if explicit_objective_contract:
            num_canvas_atoms_per_sample = []
            for reaction, features in zip(augmented_reaction_batch, num_nodes_per_reaction):
                item_canvas_lists = list(reaction.get("canvas_index_lists", []))
                if len(item_canvas_lists) != features:
                    raise ValueError("canvas_index_lists must contain one sequence per model input molecule.")
                canvas_index_lists.extend(item_canvas_lists)
                item_map_ids = [int(value) for value in reaction.get("canvas_map_ids", [])]
                canvas_map_ids.append(item_map_ids)
                num_canvas_atoms = int(reaction.get("num_canvas_atoms", len(item_map_ids)))
                if num_canvas_atoms != len(item_map_ids):
                    raise ValueError("num_canvas_atoms must equal len(canvas_map_ids) for the local objective contract.")
                num_canvas_atoms_per_sample.append(num_canvas_atoms)
            max_r_dim = max(num_canvas_atoms_per_sample, default=0)
        else:
            all_map_nums = [int(mn) for ml in map_lists for mn in ml.keys()]
            max_r_dim = max(all_map_nums) if all_map_nums else 0
            if (is_pretraining or is_finetuning) and 'r_matrix_targets' in augmented_reaction_batch[0]:
                max_r_dim_from_targets = max(r['r_matrix_targets'].shape[0] for r in augmented_reaction_batch)
                max_r_dim = max(max_r_dim, max_r_dim_from_targets)
            num_canvas_atoms_per_sample = [max_r_dim] * batch_size
            canvas_map_ids = [list(range(1, max_r_dim + 1)) for _ in range(batch_size)]
            for mol_dict, mol_map_list in zip(molecule_feature_dicts, map_lists):
                atom_count = int(len(mol_dict.get("atomic_num", [])))
                atom_to_canvas = [-1] * atom_count
                for map_num, local_idx in mol_map_list.items():
                    if 0 < int(map_num) <= max_r_dim and 0 <= int(local_idx) < atom_count:
                        atom_to_canvas[int(local_idx)] = int(map_num) - 1
                canvas_index_lists.append(atom_to_canvas)

        reaction_canvas_mask = torch.zeros(batch_size, max_r_dim, dtype=torch.bool)
        mol_offset = 0
        for reaction_idx, num_molecules in enumerate(num_nodes_per_reaction):
            for atom_to_canvas in canvas_index_lists[mol_offset : mol_offset + num_molecules]:
                for canvas_idx in atom_to_canvas:
                    if 0 <= int(canvas_idx) < max_r_dim:
                        reaction_canvas_mask[reaction_idx, int(canvas_idx)] = True
            mol_offset += num_molecules

        pair_prior = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.float)
        pair_prior_observed_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
        if explicit_objective_contract:
            for i, reaction in enumerate(augmented_reaction_batch):
                prior = reaction.get("pair_prior")
                observed = reaction.get("pair_prior_observed_mask")
                if prior is None:
                    continue
                prior = torch.as_tensor(prior, dtype=torch.float)
                dim = int(num_canvas_atoms_per_sample[i])
                if tuple(prior.shape) != (dim, dim):
                    raise ValueError(f"pair_prior shape {tuple(prior.shape)} does not match local canvas {(dim, dim)}.")
                pair_prior[i, :dim, :dim] = prior
                if observed is None:
                    pair_prior_observed_mask[i, :dim, :dim] = reaction_canvas_mask[i, :dim, None] & reaction_canvas_mask[i, None, :dim]
                else:
                    observed = torch.as_tensor(observed, dtype=torch.bool)
                    if tuple(observed.shape) != (dim, dim):
                        raise ValueError("pair_prior_observed_mask must match pair_prior shape.")
                    pair_prior_observed_mask[i, :dim, :dim] = observed
        elif is_pretraining:
            # Historical reconstruction prior: complete reactant+product BE sum.
            for i, r_sample in enumerate(augmented_reaction_batch):
                be_react_np = self._combine_be_matrices(r_sample['reactants_features'], max_r_dim)
                be_prod_np = self._combine_be_matrices(r_sample['products_features'], max_r_dim)
                pair_prior[i] = torch.from_numpy(be_react_np + be_prod_np)
                pair_prior_observed_mask[i] = reaction_canvas_mask[i, :, None] & reaction_canvas_mask[i, None, :]

        padded_be_matrix = pair_prior  # legacy alias

        # 6. 组装最终的batch字典
        final_batch = {}
        final_batch.update(molecule_batch)
        final_batch.update({
            "src_key_padding_mask": src_key_padding_mask,
            "padded_H_in": padded_H_in,
            "padded_H_out": padded_H_out,
            "padded_H_context": padded_H_context,
            "padded_context_role_ids": padded_context_role_ids,
            "context_role_ids": torch.tensor(
                [role_id for role_ids in model_input_role_ids for role_id in role_ids],
                dtype=torch.long,
            ),
            "num_nodes_per_sample": torch.tensor(num_nodes_per_reaction),
            "max_nodes_in_batch": max_nodes_in_reaction,
            "map_lists": map_lists,
            "canvas_index_lists": canvas_index_lists,
            "canvas_map_ids": canvas_map_ids,
            "num_canvas_atoms_per_sample": torch.tensor(num_canvas_atoms_per_sample, dtype=torch.long),
            "model_input_roles": model_input_roles,
            "context_molecule_mask": torch.tensor(
                [role == "context" for roles in model_input_roles for role in roles],
                dtype=torch.bool,
            ),
            "core_molecule_mask": torch.tensor(
                [role != "context" for roles in model_input_roles for role in roles],
                dtype=torch.bool,
            ),
            "batch_vec": batch_vec,
            "reaction_canvas_mask": reaction_canvas_mask,
            "pair_prior": pair_prior,
            "pair_prior_observed_mask": pair_prior_observed_mask,
            "padded_be_matrix": padded_be_matrix,
        })

        if is_dual_view_batch:
            final_batch["view_pair_index"] = torch.tensor(
                [int(sample["_view_pair_index"]) for sample in augmented_reaction_batch],
                dtype=torch.long,
            )
            final_batch["view_direction_id"] = torch.tensor(
                [int(sample["view_direction_id"]) for sample in augmented_reaction_batch],
                dtype=torch.long,
            )
            final_batch["view_direction"] = [
                str(sample.get("view_direction", "")) for sample in augmented_reaction_batch
            ]
            final_batch["num_view_pairs"] = int(next_pair_index)

        condition_dim = 514
        for sample in augmented_reaction_batch:
            value = sample.get("condition_vector")
            if torch.is_tensor(value):
                condition_dim = int(value.numel())
                break
        condition_vectors = []
        has_condition_vectors = []
        for sample in augmented_reaction_batch:
            value = sample.get("condition_vector")
            if torch.is_tensor(value):
                condition_vectors.append(value.float().view(-1))
            else:
                condition_vectors.append(torch.zeros(condition_dim, dtype=torch.float32))
            has_condition_vectors.append(bool(sample.get("has_condition_vector", False)))
        final_batch["condition_vector"] = torch.stack(condition_vectors, dim=0)
        final_batch["has_condition_vector"] = torch.tensor(has_condition_vectors, dtype=torch.bool)
        if "unmapped_components" in augmented_reaction_batch[0]:
            final_batch["unmapped_components"] = [sample.get("unmapped_components", "") for sample in augmented_reaction_batch]
        if "context_skipped_components" in augmented_reaction_batch[0]:
            final_batch["context_skipped_components"] = [
                list(sample.get("context_skipped_components", []))
                for sample in augmented_reaction_batch
            ]
        if "target_projection_audit" in augmented_reaction_batch[0]:
            final_batch["target_projection_audit"] = [
                sample.get("target_projection_audit", {}) for sample in augmented_reaction_batch
            ]

        # 7. 根据模式，添加并填充特定任务的标签
        if is_pretraining:
            if 'r_matrix_targets' in augmented_reaction_batch[0] or 'delta_be_targets' in augmented_reaction_batch[0]:
                delta_be_targets = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.float)
                train_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
                eval_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
                valid_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
                for i, reaction in enumerate(augmented_reaction_batch):
                    target = torch.as_tensor(
                        reaction.get("delta_be_targets", reaction.get("r_matrix_targets")),
                        dtype=torch.float,
                    )
                    dim = int(target.shape[0])
                    if target.ndim != 2 or target.shape[1] != dim or dim > max_r_dim:
                        raise ValueError("Delta-BE target must be square and fit the input-derived reaction canvas.")
                    legacy_mask = reaction.get("r_matrix_mask")
                    item_train_mask = reaction.get("r_matrix_train_mask", reaction.get("train_mask", legacy_mask))
                    item_eval_mask = reaction.get("r_matrix_eval_mask", reaction.get("eval_mask", item_train_mask))
                    item_valid_mask = reaction.get(
                        "r_matrix_valid_mask",
                        reaction.get("valid_pair_mask", reaction.get("valid_mask", item_eval_mask)),
                    )
                    delta_be_targets[i, :dim, :dim] = target
                    train_mask[i, :dim, :dim] = torch.as_tensor(item_train_mask, dtype=torch.bool)
                    eval_mask[i, :dim, :dim] = torch.as_tensor(item_eval_mask, dtype=torch.bool)
                    valid_mask[i, :dim, :dim] = torch.as_tensor(item_valid_mask, dtype=torch.bool)
                final_batch.update(
                    {
                        "delta_be_targets": delta_be_targets,
                        "r_matrix_targets": delta_be_targets,  # legacy alias
                        "r_matrix_train_mask": train_mask,
                        "r_matrix_eval_mask": eval_mask,
                        "r_matrix_valid_mask": valid_mask,
                        "valid_pair_mask": valid_mask,
                        "train_mask": train_mask,
                        "eval_mask": eval_mask,
                        "valid_mask": valid_mask,
                        "r_matrix_mask": train_mask,  # legacy alias
                    }
                )
            # (在这里可以添加其他预训练任务标签的填充逻辑, e.g., 几何任务)

        elif is_finetuning:
            # 填充微调任务所需的标签
            # a. 超边级任务标签
            if 'label' in augmented_reaction_batch[0]:
                labels_list = [r.get('label', -1) for r in augmented_reaction_batch]
                final_batch['labels'] = torch.from_numpy(np.array(labels_list, dtype=np.int64))

            # b. 逆合成R矩阵任务标签 (与预训练格式相同)
            if 'r_matrix_targets' in augmented_reaction_batch[0]:
                padded_r_matrix_targets = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.float)
                padded_r_matrix_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
                for i, r in enumerate(augmented_reaction_batch):
                    dim = r['r_matrix_targets'].shape[0]
                    padded_r_matrix_targets[i, :dim, :dim] = r['r_matrix_targets']
                    padded_r_matrix_mask[i, :dim, :dim] = r['r_matrix_mask']
                final_batch["r_matrix_targets"] = padded_r_matrix_targets
                final_batch["r_matrix_mask"] = padded_r_matrix_mask

            # c. 逆合成原子身份任务标签
            if 'atom_identity_targets' in augmented_reaction_batch[0]:
                atom_identity_targets_list = [r['atom_identity_targets'] for r in augmented_reaction_batch]
                padded_atom_identity_targets = torch.nn.utils.rnn.pad_sequence(
                    atom_identity_targets_list, batch_first=True, padding_value=-100
                )
                final_batch['atom_identity_targets'] = padded_atom_identity_targets

        return final_batch


class SynthesisCollatorForGT:
    """
    一个统一的、适配HyperGT模型的顶层Collator，
    能够同时为【正向合成】和【逆向合成】微调任务准备数据。
    """

    def __init__(self, task: str, mode: str = 'finetune'):
        """
        初始化。
        Args:
            task (str): 'forwardsynthesis' 或 'retrosynthesis'。
            mode (str): 'finetune' 或 'pretrain' (尽管这个collator主要为finetune设计)。
        """
        self.task = task
        if self.task not in ['forwardsynthesis', 'retrosynthesis']:
            raise ValueError("任务(task)必须是 'forwardsynthesis' 或 'retrosynthesis'")

        self.mode = mode
        self.molecule_collator = MoleculeCollator()

    def _combine_be_matrices(self, dicts_list, max_dim):
        """一个内部辅助函数，用于合并一组分子的BE矩阵。"""
        combined = np.zeros((max_dim, max_dim), dtype=np.float32)
        for d in dicts_list:
            if 'be_matrix' in d:
                mat = d['be_matrix']
                if torch.is_tensor(mat): mat = mat.numpy()
                combined[:mat.shape[0], :mat.shape[1]] += mat
        return combined

    def __call__(self, reaction_batch: list[dict]):
        """
        将一批合成任务样本转换为一个批处理张量字典。
        """
        # 过滤掉由Dataset返回的、可能因加载失败而产生的空字典
        reaction_batch = [r for r in reaction_batch if r and r.get('input_features')]
        if not reaction_batch:
            return {}

        batch_size = len(reaction_batch)

        # 1. 收集【输入数据】
        #    'input_features' 键由 SynthesisDataset 根据task准备好
        #    正向合成时，这里是反应物；逆向合成时，这里是产物。
        molecule_feature_dicts = []
        num_nodes_per_reaction = [len(r['input_features']) for r in reaction_batch]

        for reaction in reaction_batch:
            molecule_feature_dicts.extend(reaction['input_features'])

        # 2. 委托 MoleculeCollator 处理所有【输入分子】的原子级数据
        molecule_batch = self.molecule_collator(molecule_feature_dicts, self.mode)

        # 3. 构建Transformer序列长度和掩码
        max_nodes_in_reaction = max(num_nodes_per_reaction) if num_nodes_per_reaction else 0
        sequence_length = max_nodes_in_reaction + 1  # +1 for the hyperedge
        src_key_padding_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
        for i in range(batch_size):
            num_nodes = num_nodes_per_reaction[i]
            src_key_padding_mask[i, :num_nodes] = False  # 标记真实输入节点
            src_key_padding_mask[i, max_nodes_in_reaction] = False  # 标记真实超边节点

        # 4. 构建超图结构 (H_in, H_out)
        #    无论正向还是逆向，'input_features' 在模型视角下都扮演“输入/反应物”的角色 (Tail)
        padded_H_in = torch.zeros(batch_size, max_nodes_in_reaction, 1, dtype=torch.float)
        padded_H_out = torch.zeros(batch_size, max_nodes_in_reaction, 1, dtype=torch.float)
        for i in range(batch_size):
            num_nodes = num_nodes_per_reaction[i]
            if num_nodes > 0:
                padded_H_in[i, :num_nodes, 0] = 1.0  # 所有输入分子都连接到H_in

        # 5. 构建RMatrixHead所需的映射信息
        map_lists = [_molecule_map_list(d) for d in molecule_feature_dicts]
        batch_vec = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(num_nodes_per_reaction)).long()

        # 从标签中（如果存在）或map_lists中，安全地计算max_r_dim
        max_r_dim = 0
        if 'r_matrix_targets' in reaction_batch[0]:
            max_r_dim = max(r['r_matrix_targets'].shape[0] for r in reaction_batch)
        else:
            all_map_nums = [mn for ml in map_lists for mn in ml.keys()]
            if all_map_nums:
                max_r_dim = max(all_map_nums)

        reaction_canvas_mask = torch.zeros(batch_size, max_r_dim, dtype=torch.bool)
        for mol_idx, mol_map_list in enumerate(map_lists):
            reaction_idx = batch_vec[mol_idx].item()
            for map_num in mol_map_list.keys():
                if (map_num - 1) < max_r_dim:
                    reaction_canvas_mask[reaction_idx, map_num - 1] = True

        # padded_be_matrix = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.float)
        #
        # if self.task == 'forwardsynthesis':
        #     # 正向合成：图结构是【反应物】
        #     # Dataset的'input_features'就是反应物
        #     for i, r_sample in enumerate(reaction_batch):
        #         reactant_dicts = r_sample['input_features']
        #         be_react_np = self._combine_be_matrices(reactant_dicts, max_r_dim)
        #         padded_be_matrix[i] = torch.from_numpy(be_react_np)
        #
        # elif self.task == 'retrosynthesis':
        #     # 逆向合成：图结构是【产物】
        #     # Dataset的'input_features'就是产物
        #     for i, r_sample in enumerate(reaction_batch):
        #         product_dicts = r_sample['input_features']
        #         be_prod_np = self._combine_be_matrices(product_dicts, max_r_dim)
        #         padded_be_matrix[i] = torch.from_numpy(be_prod_np)

        # 6. 组装最终的batch字典 (除了标签)
        final_batch = {}
        final_batch.update(molecule_batch)
        final_batch.update({
            "src_key_padding_mask": src_key_padding_mask,
            "padded_H_in": padded_H_in,
            "padded_H_out": padded_H_out,
            "num_nodes_per_sample": torch.tensor(num_nodes_per_reaction),
            "max_nodes_in_batch": max_nodes_in_reaction,
            "map_lists": map_lists,
            "batch_vec": batch_vec,
            "reaction_canvas_mask": reaction_canvas_mask,
            # "padded_be_matrix": padded_be_matrix,
        })

        # 7. 根据数据中存在的键，填充所有【标签】

        # a. R矩阵任务标签 (正向和逆向都可能需要)
        if 'r_matrix_targets' in reaction_batch[0]:
            padded_r_matrix_targets = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.float)
            padded_r_matrix_mask = torch.zeros(batch_size, max_r_dim, max_r_dim, dtype=torch.bool)
            for i, r in enumerate(reaction_batch):
                dim = r['r_matrix_targets'].shape[0]
                padded_r_matrix_targets[i, :dim, :dim] = r['r_matrix_targets']
                padded_r_matrix_mask[i, :dim, :dim] = r['r_matrix_mask']
            final_batch["padded_r_matrix_targets"] = padded_r_matrix_targets
            final_batch["padded_r_matrix_mask"] = padded_r_matrix_mask

        # b. 原子身份任务标签 (仅逆向需要)
        if self.task == 'retrosynthesis' and 'atom_identity_targets' in reaction_batch[0]:
            atom_identity_targets_list = [r['atom_identity_targets'] for r in reaction_batch]
            padded_atom_identity_targets = pad_sequence(
                atom_identity_targets_list, batch_first=True, padding_value=-100
            )
            final_batch['padded_atom_identity_targets'] = padded_atom_identity_targets

        # d. Full USPTO50K-aligned auxiliary targets.
        if self.task == 'retrosynthesis' and 'atom_state_mask' in reaction_batch[0]:
            bool_sequence_keys = [
                'atom_state_mask',
                'chiral_changed',
                'aromatic_changed',
                'attachment_site_targets',
            ]
            int_sequence_keys = [
                'charge_delta',
                'total_h_delta',
                'atom_num_delta',
                'reactant_charge_targets',
                'reactant_total_h_targets',
                'reactant_chiral_targets',
                'reactant_aromatic_targets',
            ]
            float_sequence_keys = ['attachment_bond_order_targets']
            for key in bool_sequence_keys:
                final_batch[key] = pad_sequence(
                    [r[key].bool() for r in reaction_batch],
                    batch_first=True,
                    padding_value=False,
                )
            for key in int_sequence_keys:
                final_batch[key] = pad_sequence(
                    [r[key].long() for r in reaction_batch],
                    batch_first=True,
                    padding_value=-100,
                )
            for key in float_sequence_keys:
                final_batch[key] = pad_sequence(
                    [r[key].float() for r in reaction_batch],
                    batch_first=True,
                    padding_value=0.0,
                )
            for key in ['requires_completion', 'has_bond_edit', 'has_atom_state_change', 'has_attachment_site']:
                final_batch[key] = torch.tensor([bool(r.get(key, False)) for r in reaction_batch], dtype=torch.bool)
            final_batch['full_task_type'] = [r.get('full_task_type', '') for r in reaction_batch]
            final_batch['shared_reactant_core_smiles'] = [
                r.get('shared_reactant_core_smiles', '') for r in reaction_batch
            ]
            final_batch['extra_reactant_fragment_smiles'] = [
                r.get('extra_reactant_fragment_smiles', '') for r in reaction_batch
            ]
            final_batch['unmapped_reactant_fragment_smiles'] = [
                r.get('unmapped_reactant_fragment_smiles', '') for r in reaction_batch
            ]
            final_batch['reactant_only_mapped_fragment_smiles'] = [
                r.get('reactant_only_mapped_fragment_smiles', '') for r in reaction_batch
            ]
            if 'lgm_fragment_target' in reaction_batch[0]:
                final_batch['lgm_fragment_targets'] = torch.stack(
                    [r['lgm_fragment_target'].long() for r in reaction_batch],
                    dim=0,
                )
                final_batch['lgm_fragment_smiles'] = [
                    r.get('lgm_fragment_smiles', '') for r in reaction_batch
                ]
            if 'lg_atom_features' in reaction_batch[0]:
                max_lg_atoms = max(int(r['lg_atom_features'].shape[0]) for r in reaction_batch)
                lg_feature_dim = int(reaction_batch[0]['lg_atom_features'].shape[-1])
                lg_atom_features = torch.zeros(
                    batch_size,
                    max_lg_atoms,
                    lg_feature_dim,
                    dtype=torch.long,
                )
                lg_bond_matrix = torch.zeros(batch_size, max_lg_atoms, max_lg_atoms, dtype=torch.float)
                lg_atom_mask = torch.zeros(batch_size, max_lg_atoms, dtype=torch.bool)
                lgc_pair_targets = torch.zeros(batch_size, max_r_dim, max_lg_atoms, dtype=torch.bool)
                lgc_pair_mask = torch.zeros(batch_size, max_r_dim, max_lg_atoms, dtype=torch.bool)
                lgc_bond_order_targets = torch.full(
                    (batch_size, max_r_dim, max_lg_atoms),
                    -100,
                    dtype=torch.long,
                )
                for i, reaction in enumerate(reaction_batch):
                    lg_atoms = int(reaction['lg_atom_features'].shape[0])
                    product_dim = int(reaction['lgc_pair_targets'].shape[0])
                    lg_atom_features[i, :lg_atoms] = reaction['lg_atom_features'].long()
                    lg_bond_matrix[i, :lg_atoms, :lg_atoms] = reaction['lg_bond_matrix'].float()
                    lg_atom_mask[i, :lg_atoms] = reaction['lg_atom_mask'].bool()
                    lgc_pair_targets[i, :product_dim, :lg_atoms] = reaction['lgc_pair_targets'].bool()
                    lgc_pair_mask[i, :product_dim, :lg_atoms] = reaction['lgc_pair_mask'].bool()
                    lgc_bond_order_targets[i, :product_dim, :lg_atoms] = reaction[
                        'lgc_bond_order_targets'
                    ].long()
                final_batch['lg_atom_features'] = lg_atom_features
                final_batch['lg_bond_matrix'] = lg_bond_matrix
                final_batch['lg_atom_mask'] = lg_atom_mask
                final_batch['lgc_pair_targets'] = lgc_pair_targets
                final_batch['lgc_pair_mask'] = lgc_pair_mask
                final_batch['lgc_bond_order_targets'] = lgc_bond_order_targets
                final_batch['lgc_target_matched'] = torch.tensor(
                    [bool(r.get('lgc_target_matched', False)) for r in reaction_batch],
                    dtype=torch.bool,
                )
                final_batch['num_lgc_connections'] = torch.tensor(
                    [int(r.get('num_lgc_connections', 0)) for r in reaction_batch],
                    dtype=torch.long,
                )

        # e. 传递用于评估的原始SMILES
        if 'raw_reactant_smiles' in reaction_batch[0]:
            final_batch['raw_reactant_smiles'] = [r['raw_reactant_smiles'] for r in reaction_batch]
        if 'raw_product_smiles' in reaction_batch[0]:
            final_batch['raw_product_smiles'] = [r['raw_product_smiles'] for r in reaction_batch]
        if 'mapped_reactant_smiles' in reaction_batch[0]:
            final_batch['mapped_reactant_smiles'] = [r['mapped_reactant_smiles'] for r in reaction_batch]
        if 'mapped_product_smiles' in reaction_batch[0]:
            final_batch['mapped_product_smiles'] = [r['mapped_product_smiles'] for r in reaction_batch]
        if 'row_id' in reaction_batch[0]:
            final_batch['row_id'] = [r['row_id'] for r in reaction_batch]
        if 'matrix_index' in reaction_batch[0]:
            final_batch['matrix_index'] = [r['matrix_index'] for r in reaction_batch]

        return final_batch


class ReactionCollator:
    """Task-aware collator facade used by the new training entrypoints.

    ``pretrain`` keeps the full reactant+product HyperGT view, while ``retro``
    uses only the product/input side and produces the padded labels expected by
    ``RetroLoss``.
    """

    def __init__(self, mode: str, task: str = "pretrain"):
        if task == "retro":
            task = "retrosynthesis"
        self.mode = mode
        self.task = task
        if mode == "pretrain":
            self._collator = HypergraphCollatorForGT(mode=mode)
        elif task == "retrosynthesis":
            self._collator = SynthesisCollatorForGT(task="retrosynthesis", mode=mode)
        else:
            raise ValueError(f"Unsupported collator mode/task: mode={mode}, task={task}")

    def __call__(self, batch: list[dict]) -> dict:
        return self._collator(batch)
