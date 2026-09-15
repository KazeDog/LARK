import torch
import torch.nn as nn

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class MolProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output, dropout=0.1):
        # print d_atom, d_hid, d_output
        super(MolProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            # nn.BatchNorm1d(d_hid),
            # nn.SiLU(),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, mask=None):
        x = self.linear_seq(z)
        return x


class AngleProjection(nn.Module):
    def __init__(self, d_atom, d_hid, d_output, dropout=0.1):
        # print d_atom, d_hid, d_output
        # print_debug('AtomProjection: d_atom={}, d_hid={}, d_output={}'.format(d_atom, d_hid, d_output))
        super(AngleProjection, self).__init__()

        self.linear_seq = nn.Sequential(
            nn.Linear(d_atom, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hid, d_output)
        )

    def forward(self, z, angel_atom_table, mask=None):
        valid_entries = angel_atom_table[:, :, 0] != -1
        indices = torch.nonzero(valid_entries)
        indices_i, indices_j = indices[:, 0], indices[:, 1]

        x = z[indices_i, angel_atom_table[indices_i, indices_j, 0]] + z[
            indices_i, angel_atom_table[indices_i, indices_j, 1]] + z[
                indices_i, angel_atom_table[indices_i, indices_j, 2]]

        x = self.linear_seq(x)
        return x

class TorsionHead(nn.Module):
    """
    一个专门用于预测“边扭转角”的预测头。
    此版本接收已经填充好的、批量的原子特征和边索引，
    以及一个由collator预先计算好的、指示真实边位置的布尔掩码。
    """

    def __init__(self, d_atom, d_hid, num_torsion_bins, dropout=0.1):
        super(TorsionHead, self).__init__()
        # MLP接收2个拼接的原子嵌入
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_atom, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hid, num_torsion_bins)
        )

    def forward(self, atom_features: torch.Tensor, edge_indices: torch.Tensor, torsion_valid_mask: torch.Tensor):
        """
        前向传播。
        Args:
            atom_features (Tensor): 批量的、填充好的原子表征。
                                    形状: [batch_size, max_len, d_atom]
            edge_indices (Tensor): 填充好的、批量的边原子索引。
                                   形状: [batch_size, max_edges, 2],
                                   填充值为 -5。
            torsion_valid_mask (Tensor): 由collator提供的、指示真实边位置的布尔掩码。
                                         形状: [batch_size, max_edges]
        Returns:
            Tensor: 每个真实扭转角的logits。形状: [num_total_edges, num_torsion_bins]。
        """
        # 1. 直接使用传入的掩码找到所有真实（非填充）的边
        #    这确保了与 PretrainLoss 中的逻辑完全一致。
        #    batch_indices, edge_indices_in_sample 的形状都是 [num_total_edges]
        batch_indices, edge_indices_in_sample = torch.nonzero(torsion_valid_mask, as_tuple=True)

        # 2. 提取真实边的原子索引
        # a. 首先，根据真实边的坐标，从`edge_indices`中提取出 [atom_i, atom_j] 对
        #    real_edges 的形状: [num_total_edges, 2]
        real_edges = edge_indices[batch_indices, edge_indices_in_sample]
        # b. 分离出 atom_i 和 atom_j 的索引
        atom_i_local = real_edges[:, 0]
        atom_j_local = real_edges[:, 1]

        # 3. 提取边的两个端点的原子表征
        #    我们使用 batch_indices 和局部原子索引来共同定位
        feat_i = atom_features[batch_indices, atom_i_local]
        feat_j = atom_features[batch_indices, atom_j_local]

        # 4. 将两个原子的嵌入拼接在一起，并使其对方向不敏感
        #    形状: [num_total_edges, 2 * d_atom]
        combined_features_ij = torch.cat([feat_i, feat_j], dim=-1)
        combined_features_ji = torch.cat([feat_j, feat_i], dim=-1)
        combined_features = combined_features_ij + combined_features_ji

        # 5. 通过MLP进行预测
        #    形状: [num_total_edges, num_torsion_bins]
        logits = self.mlp(combined_features)

        return logits

class RMatrixHead(nn.Module):
    """
    一个专门用于预测R矩阵的预测头。

    它的核心工作流程是：
    1. 接收以【分子】为单位批处理的、填充过的原子表征。
    2. 利用原子图谱映射信息，将这些原子表征“绘制”到一个以【反应】为单位的、
       统一的“反应画布”上。
    3. 在这个反应画布上，通过计算所有原子对的交互来预测R矩阵。
    """

    def __init__(
        self,
        embed_dim,
        dropout=0.1,
        use_be_pair_feature: bool = False,
        be_rbf_bins: int = 81,
        be_rbf_min: float = 0.0,
        be_rbf_max: float = 8.0,
    ):
        """
        初始化RMatrixHead。
        Args:
            embed_dim (int): 输入的原子/节点表征的维度。
        """
        super().__init__()
        self.use_be_pair_feature = use_be_pair_feature
        self.be_rbf_bins = int(be_rbf_bins)
        if self.use_be_pair_feature:
            centers = torch.linspace(float(be_rbf_min), float(be_rbf_max), self.be_rbf_bins)
            width = float(centers[1] - centers[0]) if self.be_rbf_bins > 1 else 1.0
            self.register_buffer("be_rbf_centers", centers)
            self.be_rbf_gamma = 1.0 / max(2.0 * width * width, 1e-8)
        else:
            self.register_buffer("be_rbf_centers", torch.empty(0))
            self.be_rbf_gamma = 1.0

        pair_input_dim = embed_dim * 2 + (self.be_rbf_bins if self.use_be_pair_feature else 0)
        # 这个MLP用于处理一对原子表征的拼接，并预测它们之间的R矩阵值（一个标量）
        self.mlp = nn.Sequential(
            nn.Linear(pair_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 1)
        )

    def _be_rbf(self, padded_be_matrix: torch.Tensor) -> torch.Tensor:
        values = padded_be_matrix.unsqueeze(-1)
        centers = self.be_rbf_centers.to(device=values.device, dtype=values.dtype)
        return torch.exp(-self.be_rbf_gamma * (values - centers) ** 2)

    def forward(self,
                dense_atom_embeddings: torch.Tensor,
                atom_mask,
                map_lists: list[dict],
                batch_vec: torch.Tensor,
                reaction_canvas_mask: torch.Tensor,
                padded_be_matrix: torch.Tensor | None = None,
                *,
                pair_prior: torch.Tensor | None = None,
                observed_mask: torch.Tensor | None = None,
                canvas_index_lists: list | None = None):
        """
        前向传播。
        Args:
            dense_atom_embeddings (Tensor): 来自MolecularEncoder的、填充好的原子表征。
                                            形状: [num_total_molecules, max_nodes, embed_dim]
            map_lists (list[dict]): 长度为 num_total_molecules 的列表,
                                    每个元素是 {map_num: local_atom_idx} 的映射。
            batch_vec (Tensor): 标记每个分子属于哪个反应的索引。
                                形状: [num_total_molecules]
            reaction_canvas_mask (Tensor): 标记在反应画布上哪些原子图谱号是真实存在的。
                                            形状: [batch_size, max_r_dim]
        Returns:
            Tensor: 预测出的R矩阵。形状: [batch_size, max_r_dim, max_r_dim]
        """
        device = dense_atom_embeddings.device
        batch_size = reaction_canvas_mask.shape[0]
        max_r_dim = reaction_canvas_mask.shape[1]
        embed_dim = dense_atom_embeddings.shape[-1]

        # 1. 创建一个空的、统一的“反应画布”
        # 这个画布的坐标系是 (反应索引, 全局原子图谱号索引)
        # 形状: [batch_size, max_r_dim, embed_dim]
        reaction_canvas = torch.zeros(batch_size, max_r_dim, embed_dim, device=device)

        # 2. 将原子表征“绘制”或“散布”(scatter)到画布上。
        # ``canvas_index_lists`` is the strict, map-number-independent interface:
        # each entry is a sequence whose local-atom position stores a zero-based
        # reaction-canvas index (negative values mean "not on canvas").  The
        # legacy ``map_lists`` interface remains supported for old collators and
        # checkpoints.
        index_specs = canvas_index_lists if canvas_index_lists is not None else map_lists
        if index_specs is None:
            index_specs = []
        for mol_idx, index_spec in enumerate(index_specs):
            # 找到这个分子属于哪个反应
            reaction_idx = batch_vec[mol_idx].item()

            if canvas_index_lists is not None:
                if isinstance(index_spec, dict):
                    # The new dictionary form is explicitly local -> canvas.
                    local_canvas_pairs = index_spec.items()
                else:
                    if torch.is_tensor(index_spec):
                        index_spec = index_spec.detach().cpu().tolist()
                    local_canvas_pairs = enumerate(index_spec)
            else:
                # Legacy dictionaries are map-number -> local-atom index.
                local_canvas_pairs = (
                    (local_atom_idx, map_num - 1)
                    for map_num, local_atom_idx in index_spec.items()
                )

            for local_atom_idx, canvas_idx in local_canvas_pairs:
                local_atom_idx = int(local_atom_idx)
                canvas_idx = int(canvas_idx)
                if 0 <= canvas_idx < max_r_dim:
                    if local_atom_idx < atom_mask.shape[1] and atom_mask[mol_idx, local_atom_idx].item():
                        # 从分子坐标系获取原子表征
                        # 我们假设map_list中的local_atom_idx总是指向真实原子（非填充）
                        atom_embedding = dense_atom_embeddings[mol_idx, local_atom_idx]

                        # 将其放置到反应画布的正确位置上
                        # 使用 += 是一个健壮的做法，以处理多个分子贡献同一图谱号的罕见情况
                        reaction_canvas[reaction_idx, canvas_idx] += atom_embedding

        # 3. 在维度正确的反应画布上进行成对预测
        # a. 扩展维度以进行广播
        # x_i 形状: [batch_size, max_r_dim, 1, embed_dim] -> [batch_size, max_r_dim, max_r_dim, embed_dim]
        x_i = reaction_canvas.unsqueeze(2).expand(-1, -1, max_r_dim, -1)
        # x_j 形状: [batch_size, 1, max_r_dim, embed_dim] -> [batch_size, max_r_dim, max_r_dim, embed_dim]
        x_j = reaction_canvas.unsqueeze(1).expand(-1, max_r_dim, -1, -1)

        # b. 拼接成对特征
        # pair_features 形状: [batch_size, max_r_dim, max_r_dim, embed_dim * 2]
        pair_features = torch.cat([x_i, x_j], dim=-1)
        if self.use_be_pair_feature:
            # ``pair_prior`` is the strict interface.  It may contain only the
            # observed/corrupted pair information.  Falling back to
            # ``padded_be_matrix`` preserves the old forward contract.
            prior = pair_prior if pair_prior is not None else padded_be_matrix
            if prior is None:
                prior = torch.zeros(batch_size, max_r_dim, max_r_dim, device=device)
            else:
                prior = prior.to(device=device, dtype=dense_atom_embeddings.dtype)
                if prior.shape[1] != max_r_dim or prior.shape[2] != max_r_dim:
                    padded = torch.zeros(batch_size, max_r_dim, max_r_dim, device=device, dtype=dense_atom_embeddings.dtype)
                    rows = min(max_r_dim, prior.shape[1])
                    cols = min(max_r_dim, prior.shape[2])
                    padded[:, :rows, :cols] = prior[:, :rows, :cols]
                    prior = padded
            prior_features = self._be_rbf(prior)
            if observed_mask is not None:
                observed_mask = observed_mask.to(device=device, dtype=torch.bool)
                if observed_mask.shape != prior.shape:
                    raise ValueError(
                        f"observed_mask shape {tuple(observed_mask.shape)} does not match "
                        f"pair_prior shape {tuple(prior.shape)}"
                    )
                # All-zero features make an unobserved pair distinct from a
                # genuinely observed numerical zero without changing the MLP
                # input width (and therefore without breaking old checkpoints).
                prior_features = prior_features * observed_mask.unsqueeze(-1).to(prior_features.dtype)
            pair_features = torch.cat([pair_features, prior_features], dim=-1)

        # 4. 通过MLP预测
        # predictions 形状: [batch_size, max_r_dim, max_r_dim, 1]
        predictions = self.mlp(pair_features).squeeze(-1)  # 移除最后一个维度

        # 5. (可选但推荐) 强制对称性
        r_matrix_preds_raw = (predictions + predictions.transpose(1, 2)) / 2

        # 6. 使用 reaction_canvas_mask 清理无效区域
        # a. 创建成对的画布掩码
        # canvas_pair_mask 形状: [batch_size, max_r_dim, max_r_dim]
        canvas_pair_mask = reaction_canvas_mask.unsqueeze(2) * reaction_canvas_mask.unsqueeze(1)
        # b. 将无效位置（即对应原子不存在的行或列）的预测值置零
        r_matrix_preds = r_matrix_preds_raw * canvas_pair_mask

        # --- 核心修改点 ---
        # 最终返回一个元组 (tuple)，包含预测结果和中间画布
        return r_matrix_preds, reaction_canvas


class RMatrixHeadGAT(nn.Module):
    """
    一个专门用于预测R矩阵的预测头。

    它的核心工作流程是：
    1. 接收以【分子】为单位批处理的、填充过的原子表征。
    2. 利用原子图谱映射信息，将这些原子表征“绘制”到一个以【反应】为单位的、
       统一的“反应画布”上。
    3. 在这个反应画布上，通过计算所有原子对的交互来预测R矩阵。
    """

    def __init__(self, embed_dim, num_gnn_layers=2, num_heads=4, dropout=0.1):
        """
        初始化RMatrixHead。
        Args:
            embed_dim (int): 输入的原子/节点表征的维度。
        """
        super().__init__()

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(num_gnn_layers):
            # GATConv 能学习不同邻居的重要性
            self.gnn_layers.append(
                GATConv(embed_dim, embed_dim, heads=num_heads, concat=False, dropout=dropout)
            )
            self.gnn_norms.append(nn.LayerNorm(embed_dim))

        self.gnn_activation = nn.GELU()
        self.gnn_dropout = nn.Dropout(p=dropout)

        # 这个MLP用于处理一对原子表征的拼接，并预测它们之间的R矩阵值（一个标量）
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 1)
        )

    def forward(self,
                dense_atom_embeddings: torch.Tensor,
                atom_mask,
                map_lists: list[dict],
                batch_vec: torch.Tensor,
                reaction_canvas_mask: torch.Tensor,
                padded_be_matrix: torch.Tensor):
        """
        前向传播。
        Args:
            dense_atom_embeddings (Tensor): 来自MolecularEncoder的、填充好的原子表征。
                                            形状: [num_total_molecules, max_nodes, embed_dim]
            map_lists (list[dict]): 长度为 num_total_molecules 的列表,
                                    每个元素是 {map_num: local_atom_idx} 的映射。
            batch_vec (Tensor): 标记每个分子属于哪个反应的索引。
                                形状: [num_total_molecules]
            reaction_canvas_mask (Tensor): 标记在反应画布上哪些原子图谱号是真实存在的。
                                            形状: [batch_size, max_r_dim]
        Returns:
            Tensor: 预测出的R矩阵。形状: [batch_size, max_r_dim, max_r_dim]
        """
        device = dense_atom_embeddings.device
        batch_size = reaction_canvas_mask.shape[0]
        max_r_dim = reaction_canvas_mask.shape[1]
        embed_dim = dense_atom_embeddings.shape[-1]

        # 1. 创建一个空的、统一的“反应画布”
        # 这个画布的坐标系是 (反应索引, 全局原子图谱号索引)
        # 形状: [batch_size, max_r_dim, embed_dim]
        reaction_canvas = torch.zeros(batch_size, max_r_dim, embed_dim, device=device)

        # 2. 将原子表征“绘制”或“散布”(scatter)到画布上
        # 遍历批次中的每一个分子
        for mol_idx, mol_map_list in enumerate(map_lists):
            # 找到这个分子属于哪个反应
            reaction_idx = batch_vec[mol_idx].item()

            # 遍历这个分子的所有映射原子
            for map_num, local_atom_idx in mol_map_list.items():
                # 确保图谱号在当前批次的画布尺寸范围内
                if (map_num - 1) < max_r_dim:
                    if local_atom_idx < atom_mask.shape[1] and atom_mask[mol_idx, local_atom_idx].item():
                        # 从分子坐标系获取原子表征
                        # 我们假设map_list中的local_atom_idx总是指向真实原子（非填充）
                        atom_embedding = dense_atom_embeddings[mol_idx, local_atom_idx]

                        # 将其放置到反应画布的正确位置上
                        # 使用 += 是一个健壮的做法，以处理多个分子贡献同一图谱号的罕见情况
                        reaction_canvas[reaction_idx, map_num - 1] += atom_embedding

        adj = (padded_be_matrix > 0.5).float()
        edge_index, _ = dense_to_sparse(adj)
        x = reaction_canvas
        x_flat = x.view(-1, embed_dim)

        # gnn_batch_vec = torch.arange(batch_size, device=device).repeat_interleave(max_r_dim)

        for i in range(len(self.gnn_layers)):
            identity = x_flat
            x_flat = self.gnn_layers[i](x_flat, edge_index)
            x_flat = self.gnn_dropout(x_flat)
            x_flat = identity + x_flat
            x_flat = self.gnn_activation(x_flat)
            x_flat = self.gnn_norms[i](x_flat)

        # 将更新后的节点表征重塑回批处理格式
        updated_reaction_canvas = x_flat.view(batch_size, max_r_dim, embed_dim)

        # 3. 在维度正确的反应画布上进行成对预测
        # a. 扩展维度以进行广播
        # x_i 形状: [batch_size, max_r_dim, 1, embed_dim] -> [batch_size, max_r_dim, max_r_dim, embed_dim]
        x_i = updated_reaction_canvas.unsqueeze(2).expand(-1, -1, max_r_dim, -1)
        # x_j 形状: [batch_size, 1, max_r_dim, embed_dim] -> [batch_size, max_r_dim, max_r_dim, embed_dim]
        x_j = updated_reaction_canvas.unsqueeze(1).expand(-1, max_r_dim, -1, -1)

        # b. 拼接成对特征
        # pair_features 形状: [batch_size, max_r_dim, max_r_dim, embed_dim * 2]
        pair_features = torch.cat([x_i, x_j], dim=-1)

        # 4. 通过MLP预测
        # predictions 形状: [batch_size, max_r_dim, max_r_dim, 1]
        predictions = self.mlp(pair_features).squeeze(-1)  # 移除最后一个维度

        # 5. (可选但推荐) 强制对称性
        r_matrix_preds_raw = (predictions + predictions.transpose(1, 2)) / 2

        # 6. 使用 reaction_canvas_mask 清理无效区域
        # a. 创建成对的画布掩码
        # canvas_pair_mask 形状: [batch_size, max_r_dim, max_r_dim]
        canvas_pair_mask = reaction_canvas_mask.unsqueeze(2) * reaction_canvas_mask.unsqueeze(1)
        # b. 将无效位置（即对应原子不存在的行或列）的预测值置零
        r_matrix_preds = r_matrix_preds_raw * canvas_pair_mask

        # --- 核心修改点 ---
        # 最终返回一个元组 (tuple)，包含预测结果和中间画布
        return r_matrix_preds, updated_reaction_canvas
