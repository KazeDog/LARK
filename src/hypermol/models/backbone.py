import torch
import torch.nn as nn

from hypermol.data.preprocess import CompoundKit
from hypermol.models.molecular_encoder import MolecularEncoder
from hypermol.models.hypergt import DirectedHyperGT


_AUX_EDGE_FUSION_MODES = {"mean", "attention", "role_attention"}


class FusionBackbone(nn.Module):
    """
    一个融合了MolecularEncoder和DirectedHyperGT的端到端模型。
    它负责将由Collator准备的、分离的原子级和反应级数据，
    动态地组装成一个Transformer可以处理的序列，并执行完整的预测流程。
    """

    def __init__(self,
                 mode: str = 'pretrain',
                 input_num = 2,
                 feature = 'hyperedge',
                 # 分子编码器参数
                 mol_embed_dim: int = 256,
                 mol_num_kernel = 256,
                 mol_num_heads: int = 16,
                 mol_num_layers: int = 6,
                 mol_hidden_size: int = 256,
                 # 超图网络参数
                 hg_embed_dim: int = 256,
                 hg_num_heads: int = 16,
                 hg_layers: int = 6,
                 # 通用参数
                 dropout: float = 0.1,
                 num_tasks=1,
                 condition_enabled: bool = False,
                 condition_dim: int = 514,
                 condition_dropout_prob: float = 0.0,
                 context_role_enabled: bool = False,
                 context_role_aware: bool = False,
                 aux_edge_features_enabled: bool = False,
                 aux_edge_fusion: str = "mean",
                 aux_num_heads: int = 8,
                 role_aware_aux: bool = False,
                 aux_role_vocab_size: int = 6,
                 **mol_encoder_kwargs):
        """
        初始化 FusionHyperGT 模型。

        Args:
            mode (str): 'finetune' 或 'pretrain'。
            feature (str): 微调任务的类型，'node' 或 'hyperedge'。
            num_tasks (int): 微调任务的输出维度。
            vocab_size (int): 原子词汇表大小，用于分子编码器。
            mol_embed_dim (int): 分子编码器的嵌入维度。
            gt_embed_dim (int): HyperGT模块的嵌入维度。
            gt_num_heads (int): HyperGT中Transformer的注意力头数。
            gt_layers (int): HyperGT中Transformer的层数。
            dropout (float): Dropout概率。
            **mol_encoder_kwargs: 其他所有需要传递给MolecularEncoder的参数。
        """
        super().__init__()
        self.mode = mode
        self.input_num = input_num

        self.mol_embed_dim = mol_embed_dim
        self.mol_num_kernel = mol_num_kernel
        self.mol_num_heads = mol_num_heads
        self.mol_num_layers = mol_num_layers
        self.mol_hidden_size = mol_hidden_size

        self.hg_embed_dim = hg_embed_dim
        self.hg_layers = hg_layers

        self.num_tasks = num_tasks
        self.feature = feature
        self.dropout = dropout
        self.condition_enabled = bool(condition_enabled)
        self.condition_dim = int(condition_dim)
        self.condition_dropout_prob = float(condition_dropout_prob)
        self.context_role_enabled = bool(context_role_enabled)
        self.context_role_aware = bool(context_role_aware)
        self.aux_edge_features_enabled = bool(aux_edge_features_enabled)
        self.aux_edge_fusion = str(aux_edge_fusion).lower()
        self.role_aware_aux = bool(role_aware_aux)
        self.aux_role_vocab_size = int(aux_role_vocab_size)
        if self.role_aware_aux and self.aux_role_vocab_size <= 0:
            raise ValueError("aux_role_vocab_size must be > 0 when role_aware_aux is enabled.")
        if self.context_role_aware and self.aux_role_vocab_size <= 0:
            raise ValueError("aux_role_vocab_size must be > 0 when context_role_aware is enabled.")
        if self.aux_edge_fusion not in _AUX_EDGE_FUSION_MODES:
            raise ValueError(
                f"Unsupported aux_edge_fusion: {self.aux_edge_fusion}. "
                f"Choose from {sorted(_AUX_EDGE_FUSION_MODES)}."
            )

        # --- 1. 初始化子模块 ---

        # a. 分子编码器：负责将原子级数据转换为分子级嵌入
        self.molecular_encoder = MolecularEncoder(
            mode=self.mode, atom_names=CompoundKit.atom_vocab_dict.keys(),
            bond_names=CompoundKit.bond_vocab_dict.keys(),
            embed_dim=mol_embed_dim,
            num_kernel=mol_num_kernel,
            layer_num=mol_num_layers,
            num_heads=mol_num_heads,
            hidden_size=mol_hidden_size,
            cross_layers=100,
            dropout=dropout
        )

        # b. 超图Transformer核心：负责在节点和超边序列上进行全局注意力计算
        self.hypergt_net_base = DirectedHyperGT(
            embed_dim=hg_embed_dim,
            num_heads=hg_num_heads,
            num_layers=hg_layers,
            dropout=dropout,
            context_role_enabled=self.context_role_enabled,
        )

        if self.condition_enabled:
            self.condition_encoder = nn.Sequential(
                nn.LayerNorm(self.condition_dim),
                nn.Linear(self.condition_dim, self.hg_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hg_embed_dim, self.hg_embed_dim),
            )
        else:
            self.condition_encoder = None

        # c. 投影层：用于匹配分子编码器和HyperGT之间的维度
        self.projection = nn.Linear(self.mol_embed_dim, self.hg_embed_dim) if self.mol_embed_dim != self.hg_embed_dim else nn.Identity()
        self.context_projection = nn.Linear(self.hg_embed_dim, self.mol_embed_dim) if self.mol_embed_dim != self.hg_embed_dim else nn.Identity()
        self.context_type_role_embedding = (
            nn.Embedding(
                self.aux_role_vocab_size,
                self.hg_embed_dim,
                padding_idx=0,
            )
            if self.context_role_aware
            else None
        )
        self.edge_aux_role_embedding = (
            nn.Embedding(self.aux_role_vocab_size, self.mol_embed_dim) if self.role_aware_aux else None
        )
        self.edge_role_proj = None
        self.edge_role_embedding = None
        self.edge_role_attention = None
        self.edge_role_interaction = None
        self.edge_role_gate = None

        if self.aux_edge_features_enabled:
            if self.aux_edge_fusion == "attention":
                aux_num_heads = int(aux_num_heads)
                if aux_num_heads <= 0:
                    raise ValueError("aux_num_heads must be > 0.")
                if self.hg_embed_dim % aux_num_heads != 0:
                    raise ValueError(
                        f"hg_embed_dim={self.hg_embed_dim} must be divisible by aux_num_heads={aux_num_heads}."
                    )
                self.edge_aux_proj = None
                self.edge_aux_attention = nn.MultiheadAttention(
                    embed_dim=self.hg_embed_dim,
                    num_heads=aux_num_heads,
                    dropout=dropout,
                    batch_first=True,
                    kdim=self.mol_embed_dim,
                    vdim=self.mol_embed_dim,
                )
                self.edge_aux_norm = nn.LayerNorm(self.hg_embed_dim)
            elif self.aux_edge_fusion == "role_attention":
                aux_num_heads = int(aux_num_heads)
                if aux_num_heads <= 0:
                    raise ValueError("aux_num_heads must be > 0.")
                if self.hg_embed_dim % aux_num_heads != 0:
                    raise ValueError(
                        f"hg_embed_dim={self.hg_embed_dim} must be divisible by aux_num_heads={aux_num_heads}."
                    )
                if self.aux_role_vocab_size <= 0:
                    raise ValueError("aux_role_vocab_size must be > 0 when aux_edge_fusion='role_attention'.")
                self.edge_aux_proj = None
                self.edge_aux_attention = None
                self.edge_aux_norm = None
                self.edge_role_proj = nn.Sequential(
                    nn.Linear(self.mol_embed_dim, self.hg_embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hg_embed_dim),
                )
                self.edge_role_embedding = nn.Embedding(self.aux_role_vocab_size, self.hg_embed_dim)
                self.edge_role_attention = nn.MultiheadAttention(
                    embed_dim=self.hg_embed_dim,
                    num_heads=aux_num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.edge_role_interaction = nn.Sequential(
                    nn.Linear(self.hg_embed_dim * 3, self.hg_embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hg_embed_dim, self.hg_embed_dim),
                    nn.LayerNorm(self.hg_embed_dim),
                )
                self.edge_role_gate = nn.Sequential(
                    nn.Linear(self.hg_embed_dim * 2, self.hg_embed_dim),
                    nn.Sigmoid(),
                )
            else:
                self.edge_aux_proj = nn.Sequential(
                    nn.Linear(self.mol_embed_dim, self.hg_embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hg_embed_dim),
                )
                self.edge_aux_attention = None
                self.edge_aux_norm = None
        else:
            self.edge_aux_proj = None
            self.edge_aux_attention = None
            self.edge_aux_norm = None

        self.gate = nn.Sequential(
            nn.Linear(mol_embed_dim * 2, mol_embed_dim),
            nn.Sigmoid()
        )

    def _encode_aux_molecules(self, batch: dict, batch_size: int, device: torch.device):
        aux_batch = batch.get("edge_aux_batch") or {}
        aux_owner = batch.get("edge_aux_owner")
        if (
            not self.aux_edge_features_enabled
            or not aux_batch
            or aux_owner is None
            or aux_owner.numel() == 0
        ):
            return (
                torch.zeros(0, self.mol_embed_dim, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        aux_outputs = self.molecular_encoder(aux_batch)
        aux_features = aux_outputs["mol_features"]
        aux_owner = aux_owner.to(device=aux_features.device, dtype=torch.long)
        valid_owner = (aux_owner >= 0) & (aux_owner < batch_size)
        aux_features = aux_features[valid_owner]
        aux_owner = aux_owner[valid_owner]
        role_ids = batch.get("edge_aux_role_ids")
        if role_ids is None or role_ids.numel() != valid_owner.numel():
            role_ids = torch.zeros(valid_owner.numel(), dtype=torch.long, device=aux_features.device)
        else:
            role_ids = role_ids.to(device=aux_features.device, dtype=torch.long)
        role_ids = role_ids[valid_owner].clamp(min=0, max=self.aux_role_vocab_size - 1)
        if self.edge_aux_role_embedding is not None:
            aux_features = aux_features + self.edge_aux_role_embedding(role_ids)
        return aux_features, aux_owner, role_ids

    def _mean_pool_aux_edge_features(self, aux_features: torch.Tensor, aux_owner: torch.Tensor, batch_size: int) -> torch.Tensor:
        pooled = aux_features.new_zeros(batch_size, self.mol_embed_dim)
        if aux_owner.numel() > 0:
            pooled.index_add_(0, aux_owner, aux_features)
            counts = torch.bincount(aux_owner, minlength=batch_size).clamp_min(1)
            pooled = pooled / counts.to(dtype=pooled.dtype).unsqueeze(-1)
        return self.edge_aux_proj(pooled)

    def _attention_pool_aux_edge_features(
        self,
        aux_features: torch.Tensor,
        aux_owner: torch.Tensor,
        batch_size: int,
        edge_query: torch.Tensor,
    ) -> torch.Tensor:
        if aux_features.numel() == 0:
            return edge_query.new_zeros(batch_size, self.hg_embed_dim)
        counts = torch.bincount(aux_owner, minlength=batch_size)
        max_aux = max(int(counts.max().item()) if counts.numel() > 0 else 0, 1)
        aux_seq = aux_features.new_zeros(batch_size, max_aux, self.mol_embed_dim)
        key_padding_mask = torch.ones(batch_size, max_aux, dtype=torch.bool, device=aux_features.device)
        cursor = torch.zeros(batch_size, dtype=torch.long, device=aux_features.device)
        for feature, owner in zip(aux_features, aux_owner):
            owner_idx = int(owner.item())
            position = int(cursor[owner_idx].item())
            aux_seq[owner_idx, position] = feature
            key_padding_mask[owner_idx, position] = False
            cursor[owner_idx] += 1
        has_aux = counts > 0
        if torch.any(~has_aux):
            key_padding_mask[~has_aux, 0] = False
        aux_context, _ = self.edge_aux_attention(
            query=edge_query.unsqueeze(1),
            key=aux_seq,
            value=aux_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        aux_context = self.edge_aux_norm(aux_context.squeeze(1))
        return aux_context * has_aux.to(dtype=aux_context.dtype).unsqueeze(-1)

    def _role_attention_pool_aux_edge_features(
        self,
        aux_features: torch.Tensor,
        aux_owner: torch.Tensor,
        aux_role_ids: torch.Tensor,
        batch_size: int,
        edge_query: torch.Tensor,
    ) -> torch.Tensor:
        if aux_features.numel() == 0:
            return edge_query.new_zeros(batch_size, self.hg_embed_dim)

        num_roles = self.aux_role_vocab_size
        flat_role_index = aux_owner * num_roles + aux_role_ids
        flat_pooled = aux_features.new_zeros(batch_size * num_roles, self.mol_embed_dim)
        flat_pooled.index_add_(0, flat_role_index, aux_features)
        flat_counts = torch.bincount(flat_role_index, minlength=batch_size * num_roles).to(device=aux_features.device)
        flat_pooled = flat_pooled / flat_counts.clamp_min(1).to(dtype=flat_pooled.dtype).unsqueeze(-1)

        role_pooled = flat_pooled.view(batch_size, num_roles, self.mol_embed_dim)
        has_role = flat_counts.view(batch_size, num_roles) > 0
        has_aux = has_role.any(dim=1)

        role_context = self.edge_role_proj(role_pooled)
        role_ids = torch.arange(num_roles, device=aux_features.device, dtype=torch.long)
        role_emb = self.edge_role_embedding(role_ids).unsqueeze(0)
        role_context = role_context + role_emb * has_role.to(dtype=role_context.dtype).unsqueeze(-1)

        key_padding_mask = ~has_role
        if torch.any(~has_aux):
            key_padding_mask[~has_aux, 0] = False
            role_context[~has_aux, 0] = 0.0

        aux_context, _ = self.edge_role_attention(
            query=edge_query.unsqueeze(1),
            key=role_context,
            value=role_context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        aux_context = aux_context.squeeze(1)
        interaction_input = torch.cat([edge_query, aux_context, edge_query * aux_context], dim=-1)
        role_delta = self.edge_role_interaction(interaction_input)
        gate = self.edge_role_gate(torch.cat([edge_query, aux_context], dim=-1))
        return gate * role_delta * has_aux.to(dtype=role_delta.dtype).unsqueeze(-1)

    def _build_aux_edge_features(self, batch: dict, batch_size: int, edge_query: torch.Tensor) -> torch.Tensor:
        aux_features, aux_owner, aux_role_ids = self._encode_aux_molecules(batch, batch_size, edge_query.device)
        if self.aux_edge_fusion == "role_attention":
            return self._role_attention_pool_aux_edge_features(
                aux_features,
                aux_owner,
                aux_role_ids,
                batch_size,
                edge_query,
            )
        if self.aux_edge_fusion == "attention":
            return self._attention_pool_aux_edge_features(aux_features, aux_owner, batch_size, edge_query)
        return self._mean_pool_aux_edge_features(aux_features, aux_owner, batch_size)


    # @profile
    def forward(self, batch: dict):
        """
        定义了从原始数据到最终预测的完整前向传播路径。
        这个方法能够根据 self.mode 和 self.feature 的设置，执行不同的计算图。

        Args:
            batch (dict): 从 HypergraphCollatorForGT 接收到的、包含所有预处理好张量的字典。

        Returns:
            dict: 一个包含预测结果的字典。
                  键名可能包括 'logits' (用于微调) 或 'r_matrix_preds', 'mam_logits' (用于预训练)。
        """
        # --- 步骤一: 分子编码 (在所有模式下共享) ---
        # a. 调用 MolecularEncoder，从原子级数据计算出分子级的嵌入
        #    'molecule_inputs' 是由 HypergraphCollatorForGT 准备的，包含了 'padded_atom_features' 等
        mol_encoder_output = self.molecular_encoder(batch)

        # 提取原子级表征（用于预训练）和分子级表征（用于超图）
        dense_atom_embeddings = mol_encoder_output['atom_features']  # 形状: [total_mols, max_atoms_in_mol, D_mol]
        initial_node_embeddings = mol_encoder_output['mol_features']  # 形状: [total_mols, D_mol]

        if "padded_H_in" not in batch:
            return {
                "dense_atom_embeddings": dense_atom_embeddings,
                "final_node_embeddings": initial_node_embeddings
            }

        # b. 应用投影层，确保分子嵌入的维度与HyperGT匹配
        projected_embeddings = self.projection(initial_node_embeddings)  # 形状: [total_mols, D_gt]
        if self.context_type_role_embedding is not None:
            context_role_ids = batch.get("context_role_ids")
            context_mask = batch.get("context_molecule_mask")
            if context_role_ids is None or context_mask is None:
                raise ValueError(
                    "context_role_aware requires context_role_ids and context_molecule_mask."
                )
            if (
                context_role_ids.numel() != projected_embeddings.shape[0]
                or context_mask.numel() != projected_embeddings.shape[0]
            ):
                raise ValueError(
                    "Context role tensors must align with flattened molecule embeddings."
                )
            context_role_ids = context_role_ids.to(
                device=projected_embeddings.device,
                dtype=torch.long,
            ).clamp(min=0, max=self.aux_role_vocab_size - 1)
            context_mask = context_mask.to(
                device=projected_embeddings.device,
                dtype=projected_embeddings.dtype,
            ).unsqueeze(-1)
            projected_embeddings = (
                projected_embeddings
                + self.context_type_role_embedding(context_role_ids) * context_mask
            )

        # --- 步骤二: 在forward中动态构建Transformer的输入序列 'src' (在所有模式下共享) ---
        batch_size = batch['src_key_padding_mask'].shape[0]
        max_nodes_in_reaction = batch['max_nodes_in_batch']
        embed_dim = projected_embeddings.shape[1]
        device = projected_embeddings.device

        # a. 创建空的序列画布
        sequence_length = max_nodes_in_reaction + 1  # +1 是为超边节点留出位置
        src_initial = torch.zeros(batch_size, sequence_length, embed_dim, device=device)

        # b. 将计算出的分子嵌入“scatter”(散布)到序列画布的正确位置上
        mol_offset = 0
        for i in range(batch_size):
            num_nodes = batch['num_nodes_per_sample'][i].item()  # .item() 转换为整数
            if num_nodes > 0:
                src_initial[i, :num_nodes] = projected_embeddings[mol_offset: mol_offset + num_nodes]
            mol_offset += num_nodes

        # c. 计算并填充超边（反应）的初始特征
        for i in range(batch_size):
            num_nodes = batch['num_nodes_per_sample'][i].item()
            if num_nodes > 0:
                src_initial[i, max_nodes_in_reaction] = src_initial[i, :num_nodes].mean(dim=0)

        edge_state = src_initial[:, max_nodes_in_reaction].clone()

        if self.condition_enabled and self.condition_encoder is not None and "condition_vector" in batch:
            condition_vector = batch["condition_vector"].to(device=device, dtype=src_initial.dtype)
            condition_emb = self.condition_encoder(condition_vector)
            has_condition = batch.get("has_condition_vector")
            if has_condition is not None:
                condition_emb = condition_emb * has_condition.to(device=device).float().unsqueeze(-1)
            if self.training and self.condition_dropout_prob > 0.0:
                keep_mask = torch.rand(batch_size, 1, device=device) > self.condition_dropout_prob
                condition_emb = condition_emb * keep_mask.to(condition_emb.dtype)
            edge_state = edge_state + condition_emb

        if self.aux_edge_features_enabled:
            aux_edge_features = self._build_aux_edge_features(
                batch,
                batch_size=batch_size,
                edge_query=edge_state,
            )
            edge_state = edge_state + aux_edge_features

        src_initial = src_initial.clone()
        src_initial[:, max_nodes_in_reaction] = edge_state

        # --- 步骤三: 调用HyperGT核心模块进行全局信息交换 (在所有模式下共享) ---
        transformer_output = self.hypergt_net_base(
            src_initial=src_initial,
            src_key_padding_mask=batch['src_key_padding_mask'],
            padded_H_in=batch['padded_H_in'],
            padded_H_out=batch['padded_H_out'],
            padded_H_context=batch.get('padded_H_context'),
        )
        # transformer_output 形状: [B, L, D]

        # 提取更新后的【分子级】表征
        max_nodes_in_reaction = batch['max_nodes_in_batch']
        final_node_embeddings_padded = transformer_output[:, :max_nodes_in_reaction, :]
        final_edge_features = transformer_output[:, max_nodes_in_reaction, :]

        # --- 步骤三：统一的信息融合 (共享) ---
        # a. "Unpad" 更新后的分子表征，使其与原子表征的 batch size (total_mols) 对齐
        node_padding_mask = ~batch['src_key_padding_mask'][:, :max_nodes_in_reaction]
        final_node_embeddings_flat = final_node_embeddings_padded[node_padding_mask]  # 形状: [total_mols, D]

        # Consistency is defined on the mapped one-sided view.  Context nodes
        # can influence it through attention, but cannot become a trivial
        # shared token that is pooled directly into both directions.
        padded_context = batch.get("padded_H_context")
        if padded_context is None:
            core_node_mask = node_padding_mask
        else:
            core_node_mask = node_padding_mask & ~(padded_context[..., 0] > 0)
        core_weights = core_node_mask.to(final_node_embeddings_padded.dtype).unsqueeze(-1)
        core_view_features = (
            final_node_embeddings_padded * core_weights
        ).sum(dim=1) / core_weights.sum(dim=1).clamp_min(1.0)

        # b. 扩展分子级上下文，准备与原子级表征融合
        atom_dim_context = self.context_projection(final_node_embeddings_flat)
        context_expanded = atom_dim_context.unsqueeze(1)
        target_num_nodes = dense_atom_embeddings.shape[1]  # max_atoms_in_mol
        context_per_atom = context_expanded.expand(-1, target_num_nodes, -1)

        # c. 执行门控融合，得到最终的、富含双重上下文的原子表征
        combined_info = torch.cat([dense_atom_embeddings, context_per_atom], dim=-1)
        g = self.gate(combined_info)
        context_aware_atom_embeddings = (1 - g) * dense_atom_embeddings + g * context_per_atom

        return {
            "initial_atom_embeddings": dense_atom_embeddings,
            "initial_node_embeddings": initial_node_embeddings,
            # Projected molecule identities before DirectedHyperGT mixing.
            # Downstream-only role heads can combine this frozen, molecule-
            # local representation with the contextualized node states
            # without changing the pretrained backbone contract or weights.
            "initial_node_embeddings_padded": src_initial[:, :max_nodes_in_reaction, :],
            "final_node_embeddings": final_node_embeddings_flat,
            "final_node_embeddings_padded": final_node_embeddings_padded,
            "final_edge_features": final_edge_features,
            "node_padding_mask": node_padding_mask,
            "padded_context_role_ids": batch.get("padded_context_role_ids"),
            "core_view_features": core_view_features,
            "context_aware_atom_embeddings": context_aware_atom_embeddings
        }


# Backward-compatible alias for checkpoints or exploratory notebooks.
FusionHyperGTBackbone = FusionBackbone
