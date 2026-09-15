import torch
import torch.nn as nn

from hypermol.models.layers import Transformer


class HypergraphReactionLayer(nn.Module):
    """在超图上执行一轮有向消息传递。"""

    def __init__(self, molecular_dim: int, dropout: float = 0.1):
        super().__init__()
        self.reactant_to_reaction = nn.Linear(molecular_dim, molecular_dim)
        self.reaction_to_product = nn.Linear(molecular_dim, molecular_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(molecular_dim)
        self.activation = nn.GELU()

    def forward(self, node_features: torch.Tensor, H_in: torch.Tensor, H_out: torch.Tensor):
        identity = node_features

        # 步骤 1: 信息从节点 -> 超边 (聚合)
        reactant_msgs = self.activation(self.reactant_to_reaction(node_features))
        reaction_features = torch.sparse.mm(H_in.t(), reactant_msgs)

        # 步骤 2: 信息从超边 -> 节点 (传播)
        product_msgs = self.activation(self.reaction_to_product(reaction_features))
        aggregated_product_info = torch.sparse.mm(H_out, product_msgs)

        output = identity + self.dropout(aggregated_product_info)
        return self.norm(output)


class HypergraphReactionNet(nn.Module):
    """
    模型的超图处理部分。
    它接收初始的分子嵌入向量，并通过多轮超图消息传递来更新它们。
    这是一个可以被独立替换的“黑盒”组件。
    """

    def __init__(self, embed_dim: int = 64, hg_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hypergraph_layers = nn.ModuleList(
            [HypergraphReactionLayer(embed_dim, dropout) for _ in range(hg_layers)]
        )
        # 用于监督学习的最终预测头
        self.prediction_head = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, node_features: torch.Tensor, H_in: torch.Tensor, H_out: torch.Tensor):
        x_nodes = node_features
        for layer in self.hypergraph_layers:
            x_nodes = layer(x_nodes, H_in, H_out)
        # return self.prediction_head(x_nodes)
        return x_nodes


class DirectedPositionalEncoding(nn.Module):
    """
    为有向超图的【节点（分子）】和【超边（反应）】生成位置编码。
    """

    def __init__(self, embed_dim, context_role_enabled: bool = False):
        super().__init__()
        # nn.Parameter 的初始化方式更标准
        self.reactant_role_embedding = nn.Parameter(torch.randn(embed_dim))
        self.product_role_embedding = nn.Parameter(torch.randn(embed_dim))
        self.context_role_enabled = bool(context_role_enabled)
        self.context_role_embedding = (
            nn.Parameter(torch.randn(embed_dim)) if self.context_role_enabled else None
        )
        # 添加层归一化以稳定输出
        self.norm_node = nn.LayerNorm(embed_dim)
        self.norm_edge = nn.LayerNorm(embed_dim)

    def forward(self, padded_H_in, padded_H_out, padded_H_context=None):
        # padded_H_in, padded_H_out 形状: [B, max_nodes, 1]

        # 1. & 2. 使用 H 矩阵作为权重来“激活”角色嵌入
        # 将角色嵌入的形状调整为 [1, 1, D] 以便广播
        reactant_emb = self.reactant_role_embedding.view(1, 1, -1)
        product_emb = self.product_role_embedding.view(1, 1, -1)

        pv_reactant = (padded_H_in > 0).float() * reactant_emb
        pv_product = (padded_H_out > 0).float() * product_emb
        if padded_H_context is None:
            padded_H_context = torch.zeros_like(padded_H_in)
        has_context = bool(torch.any(padded_H_context > 0).item())
        if has_context and self.context_role_embedding is None:
            raise ValueError(
                "Batch contains context nodes but the model was built with context_role_enabled=False."
            )
        if self.context_role_embedding is None:
            pv_context = torch.zeros_like(pv_reactant)
        else:
            context_emb = self.context_role_embedding.view(1, 1, -1)
            pv_context = (padded_H_context > 0).float() * context_emb

        # 3. 将两个角色的编码相加，得到节点的最终位置编码
        # node_pos_encoding 形状: [B, max_nodes, D]
        node_pos_encoding = self.norm_node(pv_reactant + pv_product + pv_context)

        # --- 4. 核心修正：精确地计算超边的位置编码 ---

        # a. 创建一个掩码，精确标记每个反应中真实节点的位置
        #    node_mask 形状: [B, max_nodes, 1]
        node_mask = ((padded_H_in + padded_H_out + padded_H_context) > 0).float()

        # b. 在求和之前，使用掩码将所有【填充节点】的位置编码置为零
        #    这样它们就不会对求和/求平均产生贡献
        masked_node_pos = node_pos_encoding * node_mask

        # c. 对【真实节点】的位置编码求和
        #    .sum(dim=1) 的结果形状是 [B, D]
        summed_node_pos = masked_node_pos.sum(dim=1)

        # d. 计算每个反应的真实节点数
        #    .sum(dim=1) 的结果形状是 [B, 1]
        num_nodes_per_sample = node_mask.sum(dim=1)

        # e. 求平均
        #    [B, D] / [B, 1] -> 广播正常工作
        avg_node_pos = summed_node_pos / num_nodes_per_sample.clamp(min=1.0)

        # f. 增加一个维度，以匹配序列的格式 [B, 1, D]
        #    edge_pos_encoding 的形状现在保证是 [B, 1, D]
        edge_pos_encoding = self.norm_edge(avg_node_pos).unsqueeze(1)

        return node_pos_encoding, edge_pos_encoding


class HyperGraphTransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_dim, ffn_hidden_dim, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            Transformer(  # 您的Transformer类实际上是一个功能完备的Transformer Layer
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)  # 可选，但推荐

    def forward(self, src, src_key_padding_mask=None, attn_bias=None):
        """
        Args:
            src (Tensor): 输入序列, 形状 [B, L, D]
            src_key_padding_mask (Tensor): 布尔掩码, 形状 [B, L]
            attn_bias (Tensor): 注意力偏置, 形状 [B, L, L]
        """
        output = src

        if src_key_padding_mask is not None:
            # src_key_padding_mask follows PyTorch convention: True means padding.
            # The local attention implementation keeps mask==1 and masks mask==0.
            valid_key_mask = ~src_key_padding_mask.bool()
            attn_mask = valid_key_mask.unsqueeze(1).expand(-1, src.size(1), -1)
        else:
            attn_mask = None

        for layer in self.layers:
            # 您的 Transformer layer forward 接收 x, y, attn_mask, attn_bias
            # 对于自注意力, x 和 y 是同一个
            output, _, _ = layer(output, output, attn_mask=attn_mask, attn_bias=attn_bias)

        return self.final_layer_norm(output)

class DirectedHyperGT(nn.Module):
    """
    一个纯粹的超图Transformer模块。
    它接收一个由节点和超边表征构成的序列，并对其进行更新。
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        dropout=0.1,
        context_role_enabled: bool = False,
    ):
        super().__init__()
        self.positional_encoder = DirectedPositionalEncoding(
            embed_dim,
            context_role_enabled=context_role_enabled,
        )

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=embed_dim, nhead=num_heads,
        #     dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        # )
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder = HyperGraphTransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            ffn_hidden_dim=embed_dim,
            dropout=dropout,
            attn_dropout=dropout
        )

    def forward(
        self,
        src_initial,
        src_key_padding_mask,
        padded_H_in,
        padded_H_out,
        padded_H_context=None,
    ):
        """
        Args:
            src_initial (Tensor): 初始的、拼接好的节点和超边表征序列。形状: [B, L, D]。
            ...
        """
        max_nodes = padded_H_in.shape[1]

        # 1. 计算位置编码 (这部分逻辑已经是正确的)
        node_pos, edge_pos = self.positional_encoder(
            padded_H_in,
            padded_H_out,
            padded_H_context,
        )
        # node_pos 形状: [B, max_nodes, D]
        # edge_pos 形状: [B, 1, D]

        # --- 核心修改点：避免原地操作，改用拼接 ---

        # a. 准备【节点部分】的最终输入
        #    初始节点特征 + 节点位置编码
        src_nodes = src_initial[:, :max_nodes, :] + node_pos

        # b. 準備【超邊部分】的最終輸入
        #    初始超邊特徵 + 超邊位置編碼
        src_edge = src_initial[:, max_nodes:, :] + edge_pos

        # c. 使用 torch.cat() 将两部分拼接成最终的输入序列
        #    [B, max_nodes, D] cat [B, 1, D] -> [B, max_nodes + 1, D]
        src_final = torch.cat([src_nodes, src_edge], dim=1)

        # 2. 通过Transformer
        #    现在，我们传递的是一个全新的、内存连续的张量 src_final
        transformer_output = self.transformer_encoder(
            src=src_final,
            src_key_padding_mask=src_key_padding_mask
        )

        return transformer_output
