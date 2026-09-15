import torch
import torch.nn as nn
from torch.nn import Embedding
from torch import Tensor

from hypermol.data.preprocess import CompoundKit
from typing import Dict

from hypermol.data.edge_types import EDGE_TYPES_DICT


@torch.jit.script
def gaussian(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    # noinspection PyTypeChecker
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (torch.sqrt(2 * torch.pi) * std)

# noinspection PyPep8Naming
class GaussianKernel(nn.Module):
    def __init__(self, K: int = 128, std_width: float = 1.0, start: float = 0.0, stop: float = 9.0):
        super().__init__()
        self.K = K
        mean = torch.linspace(start, stop, K)
        std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mul = Embedding(1, 1, padding_idx=0)
        self.bias = Embedding(1, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1.0)

    # 128, 59 --> 128, 59, 256
    def forward(self, x: Tensor) -> Tensor:
        # # x shape: (batch_size, atom_num), and atom_num is max(atom_num) of mol in the batch. padding is float 0.0
        # # assert that every mol has atom_num more than 1 (note that padding is 0.0)
        # def get_zero_mask_num(x: Tensor, threshold: float = 0.0001) -> Tensor:
        #     """
        #     x is shape of (1, xxx)
        #     Args:
        #         x:
        #         threshold:
        #
        #     Returns:
        #
        #     """
        #     return (x.abs() < threshold).sum()
        # max_len_mol = x.shape[1]
        # assert all (get_zero_mask_num(x[i]) < max_len_mol - 1 for i in range(x.shape[0]))
        mul = self.mul.weight
        bias = self.bias.weight
        x = (mul * x.unsqueeze(-1)) + bias
        expand_shape = [-1] * len(x.shape)
        expand_shape[-1] = self.K
        x = x.expand(expand_shape)
        mean = self.mean.float()
        # if torch.isnan(gaussian(x.float(), mean, self.std)).any():
        #     print(gaussian(x.float(), mean, self.std))‘
        results = gaussian(x.float(), mean, self.std)
        return results

class AtomEmbedding(nn.Module):

    def __init__(self, atom_names, embed_dim, num_kernel):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim, padding_idx=0)
            self.embed_list.append(embed)

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.graph_finger_print = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mass_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.van_der_waals_radis_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.partial_charge_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim)
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, node_features: Dict[str, Tensor]):
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[name])
        if "mass" in node_features:
            mass_embed = self.mass_embedding(node_features["mass"])
            out_embed += mass_embed
        if "van_der_waals_radis" in node_features:
            van_der_waals_radis_embed = self.van_der_waals_radis_embedding(node_features["van_der_waals_radis"])
            out_embed += van_der_waals_radis_embed
        if "partial_charge" in node_features:
            partial_charge_embed = self.partial_charge_embedding(node_features["partial_charge"])
            out_embed += partial_charge_embed

        graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(out_embed.size()[0], 1, 1)
            # print_debug(f"WARNING: Inited with graph_embedding instead of morgan2048_fp")
        # graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(out_embed.size()[0], 1, 1)   # 把这个注了
        # graph_token_embed = self.graph_finger_print(node_features["morgan2048_fp"].to(torch.float32)).unsqueeze(1)

        out_embed = torch.cat([graph_token_embed, out_embed], dim=1)
        # normalize
        # out_embed = out_embed / (out_embed.norm(dim=-1, keepdim=True) + 1e-5)
        out_embed = self.final_layer_norm(out_embed)
        return out_embed

class BondEmbedding(nn.Module):
    def __init__(self, bond_names, embed_dim, num_kernel):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim, padding_idx=0)
            self.embed_list.append(embed)

        self.edge_type_embed = nn.Embedding(len(EDGE_TYPES_DICT) + 4, embed_dim, padding_idx=0)

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.distance_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
            nn.Softplus()
        )

    def forward(self, edge_features: Dict[str, Tensor]) -> Tensor:
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            # print(name)
            out_embed += self.embed_list[i](edge_features[name].to(torch.int64))
        edge_type_embed = self.edge_type_embed(edge_features["edge_types"])

        out_embed += edge_type_embed
        distance_embed = self.distance_embedding(edge_features["bond_length"])
        out_embed += distance_embed
        graph_token_embed = self.graph_embedding.weight.unsqueeze(0).repeat(out_embed.size()[0], 1, 1)
        out_embed = torch.cat([graph_token_embed, out_embed], dim=1)

        return out_embed

# noinspection PyUnresolvedReferences
class PairEmbedding(nn.Module):
    def __init__(self, bond_names, embed_dim, num_kernel):
        super(PairEmbedding, self).__init__()
        self.bond_names = bond_names

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.distance_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
            nn.Softplus()
        )

    def forward(self, pair_distances):
        # print_debug("Device of pair_distances: ", pair_distances.device)
        out_embed = self.distance_embedding(pair_distances)
        # print_debug("Device of out_embed: ", out_embed.device)
        graph_token_embed = self.graph_embedding.weight.view(1, 1, -1)

        bond_embed = torch.zeros(out_embed.size()[0], out_embed.size()[1] + 1, out_embed.size()[2] + 1,
                                 out_embed.size()[3], device=out_embed.device)
        bond_embed[:, 0, 1:, :] = graph_token_embed
        bond_embed[:, 1:, 0, :] = graph_token_embed
        bond_embed[:, 1:, 1:, :] = out_embed

        return bond_embed.permute(0, 3, 1, 2)

class SpatialPosEncoder(nn.Module):
    """
    计算完整的空间位置偏置 (Spatial Encoding)，包括虚拟节点。
    输出一个形状为 [B, num_heads, N+1, N+1] 的偏置矩阵。
    """

    def __init__(self, embed_dim, num_spatial_dist):
        super().__init__()
        self.embed_dim = embed_dim

        # 为真实节点间的距离创建嵌入层
        # num_embeddings = max_dist + 2 (一个给真实距离[0, max_dist]，一个给padding)
        num_embeddings = num_spatial_dist + 2
        self.encoder = nn.Embedding(num_embeddings, self.embed_dim, padding_idx=0)

        # 为虚拟节点与真实节点间的“特殊距离”创建专属嵌入
        self.virtual_node_distance = nn.Embedding(1, self.embed_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        B = len(batch['atom_length'])
        # N_real = batch['atom_length'].max().item()
        N_real = batch['atomic_num'].shape[1]
        N_total = N_real + 1
        device = batch['atom_distances_2d'].device

        # 1. 计算真实节点间的偏置
        spatial_pos = batch['atom_distances_2d']
        spatial_pos_processed = spatial_pos + 1
        real_nodes_bias = self.encoder(spatial_pos_processed).permute(0, 3, 1, 2)

        # 2. 获取虚拟节点偏置
        virtual_dist_bias = self.virtual_node_distance.weight.view(1, self.embed_dim, 1).repeat(B, 1, 1)

        # 3. 组合成 (N+1, N+1) 的完整矩阵
        total_bias = torch.zeros(B, self.embed_dim, N_total, N_total, device=device)

        # 填充右下角
        total_bias[:, :, 1:, 1:] = real_nodes_bias

        # 填充第一行和第一列 (修正后)
        total_bias[:, :, 0, 1:] = virtual_dist_bias
        total_bias[:, :, 1:, 0] = virtual_dist_bias  # PyTorch的广播机制会自动处理

        return total_bias

# noinspection PyIncorrectDocstring,PyPep8Naming
class BondAngelEmbedding(nn.Module):
    def __init__(self, bond_names, embed_dim, num_kernel):
        super(BondAngelEmbedding, self).__init__()
        self.bond_names = bond_names

        self.graph_embedding = nn.Embedding(1, embed_dim)

        self.distance_embedding = nn.Sequential(
            GaussianKernel(K=num_kernel, std_width=1.0, start=0.0, stop=9.0),
            nn.Linear(num_kernel, embed_dim),
            nn.Softplus()
        )

    # noinspection PyUnresolvedReferences
    def forward(self, BondAngel):
        out_embed = self.distance_embedding(BondAngel)
        # print_debug("Device of out_embed: ", out_embed.device)
        graph_token_embed = self.graph_embedding.weight.view(1, 1, -1)
        bond_embed = torch.zeros(out_embed.size()[0], out_embed.size()[1] + 1, out_embed.size()[2] + 1,
                                 out_embed.size()[3], device=out_embed.device)
        bond_embed[:, 0, :, :] = graph_token_embed
        bond_embed[:, 1:, 0, :] = graph_token_embed
        bond_embed[:, 1:, 1:, :] = out_embed

        return bond_embed

class EdgeEncoder(nn.Module):
    """
    计算完整的边路径特征偏置 (Edge Encoding)，包括为虚拟节点留出空位。
    输出一个形状为 [B, num_heads, N+1, N+1] 的偏置矩阵。
    """

    def __init__(self, bond_id_names, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.bond_id_names = bond_id_names

        # Embedding 层只为真实路径上的边服务
        self.edge_encoders = nn.ModuleDict()
        for name in self.bond_id_names:
            # num_embeddings = 原始词典大小 + 1 (为 padding_idx=0 留出位置)
            num_embeddings = len(CompoundKit.bond_vocab_dict[name]) + 1
            self.edge_encoders[name] = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=self.embed_dim,
                padding_idx=0
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch (Dict): 包含路径特征 (如 'bond_type') 和 'atom_length' 的批处理数据。

        Returns:
            torch.Tensor: 形状为 [B, num_heads, N+1, N+1] 的完整边偏置。
        """
        # 获取维度信息
        B = len(batch['atom_length'])
        # N_real = batch['atom_length'].max().item()
        N_real = batch['atomic_num'].shape[1]
        N_total = N_real + 1

        # 需要从 batch 中找到一个 tensor 来确定 device
        ref_key = self.bond_id_names[0]
        if ref_key not in batch or batch[ref_key] is None:
            # 如果批次中没有任何路径特征，返回一个 (N+1, N+1) 的零矩阵
            # 需要找到一个 device
            device_tensor = next(t for t in batch.values() if isinstance(t, torch.Tensor))
            return torch.zeros(B, self.embed_dim, N_total, N_total, device=device_tensor.device)

        device = batch[ref_key].device

        # 1. 计算真实节点间的偏置
        # 初始化一个 [B, N_real, N_real, num_heads] 的零矩阵来累加结果
        total_real_edge_bias_raw = torch.zeros(B, N_real, N_real, self.embed_dim, device=device)

        for name in self.bond_id_names:
            if name not in batch or batch[name] is None:
                continue

            # 获取该属性的路径特征张量 (已经是 +1 处理过的)
            path_feature_tensor = batch[name]  # 形状: [B, N_real, N_real, D_path]

            # 通过 Embedding 层编码
            path_embedding = self.edge_encoders[name](path_feature_tensor)  # -> [B, N, N, D_path, num_heads]

            # 对路径长度维度 D_path 求平均
            path_mask = (path_feature_tensor != 0).unsqueeze(-1)
            summed_path_embedding = path_embedding.sum(dim=3)
            path_lengths = path_mask.sum(dim=3).clamp(min=1)
            avg_path_embedding = summed_path_embedding / path_lengths  # -> [B, N, N, num_heads]

            # 累加
            total_real_edge_bias_raw += avg_path_embedding

        # 调整维度
        real_nodes_bias = total_real_edge_bias_raw.permute(0, 3, 1, 2)  # -> [B, num_heads, N_real, N_real]

        # 2. 组合成 (N+1, N+1) 的完整矩阵
        # 虚拟节点没有边路径特征，所以它的行和列都是0
        total_bias = torch.zeros(B, self.embed_dim, N_total, N_total, device=device)

        total_bias[:, :, 1:, 1:] = real_nodes_bias

        return total_bias
