from typing import Dict

import loguru
import torch
import torch.nn as nn
# from hypermol.utils.debug import profile
from torch import Tensor
import sys

from torch.nn.functional import dropout

from hypermol.models.layers import EncoderAtomLayer
from hypermol.models.embedding import AtomEmbedding, BondEmbedding, PairEmbedding, BondAngelEmbedding, SpatialPosEncoder, EdgeEncoder


class MolecularEncoder(nn.Module):
    def __init__(self, mode, atom_names, bond_names, embed_dim, num_kernel,
                 layer_num, num_heads, hidden_size, cross_layers, dropout, num_tasks=1):
        super(MolecularEncoder, self).__init__()

        self.mode = mode
        self.atom_names = list(atom_names)
        self.bond_names = list(bond_names)
        self.embed_dim = embed_dim
        self.num_kernel = num_kernel
        self.layer_num = layer_num
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_tasks = num_tasks

        self.atom_feature = AtomEmbedding(self.atom_names, self.embed_dim, self.num_kernel)
        self.bond_feature = BondEmbedding(self.bond_names, self.embed_dim, self.num_kernel)

        self.pair_feature = PairEmbedding(self.bond_names, self.num_heads, self.num_kernel)
        self.spatial_encoder = SpatialPosEncoder(self.num_heads, 511)
        self.edge_encoder = EdgeEncoder(self.bond_names, self.num_heads)
        self.angle_feature = BondAngelEmbedding(self.bond_names, self.num_heads, self.num_kernel)

        self.EncoderAtomList = nn.ModuleList()
        self.EncoderBondList = nn.ModuleList()
        for i in range(self.layer_num):
            cross_attn_flag = i > cross_layers
            self.EncoderAtomList.append(
                EncoderAtomLayer(self.embed_dim, self.hidden_size, self.num_heads, self.dropout, self.dropout, cross_attn=cross_attn_flag)
            )

    def forward(self, batched_data: Dict[str, Tensor]):
        # any tensor value of the batched_data should have the first dim `batch_size`
        atom: Tensor = self.atom_feature(batched_data)
        batch, atom_num, _ = atom.shape
        spatial_bias = self.spatial_encoder(batched_data)
        edge_bias = self.edge_encoder(batched_data)

        # atom_distances_2d = self.spatial_encoder(len(batched_data["atom_length"]), batched_data["atom_pos"],
        #                                          batched_data["atom_distances_2d"])

        atom_attention_mask: Tensor = batched_data["atom_attention_mask"]

        bias = spatial_bias + edge_bias

        for i in range(self.layer_num):
            atom, bias = self.EncoderAtomList[i](atom, atom_attention_mask, bias)

        predictions = {}
        predictions['mol_features'] = atom[:, 0, :]
        predictions['atom_features'] = atom[:, 1:, :]

        return predictions
