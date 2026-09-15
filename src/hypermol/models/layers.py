from typing import List

import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
import math


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, activation_fn="GELU", dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x += residual
        x = self.ffn_layer_norm(x)
        return x

# noinspection DuplicatedCode
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
                 use_super_node=True, cross_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.d_k = hidden_dim // num_heads
        self.num_heads = num_heads  # number of heads
        self.temperature = temperature
        self.use_super_node = use_super_node
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)

        self.w_gate_x = nn.Linear(hidden_dim, hidden_dim)
        self.w_gate_c = nn.Linear(hidden_dim, hidden_dim)

        # self.a_proj_multiscale = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(3)])

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attn = cross_attn

    def forward(self, x, y, mask=None, attn_bias=None):
        residual = x
        batch_size = x.size(0)

        if not self.cross_attn:
            y = x
        query = self.q_proj(x)  # (batch_size, atom_num, hidden_dim)
        key = self.k_proj(y)  # (batch_size, atom_num, hidden_dim)
        value = self.v_proj(y)  # (batch_size, atom_num, hidden_dim)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ScaledDotProductAttention
        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if attn_bias is not None:
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # bacth * len * len
            scores = scores + attn_bias

        if mask is not None:
            # A hard-coded -1e12 overflows when attention runs in float16
            # (notably with AMP on recent PyTorch/RTX 50-series setups).
            # Keep the mask finite, as before, but choose a value representable
            # by the score tensor's actual dtype.
            mask_value = torch.finfo(scores.dtype).min
            if scores.shape == mask.shape:  # different heads have different mask
                scores = scores * mask
                scores = scores.masked_fill(scores == 0, mask_value)
            else:
                scores = scores.masked_fill(mask == 0, mask_value)

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        gate_scores = torch.sigmoid(self.w_gate_x(x))
        # gate_scores = torch.sigmoid(self.w_gate_x(x) + self.w_gate_c(context))
        gated_context = context * gate_scores
        out = self.dropout(self.a_proj(gated_context))
        # out = self.dropout(self.a_proj(context))
        out += residual
        out = self.layer_norm(out)

        return out, attn, scores


class Transformer(nn.Module):
    def __init__(self, num_heads, hidden_dim, ffn_hidden_dim, dropout=0.1, attn_dropout=0.1, temperature=1,
                 activation_fn='GELU', cross_attn=False):
        super(Transformer, self).__init__()
        assert hidden_dim % num_heads == 0
        self.cross_attn_flag = cross_attn
        self.self_attention = MultiHeadAttention(num_heads, hidden_dim, dropout, attn_dropout, temperature,
                                                 cross_attn=False)
        self.self_ffn_layer = PositionWiseFeedForward(hidden_dim, ffn_hidden_dim, activation_fn=activation_fn)

        if self.cross_attn_flag:
            self.cross_attention = MultiHeadAttention(num_heads, hidden_dim, dropout, attn_dropout, temperature,
                                                      cross_attn=True)
            self.cross_ffn_layer = PositionWiseFeedForward(hidden_dim, ffn_hidden_dim, activation_fn=activation_fn)

    def forward(self, x, y, attn_mask, attn_bias=None):
        if self.cross_attn_flag and y is not None:
            x, attn, attn_bias = self.cross_attention(x, y, mask=attn_mask, attn_bias=attn_bias)
            x = self.cross_ffn_layer(x)
        else:
            x, attn, attn_bias = self.self_attention(x, y, mask=attn_mask, attn_bias=attn_bias)
            x = self.self_ffn_layer(x)

        return x, attn, attn_bias


class EncoderAtomLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, num_heads, dropout=0.1, attn_dropout=0.1, temperature=1,
                 activation_fn='GELU', cross_attn=False):
        super(EncoderAtomLayer, self).__init__()
        # self.transformer_self = Transformer_Layer(num_heads, hidden_dim, ffn_hidden_dim, dropout, attn_dropout, temperature,
        #                                      activation_fn)
        self.transformer_self = Transformer(num_heads, hidden_dim, ffn_hidden_dim, dropout, attn_dropout, temperature,
                                             activation_fn, cross_attn=cross_attn)
        # self.transformer_self = KANTransformer(num_heads, hidden_dim, ffn_hidden_dim, dropout, attn_dropout, temperature,
        #                                      activation_fn, cross_attn=cross_attn)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        # self.reset_parameters()
        self.angel_bias_linear = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            # nn.LayerNorm(1),
            # nn.SiLU(),
            nn.Dropout(p=dropout),
            # nn.Softplus()
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer_norm.weight)

    def forward(self, x, attn_mask, attn_bias=None):
        batch_size, atom_num, hidden_dim = x.size()
        # 假设 batch_size, atom_num, hidden_dim = x.size()
        x, attn, attn_bias = self.transformer_self(x, None, attn_mask, attn_bias)


        return x, attn_bias



