"""Implements MLP trained on embeddings for probing tasks
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim=64, dropout=0.0):
        super().__init__()
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.q_vector = nn.Linear(head_dim, 1)
        self.k_layer = nn.Linear(input_dim, head_dim)
        self.v_layer = nn.Linear(input_dim, head_dim)
        self.scale = math.sqrt(head_dim)

        self.ff_layer = nn.Linear(head_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, **kwargs):
        # X, k, v: [batch size, max seq len, input_dim]
        # q: [batch size, max seq len]
        k = self.k_layer(X)
        q = self.q_vector(k).squeeze()
        q.div_(self.scale)

        # mask: [batch size, max seq len]
        # attn_weights: [batch size, 1,  max seq len]
        # attentioned: [batch size, max seq len]
        mask = ~ torch.any(X == 0, dim=2)
        q.masked_fill_(mask == 0, -1e9)
        attn_weights = F.softmax(q, dim=1).unsqueeze(1)

        v = self.v_layer(X)
        attentioned = attn_weights.bmm(v).squeeze()

        X = self.dropout(attentioned)
        X = F.relu(X)

        if self.output_dim == 1:
            X = self.ff_layer(X)
            return X.squeeze(-1)
        else:
            X = F.softmax(self.ff_layer(X), dim=1)
            return X
