"""Implements MLP trained on embeddings for probing tasks
"""
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.layer1(X))
        X = self.dropout(X)
        X = F.softmax(self.layer2(X), dim=1)
        return X
