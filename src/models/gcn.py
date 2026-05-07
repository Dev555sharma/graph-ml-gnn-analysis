"""
Graph Convolutional Network (Kipf & Welling, 2017)
for node classification on ogbn-arxiv.

Architecture:
    Input(128) → [GCNConv → BN → ReLU → Dropout] x num_layers → Linear(40)

Key design choices for ogbn-arxiv leaderboard:
    - Batch normalization after each conv (critical for deep GCNs)
    - Residual connections from layer 2 onward
    - Label smoothing in loss for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,       # 128 for ogbn-arxiv
        hidden_channels: int,   # 256 recommended
        out_channels: int,      # 40 classes
        num_layers: int = 3,
        dropout: float = 0.5,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Residual: add input if same shape (from layer 2 onward)
            if self.use_residual and h.shape == x.shape:
                h = h + x
            x = h
        x = self.convs[-1](x, edge_index)
        return x  # raw logits — apply softmax externally


def build_gcn(num_classes: int = 40) -> GCN:
    """Default config tuned for ogbn-arxiv."""
    return GCN(
        in_channels=128,
        hidden_channels=256,
        out_channels=num_classes,
        num_layers=3,
        dropout=0.5,
        use_residual=True,
    )