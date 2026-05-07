"""
Graph Attention Network v2 (Brody et al., 2022)
for node classification on ogbn-arxiv.

Uses GATv2Conv (fixes rank collapse issue in original GAT).

Architecture:
    Input(128) → [GATv2Conv → BN → ELU → Dropout] x num_layers → Linear(40)

Key design choices:
    - GATv2 instead of GAT (strictly more expressive)
    - Multi-head attention with concatenation in hidden, averaging at output
    - ELU activation (standard for attention-based GNNs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,           # 128
        hidden_channels: int,       # 128 per head
        out_channels: int,          # 40
        num_layers: int = 3,
        heads: int = 4,             # attention heads (hidden layers)
        output_heads: int = 1,      # attention heads (output layer)
        dropout: float = 0.5,
        attn_dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads,
                      dropout=attn_dropout, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads,
                          dropout=attn_dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Output layer — average heads
        self.convs.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=output_heads,
                      dropout=attn_dropout, concat=False)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # raw logits


def build_gat(num_classes: int = 40) -> GAT:
    """Default config tuned for ogbn-arxiv."""
    return GAT(
        in_channels=128,
        hidden_channels=128,
        out_channels=num_classes,
        num_layers=3,
        heads=4,
        output_heads=1,
        dropout=0.5,
        attn_dropout=0.3,
    )