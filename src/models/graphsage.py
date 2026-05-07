"""
GraphSAGE (Hamilton et al., 2017)
for node classification on ogbn-arxiv.

Key advantage over GCN/GAT: inductive — generalizes to unseen nodes.
Uses mean aggregation (strongest on ogbn-arxiv benchmarks).

Architecture:
    Input(128) → [SAGEConv → BN → ReLU → Dropout] x num_layers → Linear(40)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = "mean",     # "mean" | "max" | "lstm"
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # raw logits


def build_graphsage(num_classes: int = 40) -> GraphSAGE:
    """Default config tuned for ogbn-arxiv."""
    return GraphSAGE(
        in_channels=128,
        hidden_channels=256,
        out_channels=num_classes,
        num_layers=3,
        dropout=0.5,
        aggr="mean",
    )