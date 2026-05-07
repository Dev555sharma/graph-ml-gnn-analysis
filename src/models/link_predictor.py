"""
Link Predictor for ogbl-ddi (drug-drug interaction).

Since DDI has NO node features, we use learnable node embeddings
fed into GraphSAGE layers, then an MLP dot-product predictor head.

Architecture:
    NodeEmbedding(4267, 256)
        → GraphSAGE encoder (256 → 256)
            → MLPPredictor(h_u ⊙ h_v → score)

This follows the top OGB-DDI leaderboard design pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class LinkPredictor(nn.Module):
    """
    MLP that scores a pair of node embeddings.
    Input: element-wise product of (h_u, h_v).
    Output: scalar edge score (pre-sigmoid).
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.lins.append(nn.Linear(hidden_channels, 1))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_i: embeddings of source nodes  [E, D]
            x_j: embeddings of target nodes  [E, D]
        Returns:
            scores: [E, 1]  (raw logits, apply sigmoid for probability)
        """
        x = x_i * x_j  # element-wise product
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = lin(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x


class DDIEncoder(nn.Module):
    """
    Full encoder for ogbl-ddi:
        Learnable embeddings → GraphSAGE layers → node representations
    """
    def __init__(
        self,
        num_nodes: int,         # 4,267 for DDI
        embedding_dim: int,     # 256
        hidden_channels: int,   # 256
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(embedding_dim, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        nn.init.xavier_uniform_(self.embedding.weight)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns embeddings for ALL nodes."""
        x = self.embedding.weight  # [N, D]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [N, hidden_channels]


def build_ddi_model(num_nodes: int = 4267):
    """Returns (encoder, predictor) tuple ready for training."""
    encoder = DDIEncoder(
        num_nodes=num_nodes,
        embedding_dim=256,
        hidden_channels=256,
        num_layers=2,
        dropout=0.5,
    )
    predictor = LinkPredictor(
        in_channels=256,
        hidden_channels=256,
        num_layers=3,
    )
    return encoder, predictor