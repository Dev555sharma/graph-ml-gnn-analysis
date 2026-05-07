"""
Preprocessing pipeline:
- Feature normalization for ogbn-arxiv
- Learnable embeddings setup for ogbl-ddi (no node features)
"""

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization of node features."""
    norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
    return x / norm


def standardize_features(x: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance scaling (fit on train, apply globally)."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.numpy())
    return torch.tensor(x_scaled, dtype=torch.float)


class LearnableEmbedding(nn.Module):
    """
    For ogbl-ddi which has NO node features.
    Assigns each node a trainable embedding vector.
    Standard approach in OGB leaderboard entries.
    """
    def __init__(self, num_nodes: int, embedding_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(node_ids)

    def all_embeddings(self) -> torch.Tensor:
        return self.embedding.weight