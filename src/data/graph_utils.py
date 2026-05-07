"""
Graph utility functions — framework-agnostic helpers
for both PyG and DGL workflows.
"""

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ─── PyG helpers ─────────────────────────

def to_undirected_pyg(data):
    """Add reverse edges to make a directed PyG graph undirected."""
    from torch_geometric.utils import to_undirected
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    return data


def add_self_loops_pyg(data):
    from torch_geometric.utils import add_self_loops
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return data


def get_degree_stats_pyg(edge_index, num_nodes):
    from torch_geometric.utils import degree
    deg = degree(edge_index[0], num_nodes=num_nodes)
    print(f"Degree | mean: {deg.mean():.2f} | max: {deg.max():.0f} | "
          f"min: {deg.min():.0f} | std: {deg.std():.2f}")
    return deg


# ─── DGL helpers ─────────────────────────

def get_degree_stats_dgl(graph):
    deg = graph.in_degrees().float()
    print(f"Degree | mean: {deg.mean():.2f} | max: {deg.max():.0f} | "
          f"min: {deg.min():.0f} | std: {deg.std():.2f}")
    return deg


# ─── Negative sampling for link prediction ───

def negative_sample_pyg(edge_index, num_nodes, num_neg_samples=None):
    """Fast negative sampling (uniform random, no guaranteed non-overlap)."""
    from torch_geometric.utils import negative_sampling
    num_neg = num_neg_samples or edge_index.size(1)
    neg_edge = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg
    )
    return neg_edge


# ─── Visualization ───────────────────────

def plot_degree_distribution(deg_tensor, title="Degree Distribution", log_scale=True):
    deg = deg_tensor.numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(deg, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Count (log)" if log_scale else "Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("reports/figures/degree_distribution.png", dpi=150)
    plt.show()
    print("Saved → reports/figures/degree_distribution.png")


def plot_label_distribution(labels, num_classes, title="Label Distribution"):
    counts = torch.bincount(labels.squeeze(), minlength=num_classes).numpy()
    plt.figure(figsize=(12, 4))
    plt.bar(range(num_classes), counts, color="coral", edgecolor="white", linewidth=0.3)
    plt.xlabel("Class")
    plt.ylabel("Node count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("reports/figures/label_distribution.png", dpi=150)
    plt.show()
    print("Saved → reports/figures/label_distribution.png")