"""
Dataset loaders for OGB-arxiv (node classification)
and OGB-ddi (link prediction).
Framework-agnostic: returns both PyG and DGL formats.
"""

import functools
import numpy as np
import torch

# ─── PyTorch 2.6 compatibility patch ─────────────────────────────────────────
# OGB was built before torch.load changed default to weights_only=True in 2.6.
# Monkey-patch torch.load to force weights_only=False for this script.
_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)

torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────────────────────────

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

try:
    from ogb.nodeproppred import DglNodePropPredDataset
    from ogb.linkproppred import DglLinkPropPredDataset
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False


# ─────────────────────────────────────────
# OGB-arxiv  →  Node Classification
# ─────────────────────────────────────────

def load_arxiv_pyg(root: str = "data/raw"):
    from ogb.nodeproppred import Evaluator
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-arxiv")
    print(f"[PyG] ogbn-arxiv | Nodes: {data.num_nodes:,} | "
          f"Edges: {data.num_edges:,} | Features: {data.num_node_features} | "
          f"Classes: {dataset.num_classes}")
    return data, split_idx, evaluator


def load_arxiv_dgl(root: str = "data/raw"):
    if not DGL_AVAILABLE:
        raise ImportError("DGL not available. Run: pip install dgl")
    from ogb.nodeproppred import Evaluator
    import dgl
    dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=root)
    graph, labels = dataset[0]
    graph = dgl.add_reverse_edges(graph)
    graph = dgl.add_self_loop(graph)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-arxiv")
    print(f"[DGL] ogbn-arxiv | Nodes: {graph.num_nodes():,} | Edges: {graph.num_edges():,}")
    return graph, labels, split_idx, evaluator


# ─────────────────────────────────────────
# OGB-ddi  →  Link Prediction
# ─────────────────────────────────────────

def load_ddi_pyg(root: str = "data/raw"):
    from ogb.linkproppred import Evaluator
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=root)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name="ogbl-ddi")
    print(f"[PyG] ogbl-ddi | Nodes: {data.num_nodes:,} | "
          f"Edges: {data.num_edges:,} | Node features: NONE (use embeddings)")
    return data, split_edge, evaluator


def load_ddi_dgl(root: str = "data/raw"):
    if not DGL_AVAILABLE:
        raise ImportError("DGL not available. Run: pip install dgl")
    from ogb.linkproppred import Evaluator
    dataset = DglLinkPropPredDataset(name="ogbl-ddi", root=root)
    graph = dataset[0]
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name="ogbl-ddi")
    print(f"[DGL] ogbl-ddi | Nodes: {graph.num_nodes():,} | Edges: {graph.num_edges():,}")
    return graph, split_edge, evaluator


# ─────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Loading ogbn-arxiv (PyG)...")
    data, split_idx, _ = load_arxiv_pyg()
    print(f"  Train: {len(split_idx['train']):,} | Valid: {len(split_idx['valid']):,} | Test: {len(split_idx['test']):,}")

    print("=" * 50)
    print("Loading ogbl-ddi (PyG)...")
    data, split_edge, _ = load_ddi_pyg()
    print(f"  Train edges: {len(split_edge['train']['edge']):,}")

    if DGL_AVAILABLE:
        print("=" * 50)
        print("Loading ogbn-arxiv (DGL)...")
        load_arxiv_dgl()
        print("=" * 50)
        print("Loading ogbl-ddi (DGL)...")
        load_ddi_dgl()
    else:
        print("\n[!] DGL not installed — skipping DGL loaders.")
        print("    To enable: pip install dgl -f https://data.dgl.ai/wheels/repo.html")

    print("\n✓ Done.")