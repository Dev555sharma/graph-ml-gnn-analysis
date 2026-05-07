"""
Metric utilities wrapping OGB evaluators.
Provides clean, reusable functions for both tasks.
"""

import torch
import numpy as np
from typing import Dict


# ─── Node Classification ─────────────────────────────────────────────────────

def compute_accuracy(y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     evaluator) -> float:
    """
    Compute OGB accuracy for node classification.

    Args:
        y_true : [N, 1] ground truth labels
        y_pred : [N, 1] predicted class indices
        evaluator : OGB Evaluator(name='ogbn-arxiv')
    Returns:
        accuracy as float
    """
    return evaluator.eval({
        "y_true": y_true,
        "y_pred": y_pred,
    })["acc"]


def node_metrics_report(results: Dict[str, float]) -> str:
    """Format train/valid/test accuracy dict into a readable string."""
    return (f"Train: {results['train']:.4f} | "
            f"Val: {results['valid']:.4f} | "
            f"Test: {results['test']:.4f}")


# ─── Link Prediction ─────────────────────────────────────────────────────────

def compute_hits_at_k(y_pred_pos: torch.Tensor,
                      y_pred_neg: torch.Tensor,
                      evaluator,
                      k: int = 20) -> float:
    """
    Compute OGB Hits@K for link prediction.

    Args:
        y_pred_pos : scores for positive edges [E_pos]
        y_pred_neg : scores for negative edges [E_neg]
        evaluator  : OGB Evaluator(name='ogbl-ddi')
        k          : K for Hits@K (default 20)
    Returns:
        hits@k as float
    """
    return evaluator.eval({
        "y_pred_pos": y_pred_pos,
        "y_pred_neg": y_pred_neg,
    })[f"hits@{k}"]


def link_metrics_report(results: Dict[str, float], k: int = 20) -> str:
    """Format valid/test hits dict into a readable string."""
    return (f"Val Hits@{k}: {results['valid']:.4f} | "
            f"Test Hits@{k}: {results['test']:.4f}")


# ─── General ─────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module, model_name: str = "Model"):
    """Print parameter count and layer breakdown."""
    total = count_parameters(model)
    print(f"\n{'─'*40}")
    print(f"  {model_name} — {total:,} trainable parameters")
    print(f"{'─'*40}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<40} {str(list(param.shape)):<20} "
                  f"{param.numel():>10,}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.models.gcn import build_gcn
    from src.models.gat import build_gat
    from src.models.graphsage import build_graphsage
    from src.models.link_predictor import build_ddi_model

    for name, model in [("GCN", build_gcn()),
                         ("GAT", build_gat()),
                         ("GraphSAGE", build_graphsage())]:
        print(f"{name}: {count_parameters(model):,} params")

    enc, pred = build_ddi_model()
    print(f"DDI total: {count_parameters(enc)+count_parameters(pred):,} params")