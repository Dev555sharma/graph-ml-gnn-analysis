"""
Visualization utilities for GNN project.
All plot functions save to reports/figures/ automatically.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_learning_curves(history: Dict[str, List[float]],
                         model_name: str,
                         best_epoch: Optional[int] = None,
                         filename: Optional[str] = None):
    """
    Plot train/valid/test accuracy over epochs for one model.

    Args:
        history    : dict with keys 'train', 'valid', 'test', 'loss'
        model_name : display name for title
        best_epoch : draw vertical line at best epoch
        filename   : save filename (auto-generated if None)
    """
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Accuracy
    axes[0].plot(epochs, history["train"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["valid"], label="Valid",
                 linewidth=2, linestyle="--")
    axes[0].plot(epochs, history["test"],  label="Test",
                 linewidth=1.5, linestyle=":", alpha=0.8)
    if best_epoch:
        axes[0].axvline(best_epoch, color="gray", linestyle="--",
                        alpha=0.5, label=f"Best @ {best_epoch}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(epochs, history["loss"], color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title(f"{model_name} — Training Loss")

    plt.suptitle(f"{model_name} on ogbn-arxiv", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = filename or f"curves_{model_name.lower()}.png"
    _save(fname)
    plt.show()


def plot_all_learning_curves(results: Dict[str, dict]):
    """Plot learning curves for all models side by side."""
    colors = {"gcn": "#4C72B0", "gat": "#C44E52", "graphsage": "#55A868"}
    titles = {"gcn": "GCN", "gat": "GAT (v2)", "graphsage": "GraphSAGE"}
    n = len(results)

    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        h = r["history"]
        epochs = range(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], color=colors[name], linewidth=2, label="Train")
        ax.plot(epochs, h["valid"], color=colors[name], linewidth=2,
                linestyle="--", label="Valid")
        ax.plot(epochs, h["test"],  color=colors[name], linewidth=1.5,
                linestyle=":", alpha=0.7, label="Test")
        ax.axvline(r["best_epoch"], color="gray", linestyle="--",
                   alpha=0.5, label=f"Best @ {r['best_epoch']}")
        ax.set_title(f"{titles[name]}\nTest: {r['test_acc']*100:.2f}%")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Accuracy")
    plt.suptitle("Learning Curves — ogbn-arxiv", fontsize=14, y=1.02)
    plt.tight_layout()
    _save("learning_curves_all.png")
    plt.show()


# ─── Model comparison ────────────────────────────────────────────────────────

def plot_model_comparison(results: Dict[str, dict]):
    """Bar chart comparing train/val/test accuracy for all models."""
    models     = [r["model"] for r in results.values()]
    test_accs  = [r["test_acc"] * 100 for r in results.values()]
    val_accs   = [r["val_acc"]  * 100 for r in results.values()]
    train_accs = [r["train_acc"] * 100 for r in results.values()]

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w, train_accs, w, label="Train", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,     val_accs,   w, label="Valid", color="#55A868", alpha=0.85)
    b3 = ax.bar(x + w, test_accs,  w, label="Test",  color="#C44E52", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 90)
    ax.set_title("GNN Model Comparison — ogbn-arxiv")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save("model_comparison_bar.png")
    plt.show()


# ─── Ablation ────────────────────────────────────────────────────────────────

def plot_depth_ablation(depth_results: Dict[int, dict], k: int = 20):
    """Line plot of accuracy vs number of GNN layers."""
    layers = sorted(depth_results.keys())
    vals   = [depth_results[l]["val"]  * 100 for l in layers]
    tests  = [depth_results[l]["test"] * 100 for l in layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, vals,  "o--", color="#4C72B0", linewidth=2,
            markersize=8, label="Validation")
    ax.plot(layers, tests, "s-",  color="#C44E52", linewidth=2,
            markersize=8, label="Test")

    for l, v, t in zip(layers, vals, tests):
        ax.annotate(f"{v:.1f}%", (l, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#4C72B0")
        ax.annotate(f"{t:.1f}%", (l, t), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=9, color="#C44E52")

    ax.set_xlabel("Number of GraphSAGE Layers")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation: Effect of Depth on GraphSAGE")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()
    _save("depth_ablation.png")
    plt.show()


if __name__ == "__main__":
    print("✓ visualization.py loaded — all plot functions ready")
    print(f"  Output directory: {FIGURES_DIR}")