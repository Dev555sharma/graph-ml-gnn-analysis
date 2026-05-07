"""
Training loop for node classification on ogbn-arxiv.
Supports GCN, GAT, GraphSAGE interchangeably.

Features:
    - OGB accuracy evaluator
    - Early stopping
    - LR scheduling (ReduceLROnPlateau)
    - Weights & Biases logging
    - Model checkpointing
"""

import os
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb

from src.data.load_data import load_arxiv_pyg
from src.models.gcn import build_gcn
from src.models.gat import build_gat
from src.models.graphsage import build_graphsage


# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "model":          "graphsage",   # "gcn" | "gat" | "graphsage"
    "dataset":        "ogbn-arxiv",
    "epochs":         500,
    "lr":             0.01,
    "weight_decay":   0.0,
    "dropout":        0.5,
    "hidden_channels": 256,
    "num_layers":     3,
    "early_stop_patience": 30,
    "checkpoint_dir": "experiments/results",
    "use_wandb":      False,          # set True when you have a W&B account
}

# ─── Device ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─── Model factory ───────────────────────────────────────────────────────────

def get_model(name: str, num_classes: int = 40):
    name = name.lower()
    if name == "gcn":
        return build_gcn(num_classes)
    elif name == "gat":
        return build_gat(num_classes)
    elif name == "graphsage":
        return build_graphsage(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose gcn | gat | graphsage")


# ─── Train / Eval ────────────────────────────────────────────────────────────

def train(model, data, optimizer, split_idx):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)  # [N, 40]
    train_idx = split_idx["train"]

    loss = F.cross_entropy(out[train_idx], data.y[train_idx].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, split_idx, evaluator):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1, keepdim=True)  # [N, 1]

    results = {}
    for split in ["train", "valid", "test"]:
        idx = split_idx[split]
        acc = evaluator.eval({
            "y_true": data.y[idx],
            "y_pred": pred[idx],
        })["acc"]
        results[split] = acc
    return results


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 30, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_val = 0.0
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_acc: float, model, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_acc > self.best_val:
            self.best_val = val_acc
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── Main ────────────────────────────────────────────────────────────────────

def main(config: dict = CONFIG):
    # Init W&B
    if config["use_wandb"]:
        wandb.init(project="gnn-node-classification", config=config)

    # Data
    data, split_idx, evaluator = load_arxiv_pyg()
    data = data.to(DEVICE)
    split_idx = {k: v.to(DEVICE) for k, v in split_idx.items()}

    # Model
    model = get_model(config["model"]).to(DEVICE)
    print(f"\nModel: {config['model'].upper()} | "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=10, min_lr=1e-5
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(
        config["checkpoint_dir"], f"{config['model']}_best.pt"
    )
    stopper = EarlyStopping(
        patience=config["early_stop_patience"],
        checkpoint_path=checkpoint_path
    )

    # Training loop
    print(f"\nTraining for up to {config['epochs']} epochs...\n")
    best_results = {}

    for epoch in tqdm(range(1, config["epochs"] + 1), desc="Training"):
        loss = train(model, data, optimizer, split_idx)
        results = evaluate(model, data, split_idx, evaluator)
        scheduler.step(results["valid"])

        if config["use_wandb"]:
            wandb.log({
                "loss": loss,
                "train_acc": results["train"],
                "val_acc":   results["valid"],
                "test_acc":  results["test"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        if results["valid"] > stopper.best_val:
            best_results = results.copy()

        if stopper.step(results["valid"], model, epoch):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val @ epoch {stopper.best_epoch})")
            break

        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {results['train']:.4f} | "
                  f"Val: {results['valid']:.4f} | "
                  f"Test: {results['test']:.4f}")

    # Final results
    print("\n" + "=" * 55)
    print(f"Best Results ({config['model'].upper()} on ogbn-arxiv)")
    print("=" * 55)
    print(f"  Train Acc : {best_results.get('train', 0):.4f}")
    print(f"  Valid Acc : {best_results.get('valid', 0):.4f}")
    print(f"  Test  Acc : {best_results.get('test',  0):.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 55)

    if config["use_wandb"]:
        wandb.finish()

    return best_results


if __name__ == "__main__":
    main()