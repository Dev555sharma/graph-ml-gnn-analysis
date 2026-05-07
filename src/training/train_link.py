"""
Training loop for link prediction on ogbl-ddi.
Mini-batch version — runs efficiently on CPU.
Uses edge mini-batches instead of full-graph forward pass.
"""

import os
import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.load_data import load_ddi_pyg
from src.models.link_predictor import build_ddi_model

# ─── Config ──────────────────────────────────────────────────────────────────

CONFIG = {
    "dataset":              "ogbl-ddi",
    "epochs":               200,
    "lr":                   0.01,
    "embedding_dim":        128,        # smaller = faster on CPU
    "hidden_channels":      128,
    "num_sage_layers":      2,
    "dropout":              0.3,
    "batch_size":           8192,       # edges per mini-batch
    "hits_k":               20,
    "early_stop_patience":  30,
    "checkpoint_dir":       "experiments/results",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─── Train (mini-batch) ───────────────────────────────────────────────────────

def train_epoch(encoder, predictor, train_edges, all_edges, num_nodes, optimizer, config):
    encoder.train()
    predictor.train()

    # Full graph edge index for message passing
    edge_index = all_edges.to(DEVICE)

    # Shuffle training edges into mini-batches
    perm = torch.randperm(train_edges.size(0))
    train_edges = train_edges[perm]

    loader = DataLoader(range(train_edges.size(0)),
                        batch_size=config["batch_size"], shuffle=False)

    total_loss = 0
    total_examples = 0

    # Compute all node embeddings once per epoch
    with torch.no_grad():
        h_full = encoder(edge_index)

    for batch_idx in loader:
        optimizer.zero_grad()

        # Recompute embeddings with grad for this batch
        h = encoder(edge_index)

        pos_edge = train_edges[batch_idx].t().to(DEVICE)     # [2, B]

        # Positive scores
        pos_scores = predictor(h[pos_edge[0]], h[pos_edge[1]])

        # Negative sampling
        neg_edge = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1),
        )
        neg_scores = predictor(h[neg_edge[0]], h[neg_edge[1]])

        loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        ) + F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * pos_edge.size(1)
        total_examples += pos_edge.size(1)

    return total_loss / total_examples


# ─── Evaluate ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, predictor, split_edge, evaluator, all_edges, k=20):
    encoder.eval()
    predictor.eval()

    edge_index = all_edges.to(DEVICE)
    h = encoder(edge_index)

    results = {}
    for split in ["valid", "test"]:
        pos_edge = split_edge[split]["edge"].to(DEVICE).t()
        neg_edge = split_edge[split]["edge_neg"].to(DEVICE).t()

        pos_s = predictor(h[pos_edge[0]], h[pos_edge[1]]).squeeze()
        neg_s = predictor(h[neg_edge[0]], h[neg_edge[1]]).squeeze()

        hits = evaluator.eval({
            "y_pred_pos": pos_s,
            "y_pred_neg": neg_s,
        })[f"hits@{k}"]
        results[split] = hits

    return results


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience, enc_path, pred_path):
        self.patience   = patience
        self.enc_path   = enc_path
        self.pred_path  = pred_path
        self.best_val   = 0.0
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_hits, encoder, predictor, epoch):
        if val_hits > self.best_val:
            self.best_val   = val_hits
            self.counter    = 0
            self.best_epoch = epoch
            torch.save(encoder.state_dict(),   self.enc_path)
            torch.save(predictor.state_dict(), self.pred_path)
        else:
            self.counter += 1
        return self.counter >= self.patience


# ─── Main ────────────────────────────────────────────────────────────────────

def main(config=CONFIG):
    data, split_edge, evaluator = load_ddi_pyg()
    data = data.to(DEVICE)

    # All edges for message passing
    all_edges = split_edge["train"]["edge"].t().contiguous()   # [2, E_train]

    # Training edge list [E, 2] for mini-batch sampling
    train_edges = split_edge["train"]["edge"]                  # [E, 2]

    encoder, predictor = build_ddi_model(num_nodes=data.num_nodes)

    # Override dims to match CPU-friendly config
    from src.models.link_predictor import DDIEncoder, LinkPredictor
    encoder = DDIEncoder(
        num_nodes=data.num_nodes,
        embedding_dim=config["embedding_dim"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_sage_layers"],
        dropout=config["dropout"],
    ).to(DEVICE)
    predictor = LinkPredictor(
        in_channels=config["hidden_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=3,
    ).to(DEVICE)

    total_params = (sum(p.numel() for p in encoder.parameters()) +
                    sum(p.numel() for p in predictor.parameters()))
    print(f"Encoder + Predictor | Total params: {total_params:,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config["lr"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    stopper = EarlyStopping(
        patience=config["early_stop_patience"],
        enc_path=os.path.join(config["checkpoint_dir"], "ddi_encoder_best.pt"),
        pred_path=os.path.join(config["checkpoint_dir"], "ddi_predictor_best.pt"),
    )

    print(f"\nTraining for up to {config['epochs']} epochs...\n")
    best_results = {}

    for epoch in tqdm(range(1, config["epochs"] + 1), desc="Training"):
        loss = train_epoch(encoder, predictor, train_edges,
                           all_edges, data.num_nodes, optimizer, config)
        results = evaluate(encoder, predictor, split_edge,
                           evaluator, all_edges, config["hits_k"])
        scheduler.step(results["valid"])

        if results["valid"] > stopper.best_val:
            best_results = results.copy()

        if stopper.step(results["valid"], encoder, predictor, epoch):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val @ epoch {stopper.best_epoch})")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Val Hits@{config['hits_k']}: {results['valid']:.4f} | "
                  f"Test Hits@{config['hits_k']}: {results['test']:.4f}")

    print("\n" + "=" * 55)
    print("Best Results (DDIEncoder on ogbl-ddi)")
    print("=" * 55)
    print(f"  Val  Hits@{config['hits_k']} : {best_results.get('valid', 0):.4f}")
    print(f"  Test Hits@{config['hits_k']} : {best_results.get('test',  0):.4f}")
    print("=" * 55)
    return best_results


if __name__ == "__main__":
    main()