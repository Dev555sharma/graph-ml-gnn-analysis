"""
main.py — CLI entry point for graph-ml-gnn-analysis.
Run any part of the pipeline from one place.

Usage:
    python main.py --task train_node --model graphsage
    python main.py --task train_node --model gcn
    python main.py --task train_node --model gat
    python main.py --task train_link
    python main.py --task evaluate
    python main.py --task data_check
"""

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph ML — GNN Node Classification & Link Prediction"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["train_node", "train_link", "evaluate", "data_check"],
        help="Task to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graphsage",
        choices=["gcn", "gat", "graphsage"],
        help="Model for node classification (default: graphsage)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override default learning rate"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=None,
        help="Override hidden channel size"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  Graph ML Pipeline")
    print(f"  Task  : {args.task}")
    if args.task == "train_node":
        print(f"  Model : {args.model.upper()}")
    print(f"{'='*55}\n")

    if args.task == "data_check":
        from src.data.load_data import load_arxiv_pyg, load_ddi_pyg
        print("Checking ogbn-arxiv...")
        data, split_idx, _ = load_arxiv_pyg()
        print(f"  Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | "
              f"Features: {data.num_node_features} | Classes: 40")
        print(f"  Train: {len(split_idx['train']):,} | "
              f"Valid: {len(split_idx['valid']):,} | "
              f"Test: {len(split_idx['test']):,}")
        print("\nChecking ogbl-ddi...")
        data, split_edge, _ = load_ddi_pyg()
        print(f"  Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,}")
        print(f"  Train edges: {len(split_edge['train']['edge']):,}")
        print("\n✓ Both datasets OK")

    elif args.task == "train_node":
        from src.training.train_node import main as train_node_main, CONFIG
        config = CONFIG.copy()
        config["model"] = args.model
        if args.epochs: config["epochs"] = args.epochs
        if args.lr:     config["lr"] = args.lr
        if args.hidden: config["hidden_channels"] = args.hidden
        if args.wandb:  config["use_wandb"] = True
        train_node_main(config)

    elif args.task == "train_link":
        from src.training.train_link import main as train_link_main, CONFIG
        config = CONFIG.copy()
        if args.epochs: config["epochs"] = args.epochs
        if args.lr:     config["lr"] = args.lr
        if args.wandb:  config["use_wandb"] = True
        train_link_main(config)

    elif args.task == "evaluate":
        from src.training.evaluate import eval_node_model, eval_link_model
        eval_node_model(model_name="graphsage")
        eval_link_model()


if __name__ == "__main__":
    main()