"""
Centralized configuration for all models and training runs.
Import CONFIG_NODE or CONFIG_LINK instead of redefining dicts per script.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeConfig:
    """Configuration for node classification (ogbn-arxiv)."""
    model:                str   = "graphsage"   # gcn | gat | graphsage
    dataset:              str   = "ogbn-arxiv"
    epochs:               int   = 500
    lr:                   float = 0.01
    weight_decay:         float = 0.0
    dropout:              float = 0.5
    hidden_channels:      int   = 256
    num_layers:           int   = 3
    early_stop_patience:  int   = 30
    checkpoint_dir:       str   = "experiments/results"
    use_wandb:            bool  = False
    data_root:            str   = "data/raw"


@dataclass
class LinkConfig:
    """Configuration for link prediction (ogbl-ddi)."""
    dataset:              str   = "ogbl-ddi"
    epochs:               int   = 200
    lr:                   float = 0.01
    embedding_dim:        int   = 128
    hidden_channels:      int   = 128
    num_sage_layers:      int   = 2
    dropout:              float = 0.3
    batch_size:           int   = 8192
    hits_k:               int   = 20
    early_stop_patience:  int   = 30
    checkpoint_dir:       str   = "experiments/results"
    use_wandb:            bool  = False
    data_root:            str   = "data/raw"


# Default instances — import these directly
CONFIG_NODE = NodeConfig()
CONFIG_LINK = LinkConfig()


def config_to_dict(cfg) -> dict:
    """Convert dataclass config to plain dict (for W&B logging)."""
    import dataclasses
    return dataclasses.asdict(cfg)


def print_config(cfg):
    """Pretty-print a config dataclass."""
    name = type(cfg).__name__
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    for k, v in config_to_dict(cfg).items():
        print(f"  {k:<25} {v}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    print_config(CONFIG_NODE)
    print_config(CONFIG_LINK)