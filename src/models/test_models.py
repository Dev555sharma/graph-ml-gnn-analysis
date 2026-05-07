import sys
sys.path.insert(0, ".")  # tells Python to look from project root

from src.models.gcn import build_gcn
from src.models.gat import build_gat
from src.models.graphsage import build_graphsage
from src.models.link_predictor import build_ddi_model
import torch

x = torch.randn(10, 128)
edge_index = torch.randint(0, 10, (2, 30))

for name, model in [("GCN", build_gcn()), ("GAT", build_gat()), ("SAGE", build_graphsage())]:
    out = model(x, edge_index)
    print(f"{name}: input {x.shape} → output {out.shape}")

enc, pred = build_ddi_model(num_nodes=10)
h = enc(edge_index)
scores = pred(h[edge_index[0]], h[edge_index[1]])
print(f"DDI: node emb {h.shape} → edge scores {scores.shape}")