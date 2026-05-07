"""
Standalone evaluation — loads a saved checkpoint and reports final metrics.
Use this after training to get clean numbers for your report/README.
"""

import sys
sys.path.insert(0, ".")

import torch
from src.data.load_data import load_arxiv_pyg, load_ddi_pyg
from src.models.gcn import build_gcn
from src.models.gat import build_gat
from src.models.graphsage import build_graphsage
from src.models.link_predictor import build_ddi_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_node_model(model_name: str = "graphsage",
                    checkpoint: str = "experiments/results/graphsage_best.pt"):
    data, split_idx, evaluator = load_arxiv_pyg()
    data = data.to(DEVICE)

    model_map = {"gcn": build_gcn, "gat": build_gat, "graphsage": build_graphsage}
    model = model_map[model_name]().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1, keepdim=True)

    print(f"\n{'='*45}")
    print(f"Final Evaluation: {model_name.upper()} on ogbn-arxiv")
    print(f"{'='*45}")
    for split in ["train", "valid", "test"]:
        idx = split_idx[split].to(DEVICE)
        acc = evaluator.eval({"y_true": data.y[idx], "y_pred": pred[idx]})["acc"]
        print(f"  {split.capitalize():6s} Accuracy: {acc:.4f} ({acc*100:.2f}%)")


def eval_link_model(
    enc_path: str = "experiments/results/ddi_encoder_best.pt",
    pred_path: str = "experiments/results/ddi_predictor_best.pt",
    hits_k: int = 20,
):
    data, split_edge, evaluator = load_ddi_pyg()
    data = data.to(DEVICE)

    encoder, predictor = build_ddi_model(num_nodes=data.num_nodes)
    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    predictor.load_state_dict(torch.load(pred_path, map_location=DEVICE))
    encoder, predictor = encoder.to(DEVICE), predictor.to(DEVICE)
    encoder.eval()
    predictor.eval()

    train_edge = split_edge["train"]["edge"].t().contiguous().to(DEVICE)
    with torch.no_grad():
        h = encoder(train_edge)

    print(f"\n{'='*45}")
    print(f"Final Evaluation: DDI Link Predictor on ogbl-ddi")
    print(f"{'='*45}")
    for split in ["valid", "test"]:
        pos_edge = split_edge[split]["edge"].to(DEVICE).t()
        neg_edge = split_edge[split]["edge_neg"].to(DEVICE).t()
        with torch.no_grad():
            pos_s = predictor(h[pos_edge[0]], h[pos_edge[1]]).squeeze()
            neg_s = predictor(h[neg_edge[0]], h[neg_edge[1]]).squeeze()
        hits = evaluator.eval({"y_pred_pos": pos_s, "y_pred_neg": neg_s})[f"hits@{hits_k}"]
        print(f"  {split.capitalize():6s} Hits@{hits_k}: {hits:.4f} ({hits*100:.2f}%)")


if __name__ == "__main__":
    eval_node_model("graphsage")
    eval_link_model()