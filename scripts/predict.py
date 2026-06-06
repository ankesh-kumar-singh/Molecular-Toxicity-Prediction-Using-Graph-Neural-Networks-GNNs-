#!/usr/bin/env python3
"""
scripts/predict.py
──────────────────
Predict toxicity for one or more molecules given as SMILES strings.

Usage:
    # Single molecule (Aspirin)
    python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

    # Multiple molecules from a file (one SMILES per line)
    python scripts/predict.py --smiles_file molecules.txt

    # Specify a different checkpoint
    python scripts/predict.py --smiles "CCO" --checkpoint results/checkpoints/best_GIN.pt
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import smiles_to_graph, ATOM_FEATURE_DIM
from models.gnn import build_model
from utils.visualize import draw_molecule

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def predict_smiles(
    smiles_list: list[str],
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict]:
    """
    Run inference on a list of SMILES.

    Returns a list of dicts:
        { smiles, valid, predictions: {task: prob}, toxic_tasks: [...] }
    """
    from torch_geometric.data import Batch

    model.eval()
    results = []

    valid_smiles, valid_graphs, valid_idx = [], [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is None:
            results.append({"smiles": smi, "valid": False,
                            "predictions": {}, "toxic_tasks": []})
        else:
            valid_smiles.append(smi)
            valid_graphs.append(g)
            valid_idx.append(i)
            results.append(None)        # placeholder

    if not valid_graphs:
        return results

    batch = Batch.from_data_list(valid_graphs).to(device)

    with torch.no_grad():
        logits = model(batch)           # (N, 12)
        probs  = torch.sigmoid(logits).cpu().numpy()

    for j, (smi, p) in enumerate(zip(valid_smiles, probs)):
        pred_dict = {task: float(p[k]) for k, task in enumerate(TOX21_TASKS)}
        toxic = [t for t, v in pred_dict.items() if v >= 0.5]
        results[valid_idx[j]] = {
            "smiles":      smi,
            "valid":       True,
            "predictions": pred_dict,
            "toxic_tasks": toxic,
        }

    return results


def print_predictions(result: dict) -> None:
    smi = result["smiles"]
    print(f"\n  Molecule : {smi}")
    if not result["valid"]:
        print("  ❌  Invalid SMILES — could not parse.")
        return

    print(f"  {'Task':<20} {'Prob':>6}  {'Label':>8}")
    print(f"  {'-'*38}")
    for task, prob in result["predictions"].items():
        label = "🔴 TOXIC" if prob >= 0.5 else "🟢 SAFE"
        bar = "█" * int(prob * 20)
        print(f"  {task:<20} {prob:>5.3f}  {label:<8}")

    if result["toxic_tasks"]:
        print(f"\n  ⚠️  Predicted toxic in: {', '.join(result['toxic_tasks'])}")
    else:
        print(f"\n  ✅  Predicted non-toxic across all assays.")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles",      type=str, help="Single SMILES string")
    g.add_argument("--smiles_file", type=str, help="File with one SMILES per line")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--config",     default=os.path.join(_root, "configs", "config.yaml"))
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    p.add_argument("--model",      default=None, choices=["gcn", "gat", "gin"])
    p.add_argument("--visualize",  action="store_true", help="Draw molecule plot")
    p.add_argument("--no_cuda",    action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"]["type"] = args.model

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    # Auto-detect checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        model_name = cfg["model"]["type"].upper()
        ckpt_path  = os.path.join(_root, "results", "checkpoints", f"best_{model_name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"❌  Checkpoint not found: {ckpt_path}")
        print("    Train first: python scripts/train.py")
        sys.exit(1)

    model = build_model(cfg, in_dim=ATOM_FEATURE_DIM).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"✅  Loaded checkpoint: {ckpt_path}  (best AUC={ckpt.get('best_auc', '?'):.4f})")

    # Gather SMILES
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        with open(args.smiles_file) as f:
            smiles_list = [l.strip() for l in f if l.strip()]

    print(f"\n🔬  Predicting for {len(smiles_list)} molecule(s) …")
    results = predict_smiles(smiles_list, model, device)

    print("\n" + "=" * 50)
    for result in results:
        print_predictions(result)

    if args.visualize and len(results) == 1 and results[0]["valid"]:
        draw_molecule(
            smiles_list[0],
            predictions=results[0]["predictions"],
            task_names=TOX21_TASKS,
            save_path="results/plots/molecule_prediction.png",
        )


if __name__ == "__main__":
    main()
