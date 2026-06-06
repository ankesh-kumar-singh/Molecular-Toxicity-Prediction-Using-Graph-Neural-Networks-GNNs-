#!/usr/bin/env python3
"""
scripts/train.py
─────────────────
Main training entrypoint for the Tox21 GNN classifier.

Usage:
    python scripts/train.py                          # use configs/config.yaml
    python scripts/train.py --model gat --epochs 100
    python scripts/train.py --model gin --hidden_dim 256 --batch_size 128
"""

import argparse
import os
import sys
import json
import yaml
import torch

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader

from data.dataset import MoleculeDataset, ATOM_FEATURE_DIM
from models.gnn import build_model
from utils.train import train
from utils.evaluate import evaluate, print_evaluation_report
from utils.visualize import plot_training_curves, plot_task_aucs, plot_roc_curves

import numpy as np


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GNN on Tox21")
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--config", default=os.path.join(_root, "configs", "config.yaml"))
    p.add_argument("--model", choices=["gcn", "gat", "gin"], help="Override model type")
    p.add_argument("--hidden_dim", type=int, help="Override hidden dim")
    p.add_argument("--num_layers", type=int, help="Override num layers")
    p.add_argument("--epochs", type=int, help="Override epochs")
    p.add_argument("--batch_size", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--dropout", type=float, help="Override dropout")
    p.add_argument("--data_dir", default=os.path.join(_root, "data", "tox21"))
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.model:      cfg["model"]["type"]       = args.model
    if args.hidden_dim: cfg["model"]["hidden_dim"]  = args.hidden_dim
    if args.num_layers: cfg["model"]["num_layers"]  = args.num_layers
    if args.epochs:     cfg["training"]["epochs"]   = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.lr:         cfg["training"]["learning_rate"] = args.lr
    if args.dropout:    cfg["model"]["dropout"]     = args.dropout

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\n🔧  Device   : {device}")
    print(f"🧠  Model    : {cfg['model']['type'].upper()}")
    print(f"📐  Hidden   : {cfg['model']['hidden_dim']}  |  "
          f"Layers: {cfg['model']['num_layers']}  |  "
          f"Dropout: {cfg['model']['dropout']}")

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        print(f"\n⚠️  Data not found at {data_dir}.")
        print("    Run:  python data/download_data.py")
        sys.exit(1)

    print("\n📂  Loading datasets …")
    train_ds = MoleculeDataset(os.path.join(data_dir, "train.csv"))
    val_ds   = MoleculeDataset(os.path.join(data_dir, "val.csv"))
    test_ds  = MoleculeDataset(os.path.join(data_dir, "test.csv"))

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(cfg, in_dim=ATOM_FEATURE_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢  Parameters: {n_params:,}")

    # ── Training ─────────────────────────────────────────────────────────
    ckpt_dir = cfg["logging"]["checkpoint_dir"]
    model_name = cfg["model"]["type"].upper()
    ckpt_path  = os.path.join(ckpt_dir, f"best_{model_name}.pt")

    history = train(model, train_loader, val_loader, cfg, device, ckpt_path)

    # ── Test Evaluation ───────────────────────────────────────────────────
    print("\n🧪  Loading best checkpoint for test evaluation …")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print_evaluation_report(test_metrics, split="Test")

    # ── Save results ─────────────────────────────────────────────────────
    plot_dir = cfg["logging"]["plot_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    plot_training_curves(
        history,
        save_path=os.path.join(plot_dir, f"training_curves_{model_name}.png"),
    )
    plot_task_aucs(
        test_metrics,
        model_name=model_name,
        save_path=os.path.join(plot_dir, f"task_aucs_{model_name}.png"),
    )

    # Save JSON results
    results_path = os.path.join("results", f"test_results_{model_name}.json")
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        # Convert float values for JSON serialization
        serializable = {
            "model": model_name,
            "mean_auc": float(test_metrics["mean_auc"]),
            "mean_ap":  float(test_metrics["mean_ap"]),
            "loss":     float(test_metrics["loss"]),
            "per_task": {
                t: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                    for k, v in m.items()}
                for t, m in test_metrics["per_task"].items()
            },
        }
        json.dump(serializable, f, indent=2)
    print(f"  💾  Results → {results_path}")


if __name__ == "__main__":
    main()
