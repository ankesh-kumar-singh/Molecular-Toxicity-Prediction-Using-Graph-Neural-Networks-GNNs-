#!/usr/bin/env python3
"""
scripts/hyperparameter_search.py
─────────────────────────────────
Optuna-based hyperparameter optimization for the Tox21 GNN.

Searches over:
  - model type (gcn / gat / gin)
  - hidden_dim
  - num_layers
  - dropout
  - learning_rate
  - batch_size
  - readout

Usage:
    python scripts/hyperparameter_search.py --n_trials 50
    python scripts/hyperparameter_search.py --n_trials 20 --model gin
"""

import argparse
import os
import sys
import yaml
import copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader
from data.dataset import MoleculeDataset, ATOM_FEATURE_DIM
from models.gnn import build_model
from utils.train import train
from utils.evaluate import evaluate

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, base_cfg: dict, data_dir: str,
              device: torch.device, model_filter: str = None) -> float:
    cfg = copy.deepcopy(base_cfg)

    # ── Search space ──────────────────────────────────────────────────────
    model_type = model_filter or trial.suggest_categorical("model", ["gcn", "gat", "gin"])
    cfg["model"]["type"]       = model_type
    cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    cfg["model"]["num_layers"] = trial.suggest_int("num_layers", 2, 6)
    cfg["model"]["dropout"]    = trial.suggest_float("dropout", 0.1, 0.5)
    cfg["model"]["readout"]    = trial.suggest_categorical("readout", ["mean", "sum", "max"])

    cfg["training"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg["training"]["batch_size"]    = trial.suggest_categorical("batch_size", [32, 64, 128])
    cfg["training"]["epochs"]        = 50   # short runs for HPO
    cfg["training"]["patience"]      = 10

    if model_type == "gat":
        cfg["gat"]["heads"]  = trial.suggest_categorical("heads", [2, 4, 8])
        cfg["gat"]["concat"] = trial.suggest_categorical("concat", [True, False])

    if model_type == "gin":
        cfg["gin"]["mlp_layers"] = trial.suggest_int("mlp_layers", 1, 3)

    # ── Data ─────────────────────────────────────────────────────────────
    bs = cfg["training"]["batch_size"]
    train_ds = MoleculeDataset(os.path.join(data_dir, "train.csv"))
    val_ds   = MoleculeDataset(os.path.join(data_dir, "val.csv"))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, in_dim=ATOM_FEATURE_DIM).to(device)

    ckpt_path = f"/tmp/hpo_trial_{trial.number}.pt"
    try:
        train(model, train_loader, val_loader, cfg, device, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        val_metrics = evaluate(model, val_loader, device)
        return val_metrics["mean_auc"]
    except Exception as e:
        print(f"  ⚠️  Trial {trial.number} failed: {e}")
        return 0.0
    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials",  type=int, default=30)
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--config",    default=os.path.join(_root, "configs", "config.yaml"))
    p.add_argument("--data_dir",  default=os.path.join(_root, "data", "tox21"))
    p.add_argument("--model",     default=None, choices=["gcn", "gat", "gin"])
    p.add_argument("--no_cuda",   action="store_true")
    p.add_argument("--study_name", default="tox21_gnn_hpo")
    args = p.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"\n🔍  Hyperparameter search: {args.n_trials} trials on {device}")
    if args.model:
        print(f"    Model locked to: {args.model.upper()}")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg, args.data_dir, device, args.model),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n🏆  Best Trial #{best.number}")
    print(f"    Val AUC : {best.value:.4f}")
    print(f"    Params  :")
    for k, v in best.params.items():
        print(f"      {k:<20}: {v}")

    # Save best config
    out_path = "results/best_hyperparams.yaml"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump({"best_auc": best.value, "params": best.params}, f)
    print(f"\n  💾  Saved → {out_path}")
    print(f"\n  Re-train with best params:")
    for k, v in best.params.items():
        print(f"    --{k}={v}", end="")
    print()


if __name__ == "__main__":
    main()
