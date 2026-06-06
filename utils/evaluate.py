"""
utils/evaluate.py
─────────────────
Evaluation metrics for multi-task binary molecular classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def masked_bce_loss(logits, targets, pos_weight=None):
    """Inline — no import from utils.train to avoid circular import."""
    mask    = targets != -1
    clamped = targets.clamp(min=0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none"
    )
    loss_all    = criterion(logits, clamped)
    loss_masked = loss_all * mask.float()
    n_valid     = mask.float().sum()
    if n_valid == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return loss_masked.sum() / n_valid


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor = None,
    task_names: list = TOX21_TASKS,
) -> Dict:
    model.eval()

    all_logits  = []
    all_targets = []
    total_loss  = 0.0
    total_graphs = 0

    for batch in loader:
        batch   = batch.to(device)
        targets = batch.y.squeeze(1)
        logits  = model(batch)

        loss = masked_bce_loss(logits, targets, pos_weight)
        total_loss   += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits  = torch.cat(all_logits,  dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs   = 1 / (1 + np.exp(-all_logits))

    mean_loss = total_loss / max(total_graphs, 1)

    per_task: Dict[str, Dict[str, float]] = {}
    aucs, aps = [], []

    for i, task in enumerate(task_names):
        mask   = all_targets[:, i] != -1
        y_true = all_targets[mask, i]
        y_prob = all_probs[mask, i]
        y_pred = (y_prob >= 0.5).astype(int)

        task_metrics: Dict[str, float] = {}

        if len(np.unique(y_true)) < 2:
            task_metrics["auc"] = float("nan")
            task_metrics["ap"]  = float("nan")
        else:
            auc = roc_auc_score(y_true, y_prob)
            ap  = average_precision_score(y_true, y_prob)
            task_metrics["auc"] = auc
            task_metrics["ap"]  = ap
            aucs.append(auc)
            aps.append(ap)

        task_metrics["acc"]   = accuracy_score(y_true, y_pred)
        task_metrics["f1"]    = f1_score(y_true, y_pred, zero_division=0)
        task_metrics["n_pos"] = int(y_true.sum())
        task_metrics["n_neg"] = int((1 - y_true).sum())

        per_task[task] = task_metrics

    mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
    mean_ap  = float(np.nanmean(aps))  if aps  else float("nan")

    return {
        "loss":     mean_loss,
        "mean_auc": mean_auc,
        "mean_ap":  mean_ap,
        "per_task": per_task,
    }


def print_evaluation_report(metrics: Dict, split: str = "Test") -> None:
    print(f"\n{'─'*65}")
    print(f"  {split} Evaluation Report")
    print(f"{'─'*65}")
    print(f"  Mean Loss    : {metrics['loss']:.4f}")
    print(f"  Mean ROC-AUC : {metrics['mean_auc']:.4f}")
    print(f"  Mean PR-AUC  : {metrics['mean_ap']:.4f}")
    print(f"\n  {'Task':<20} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'Acc':>7} {'F1':>7} {'Pos':>6} {'Neg':>6}")
    print(f"  {'-'*62}")
    for task, m in metrics["per_task"].items():
        auc_s = f"{m['auc']:.4f}" if not np.isnan(m["auc"]) else "  N/A  "
        ap_s  = f"{m['ap']:.4f}"  if not np.isnan(m["ap"])  else "  N/A  "
        print(
            f"  {task:<20} {auc_s:>8} {ap_s:>8} "
            f"{m['acc']:>7.4f} {m['f1']:>7.4f} "
            f"{m['n_pos']:>6} {m['n_neg']:>6}"
        )
    print(f"{'─'*65}\n")