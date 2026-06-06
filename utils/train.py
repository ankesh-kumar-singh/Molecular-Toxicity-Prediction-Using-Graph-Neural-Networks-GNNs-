"""
utils/train.py
──────────────
Training loop + early stopping for multi-task binary classification.

Key design choices:
  • BCEWithLogitsLoss with pos_weight for class imbalance
  • Masking of missing labels (label == -1)
  • Early stopping on validation mean ROC-AUC
  • Cosine annealing LR scheduler
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.loader import DataLoader

from utils.evaluate import evaluate


# ─── Masked loss ─────────────────────────────────────────────────────────────

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCEWithLogitsLoss that ignores missing labels (encoded as -1).

    Args:
        logits   : (B, T) raw model outputs
        targets  : (B, T) binary labels, -1 = missing
        pos_weight: (T,) per-task positive class weights

    Returns:
        Scalar mean loss over valid (non-masked) entries.
    """
    mask = targets != -1                        # (B, T) bool

    # Clamp targets so BCE sees only 0/1 (masked positions don't matter)
    clamped = targets.clamp(min=0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction="none"
    )
    loss_all = criterion(logits, clamped)       # (B, T)
    loss_masked = loss_all * mask.float()       # zero out missing

    n_valid = mask.float().sum()
    if n_valid == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return loss_masked.sum() / n_valid


# ─── One epoch ───────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.squeeze(1)            # (B, 12)

        optimizer.zero_grad()
        logits = model(batch)                   # (B, 12)
        loss = masked_bce_loss(logits, targets, pos_weight)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_loss / max(total_graphs, 1)


# ─── Full training run ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0
        self.stopped = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


def build_pos_weight(train_loader: DataLoader, num_tasks: int,
                     base_weight: float = 10.0) -> torch.Tensor:
    """
    Compute per-task positive class weight from training data.
    Falls back to `base_weight` for tasks with no positives.
    """
    pos = torch.zeros(num_tasks)
    neg = torch.zeros(num_tasks)
    for batch in train_loader:
        y = batch.y.squeeze(1)
        mask = y != -1
        pos += ((y == 1) & mask).float().sum(dim=0)
        neg += ((y == 0) & mask).float().sum(dim=0)

    # Avoid division by zero
    weights = torch.where(pos > 0, neg / pos.clamp(min=1), torch.full_like(pos, base_weight))
    return weights.clamp(max=50.0)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: str = "./results/checkpoints/best.pt",
) -> Dict[str, list]:
    """
    Full training loop.

    Returns a history dict with keys:
        train_loss, val_loss, val_auc, lr
    """
    import os; os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    epochs    = cfg["training"]["epochs"]
    lr        = cfg["training"]["learning_rate"]
    wd        = cfg["training"]["weight_decay"]
    patience  = cfg["training"]["patience"]
    base_pw   = cfg["training"]["pos_weight"]
    scheduler_type = cfg["training"].get("scheduler", "cosine")
    log_every = cfg["logging"].get("log_every", 5)
    num_tasks = cfg["model"]["num_tasks"]

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = None

    pos_weight = build_pos_weight(train_loader, num_tasks, base_pw).to(device)
    stopper = EarlyStopping(patience=patience)

    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_auc": [], "lr": []
    }

    best_auc = -1.0
    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  "
          f"{'Val AUC':>8}  {'LR':>10}  {'Time':>6}")
    print("─" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device, pos_weight)

        val_loss = val_metrics["loss"]
        val_auc  = val_metrics["mean_auc"]

        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Checkpoint best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "best_auc": best_auc}, checkpoint_path)

        elapsed = time.time() - t0
        if epoch % log_every == 0 or epoch == 1:
            print(f"{epoch:>6}  {tr_loss:>11.4f}  {val_loss:>9.4f}  "
                  f"{val_auc:>8.4f}  {optimizer.param_groups[0]['lr']:>10.6f}  "
                  f"{elapsed:>5.1f}s")

        if stopper.step(val_auc):
            print(f"\n⏹  Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
            break

    print(f"\n✅  Training complete. Best Val AUC = {best_auc:.4f}")
    print(f"    Checkpoint saved → {checkpoint_path}")
    return history
