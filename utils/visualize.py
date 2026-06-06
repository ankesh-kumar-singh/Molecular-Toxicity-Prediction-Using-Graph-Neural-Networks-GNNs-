"""
utils/visualize.py
──────────────────
Plotting functions:
  • plot_training_curves  — loss & AUC over epochs
  • plot_task_aucs        — bar chart of per-task AUC
  • plot_roc_curves       — ROC curve overlay for all tasks
  • draw_molecule         — render a molecule from SMILES with prediction
  • plot_attention_weights— (GAT only) visualize attention on molecule graph
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Try RDKit drawing (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ─── Style ───────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#4C72B0", "#DD8452",
]


def _savefig(fig, path: Optional[str], tight: bool = True):
    if tight:
        fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  📊  Saved → {path}")
    plt.show()
    plt.close(fig)


# ─── Training curves ─────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, list],
    save_path: Optional[str] = None,
) -> None:
    """Plot train/val loss and val ROC-AUC over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0")
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    # AUC
    ax = axes[1]
    ax.plot(epochs, history["val_auc"], label="Val ROC-AUC", color="#55A868")
    ax.set_xlabel("Epoch"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Validation ROC-AUC"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    _savefig(fig, save_path)


# ─── Per-task AUC bar chart ───────────────────────────────────────────────────

def plot_task_aucs(
    metrics: Dict,
    model_name: str = "GNN",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of per-task ROC-AUC."""
    tasks = list(metrics["per_task"].keys())
    aucs  = [metrics["per_task"][t]["auc"] for t in tasks]
    mean_auc = metrics["mean_auc"]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0" if not np.isnan(a) else "#cccccc" for a in aucs]
    aucs_plot = [a if not np.isnan(a) else 0 for a in aucs]

    bars = ax.barh(tasks[::-1], aucs_plot[::-1], color=colors[::-1], edgecolor="white")

    ax.axvline(mean_auc, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"Mean AUC = {mean_auc:.3f}")
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, label="Random = 0.5")

    # Value labels
    for bar, auc in zip(bars, aucs_plot[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", fontsize=9)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"{model_name} — Per-Task ROC-AUC on Tox21", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    _savefig(fig, save_path)


# ─── ROC curves ──────────────────────────────────────────────────────────────

def plot_roc_curves(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    task_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Overlay ROC curves for all tasks."""
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")

    for i, (task, color) in enumerate(zip(task_names, PALETTE)):
        mask = all_targets[:, i] != -1
        y_true = all_targets[mask, i]
        y_prob = all_probs[mask, i]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f"{task} ({auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Tox21 Tasks", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    _savefig(fig, save_path)


# ─── Molecule drawing ─────────────────────────────────────────────────────────

def draw_molecule(
    smiles: str,
    predictions: Optional[Dict[str, float]] = None,
    task_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Draw a molecule with optional per-task toxicity predictions.
    Requires RDKit.
    """
    if not HAS_RDKIT:
        print("⚠️  RDKit not available — skipping molecule drawing.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️  Could not parse SMILES: {smiles}")
        return

    AllChem.Compute2DCoords(mol)

    fig = plt.figure(figsize=(14, 5))

    # Left: molecule image
    ax_mol = fig.add_axes([0.0, 0.0, 0.4, 1.0])
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    from io import StringIO
    import xml.etree.ElementTree as ET
    # Fallback: use PIL if available
    try:
        from PIL import Image
        from rdkit.Chem.Draw import MolToImage
        img = MolToImage(mol, size=(400, 400))
        ax_mol.imshow(img)
    except Exception:
        ax_mol.text(0.5, 0.5, smiles, ha="center", va="center",
                    wrap=True, fontsize=9)
    ax_mol.axis("off")
    ax_mol.set_title(f"SMILES: {smiles[:40]}…" if len(smiles) > 40 else smiles,
                     fontsize=9)

    # Right: bar chart of predictions
    if predictions and task_names:
        ax_bar = fig.add_axes([0.42, 0.05, 0.55, 0.85])
        probs = [predictions.get(t, 0.0) for t in task_names]
        colors = ["#C44E52" if p >= 0.5 else "#55A868" for p in probs]
        y_pos = range(len(task_names))

        ax_bar.barh(list(y_pos), probs, color=colors, edgecolor="white")
        ax_bar.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax_bar.set_yticks(list(y_pos))
        ax_bar.set_yticklabels(task_names, fontsize=9)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Toxicity Probability")
        ax_bar.set_title("Predicted Toxicity per Assay", fontsize=11, fontweight="bold")
        ax_bar.grid(axis="x", alpha=0.3)

        toxic_patch = mpatches.Patch(color="#C44E52", label="Toxic (≥0.5)")
        safe_patch  = mpatches.Patch(color="#55A868", label="Non-toxic (<0.5)")
        ax_bar.legend(handles=[toxic_patch, safe_patch], fontsize=8)

    fig.suptitle("Molecule Toxicity Prediction", fontsize=13, fontweight="bold", y=1.02)
    _savefig(fig, save_path, tight=False)
