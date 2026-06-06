"""
models/layers.py
────────────────
Custom layers used by GNN models:
  - MLPLayer        — multi-layer perceptron block
  - AttentionReadout — global attention pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class MLPLayer(nn.Module):
    """
    A standard MLP block:
        Linear → BatchNorm → ReLU → Dropout  (repeated `num_layers` times)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        assert num_layers >= 1

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}[activation]

        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no BN/act/drop after last linear
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionReadout(nn.Module):
    """
    Global attention pooling:
        gate = sigmoid(W_gate · h)
        out  = Σ_i gate_i * (W_feat · h_i)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.gate_nn = nn.Linear(in_dim, 1)
        self.feat_nn = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_nn(x))      # (N, 1)
        feat = self.feat_nn(x)                     # (N, out_dim)
        weighted = gate * feat                     # (N, out_dim)

        # Sum per graph
        from torch_scatter import scatter_add
        num_graphs = int(batch.max().item()) + 1
        out = scatter_add(weighted, batch, dim=0, dim_size=num_graphs)
        return out                                 # (B, out_dim)


def get_readout(readout: str, hidden_dim: int):
    """Return (readout_fn, out_dim) pair."""
    if readout == "mean":
        return global_mean_pool, hidden_dim
    elif readout == "sum":
        return global_add_pool, hidden_dim
    elif readout == "max":
        return global_max_pool, hidden_dim
    elif readout == "attention":
        # AttentionReadout is a module — caller must handle differently
        return None, hidden_dim
    else:
        raise ValueError(f"Unknown readout: {readout}")
