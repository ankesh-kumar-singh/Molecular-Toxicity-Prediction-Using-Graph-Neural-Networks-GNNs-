"""
models/gnn.py
─────────────
Three GNN architectures for multi-task molecular classification:

  GCN  — Graph Convolutional Network  (Kipf & Welling, 2017)
  GAT  — Graph Attention Network      (Veličković et al., 2018)
  GIN  — Graph Isomorphism Network    (Xu et al., 2019)

All models:
  • Accept pre-computed node (and optionally edge) features
  • Use BatchNorm + Dropout between layers
  • Apply global graph readout → MLP classifier head
  • Output logits of shape (B, num_tasks); sigmoid applied outside
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)

from models.layers import MLPLayer, AttentionReadout


# ─── Shared readout helper ────────────────────────────────────────────────────

def build_readout(readout: str, hidden_dim: int):
    """Returns (pooling_callable_or_module, bool is_attention_module)."""
    if readout == "mean":
        return global_mean_pool, False
    elif readout == "sum":
        return global_add_pool, False
    elif readout == "max":
        return global_max_pool, False
    elif readout == "attention":
        return AttentionReadout(hidden_dim, hidden_dim), True
    else:
        raise ValueError(f"Unknown readout: {readout!r}")


# ─── GCN ─────────────────────────────────────────────────────────────────────

class GCNModel(nn.Module):
    """
    GCN: stacked GCNConv layers with residual connections.

    Args:
        in_dim      : atom feature dimension
        hidden_dim  : hidden channels
        num_layers  : number of GCN layers
        num_tasks   : number of output binary tasks
        dropout     : dropout probability
        readout     : 'mean' | 'sum' | 'max' | 'attention'
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.3,
        readout: str = "mean",
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Readout
        pool, self._is_att = build_readout(readout, hidden_dim)
        if self._is_att:
            self.pool = pool
        else:
            self._pool_fn = pool

        # Classifier head
        self.classifier = MLPLayer(hidden_dim, hidden_dim // 2, num_tasks,
                                   num_layers=2, dropout=dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.input_proj(x))

        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h          # residual

        # Graph-level readout
        if self._is_att:
            g = self.pool(x, batch)
        else:
            g = self._pool_fn(x, batch)

        return self.classifier(g)   # (B, num_tasks)


# ─── GAT ─────────────────────────────────────────────────────────────────────

class GATModel(nn.Module):
    """
    GAT: multi-head attention convolutions.

    Args:
        heads   : number of attention heads
        concat  : if True, concatenate head outputs; else average
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.3,
        readout: str = "mean",
        heads: int = 4,
        concat: bool = True,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim * heads if (concat and i > 0) else hidden_dim
            out_ch = hidden_dim
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads, concat=concat,
                        dropout=dropout)
            )
            bn_dim = out_ch * heads if concat else out_ch
            self.bns.append(nn.BatchNorm1d(bn_dim))

        final_dim = hidden_dim * heads if concat else hidden_dim
        pool, self._is_att = build_readout(readout, final_dim)
        if self._is_att:
            self.pool = pool
        else:
            self._pool_fn = pool

        self.classifier = MLPLayer(final_dim, hidden_dim, num_tasks,
                                   num_layers=2, dropout=dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.input_proj(x))

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self._is_att:
            g = self.pool(x, batch)
        else:
            g = self._pool_fn(x, batch)

        return self.classifier(g)


# ─── GIN ─────────────────────────────────────────────────────────────────────

class GINModel(nn.Module):
    """
    GIN: most expressive 1-WL equivalent architecture.
    Uses sum aggregation and a learned ε parameter.
    Readout is concatenation of representations from all layers (JK).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_tasks: int = 12,
        dropout: float = 0.3,
        readout: str = "sum",
        eps: float = 0.0,
        train_eps: bool = True,
        mlp_layers: int = 2,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = MLPLayer(hidden_dim, hidden_dim, hidden_dim,
                           num_layers=mlp_layers, dropout=0.0)
            self.convs.append(GINConv(mlp, train_eps=train_eps, eps=eps))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JK (jumping knowledge): concat all layer outputs
        jk_dim = hidden_dim * (num_layers + 1)   # +1 for input projection

        pool, self._is_att = build_readout(readout, hidden_dim)
        if self._is_att:
            self.pool = pool
        else:
            self._pool_fn = pool

        self.classifier = MLPLayer(jk_dim, hidden_dim, num_tasks,
                                   num_layers=2, dropout=dropout)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = F.relu(self.input_proj(x))
        layer_outs = [h]

        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outs.append(h)

        # Jumping Knowledge: pool each layer, then concat
        pooled = []
        for h_l in layer_outs:
            if self._is_att:
                pooled.append(self.pool(h_l, batch))
            else:
                pooled.append(self._pool_fn(h_l, batch))

        g = torch.cat(pooled, dim=-1)   # (B, hidden * (L+1))
        return self.classifier(g)


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(cfg: dict, in_dim: int) -> nn.Module:
    """
    Instantiate a model from a config dict.

    Expected keys: model.type, model.hidden_dim, model.num_layers,
                   model.num_tasks, model.dropout, model.readout,
                   gat.heads, gat.concat, gin.eps, gin.train_eps
    """
    mtype = cfg["model"]["type"].lower()
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_tasks=cfg["model"]["num_tasks"],
        dropout=cfg["model"]["dropout"],
        readout=cfg["model"]["readout"],
    )

    if mtype == "gcn":
        return GCNModel(**kwargs)
    elif mtype == "gat":
        return GATModel(
            **kwargs,
            heads=cfg["gat"]["heads"],
            concat=cfg["gat"]["concat"],
        )
    elif mtype == "gin":
        return GINModel(
            **kwargs,
            eps=cfg["gin"]["eps"],
            train_eps=cfg["gin"]["train_eps"],
            mlp_layers=cfg["gin"]["mlp_layers"],
        )
    else:
        raise ValueError(f"Unknown model type: {mtype!r}. Choose gcn/gat/gin.")


if __name__ == "__main__":
    # Quick architecture smoke-test
    from torch_geometric.data import Batch, Data

    def dummy_batch(n=4, num_atoms=12, in_dim=77):
        graphs = []
        for _ in range(n):
            x = torch.randn(num_atoms, in_dim)
            ei = torch.randint(0, num_atoms, (2, 20))
            graphs.append(Data(x=x, edge_index=ei))
        return Batch.from_data_list(graphs)

    IN_DIM = 77
    batch = dummy_batch(in_dim=IN_DIM)

    for name, Model in [("GCN", GCNModel), ("GAT", GATModel), ("GIN", GINModel)]:
        if name == "GAT":
            m = Model(in_dim=IN_DIM)
        else:
            m = Model(in_dim=IN_DIM)
        out = m(batch)
        params = sum(p.numel() for p in m.parameters())
        print(f"{name:4s}  output={tuple(out.shape)}  params={params:,}")
