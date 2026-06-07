"""
backend/main.py
"""

from __future__ import annotations

import io
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataset import smiles_to_graph, ATOM_FEATURE_DIM
from models.gnn import build_model

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

TASK_DESCRIPTIONS = {
    "NR-AR":        "Androgen Receptor — steroid hormone receptor involved in male development",
    "NR-AR-LBD":    "Androgen Receptor Ligand Binding Domain",
    "NR-AhR":       "Aryl Hydrocarbon Receptor — mediates dioxin & PAH toxicity",
    "NR-Aromatase": "Aromatase enzyme — involved in estrogen biosynthesis",
    "NR-ER":        "Estrogen Receptor Alpha — key regulator of female development",
    "NR-ER-LBD":    "Estrogen Receptor Alpha Ligand Binding Domain",
    "NR-PPAR-gamma":"PPAR-gamma — lipid metabolism and adipogenesis regulator",
    "SR-ARE":       "Antioxidant Response Element — oxidative stress pathway",
    "SR-ATAD5":     "ATAD5 — DNA damage response and genome integrity",
    "SR-HSE":       "Heat Shock Factor Response Element — cellular stress response",
    "SR-MMP":       "Mitochondrial Membrane Potential — mitochondrial toxicity",
    "SR-p53":       "p53 Tumor Suppressor — DNA damage and apoptosis pathway",
}

MODEL_INFO = {
    "gcn": {"name": "GCN", "full_name": "Graph Convolutional Network",
            "params": 319628, "val_auc": 0.8804, "speed": "Fast",
            "description": "Spectral graph convolutions with residual connections"},
    "gat": {"name": "GAT", "full_name": "Graph Attention Network",
            "params": 3714060, "val_auc": 0.8536, "speed": "Medium",
            "description": "Multi-head attention over molecular neighborhoods"},
    "gin": {"name": "GIN", "full_name": "Graph Isomorphism Network",
            "params": 881680, "val_auc": 0.8821, "speed": "Fast",
            "description": "Most expressive GNN with jumping knowledge readout"},
}

CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    os.path.join(ROOT, "results", "checkpoints")
)
CONFIG_PATH = os.environ.get(
    "CONFIG_PATH",
    os.path.join(ROOT, "configs", "config.yaml")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_HISTORY = 50

_model_cache: Dict[str, torch.nn.Module] = {}
_history: deque = deque(maxlen=MAX_HISTORY)


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _list_available() -> List[str]:
    if not os.path.isdir(CHECKPOINT_DIR):
        return []
    return [
        f.replace("best_", "").replace(".pt", "").lower()
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("best_") and f.endswith(".pt")
    ]


def _get_model(model_name: str) -> torch.nn.Module:
    key = model_name.lower()
    if key in _model_cache:
        return _model_cache[key]
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{key.upper()}.pt")
    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found for '{key}'.")
    cfg = _load_config()
    cfg["model"]["type"] = key
    model = build_model(cfg, in_dim=ATOM_FEATURE_DIM).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model_cache[key] = model
    return model


def _run_inference(smiles_list: List[str], model_name: str) -> List[dict]:
    from torch_geometric.data import Batch
    model = _get_model(model_name)
    results = []
    valid_idx, valid_graphs = [], []

    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi.strip())
        if g is None:
            results.append({
                "smiles": smi, "valid": False,
                "predictions": {}, "toxic_tasks": [],
                "max_prob": 0.0, "risk_level": "invalid",
            })
        else:
            results.append(None)
            valid_idx.append(i)
            valid_graphs.append(g)

    if not valid_graphs:
        return results

    batch = Batch.from_data_list(valid_graphs).to(DEVICE)
    with torch.no_grad():
        logits = model(batch)
        probs  = torch.sigmoid(logits).cpu().numpy()

    for j, idx in enumerate(valid_idx):
        p           = probs[j]
        pred_dict   = {task: round(float(p[k]), 4) for k, task in enumerate(TOX21_TASKS)}
        toxic_tasks = [t for t, v in pred_dict.items() if v >= 0.5]
        max_prob    = round(float(p.max()), 4)
        risk_level  = "high" if max_prob >= 0.5 else "moderate" if max_prob >= 0.3 else "low"
        results[idx] = {
            "smiles":      smiles_list[idx],
            "valid":       True,
            "predictions": pred_dict,
            "toxic_tasks": toxic_tasks,
            "max_prob":    max_prob,
            "risk_level":  risk_level,
        }
    return results


def _smiles_to_svg(smiles: str, width: int = 300, height: int = 220, theme: str = "dark") -> Optional[str]:
    """Generate SVG from SMILES using RDKit on the backend."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)

        # Theme-aware drawing options
        opts = drawer.drawOptions()
        if theme == "dark":
            opts.backgroundColour = (0, 0, 0, 0)        # transparent
            opts.atomLabelColour  = (0.91, 0.92, 0.94)  # light text
            opts.bondLineColour   = (0.91, 0.92, 0.94)  # light bonds
        else:
            opts.backgroundColour = (0, 0, 0, 0)        # transparent
            opts.atomLabelColour  = (0.1,  0.11, 0.18)  # dark text
            opts.bondLineColour   = (0.1,  0.11, 0.18)  # dark bonds

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"  Device: {DEVICE}")
    for name in _list_available():
        try:
            _get_model(name)
            info = MODEL_INFO.get(name, {})
            print(f"  Loaded {name.upper()} (AUC={info.get('val_auc','?')})")
        except Exception as e:
            print(f"  Could not load {name}: {e}")
    yield
    _model_cache.clear()


app = FastAPI(
    title="Tox21 GNN API",
    description="Molecular Toxicity Classifier — GCN, GAT, GIN",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    smiles: str = Field(..., example="CC(=O)Oc1ccccc1C(=O)O")
    model:  str = Field("gin", example="gin")

class BatchRequest(BaseModel):
    smiles_list: List[str] = Field(..., min_length=1, max_length=500)
    model: str = Field("gin")

class ValidateRequest(BaseModel):
    smiles: str

class MolSVGRequest(BaseModel):
    smiles: str
    width:  int = 300
    height: int = 220
    theme:  str = "dark"


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "device":        str(DEVICE),
        "models_loaded": list(_model_cache.keys()),
        "available":     _list_available(),
        "timestamp":     time.time(),
    }


@app.get("/models")
def list_models():
    available = _list_available()
    result = []
    for key in ["gcn", "gat", "gin"]:
        info = MODEL_INFO.get(key, {}).copy()
        info["available"] = key in available
        info["loaded"]    = key in _model_cache
        info["id"]        = key
        result.append(info)
    return result


@app.get("/tasks")
def list_tasks():
    return [
        {"id": task, "description": TASK_DESCRIPTIONS[task]}
        for task in TOX21_TASKS
    ]


@app.post("/validate")
def validate_smiles(req: ValidateRequest):
    try:
        from rdkit import Chem
        mol   = Chem.MolFromSmiles(req.smiles.strip())
        valid = mol is not None
        return {
            "smiles": req.smiles,
            "valid":  valid,
            "atoms":  mol.GetNumAtoms() if valid else 0,
            "bonds":  mol.GetNumBonds() if valid else 0,
        }
    except Exception:
        return {"smiles": req.smiles, "valid": False, "atoms": 0, "bonds": 0}


@app.post("/mol/svg")
def mol_svg(req: MolSVGRequest):
    """Generate molecule SVG on the backend using RDKit."""
    if not req.smiles.strip():
        raise HTTPException(400, "SMILES cannot be empty")
    svg = _smiles_to_svg(req.smiles, req.width, req.height, req.theme)
    if svg is None:
        raise HTTPException(400, "Could not render molecule — invalid SMILES")
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/predict")
def predict_single(req: PredictRequest):
    t0     = time.time()
    result = _run_inference([req.smiles], req.model)[0]
    result["model"]      = req.model
    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
    result["id"]         = str(uuid.uuid4())[:8]
    result["timestamp"]  = time.time()
    _history.appendleft({**result})
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if len(req.smiles_list) > 500:
        raise HTTPException(400, "Max 500 molecules per batch.")
    results = _run_inference(req.smiles_list, req.model)
    for r in results:
        r["model"] = req.model
    return results


@app.post("/predict/compare")
def predict_compare(req: ValidateRequest):
    available = _list_available()
    if not available:
        raise HTTPException(404, "No models available.")
    comparison = {}
    for model_name in available:
        t0     = time.time()
        result = _run_inference([req.smiles], model_name)[0]
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        comparison[model_name] = result
    return {"smiles": req.smiles, "models": comparison, "timestamp": time.time()}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), model: str = Query("gin")):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a .csv")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    if "smiles" not in df.columns:
        raise HTTPException(400, "CSV must have a 'smiles' column.")
    if len(df) > 500:
        raise HTTPException(400, "Max 500 rows per CSV.")
    smiles_list = df["smiles"].astype(str).tolist()
    results     = _run_inference(smiles_list, model)
    rows = []
    for r in results:
        row = {
            "smiles":      r["smiles"],
            "valid":       r["valid"],
            "risk_level":  r.get("risk_level", ""),
            "max_prob":    r["max_prob"],
            "toxic_tasks": ", ".join(r["toxic_tasks"]),
        }
        row.update(r["predictions"])
        rows.append(row)
    out_csv = pd.DataFrame(rows).to_csv(index=False)
    return Response(
        content=out_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tox21_predictions.csv"},
    )


@app.get("/history")
def get_history(limit: int = Query(20, le=50)):
    return list(_history)[:limit]


@app.delete("/history")
def clear_history():
    _history.clear()
    return {"message": "History cleared"}
