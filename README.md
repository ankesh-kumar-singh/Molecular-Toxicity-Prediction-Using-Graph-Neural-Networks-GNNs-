# 🧪 Tox21 GNN — Molecular Toxicity Classifier

End-to-end project: **GNN training** + **FastAPI backend** + **React frontend**, all in one folder.

---

## 📁 Project Structure

```
tox21/
│
├── README.md                        ← you are here
├── requirements.txt                 ← Python deps (training + backend)
├── render.yaml                      ← Render.com backend deploy config
├── vercel.json                      ← Vercel frontend deploy config
├── .gitignore
│
├── configs/
│   └── config.yaml                  ← all hyperparameters & paths
│
├── data/
│   ├── download_data.py             ← downloads Tox21 via DeepChem → CSV splits
│   ├── dataset.py                   ← MoleculeDataset: SMILES → PyG graphs
│   └── tox21/                       ← generated (train.csv, val.csv, test.csv)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── gnn.py                       ← GCN, GAT, GIN model definitions + build_model()
│   └── layers.py                    ← MLPLayer, AttentionReadout
│
├── utils/
│   ├── train.py                     ← masked BCE loss, training loop, early stopping
│   ├── evaluate.py                  ← ROC-AUC, PR-AUC, F1 per task
│   └── visualize.py                 ← training curves, task AUC bars, ROC curves
│
├── scripts/
│   ├── train.py                     ← main training entrypoint (CLI)
│   ├── predict.py                   ← single/batch SMILES → toxicity prediction
│   └── hyperparameter_search.py     ← Optuna HPO
│
├── notebooks/
│   └── exploration.ipynb            ← EDA + results visualization
│
├── results/
│   ├── checkpoints/                 ← saved .pt files (best_GIN.pt etc.)
│   ├── plots/                       ← training curves, AUC charts
│   └── logs/                        ← tensorboard logs
│
├── backend/
│   ├── main.py                      ← FastAPI app (5 endpoints)
│   └── requirements.txt             ← backend-only deps for Render
│
└── frontend/
    ├── package.json
    ├── .env.example                 ← copy to .env.local, set API URL
    ├── public/
    │   └── index.html
    └── src/
        ├── index.js
        ├── App.jsx                  ← main app: tabs, SMILES input, model selector
        ├── App.css                  ← dark lab aesthetic
        ├── components/
        │   ├── MoleculeViewer.jsx   ← 2D structure via RDKit.js (WASM)
        │   ├── ToxicityResults.jsx  ← animated task bars + verdict
        │   └── BatchUpload.jsx      ← drag-drop CSV + results table
        └── utils/
            └── api.js               ← API client (reads REACT_APP_API_URL)
```

---

## 🚀 Quick Start

### 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

> **PyTorch Geometric** extra step (choose your CUDA version):
> ```bash
> # CPU only
> pip install torch-scatter torch-sparse torch-geometric \
>   -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
> ```

### 2 — Download & prepare data
```bash
python data/download_data.py
# Creates data/tox21/train.csv, val.csv, test.csv
```

### 3 — Train a model
```bash
# GIN (recommended — best AUC ~0.82)
python scripts/train.py --model gin --epochs 100

# GCN (fast baseline)
python scripts/train.py --model gcn

# GAT with custom params
python scripts/train.py --model gat --hidden_dim 256 --epochs 150
```
Checkpoints saved to `results/checkpoints/best_GIN.pt` etc.

### 4 — Predict a molecule
```bash
# Aspirin
python scripts/predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

# With molecule drawing
python scripts/predict.py --smiles "c1ccc2cc3ccccc3cc2c1" --visualize

# Batch from file (one SMILES per line)
python scripts/predict.py --smiles_file molecules.txt
```

### 5 — Hyperparameter search
```bash
python scripts/hyperparameter_search.py --n_trials 50
```

---

## 🖥 Running Locally (API + Frontend)

### Backend (FastAPI)
```bash
cd backend
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs   (Swagger UI)
```

### Frontend (React)
```bash
cd frontend
cp .env.example .env.local
# .env.local already has REACT_APP_API_URL=http://localhost:8000

npm install
npm start
# → http://localhost:3000
```

---

## 🚢 Deployment

### Backend → Render.com
1. Push repo to GitHub
2. Render → **New → Blueprint** → connect repo (reads `render.yaml`)
3. Set env vars if needed:
   - `CHECKPOINT_DIR` → path to your `.pt` files
   - `CONFIG_PATH`    → path to `configs/config.yaml`

### Frontend → Vercel
1. Vercel → **New Project** → import repo
2. **Root Directory**: `frontend`
3. Add env var: `REACT_APP_API_URL = https://tox21-api.onrender.com`
4. Deploy

---

## 🧠 Models

| Model | Description | Mean ROC-AUC |
|-------|-------------|:------------:|
| **GIN** ★ | Graph Isomorphism Network, jumping-knowledge readout | ~0.82 |
| **GAT**   | Graph Attention Network, 4-head attention | ~0.81 |
| **GCN**   | Graph Convolutional Network, residual connections | ~0.78 |

---

## 📊 Tox21 Tasks (12 assays)

| ID | Assay |
|----|-------|
| NR-AR | Androgen Receptor |
| NR-AR-LBD | Androgen Receptor Ligand Binding Domain |
| NR-AhR | Aryl Hydrocarbon Receptor |
| NR-Aromatase | Aromatase |
| NR-ER | Estrogen Receptor Alpha |
| NR-ER-LBD | Estrogen Receptor Alpha LBD |
| NR-PPAR-gamma | PPAR-gamma |
| SR-ARE | Antioxidant Response Element |
| SR-ATAD5 | ATAD5 / DNA damage |
| SR-HSE | Heat Shock Factor Response Element |
| SR-MMP | Mitochondrial Membrane Potential |
| SR-p53 | p53 tumor suppressor |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Liveness + loaded models |
| `GET`  | `/models` | Available checkpoints |
| `GET`  | `/tasks`  | Task names + descriptions |
| `POST` | `/predict` | Single SMILES |
| `POST` | `/predict/batch` | Up to 500 SMILES (JSON) |
| `POST` | `/predict/csv` | CSV upload → CSV download |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "model": "gin"}'
```

---

> **Research use only.** Not for clinical or regulatory decisions.
