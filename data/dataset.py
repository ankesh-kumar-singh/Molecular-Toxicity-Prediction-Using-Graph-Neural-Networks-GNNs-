"""
data/dataset.py
───────────────
MoleculeDataset: converts SMILES → PyG Data objects.

Atom features (node):
  - Atomic number (one-hot, 44 common elements)
  - Degree (0–10)
  - Formal charge
  - Chiral tag
  - Hybridization
  - Aromaticity
  - Total Hs

Bond features (edge):
  - Bond type (single/double/triple/aromatic)
  - Conjugation flag
  - Ring membership
  - Stereo configuration
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("Install RDKit: pip install rdkit")

# ─── Feature vocabulary ───────────────────────────────────────────────────────

ALLOWABLE_ATOMS = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
    "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
    "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge",
    "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb",
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


# ─── Feature helpers ──────────────────────────────────────────────────────────

def one_hot(value, allowable: list, include_other: bool = True) -> List[float]:
    """One-hot encode `value` against `allowable` set."""
    enc = [float(value == v) for v in allowable]
    if include_other:
        enc.append(float(value not in allowable))
    return enc


def atom_features(atom) -> List[float]:
    """Return feature vector for a single RDKit atom."""
    return (
        one_hot(atom.GetSymbol(), ALLOWABLE_ATOMS)          # 45
        + one_hot(atom.GetDegree(), list(range(11)))         # 12
        + [float(atom.GetFormalCharge())]                    # 1
        + one_hot(atom.GetChiralTag(), [0, 1, 2, 3])         # 5
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)  # 7
        + [float(atom.GetIsAromatic())]                      # 1
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])    # 6
    )
    # Total: 45 + 12 + 1 + 5 + 7 + 1 + 6 = 77


def bond_features(bond) -> List[float]:
    """Return feature vector for a single RDKit bond."""
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)             # 5
        + [float(bond.GetIsConjugated())]                   # 1
        + [float(bond.IsInRing())]                          # 1
        + one_hot(bond.GetStereo(), STEREO_TYPES)           # 5
    )
    # Total: 5 + 1 + 1 + 5 = 12


ATOM_FEATURE_DIM = len(atom_features(
    Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FEATURE_DIM = len(bond_features(
    Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


# ─── SMILES → PyG Data ───────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, y: Optional[torch.Tensor] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float,
    )  # (num_atoms, ATOM_FEATURE_DIM)

    # Edge indices + features (bidirectional)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feat, feat]

    if len(edge_indices) == 0:
        # Isolated atom / single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).T  # (2, E)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)       # (E, BOND_FEATURE_DIM)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = y
    data.smiles = smiles
    return data


# ─── Dataset ─────────────────────────────────────────────────────────────────

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class MoleculeDataset(Dataset):
    """
    PyG Dataset wrapping a CSV with columns:
        smiles, NR-AR, NR-AR-LBD, … (12 task columns)

    Missing labels (NaN) are encoded as -1 and masked during training.
    """

    def __init__(
        self,
        csv_path: str,
        tasks: List[str] = TOX21_TASKS,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform,
                         pre_transform=pre_transform)
        self.tasks = tasks
        self._load(csv_path)

    def _load(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        self._data_list: List[Data] = []
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  Featurizing {os.path.basename(csv_path)}"):
            smiles = row["smiles"]
            labels = row[self.tasks].values.astype(float)
            # NaN → -1 (mask sentinel)
            labels = np.where(np.isnan(labels), -1.0, labels)
            y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)  # (1, 12)

            data = smiles_to_graph(smiles, y=y)
            if data is None:
                skipped += 1
                continue
            self._data_list.append(data)

        if skipped:
            print(f"  ⚠️  Skipped {skipped} unparseable molecules.")

    # ── PyG Dataset interface ──────────────────────────────────────────────

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    @property
    def num_node_features(self) -> int:
        return ATOM_FEATURE_DIM

    @property
    def num_edge_features(self) -> int:
        return BOND_FEATURE_DIM

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Atom feature dim : {ATOM_FEATURE_DIM}")
    print(f"Bond feature dim : {BOND_FEATURE_DIM}")

    # Quick sanity check on aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    g = smiles_to_graph(smiles)
    print(f"\nAspirin ({smiles})")
    print(f"  Nodes : {g.x.shape}")
    print(f"  Edges : {g.edge_index.shape}")
    print(f"  Edge features: {g.edge_attr.shape}")
