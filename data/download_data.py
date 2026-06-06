"""
data/download_data.py
─────────────────────
Downloads the Tox21 dataset via DeepChem, saves raw SMILES + labels as CSV,
and prints a quick summary.
"""

import os
import numpy as np
import pandas as pd

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


def download_tox21(save_dir: str = "./data/tox21") -> None:
    """Download Tox21 via DeepChem and save splits as CSV."""
    try:
        import deepchem as dc
    except ImportError:
        raise ImportError("Install deepchem: pip install deepchem")

    os.makedirs(save_dir, exist_ok=True)

    print("📥  Downloading Tox21 dataset via DeepChem …")
    featurizer = dc.feat.DummyFeaturizer()          # raw SMILES only
    loader = dc.data.CSVLoader(
        tasks=TOX21_TASKS,
        feature_field="smiles",
        featurizer=featurizer,
    )

    # Use DeepChem's built-in MoleculeNet loader for convenience
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer="Raw",   # returns SMILES strings
        splitter="scaffold",
    )
    train_ds, val_ds, test_ds = datasets

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        smiles = ds.ids                              # SMILES strings
        labels = ds.y                                # (N, 12) float array, NaN = missing
        df = pd.DataFrame(labels, columns=TOX21_TASKS)
        df.insert(0, "smiles", smiles)
        path = os.path.join(save_dir, f"{split_name}.csv")
        df.to_csv(path, index=False)
        print(f"  ✅  {split_name:5s}: {len(df):5d} molecules → {path}")

    _print_summary(save_dir)


def _print_summary(save_dir: str) -> None:
    print("\n📊  Dataset Summary")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(save_dir, f"{split}.csv"))
        n_total = len(df)
        print(f"\n  {split.upper()} ({n_total} molecules)")
        print(f"  {'Task':<20} {'Pos':>6} {'Neg':>6} {'Missing':>8} {'Pos%':>6}")
        print(f"  {'-'*50}")
        for task in TOX21_TASKS:
            col = df[task]
            pos = int((col == 1).sum())
            neg = int((col == 0).sum())
            missing = int(col.isna().sum())
            pct = 100 * pos / (pos + neg) if (pos + neg) > 0 else 0
            print(f"  {task:<20} {pos:>6} {neg:>6} {missing:>8} {pct:>5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    download_tox21()
