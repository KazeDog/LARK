"""Processed classification and regression molecular-property datasets.

The dataset deliberately reads only the normalized columns produced by the
MoleculeNet preparation pipeline.  In particular, descriptor columns present
in the raw BACE table can never enter the model through this loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hypermol.data.mol_collator import MoleculeCollator
from hypermol.data.molecule_store import open_molecule_store


PROPERTY_COLUMNS = ("canonical_smiles", "label", "source_index")
PROPERTY_BASE_COLUMNS = ("canonical_smiles", "source_index")


class MolecularPropertyDataset(Dataset):
    """Masked molecular-property dataset backed by a molecule store.

    ``split_csv_path`` is a processed split, not a raw MoleculeNet table.  Only
    ``canonical_smiles``, ``label`` and ``source_index`` are read, which makes
    accidental use of BACE's precomputed descriptor columns impossible.
    """

    def __init__(
        self,
        split_csv_path: str,
        molecule_store_path: str,
        *,
        label_columns: Sequence[str] | None = None,
        task_names: Sequence[str] | None = None,
        task_type: str = "classification",
    ):
        self.split_csv_path = str(Path(split_csv_path).expanduser().resolve())
        self.molecule_store_path = str(Path(molecule_store_path).expanduser().resolve())
        header = pd.read_csv(self.split_csv_path, nrows=0).columns.tolist()
        if label_columns is None:
            if "label" in header:
                label_columns = ["label"]
            else:
                label_columns = sorted(
                    column for column in header if column.startswith("label_") and column[6:].isdigit()
                )
        self.label_columns = [str(column) for column in label_columns]
        if not self.label_columns or any(column not in header for column in self.label_columns):
            raise ValueError(
                f"Could not resolve processed property label columns in {self.split_csv_path}: "
                f"{self.label_columns}"
            )
        self.task_names = [str(value) for value in (task_names or self.label_columns)]
        if len(self.task_names) != len(self.label_columns):
            raise ValueError("task_names and label_columns must have the same length")
        self.task_type = str(task_type).strip().lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported property task_type: {self.task_type!r}")
        usecols = [*PROPERTY_BASE_COLUMNS, *self.label_columns]
        self.df = pd.read_csv(self.split_csv_path, usecols=usecols)
        self.df = self.df.loc[:, usecols].copy()
        self._validate_table()
        self.store = None

    @property
    def num_tasks(self) -> int:
        return len(self.label_columns)

    @property
    def labels_matrix(self) -> np.ndarray:
        return self.df.loc[:, self.label_columns].to_numpy(dtype=np.float32, copy=True)

    def _validate_table(self) -> None:
        if self.df.empty:
            raise ValueError(f"Property split is empty: {self.split_csv_path}")
        if self.df[list(PROPERTY_BASE_COLUMNS)].isnull().any().any():
            raise ValueError(f"Property split contains missing required metadata: {self.split_csv_path}")

        smiles = self.df["canonical_smiles"].astype(str).str.strip()
        if (smiles == "").any():
            raise ValueError(f"Property split contains empty canonical_smiles: {self.split_csv_path}")
        self.df["canonical_smiles"] = smiles

        for column in self.label_columns:
            labels = pd.to_numeric(self.df[column], errors="raise")
            observed = labels.dropna()
            if not np.isfinite(observed.to_numpy(dtype=np.float64)).all():
                raise ValueError(
                    f"Observed property labels must be finite for {column} in {self.split_csv_path}"
                )
            if self.task_type == "classification" and not observed.isin([0, 1]).all():
                values = sorted(set(observed.tolist()))
                raise ValueError(
                    f"Binary property labels must be 0/1 or missing; observed {values} "
                    f"for {column} in {self.split_csv_path}"
                )
            self.df[column] = labels.astype("float32")
        if self.df[self.label_columns].notna().sum(axis=1).eq(0).any():
            raise ValueError(f"Property split contains rows with no observed task label: {self.split_csv_path}")

        source_indices = pd.to_numeric(self.df["source_index"], errors="raise")
        integer_source_indices = source_indices.astype("int64")
        if not (source_indices == integer_source_indices).all():
            raise ValueError(f"source_index must contain integers: {self.split_csv_path}")
        self.df["source_index"] = integer_source_indices

    def __len__(self) -> int:
        return len(self.df)

    def _ensure_store(self):
        if self.store is None:
            self.store = open_molecule_store(self.molecule_store_path)
        return self.store

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        canonical_smiles = str(row["canonical_smiles"])
        try:
            molecule_dict = self._ensure_store().get_mol_dict(canonical_smiles)
        except KeyError as exc:
            raise KeyError(
                f"Processed property molecule is missing from the store: {canonical_smiles} "
                f"(split={self.split_csv_path}, source_index={int(row['source_index'])})"
            ) from exc
        labels = row[self.label_columns].to_numpy(dtype=np.float32, copy=True)
        label_mask = np.isfinite(labels)
        label_value = float(labels[0]) if self.num_tasks == 1 else labels
        mask_value = bool(label_mask[0]) if self.num_tasks == 1 else label_mask
        return {
            "molecule_dict": molecule_dict,
            "label": label_value,
            "label_mask": mask_value,
            "canonical_smiles": canonical_smiles,
            "source_index": int(row["source_index"]),
        }

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["store"] = None
        return state

    def __del__(self) -> None:
        store = getattr(self, "store", None)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
            self.store = None


class MolecularPropertyCollator:
    """Collate single-molecule examples for ``MolecularPropertyModel``."""

    def __init__(self, task_names: Sequence[str] | None = None):
        self.molecule_collator = MoleculeCollator()
        self.task_names = [str(value) for value in task_names] if task_names is not None else None

    def __call__(self, batch: List[Optional[Dict]]) -> Dict:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return {}
        first_label = np.asarray(valid_batch[0]["label"])
        if first_label.ndim == 0:
            labels = torch.tensor([item["label"] for item in valid_batch], dtype=torch.float32)
            label_mask = torch.tensor([item["label_mask"] for item in valid_batch], dtype=torch.bool)
        else:
            labels = torch.from_numpy(
                np.stack([np.asarray(item["label"], dtype=np.float32) for item in valid_batch])
            )
            label_mask = torch.from_numpy(
                np.stack([np.asarray(item["label_mask"], dtype=np.bool_) for item in valid_batch])
            )
        return {
            "molecule_batch": self.molecule_collator(
                [item["molecule_dict"] for item in valid_batch],
                mode="finetune",
            ),
            "labels": labels,
            "label_mask": label_mask,
            "task_names": self.task_names,
            "smiles": [str(item["canonical_smiles"]) for item in valid_batch],
            "source_index": torch.tensor(
                [item["source_index"] for item in valid_batch],
                dtype=torch.long,
            ),
        }


__all__ = [
    "PROPERTY_COLUMNS",
    "PROPERTY_BASE_COLUMNS",
    "MolecularPropertyDataset",
    "MolecularPropertyCollator",
]
