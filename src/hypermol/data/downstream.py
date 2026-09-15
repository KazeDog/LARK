from __future__ import annotations

import json
import hashlib
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from hypermol.data.condition_roles import aux_role_to_id
from hypermol.data.component_merge import ensure_atom_map_num, merge_component_dicts
from hypermol.data.collator import HypergraphCollatorForGT
from hypermol.data.molecule_store import open_molecule_store
from hypermol.data.mol_collator import MoleculeCollator


CONDITION_FIELDS = ["catalyst1", "solvent1", "solvent2", "reagent1", "reagent2"]


def apply_aux_role_ablation(
    role_ids: Sequence[int],
    *,
    mode: str,
    seed: int,
    row_index: int,
) -> List[int]:
    """Apply the deterministic role negative control without changing molecules."""

    values = [int(value) for value in role_ids]
    normalized = str(mode).strip().lower()
    if normalized == "true":
        return values
    if normalized == "none":
        return [0] * len(values)
    if normalized != "shuffled":
        raise ValueError("Role ablation mode must be true, none, or shuffled.")
    if len(values) <= 1:
        return values
    digest = hashlib.sha256(
        f"{int(seed)}\x1f{int(row_index)}".encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
    rng.shuffle(values)
    return values


def _safe_int(value: object, default: int = -100) -> int:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if text == "":
        return default
    return int(float(text))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def _json_list(value: object) -> List:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if text == "":
        return []
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else []


class _LMDBMixin:
    """Backward-compatible molecule-store mixin.

    The class name is kept because older datasets pass ``molecule_lmdb_path``,
    but the implementation now supports both LMDB and pickle-backed stores.
    """

    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self.store = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["store"] = None
        return state

    def _ensure_store(self):
        if self.store is None:
            self.store = open_molecule_store(self.lmdb_path)
        return self.store

    def _get_payload_by_key(self, key: str | bytes) -> Optional[Dict]:
        try:
            return self._ensure_store().get_mol_dict_by_key(key)
        except Exception:
            return None

    def _get_mol_by_smiles(self, smiles: str) -> Optional[Dict]:
        try:
            return self._ensure_store().get_mol_dict(smiles)
        except Exception:
            return None

    def __del__(self):
        if getattr(self, "store", None) is not None:
            try:
                self.store.close()
            except Exception:
                pass
            self.store = None


class ReactionClassificationDataset(_LMDBMixin, Dataset):
    """Reaction-pair dataset for Schneider/USPTO-style reaction classification."""

    def __init__(self, split_csv_path: str, molecule_lmdb_path: str, label_column: str = ""):
        _LMDBMixin.__init__(self, molecule_lmdb_path)
        self.df = pd.read_csv(split_csv_path)
        if not label_column:
            for candidate in ("reaction_type", "class", "label"):
                if candidate in self.df.columns:
                    label_column = candidate
                    break
        if not label_column or label_column not in self.df.columns:
            raise ValueError(f"Could not find label column in {split_csv_path}")
        self.label_column = label_column
        self.num_classes = int(pd.to_numeric(self.df[self.label_column]).max()) + 1

    def __len__(self) -> int:
        return len(self.df)

    def _load_components(self, smiles_text: str) -> Optional[List[Dict]]:
        components = []
        for smi in str(smiles_text).split("."):
            mol_dict = self._get_mol_by_smiles(smi)
            if mol_dict is None:
                return None
            components.append(ensure_atom_map_num(mol_dict))
        return components

    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df.iloc[idx]
        reactants = self._load_components(str(row["reactant_smiles"]))
        products = self._load_components(str(row["prod_smiles"]))
        if reactants is None or products is None:
            return None
        return {
            "reactant_dict": merge_component_dicts(reactants),
            "product_dict": merge_component_dicts(products),
            "reactant_component_dicts": reactants,
            "product_component_dicts": products,
            "label": int(row[self.label_column]),
            "reaction_smiles": f"{row['reactant_smiles']}>>{row['prod_smiles']}",
        }


class GraphConditionDataset(_LMDBMixin, Dataset):
    """Processed USPTO-Condition graph dataset backed by hashed molecule LMDB."""

    def __init__(self, split_csv_path: str, molecule_lmdb_path: str):
        _LMDBMixin.__init__(self, molecule_lmdb_path)
        self.df = pd.read_csv(split_csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def _load_hashes(self, row: pd.Series, column: str) -> Optional[List[Dict]]:
        hashes = _json_list(row[column])
        components = [self._get_payload_by_key(h) for h in hashes]
        if any(component is None for component in components):
            return None
        return components

    def __getitem__(self, idx: int) -> Optional[Dict]:
        row = self.df.iloc[idx]
        reactants = self._load_hashes(row, "reactant_hashes")
        products = self._load_hashes(row, "product_hashes")
        if reactants is None or products is None:
            return None
        labels = {field: _safe_int(row.get(f"label_{field}"), default=-100) for field in CONDITION_FIELDS}
        return {
            "reactant_dict": merge_component_dicts(reactants),
            "product_dict": merge_component_dicts(products),
            "labels": labels,
            "reg_targets": {
                key: _safe_float(row.get(f"target_{key}"), default=0.0) for key in ("temperature", "time")
            },
            "reg_masks": {
                key: bool(_safe_int(row.get(f"mask_{key}"), default=0)) for key in ("temperature", "time")
            },
            "reaction_smiles": str(row.get("reaction_smiles", "")),
            "source": str(row.get("source", "")),
        }


class YieldDataset(Dataset):
    """Processed yield dataset backed by a hashed molecule feature store."""

    target_column = "yield"

    def __init__(
        self,
        split_csv_path: str,
        molecule_store_path: str,
        target_column: str | None = None,
        aux_role_mode: str = "true",
        aux_role_seed: int = 42,
    ):
        self.df = pd.read_csv(split_csv_path)
        self.molecule_store_path = molecule_store_path
        self.molecule_store = open_molecule_store(molecule_store_path)
        self.target_column = target_column or self.target_column
        role_mode = str(aux_role_mode).strip().lower()
        aliases = {"observed": "true", "off": "none", "zero": "none", "shuffle": "shuffled"}
        self.aux_role_mode = aliases.get(role_mode, role_mode)
        self.aux_role_seed = int(aux_role_seed)
        if self.aux_role_mode not in {"true", "none", "shuffled"}:
            raise ValueError(
                "aux_role_mode must be one of 'true', 'none', or 'shuffled'."
            )
        if self.target_column not in self.df.columns:
            raise ValueError(f"Could not find target column '{self.target_column}' in {split_csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _get_mol(self, key: str) -> Dict:
        return self.molecule_store.get_mol_dict_by_key(key)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["molecule_store"] = open_molecule_store(self.molecule_store_path)
        return state

    def __del__(self):
        store = getattr(self, "molecule_store", None)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        reactant_hashes = _json_list(row["reactant_hashes"])
        product_hashes = _json_list(row["product_hashes"])
        aux_hashes = _json_list(row.get("aux_hashes", "[]"))
        aux_roles = _json_list(row.get("aux_roles", "[]"))
        if len(aux_roles) < len(aux_hashes):
            aux_roles = list(aux_roles) + ["unknown"] * (len(aux_hashes) - len(aux_roles))
        reactants = [ensure_atom_map_num(self._get_mol(h)) for h in reactant_hashes]
        products = [ensure_atom_map_num(self._get_mol(h)) for h in product_hashes]
        aux_components = []
        aux_role_ids = []
        for aux_hash, aux_role in zip(aux_hashes, aux_roles):
            if not self.molecule_store.contains_key(aux_hash):
                continue
            aux_components.append(ensure_atom_map_num(self._get_mol(aux_hash)))
            aux_role_ids.append(aux_role_to_id(aux_role))
        aux_role_ids = apply_aux_role_ablation(
            aux_role_ids,
            mode=self.aux_role_mode,
            seed=self.aux_role_seed,
            row_index=int(idx),
        )
        return {
            "reactant_dict": merge_component_dicts(reactants),
            "product_dict": merge_component_dicts(products),
            "reactant_component_dicts": reactants,
            "product_component_dicts": products,
            "aux_component_dicts": aux_components,
            "aux_role_ids": aux_role_ids,
            "aux_role_mode": self.aux_role_mode,
            self.target_column: float(row[self.target_column]),
            "reaction_smiles": str(row.get("reaction_smiles", "")),
            "aux_smiles": str(row.get("aux_smiles", "")),
            "template_smiles": str(row.get("template_smiles", "")),
            "source": str(row.get("source", "")),
        }


class SelectivityDataset(YieldDataset):
    """Processed enantio-/regio-selectivity dataset using the yield data format."""

    target_column = "selectivity"

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        if self.target_column != "selectivity":
            item["selectivity"] = item.pop(self.target_column)
        return item


class YieldRankingDataset(Dataset):
    """Pairwise condition-ranking dataset built from yield candidates.

    Each item contains two candidates from the same reaction. The first candidate
    has a higher observed yield than the second candidate.
    """

    def __init__(
        self,
        split_csv_path: str,
        molecule_store_path: str,
        group_column: str = "reaction_smiles",
        group_columns: str | Sequence[str] | None = None,
        candidate_column: str = "",
        yield_column: str = "yield",
        min_yield_delta: float = 0.0,
        max_pairs_per_group: int = 0,
        seed: int = 42,
    ):
        self.base = YieldDataset(split_csv_path=split_csv_path, molecule_store_path=molecule_store_path)
        self.df = self.base.df
        if group_columns is None:
            group_columns = group_column
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        self.group_columns = [str(column) for column in group_columns if str(column)]
        self.group_column = self.group_columns[0] if len(self.group_columns) == 1 else "||".join(self.group_columns)
        self.candidate_column = str(candidate_column or "")
        self.yield_column = yield_column
        self.min_yield_delta = float(min_yield_delta)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self.seed = int(seed)
        self._group_keys = self._build_group_keys()
        self.pairs = self._build_pairs()

    def _build_group_keys(self) -> pd.Series:
        missing = [column for column in self.group_columns if column not in self.df.columns]
        if missing:
            raise ValueError(f"Yield ranking requires group column(s) {missing} in split CSV.")
        if self.candidate_column and self.candidate_column not in self.df.columns:
            raise ValueError(f"Yield ranking requires candidate column '{self.candidate_column}' in split CSV.")
        return self.df[self.group_columns].fillna("").astype(str).agg("||".join, axis=1)

    def _build_pairs(self) -> List[Tuple[int, int, float]]:
        if self.yield_column not in self.df.columns:
            raise ValueError(f"Yield ranking requires yield column '{self.yield_column}' in split CSV.")

        rng = random.Random(self.seed)
        pairs: List[Tuple[int, int, float]] = []
        yields = pd.to_numeric(self.df[self.yield_column], errors="coerce")
        candidates = self.df[self.candidate_column].fillna("").astype(str) if self.candidate_column else None
        for _, indices_raw in self._group_keys.groupby(self._group_keys, sort=False).groups.items():
            indices = [int(index) for index in indices_raw]
            group_pairs: List[Tuple[int, int, float]] = []
            for left_pos in range(len(indices)):
                left_idx = int(indices[left_pos])
                left_yield = float(yields.iloc[left_idx])
                if math.isnan(left_yield):
                    continue
                for right_idx_raw in indices[left_pos + 1 :]:
                    right_idx = int(right_idx_raw)
                    if candidates is not None and candidates.iloc[left_idx] == candidates.iloc[right_idx]:
                        continue
                    right_yield = float(yields.iloc[right_idx])
                    if math.isnan(right_yield):
                        continue
                    delta = left_yield - right_yield
                    if abs(delta) <= self.min_yield_delta:
                        continue
                    if delta > 0:
                        group_pairs.append((left_idx, right_idx, abs(delta)))
                    else:
                        group_pairs.append((right_idx, left_idx, abs(delta)))
            if self.max_pairs_per_group > 0 and len(group_pairs) > self.max_pairs_per_group:
                group_pairs = rng.sample(group_pairs, self.max_pairs_per_group)
            pairs.extend(group_pairs)
        if not pairs:
            raise ValueError(
                f"No ranking pairs were built from {len(self.df)} rows. "
                "Use a lower min_yield_delta or check grouping."
            )
        rng.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        better_idx, worse_idx, yield_diff = self.pairs[idx]
        better = self.base[better_idx]
        worse = self.base[worse_idx]
        return {
            "better": better,
            "worse": worse,
            "yield_diff": float(yield_diff),
            "group": str(self._group_keys.iloc[better_idx]),
        }


class ReactionPairCollator:
    """Collate downstream reaction-pair samples into nested encoder batches."""

    def __init__(self, task: str, aux_input_mode: str = "edge"):
        self.task = task
        self.aux_input_mode = str(aux_input_mode).lower()
        if self.aux_input_mode not in {"edge", "context_node"}:
            raise ValueError("aux_input_mode must be 'edge' or 'context_node'.")
        self.molecule_collator = MoleculeCollator()
        self.hypergraph_collator = (
            HypergraphCollatorForGT(mode="finetune") if task in {"reaction_class", "yield", "selectivity"} else None
        )

    def __call__(self, batch: List[Optional[Dict]]) -> Dict:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return {}
        reactants = [item["reactant_dict"] for item in valid_batch]
        products = [item["product_dict"] for item in valid_batch]
        out = {
            "reactant_batch": self.molecule_collator(reactants, mode="finetune"),
            "product_batch": self.molecule_collator(products, mode="finetune"),
        }
        if self.task == "reaction_class":
            out["labels"] = torch.tensor([item["label"] for item in valid_batch], dtype=torch.long)
            graph_samples = [
                {
                    "reactants_features": item.get("reactant_component_dicts", [item["reactant_dict"]]),
                    "products_features": item.get("product_component_dicts", [item["product_dict"]]),
                }
                for item in valid_batch
            ]
            out["reaction_graph_batch"] = self.hypergraph_collator(graph_samples) if self.hypergraph_collator is not None else {}
        elif self.task == "condition":
            out["labels"] = {
                key: torch.tensor([item["labels"][key] for item in valid_batch], dtype=torch.long)
                for key in CONDITION_FIELDS
            }
            out["reg_targets"] = {
                key: torch.tensor([item["reg_targets"][key] for item in valid_batch], dtype=torch.float32)
                for key in ("temperature", "time")
            }
            out["reg_masks"] = {
                key: torch.tensor([item["reg_masks"][key] for item in valid_batch], dtype=torch.bool)
                for key in ("temperature", "time")
            }
        elif self.task in {"yield", "selectivity"}:
            target_key = "yield" if self.task == "yield" else "selectivity"
            out[target_key] = torch.tensor([item[target_key] for item in valid_batch], dtype=torch.float32)
            out["reaction_smiles"] = [item.get("reaction_smiles", "") for item in valid_batch]
            out["aux_smiles"] = [item.get("aux_smiles", "") for item in valid_batch]
            out["template_smiles"] = [item.get("template_smiles", "") for item in valid_batch]
            out["source"] = [item.get("source", "") for item in valid_batch]
            aux_components = []
            aux_owner = []
            aux_role_ids = []
            aux_counts = []
            for owner_idx, item in enumerate(valid_batch):
                item_aux = item.get("aux_component_dicts", [])
                item_aux_role_ids = list(item.get("aux_role_ids", []))
                if len(item_aux_role_ids) < len(item_aux):
                    item_aux_role_ids.extend([0] * (len(item_aux) - len(item_aux_role_ids)))
                aux_counts.append(len(item_aux))
                aux_components.extend(item_aux)
                aux_owner.extend([owner_idx] * len(item_aux))
                aux_role_ids.extend(item_aux_role_ids[: len(item_aux)])
            out["aux_counts"] = torch.tensor(aux_counts, dtype=torch.long)
            out["aux_owner"] = torch.tensor(aux_owner, dtype=torch.long)
            out["aux_role_ids"] = torch.tensor(aux_role_ids, dtype=torch.long)
            out["aux_batch"] = self.molecule_collator(aux_components, mode="finetune") if aux_components else {}
            graph_samples = [
                {
                    "reactants_features": item.get("reactant_component_dicts", [item["reactant_dict"]]),
                    "products_features": item.get("product_component_dicts", [item["product_dict"]]),
                    **(
                        {
                            "context_features": item.get("aux_component_dicts", []),
                            "context_role_ids": item.get("aux_role_ids", []),
                        }
                        if self.aux_input_mode == "context_node"
                        else {}
                    ),
                }
                for item in valid_batch
            ]
            graph_batch = self.hypergraph_collator(graph_samples) if self.hypergraph_collator is not None else {}
            if graph_batch and self.aux_input_mode == "edge":
                graph_batch["edge_aux_batch"] = out["aux_batch"]
                graph_batch["edge_aux_owner"] = out["aux_owner"]
                graph_batch["edge_aux_role_ids"] = out["aux_role_ids"]
            out["reaction_graph_batch"] = graph_batch
        else:
            raise ValueError(f"Unsupported downstream task: {self.task}")
        return out


class YieldRankingCollator:
    """Collate pairwise yield-ranking samples into a flattened yield batch."""

    def __init__(self):
        self.yield_collator = ReactionPairCollator(task="yield")

    def __call__(self, batch: List[Optional[Dict]]) -> Dict:
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return {}
        candidates = []
        for item in valid_batch:
            candidates.append(item["better"])
            candidates.append(item["worse"])
        out = self.yield_collator(candidates)
        out["pair_count"] = len(valid_batch)
        out["yield_diff"] = torch.tensor([item["yield_diff"] for item in valid_batch], dtype=torch.float32)
        out["pair_group"] = [item.get("group", "") for item in valid_batch]
        return out
