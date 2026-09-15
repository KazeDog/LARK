# reaction_dataset.py
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
	
from hypermol.data.atom_mapping import has_atom_mapping_markers
from hypermol.data.be_matrix import (
    audit_reaction_be_matrices,
    compute_be_matrix,
    compute_reaction_delta_be,
    sum_be_matrices,
    validate_aromatic_mode,
)
from hypermol.data.molecule_store import open_molecule_store, smiles_key_bytes
from hypermol.data.preprocess import CompoundKit

# 假设您的特征配置字典在这里或从外部导入
# from features_config import atom_masked_features, atom_float_masked_features
ATOM_MASKED_FEATURES = {
    "atomic_num", "valence_out_shell", "formal_charge", "num_radical_e",
    "hybridization", "degree", "total_numHs", "explicit_valence", "implicit_valence"
}
ATOM_FLOAT_MASKED_FEATURES = {"van_der_waals_radis", "partial_charge"}

REACTION_OBJECTIVES = {
    "reaction_reconstruction",
    "product_to_reactant",
    "reactant_to_product",
    "bidirectional_single_side",
    "bidirectional",
}
DUAL_VIEW_OBJECTIVES = {"bidirectional_single_side", "bidirectional"}
CORRUPTION_STRATEGIES = {"target_local", "random_atoms", "none"}
PAIR_MASK_STRATEGIES = {"corruption_local", "full_input_canvas"}
PAIR_PRIOR_MODES = {"combined", "input_side", "none"}
CONTEXT_MISSING_POLICIES = {"error", "skip_component", "drop_sample"}


def _copy_mol_dict(mol_dict: dict) -> dict:
    """Copy mutable feature payloads before applying online augmentation.

    LMDB reads deserialize a fresh dictionary, but the pickle-backed molecule
    store returns cached objects.  Copy-on-read prevents one sample/epoch's
    masking from mutating the cached molecule used by later reads.
    """

    out = {}
    for key, value in mol_dict.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


# 假设您的Collator在一个单独的文件中
# from data_collator import UnifiedCollator 

def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def get_smiles_key(smiles_string: str) -> bytes:
    """计算SMILES字符串的SHA256哈希值，并返回bytes类型的键。"""
    return smiles_key_bytes(smiles_string)


class ReactionDataset(Dataset):
    """
    一个统一的数据集，能够同时为【自监督预训练】和【多种下游微调任务】
    （如正向/逆向合成、反应分类）准备数据。
    """

    def __init__(
        self,
        split_csv: str = None,
        molecule_db: str = None,
        mode: str = "pretrain",
        task: str = None,
        target: str = None,
        csv_path: str = None,
        db_path: str = None,
        # 预训练专用参数
        mask_prob: float = 0.15,
        mask_idx: int = 120,
        vocab_size: int = 121,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        neighbor_mask_prob: float = 0.3,
        aromatic_mode: str = "aromatic_1p5",
        be_audit_tolerance: float = 1e-4,
        return_be_audit: bool = False,
        condition_enabled: bool = False,
        condition_column: str = "condition_vector",
        condition_dim: int = 514,
        unmapped_components_column: str = "unmapped_components",
        context_enabled: bool = False,
        context_column: str = "unmapped_components",
        context_molecule_db: str | None = None,
        context_max_components: int = 0,
        context_missing_policy: str = "error",
        context_mask_prob: float | None = None,
        objective: str = "reaction_reconstruction",
        corruption_strategy: str | None = None,
        train_pair_mask_strategy: str | None = None,
        eval_pair_mask_strategy: str | None = None,
        pair_prior_mode: str | None = None,
        split: str = "",
        augmentation_seed: int = 42,
        deterministic_eval: bool = True,
        local_canvas: bool | None = None,
        load_all_columns: bool = True,
    ):
        """
        初始化统一的数据集。
        Args:
            split_csv (str): CSV文件路径。
            molecule_db (str): LMDB数据库路径。
            mode (str): 'pretrain' 或 'finetune'。
            task (str, optional): 'pretrain' 或 'retro'/'retrosynthesis'。
            target (str, optional): CSV中作为简单标签的列名。
            **kwargs: 所有预训练相关的参数。
        """
        if split_csv is None:
            split_csv = csv_path
        if molecule_db is None:
            molecule_db = db_path
        if split_csv is None or molecule_db is None:
            raise ValueError("ReactionDataset requires split_csv and molecule_db.")

        if task == "retro":
            task = "retrosynthesis"

        self.split_path = split_csv
        self.condition_enabled = bool(condition_enabled)
        self.condition_column = condition_column
        self.condition_dim = int(condition_dim)
        self.unmapped_components_column = unmapped_components_column
        self.context_enabled = bool(context_enabled)
        self.context_column = str(context_column)
        self.context_db_path = str(context_molecule_db) if context_molecule_db else None
        self.context_max_components = int(context_max_components)
        self.context_missing_policy = str(context_missing_policy).lower()
        self.context_mask_prob = float(mask_prob if context_mask_prob is None else context_mask_prob)
        if self.context_enabled and not self.context_db_path:
            raise ValueError("context_molecule_db is required when context_enabled=True.")
        if self.context_max_components < 0:
            raise ValueError("context_max_components must be >= 0; use 0 for no limit.")
        if self.context_missing_policy not in CONTEXT_MISSING_POLICIES:
            raise ValueError(
                f"context_missing_policy must be one of {sorted(CONTEXT_MISSING_POLICIES)}."
            )
        if not 0.0 <= self.context_mask_prob <= 1.0:
            raise ValueError("context_mask_prob must be between 0 and 1.")
        self.target_column_name = target
        self.load_all_columns = bool(load_all_columns)
        self.df = self._read_split_table(split_csv)
        if self.context_enabled and self.context_column not in self.df.columns:
            raise ValueError(
                f"Missing required context column {self.context_column!r} in {split_csv}."
            )
        self.db_path = molecule_db
        self.store = None
        self.context_store = None

        self.objective = str(objective).lower()
        if self.objective not in REACTION_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(REACTION_OBJECTIVES)}.")
        if self.objective == "bidirectional":
            self.objective = "bidirectional_single_side"
        directional = self.objective != "reaction_reconstruction"
        self.corruption_strategy = str(
            corruption_strategy if corruption_strategy is not None else ("random_atoms" if directional else "target_local")
        ).lower()
        self.train_pair_mask_strategy = str(
            train_pair_mask_strategy
            if train_pair_mask_strategy is not None
            else ("full_input_canvas" if directional else "corruption_local")
        ).lower()
        self.eval_pair_mask_strategy = str(
            eval_pair_mask_strategy
            if eval_pair_mask_strategy is not None
            else ("full_input_canvas" if directional else self.train_pair_mask_strategy)
        ).lower()
        self.pair_prior_mode = str(
            pair_prior_mode if pair_prior_mode is not None else ("none" if directional else "combined")
        ).lower()
        self.local_canvas = bool(directional if local_canvas is None else local_canvas)
        inferred_split = Path(split_csv).stem.lower()
        self.split = str(split or inferred_split or "train").lower()
        self.augmentation_seed = int(augmentation_seed)
        self.deterministic_eval = bool(deterministic_eval)

        if self.corruption_strategy not in CORRUPTION_STRATEGIES:
            raise ValueError(f"corruption_strategy must be one of {sorted(CORRUPTION_STRATEGIES)}.")
        if self.train_pair_mask_strategy not in PAIR_MASK_STRATEGIES:
            raise ValueError(f"train_pair_mask_strategy must be one of {sorted(PAIR_MASK_STRATEGIES)}.")
        if self.eval_pair_mask_strategy not in PAIR_MASK_STRATEGIES:
            raise ValueError(f"eval_pair_mask_strategy must be one of {sorted(PAIR_MASK_STRATEGIES)}.")
        if self.pair_prior_mode not in PAIR_PRIOR_MODES:
            raise ValueError(f"pair_prior_mode must be one of {sorted(PAIR_PRIOR_MODES)}.")
        if directional and self.pair_prior_mode == "combined":
            raise ValueError("pair_prior_mode='combined' is not allowed for one-side directional objectives.")
        if directional and self.corruption_strategy == "target_local":
            raise ValueError("Directional objectives require target-independent corruption; use 'random_atoms' or 'none'.")
        if directional and not self.local_canvas:
            raise ValueError("Directional objectives require local_canvas=True to avoid target-shape leakage.")
        if self.objective == "reaction_reconstruction" and self.pair_prior_mode == "input_side":
            raise ValueError("pair_prior_mode='input_side' requires a one-side directional objective.")

        self.mode = mode
        self.task = task

        if self.mode not in ['pretrain', 'finetune']:
            raise ValueError("mode 必须是 'pretrain' 或 'finetune'")

        # 将所有与masking相关的参数保存为实例属性
        self.mask_prob = mask_prob
        self.mask_idx = mask_idx
        self.vocab_size = vocab_size
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.neighbor_mask_prob = neighbor_mask_prob
        self.aromatic_mode = validate_aromatic_mode(aromatic_mode)
        self.be_audit_tolerance = be_audit_tolerance
        self.return_be_audit = return_be_audit

        weights = np.ones(self.vocab_size)
        if self.vocab_size > 0:
            weights[0] = 0
        self.random_token_weights = weights / weights.sum()

    def _read_split_table(self, split_path: str) -> pd.DataFrame:
        path = Path(split_path)
        columns = None
        if not self.load_all_columns:
            columns = ["reactant_smiles", "prod_smiles"]
            if self.target_column_name:
                columns.append(self.target_column_name)
            if self.condition_enabled:
                columns.append(self.condition_column)
            if self.context_enabled:
                columns.append(self.context_column)
            columns = list(dict.fromkeys(columns))
        if path.suffix.lower() in {".parquet", ".pq"}:
            if columns is not None:
                try:
                    import pyarrow.parquet as pq

                    available = set(pq.ParquetFile(path).schema_arrow.names)
                    required = {"reactant_smiles", "prod_smiles"}
                    if self.target_column_name:
                        required.add(self.target_column_name)
                    if self.context_enabled:
                        required.add(self.context_column)
                    missing_required = required - available
                    if missing_required:
                        raise ValueError(f"Missing required reaction columns: {sorted(missing_required)}")
                    columns = [column for column in columns if column in available]
                except ImportError:
                    pass
            return pd.read_parquet(path, columns=columns)
        if columns is None:
            return pd.read_csv(path)
        requested = set(columns)
        return pd.read_csv(path, usecols=lambda name: name in requested)

    def _read_condition_from_row(self, row: pd.Series) -> tuple[np.ndarray, bool]:
        if not self.condition_enabled or self.condition_column not in row:
            return np.zeros(self.condition_dim, dtype=np.float32), False

        value = row.get(self.condition_column)
        if value is None:
            return np.zeros(self.condition_dim, dtype=np.float32), False
        try:
            if pd.isna(value):
                return np.zeros(self.condition_dim, dtype=np.float32), False
        except (TypeError, ValueError):
            pass

        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return np.zeros(self.condition_dim, dtype=np.float32), False
        if vector.size != self.condition_dim:
            raise ValueError(
                f"Expected {self.condition_column} length {self.condition_dim}, got {vector.size} in {self.split_path}."
            )
        return vector, True

    def __getstate__(self):
        state = self.__dict__.copy()
        state["store"] = None
        state["context_store"] = None
        return state

    def split_view(self, split: str) -> "ReactionDataset":
        """Return a lightweight view sharing immutable tabular data.

        The molecule store handle remains process/view local, while the large
        reaction table is shared.  This is used to give validation a fixed RNG
        namespace without loading ORDerly twice.
        """

        view = copy.copy(self)
        view.store = None
        view.context_store = None
        view.split = str(split).lower()
        return view

    def _init_store(self):
        if self.store is None:
            self.store = open_molecule_store(self.db_path)
        if self.context_enabled and self.context_store is None:
            self.context_store = open_molecule_store(self.context_db_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        self._init_store()
        row = self.df.iloc[int(idx)]
        reaction_smiles = {
            "reactants": clean_text(row["reactant_smiles"]).split('.'),
            "products": clean_text(row["prod_smiles"]).split('.'),
        }

        # 1. 加载一个完整的“干净”反应样本
        reactants_dicts, products_dicts = [], []
        try:
            for smi in reaction_smiles['reactants']:
                mol_dict = _copy_mol_dict(self.store.get_mol_dict(smi))
                self._ensure_be_matrix(mol_dict, smi)
                reactants_dicts.append(mol_dict)
            for smi in reaction_smiles['products']:
                mol_dict = _copy_mol_dict(self.store.get_mol_dict(smi))
                self._ensure_be_matrix(mol_dict, smi)
                products_dicts.append(mol_dict)
        except (TypeError, KeyError, ValueError) as e:
            print(f"警告: 无法加载反应索引 {idx} 的数据. 错误: {e}. 跳过此样本。")
            return {}

        # 2. 准备基础样本字典，包含所有可能需要的信息
        sample = self._prepare_base_sample(reactants_dicts, products_dicts, reaction_smiles)
        if not sample: return {}

        if self.context_enabled:
            context_result = self._load_context_features(row, idx)
            if context_result is None:
                return {}
            context_features, skipped_context = context_result
            sample["context_features"] = context_features
            sample["unmapped_components"] = clean_text(row.get(self.context_column, ""))
            sample["context_skipped_components"] = skipped_context
        else:
            sample["context_features"] = []

        # 添加可选的简单标签
        if self.target_column_name and self.target_column_name in row:
            label_val = row.get(self.target_column_name)
            if hasattr(label_val, 'values'):
                label_val = label_val.values[0]
            sample["label"] = label_val
        if self.condition_enabled:
            condition_vector, has_condition = self._read_condition_from_row(row)
            sample["condition_vector"] = torch.from_numpy(condition_vector).float()
            sample["has_condition_vector"] = torch.tensor(has_condition, dtype=torch.bool)

        # 3. 根据模式决定是否进行数据增强
        if self.mode == 'pretrain':
            if self.objective == "bidirectional_single_side":
                views = []
                for direction_id, objective in enumerate(
                    ("product_to_reactant", "reactant_to_product")
                ):
                    view = self._clone_pretrain_sample(sample)
                    view = self._prepare_pretrain_objective(view, objective=objective)
                    view["view_direction"] = objective
                    view["view_direction_id"] = direction_id
                    rng = self._augmentation_rng(idx, namespace=direction_id + 1)
                    views.append(self._apply_pretrain_augmentation(view, rng=rng))
                return {
                    "pretrain_views": views,
                    "dual_view_objective": "bidirectional_single_side",
                }
            sample = self._prepare_pretrain_objective(sample)
            return self._apply_pretrain_augmentation(sample, rng=self._augmentation_rng(idx))
        else:  # finetune 模式
            # a. 准备微调任务的【输入】
            if self.task == 'forwardsynthesis':
                sample["input_features"] = sample['reactants_features']
            elif self.task == 'retrosynthesis':
                sample["input_features"] = sample['products_features']
            elif self.task == 'classification':
                sample["input_features"] = sample['reactants_features'] + sample['products_features']

            # b. 确保所有原子特征都是Tensor
            if "input_features" in sample:
                for d in sample['input_features']:
                    if 'atomic_num' in d:
                        if not torch.is_tensor(d['atomic_num']): d['atomic_num'] = torch.from_numpy(d['atomic_num'])
                        d['atomic_num'] = d['atomic_num'].long()

            return sample

    def _load_context_features(
        self,
        row: pd.Series,
        idx: int,
    ) -> tuple[list[dict], list[str]] | None:
        """Load entirely unmapped components from their independent molecule store."""

        raw = clean_text(row.get(self.context_column, ""))
        components = [part.strip() for part in raw.split(".") if part.strip()]
        if self.context_max_components > 0:
            components = components[: self.context_max_components]

        features: list[dict] = []
        skipped: list[str] = []
        for smiles in components:
            if has_atom_mapping_markers(smiles):
                message = (
                    f"Context column {self.context_column!r} contains atom mapping at "
                    f"reaction index {idx}: {smiles!r}. Context atoms must stay outside the Delta-BE canvas."
                )
                if self.context_missing_policy == "error":
                    raise ValueError(message)
                if self.context_missing_policy == "drop_sample":
                    return None
                skipped.append(smiles)
                continue
            try:
                mol_dict = _copy_mol_dict(self.context_store.get_mol_dict(smiles))
            except (TypeError, KeyError, ValueError) as exc:
                if self.context_missing_policy == "error":
                    raise KeyError(
                        f"Unable to load unmapped context component {smiles!r} at reaction index {idx} "
                        f"from {self.context_db_path}."
                    ) from exc
                if self.context_missing_policy == "drop_sample":
                    return None
                skipped.append(smiles)
                continue

            # This is a hard boundary: context molecules can participate in
            # attention and representation learning, but never in Delta-BE.
            if dict(mol_dict.get("map_list") or {}):
                if self.context_missing_policy == "error":
                    raise ValueError(
                        f"Context molecule store contains mapped atoms for {smiles!r}; rebuild the unmapped store."
                    )
                if self.context_missing_policy == "drop_sample":
                    return None
                skipped.append(smiles)
                continue
            mol_dict["map_list"] = {}
            features.append(mol_dict)
        return features, skipped

    @staticmethod
    def _clone_pretrain_sample(sample: dict) -> dict:
        """Clone mutable feature payloads so the two view corruptions are independent."""

        cloned = {}
        for key, value in sample.items():
            if key in {"reactants_features", "products_features", "context_features"}:
                cloned[key] = [_copy_mol_dict(mol_dict) for mol_dict in value]
            elif torch.is_tensor(value):
                cloned[key] = value.clone()
            elif isinstance(value, np.ndarray):
                cloned[key] = value.copy()
            elif isinstance(value, list):
                cloned[key] = list(value)
            elif isinstance(value, dict):
                cloned[key] = dict(value)
            else:
                cloned[key] = value
        return cloned

    def _ensure_be_matrix(self, mol_dict: dict, fallback_smiles: str | None = None) -> None:
        """Ensure the molecule dictionary has a BE matrix using the configured aromatic mode."""
        has_cached_default = self.aromatic_mode == "aromatic_1p5" and mol_dict.get("be_matrix") is not None
        if has_cached_default:
            mol_dict["be_matrix"] = np.asarray(mol_dict["be_matrix"], dtype=np.float32)
            return

        source = mol_dict.get("mol") or mol_dict.get("smiles") or fallback_smiles
        if source is None:
            raise ValueError("Cannot recompute BE matrix without mol or smiles.")
        be_matrix, max_map_num, map_list = compute_be_matrix(source, aromatic_mode=self.aromatic_mode)
        mol_dict["be_matrix"] = be_matrix.astype(np.float32, copy=False)
        mol_dict["max_map_num"] = max_map_num
        mol_dict["map_list"] = map_list

    def _prepare_base_sample(self, reactants_dicts, products_dicts, reaction_smiles):
        """一个内部辅助函数，负责计算所有“干净”的标签和元数据。"""
        all_mol_dicts = reactants_dicts + products_dicts
        if not all_mol_dicts:
            return None

        # a. 计算R矩阵标签
        reactant_be_matrices = [d['be_matrix'] for d in reactants_dicts if 'be_matrix' in d]
        product_be_matrices = [d['be_matrix'] for d in products_dicts if 'be_matrix' in d]
        all_be_matrices = reactant_be_matrices + product_be_matrices
        if not all_be_matrices:
            return None

        max_dim = max(m.shape[0] for m in all_be_matrices)
        be_matrix_reactants_np = sum_be_matrices(reactant_be_matrices, size=max_dim)
        be_matrix_products_np = sum_be_matrices(product_be_matrices, size=max_dim)
        r_matrix_targets_np = compute_reaction_delta_be(reactant_be_matrices, product_be_matrices)

        r_matrix_mask_np = (r_matrix_targets_np != 0)
        be_audit = audit_reaction_be_matrices(
            reactant_be_matrices,
            product_be_matrices,
            reactants_dicts,
            products_dicts,
            tolerance=self.be_audit_tolerance,
        )

        # b. 计算原子身份标签
        atom_identity_targets_np = np.full(max_dim, -100, dtype=np.int64)
        for mol_dict in reactants_dicts:
            if 'map_list' in mol_dict and 'atomic_num' in mol_dict:
                features = mol_dict['atomic_num']
                atoms = features.numpy().flatten() if torch.is_tensor(features) else features.flatten()
                for map_num, local_idx in mol_dict['map_list'].items():
                    if (map_num - 1) < max_dim and local_idx < len(atoms):
                        atom_identity_targets_np[map_num - 1] = atoms[local_idx]

        sample = {
            "reactants_features": reactants_dicts,
            "products_features": products_dicts,
            "reactant_be_matrix": torch.from_numpy(be_matrix_reactants_np),
            "product_be_matrix": torch.from_numpy(be_matrix_products_np),
            "r_matrix_targets": torch.from_numpy(r_matrix_targets_np),
            "r_matrix_mask": torch.from_numpy(r_matrix_mask_np),
            "atom_identity_targets": torch.from_numpy(atom_identity_targets_np),
            "raw_reactant_smiles": reaction_smiles['reactants'],
            "raw_product_smiles": reaction_smiles['products'],
        }
        if self.return_be_audit:
            sample["be_audit"] = be_audit.to_dict()
        return sample

    @staticmethod
    def _project_canvas_matrix(matrix: np.ndarray, canvas_map_ids: list[int], local_canvas: bool) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if not local_canvas:
            return matrix.copy()
        if not canvas_map_ids:
            return np.zeros((0, 0), dtype=np.float32)
        indices = np.asarray([map_num - 1 for map_num in canvas_map_ids], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= matrix.shape[0]):
            raise ValueError("canvas_map_ids are outside the BE/Delta-BE matrix coordinate system.")
        return matrix[np.ix_(indices, indices)].astype(np.float32, copy=False)

    @staticmethod
    def _canvas_index_lists(
        molecule_dicts: list[dict],
        map_to_canvas: dict[int, int],
    ) -> list[list[int]]:
        """Return local-atom-index -> zero-based reaction-canvas-index lists."""

        out: list[list[int]] = []
        for mol_dict in molecule_dicts:
            atom_count = int(len(mol_dict.get("atomic_num", [])))
            indices = [-1] * atom_count
            for map_num, local_atom_idx in dict(mol_dict.get("map_list") or {}).items():
                canvas_idx = map_to_canvas.get(int(map_num))
                local_atom_idx = int(local_atom_idx)
                if canvas_idx is not None and 0 <= local_atom_idx < atom_count:
                    indices[local_atom_idx] = int(canvas_idx)
            out.append(indices)
        return out

    def _prepare_pretrain_objective(self, sample: dict, objective: str | None = None) -> dict:
        """Normalize reconstruction and directional objectives to one batch contract."""

        reactants = sample["reactants_features"]
        products = sample["products_features"]
        context = sample.get("context_features", [])
        objective = str(objective or self.objective).lower()
        if objective == "product_to_reactant":
            core_input_features = products
            core_input_roles = ["product"] * len(products)
            target = -sample["r_matrix_targets"].detach().cpu().numpy().astype(np.float32, copy=False)
            input_side_be = sample["product_be_matrix"].detach().cpu().numpy()
        elif objective == "reactant_to_product":
            core_input_features = reactants
            core_input_roles = ["reactant"] * len(reactants)
            target = sample["r_matrix_targets"].detach().cpu().numpy().astype(np.float32, copy=False)
            input_side_be = sample["reactant_be_matrix"].detach().cpu().numpy()
        else:
            core_input_features = reactants + products
            core_input_roles = ["reactant"] * len(reactants) + ["product"] * len(products)
            target = sample["r_matrix_targets"].detach().cpu().numpy().astype(np.float32, copy=False)
            input_side_be = (
                sample["reactant_be_matrix"].detach().cpu().numpy()
                + sample["product_be_matrix"].detach().cpu().numpy()
            )

        model_input_features = core_input_features + context if context else core_input_features
        model_input_roles = core_input_roles + ["context"] * len(context)

        input_map_ids = sorted(
            {
                int(map_num)
                for mol_dict in model_input_features
                for map_num in dict(mol_dict.get("map_list") or {}).keys()
                if int(map_num) > 0
            }
        )
        if self.local_canvas:
            canvas_map_ids = input_map_ids
            map_to_canvas = {map_num: idx for idx, map_num in enumerate(canvas_map_ids)}
        else:
            canvas_map_ids = list(range(1, int(target.shape[0]) + 1))
            map_to_canvas = {
                map_num: map_num - 1
                for map_num in input_map_ids
                if 0 < map_num <= int(target.shape[0])
            }

        global_target = np.asarray(target, dtype=np.float32)
        target = self._project_canvas_matrix(global_target, canvas_map_ids, self.local_canvas)
        input_side_be = self._project_canvas_matrix(input_side_be, canvas_map_ids, self.local_canvas)
        canvas_index_lists = self._canvas_index_lists(model_input_features, map_to_canvas)
        canvas_dim = int(target.shape[0])
        valid_atoms = np.zeros(canvas_dim, dtype=np.bool_)
        for indices in canvas_index_lists:
            for canvas_idx in indices:
                if 0 <= int(canvas_idx) < canvas_dim:
                    valid_atoms[int(canvas_idx)] = True
        valid_pair_mask = valid_atoms[:, None] & valid_atoms[None, :]

        def changed_counts(matrix: np.ndarray) -> tuple[int, int]:
            changed = np.abs(matrix) > float(getattr(self, "be_audit_tolerance", 1e-4))
            offdiag = int(np.triu(changed, k=1).sum())
            diagonal = int(np.diag(changed).sum())
            return offdiag, diagonal

        global_offdiag, global_diagonal = changed_counts(global_target)
        retained_offdiag, retained_diagonal = changed_counts(target)
        target_projection_audit = {
            "global_canvas_dim": int(global_target.shape[0]),
            "input_canvas_dim": int(target.shape[0]),
            "global_changed_offdiag_pairs": global_offdiag,
            "retained_changed_offdiag_pairs": retained_offdiag,
            "dropped_changed_offdiag_pairs": max(global_offdiag - retained_offdiag, 0),
            "global_changed_diagonal_entries": global_diagonal,
            "retained_changed_diagonal_entries": retained_diagonal,
            "dropped_changed_diagonal_entries": max(global_diagonal - retained_diagonal, 0),
        }

        if self.pair_prior_mode == "none":
            pair_prior = np.zeros_like(target, dtype=np.float32)
            pair_prior_observed_mask = np.zeros_like(valid_pair_mask, dtype=np.bool_)
        elif self.pair_prior_mode == "input_side":
            pair_prior = input_side_be.astype(np.float32, copy=True)
            pair_prior_observed_mask = valid_pair_mask.copy()
        else:
            combined_be = (
                sample["reactant_be_matrix"].detach().cpu().numpy()
                + sample["product_be_matrix"].detach().cpu().numpy()
            )
            pair_prior = self._project_canvas_matrix(combined_be, canvas_map_ids, self.local_canvas)
            pair_prior_observed_mask = valid_pair_mask.copy()

        sample.update(
            {
                "objective": objective,
                "model_input_features": model_input_features,
                "model_input_roles": model_input_roles,
                "canvas_index_lists": canvas_index_lists,
                "canvas_map_ids": canvas_map_ids,
                "num_canvas_atoms": canvas_dim,
                "target_projection_audit": target_projection_audit,
                "delta_be_targets": torch.from_numpy(target.copy()),
                "r_matrix_targets": torch.from_numpy(target.copy()),  # legacy alias
                "valid_pair_mask": torch.from_numpy(valid_pair_mask),
                "valid_mask": torch.from_numpy(valid_pair_mask.copy()),
                "r_matrix_valid_mask": torch.from_numpy(valid_pair_mask.copy()),
                "pair_prior": torch.from_numpy(pair_prior),
                "pair_prior_observed_mask": torch.from_numpy(pair_prior_observed_mask),
                # Used only by the Dataset's target-independent neighbor expansion;
                # the collator intentionally does not forward this private field.
                "_corruption_adjacency": input_side_be.astype(np.float32, copy=True),
            }
        )
        return sample

    def _augmentation_rng(self, idx: int, namespace: int = 0):
        if not self.deterministic_eval or self.split not in {"valid", "val", "validation", "test", "eval"}:
            return None
        split_offsets = {"valid": 1_000_000, "val": 1_000_000, "validation": 1_000_000, "test": 2_000_000, "eval": 3_000_000}
        return np.random.default_rng(
            self.augmentation_seed
            + split_offsets[self.split]
            + int(namespace) * 10_000_000
            + int(idx)
        )

    def _mask_molecule_atoms(self, mol_dict: dict, local_mask_indices: list[int], rng) -> None:
        """Apply the shared BERT-style atom corruption to selected local atoms."""

        if not local_mask_indices:
            return
        num_atoms = int(len(mol_dict.get("atomic_num", [])))
        local_mask_indices = sorted({int(index) for index in local_mask_indices if 0 <= int(index) < num_atoms})
        if not local_mask_indices:
            return

        original_atoms = (
            mol_dict["atomic_num"].detach().cpu().numpy()
            if torch.is_tensor(mol_dict["atomic_num"])
            else np.asarray(mol_dict["atomic_num"])
        ).reshape(-1)
        mam_targets = np.full(num_atoms, -1, dtype=np.int64)
        mam_targets[local_mask_indices] = original_atoms[local_mask_indices]
        mol_dict["mam_targets"] = torch.from_numpy(mam_targets)

        augmented_atoms = np.array(original_atoms, copy=True)
        dice_roll = rng.random(len(local_mask_indices))
        random_token_thresh = self.leave_unmasked_prob + self.random_token_prob
        for position, local_idx in enumerate(local_mask_indices):
            roll = float(dice_roll[position])
            if roll < self.leave_unmasked_prob:
                continue
            if roll < random_token_thresh:
                augmented_atoms[local_idx] = rng.choice(
                    self.vocab_size,
                    p=self.random_token_weights,
                )
            else:
                augmented_atoms[local_idx] = self.mask_idx
        mol_dict["atomic_num"] = torch.from_numpy(augmented_atoms).long()

        for key in CompoundKit.atom_vocab_dict.keys():
            if key == "atomic_num" or key not in mol_dict:
                continue
            values = mol_dict[key].detach().cpu().numpy() if torch.is_tensor(mol_dict[key]) else np.asarray(mol_dict[key])
            values = np.array(values, copy=True)
            values.flat[local_mask_indices] = -1
            mol_dict[key] = torch.from_numpy(values).long()

        for key in CompoundKit.atom_float_names:
            if key not in mol_dict:
                continue
            values = mol_dict[key].detach().cpu().numpy() if torch.is_tensor(mol_dict[key]) else np.asarray(mol_dict[key])
            values = np.array(values, copy=True)
            values.flat[local_mask_indices] = -1.0
            mol_dict[key] = torch.from_numpy(values).float()

    def _apply_pretrain_augmentation(self, sample: dict, rng=None) -> dict:
        # """一个专门负责执行预训练数据增强（协同Masking）的内部函数。"""
        # all_mol_dicts = sample['reactants_features'] + sample['products_features']
        # original_r_matrix_np = sample['r_matrix_targets'].numpy()
        #
        # # 1. 做出统一的Masking决策
        # total_map_indices = original_r_matrix_np.shape[0]
        # num_to_mask = int(np.ceil(total_map_indices * self.mask_prob))
        # if num_to_mask == 0 and total_map_indices > 0: num_to_mask = 1
        # masked_indices_global = np.random.choice(total_map_indices, num_to_mask, replace=False)
        #
        # # 2. 创建 R 矩阵的【损失掩码】
        # original_r_matrix_np = sample['r_matrix_targets'].numpy()
        # r_matrix_mask_np_pretrain = np.zeros_like(original_r_matrix_np, dtype=np.bool_)
        # if masked_indices_global.size > 0:
        #     r_matrix_mask_np_pretrain[masked_indices_global, :] = True
        #     r_matrix_mask_np_pretrain[:, masked_indices_global] = True
        # # 用预训练的mask【覆盖】之前为微调准备的默认mask
        # sample['r_matrix_mask'] = torch.from_numpy(r_matrix_mask_np_pretrain)
        #
        # # 3. 逐个分子应用协同Masking
        # for mol_dict in all_mol_dicts:
        #     num_atoms_in_mol = len(mol_dict.get('atomic_num', []))
        #     mol_dict['mam_targets'] = torch.full((num_atoms_in_mol,), -1, dtype=torch.long)
        #
        #     local_mask_indices = []
        #     if 'map_list' in mol_dict:
        #         for map_num, local_idx in mol_dict['map_list'].items():
        #             if (map_num - 1) in masked_indices_global and local_idx < num_atoms_in_mol:
        #                 local_mask_indices.append(local_idx)
        #
        #     if not local_mask_indices:
        #         continue

        rng = np.random if rng is None else rng
        all_mol_dicts = sample.get("model_input_features") or (
            sample['reactants_features'] + sample['products_features']
        )
        original_r_matrix_np = sample['r_matrix_targets'].detach().cpu().numpy()
        canvas_dim = int(original_r_matrix_np.shape[0])
        canvas_index_lists = sample.get("canvas_index_lists")
        if canvas_index_lists is None:
            canvas_index_lists = []
            for mol_dict in all_mol_dicts:
                atom_count = int(len(mol_dict.get("atomic_num", [])))
                indices = [-1] * atom_count
                for map_num, local_idx in dict(mol_dict.get("map_list") or {}).items():
                    if 0 < int(map_num) <= canvas_dim and 0 <= int(local_idx) < atom_count:
                        indices[int(local_idx)] = int(map_num) - 1
                canvas_index_lists.append(indices)

        mapped_atom_indices = sorted(
            {
                int(canvas_idx)
                for indices in canvas_index_lists
                for canvas_idx in indices
                if 0 <= int(canvas_idx) < canvas_dim
            }
        )
        corruption_strategy = getattr(self, "corruption_strategy", "target_local")

        # Select corruption locations.  random_atoms is deliberately independent
        # of Delta-BE; target_local preserves the historical reconstruction task.
        core_atoms = np.asarray([], dtype=np.int64)
        if corruption_strategy == "target_local":
            non_diagonal_changes = np.triu(original_r_matrix_np, k=1)
            changed_bonds_indices = np.argwhere(np.abs(non_diagonal_changes) > 0.1)
            if len(changed_bonds_indices) > 0:
                num_bonds_to_mask = max(1, int(np.ceil(len(changed_bonds_indices) * self.mask_prob)))
                num_bonds_to_mask = min(num_bonds_to_mask, len(changed_bonds_indices))
                chosen = rng.choice(len(changed_bonds_indices), num_bonds_to_mask, replace=False)
                core_atoms = np.unique(changed_bonds_indices[chosen].flatten())
        if corruption_strategy == "random_atoms" or (
            corruption_strategy == "target_local" and core_atoms.size == 0
        ):
            if mapped_atom_indices:
                num_atoms_to_mask = max(1, int(np.ceil(len(mapped_atom_indices) * self.mask_prob)))
                num_atoms_to_mask = min(num_atoms_to_mask, len(mapped_atom_indices))
                core_atoms = np.asarray(
                    rng.choice(mapped_atom_indices, num_atoms_to_mask, replace=False),
                    dtype=np.int64,
                )

        adjacency = sample.pop("_corruption_adjacency", None)
        if adjacency is None:
            adjacency = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
            for mol_dict in all_mol_dicts:
                matrix = mol_dict.get("be_matrix")
                if matrix is not None:
                    matrix = np.asarray(matrix, dtype=np.float32)
                    rows = min(canvas_dim, matrix.shape[0])
                    cols = min(canvas_dim, matrix.shape[1])
                    adjacency[:rows, :cols] += matrix[:rows, :cols]
        else:
            adjacency = np.asarray(adjacency, dtype=np.float32)

        neighborhood_atoms = set(int(idx) for idx in core_atoms)
        for atom_idx in core_atoms:
            atom_idx = int(atom_idx)
            if 0 <= atom_idx < adjacency.shape[0]:
                neighborhood_atoms.update(int(idx) for idx in np.where(adjacency[atom_idx, :] > 0.5)[0])

        final_masked_atoms = set(int(idx) for idx in core_atoms)
        for neighbor in neighborhood_atoms:
            if neighbor not in final_masked_atoms and float(rng.random()) < self.neighbor_mask_prob:
                final_masked_atoms.add(int(neighbor))
        masked_indices_global = np.asarray(
            sorted(idx for idx in final_masked_atoms if 0 <= idx < canvas_dim),
            dtype=np.int64,
        )

        corruption_atom_mask = np.zeros(canvas_dim, dtype=np.bool_)
        if masked_indices_global.size > 0:
            corruption_atom_mask[masked_indices_global] = True
        corruption_pair_mask = corruption_atom_mask[:, None] & corruption_atom_mask[None, :]
        valid_pair_mask = sample.get("valid_pair_mask")
        if torch.is_tensor(valid_pair_mask):
            valid_pair_mask = valid_pair_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
        elif valid_pair_mask is None:
            valid_pair_mask = np.ones_like(original_r_matrix_np, dtype=np.bool_)
        else:
            valid_pair_mask = np.asarray(valid_pair_mask, dtype=np.bool_)

        def build_pair_mask(strategy: str) -> np.ndarray:
            if strategy == "full_input_canvas":
                return valid_pair_mask.copy()
            return corruption_pair_mask & valid_pair_mask

        train_strategy = getattr(self, "train_pair_mask_strategy", "corruption_local")
        eval_strategy = getattr(self, "eval_pair_mask_strategy", train_strategy)
        train_mask = build_pair_mask(train_strategy)
        eval_mask = build_pair_mask(eval_strategy)
        sample.update(
            {
                "corruption_atom_mask": torch.from_numpy(corruption_atom_mask),
                "r_matrix_train_mask": torch.from_numpy(train_mask),
                "r_matrix_eval_mask": torch.from_numpy(eval_mask),
                "train_mask": torch.from_numpy(train_mask.copy()),
                "eval_mask": torch.from_numpy(eval_mask.copy()),
                "valid_mask": torch.from_numpy(valid_pair_mask.copy()),
                "valid_pair_mask": torch.from_numpy(valid_pair_mask.copy()),
                "r_matrix_valid_mask": torch.from_numpy(valid_pair_mask.copy()),
                "r_matrix_mask": torch.from_numpy(train_mask.copy()),  # legacy alias
            }
        )

        # Initialize every exposed molecule, including unmapped context, with
        # an explicit no-target MAM vector before selecting either loss family.
        for mol_dict in all_mol_dicts:
            num_atoms_in_mol = len(mol_dict.get("atomic_num", []))
            mol_dict["mam_targets"] = torch.full((num_atoms_in_mol,), -1, dtype=torch.long)

        # Core corruption follows the mapped local canvas only.
        for mol_dict, atom_to_canvas in zip(all_mol_dicts, canvas_index_lists):
            local_mask_indices = [
                local_idx
                for local_idx, canvas_idx in enumerate(atom_to_canvas)
                if int(canvas_idx) in final_masked_atoms
            ]
            self._mask_molecule_atoms(mol_dict, local_mask_indices, rng)

        # Context corruption is sampled in local atom coordinates and is
        # deliberately disconnected from atom maps and the Delta-BE canvas.
        roles = list(sample.get("model_input_roles", []))
        context_atom_positions = [
            (mol_idx, atom_idx)
            for mol_idx, (mol_dict, role) in enumerate(zip(all_mol_dicts, roles))
            if role == "context"
            for atom_idx in range(len(mol_dict.get("atomic_num", [])))
        ]
        context_mask_prob = float(getattr(self, "context_mask_prob", self.mask_prob))
        if corruption_strategy != "none" and context_atom_positions and context_mask_prob > 0.0:
            num_context_atoms = len(context_atom_positions)
            num_to_mask = min(
                num_context_atoms,
                max(1, int(np.ceil(num_context_atoms * context_mask_prob))),
            )
            chosen_positions = np.asarray(
                rng.choice(num_context_atoms, num_to_mask, replace=False),
                dtype=np.int64,
            ).reshape(-1)
            by_molecule: dict[int, list[int]] = {}
            for position in chosen_positions:
                mol_idx, atom_idx = context_atom_positions[int(position)]
                by_molecule.setdefault(mol_idx, []).append(atom_idx)
            for mol_idx, atom_indices in by_molecule.items():
                self._mask_molecule_atoms(all_mol_dicts[mol_idx], atom_indices, rng)

        return sample


def get_dataloader(csv_path: str, db_path: str, batch_size: int = 1, shuffle: bool = False, mode: str = "pretrain", task: str = None,
                   target: str = None, num_workers: int = 0, **kwargs):
    """
    统一的工厂函数，根据mode和task创建和配置DataLoader。
    """
    dataset = ReactionDataset(split_csv=csv_path, molecule_db=db_path, mode=mode, task=task, target=target, **kwargs)

    # collator = ... (根据mode和task选择正确的Collator)
    # return DataLoader(dataset, ..., collate_fn=collator)

    return dataset
