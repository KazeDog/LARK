"""Prepare MoleculeNet classification/regression datasets for SMILES-only prediction.

The raw tables are read through a strict two-column allowlist.  BACE's
precomputed descriptor columns are therefore never copied into a processed
table, molecule store, batch, or model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import lmdb
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from hypermol.data.molecule_store import smiles_key_bytes
from hypermol.data.preprocess import Compound3DKit, CompoundKit, mol_to_data_pretrain


DATASET_SPECS = {
    "bace": {
        "filename": "bace.csv",
        "smiles_column": "mol",
        "label_columns": ("Class",),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    },
    "bbbp": {
        "filename": "BBBP.csv",
        "smiles_column": "smiles",
        "label_columns": ("p_np",),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    },
    "clintox": {
        "filename": "clintox.csv.gz",
        "smiles_column": "smiles",
        "label_columns": ("FDA_APPROVED", "CT_TOX"),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
    },
    "tox21": {
        "filename": "tox21.csv.gz",
        "smiles_column": "smiles",
        "label_columns": (
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ),
        "ignored_columns": ("mol_id",),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    },
    "toxcast": {
        "filename": "toxcast_data.csv.gz",
        "smiles_column": "smiles",
        "label_columns": None,
        "ignored_columns": (),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
    },
    "sider": {
        "filename": "sider.csv.gz",
        "smiles_column": "smiles",
        "label_columns": None,
        "ignored_columns": (),
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
    },
    "esol": {
        "filename": "delaney-processed.csv",
        "smiles_column": "smiles",
        "label_columns": ("measured log solubility in mols per litre",),
        "task_type": "regression",
        "selection_metric": "rmse",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    },
    "freesolv": {
        "filename": "SAMPL.csv",
        "smiles_column": "smiles",
        "label_columns": ("expt",),
        "task_type": "regression",
        "selection_metric": "rmse",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    },
    "lipophilicity": {
        "filename": "Lipophilicity.csv",
        "smiles_column": "smiles",
        "label_columns": ("exp",),
        "task_type": "regression",
        "selection_metric": "rmse",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    },
    "hiv": {
        "filename": "HIV.csv",
        "smiles_column": "smiles",
        "label_columns": ("HIV_active",),
        "task_type": "classification",
        "selection_metric": "roc_auc",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    },
    "muv": {
        "filename": "muv.csv.gz",
        "smiles_column": "smiles",
        "label_columns": None,
        "ignored_columns": ("mol_id",),
        "task_type": "classification",
        "selection_metric": "average_precision",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    },
}

SPLIT_NAMES = ("train", "valid", "test")
FINGERPRINT_FIELDS = {
    "morgan_fp",
    "morgan2048_fp",
    "maccs_fp",
    "rdkit_fp",
    "torsion_fp",
    "atom_pair_fp",
    "layered_fp",
}
ENCODER_STORE_FIELDS = (
    set(CompoundKit.atom_vocab_dict)
    | set(CompoundKit.atom_float_names)
    | set(CompoundKit.bond_vocab_dict)
    | {"edges", "angles_atom_index", "atom_distances_2d"}
)
PROCESSED_METADATA_COLUMNS = ("molecule_id", "source_index", "canonical_smiles")


def resolve_raw_label_columns(csv_path: str | os.PathLike[str], spec: Mapping) -> list[str]:
    """Resolve the exact target allowlist without reading any feature columns."""

    configured = spec.get("label_columns")
    if configured is not None:
        columns = [str(value) for value in configured]
    else:
        header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        excluded = {str(spec["smiles_column"]), *map(str, spec.get("ignored_columns", ()))}
        columns = [str(column) for column in header if str(column) not in excluded]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"Invalid MoleculeNet task columns: {columns}")
    return columns


def processed_label_columns(num_tasks: int) -> list[str]:
    if int(num_tasks) <= 0:
        raise ValueError("num_tasks must be positive")
    return ["label"] if int(num_tasks) == 1 else [f"label_{index:03d}" for index in range(int(num_tasks))]


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles(smiles: object) -> tuple[str, str]:
    """Return canonical isomeric SMILES and its achiral Murcko scaffold."""

    if smiles is None or pd.isna(smiles):
        raise ValueError("missing_smiles")
    text = str(smiles).strip()
    if not text:
        raise ValueError("empty_smiles")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(text)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid_smiles")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return canonical, scaffold


def _binary_label(value: object) -> int:
    if value is None or pd.isna(value):
        raise ValueError("missing_label")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_label") from exc
    if numeric not in (0.0, 1.0):
        raise ValueError("non_binary_label")
    return int(numeric)


def _optional_binary_label(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return _binary_label(value)


def _optional_regression_label(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_label") from exc
    if not math.isfinite(numeric):
        raise ValueError("non_finite_label")
    return numeric


def load_moleculenet_rows(
    csv_path: str | os.PathLike[str],
    smiles_column: str,
    label_column: str | Sequence[str],
    *,
    task_type: str = "classification",
    return_failures: bool = False,
):
    """Load valid rows using only the raw SMILES and target columns.

    Duplicate rows are deliberately retained to match MoleculeNet/DeepChem
    loading semantics.  Canonical identity and scaffold grouping later ensure
    that duplicates cannot cross splits.
    """

    task_type = str(task_type).strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported MoleculeNet task_type: {task_type!r}")
    label_columns = [label_column] if isinstance(label_column, str) else list(label_column)
    if not label_columns:
        raise ValueError("At least one MoleculeNet label column is required")
    frame = pd.read_csv(csv_path, usecols=[smiles_column, *label_columns])
    rows = []
    failures = []
    for source_index, row in frame.iterrows():
        raw_smiles = row[smiles_column]
        raw_labels = [row[column] for column in label_columns]
        try:
            label_parser = _optional_binary_label if task_type == "classification" else _optional_regression_label
            labels = [label_parser(value) for value in raw_labels]
            if len(labels) == 1 and labels[0] is None:
                raise ValueError("missing_label")
            if not any(value is not None for value in labels):
                raise ValueError("missing_all_labels")
            canonical, scaffold = canonicalize_smiles(raw_smiles)
        except ValueError as exc:
            failures.append(
                {
                    "source_index": int(source_index),
                    "smiles": "" if pd.isna(raw_smiles) else str(raw_smiles),
                    "label": json.dumps(
                        [None if pd.isna(value) else value for value in raw_labels],
                        ensure_ascii=False,
                        default=str,
                    ),
                    "failure_reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "source_index": int(source_index),
                "raw_smiles": str(raw_smiles).strip(),
                "canonical_smiles": canonical,
                "labels": labels,
                "scaffold": scaffold,
            }
        )
        if len(labels) == 1:
            rows[-1]["label"] = labels[0]
    if return_failures:
        return rows, failures
    return rows


def _as_dataframe(rows: Iterable[Mapping]) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    required = {"source_index", "canonical_smiles", "scaffold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MoleculeNet rows are missing required fields: {missing}")
    return frame.reset_index(drop=True)


def deepchem_scaffold_split(
    rows: Iterable[Mapping],
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> Dict[str, pd.DataFrame]:
    """Reproduce DeepChem's deterministic ``ScaffoldSplitter`` assignment."""

    if len(fractions) != 3 or any(float(value) < 0 for value in fractions):
        raise ValueError("fractions must contain three non-negative values")
    if abs(sum(float(value) for value in fractions) - 1.0) > 1e-8:
        raise ValueError("fractions must sum to 1")
    frame = _as_dataframe(rows)
    if frame.empty:
        raise ValueError("Cannot scaffold-split an empty table")

    scaffold_groups: Dict[str, list[int]] = {}
    for row_index, scaffold in enumerate(frame["scaffold"].astype(str).tolist()):
        scaffold_groups.setdefault(scaffold, []).append(row_index)
    for indices in scaffold_groups.values():
        indices.sort(key=lambda index: int(frame.iloc[index]["source_index"]))
    ordered_groups = sorted(
        scaffold_groups.values(),
        key=lambda indices: (len(indices), int(frame.iloc[indices[0]]["source_index"])),
        reverse=True,
    )

    train_cutoff = float(fractions[0]) * len(frame)
    valid_cutoff = (float(fractions[0]) + float(fractions[1])) * len(frame)
    assigned = {name: [] for name in SPLIT_NAMES}
    for group in ordered_groups:
        if len(assigned["train"]) + len(group) > train_cutoff:
            if len(assigned["train"]) + len(assigned["valid"]) + len(group) > valid_cutoff:
                assigned["test"].extend(group)
            else:
                assigned["valid"].extend(group)
        else:
            assigned["train"].extend(group)

    splits = {}
    for split in SPLIT_NAMES:
        split_frame = frame.iloc[assigned[split]].copy()
        split_frame["split"] = split
        splits[split] = split_frame.sort_values("source_index", kind="stable").reset_index(drop=True)
    return splits


def audit_split_disjointness(splits: Mapping[str, pd.DataFrame]) -> Dict:
    """Audit canonical identity and scaffold separation across all split pairs."""

    missing = sorted(set(SPLIT_NAMES) - set(splits))
    if missing:
        raise ValueError(f"Missing property splits: {missing}")
    pairs = {}
    passed = True
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            left_frame = _as_dataframe(splits[left])
            right_frame = _as_dataframe(splits[right])
            identity_overlap = sorted(set(left_frame["canonical_smiles"]) & set(right_frame["canonical_smiles"]))
            scaffold_overlap = sorted(set(left_frame["scaffold"]) & set(right_frame["scaffold"]))
            key = f"{left}_vs_{right}"
            pairs[key] = {
                "canonical_identity_overlap": len(identity_overlap),
                "scaffold_overlap": len(scaffold_overlap),
            }
            passed = passed and not identity_overlap and not scaffold_overlap
    return {"passed": bool(passed), "pairs": pairs}


def strip_non_encoder_features(molecule_dict: Mapping) -> Dict:
    """Allowlist only fields consumed by ``MoleculeCollator``/encoder.

    This removes fingerprints, BE matrices, conformer coordinates, RDKit Mol
    objects, and every pretraining target.  All retained fields are derived
    from the SMILES string itself.
    """

    stripped = {key: value for key, value in molecule_dict.items() if key in ENCODER_STORE_FIELDS}
    missing = sorted(ENCODER_STORE_FIELDS - set(stripped))
    if missing:
        raise ValueError(f"Generated molecule is missing encoder fields: {missing}")
    leaked = sorted(FINGERPRINT_FIELDS & set(stripped))
    if leaked:
        raise AssertionError(f"Fingerprint fields survived encoder allowlisting: {leaked}")
    return stripped


def _write_molecule_store(
    canonical_smiles: Iterable[str],
    output_path: str,
    map_size_gb: int,
) -> tuple[set[str], list[Dict]]:
    output_path = os.path.abspath(output_path)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    for path in (temporary_path, f"{temporary_path}-lock"):
        if os.path.exists(path):
            os.unlink(path)
    env = lmdb.open(temporary_path, map_size=int(map_size_gb) * 1024**3, subdir=False)
    successful: set[str] = set()
    failures: list[Dict] = []
    try:
        ordered_smiles = sorted(set(str(value) for value in canonical_smiles))
        disable_progress = os.environ.get("HYPERMOL_DISABLE_TQDM", "").lower() in {"1", "true", "yes"}
        for start in tqdm(
            range(0, len(ordered_smiles), 32),
            desc="SMILES-only molecule features",
            unit="batch",
            disable=disable_progress,
        ):
            batch = ordered_smiles[start : start + 32]
            with env.begin(write=True) as transaction:
                for smiles in batch:
                    try:
                        with rdBase.BlockLogs():
                            mol = Chem.MolFromSmiles(smiles)
                            # The molecular encoder consumes atom attributes and
                            # 2-D graph/path features, not conformer coordinates.
                            # Supplying deterministic 2-D coordinates skips the
                            # expensive MMFF search performed for pretraining-only
                            # angle/torsion targets; those targets are allowlisted
                            # out immediately below.
                            atom_poses = Compound3DKit.get_2d_atom_poses(mol) if mol is not None else None
                            generated = (
                                mol_to_data_pretrain(mol, smiles, pre_calculated_compose=atom_poses)
                                if mol is not None
                                else None
                            )
                        if generated is None:
                            raise ValueError("mol_to_data_pretrain_returned_none")
                        payload = strip_non_encoder_features(generated)
                        transaction.put(smiles_key_bytes(smiles), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                        successful.add(smiles)
                    except Exception as exc:
                        failures.append({"canonical_smiles": smiles, "failure_reason": type(exc).__name__ + ":" + str(exc)})
        env.sync()
    finally:
        env.close()
    if os.path.exists(f"{temporary_path}-lock"):
        os.unlink(f"{temporary_path}-lock")
    if os.path.exists(output_path):
        os.unlink(output_path)
    os.replace(temporary_path, output_path)
    return successful, failures


def _duplicate_audit(frame: pd.DataFrame, label_columns: Sequence[str]) -> Dict:
    group_sizes = frame.groupby("canonical_smiles", sort=False).size()
    duplicate_groups = group_sizes[group_sizes > 1]
    conflicting_identities: set[str] = set()
    conflicting_task_events = 0
    grouped = frame.groupby("canonical_smiles", sort=False)
    for column in label_columns:
        counts = grouped[column].nunique(dropna=True)
        conflicts = counts[counts > 1]
        conflicting_identities.update(map(str, conflicts.index))
        conflicting_task_events += int(len(conflicts))
    return {
        "unique_canonical_smiles": int(frame["canonical_smiles"].nunique()),
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_extra_rows": int(duplicate_groups.sum() - len(duplicate_groups)),
        "conflicting_label_groups": int(len(conflicting_identities)),
        "conflicting_label_rows": int(
            frame[frame["canonical_smiles"].isin(conflicting_identities)].shape[0]
        ),
        "conflicting_task_events": int(conflicting_task_events),
        "policy": "retain_all_rows_and_group_by_scaffold",
    }


def _task_label_stats(
    frame: pd.DataFrame,
    label_columns: Sequence[str],
    task_names: Sequence[str],
    task_type: str,
) -> Dict:
    task_stats = {}
    for task_name, column in zip(task_names, label_columns):
        observed = pd.to_numeric(frame[column], errors="coerce").dropna()
        stats = {
            "processed_column": str(column),
            "observed": int(len(observed)),
            "missing": int(len(frame) - len(observed)),
        }
        if task_type == "classification":
            counts = observed.value_counts().to_dict()
            negative = int(counts.get(0, 0))
            positive = int(counts.get(1, 0))
            stats.update(
                {
                    "negative": negative,
                    "positive": positive,
                    "positive_prevalence": float(positive / len(observed)) if len(observed) else None,
                    "roc_auc_eligible": bool(negative > 0 and positive > 0),
                }
            )
        else:
            values = observed.to_numpy(dtype=float)
            stats.update(
                {
                    "minimum": float(values.min()) if len(values) else None,
                    "maximum": float(values.max()) if len(values) else None,
                    "mean": float(values.mean()) if len(values) else None,
                    "standard_deviation": float(values.std(ddof=0)) if len(values) else None,
                    "regression_eligible": bool(len(values) > 0),
                }
            )
        task_stats[str(task_name)] = stats
    return task_stats


def _split_stats(
    frame: pd.DataFrame,
    csv_path: str,
    label_columns: Sequence[str],
    task_names: Sequence[str],
    task_type: str,
) -> Dict:
    task_stats = _task_label_stats(frame, label_columns, task_names, task_type)
    result = {
        "rows": int(len(frame)),
        "num_tasks": int(len(label_columns)),
        "observed_labels": int(sum(value["observed"] for value in task_stats.values())),
        "tasks": task_stats,
        "unique_canonical_smiles": int(frame["canonical_smiles"].nunique()),
        "unique_scaffolds": int(frame["scaffold"].nunique()),
        "csv_sha256": sha256_file(csv_path),
    }
    if task_type == "classification":
        result["roc_auc_eligible_tasks"] = int(
            sum(value["roc_auc_eligible"] for value in task_stats.values())
        )
    else:
        result["regression_eligible_tasks"] = int(
            sum(value["regression_eligible"] for value in task_stats.values())
        )
    if len(label_columns) == 1:
        single = task_stats[str(task_names[0])]
        if task_type == "classification":
            result.update(
                {
                    "negative": single["negative"],
                    "positive": single["positive"],
                    "positive_prevalence": single["positive_prevalence"],
                }
            )
        else:
            result.update(
                {key: single[key] for key in ("minimum", "maximum", "mean", "standard_deviation")}
            )
    return result


def prepare_moleculenet_dataset(
    dataset: str,
    raw_csv: str,
    output_root: str,
    *,
    map_size_gb: int = 16,
    overwrite: bool = False,
) -> Dict:
    dataset = str(dataset).lower()
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset}'; choose from {sorted(DATASET_SPECS)}")
    spec = DATASET_SPECS[dataset]
    task_type = str(spec.get("task_type", "classification")).strip().lower()
    raw_label_columns = resolve_raw_label_columns(raw_csv, spec)
    label_columns = processed_label_columns(len(raw_label_columns))
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    managed_outputs = [
        *(os.path.join(output_root, f"{split}.csv") for split in SPLIT_NAMES),
        os.path.join(output_root, "all_processed.csv"),
        os.path.join(output_root, "failed_rows.csv"),
        os.path.join(output_root, "scaffold_audit.csv"),
        os.path.join(output_root, "molecule_store.mdb"),
        os.path.join(output_root, "manifest.json"),
    ]
    existing = [path for path in managed_outputs if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(f"Processed outputs already exist; pass --overwrite: {existing[0]}")
    if overwrite:
        for path in existing:
            os.unlink(path)

    rows, failures = load_moleculenet_rows(
        raw_csv,
        spec["smiles_column"],
        raw_label_columns,
        task_type=task_type,
        return_failures=True,
    )
    valid_frame = _as_dataframe(rows)
    label_frame = pd.DataFrame(
        valid_frame["labels"].tolist(),
        columns=label_columns,
        index=valid_frame.index,
    )
    valid_frame = pd.concat(
        [valid_frame.drop(columns=[column for column in label_columns if column in valid_frame]), label_frame],
        axis=1,
    )
    molecule_store_path = os.path.join(output_root, "molecule_store.mdb")
    successful, feature_failures = _write_molecule_store(
        valid_frame["canonical_smiles"],
        molecule_store_path,
        map_size_gb=map_size_gb,
    )
    if feature_failures:
        failed_identities = {item["canonical_smiles"] for item in feature_failures}
        dropped = valid_frame[valid_frame["canonical_smiles"].isin(failed_identities)]
        failure_reason = {item["canonical_smiles"]: item["failure_reason"] for item in feature_failures}
        failures.extend(
            {
                "source_index": int(row.source_index),
                "smiles": str(row.raw_smiles),
                "label": json.dumps(list(row.labels), ensure_ascii=False),
                "failure_reason": "feature_generation_failed:" + failure_reason[str(row.canonical_smiles)],
            }
            for row in dropped.itertuples(index=False)
        )
        valid_frame = valid_frame[valid_frame["canonical_smiles"].isin(successful)].reset_index(drop=True)
    if valid_frame.empty:
        raise RuntimeError(f"No valid {dataset} rows remained after SMILES feature generation")

    splits = deepchem_scaffold_split(valid_frame)
    disjointness = audit_split_disjointness(splits)
    if not disjointness["passed"]:
        raise RuntimeError(f"Scaffold split leakage audit failed: {disjointness}")
    for split, frame in splits.items():
        if task_type == "classification":
            eligible_tasks = sum(
                int(set(pd.to_numeric(frame[column], errors="coerce").dropna().astype(int).unique()) == {0, 1})
                for column in label_columns
            )
            if eligible_tasks == 0:
                raise RuntimeError(f"{dataset} {split} split has no ROC-AUC-eligible binary task")
        elif not any(pd.to_numeric(frame[column], errors="coerce").notna().any() for column in label_columns):
            raise RuntimeError(f"{dataset} {split} split has no observed regression target")

    processed_frames = []
    split_stats = {}
    for split in SPLIT_NAMES:
        frame = splits[split].copy()
        frame["molecule_id"] = frame["canonical_smiles"].map(
            lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        )
        output_columns = [*PROCESSED_METADATA_COLUMNS, *label_columns, "split"]
        output_frame = frame.loc[:, output_columns]
        csv_path = os.path.join(output_root, f"{split}.csv")
        output_frame.to_csv(csv_path, index=False)
        split_stats[split] = _split_stats(frame, csv_path, label_columns, raw_label_columns, task_type)
        processed_frames.append(output_frame)

    all_processed = pd.concat(processed_frames, ignore_index=True).sort_values("source_index", kind="stable")
    all_processed.to_csv(os.path.join(output_root, "all_processed.csv"), index=False)
    pd.DataFrame(failures, columns=["source_index", "smiles", "label", "failure_reason"]).to_csv(
        os.path.join(output_root, "failed_rows.csv"), index=False
    )
    audit_frame = pd.concat([splits[name] for name in SPLIT_NAMES], ignore_index=True)
    audit_frame.loc[:, ["source_index", "canonical_smiles", *label_columns, "scaffold", "split"]].to_csv(
        os.path.join(output_root, "scaffold_audit.csv"), index=False
    )

    raw_rows = len(rows) + len(failures) - sum(
        1 for item in failures if str(item.get("failure_reason", "")).startswith("feature_generation_failed:")
    )
    failure_counts = pd.Series([item["failure_reason"].split(":", 1)[0] for item in failures]).value_counts().to_dict()
    manifest = {
        "dataset": dataset,
        "task_type": task_type,
        "selection_metric": str(spec.get("selection_metric", "roc_auc")),
        "status": "diagnostic_transfer_dataset",
        "raw_csv": os.path.abspath(raw_csv),
        "raw_sha256": sha256_file(raw_csv),
        "download_url": spec["download_url"],
        "raw_rows": int(raw_rows),
        "processed_rows": int(len(valid_frame)),
        "failed_rows": int(len(failures)),
        "failure_counts": {str(key): int(value) for key, value in failure_counts.items()},
        "source_columns_consumed": [spec["smiles_column"], *raw_label_columns],
        "smiles_column": spec["smiles_column"],
        "task_names": list(raw_label_columns),
        "num_tasks": int(len(raw_label_columns)),
        "processed_label_columns": list(label_columns),
        "descriptor_columns_consumed": [],
        "processed_columns": [*PROCESSED_METADATA_COLUMNS, *label_columns, "split"],
        "missing_label_policy": "retain rows with at least one observed task and mask missing task labels",
        "rdkit_version": rdBase.rdkitVersion,
        "canonicalization": "RDKit canonical isomeric SMILES",
        "splitter": {
            "name": "DeepChem-compatible Bemis-Murcko scaffold split",
            "fractions": [0.8, 0.1, 0.1],
            "include_chirality": False,
            "seed": None,
        },
        "duplicates": _duplicate_audit(valid_frame, label_columns),
        "splits": split_stats,
        "disjointness": disjointness,
        "molecule_store": {
            "path": molecule_store_path,
            "unique_entries": int(len(successful)),
            "feature_source": "SMILES-derived RDKit atom attributes and 2-D graph/topology only",
            "allowed_fields": sorted(ENCODER_STORE_FIELDS),
            "fingerprint_fields_present": [],
            "file_sha256": sha256_file(molecule_store_path),
        },
    }
    with open(os.path.join(output_root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--raw_csv", default="")
    parser.add_argument("--raw_root", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--map_size_gb", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    spec = DATASET_SPECS[args.dataset]
    raw_csv = args.raw_csv or os.path.join(args.raw_root, spec["filename"])
    if not raw_csv:
        parser.error("Provide --raw_csv or --raw_root")
    manifest = prepare_moleculenet_dataset(
        args.dataset,
        raw_csv,
        args.output_root,
        map_size_gb=args.map_size_gb,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
