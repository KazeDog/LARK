"""Prepare LARK-owned reaction classification datasets."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from typing import Dict, Iterable, List, Tuple

import lmdb
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from hypermol.data.preprocess import mol_to_data_pretrain
from hypermol.data.reaction_dataset import get_smiles_key


REQUIRED_COLUMNS = {"reactant_smiles", "prod_smiles", "reaction_type"}
SPLIT_FILES = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv"}


def split_components(smiles_text: object) -> List[str]:
    return [part.strip() for part in str(smiles_text).split(".") if part.strip()]


def load_split(csv_path: str, limit_rows: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    if limit_rows > 0:
        df = df.head(limit_rows).copy()

    kept = []
    failed = []
    for row in df.to_dict("records"):
        reactants = split_components(row["reactant_smiles"])
        products = split_components(row["prod_smiles"])
        try:
            label = int(float(str(row["reaction_type"]).strip()))
        except Exception:
            failed.append(row)
            continue
        if not reactants or not products:
            failed.append(row)
            continue
        kept.append(
            {
                "reactant_smiles": ".".join(reactants),
                "prod_smiles": ".".join(products),
                "reaction_type": label,
                "reaction_smiles": ".".join(reactants) + ">>" + ".".join(products),
            }
        )
    return pd.DataFrame(kept), pd.DataFrame(failed)


def collect_unique_smiles(frames: Iterable[pd.DataFrame]) -> List[str]:
    smiles_set = set()
    for df in frames:
        for column in ("reactant_smiles", "prod_smiles"):
            for value in df[column].tolist():
                smiles_set.update(split_components(value))
    return sorted(smiles_set)


def row_smiles(row: pd.Series) -> List[str]:
    return split_components(row["reactant_smiles"]) + split_components(row["prod_smiles"])


def filter_missing_smiles(
    frames: Dict[str, pd.DataFrame],
    failed_frames: Dict[str, pd.DataFrame],
    missing_smiles: List[str],
) -> Dict[str, int]:
    missing_set = set(missing_smiles)
    dropped = {}
    for split, df in frames.items():
        if df.empty:
            dropped[split] = 0
            continue
        valid_mask = df.apply(lambda row: not any(smiles in missing_set for smiles in row_smiles(row)), axis=1)
        missing_rows = df.loc[~valid_mask].copy()
        if not missing_rows.empty:
            missing_rows["failure_reason"] = "missing_reused_molecule_db_entry"
            failed_frames[split] = pd.concat([failed_frames[split], missing_rows], ignore_index=True)
        frames[split] = df.loc[valid_mask].reset_index(drop=True)
        dropped[split] = int((~valid_mask).sum())
    return dropped


def verify_lmdb_contains(lmdb_path: str, smiles_list: List[str]) -> List[str]:
    missing = []
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, subdir=False)
    try:
        with env.begin(write=False) as txn:
            for smiles in tqdm(smiles_list, desc="verify molecule db", leave=False):
                if txn.get(get_smiles_key(smiles)) is None:
                    missing.append(smiles)
    finally:
        env.close()
    return missing


def write_lmdb(smiles_list: List[str], lmdb_path: str, smiles_txt: str, map_size_gb: int) -> Dict[str, int]:
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=int(map_size_gb) * 1024**3, subdir=False)
    failed = []
    try:
        with env.begin(write=True) as txn:
            for smiles in tqdm(smiles_list, desc="build molecule db", leave=False):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed.append(smiles)
                    continue
                try:
                    mol_dict = mol_to_data_pretrain(mol, smiles)
                except Exception:
                    mol_dict = None
                if mol_dict is None:
                    failed.append(smiles)
                    continue
                txn.put(get_smiles_key(smiles), pickle.dumps(mol_dict))
    finally:
        env.sync()
        env.close()
    with open(smiles_txt, "w", encoding="utf-8") as f:
        for smiles in smiles_list:
            if smiles not in failed:
                f.write(smiles + "\n")
    return {
        "unique_smiles": len(smiles_list),
        "stored_molecules": len(smiles_list) - len(failed),
        "failed_molecules": len(failed),
    }


def copy_or_link(src: str, dst: str, mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        if os.path.samefile(src, dst):
            return
        raise FileExistsError(f"Destination already exists: {dst}")
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported link mode: {mode}")


def process_reaction_class(
    raw_root: str,
    output_root: str,
    reuse_molecule_db: str = "",
    reuse_smiles_txt: str = "",
    link_mode: str = "hardlink",
    verify_reuse: bool = True,
    limit_rows: int = 0,
    map_size_gb: int = 256,
) -> Dict:
    os.makedirs(output_root, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    failed_frames: Dict[str, pd.DataFrame] = {}

    for split, filename in SPLIT_FILES.items():
        kept, failed = load_split(os.path.join(raw_root, filename), limit_rows=limit_rows)
        frames[split] = kept
        failed_frames[split] = failed

    unique_smiles = collect_unique_smiles(frames.values())
    molecule_db = os.path.join(output_root, "smiles.mdb")
    smiles_txt = os.path.join(output_root, "smiles.txt")
    missing_reused_smiles: List[str] = []
    dropped_missing_rows = {"train": 0, "valid": 0, "test": 0}
    if reuse_molecule_db:
        if verify_reuse:
            missing_reused_smiles = verify_lmdb_contains(reuse_molecule_db, unique_smiles)
            if missing_reused_smiles:
                dropped_missing_rows = filter_missing_smiles(frames, failed_frames, missing_reused_smiles)
                unique_smiles = collect_unique_smiles(frames.values())

    for split, filename in SPLIT_FILES.items():
        frames[split].to_csv(os.path.join(output_root, filename), index=False)
        failed_frames[split].to_csv(os.path.join(output_root, f"{split}_failed_rows.csv"), index=False)
    pd.DataFrame({"smiles": unique_smiles}).to_csv(os.path.join(output_root, "unique_smiles_all.csv"), index=False)

    if reuse_molecule_db:
        copy_or_link(reuse_molecule_db, molecule_db, mode=link_mode)
        if reuse_smiles_txt and not missing_reused_smiles:
            copy_or_link(reuse_smiles_txt, smiles_txt, mode=link_mode)
        else:
            with open(smiles_txt, "w", encoding="utf-8") as f:
                for smiles in unique_smiles:
                    f.write(smiles + "\n")
        mol_stats = {
            "unique_smiles": len(unique_smiles),
            "stored_molecules": len(unique_smiles),
            "failed_molecules": 0,
            "reuse_molecule_db": reuse_molecule_db,
            "link_mode": link_mode,
            "missing_reused_molecules": len(missing_reused_smiles),
            "dropped_rows_missing_reuse": dropped_missing_rows,
        }
    else:
        mol_stats = write_lmdb(unique_smiles, molecule_db, smiles_txt, map_size_gb=map_size_gb)

    stats = {
        split: {
            "input_rows": int(len(frames[split]) + len(failed_frames[split])),
            "processed_rows": int(len(frames[split])),
            "failed_rows": int(len(failed_frames[split])),
        }
        for split in ("train", "valid", "test")
    }
    stats["molecule_store"] = mol_stats
    stats["label_distribution"] = {
        split: {
            "count": int(len(frames[split])),
            "num_classes": int(frames[split]["reaction_type"].nunique()),
            "class_counts": {
                str(int(label)): int(count)
                for label, count in frames[split]["reaction_type"].value_counts().sort_index().items()
            },
        }
        for split in ("train", "valid", "test")
    }
    with open(os.path.join(output_root, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_mode": "schneider_reaction_class",
                "raw_root": raw_root,
                "output_root": output_root,
                "processor": "hypermol.data.process_reaction_class",
                "reuse_molecule_db": reuse_molecule_db,
                "reuse_smiles_txt": reuse_smiles_txt,
                "link_mode": link_mode,
                "verify_reuse": verify_reuse,
                "limit_rows": limit_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--reuse_molecule_db", default="")
    parser.add_argument("--reuse_smiles_txt", default="")
    parser.add_argument("--link_mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--skip_verify_reuse", action="store_true")
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--map_size_gb", type=int, default=256)
    args = parser.parse_args()
    stats = process_reaction_class(
        raw_root=args.raw_root,
        output_root=args.output_root,
        reuse_molecule_db=args.reuse_molecule_db,
        reuse_smiles_txt=args.reuse_smiles_txt,
        link_mode=args.link_mode,
        verify_reuse=not args.skip_verify_reuse,
        limit_rows=args.limit_rows,
        map_size_gb=args.map_size_gb,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
