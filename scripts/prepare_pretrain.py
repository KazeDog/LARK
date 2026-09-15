"""Build pretraining LMDB stores from mapped train/test reaction CSVs."""
import argparse
from pathlib import Path
import shutil

import pandas as pd
from rdkit import Chem

from hypermol.data.preprocess import process_smiles_parallel_streaming
from hypermol.data.molecule_store import open_molecule_store


def prepare(raw_root, output_root, workers=4, map_size_gb=64):
    raw_root, output_root = Path(raw_root), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_root}")
    core, context = set(), set()
    for split in ("train", "test"):
        frame = pd.read_csv(raw_root / f"{split}.csv", keep_default_na=False)
        required = {"reactant_smiles", "prod_smiles", "unmapped_components"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{split}.csv must be nonempty and contain {sorted(required)}")
        for index, row in frame.iterrows():
            side_atoms = {}
            for column in sorted(required):
                identities = {}
                text = str(row[column]).strip()
                if column != "unmapped_components" and not text:
                    raise ValueError(f"Empty {column} at {split} row {index}")
                for smiles in filter(None, (part.strip() for part in text.split("."))):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        raise ValueError(f"Invalid SMILES at {split} row {index}: {smiles}")
                    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
                    if column == "unmapped_components":
                        if any(maps):
                            raise ValueError("Context components must have no atom maps")
                        context.add(smiles)
                    else:
                        if not all(maps) or len(set(maps)) != len(maps):
                            raise ValueError("Core atoms require unique positive atom maps")
                        for atom in mol.GetAtoms():
                            map_id = atom.GetAtomMapNum()
                            if map_id in identities:
                                raise ValueError("Atom maps must be unique across all components of one side")
                            identities[map_id] = atom.GetAtomicNum()
                        core.add(smiles)
                side_atoms[column] = identities
            reactants, products = side_atoms["reactant_smiles"], side_atoms["prod_smiles"]
            shared = reactants.keys() & products.keys()
            if not shared or any(reactants[key] != products[key] for key in shared):
                raise ValueError(f"Inconsistent atom correspondence at {split} row {index}")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, smiles_set in (("smiles", core), ("unmapped_smiles", context)):
        txt = output_root / f"{name}.txt"
        txt.write_text("".join(s + "\n" for s in sorted(smiles_set)))
        database = output_root / f"{name}.lmdb"
        process_smiles_parallel_streaming(str(txt), str(database),
            num_cores=workers, map_size=map_size_gb * 1024**3)
        store = open_molecule_store(str(database))
        try:
            for smiles in smiles_set:
                store.get_mol_dict(smiles)
        finally:
            store.close()
    for split in ("train", "test"):
        shutil.copyfile(raw_root / f"{split}.csv", output_root / f"{split}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--map-size-gb", type=int, default=64)
    args = parser.parse_args()
    prepare(args.raw_root, args.output_root, args.workers, args.map_size_gb)
