"""Role heuristics for auxiliary reaction-condition molecules."""

from __future__ import annotations

from typing import Iterable, List

from rdkit import Chem


AUX_ROLE_NAMES = ("unknown", "ligand", "pd_source", "base", "solvent", "additive")
AUX_ROLE_TO_ID = {name: idx for idx, name in enumerate(AUX_ROLE_NAMES)}
UNKNOWN_AUX_ROLE_ID = AUX_ROLE_TO_ID["unknown"]

_SOLVENT_SMILES = {
    "C1CCOC1",
    "CC#N",
    "CCc1cccc(CC)c1",
    "CN(C)C=O",
    "CO",
    "CS(C)=O",
    "O",
}

_BASE_SMILES = {
    "CCN=P(N=P(N(C)C)(N(C)C)N(C)C)(N(C)C)N(C)C",
    "CC(C)(C)[O-]",
    "CCN(CC)CC",
    "CN(C)C(=NC(C)(C)C)N(C)C",
    "CN1CCCN2CCCN=C12",
    "O=C([O-])O",
    "O=P([O-])([O-])[O-]",
    "[Cs+]",
    "[F-]",
    "[K+]",
    "[Li+]",
    "[Na+]",
    "[OH-]",
}

_LIGAND_AUX_SMILES = {
    "[Fe+2]",
    "[Fe]",
}


def _mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(str(smiles).strip())


def has_atom(smiles: str, symbol: str) -> bool:
    mol = _mol_from_smiles(smiles)
    return bool(mol is not None and any(atom.GetSymbol() == symbol for atom in mol.GetAtoms()))


def is_phosphine_ligand(smiles: str) -> bool:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "P":
            continue
        carbon_neighbors = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "C")
        if carbon_neighbors >= 2:
            return True
    return False


def classify_aux_role(smiles: str) -> str:
    text = str(smiles).strip()
    if not text:
        return "unknown"
    if has_atom(text, "Pd"):
        return "pd_source"
    if is_phosphine_ligand(text) or text in _LIGAND_AUX_SMILES:
        return "ligand"
    if text in _BASE_SMILES:
        return "base"
    if text in _SOLVENT_SMILES:
        return "solvent"
    return "additive"


def classify_aux_roles(aux_smiles: Iterable[str]) -> List[str]:
    return [classify_aux_role(smiles) for smiles in aux_smiles]


def aux_role_to_id(role: object) -> int:
    return AUX_ROLE_TO_ID.get(str(role).strip(), UNKNOWN_AUX_ROLE_ID)
