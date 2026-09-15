"""Bond-electron matrix helpers.

The project uses ``r_matrix`` as a compatibility name for ``delta_be``:

    delta_be = product_be_matrix - reactant_be_matrix

The BE matrix itself follows the prototype implementation: off-diagonal values
store bond order and diagonal values store non-bonded valence electrons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Tuple

import numpy as np
from rdkit.Chem.rdchem import Mol

from hypermol.data.preprocess import get_be_matrix_with_map_info


AROMATIC_MODES = {"aromatic_1p5", "kekule"}


@dataclass
class BEMatrixAudit:
    """Lightweight quality report for a reaction-level BE/Delta-BE label."""

    num_reactants: int = 0
    num_products: int = 0
    num_reactant_maps: int = 0
    num_product_maps: int = 0
    num_shared_maps: int = 0
    num_reactant_only_maps: int = 0
    num_product_only_maps: int = 0
    reactant_sum: float = 0.0
    product_sum: float = 0.0
    delta_sum: float = 0.0
    max_abs_delta: float = 0.0
    reactant_symmetric: bool = True
    product_symmetric: bool = True
    delta_symmetric: bool = True
    reactant_negative_diagonal: int = 0
    product_negative_diagonal: int = 0
    duplicate_reactant_maps: int = 0
    duplicate_product_maps: int = 0
    electron_conserved: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def validate_aromatic_mode(aromatic_mode: str) -> str:
    if aromatic_mode not in AROMATIC_MODES:
        raise ValueError(f"aromatic_mode must be one of {sorted(AROMATIC_MODES)}.")
    return aromatic_mode


def compute_be_matrix(
    mol_or_smiles: str | Mol,
    aromatic_mode: str = "aromatic_1p5",
) -> Tuple[np.ndarray, int, dict[int, int]]:
    """Return the BE matrix, maximum atom-map number, and map-to-RDKit index."""
    be_matrix, max_map_num, map_to_idx = get_be_matrix_with_map_info(
        mol_or_smiles,
        aromatic_mode=validate_aromatic_mode(aromatic_mode),
    )
    if be_matrix is None or map_to_idx is None:
        raise ValueError("Could not compute BE matrix. Input must be mapped and RDKit-parseable.")
    return be_matrix, max_map_num, map_to_idx


def sum_be_matrices(matrices: Iterable[np.ndarray], size: int | None = None) -> np.ndarray:
    """Pad and sum BE matrices in a shared atom-map coordinate system."""
    matrices = list(matrices)
    if not matrices and size is None:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(size if size is not None else max(mat.shape[0] for mat in matrices))
    out = np.zeros((dim, dim), dtype=np.float32)
    for mat in matrices:
        out[: mat.shape[0], : mat.shape[1]] += mat.astype(np.float32, copy=False)
    return out


def compute_delta_be(product_be: np.ndarray, reactant_be: np.ndarray) -> np.ndarray:
    """Compute product-minus-reactant BE change matrix."""
    dim = max(product_be.shape[0], reactant_be.shape[0])
    product = sum_be_matrices([product_be], size=dim)
    reactant = sum_be_matrices([reactant_be], size=dim)
    return product - reactant


def compute_reaction_delta_be(
    reactant_be_matrices: Iterable[np.ndarray],
    product_be_matrices: Iterable[np.ndarray],
) -> np.ndarray:
    """Compute the overall reaction delta-BE matrix."""
    reactants = list(reactant_be_matrices)
    products = list(product_be_matrices)
    if not reactants and not products:
        return np.zeros((0, 0), dtype=np.float32)
    dim = max([m.shape[0] for m in reactants + products])
    return sum_be_matrices(products, size=dim) - sum_be_matrices(reactants, size=dim)


def map_numbers_from_mol_dict(mol_dict: Mapping) -> list[int]:
    """Return atom-map numbers present in a stored molecule dictionary."""
    map_list = mol_dict.get("map_list", {}) if mol_dict else {}
    return [int(map_num) for map_num in map_list.keys()]


def collect_map_numbers(mol_dicts: Iterable[Mapping]) -> list[int]:
    maps = []
    for mol_dict in mol_dicts:
        maps.extend(map_numbers_from_mol_dict(mol_dict))
    return maps


def count_duplicates(values: Iterable[int]) -> int:
    seen = set()
    dup = 0
    for value in values:
        if value in seen:
            dup += 1
        else:
            seen.add(value)
    return dup


def audit_reaction_be_matrices(
    reactant_be_matrices: Iterable[np.ndarray],
    product_be_matrices: Iterable[np.ndarray],
    reactant_mol_dicts: Iterable[Mapping] | None = None,
    product_mol_dicts: Iterable[Mapping] | None = None,
    tolerance: float = 1e-4,
) -> BEMatrixAudit:
    """Check conservation and structural invariants for reaction-level BE data."""
    reactants = list(reactant_be_matrices)
    products = list(product_be_matrices)
    reactant_mol_dicts = list(reactant_mol_dicts or [])
    product_mol_dicts = list(product_mol_dicts or [])
    if not reactants and not products:
        return BEMatrixAudit()

    dim = max([mat.shape[0] for mat in reactants + products])
    reactant_be = sum_be_matrices(reactants, size=dim)
    product_be = sum_be_matrices(products, size=dim)
    delta_be = product_be - reactant_be

    reactant_maps = collect_map_numbers(reactant_mol_dicts)
    product_maps = collect_map_numbers(product_mol_dicts)
    reactant_set = set(reactant_maps)
    product_set = set(product_maps)

    audit = BEMatrixAudit(
        num_reactants=len(reactants),
        num_products=len(products),
        num_reactant_maps=len(reactant_set),
        num_product_maps=len(product_set),
        num_shared_maps=len(reactant_set & product_set),
        num_reactant_only_maps=len(reactant_set - product_set),
        num_product_only_maps=len(product_set - reactant_set),
        reactant_sum=float(reactant_be.sum()),
        product_sum=float(product_be.sum()),
        delta_sum=float(delta_be.sum()),
        max_abs_delta=float(np.abs(delta_be).max()) if delta_be.size else 0.0,
        reactant_symmetric=bool(np.allclose(reactant_be, reactant_be.T, atol=tolerance)),
        product_symmetric=bool(np.allclose(product_be, product_be.T, atol=tolerance)),
        delta_symmetric=bool(np.allclose(delta_be, delta_be.T, atol=tolerance)),
        reactant_negative_diagonal=int((np.diag(reactant_be) < -tolerance).sum()),
        product_negative_diagonal=int((np.diag(product_be) < -tolerance).sum()),
        duplicate_reactant_maps=count_duplicates(reactant_maps),
        duplicate_product_maps=count_duplicates(product_maps),
        electron_conserved=bool(abs(float(delta_be.sum())) <= tolerance),
    )
    return audit


def is_bad_be_audit(audit: BEMatrixAudit) -> bool:
    """Return True when the audit reveals a likely label-quality problem."""
    return (
        not audit.electron_conserved
        or not audit.reactant_symmetric
        or not audit.product_symmetric
        or not audit.delta_symmetric
        or audit.reactant_negative_diagonal > 0
        or audit.product_negative_diagonal > 0
        or audit.duplicate_reactant_maps > 0
        or audit.duplicate_product_maps > 0
        or audit.num_reactant_only_maps > 0
        or audit.num_product_only_maps > 0
    )
