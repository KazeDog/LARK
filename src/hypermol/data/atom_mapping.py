"""Atom-mapping helpers backed by RXNMapper.

The functions in this module are intentionally small and dependency-light at
import time. ``rxnmapper`` is imported lazily only when a mapping function is
called, so preprocessing scripts can still be inspected and tested in
environments where RXNMapper is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from tqdm import tqdm


COMMON_MAPPED_RXN_COLUMNS = (
    "mapped_rxn",
    "mapped_reaction",
    "mapped_reaction_smiles",
    "atom_mapped_rxn",
    "mapped_canonical_rxn",
    "mapped_smiles",
)

_ATOM_MAP_PATTERN = re.compile(r":\d+\]")
_RXN_MAPPER = None


@dataclass(frozen=True)
class MappingResult:
    """Normalized result for one reaction."""

    mapped_rxn: str
    reason: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.mapped_rxn) and has_atom_mapping_markers(self.mapped_rxn)

    def to_dict(self) -> dict[str, str]:
        return {
            "mapped_rxn": self.mapped_rxn,
            "reason": self.reason,
            "detail": self.detail,
        }


def has_atom_mapping_markers(smiles: str | None) -> bool:
    """Return True when a SMILES/reaction SMILES string contains atom-map ids."""

    if not isinstance(smiles, str):
        return False
    return _ATOM_MAP_PATTERN.search(smiles) is not None


def split_reaction_smiles(rxn: str | None) -> Optional[tuple[str, str]]:
    """Split reaction SMILES into reactant and product sides.

    Both ``reactants>>products`` and ``reactants>agents>products`` are accepted.
    Agents are ignored because LARK stores reaction conditions separately.
    """

    if not isinstance(rxn, str):
        return None
    parts = rxn.strip().split(">")
    if len(parts) != 3:
        return None
    reactants, _, products = parts
    reactants = reactants.strip()
    products = products.strip()
    if not reactants or not products:
        return None
    return reactants, products


def join_reaction_smiles(reactants: Any, products: Any) -> str:
    """Build a reaction SMILES from ORDerly-style reactant/product fields."""

    reactant_text = "" if reactants is None else str(reactants).strip()
    product_text = "" if products is None else str(products).strip()
    if not reactant_text or not product_text:
        return ""
    return f"{reactant_text}>>{product_text}"


def find_existing_mapped_rxn(row: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Find an already mapped reaction column in a row-like dictionary."""

    for key in COMMON_MAPPED_RXN_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and has_atom_mapping_markers(value):
            return key, value.strip()
    for key, value in row.items():
        lower = str(key).lower()
        if "mapped" not in lower:
            continue
        if "rxn" not in lower and "reaction" not in lower and "smiles" not in lower:
            continue
        if isinstance(value, str) and has_atom_mapping_markers(value):
            return str(key), value.strip()
    return None


def get_rxn_mapper():
    """Create or return the singleton RXNMapper instance."""

    global _RXN_MAPPER
    if _RXN_MAPPER is None:
        from rxnmapper import RXNMapper

        _RXN_MAPPER = RXNMapper()
    return _RXN_MAPPER


def ensure_rxnmapper_available() -> None:
    """Raise a clear error when RXNMapper cannot be used."""

    try:
        get_rxn_mapper()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RXNMapper is required for atom mapping, but it is not installed in "
            "the active Python environment. Install `rxnmapper` first."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"RXNMapper is installed but failed to initialize: {type(exc).__name__}: {str(exc).strip()}"
        ) from exc


def parse_rxnmapper_result(result: object) -> MappingResult:
    """Normalize RXNMapper output across package versions."""

    mapped_rxn = None
    if isinstance(result, dict):
        mapped_rxn = (
            result.get("mapped_rxn")
            or result.get("mapped_reaction")
            or result.get("mapped_reaction_smiles")
            or result.get("rxn")
        )
    elif isinstance(result, str):
        mapped_rxn = result
    else:
        return MappingResult("", "rxnmapper_invalid_result_type", type(result).__name__)

    if mapped_rxn is None:
        return MappingResult("", "rxnmapper_empty_result")
    mapped_rxn = str(mapped_rxn).strip()
    if mapped_rxn == "":
        return MappingResult("", "rxnmapper_empty_output")
    if not has_atom_mapping_markers(mapped_rxn):
        return MappingResult(mapped_rxn, "rxnmapper_output_without_atom_map")
    return MappingResult(mapped_rxn, "rxnmapper_success")


def _map_rxnmapper_batch_recursive(
    rxns: list[str],
    output_indices: list[int],
    outputs: list[Optional[MappingResult]],
) -> None:
    """Map a batch, recursively splitting failed batches down to singletons."""

    try:
        results = get_rxn_mapper().get_attention_guided_atom_maps(rxns)
        if not isinstance(results, list) or len(results) != len(rxns):
            raise RuntimeError(
                f"Unexpected RXNMapper batch output length: got="
                f"{len(results) if isinstance(results, list) else 'non-list'} expected={len(rxns)}"
            )
    except Exception as exc:
        if len(rxns) == 1:
            outputs[output_indices[0]] = MappingResult(
                "",
                "rxnmapper_exception",
                f"{type(exc).__name__}: {str(exc).strip()}",
            )
            return
        mid = max(1, len(rxns) // 2)
        _map_rxnmapper_batch_recursive(rxns[:mid], output_indices[:mid], outputs)
        _map_rxnmapper_batch_recursive(rxns[mid:], output_indices[mid:], outputs)
        return

    for out_idx, result in zip(output_indices, results):
        outputs[out_idx] = parse_rxnmapper_result(result)


def batch_map_reactions_with_rxnmapper_safe(
    rxns: Iterable[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[MappingResult]:
    """Map reaction SMILES with RXNMapper and keep per-row failure reasons."""

    rxn_list = [str(rxn).strip() for rxn in rxns]
    if len(rxn_list) == 0:
        return []

    batch_size = max(1, int(batch_size))
    outputs: list[Optional[MappingResult]] = [None] * len(rxn_list)
    iterator = range(0, len(rxn_list), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="rxnmapper_batches", leave=False)

    for start in iterator:
        end = min(len(rxn_list), start + batch_size)
        batch_rxns = rxn_list[start:end]
        batch_indices = list(range(start, end))
        _map_rxnmapper_batch_recursive(batch_rxns, batch_indices, outputs)

    return [
        item if item is not None else MappingResult("", "rxnmapper_unknown")
        for item in outputs
    ]
