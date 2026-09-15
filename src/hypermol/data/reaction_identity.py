"""Canonical, atom-map-invariant identities for reaction sides.

This module is deliberately lightweight so CPU preprocessing workers can import
it without importing the training stack or initializing CUDA-related modules.
"""

from __future__ import annotations

import hashlib
import re

from rdkit import Chem, rdBase


REACTION_IDENTITY_METHOD = "rdkit_canonical_map_free_isomeric_sorted_components_v2"
REACTION_IDENTITY_FALLBACK = "map_annotation_stripped_sorted_components"
MAIN_PRODUCT_IDENTITY_METHOD = "largest_heavy_atom_component_from_" + REACTION_IDENTITY_METHOD

_ATOM_MAP_ANNOTATION = re.compile(r":\d+(?=\])")


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def canonical_side_identities(smiles_text: object) -> tuple[str, str]:
    """Return ``(full_side_identity, largest_component_identity)``.

    Atom-map numbers and dot-component order are ignored.  The largest
    component is selected by heavy-atom count and is useful for strict product
    holdout when salts or small mapped byproducts differ between corpora.
    Malformed placeholders receive deterministic textual fallbacks rather than
    collapsing to a shared empty identity.
    """

    raw = _clean_text(smiles_text)
    components = [part.strip() for part in raw.split(".") if part.strip()]
    canonical_components: list[tuple[str, int]] = []
    parse_failed = not components
    if components:
        for component in components:
            try:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(component)
                if mol is None:
                    parse_failed = True
                    break
                mol = Chem.Mol(mol)
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                canonical_components.append(
                    (
                        Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                        int(mol.GetNumHeavyAtoms()),
                    )
                )
            except Exception:
                parse_failed = True
                break

    if not parse_failed:
        sorted_smiles = sorted(smiles for smiles, _ in canonical_components)
        main_smiles, _ = max(canonical_components, key=lambda item: (item[1], item[0]))
        return "rdkit|" + ".".join(sorted_smiles), "rdkit|" + main_smiles

    fallback_components = [
        _ATOM_MAP_ANNOTATION.sub("", re.sub(r"\s+", "", part))
        for part in components
    ]
    fallback_components = fallback_components or ["<empty>"]
    full_identity = "fallback|" + ".".join(sorted(fallback_components))
    main_component = max(fallback_components, key=lambda value: (len(value), value))
    return full_identity, "fallback|" + main_component


def canonical_side_identity(smiles_text: object) -> str:
    return canonical_side_identities(smiles_text)[0]


def canonical_main_component_identity(smiles_text: object) -> str:
    return canonical_side_identities(smiles_text)[1]


def identity_digest(identity: str) -> bytes:
    return hashlib.sha256(identity.encode("utf-8")).digest()


def canonical_side_digest(smiles_text: object) -> bytes:
    return identity_digest(canonical_side_identity(smiles_text))


def canonical_side_digest_bundle(smiles_text: object) -> tuple[bytes, bytes]:
    full_identity, main_identity = canonical_side_identities(smiles_text)
    return identity_digest(full_identity), identity_digest(main_identity)
