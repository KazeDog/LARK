from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from hypermol.data.preprocess import CompoundKit


FINGERPRINT_KEYS = (
    "morgan_fp",
    "morgan2048_fp",
    "maccs_fp",
    "rdkit_fp",
    "torsion_fp",
    "atom_pair_fp",
    "layered_fp",
)


def _as_1d(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _as_2d(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _as_3d(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def ensure_atom_map_num(mol_dict: Dict[str, object]) -> Dict[str, object]:
    """Return a shallow copy with an ``atom_map_num`` vector.

    New LARK pretraining molecules keep ``map_list`` as ``map_num -> local_idx``.
    The downstream center-aware pooling code works more naturally with a padded
    per-atom map-number tensor, so we derive it here when needed.
    """
    out = dict(mol_dict)
    atom_count = int(len(out.get("atomic_num", [])))
    if out.get("map_list") is None:
        out["map_list"] = {}
    if "atom_pos" not in out:
        out["atom_pos"] = np.zeros((atom_count, 3), dtype=np.float32)
    if "atom_map_num" not in out:
        atom_map_num = np.zeros(atom_count, dtype=np.int64)
        map_list = out.get("map_list") or {}
        for map_num, local_idx in dict(map_list).items():
            local_idx = int(local_idx)
            if 0 <= local_idx < atom_count:
                atom_map_num[local_idx] = int(map_num)
        out["atom_map_num"] = atom_map_num
    return out


def merge_component_dicts(component_dicts: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Merge molecular component feature dictionaries into one disconnected graph."""
    if len(component_dicts) == 0:
        raise ValueError("component_dicts must not be empty")

    components = [ensure_atom_map_num(component) for component in component_dicts]
    if len(components) == 1:
        return dict(components[0])

    atom_int_keys = list(CompoundKit.atom_vocab_dict.keys())
    atom_float_keys = list(CompoundKit.atom_float_names)
    atom_keys = atom_int_keys + atom_float_keys + ["atom_map_num"]
    bond_keys = list(CompoundKit.bond_vocab_dict.keys())

    merged: Dict[str, object] = {}
    atom_dist_blocks = []
    optional_2d_blocks: Dict[str, list[np.ndarray]] = {"laplacian_eig": [], "rw_diag": []}
    bond_blocks = {key: [] for key in bond_keys}
    merged_map_list: Dict[int, int] = {}
    offset = 0

    for component in components:
        atom_count = int(len(component["atomic_num"]))
        for key in atom_keys:
            if key not in component and key != "atom_map_num":
                continue
            if key in atom_float_keys:
                default = np.zeros(atom_count, dtype=np.float32)
                dtype = np.float32
            else:
                default = np.zeros(atom_count, dtype=np.int64)
                dtype = np.int64
            arr = _as_1d(component.get(key, default))
            if arr.shape[0] == 0:
                arr = default
            elif arr.shape[0] == 1 and atom_count > 1:
                arr = np.repeat(arr, atom_count, axis=0)
            elif arr.shape[0] != atom_count:
                arr = np.resize(arr, atom_count)
            merged.setdefault(key, []).append(arr.astype(dtype, copy=False))

        atom_pos = component.get("atom_pos", np.zeros((atom_count, 3), dtype=np.float32))
        merged.setdefault("atom_pos", []).append(_as_2d(atom_pos).astype(np.float32, copy=False))
        merged.setdefault("edges", []).append((_as_2d(component["edges"]) + offset).astype(np.int64, copy=False))
        merged.setdefault("angles_atom_index", []).append(
            (_as_2d(component["angles_atom_index"]) + offset).astype(np.int64, copy=False)
        )
        atom_dist_blocks.append(_as_2d(component["atom_distances_2d"]))

        for key in optional_2d_blocks:
            if key in component:
                optional_2d_blocks[key].append(_as_2d(component[key]).astype(np.float32, copy=False))
        for key in bond_keys:
            if key in component:
                bond_blocks[key].append(_as_3d(component[key]).astype(np.int64, copy=False))
        for key in FINGERPRINT_KEYS:
            if key in component:
                merged.setdefault(key, []).append(_as_1d(component[key]).astype(np.int64, copy=False))
        for map_num, local_idx in dict(component.get("map_list") or {}).items():
            merged_map_list[int(map_num)] = int(local_idx) + offset
        offset += atom_count

    for key, values in list(merged.items()):
        if key in FINGERPRINT_KEYS:
            merged[key] = np.maximum.reduce(values)
        else:
            merged[key] = np.concatenate(values, axis=0)

    total_atoms = int(len(merged["atomic_num"]))
    atom_dist = np.full((total_atoms, total_atoms), -1, dtype=atom_dist_blocks[0].dtype)
    start = 0
    for block in atom_dist_blocks:
        end = start + block.shape[0]
        atom_dist[start:end, start:end] = block
        start = end
    merged["atom_distances_2d"] = atom_dist

    for key, blocks in optional_2d_blocks.items():
        if len(blocks) != len(components):
            continue
        width = blocks[0].shape[1]
        out = np.zeros((total_atoms, width), dtype=blocks[0].dtype)
        start = 0
        for block in blocks:
            end = start + block.shape[0]
            out[start:end, :] = block
            start = end
        merged[key] = out

    for key, blocks in bond_blocks.items():
        if len(blocks) != len(components):
            continue
        path_dim = blocks[0].shape[2]
        out = np.full((total_atoms, total_atoms, path_dim), -1, dtype=blocks[0].dtype)
        start = 0
        for block in blocks:
            end = start + block.shape[0]
            out[start:end, start:end, :] = block
            start = end
        merged[key] = out

    if merged_map_list:
        merged["map_list"] = merged_map_list
    return merged
