"""Molecule feature store backends.

The historical LARK datasets read molecule dictionaries from LMDB. This
module keeps that behavior and adds a pickle-backed store that can be loaded
once into memory to avoid repeated random disk reads during training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import pickle
import struct
from typing import Any, Iterator

import lmdb


PKL_FORMAT = "hypermol_molecule_store_pickle_v1"
PKL_KEY_TYPE = "sha256_hex"
PKL_VALUE_ENCODING = "pickle_bytes"
PKL_VALUE_ENCODING_ZSTD = "zstd_pickle_bytes"
PKL_END = "__hypermol_molecule_store_end__"

ZSTD_VALUE_MAGIC = b"HYPERMOL_ZSTD_V1\0"
ZSTD_VALUE_HEADER = struct.Struct(">Q")

_PICKLE_CACHE: dict[str, dict[str, Any]] = {}
_ZSTD_COMPRESSORS: dict[int, Any] = {}
_ZSTD_DECOMPRESSOR: Any = None


def _zstandard_module():
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "This molecule store contains zstd-compressed values. Install the "
            "'zstandard' package (also listed in environment.yml)."
        ) from exc
    return zstandard


def compress_store_value(payload: bytes, *, level: int = 1) -> bytes:
    """Compress one LMDB value while retaining random record access."""

    raw = bytes(payload)
    selected_level = int(level)
    compressor = _ZSTD_COMPRESSORS.get(selected_level)
    if compressor is None:
        compressor = _zstandard_module().ZstdCompressor(level=selected_level)
        _ZSTD_COMPRESSORS[selected_level] = compressor
    compressed = compressor.compress(raw)
    return ZSTD_VALUE_MAGIC + ZSTD_VALUE_HEADER.pack(len(raw)) + compressed


def decompress_store_value(payload: bytes | bytearray | memoryview) -> bytes:
    """Return the original pickle bytes for legacy or compressed values."""

    global _ZSTD_DECOMPRESSOR
    raw = bytes(payload)
    if not raw.startswith(ZSTD_VALUE_MAGIC):
        return raw
    header_start = len(ZSTD_VALUE_MAGIC)
    header_end = header_start + ZSTD_VALUE_HEADER.size
    if len(raw) < header_end:
        raise ValueError("Truncated LARK zstd value header.")
    (expected_size,) = ZSTD_VALUE_HEADER.unpack(raw[header_start:header_end])
    if _ZSTD_DECOMPRESSOR is None:
        _ZSTD_DECOMPRESSOR = _zstandard_module().ZstdDecompressor()
    restored = _ZSTD_DECOMPRESSOR.decompress(
        raw[header_end:],
        max_output_size=expected_size,
    )
    if len(restored) != expected_size:
        raise ValueError(
            "LARK zstd value size mismatch: "
            f"expected {expected_size}, restored {len(restored)}."
        )
    return restored


def smiles_key_bytes(smiles: str) -> bytes:
    return hashlib.sha256(str(smiles).encode("utf-8")).hexdigest().encode("utf-8")


def smiles_key_hex(smiles: str) -> str:
    return hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()


def normalize_store_key(key: str | bytes) -> str:
    if isinstance(key, bytes):
        return key.decode("utf-8")
    return str(key)


def store_key_bytes(key: str | bytes) -> bytes:
    if isinstance(key, bytes):
        return key
    return str(key).encode("utf-8")


def normalize_mol_payload(payload: Any) -> dict:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = pickle.loads(decompress_store_value(payload))
    if isinstance(payload, dict) and "mol_dict" in payload:
        payload = payload["mol_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Molecule payload is not a dictionary.")
    return payload


def _load_pickle_store(path: str) -> dict[str, Any]:
    path = os.path.abspath(path)
    if path in _PICKLE_CACHE:
        return _PICKLE_CACHE[path]

    with open(path, "rb") as f:
        first = pickle.load(f)

        if isinstance(first, dict) and first.get("format") == PKL_FORMAT:
            store: dict[str, Any] = {}
            while True:
                try:
                    item = pickle.load(f)
                except EOFError:
                    break
                if isinstance(item, dict) and item.get(PKL_END):
                    break
                key, value = item
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                store[str(key)] = value
        elif isinstance(first, dict) and "molecules" in first:
            molecules = first["molecules"]
            store = {
                (key.decode("utf-8") if isinstance(key, bytes) else str(key)): value
                for key, value in molecules.items()
            }
        elif isinstance(first, dict):
            store = {
                (key.decode("utf-8") if isinstance(key, bytes) else str(key)): value
                for key, value in first.items()
            }
        else:
            raise ValueError(f"Unsupported molecule pickle store format: {path}")

    _PICKLE_CACHE[path] = store
    return store


@dataclass
class MoleculeStore:
    path: str

    def __post_init__(self) -> None:
        self.path = os.path.abspath(self.path)
        self.kind = infer_store_kind(self.path)
        self.env = None
        self._pickle_store = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["env"] = None
        if self.kind == "lmdb":
            state["_pickle_store"] = None
        return state

    def _init_lmdb(self) -> None:
        if self.env is None:
            self.env = lmdb.open(
                self.path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
                max_readers=256,
            )

    def _init_pickle(self) -> None:
        if self._pickle_store is None:
            self._pickle_store = _load_pickle_store(self.path)

    def get_raw_by_key(self, key: str | bytes) -> Any:
        if self.kind == "lmdb":
            self._init_lmdb()
            with self.env.begin(write=False) as txn:
                raw = txn.get(store_key_bytes(key))
            if raw is None:
                raise KeyError(f"missing_molecule_key:{normalize_store_key(key)}")
            return decompress_store_value(raw)

        self._init_pickle()
        key = normalize_store_key(key)
        try:
            value = self._pickle_store[key]
        except KeyError as exc:
            raise KeyError(f"missing_molecule_key:{key}") from exc
        if isinstance(value, (bytes, bytearray, memoryview)):
            return decompress_store_value(value)
        return value

    def contains_key(self, key: str | bytes) -> bool:
        if self.kind == "lmdb":
            self._init_lmdb()
            with self.env.begin(write=False) as txn:
                return txn.get(store_key_bytes(key)) is not None

        self._init_pickle()
        return normalize_store_key(key) in self._pickle_store

    def get_raw(self, smiles: str) -> Any:
        try:
            return self.get_raw_by_key(smiles_key_hex(smiles))
        except KeyError as exc:
            raise KeyError(f"missing_molecule:{smiles}") from exc

    def get_mol_dict_by_key(self, key: str | bytes) -> dict:
        return normalize_mol_payload(self.get_raw_by_key(key))

    def get_mol_dict(self, smiles: str) -> dict:
        return normalize_mol_payload(self.get_raw(smiles))

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def infer_store_kind(path: str) -> str:
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix in {".mdb", ".lmdb"}:
        return "lmdb"
    raise ValueError(f"Unsupported molecule store extension for {path}; expected .mdb/.lmdb/.pkl/.pickle")


def open_molecule_store(path: str) -> MoleculeStore:
    return MoleculeStore(path)


def iter_lmdb_records(lmdb_path: str) -> Iterator[tuple[str, bytes]]:
    env = lmdb.open(
        os.path.abspath(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
        max_readers=256,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                yield str(key), decompress_store_value(value)
    finally:
        env.close()


def write_pickle_store_from_lmdb(
    lmdb_path: str,
    output_path: str,
    limit: int = 0,
    *,
    zstd_level: int = 0,
) -> dict[str, Any]:
    selected_level = int(zstd_level)
    if selected_level < 0:
        raise ValueError("zstd_level must be >= 0; use 0 to keep raw pickle bytes.")
    value_encoding = (
        PKL_VALUE_ENCODING_ZSTD if selected_level > 0 else PKL_VALUE_ENCODING
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = 0
    raw_value_bytes = 0
    stored_value_bytes = 0
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "format": PKL_FORMAT,
                "key_type": PKL_KEY_TYPE,
                "value_encoding": value_encoding,
                "source_lmdb": os.path.abspath(lmdb_path),
                "zstd_level": selected_level,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for key, raw_value in iter_lmdb_records(lmdb_path):
            stored_value = (
                compress_store_value(raw_value, level=selected_level)
                if selected_level > 0
                else raw_value
            )
            pickle.dump((key, stored_value), f, protocol=pickle.HIGHEST_PROTOCOL)
            count += 1
            raw_value_bytes += len(raw_value)
            stored_value_bytes += len(stored_value)
            if limit > 0 and count >= limit:
                break
        pickle.dump({PKL_END: True, "count": count}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "source_lmdb": os.path.abspath(lmdb_path),
        "output_path": os.path.abspath(output_path),
        "count": int(count),
        "format": PKL_FORMAT,
        "value_encoding": value_encoding,
        "zstd_level": selected_level,
        "raw_value_bytes": raw_value_bytes,
        "stored_value_bytes": stored_value_bytes,
        "compression_ratio": (
            stored_value_bytes / raw_value_bytes if raw_value_bytes else None
        ),
    }
