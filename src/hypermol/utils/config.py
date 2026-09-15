"""YAML configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict

import yaml

from hypermol.utils.paths import resolve_path_context, resolve_path_tree


@dataclass
class Config:
    raw: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_yaml_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    requested_server = data.get("server")
    if requested_server is not None and not isinstance(requested_server, str):
        raise ValueError("Top-level config field 'server' must be a profile name or 'auto'.")
    path_context = resolve_path_context(requested_server)
    base_dir = os.path.dirname(os.path.abspath(path))
    resolved = resolve_path_tree(
        data,
        base_dir=base_dir,
        path_context=path_context,
    )
    if requested_server is not None:
        # Preserve both the selected server and all concrete roots in every
        # config_resolved.json written by the training entrypoints.
        resolved["server"] = path_context.server_name
        resolved["path_roots"] = {
            "project_root": path_context.project_root,
            "data_root": path_context.data_root,
            "results_root": path_context.results_root,
            "profile_path": path_context.profile_path,
        }
    return Config(resolved)
