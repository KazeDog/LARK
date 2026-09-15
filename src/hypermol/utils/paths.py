"""Project paths and server-profile-aware config placeholder expansion."""

from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from typing import Any, Mapping

import yaml


# This is the checkout from which LARK was imported.  A selected server
# profile may intentionally point PROJECT_ROOT elsewhere, but this path is
# always used to locate the tracked profile registry itself.
SOURCE_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PROJECT_ROOT = SOURCE_PROJECT_ROOT
DATA_ROOT = os.path.abspath(
    os.path.expanduser(
        os.environ.get("HYPERMOL_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
    )
)
RESULTS_ROOT = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "HYPERMOL_RESULTS_ROOT",
            os.path.join(PROJECT_ROOT, "results"),
        )
    )
)
DEFAULT_SERVER_PROFILES_PATH = os.path.join(
    SOURCE_PROJECT_ROOT,
    "configs",
    "server_profiles.yaml",
)


@dataclass(frozen=True)
class PathContext:
    """Resolved roots for one config-loading process."""

    server_name: str
    project_root: str
    data_root: str
    results_root: str
    profile_path: str

    @property
    def placeholders(self) -> dict[str, str]:
        return {
            "PROJECT_ROOT": self.project_root,
            "DATA_ROOT": self.data_root,
            "RESULTS_ROOT": self.results_root,
            "SERVER_NAME": self.server_name,
        }


def _profile_registry_path(profile_path: str | os.PathLike[str] | None = None) -> str:
    selected = (
        str(profile_path)
        if profile_path is not None
        else os.environ.get(
            "HYPERMOL_SERVER_PROFILES",
            DEFAULT_SERVER_PROFILES_PATH,
        )
    )
    return os.path.abspath(os.path.expanduser(os.path.expandvars(selected)))


def load_server_profiles(
    profile_path: str | os.PathLike[str] | None = None,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load and minimally validate the tracked server profile registry."""

    registry_path = _profile_registry_path(profile_path)
    with open(registry_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError(
            f"Server profile registry {registry_path} must contain a non-empty "
            "'profiles' mapping."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_profile in profiles.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"Empty server profile name in {registry_path}.")
        if not isinstance(raw_profile, Mapping):
            raise ValueError(
                f"Server profile {name!r} in {registry_path} must be a mapping."
            )
        missing = [
            key
            for key in ("project_root", "data_root", "results_root")
            if not str(raw_profile.get(key, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"Server profile {name!r} in {registry_path} is missing {missing}."
            )
        normalized[name] = dict(raw_profile)
    return registry_path, normalized


def detect_server_name(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    hostname: str | None = None,
) -> str | None:
    """Return the unique profile matching the current hostname."""

    host = str(hostname or socket.gethostname()).strip().lower()
    short_host = host.split(".", 1)[0]
    matches: list[str] = []
    for name, profile in profiles.items():
        aliases = {
            str(name).strip().lower(),
            *{
                str(alias).strip().lower()
                for alias in (profile.get("hostnames") or [])
                if str(alias).strip()
            },
        }
        if host in aliases or short_host in aliases:
            matches.append(str(name))
    if len(matches) > 1:
        raise ValueError(
            f"Hostname {host!r} matches multiple LARK server profiles: {matches}."
        )
    return matches[0] if matches else None


def _expand_profile_root(value: Any, replacements: Mapping[str, str]) -> str:
    text = str(value)
    for name, root in replacements.items():
        text = text.replace(f"${{{name}}}", root)
    return os.path.abspath(os.path.expanduser(os.path.expandvars(text)))


def resolve_path_context(
    server_name: str | None = None,
    *,
    profile_path: str | os.PathLike[str] | None = None,
    hostname: str | None = None,
) -> PathContext:
    """Resolve roots from a named/automatic profile with environment overrides.

    Selection precedence is ``HYPERMOL_SERVER`` > the config's ``server``
    field.  Root-specific environment variables then override the selected
    profile.  Configs without a server field retain the historical behavior.
    """

    environment_server = os.environ.get("HYPERMOL_SERVER", "").strip()
    requested = environment_server or str(server_name or "").strip()
    if not requested:
        project_root = os.path.abspath(
            os.path.expanduser(
                os.environ.get("HYPERMOL_PROJECT_ROOT", PROJECT_ROOT)
            )
        )
        data_root = os.path.abspath(
            os.path.expanduser(os.environ.get("HYPERMOL_DATA_ROOT", DATA_ROOT))
        )
        results_root = os.path.abspath(
            os.path.expanduser(
                os.environ.get(
                    "HYPERMOL_RESULTS_ROOT",
                    os.path.join(project_root, "results"),
                )
            )
        )
        return PathContext(
            server_name="legacy",
            project_root=project_root,
            data_root=data_root,
            results_root=results_root,
            profile_path="",
        )

    registry_path, profiles = load_server_profiles(profile_path)
    selected = requested
    if requested.lower() == "auto":
        selected = detect_server_name(profiles, hostname=hostname) or ""
        if not selected:
            known_hosts = sorted(
                {
                    str(alias)
                    for profile in profiles.values()
                    for alias in (profile.get("hostnames") or [])
                }
            )
            current_host = str(hostname or socket.gethostname())
            raise ValueError(
                f"No LARK server profile matches hostname {current_host!r}. "
                f"Known hostnames: {known_hosts}. Set HYPERMOL_SERVER to a "
                "profile name or add the host to the registry."
            )
    if selected not in profiles:
        raise ValueError(
            f"Unknown LARK server profile {selected!r}; available profiles: "
            f"{sorted(profiles)}."
        )

    profile = profiles[selected]
    project_root = _expand_profile_root(profile["project_root"], {})
    project_root = _expand_profile_root(
        os.environ.get("HYPERMOL_PROJECT_ROOT", project_root),
        {},
    )
    roots = {"PROJECT_ROOT": project_root}
    data_root = _expand_profile_root(
        os.environ.get("HYPERMOL_DATA_ROOT", profile["data_root"]),
        roots,
    )
    results_root = _expand_profile_root(
        os.environ.get("HYPERMOL_RESULTS_ROOT", profile["results_root"]),
        {**roots, "DATA_ROOT": data_root},
    )
    return PathContext(
        server_name=selected,
        project_root=project_root,
        data_root=data_root,
        results_root=results_root,
        profile_path=registry_path,
    )


def expand_path_placeholders(
    value: str,
    *,
    path_context: PathContext | None = None,
) -> str:
    context = path_context or resolve_path_context()
    text = str(value)
    for name, root in context.placeholders.items():
        text = text.replace(f"${{{name}}}", root)
    return os.path.expanduser(os.path.expandvars(text))


def is_probably_path(value: Any) -> bool:
    if not isinstance(value, str) or value.strip() == "":
        return False
    text = value.strip()
    if any(
        token in text
        for token in (
            "${PROJECT_ROOT}",
            "${DATA_ROOT}",
            "${RESULTS_ROOT}",
            "$HOME",
        )
    ):
        return True
    if text.startswith(("/", "./", "../", "~")):
        return True
    if "/" in text or "\\" in text:
        return True
    return text.endswith(
        (".yaml", ".yml", ".csv", ".json", ".pt", ".pth", ".mdb", ".lmdb", ".txt")
    )


def resolve_path(
    value: str,
    base_dir: str | None = None,
    *,
    path_context: PathContext | None = None,
) -> str:
    expanded = expand_path_placeholders(value, path_context=path_context)
    if expanded == "":
        return expanded
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    if base_dir:
        return os.path.abspath(os.path.join(base_dir, expanded))
    return expanded


def resolve_path_tree(
    obj: Any,
    base_dir: str | None = None,
    *,
    path_context: PathContext | None = None,
) -> Any:
    context = path_context or resolve_path_context()
    if isinstance(obj, dict):
        return {
            key: resolve_path_tree(
                value,
                base_dir=base_dir,
                path_context=context,
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            resolve_path_tree(value, base_dir=base_dir, path_context=context)
            for value in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            resolve_path_tree(value, base_dir=base_dir, path_context=context)
            for value in obj
        )
    if is_probably_path(obj):
        return resolve_path(obj, base_dir=base_dir, path_context=context)
    return obj
