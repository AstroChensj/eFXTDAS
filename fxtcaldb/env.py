"""Environment helpers for the local EP/FXT CALDB."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _require_env(var_name: str, explicit: str | None = None) -> str:
    """Return a required environment variable or explicit override."""
    value = explicit or os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable {var_name}")
    return value


def _join_caldb(root: str, *parts: str) -> str:
    """Build a normalized path under the CALDB root."""
    return os.path.normpath(os.path.join(root, *parts))


def _parse_caldb_config(root: str, config_path: str, mission_tag: str) -> tuple[str, str]:
    """Read the CALDB config entry for one mission/instrument block."""
    with open(config_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.startswith(mission_tag):
                continue
            parts = stripped.split()
            if len(parts) < 7:
                raise RuntimeError(
                    f"Malformed CALDB config line: {line.strip()} (expected at least 7 tokens)"
                )
            index_path = _join_caldb(root, parts[3], parts[4])
            data_root = _join_caldb(root, parts[6])
            return index_path, data_root
    raise RuntimeError(f"No CALDB config line starting with '{mission_tag}' found in {config_path}")


@dataclass(frozen=True)
class CaldbPaths:
    """Resolved filesystem paths for the local EP/FXT CALDB."""

    root: str
    config: str
    index: str
    data_root: str

    @classmethod
    def resolve(
        cls,
        caldb_root: str | None = None,
        caldb_config: str | None = None,
        mission_tag: str = "EP FXT",
    ) -> "CaldbPaths":
        """Resolve the active CALDB root, config, and index file."""
        root = os.path.abspath(_require_env("CALDB", caldb_root))
        config = os.path.abspath(_require_env("CALDBCONFIG", caldb_config))
        index_path, data_root = _parse_caldb_config(root, config, mission_tag)
        return cls(root=root, config=config, index=index_path, data_root=data_root)
