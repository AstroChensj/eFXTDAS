"""Internal configuration constants for ``fxteefmap``."""

from __future__ import annotations

import os


def _parse_override(name: str, current: object, raw: str) -> object:
    """Parse an environment override using the current constant type."""
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return raw
    if isinstance(current, tuple):
        if not current:
            raise ValueError("empty tuple constants are not overrideable")
        elem_types = {type(item) for item in current}
        if elem_types <= {int}:
            caster = int
        elif elem_types <= {int, float}:
            caster = float
        else:
            raise ValueError("only numeric tuple constants are overrideable")
        parts = [item.strip() for item in raw.split(",")]
        if not parts or any(not item for item in parts):
            raise ValueError(f"expected a comma-separated tuple, got {raw!r}")
        return tuple(caster(item) for item in parts)
    raise ValueError(f"unsupported override type {type(current).__name__}")


def _apply_env_overrides(prefix: str) -> None:
    """Override uppercase constants from environment variables."""
    for name, current in list(globals().items()):
        if not name.isupper() or name.startswith("_"):
            continue
        env_name = f"{prefix}{name}"
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        try:
            globals()[name] = _parse_override(name, current, raw)
        except Exception as exc:  # pragma: no cover - exercised via import failure paths
            raise ValueError(
                f"Invalid override for fxteefmap constant {name} via {env_name}: "
                f"expected {type(current).__name__}, got {raw!r}"
            ) from exc


DEFAULT_PIXEL_SCALE_ARCSEC = 9.6


_apply_env_overrides("FXTEEFMAP_")
