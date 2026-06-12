"""Configuration constants for FXT extraction-region generation."""

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
                f"Invalid override for fxtregions constant {name} via {env_name}: "
                f"expected {type(current).__name__}, got {raw!r}"
            ) from exc

# Representative EP-FXT source-position accuracy at 90% confidence from the
# mission handbook, considering PSF shape and typical source brightness. 
# Use this for target-to-catalog cross matching rather than the PSF half-power 
# diameter.
FXT_POSITION_ERR90_ARCSEC = 8.6

# Default source extraction radius, in degrees, used when auto mode falls back
# to manual mode for an undetected target.
DEFAULT_SOURCE_RADIUS_DEG = 0.00944444

# Default background annulus inner radius, in degrees, used in the same manual
# fallback branch.
DEFAULT_BKG_INNER_RADIUS_DEG = 0.0233333

# Default background annulus outer radius, in degrees, used in the same manual
# fallback branch.
DEFAULT_BKG_OUTER_RADIUS_DEG = 0.118

# Minimum allowed automatically generated source radius.
MIN_SOURCE_RADIUS_ARCSEC = 15.0

# Smallest contaminant exclusion radius allowed in either source or
# background-region carving.
MIN_EXCLUDE_RADIUS_ARCSEC = 5.0

# Neighbors closer than this are treated as blended rather than carved as
# independent confusing sources.
MIN_EXCLUDE_DIST_ARCSEC = 10.0

# Initial guess for the background annulus inner radius relative to the source
# radius in auto mode.
INITIAL_SRC_TO_BKG_INNER_RATIO = 2.0

# Target source-wing to local-background surface-brightness ratio at the inner
# edge of the background annulus.
MAX_SRC_TO_BKG_RATIO = 0.05

# Desired background annulus area relative to the source extraction area.
BACK_TO_SRC_AREA_RATIO = 10.0

# Contaminant-to-target surface-brightness threshold used when carving
# confusing sources out of the source extraction region.
MAX_CONF_TO_SRC_RATIO = 0.20

# Contaminant-to-background surface-brightness threshold used when carving
# confusing sources out of the background annulus.
MAX_CONF_TO_BACK_RATIO = 0.10

# Upper cap on the auto background inner radius in units of the local r99.
MAX_BACK_R1_TO_R99_RATIO = 3.0

# Maximum allowed width of the background annulus.
MAX_BACK_ANNULUS_WIDTH_ARCSEC = 120.0


_apply_env_overrides("FXTREGIONS_")
