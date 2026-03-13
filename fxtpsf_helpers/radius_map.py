"""Helpers for reading precomputed EEF-radius maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def load_radius_map_bundle(path: Path) -> dict[str, Any]:
    """Load a precomputed EEF-radius map product.

    Parameters
    ----------
    path : Path
        FITS file path produced by ``fxteefmap``.

    Returns
    -------
    bundle : dict[str, Any]
        Dictionary with keys ``header`` and ``maps``. ``maps`` is keyed by
        extension name, such as ``R50`` or ``R90``.
    """
    maps: dict[str, np.ndarray] = {}
    with fits.open(path) as hdul:
        primary_header = hdul[0].header.copy()
        for hdu in hdul[1:]:
            if hdu.data is None:
                continue
            maps[str(hdu.name).upper()] = np.asarray(hdu.data, dtype=np.float64)
    if not maps:
        # Single-image fallback.
        maps["R75"] = np.asarray(fits.getdata(path), dtype=np.float64)
    return {"header": primary_header, "maps": maps}


def sample_radius_map(bundle: dict[str, Any] | None, frac_name: str, x_ima: float, y_ima: float) -> float | None:
    """Sample one radius map at an image position.

    Parameters
    ----------
    bundle : dict[str, Any] | None
        Radius-map bundle returned by :func:`load_radius_map_bundle`.
    frac_name : str
        Radius-map name such as ``R50`` or ``R90``.
    x_ima : float
        X image coordinate in 1-based pixels.
    y_ima : float
        Y image coordinate in 1-based pixels.

    Returns
    -------
    radius_pix : float | None
        Sampled radius in pixels, or ``None`` when unavailable.
    """
    if bundle is None:
        return None
    maps = bundle.get("maps", {})
    arr = maps.get(str(frac_name).upper())
    if arr is None:
        return None
    x_idx = int(round(float(x_ima) - 1.0))
    y_idx = int(round(float(y_ima) - 1.0))
    if y_idx < 0 or y_idx >= arr.shape[0] or x_idx < 0 or x_idx >= arr.shape[1]:
        return None
    value = float(arr[y_idx, x_idx])
    return value if np.isfinite(value) and value > 0.0 else None
