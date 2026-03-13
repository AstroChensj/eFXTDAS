from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

from fxtsrcdet.models import CatalogRow


def augment_rows_with_wcs(rows: list[CatalogRow], wcs: Any | None) -> list[CatalogRow]:
    """Add sky coordinates and angular sizes to source rows.

    Parameters
    ----------
    rows : list[CatalogRow]
        Catalog rows to augment in place.
    wcs : Any | None
        Celestial WCS object. If ``None``, the input rows are returned unchanged.

    Returns
    -------
    list[CatalogRow]
        The input rows, updated with sky coordinates and angular-size columns when
        a usable celestial WCS is available.
    """
    if wcs is None:
        return rows
    pixel_scales_deg = proj_plane_pixel_scales(wcs)
    if len(pixel_scales_deg) < 2:
        return rows
    x_pix = np.array([row.x_ima for row in rows], dtype=np.float64)
    y_pix = np.array([row.y_ima for row in rows], dtype=np.float64)
    ra_deg, dec_deg = wcs.all_pix2world(x_pix, y_pix, 1)
    major_scale_arcsec = abs(float(pixel_scales_deg[0])) * 3600.0
    minor_scale_arcsec = abs(float(pixel_scales_deg[1])) * 3600.0
    for idx, row in enumerate(rows):
        row.ra = float(ra_deg[idx])
        row.dec = float(dec_deg[idx])
        row.major_arcsec = float(row.major * major_scale_arcsec)
        row.minor_arcsec = float(row.minor * minor_scale_arcsec)
        row.radius_arcsec = float(math.sqrt(row.major_arcsec * row.minor_arcsec))
    return rows


def infer_pixel_scale_arcsec(wcs: Any | None, fallback: float) -> float:
    """Infer the image pixel scale from WCS metadata.

    Parameters
    ----------
    wcs : Any | None
        Celestial WCS object. If ``None``, ``fallback`` is returned.
    fallback : float
        Pixel scale to use when the WCS is unavailable or unusable.

    Returns
    -------
    float
        Geometric-mean sky pixel scale in arcsec per pixel.
    """
    if wcs is None:
        return float(fallback)
    pixel_scales_deg = proj_plane_pixel_scales(wcs)
    if len(pixel_scales_deg) < 2:
        return float(fallback)
    pixel_scale_arcsec = float(np.sqrt(abs(pixel_scales_deg[0] * pixel_scales_deg[1])) * 3600.0)
    return pixel_scale_arcsec if pixel_scale_arcsec > 0 else float(fallback)
