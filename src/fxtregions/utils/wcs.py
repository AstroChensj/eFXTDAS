"""WCS helpers for FXT extraction-region generation."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def infer_pixel_scale_arcsec(wcs: WCS) -> float:
    """Compute the geometric mean pixel scale from WCS.

    Parameters
    ----------
    wcs : WCS
        Celestial WCS object.

    Returns
    -------
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.
    """
    scales = proj_plane_pixel_scales(wcs.celestial)
    return float(np.sqrt(abs(scales[0] * scales[1])) * 3600.0)
