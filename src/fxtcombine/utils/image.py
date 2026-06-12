#!/usr/bin/env python3
"""

"""
import numpy as np
from astropy.wcs import WCS


def reproject_events_xy_to_refwcs(
    x_evt, y_evt,
    wcs_evt: WCS,
    wcs_ref: WCS,
    shape_ref=None,          # (ny, nx)
    weight=None,
    method="nearest",   # "nearest" or "floor"
    event_origin=1.0,
):
    """Project event coordinates onto a reference WCS image grid.

    Parameters
    ----------
    x_evt : array-like
        Event x coordinates in the event/image pixel convention.
    y_evt : array-like
        Event y coordinates in the event/image pixel convention.
    wcs_evt : astropy.wcs.WCS
        WCS describing the input event/image frame.
    wcs_ref : astropy.wcs.WCS
        WCS describing the output reference frame.
    shape_ref : tuple[int, int]
        Output image shape as ``(ny, nx)``.
    weight : array-like | None, optional
        Optional per-event weights. When omitted, each event contributes one
        count.
    method : {"nearest", "floor"}, optional
        Pixelization rule used when mapping floating-point reference
        coordinates onto integer pixels.
    event_origin : float, optional
        Pixel-origin convention of the input event coordinates. Use ``1.0``
        for FITS-style 1-based coordinates.

    Returns
    -------
    numpy.ndarray
        Reprojected image on the reference grid.
    """
    x_evt = np.asarray(x_evt, dtype=float)
    y_evt = np.asarray(y_evt, dtype=float)
    ny, nx = shape_ref

    # Astropy WCS pixel_to_world_values uses 0-based pixel coordinates.
    # FITS event/image coordinates are typically 1-based, so shift them here.
    x_evt_wcs = x_evt - float(event_origin)
    y_evt_wcs = y_evt - float(event_origin)

    # 1) event pixel -> world (deg)
    ra, dec = wcs_evt.pixel_to_world_values(x_evt_wcs, y_evt_wcs)

    # 2) world -> reference pixel
    x_ref, y_ref = wcs_ref.world_to_pixel_values(ra, dec)

    # 3) choose how to map float pixel coords to integer pixels
    if method == "nearest":
        ix = np.rint(x_ref).astype(np.int64)
        iy = np.rint(y_ref).astype(np.int64)
    elif method == "floor":
        ix = np.floor(x_ref).astype(np.int64)
        iy = np.floor(y_ref).astype(np.int64)
    else:
        raise ValueError("method must be 'nearest' or 'floor'")

    # mask inside bounds + finite
    m = np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[m]
    iy = iy[m]
    w = None if weight is None else np.asarray(weight, dtype=float)[m]

    img = np.zeros((ny, nx), dtype=np.float64 if w is not None else np.int32)
    if w is None:
        np.add.at(img, (iy, ix), 1)
    else:
        np.add.at(img, (iy, ix), w)

    return img
