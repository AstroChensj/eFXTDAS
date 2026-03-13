"""Measurement helpers for FXT extraction-region generation."""

from __future__ import annotations

import math

import numpy as np


def sample_bkg_from_map(bkg_image: np.ndarray, x_cen: float, y_cen: float) -> float:
    """Sample a 2D background map at one image position.

    Parameters
    ----------
    bkg_image : np.ndarray
        Background map in counts per pixel.
    x_cen : float
        Source x coordinate in 0-based pixels.
    y_cen : float
        Source y coordinate in 0-based pixels.

    Returns
    -------
    bkg_per_pix : float
        Sampled local background in counts per pixel.
    """
    return float(
        bkg_image[
            int(max(min(round(y_cen), bkg_image.shape[0] - 1), 0)),
            int(max(min(round(x_cen), bkg_image.shape[1] - 1), 0)),
        ]
    )


def ml_bkg_to_bkg_per_pix(ml_bkg: float, pixel_scale_arcsec: float) -> float:
    """Convert ``ML_BKG_0`` from count/arcmin2 to count/pixel.

    Parameters
    ----------
    ml_bkg : float
        Catalog ``ML_BKG_0`` value in count/arcmin2.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.

    Returns
    -------
    bkg_per_pix : float
        Local background in counts per pixel.
    """
    pixel_area_arcmin2 = (pixel_scale_arcsec / 60.0) ** 2
    return float(max(ml_bkg, 0.0) * pixel_area_arcmin2)


def kernel_cumulative_curve(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D source kernel into a cumulative radial EEF curve.

    Parameters
    ----------
    kernel : np.ndarray
        Two-dimensional non-negative source kernel, usually a local PSF or a
        PSF broadened by an empirical extent model.

    Returns
    -------
    rr : np.ndarray
        One-dimensional array of pixel radii, sorted in ascending order.
    cum : np.ndarray
        Cumulative enclosed fraction evaluated on ``rr``. The final value is
        normalized to unity.
    """
    cy = (kernel.shape[0] - 1) / 2.0
    cx = (kernel.shape[1] - 1) / 2.0
    yy, xx = np.indices(kernel.shape, dtype=np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    ww = kernel.ravel()
    order = np.argsort(rr)
    rr = rr[order]
    ww = ww[order]
    cum = np.cumsum(ww)
    return rr, np.clip(cum / max(float(cum[-1]), 1e-12), 0.0, 1.0)


def kernel_surface_profile(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute an azimuthally averaged surface-brightness profile.

    Parameters
    ----------
    kernel : np.ndarray
        Two-dimensional source kernel.

    Returns
    -------
    radii : np.ndarray
        Radial bin centers in pixels.
    prof : np.ndarray
        Mean kernel value in each annular bin.
    """
    cy = (kernel.shape[0] - 1) / 2.0
    cx = (kernel.shape[1] - 1) / 2.0
    yy, xx = np.indices(kernel.shape, dtype=np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    bins = np.arange(0.0, float(np.ceil(rr.max())) + 1.5, 1.0)
    which = np.digitize(rr.ravel(), bins) - 1
    radii = []
    prof = []
    flat = kernel.ravel()
    for idx in range(len(bins) - 1):
        mask = which == idx
        if not np.any(mask):
            continue
        radii.append(0.5 * (bins[idx] + bins[idx + 1]))
        prof.append(float(np.mean(flat[mask])))
    return np.asarray(radii, dtype=np.float64), np.asarray(prof, dtype=np.float64)


def estimate_net_counts(
    image: np.ndarray,
    bkg_per_pix: float,
    x0: float,
    y0: float,
    radius_pix: float,
    aperture_fraction: float = 1.0,
) -> float:
    """Estimate aperture-corrected net counts from a local circular aperture.

    Parameters
    ----------
    image : np.ndarray
        Counts image.
    bkg_per_pix : float
        Local background level in counts per pixel.
    x0 : float
        Source x coordinate in 0-based pixels.
    y0 : float
        Source y coordinate in 0-based pixels.
    radius_pix : float
        Aperture radius in pixels.
    aperture_fraction : float, default=1.0
        Enclosed source fraction for the chosen aperture. When smaller than
        unity, the measured aperture counts are corrected for PSF loss by
        dividing by this fraction.

    Returns
    -------
    net_counts : float
        Aperture-corrected background-subtracted counts estimate.
    """
    y_min = max(0, int(math.floor(y0 - radius_pix)))
    y_max = min(image.shape[0] - 1, int(math.ceil(y0 + radius_pix)))
    x_min = max(0, int(math.floor(x0 - radius_pix)))
    x_max = min(image.shape[1] - 1, int(math.ceil(x0 + radius_pix)))
    yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius_pix * radius_pix
    sub_img = image[y_min:y_max + 1, x_min:x_max + 1][mask]
    area_pix = float(np.count_nonzero(mask))
    aperture_net = float(np.clip(np.sum(sub_img) - bkg_per_pix * area_pix, 0.0, None))
    return aperture_net / max(float(aperture_fraction), 1e-12)
