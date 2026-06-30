"""APER-mode sensitivity-map calculations."""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from fxtsrcdet.utils.measure import aperture_pixels

DEFAULT_ECF = 1.3787e11


def poisson_tail_probability(observed_counts: int, background_counts: float) -> float:
    """Evaluate the Poisson probability of observing at least ``n`` counts.

    Parameters
    ----------
    observed_counts : int
        Integer aperture-count threshold.
    background_counts : float
        Expected background counts in the aperture.

    Returns
    -------
    float
        Poisson survival probability under the background-only model.
    """
    n = int(observed_counts)
    if n <= 0:
        return 1.0
    bkg = max(float(background_counts), 0.0)
    return float(special.gammainc(float(n), bkg))


def poisson_source_counts_for_likelihood(background_counts: float, likemin: float) -> float:
    """Solve the aperture-contained source counts required for ``likemin``.

    Parameters
    ----------
    background_counts : float
        Expected background counts in the aperture.
    likemin : float
        Detection likelihood threshold, defined as ``-ln(p_tail)``.

    Returns
    -------
    float
        Minimum aperture-contained source counts.
    """
    bkg = max(float(background_counts), 0.0)
    target_tail = math.exp(-float(likemin))
    low = max(0, int(math.floor(bkg)))
    high = max(1, int(math.ceil(bkg + 1.0)))
    while poisson_tail_probability(high, bkg) > target_tail:
        step = max(high - low, 1)
        low = high
        high += max(2 * step, 1)
    while high - low > 1:
        mid = (low + high) // 2
        if poisson_tail_probability(mid, bkg) <= target_tail:
            high = mid
        else:
            low = mid
    return max(float(high) - bkg, 0.0)


def compute_sensitivity_map(
    bkgmap: np.ndarray,
    expmap: np.ndarray,
    radius_pix_map: np.ndarray,
    *,
    eef: float,
    ecf: float = DEFAULT_ECF,
    likemin: float = 6.0,
) -> np.ndarray:
    """Compute a full APER-mode flux sensitivity map.

    Parameters
    ----------
    bkgmap : np.ndarray
        Expected background counts per image pixel.
    expmap : np.ndarray
        Exposure map in seconds.
    radius_pix_map : np.ndarray
        Aperture radius map in image pixels.
    eef : float
        Encircled-energy fraction captured by the aperture radius.
    ecf : float, optional
        Count-rate to energy-flux conversion in
        ``ct s^-1 / (erg cm^-2 s^-1)``.
    likemin : float, optional
        Detection likelihood threshold.

    Returns
    -------
    np.ndarray
        Flux sensitivity map in ``erg cm^-2 s^-1``.
    """
    bkg = np.asarray(bkgmap, dtype=np.float64)
    exp = np.asarray(expmap, dtype=np.float64)
    radius = np.asarray(radius_pix_map, dtype=np.float64)
    if bkg.shape != exp.shape or bkg.shape != radius.shape:
        raise ValueError("bkgmap, expmap, and radius_pix_map must have identical shapes.")
    if not 0.0 < float(eef) <= 1.0:
        raise ValueError("eef must be in (0, 1].")
    if float(ecf) <= 0.0:
        raise ValueError("ecf must be positive.")

    sensitivity = np.full(bkg.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(bkg) & (bkg >= 0.0) & np.isfinite(exp) & (exp > 0.0)
    valid &= np.isfinite(radius) & (radius > 0.0)
    for y_idx, x_idx in np.argwhere(valid):
        pixels = aperture_pixels(bkg.shape, float(x_idx), float(y_idx), float(radius[y_idx, x_idx]))
        if len(pixels) == 0:
            continue
        aperture_bkg = float(np.nansum(bkg[pixels[:, 0], pixels[:, 1]]))
        source_ap_counts = poisson_source_counts_for_likelihood(aperture_bkg, likemin)
        denom = float(exp[y_idx, x_idx]) * float(ecf) * float(eef)
        if denom > 0.0:
            sensitivity[y_idx, x_idx] = source_ap_counts / denom
    return sensitivity
