from __future__ import annotations

import math

import numpy as np


def ellipse_from_pixels(
    pixels: np.ndarray,
    weights: np.ndarray,
    ellsigma: float,
) -> tuple[float, float, float, float, float]:
    """Measure a weighted centroid and ellipse from source pixels."""
    y = pixels[:, 0].astype(np.float64)
    x = pixels[:, 1].astype(np.float64)
    w = np.clip(weights.astype(np.float64), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    wsum = w.sum()
    xc = float((w * x).sum() / wsum)
    yc = float((w * y).sum() / wsum)
    dx = x - xc
    dy = y - yc
    cxx = float((w * dx * dx).sum() / wsum)
    cyy = float((w * dy * dy).sum() / wsum)
    cxy = float((w * dx * dy).sum() / wsum)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    major = ellsigma * math.sqrt(max(float(evals[0]), np.finfo(np.float64).eps))
    minor = ellsigma * math.sqrt(max(float(evals[1]), np.finfo(np.float64).eps))
    theta = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
    return xc, yc, major, minor, theta


def aperture_pixels(shape: tuple[int, int], x0: float, y0: float, radius: float) -> np.ndarray:
    """Enumerate image pixels inside a circular aperture."""
    h, w = shape
    r = max(float(radius), 1.0)
    x_min = max(0, int(math.floor(x0 - r)))
    x_max = min(w - 1, int(math.ceil(x0 + r)))
    y_min = max(0, int(math.floor(y0 - r)))
    y_max = min(h - 1, int(math.ceil(y0 + r)))
    yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r * r
    return np.column_stack((yy[mask], xx[mask])).astype(np.int64)


def net_profile(image: np.ndarray, x0: float, y0: float, bkg_per_pix: float, max_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a cumulative net-count radial profile around one source."""
    pixels = aperture_pixels(image.shape, x0, y0, max_radius)
    if len(pixels) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    yy = pixels[:, 0].astype(np.float64)
    xx = pixels[:, 1].astype(np.float64)
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    vals = image[pixels[:, 0], pixels[:, 1]].astype(np.float64) - bkg_per_pix
    order = np.argsort(rr)
    rr = rr[order]
    vals = vals[order]
    cum = np.cumsum(vals)
    cum = np.maximum(cum, 0.0)
    return rr, cum


def radius_at_fraction(rr: np.ndarray, cum: np.ndarray, frac: float) -> float:
    """Interpolate the radius where a cumulative profile reaches a target fraction."""
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 0.0:
        return 0.0
    idx = int(np.searchsorted(cum, frac * total, side="left"))
    idx = min(max(idx, 0), len(rr) - 1)
    return float(rr[idx])


def fraction_at_radius(rr: np.ndarray, frac: np.ndarray, radius: float) -> float:
    """Interpolate the encircled fraction at a target radius."""
    if len(rr) == 0:
        return 0.0
    return float(np.interp(radius, rr, frac, left=0.0, right=float(frac[-1])))


def exposure_at_position(exposure: np.ndarray | None, x: float, y: float) -> float:
    """Sample the exposure map at the nearest image pixel."""
    if exposure is None:
        return math.nan
    iy = int(np.clip(round(y - 1.0), 0, exposure.shape[0] - 1))
    ix = int(np.clip(round(x - 1.0), 0, exposure.shape[1] - 1))
    return float(exposure[iy, ix])


def mask_fraction(
    exposure: np.ndarray | None,
    x: float,
    y: float,
    radius_pix: float,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Approximate the valid-area fraction inside a source cutout."""
    if valid_mask is None and exposure is None:
        return 1.0
    ref_shape = valid_mask.shape if valid_mask is not None else exposure.shape
    pixels = aperture_pixels(ref_shape, x - 1.0, y - 1.0, radius_pix)
    if len(pixels) == 0:
        return 0.0
    yy = pixels[:, 0]
    xx = pixels[:, 1]
    if valid_mask is not None:
        return float(np.count_nonzero(valid_mask[yy, xx]) / len(pixels))
    return float(np.count_nonzero(exposure[yy, xx] > 0) / len(pixels))


def template_radius_at_fraction(template: np.ndarray, frac: float) -> float:
    """Measure the encircled-energy radius of a normalized 2D template.

    Parameters
    ----------
    template : np.ndarray
        Normalized 2D surface-brightness template.
    frac : float
        Target encircled fraction.

    Returns
    -------
    radius_pix : float
        Radius in pixels enclosing the requested fraction.
    """
    if template.size == 0:
        return 0.0
    yy, xx = np.indices(template.shape, dtype=np.float64)
    cy = 0.5 * (template.shape[0] - 1.0)
    cx = 0.5 * (template.shape[1] - 1.0)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    ww = np.clip(template.ravel(), 0.0, None)
    if float(np.sum(ww)) <= 0.0:
        return 0.0
    order = np.argsort(rr)
    rr = rr[order]
    cum = np.cumsum(ww[order])
    cum /= float(cum[-1])
    idx = int(np.searchsorted(cum, frac, side="left"))
    idx = min(max(idx, 0), len(rr) - 1)
    return float(rr[idx])
