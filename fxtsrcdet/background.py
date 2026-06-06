from __future__ import annotations

import math

import numpy as np

from fxtpsfgen.mapper import ObservationPSFMapper, StackedPSFMapper
from fxtsrcdet.config import (
    BACKGROUND_CARVE_MIN_COUNTS,
    BACKGROUND_CARVE_MIN_SUPPORT_SCALES,
    BACKGROUND_CARVE_MIN_RADIUS_PIX,
    BACKGROUND_CARVE_R90_FACTOR,
    BACKGROUND_CARVE_SCALE_FACTOR,
    BACKGROUND_MIN_SUPPORT_WEIGHT,
    BACKGROUND_RATE_CAP_FACTOR,
    BACKGROUND_RATE_CAP_PERCENTILE,
    BACKGROUND_SIGMA_FLOOR_PIX,
    BACKGROUND_TARGET_COUNTS,
    EPS,
)
from fxtsrcdet.utils.imageops import smooth_image
from fxtsrcdet.utils.measure import aperture_pixels


def create_background_map(
    image: np.ndarray,
    rows: list,
    pixel_scale_arcsec: float,
    psf_mapper: ObservationPSFMapper | StackedPSFMapper | None = None,
    exposure_map: np.ndarray | None = None,
    analysis_mask: np.ndarray | None = None,
    sigma_grid: tuple[float, ...] | list[float] | np.ndarray = (4.0, 8.0, 16.0, 32.0, 64.0),
) -> np.ndarray:
    """Build an exposure-aware, source-masked smoothed background map.

    Parameters
    ----------
    image : np.ndarray
        Counts image.
    rows : list[dict]
        Initial detection candidates used to carve the source mask.
    pixel_scale_arcsec : float
        Image pixel scale in arcsec/pixel.
    psf_mapper : ObservationPSFMapper | StackedPSFMapper | None
        PSF mapper used to query local PSF radii.
    exposure_map : np.ndarray | None
        Optional exposure map matched to ``image``.
    analysis_mask : np.ndarray | None
        Optional boolean mask selecting globally valid pixels for background
        estimation.
    sigma_grid : tuple[float, ...] | list[float] | np.ndarray
        Gaussian smoothing scales in pixels available to the adaptive
        background model.

    Returns
    -------
    background : np.ndarray
        Smoothed background map in counts/pixel.

    Notes
    -----
    The algorithm is a simplified, exposure-aware source-masked background
    estimator:

    1. Build a valid-pixel mask from the exposure map, or use the full image
       when no exposure map is provided.
    2. Carve out detected sources from that valid mask using a radius tied to
       the local PSF size and the wavelet scale of each candidate.
    3. Smooth the masked counts image and the masked exposure image on a grid
       of Gaussian smoothing scales.
    4. At each smoothing scale, convert the smoothed counts and smoothed
       exposure into a local background rate, then back into counts/pixel.
    5. Estimate the effective background-count support at each scale.
    6. For each pixel, choose the smallest smoothing scale that reaches the
       target support level, with interpolation between neighboring scales to
       avoid sharp boundaries.
    7. Zero the result only in globally invalid pixels.

    This produces a background map that preserves more structure where local
    support is strong, while automatically switching to broader smoothing where
    the source-free background is sparse.
    """
    if psf_mapper is None:
        raise ValueError("A PSF mapper is required for PSF-aware background estimation.")

    #--- valid region for bkg estimation: must have non-zero exposure
    if exposure_map is not None:
        valid_mask = exposure_map > 0.0
    else:
        valid_mask = np.ones_like(image, dtype=bool)
    if analysis_mask is not None:
        valid_mask &= np.asarray(analysis_mask, dtype=bool)

    #--- valid region for bkg estimation: must carve detected sources out
    source_free_mask = valid_mask.copy()
    for row in rows:
        if len(getattr(row, "support_scales", [])) < BACKGROUND_CARVE_MIN_SUPPORT_SCALES:
            continue
        if float(getattr(row, "counts", 0.0)) < BACKGROUND_CARVE_MIN_COUNTS:
            continue
        x_ima = float(row.x)
        y_ima = float(row.y)
        local_psf_r90_pix = psf_mapper.radius_at_position(x_ima, y_ima, 0.90)
        x0 = float(x_ima - 1.0)
        y0 = float(y_ima - 1.0)
        ##--- carving out radius dependent on local R90 size, and source extendedness
        carve_radius = max(
            BACKGROUND_CARVE_R90_FACTOR * float(local_psf_r90_pix),
            BACKGROUND_CARVE_SCALE_FACTOR * float(row.scale),
            BACKGROUND_CARVE_MIN_RADIUS_PIX,
        )
        pixels = aperture_pixels(image.shape, x0, y0, carve_radius)
        if len(pixels) > 0:
            source_free_mask[pixels[:, 0], pixels[:, 1]] = False

    masked_weight = source_free_mask.astype(np.float64)
    if exposure_map is not None:
        masked_counts = image.astype(np.float64, copy=False) * masked_weight
        masked_exposure = exposure_map.astype(np.float64, copy=False) * masked_weight
        source_free_rate = np.divide(
            masked_counts,
            np.maximum(masked_exposure, EPS),
            out=np.zeros_like(masked_counts, dtype=np.float64),
            where=masked_exposure > 0.0,
        )
        rate_samples = source_free_rate[source_free_mask & valid_mask]
        if rate_samples.size > 0:
            rate_cap = max(
                float(np.percentile(rate_samples, BACKGROUND_RATE_CAP_PERCENTILE)) * BACKGROUND_RATE_CAP_FACTOR,
                EPS,
            )
        else:
            rate_cap = np.inf
    else:
        masked_image = image.astype(np.float64, copy=False) * masked_weight

    #--- try different smoothing scales to find the one with highest spatial resolution, and at the same time enough effective counts support for stable estimation
    # small smoothing scales preserve detail but may have too few counts, so the background gets noisy/spiky
    # large smoothing scales are stable but blur structure too much
    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
    if sigma_grid.size == 0:
        raise ValueError("sigma_grid must contain at least one smoothing scale.")
    sigma_grid = np.array(
        sorted({max(float(sigma), BACKGROUND_SIGMA_FLOOR_PIX) for sigma in sigma_grid}),
        dtype=np.float64,
    )
    target_counts = BACKGROUND_TARGET_COUNTS
    min_support_weight = BACKGROUND_MIN_SUPPORT_WEIGHT
    models: list[np.ndarray] = []
    counts_support: list[np.ndarray] = []
    for sigma in sigma_grid:
        smoothed_weight = smooth_image(masked_weight, float(sigma))
        if exposure_map is not None:
            smoothed_counts = smooth_image(masked_counts, float(sigma)) # spilling counts to fill in the carved-out holes 
            smoothed_exposure = smooth_image(masked_exposure, float(sigma)) # spilling exposure to fill in the carved-out holes
            rate_model = np.divide(
                smoothed_counts,
                np.maximum(smoothed_exposure, EPS),
                out=np.zeros_like(smoothed_counts, dtype=np.float64),
                where=smoothed_exposure > 0.0,
            )   # ratio: so the carved-out hole will get normal bkg rate, estimated from neighborhood
            # NOTE: the carved out hole cannot be too large, otherwise some fraction of holes will not be spilled, and rate_model is still 0!
            rate_model = np.clip(rate_model, 0.0, rate_cap) # force rate_model to lie between 0 and rate_cap
            counts_model = rate_model * exposure_map    # smoothed bkg counts model (full image), in counts/pixel
        else:
            smoothed_image = smooth_image(masked_image, float(sigma))
            counts_model = np.divide(
                smoothed_image,
                np.maximum(smoothed_weight, EPS),
                out=np.zeros_like(smoothed_image, dtype=np.float64),
                where=smoothed_weight > min_support_weight,
            )
        eff_area = 2.0 * math.pi * float(sigma) * float(sigma) * np.maximum(smoothed_weight, 0.0)   # effective usable area around each pixel at that smoothing scale; this is a 2d image
        counts_support.append(np.clip(counts_model * eff_area, 0.0, None))  # number of background counts supporting that local estimate; this is a 2d image
        models.append(counts_model)

    #--- adaptive multi-scale background smoother driven by local effective background counts
    ##--- stack all trial scales
    support_cube = np.stack(counts_support, axis=0)
    model_cube = np.stack(models, axis=0)
    ##--- ask where support is sufficient
    ge_target = support_cube >= target_counts
    ##--- find the first scale that reaches the target
    first_idx = np.argmax(ge_target, axis=0)
    hit = np.any(ge_target, axis=0) # whether the target was ever reached at all
    ##--- define the bracketing scales (for later interpolation)
    lower_idx = np.clip(first_idx - 1, 0, len(sigma_grid) - 1)
    upper_idx = first_idx
    ##--- extract support/model at those two scales
    lower_support = np.take_along_axis(support_cube, lower_idx[None, ...], axis=0)[0]
    upper_support = np.take_along_axis(support_cube, upper_idx[None, ...], axis=0)[0]
    lower_model = np.take_along_axis(model_cube, lower_idx[None, ...], axis=0)[0]
    upper_model = np.take_along_axis(model_cube, upper_idx[None, ...], axis=0)[0]
    ##--- interpolate between the two scales (to avoid discontinuity)
    denom = np.maximum(upper_support - lower_support, EPS)
    alpha = np.clip((target_counts - lower_support) / denom, 0.0, 1.0)
    background = (1.0 - alpha) * lower_model + alpha * upper_model
    background = np.where(hit, background, model_cube[-1])  # if target is never reached, use the broadest model
    ##--- reject invalid detector pixels only; low-support pixels fall back to
    ## the broadest-scale model rather than being forced to zero.
    background[~valid_mask] = 0.0
    
    return np.clip(background, 0.0, None)
