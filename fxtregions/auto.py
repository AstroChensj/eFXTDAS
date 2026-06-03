"""Auto-sizing heuristics for FXT extraction regions."""

from __future__ import annotations

import math
import numpy as np

from fxtregions.config import (
    BACK_TO_SRC_AREA_RATIO,
    INITIAL_SRC_TO_BKG_INNER_RATIO,
    MAX_BACK_ANNULUS_WIDTH_ARCSEC,
    MAX_BACK_R1_TO_R99_RATIO,
    MAX_CONF_TO_SRC_RATIO,
    MAX_SRC_TO_BKG_RATIO,
    MIN_EXCLUDE_DIST_ARCSEC,
    MIN_EXCLUDE_RADIUS_ARCSEC,
    MIN_SOURCE_RADIUS_ARCSEC,
)


def surface_brightness_at_radius(
    radius_pix: float,
    profile_r: np.ndarray,
    profile_sb: np.ndarray,
    total_counts: float,
) -> float:
    """Evaluate the model surface brightness at a given radius.

    Parameters
    ----------
    radius_pix : float
        Radius in pixels at which to evaluate the profile.
    profile_r : np.ndarray
        Radial grid in pixels.
    profile_sb : np.ndarray
        Surface-brightness profile normalized per unit source count.
    total_counts : float
        Source counts used to scale the normalized profile.

    Returns
    -------
    sb : float
        Surface brightness at ``radius_pix`` in counts per pixel.

    Notes
    -----
    The function linearly interpolates the azimuthally averaged profile and
    scales it by the assumed total source counts. It is used to mimic the
    srctool logic of comparing contaminant or target source wings against either
    local background or target-source surface brightness.
    """
    if len(profile_r) == 0 or len(profile_sb) == 0:
        return 0.0
    sb = float(np.interp(radius_pix, profile_r, profile_sb, left=float(profile_sb[0]), right=0.0))
    return max(float(total_counts), 0.0) * sb


def auto_source_radius_pix(
    net_counts: float,
    bkg_per_pix: float,
    rr: np.ndarray,
    cum: np.ndarray,
    psf_r99_pix: float,
    pixel_scale_arcsec: float,
) -> float:
    """Choose the source extraction radius that maximizes approximate SNR.

    Parameters
    ----------
    net_counts : float
        Nominal source counts for the target source.
    bkg_per_pix : float
        Local background level in counts per pixel.
    rr : np.ndarray
        Radial grid in pixels for the source cumulative curve.
    cum : np.ndarray
        Encircled-energy fraction on ``rr``.
    psf_r99_pix : float
        Local ``r99`` in pixels, used as an upper cap.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.

    Returns
    -------
    radius_pix : float
        Source extraction radius in pixels.

    Notes
    -----
    Trial radii are scanned from the configured minimum source radius up to the
    local ``r99``. For each trial radius, the expected source counts are
    estimated from the cumulative curve and the expected background counts from
    the local scalar background. The chosen radius maximizes the approximate
    signal-to-noise ratio ``src / sqrt(src + bkg)``.
    """
    min_radius = max(MIN_SOURCE_RADIUS_ARCSEC / pixel_scale_arcsec, 1.0)
    max_radius = max(min(float(rr[-1]), float(psf_r99_pix)), min_radius)
    trial = np.arange(min_radius, max_radius + 0.5, 0.5)    # in pixels
    frac = np.interp(trial, rr, cum, left=0.0, right=1.0)
    src = net_counts * frac
    bkg = bkg_per_pix * math.pi * trial * trial
    snr = src / np.sqrt(np.maximum(src + bkg, 1e-12))
    print(float(trial[int(np.argmax(snr))]))
    return float(trial[int(np.argmax(snr))])


def auto_background_annulus_pix(
    net_counts: float,
    bkg_per_pix: float,
    profile_r: np.ndarray,
    profile_sb: np.ndarray,
    source_radius_pix: float,
    psf_r99_pix: float,
    pixel_scale_arcsec: float,
) -> tuple[float, float]:
    """Choose the background annulus using srctool-like heuristics.

    Parameters
    ----------
    net_counts : float
        Nominal source counts for the target source.
    bkg_per_pix : float
        Local background level in counts per pixel.
    profile_r : np.ndarray
        Radial grid in pixels for the target-source surface-brightness profile.
    profile_sb : np.ndarray
        Surface-brightness profile normalized per unit source count.
    source_radius_pix : float
        Source extraction radius in pixels.
    psf_r99_pix : float
        Local ``r99`` in pixels.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.

    Returns
    -------
    inner : float
        Background annulus inner radius in pixels.
    outer : float
        Background annulus outer radius in pixels.

    Notes
    -----
    The inner radius is the first radius where the source surface brightness
    drops below a fraction of the local background, subject to floors and caps
    based on the source radius and local ``r99``. The outer radius is then set
    so that the annulus area is a fixed multiple of the source-region area,
    with a maximum width cap. This follows the same policy direction as eSASS
    srctool without reusing its exact implementation.
    """
    target_sb = MAX_SRC_TO_BKG_RATIO * max(bkg_per_pix, 1e-12)
    absolute_sb = net_counts * profile_sb
    idx = np.where(absolute_sb <= target_sb)[0]
    inner_floor = max(INITIAL_SRC_TO_BKG_INNER_RATIO * source_radius_pix, source_radius_pix + 1.0)
    inner_cap = max(MAX_BACK_R1_TO_R99_RATIO * psf_r99_pix, inner_floor)
    if len(idx) == 0:
        inner = inner_cap
    else:
        inner = max(inner_floor, float(profile_r[idx[0]]))
        inner = min(inner, inner_cap)
    required_area = BACK_TO_SRC_AREA_RATIO * math.pi * source_radius_pix * source_radius_pix
    outer = math.sqrt(inner * inner + required_area / math.pi)
    max_width_pix = MAX_BACK_ANNULUS_WIDTH_ARCSEC / pixel_scale_arcsec
    if outer - inner > max_width_pix:
        outer = inner + max_width_pix
    outer = max(outer, inner + 1.0)
    return float(inner), float(outer)


def confusing_source_radius_pix(
    sep_pix: float,
    conf_counts: float,
    conf_profile_r: np.ndarray,
    conf_profile_sb: np.ndarray,
    threshold_sb: float,
    pixel_scale_arcsec: float,
    conf_r99_pix: float,
    target_counts: float | None = None,
    target_profile_r: np.ndarray | None = None,
    target_profile_sb: np.ndarray | None = None,
) -> float:
    """Compute the exclusion radius for a confusing neighbouring source.

    Parameters
    ----------
    sep_pix : float
        Separation between the target and the confusing neighbour in pixels.
    conf_counts : float
        Nominal counts of the confusing source.
    conf_profile_r : np.ndarray
        Radial grid in pixels for the confusing-source profile.
    conf_profile_sb : np.ndarray
        Surface-brightness profile of the confusing source, normalized per unit
        source count.
    threshold_sb : float
        Background-based surface-brightness threshold in counts per pixel. This
        is used for background-annulus exclusion.
    pixel_scale_arcsec : float
        Pixel scale in arcsec per pixel.
    conf_r99_pix : float
        Local ``r99`` of the confusing source in pixels.
    target_counts : float | None, optional
        Target source counts. When supplied together with the target profile,
        the exclusion radius is solved against the target source brightness
        rather than a background threshold.
    target_profile_r : np.ndarray | None, optional
        Radial grid for the target-source profile.
    target_profile_sb : np.ndarray | None, optional
        Surface-brightness profile of the target source, normalized per unit
        source count.

    Returns
    -------
    radius_pix : float
        Exclusion radius in pixels. A value of ``0`` means that the neighbour is
        too close to be treated as a separate confusing source.

    Notes
    -----
    For source-region exclusion, the confusing source is suppressed until its
    wing surface brightness falls below a fraction of the target source surface
    brightness evaluated along the line joining the two sources. For
    background-annulus exclusion, the target is replaced by a constant
    background threshold. The final radius is clipped between the configured
    minimum exclusion radius and the neighbour's local ``r99``.
    """
    min_radius = MIN_EXCLUDE_RADIUS_ARCSEC / pixel_scale_arcsec
    #--- 
    if sep_pix <= MIN_EXCLUDE_DIST_ARCSEC / pixel_scale_arcsec:
        return 0.0
    #--- iterate to find the maximum radius that satisfy the brightness threshold
    trial = np.arange(min_radius, max(float(conf_r99_pix), min_radius) + 0.5, 0.5)
    for radius in trial:
        conf_sb = surface_brightness_at_radius(radius, conf_profile_r, conf_profile_sb, conf_counts)
        ##--- mode 1: confusing source in target source extraction region
        # then the threshold is based on target source brightness at the confusing source position
        if target_counts is not None and target_profile_r is not None and target_profile_sb is not None:
            target_sb = surface_brightness_at_radius(
                max(sep_pix - radius, 0.0),
                target_profile_r,
                target_profile_sb,
                target_counts,
            )
            limit = MAX_CONF_TO_SRC_RATIO * max(target_sb, 1e-12)
        ##--- mode 2: confusing source in target source background annulus
        # then the threshold is based on the local background surface brightness
        else:
            limit = threshold_sb
        if conf_sb <= limit:
            return float(np.clip(radius, min_radius, conf_r99_pix))
    return float(max(float(conf_r99_pix), min_radius))
