"""Data models and source-model helpers for FXT extraction-region generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy import ndimage

from astropy.coordinates import SkyCoord

from fxtpsfgen.mapper import ObservationPSFMapper, StackedPSFMapper
from fxtregions.measure import kernel_cumulative_curve


@dataclass
class RegionSpec:
    """Container for one DS9 region specification."""

    shape: str
    values: tuple[float, ...]
    excludes: list[str] = field(default_factory=list)


@dataclass
class TargetContext:
    """Resolved target state for region construction.

    Attributes
    ----------
    requested_coord : SkyCoord
        User-supplied sky coordinate.
    adopted_coord : SkyCoord
        Catalog coordinate when a match is found; otherwise the original
        requested coordinate.
    matched : bool
        Whether the requested coordinate matched a catalog source.
    matched_index : int | None
        Matched catalog row index, or ``None`` when unmatched.
    match_separation_arcsec : float
        Separation between the requested coordinate and the nearest catalog
        source in arcseconds.
    requested_mode : str
        User-requested extraction mode.
    effective_mode : str
        Mode actually used by the pipeline after applying the unmatched-target
        fallback policy.
    source_type : str
        Target source type inferred from the matched catalog row, or ``point``
        for unmatched targets.
    fitted_extent_sigma_pix : float
        Target extent scale in pixels inferred from the catalog ``EXT`` column.
    background_mode : str
        Origin of the local background estimate: ``map``, ``catalog``, or
        ``nearest-catalog``.
    bkg_per_pix : float
        Local background estimate in counts per pixel.
    net_counts : float | None
        Catalog ``ML_CTS_0`` value for matched targets. Unmatched targets keep
        this field as ``None`` by design.
    forced_net_counts : float
        Aperture-corrected forced-photometry estimate at the adopted target
        coordinate, in counts.
    """

    requested_coord: SkyCoord
    adopted_coord: SkyCoord
    matched: bool
    matched_index: int | None
    match_separation_arcsec: float
    requested_mode: str
    effective_mode: str
    source_type: str
    fitted_extent_sigma_pix: float
    background_mode: str
    bkg_per_pix: float
    net_counts: float | None
    forced_net_counts: float


def source_kernel(
    psf_mapper: ObservationPSFMapper | StackedPSFMapper,
    x_ima: float,
    y_ima: float,
    source_type: str,
    fitted_extent_sigma_pix: float,
) -> tuple[object, object, object]:
    """Build the local source kernel and its cumulative radial curve.

    Parameters
    ----------
    psf_mapper : ObservationPSFMapper | StackedPSFMapper
        Observation or stacked PSF mapper used to evaluate the local kernel.
    x_ima : float
        Source x coordinate in 1-based image pixels.
    y_ima : float
        Source y coordinate in 1-based image pixels.
    source_type : str
        Source type label. When equal to ``"extended"``, the point-source PSF
        is broadened by the fitted extent scale.
    fitted_extent_sigma_pix : float
        Empirical extent scale in pixels used only when ``source_type`` is
        extended.

    Returns
    -------
    kernel : np.ndarray
        Two-dimensional local source kernel.
    rr : np.ndarray
        Radial grid in pixels for the cumulative curve.
    cum : np.ndarray
        Encircled-energy fraction on ``rr``.
    """
    kernel = psf_mapper.kernel_at_position(x_ima, y_ima)
    if str(source_type).lower() == "extended" and float(fitted_extent_sigma_pix) > 0.0:
        kernel = ndimage.gaussian_filter(kernel, float(fitted_extent_sigma_pix), mode="constant", cval=0.0)
        kernel /= kernel.sum()
    rr, cum = kernel_cumulative_curve(kernel)
    return kernel, rr, cum
