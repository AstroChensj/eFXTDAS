"""Library of dataclass models for detection candidates, fit measurements, and catalog rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np


class FieldAccessMixin:
    """Provide compatibility-style field access for dataclass records."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extras = payload.pop("extras", {})
        payload.update(extras)
        return payload

    @classmethod
    def field_names(cls) -> set[str]:
        return {item.name for item in fields(cls)}


@dataclass
class DetectionCandidate(FieldAccessMixin):
    """Wavelet-detection candidate before catalog fitting."""

    id: int = 0
    peak_x: float = 0.0
    peak_y: float = 0.0
    x: float = 0.0
    y: float = 0.0
    major: float = 0.5
    minor: float = 0.5
    theta_deg: float = 0.0
    npix: int = 0
    counts: float = 0.0
    net_counts: float = 0.0
    scale: float = 1.0
    support_scales: list[float] = field(default_factory=list)
    min_significance: float = 1.0
    wavelet_peak_score: float = 0.0
    z_peak: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FitMeasurement(FieldAccessMixin):
    """Intermediate PSF-fit measurement for one detection candidate."""

    candidate: DetectionCandidate
    det_bkg: float = 0.0
    bkg_per_pix: float = 0.0
    det_like: float = 0.0
    ext_like: float = 0.0
    best_sigma: float = 0.0
    r50_meas: float = 0.0
    r80_meas: float = 0.0
    r90_meas: float = 0.0
    point_amp: float = 0.0
    ext_amp: float = 0.0
    dx_pt: float = 0.0
    dy_pt: float = 0.0
    dx_ext: float = 0.0
    dy_ext: float = 0.0
    theta_arcmin: float = 0.0
    psf_r50_pix_nom: float = 0.0
    psf_r75_pix_nom: float = 0.0
    psf_r80_pix_nom: float = 0.0
    psf_r90_pix_nom: float = 0.0
    ext_r75_pix: float = 0.0
    stamp_radius_pix: float = 0.0
    group_stamp_radius_pix: float = 0.0
    ml_eff: float = 0.0
    fit_maskfrac: float = 1.0
    has_strong_neighbor: bool = False
    group_id: int = -1
    group_size: int = 1
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CatalogRow(FieldAccessMixin):
    """Final source catalog row.

    Attributes
    ----------
    id : int
        Internal running index used while constructing the final catalog list.
    id_src : int
        Final source number written to the science catalog.
    id_band : int
        Band identifier. The current pipeline writes ``0`` for total-band results.
    id_cluster : int
        Cluster/group identifier in the public catalog schema.
    source_type : str
        Final source classification: ``background``, ``point``, or ``extended``.
    ra : float
        Right ascension in degrees.
    ra_lowerr : float
        Lower one-sigma right-ascension error in arcsec.
    ra_uperr : float
        Upper one-sigma right-ascension error in arcsec.
    dec : float
        Declination in degrees.
    dec_lowerr : float
        Lower one-sigma declination error in arcsec.
    dec_uperr : float
        Upper one-sigma declination error in arcsec.
    radec_err : float
        Combined positional error in arcsec.
    lii : float
        Galactic longitude in degrees.
    bii : float
        Galactic latitude in degrees.
    dist_nn : float
        Distance to the nearest retained neighbor in arcsec.
    x_ima : float
        Final fitted x coordinate in image pixels, using FITS 1-based convention.
    x_ima_err : float
        One-sigma x-position error in image pixels.
    x_ima_lowerr : float
        Lower one-sigma x-position error in image pixels.
    x_ima_uperr : float
        Upper one-sigma x-position error in image pixels.
    y_ima : float
        Final fitted y coordinate in image pixels, using FITS 1-based convention.
    y_ima_err : float
        One-sigma y-position error in image pixels.
    y_ima_lowerr : float
        Lower one-sigma y-position error in image pixels.
    y_ima_uperr : float
        Upper one-sigma y-position error in image pixels.
    ml_cts : float
        Best-fit source counts from the local likelihood model.
    ml_cts_err : float
        One-sigma error estimate on ``ml_cts``.
    ml_cts_lowerr : float
        Lower one-sigma error estimate on ``ml_cts``.
    ml_cts_uperr : float
        Upper one-sigma error estimate on ``ml_cts``.
    ml_rate : float
        Vignetting-corrected source count rate in counts per second.
    ml_rate_err : float
        One-sigma error estimate on ``ml_rate``.
    ml_rate_lowerr : float
        Lower one-sigma error estimate on ``ml_rate``.
    ml_rate_uperr : float
        Upper one-sigma error estimate on ``ml_rate``.
    ml_flux : float
        Source flux derived from ``ml_rate`` and the user-supplied ECF.
    ml_flux_err : float
        One-sigma error estimate on ``ml_flux``.
    ml_flux_lowerr : float
        Lower one-sigma error estimate on ``ml_flux``.
    ml_flux_uperr : float
        Upper one-sigma error estimate on ``ml_flux``.
    ext : float
        Best-fit source extent in arcsec.
    ext_err : float
        One-sigma error estimate on ``ext``.
    ext_lowerr : float
        Lower one-sigma error estimate on ``ext``.
    ext_uperr : float
        Upper one-sigma error estimate on ``ext``.
    det_like : float
        Detection likelihood from the final PSF-aware fit.
    ext_like : float
        Extent likelihood from the final point-versus-extended comparison.
    ml_bkg : float
        Local fitted background surface brightness in counts per arcmin^2.
    ml_exp : float
        Exposure at the fitted source position in seconds.
    ml_radius : float
        Single-source fit stamp radius in arcsec.
    maskfrac : float
        Fraction of the single-source fit region covered by valid exposure.
    ml_eff : float
        Fraction of the PSF enclosed within the adopted catalog radius.
    scale : float
        Representative wavelet scale associated with the candidate.
    support_scales : list[float]
        Wavelet scales that contributed support to this source.
    wavelet_peak_score : float
        Wavelet-stage peak ranking statistic used for ordering and neighbor checks.
    min_significance : float
        Smallest pixel significance retained within the wavelet support.
    npix : int
        Number of source-mask pixels assigned to the wavelet candidate.
    counts : float
        Raw counts inside the wavelet detection support.
    net_counts : float
        Approximate net counts associated with the wavelet detection support.
    bkg_counts : float
        Approximate background counts associated with the wavelet detection support.
    major : float
        Semi-major axis of the wavelet-derived ellipse in pixels.
    minor : float
        Semi-minor axis of the wavelet-derived ellipse in pixels.
    theta_deg : float
        Position angle of the wavelet-derived ellipse in degrees.
    group_id : int
        Internal grouped-fit identifier.
    group_size : int
        Number of sources in the local grouped point-source fit.
    group_stamp_radius_pix : float
        Radius of the grouped-fit stamp in pixels.
    theta_arcmin : float
        Off-axis angle from the optical axis in arcmin.
    psf_r50_pix : float
        Local PSF r50 radius in pixels.
    psf_r75_pix : float
        Local PSF r75 radius in pixels.
    psf_r80_pix : float
        Local PSF r80 radius in pixels.
    psf_r90_pix : float
        Local PSF r90 radius in pixels.
    psf_instrument : str
        Instrument identifier used for PSF calibration lookup.
    psf_filter : str
        Filter identifier used for PSF calibration lookup.
    psf_line : str
        Calibration line name used for the PSF/EEF lookup.
    psf_energy_keV : float
        Representative calibration energy in keV used for the local PSF.
    ml_radius_pix : float
        Radius of the single-source fit stamp in pixels.
    extent_ratio : float
        Ratio of measured source size to local PSF size.
    fitted_extent_sigma_pix : float
        Best-fit extended-model core scale in pixels.
    meas_r50_pix : float
        Measured residual-profile r50 in pixels.
    meas_r80_pix : float
        Measured residual-profile r80 in pixels.
    meas_r90_pix : float
        Measured residual-profile r90 in pixels.
    catalog_shape : str
        Final catalog-region shape string, currently ``circle``.
    catalog_radius_pix : float
        Final catalog-region radius in pixels.
    catalog_radius_arcsec : float
        Final catalog-region radius in arcsec.
    major_arcsec : float
        Wavelet-ellipse semi-major axis in arcsec.
    minor_arcsec : float
        Wavelet-ellipse semi-minor axis in arcsec.
    radius_arcsec : float
        Geometric-mean wavelet radius in arcsec.
    emin_keV : float | None
        Lower energy bound of the analyzed image in keV.
    emax_keV : float | None
        Upper energy bound of the analyzed image in keV.
    extras : dict[str, Any]
        Additional auxiliary fields not covered by the main schema.

    Notes
    -----
    The fields are intentionally split into two conceptual groups:

    - public science catalog fields, which are written to the standard FITS output;
    - debug and review fields, which are only emitted when ``debug_columns=True``.

    Internal attribute names remain lowercase throughout the codebase. The public
    FITS writer in :mod:`fxtsrcdet.io` is responsible for mapping these internal
    names to the external science-catalog column names such as ``X_IMA``,
    ``ML_CTS_0``, and ``DET_LIKE_0``.
    """

    id: int = 0

    #--- public science catalog columns
    id_src: int = 0
    id_band: int = 0
    id_cluster: int = 0
    source_type: str = "background"
    ra: float = float("nan")
    ra_lowerr: float = float("nan")
    ra_uperr: float = float("nan")
    dec: float = float("nan")
    dec_lowerr: float = float("nan")
    dec_uperr: float = float("nan")
    radec_err: float = float("nan")
    lii: float = float("nan")
    bii: float = float("nan")
    dist_nn: float = float("nan")
    x_ima: float = 0.0
    x_ima_err: float = 0.0
    x_ima_lowerr: float = 0.0
    x_ima_uperr: float = 0.0
    y_ima: float = 0.0
    y_ima_err: float = 0.0
    y_ima_lowerr: float = 0.0
    y_ima_uperr: float = 0.0
    ml_cts: float = 0.0
    ml_cts_err: float = 0.0
    ml_cts_lowerr: float = 0.0
    ml_cts_uperr: float = 0.0
    ml_rate: float = float("nan")
    ml_rate_err: float = float("nan")
    ml_rate_lowerr: float = float("nan")
    ml_rate_uperr: float = float("nan")
    ml_flux: float = 0.0
    ml_flux_err: float = 0.0
    ml_flux_lowerr: float = 0.0
    ml_flux_uperr: float = 0.0
    ext: float = 0.0
    ext_err: float = 0.0
    ext_lowerr: float = 0.0
    ext_uperr: float = 0.0
    det_like: float = float("nan")
    ext_like: float = float("nan")
    ml_bkg: float = float("nan")
    ml_exp: float = float("nan")
    ml_radius: float = 0.0
    maskfrac: float = 1.0
    ml_eff: float = 0.0

    #--- debug and review columns
    scale: float = 1.0
    support_scales: list[float] = field(default_factory=list)
    wavelet_peak_score: float = 0.0
    min_significance: float = 1.0
    npix: int = 0
    counts: float = 0.0
    net_counts: float = 0.0
    bkg_counts: float = 0.0
    major: float = 0.5
    minor: float = 0.5
    theta_deg: float = 0.0
    group_id: int = -1
    group_size: int = 1
    group_stamp_radius_pix: float = 0.0
    theta_arcmin: float = 0.0
    psf_r50_pix: float = 0.0
    psf_r75_pix: float = 0.0
    psf_r80_pix: float = 0.0
    psf_r90_pix: float = 0.0
    psf_instrument: str = ""
    psf_filter: str = ""
    psf_line: str = ""
    psf_energy_keV: float = float("nan")
    ml_radius_pix: float = 0.0
    extent_ratio: float = 1.0
    fitted_extent_sigma_pix: float = 0.0
    meas_r50_pix: float = 0.0
    meas_r80_pix: float = 0.0
    meas_r90_pix: float = 0.0
    catalog_shape: str = "circle"
    catalog_radius_pix: float = 0.5
    catalog_radius_arcsec: float = 0.0
    major_arcsec: float = 0.0
    minor_arcsec: float = 0.0
    radius_arcsec: float = 0.0
    emin_keV: float | None = None
    emax_keV: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, candidate: DetectionCandidate) -> "CatalogRow":
        """Create a CatalogRow from class: DetectionCandidate."""
        return cls(
            id=int(candidate.id),
            id_src=int(candidate.id),
            x_ima=float(candidate.x),
            y_ima=float(candidate.y),
            major=float(candidate.major),
            minor=float(candidate.minor),
            theta_deg=float(candidate.theta_deg),
            support_scales=list(candidate.support_scales),
            npix=int(candidate.npix),
            counts=float(candidate.counts),
            net_counts=float(candidate.net_counts),
            scale=float(candidate.scale),
            min_significance=float(candidate.min_significance),
            wavelet_peak_score=float(candidate.wavelet_peak_score),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row using internal lowercase field names."""
        return super().to_dict()


def beta_model_kernel(shape: tuple[int, int], rc_pix: float, beta: float = 2.0 / 3.0) -> np.ndarray:
    """Build a circular beta-model surface-brightness kernel.

    Parameters
    ----------
    shape : tuple[int, int]
        Output kernel shape.
    rc_pix : float
        Beta-model core radius in pixels.
    beta : float
        Beta-model slope parameter.

    Returns
    -------
    kernel : np.ndarray
        Normalized 2D beta-model kernel.
    """
    rc_pix = max(float(rc_pix), 0.25)
    yy, xx = np.indices(shape, dtype=np.float64)
    cy = 0.5 * (shape[0] - 1.0)
    cx = 0.5 * (shape[1] - 1.0)
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    exponent = -3.0 * float(beta) + 0.5
    kernel = np.power(1.0 + rr2 / (rc_pix * rc_pix), exponent)
    kernel = np.clip(kernel, 0.0, None)
    norm = float(np.sum(kernel))
    if norm <= 0.0:
        out = np.zeros(shape, dtype=np.float64)
        out[int(round(cy)), int(round(cx))] = 1.0
        return out
    return kernel / norm
