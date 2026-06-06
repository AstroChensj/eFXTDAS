"""ARF generation and response diagnostics for ``fxtrspgen``.

The total aperture correction is computed from the joint capture integral over
the region footprint. The diagnostic columns are retained for interpretability,
but in general:

``average(coverage * vign * psf) != average(coverage) * average(vign) * average(psf)``

so ``TOT_CORR`` is authoritative and the component columns are not required to
multiply back to the total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import interpolate

from fxtcaldb.optics import compute_optical_axis_pixel
from fxtcaldb.query import ObservationMetadata
from fxtcaldb.response import read_base_arf_table
from fxtcaldb.vignetting import resolve_vignetting_table
from fxtpsfgen.mapper import ObservationPSFMapper, build_observation_psf_mapper
from fxtrspgen.regions import RegionSet, load_region_set


@dataclass(frozen=True)
class SourcePosition:
    """Resolved source position on the exposure-map pixel grid."""

    x: float
    y: float
    origin: str


@dataclass(frozen=True)
class ArfProducts:
    """ARF products and provenance returned by ``generate_arf``."""

    arf_out: str
    region_set: RegionSet
    source_position: SourcePosition


def vignfunc(theta_arcmin: np.ndarray, coef0: float, coef1: float) -> np.ndarray:
    """Evaluate the legacy scalar vignetting model.

    Parameters
    ----------
    theta_arcmin : np.ndarray
        Off-axis angle grid in arcminutes.
    coef0 : float
        Vignetting scale coefficient.
    coef1 : float
        Vignetting slope coefficient.

    Returns
    -------
    np.ndarray
        Vignetting map on the same grid.
    """
    return np.power(1.0 + (theta_arcmin / coef0) ** 2, -coef1)


# ---------------------------------------------------------------------------
# Interpolation and calibration-name helpers
# ---------------------------------------------------------------------------

def _interp_correction(energy_points: np.ndarray, values: np.ndarray, energy_bins: np.ndarray) -> np.ndarray:
    """Interpolate a per-energy correction onto the ARF energy grid.

    Parameters
    ----------
    energy_points : np.ndarray
        Native calibration energies.
    values : np.ndarray
        Correction values at ``energy_points``.
    energy_bins : np.ndarray
        Output energy grid.

    Returns
    -------
    np.ndarray
        Interpolated correction values on ``energy_bins``.
    """
    energy_points = np.asarray(energy_points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if energy_points.size == 1:
        return np.full_like(energy_bins, values[0], dtype=np.float64)
    if energy_points.size > 2:
        interp = interpolate.PchipInterpolator(energy_points, values, extrapolate=False)
        result = interp(energy_bins)
    else:
        interp = interpolate.interp1d(
            energy_points,
            values,
            kind="linear",
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )
        result = interp(energy_bins)
    result = np.asarray(result, dtype=np.float64)
    low_mask = np.isnan(result) & (energy_bins <= energy_points[0])
    high_mask = np.isnan(result) & (energy_bins >= energy_points[-1])
    result[low_mask] = values[0]
    result[high_mask] = values[-1]
    other_nan = np.isnan(result)
    if np.any(other_nan):
        result[other_nan] = np.interp(energy_bins[other_nan], energy_points, values)
    return result


def _interp_scalar(energy_points: np.ndarray, values: np.ndarray, energy: float) -> float:
    """Interpolate one scalar calibration value to a target energy.

    Parameters
    ----------
    energy_points : np.ndarray
        Native calibration energies.
    values : np.ndarray
        Scalar values on the native energy grid.
    energy : float
        Target energy in keV.

    Returns
    -------
    float
        Interpolated scalar value.
    """
    result = _interp_correction(
        np.asarray(energy_points, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        np.asarray([float(energy)], dtype=np.float64),
    )
    return float(result[0])


def _mapper_instrument(detnam: str) -> str:
    """Convert one detector token into the observation-mapper instrument name.

    Parameters
    ----------
    detnam : str
        Detector token such as ``FXTA`` or ``A``.

    Returns
    -------
    str
        Instrument token such as ``fxta``.
    """
    suffix = str(detnam).strip().upper()[-1]
    if suffix not in {"A", "B"}:
        raise ValueError(f"Unsupported FXT detector token for PSF mapping: {detnam!r}")
    return f"fxt{suffix.lower()}"


def _mapper_filter_name(filt: str | None) -> str:
    """Convert one FXT filter code into the observation-mapper family name.

    Parameters
    ----------
    filt : str | None
        Numeric or string FXT filter value.

    Returns
    -------
    str
        Filter-family token such as ``open`` or ``thin``.
    """
    mapping = {
        "0": "open",
        "00": "open",
        "1": "thin",
        "01": "thin",
        "2": "medium",
        "02": "medium",
        "3": "hole",
        "03": "hole",
        "open": "open",
        "thin": "thin",
        "medium": "medium",
        "hole": "hole",
    }
    if filt is None:
        raise ValueError("Observation metadata is missing FILTER, so the PSF mapper cannot be built.")
    key = str(filt).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported FXT filter token for PSF mapping: {filt!r}")
    return mapping[key]


def _psf_capture_fraction(
    psf_mapper: ObservationPSFMapper,
    energy_bins: np.ndarray,
    source_xy: tuple[float, float],
    weight_map: np.ndarray,
) -> np.ndarray:
    """Evaluate the active PSF model over an arbitrary image-space weight map.

    Parameters
    ----------
    psf_mapper : ObservationPSFMapper
        Observation PSF mapper for local PSF queries.
    energy_bins : np.ndarray
        Output energy grid in keV.
    source_xy : tuple[float, float]
        Source position in zero-based image pixels.
    weight_map : np.ndarray
        Pixel-space capture weights to integrate over.
    Returns
    -------
    np.ndarray
        PSF capture fraction on the requested energy grid.
    """
    return np.asarray(
        [
            psf_mapper.capture_fraction(
                source_xy[0] + 1.0,
                source_xy[1] + 1.0,
                weight_map,
                energy_keV=float(energy),
            )
            for energy in np.asarray(energy_bins, dtype=np.float64)
        ],
        dtype=np.float64,
    )


def _compute_psf_correction(
    psf_mapper: ObservationPSFMapper,
    energy_bins: np.ndarray,
    source_xy: tuple[float, float],
    region_mask: np.ndarray,
) -> np.ndarray:
    """Compute the diagnostic full-coverage PSF capture fraction.

    Parameters
    ----------
    psf_mapper : ObservationPSFMapper
        Observation PSF mapper for local PSF queries.
    energy_bins : np.ndarray
        Output energy grid in keV.
    source_xy : tuple[float, float]
        Source position in zero-based image pixels.
    region_mask : np.ndarray
        Rasterized region mask.
    Returns
    -------
    np.ndarray
        Diagnostic PSF capture fraction under full coverage and no vignetting.
    """
    return _psf_capture_fraction(psf_mapper, energy_bins, source_xy, region_mask)


def _compute_total_correction(
    psf_mapper: ObservationPSFMapper,
    source_xy: tuple[float, float],
    region_mask: np.ndarray,
    exposure_fraction: np.ndarray,
    theta_map_arcmin: np.ndarray,
    vign_table: np.recarray,
    energy_bins: np.ndarray,
) -> np.ndarray:
    """Compute the exact joint correction over coverage, vignetting, and PSF.

    Parameters
    ----------
    psf_mapper : ObservationPSFMapper
        Observation PSF mapper for local PSF queries.
    source_xy : tuple[float, float]
        Source position in zero-based image pixels.
    region_mask : np.ndarray
        Rasterized region mask.
    exposure_fraction : np.ndarray
        Relative exposure/coverage fraction on the image grid.
    theta_map_arcmin : np.ndarray
        Off-axis angle map in arcminutes.
    vign_table : np.recarray
        Vignetting calibration table.
    energy_bins : np.ndarray
        Output ARF energy grid in keV.

    Returns
    -------
    np.ndarray
        Exact total correction on the ARF energy grid.
    """
    vign_energy = np.asarray(vign_table["ENERGY"], dtype=np.float64)
    coef0_values = np.asarray(vign_table["COEF0"], dtype=np.float64)
    coef1_values = np.asarray(vign_table["COEF1"], dtype=np.float64)
    samples = []
    for energy in np.asarray(energy_bins, dtype=np.float64):
        coef0 = _interp_scalar(vign_energy, coef0_values, float(energy))
        coef1 = _interp_scalar(vign_energy, coef1_values, float(energy))
        vign_map = np.clip(vignfunc(theta_map_arcmin, coef0, coef1), 0.0, 1.0)
        joint_weight = region_mask * exposure_fraction * vign_map
        samples.append(
            float(
                psf_mapper.capture_fraction(
                    source_xy[0] + 1.0,
                    source_xy[1] + 1.0,
                    joint_weight,
                    energy_keV=float(energy),
                )
            )
        )
    return np.clip(np.asarray(samples, dtype=np.float64), 0.0, 1.0)


def resolve_source_position(
    exp_wcs: WCS,
    region_set: RegionSet,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
) -> SourcePosition:
    """Resolve the PSF anchor position in exposure-map pixel coordinates.

    Parameters
    ----------
    exp_wcs : WCS
        Exposure-map WCS.
    region_set : RegionSet
        Parsed source region set.
    srcx, srcy : float | None, optional
        Explicit image-coordinate source override.
    ra, dec : float | None, optional
        Explicit sky-coordinate source override.

    Returns
    -------
    SourcePosition
        Resolved source center in zero-based image pixels.
    """
    if (srcx is None) ^ (srcy is None):
        raise ValueError("srcx and srcy must be provided together.")
    if (ra is None) ^ (dec is None):
        raise ValueError("ra and dec must be provided together.")
    if srcx is not None and ra is not None:
        raise ValueError("Choose either srcx/srcy or ra/dec, not both.")
    if srcx is not None:
        return SourcePosition(x=float(srcx) - 1.0, y=float(srcy) - 1.0, origin="srcx/srcy")
    if ra is not None:
        xpix, ypix = exp_wcs.all_world2pix(float(ra), float(dec), 0)
        return SourcePosition(x=float(xpix), y=float(ypix), origin="ra/dec")
    xpix, ypix = region_set.first_positive_center_xy
    return SourcePosition(x=float(xpix), y=float(ypix), origin="region-center")


# ---------------------------------------------------------------------------
# ARF writer and task entrypoint
# ---------------------------------------------------------------------------

def _write_factorized_arf(
    outfile: str,
    energ_lo: np.ndarray,
    energ_hi: np.ndarray,
    specresp: np.ndarray,
    base_arf: np.ndarray,
    vign_corr: np.ndarray,
    psf_corr: np.ndarray,
    regcov_corr: np.ndarray,
    total_corr: np.ndarray,
    clobber: bool,
) -> None:
    """Write an OGIP ARF with diagnostic correction columns.

    Notes
    -----
    ``TOT_CORR`` is the exact joint aperture correction used to build
    ``SPECRESP``. The component columns are descriptive diagnostics and are not
    required to multiply back to the total.
    """
    if Path(outfile).exists() and not clobber:
        raise FileExistsError(f"Output ARF already exists: {outfile}")
    columns = [
        fits.Column(name="ENERG_LO", array=energ_lo, format="E", unit="keV"),
        fits.Column(name="ENERG_HI", array=energ_hi, format="E", unit="keV"),
        fits.Column(name="SPECRESP", array=specresp, format="D", unit="cm2"),
        fits.Column(name="BASE_ARF", array=base_arf, format="D", unit="cm2"),
        fits.Column(name="VIGN_CORR", array=vign_corr, format="D"),
        fits.Column(name="PSF_CORR", array=psf_corr, format="D"),
        fits.Column(name="REGCOV_CORR", array=regcov_corr, format="D"),
        fits.Column(name="TOT_CORR", array=total_corr, format="D"),
    ]
    primary = fits.PrimaryHDU()
    spectrum = fits.BinTableHDU.from_columns(columns, name="SPECRESP")
    spectrum.header["TELESCOP"] = "EP"
    spectrum.header["INSTRUME"] = "FXT"
    spectrum.header["HDUCLASS"] = "OGIP"
    spectrum.header["HDUCLAS1"] = "RESPONSE"
    spectrum.header["HDUCLAS2"] = "SPECRESP"
    spectrum.header["HDUVERS"] = "1.5.0"
    spectrum.header["FXTRSPBS"] = "BASE_ARF"
    spectrum.header["FXTRSPVG"] = "VIGN_CORR"
    spectrum.header["FXTRSPPS"] = "PSF_CORR"
    spectrum.header["FXTRSPRG"] = "REGCOV_CORR"
    spectrum.header["FXTRSPTT"] = "TOT_CORR"
    spectrum.header["FXTRSPNT"] = "Joint total is non-separable"
    hdul = fits.HDUList([primary, spectrum])
    hdul.writeto(outfile, overwrite=clobber)


def generate_arf(
    expfile: str,
    regionfile: str,
    outfile: str,
    metadata: ObservationMetadata,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    clobber: bool = False,
) -> ArfProducts:
    """Generate an ARF with exact total correction and diagnostic columns.

    Parameters
    ----------
    expfile : str
        Exposure-map FITS file.
    regionfile : str
        External DS9 source-region file.
    outfile : str
        Output ARF path.
    metadata : ObservationMetadata
        Observation metadata used for calibration lookup.
    srcx, srcy : float | None, optional
        Explicit image-coordinate source override.
    ra, dec : float | None, optional
        Explicit sky-coordinate source override.
    clobber : bool, optional
        Overwrite existing output files.

    Returns
    -------
    ArfProducts
        Written ARF path and provenance information.
    """
    energ_lo, energ_hi, base_arf = read_base_arf_table(metadata)
    energy_bins = 0.5 * (energ_lo + energ_hi)
    with fits.open(expfile) as hdul:
        exp_data = np.asarray(hdul[0].data, dtype=np.float64)
        exp_wcs = WCS(hdul[0].header)
    image_shape = exp_data.shape
    region_set = load_region_set(regionfile, image_shape=image_shape, wcs=exp_wcs, oversample=5)
    source = resolve_source_position(exp_wcs, region_set, srcx=srcx, srcy=srcy, ra=ra, dec=dec)
    psf_mapper = build_observation_psf_mapper(
        image_path=str(expfile),
        expmap_path=str(expfile),
        metadata=metadata,
        instrument=_mapper_instrument(metadata.detnam),
        filter_name=_mapper_filter_name(metadata.filt),
        emin_keV=float(np.min(energy_bins)),
        emax_keV=float(np.max(energy_bins)),
    )
    opt_x, opt_y = compute_optical_axis_pixel(metadata, exp_wcs.celestial)
    pixel_scale_deg = float(np.mean(proj_plane_pixel_scales(exp_wcs.celestial)))
    yy, xx = np.indices(image_shape, dtype=np.float64)
    theta_map_arcmin = np.hypot((xx + 1.0) - opt_x, (yy + 1.0) - opt_y) * pixel_scale_deg * 60.0
    exposure_positive = exp_data > 0.0
    if not np.any(exposure_positive):
        raise ValueError("The exposure map has no positive pixels.")
    max_exposure = float(np.nanmax(exp_data[exposure_positive]))
    exposure_fraction = np.clip(exp_data / max_exposure, 0.0, 1.0)
    region_mask = region_set.mask
    region_sum = float(np.sum(region_mask))
    if region_sum <= 0.0:
        raise ValueError("The DS9 region rasterizes to an empty source mask.")
    regcov_value = float(np.sum(region_mask * exposure_fraction) / region_sum)
    regcov_corr = np.full_like(energy_bins, regcov_value, dtype=np.float64)

    weighted_region = region_mask * exposure_fraction
    weight_sum = float(np.sum(weighted_region))
    vign_table = resolve_vignetting_table(metadata)
    vign_energy = np.asarray(vign_table["ENERGY"], dtype=np.float64)
    vign_samples = []
    for coef0, coef1 in zip(vign_table["COEF0"], vign_table["COEF1"]):
        vign_map = np.clip(vignfunc(theta_map_arcmin, float(coef0), float(coef1)), 0.0, 1.0)
        vign_samples.append(float(np.sum(weighted_region * vign_map) / weight_sum) if weight_sum > 0.0 else 0.0)
    vign_corr = np.clip(_interp_correction(vign_energy, np.asarray(vign_samples), energy_bins), 0.0, 1.0)

    psf_corr = _compute_psf_correction(
        psf_mapper,
        energy_bins,
        (source.x, source.y),
        region_mask,
    )
    total_corr = _compute_total_correction(
        psf_mapper,
        (source.x, source.y),
        region_mask,
        exposure_fraction,
        theta_map_arcmin,
        vign_table,
        energy_bins,
    )
    specresp = base_arf * total_corr
    _write_factorized_arf(
        outfile,
        energ_lo,
        energ_hi,
        specresp,
        base_arf,
        vign_corr,
        psf_corr,
        regcov_corr,
        total_corr,
        clobber=clobber,
    )
    return ArfProducts(arf_out=outfile, region_set=region_set, source_position=source)
