"""ARF generation and response factorization for ``fxtrspgen``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import interpolate

from fxtrspgen.caldb import (
    SpectrumMetadata,
    read_base_arf_table,
    resolve_base_arf,
    resolve_teldef,
    resolve_vignetting_table,
)
from fxtrspgen.regions import RegionSet, load_region_set
from fxtrspgen.runtime import ensure_fxtdas_py_path

ensure_fxtdas_py_path()

import attitude  # type: ignore  # noqa: E402
from eef_read import cal_eefcurve  # type: ignore  # noqa: E402


PIXEL_ARCSEC = 9.6687
BETA_EEF_THETA_MAX_ARCMIN = 3.0
BETA_CACHE: dict[str, dict[str, np.ndarray]] = {}


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
    """Evaluate the legacy scalar vignetting model."""
    return np.power(1.0 + (theta_arcmin / coef0) ** 2, -coef1)


def _interp_correction(energy_points: np.ndarray, values: np.ndarray, energy_bins: np.ndarray) -> np.ndarray:
    """Interpolate a per-energy correction onto the ARF energy grid."""
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


def _detector_prefix(detnam: str) -> str:
    """Convert ``FXTA``/``FXTB`` to ``fxta``/``fxtb``."""
    return {"A": "fxta", "B": "fxtb"}[detnam[-1].upper()]


def _filter_name(filt: str) -> str:
    """Translate the numeric FILTER code to the CALDB string suffix."""
    return {0: "open", 1: "thin", 2: "medium", 3: "hole"}[int(str(filt)[-1])]


def _read_teldef_alignment(filepath: str) -> tuple[np.ndarray, float, float, float]:
    """Read the TELDEF quantities needed for the optical-axis world position."""
    header = fits.getheader(filepath)
    mij = np.array(
        [
            [header["ALIGNM11"], header["ALIGNM12"], header["ALIGNM13"]],
            [header["ALIGNM21"], header["ALIGNM22"], header["ALIGNM23"]],
            [header["ALIGNM31"], header["ALIGNM32"], header["ALIGNM33"]],
        ],
        dtype=np.float64,
    )
    return (
        mij,
        float(header["FOCALLEN"]),
        float(header["DET_XSCL"]),
        float(header["OPTAXISX"]),
        float(header["OPTAXISY"]),
    )


def compute_optical_axis_pixel(specfile: str, exp_wcs: WCS, metadata: SpectrumMetadata) -> tuple[float, float]:
    """Project the telescope optical axis onto the exposure-map pixel grid."""
    teldef_path, _ = resolve_teldef(metadata)
    mij, focallen, pixel_size, optaxis_x, optaxis_y = _read_teldef_alignment(teldef_path)
    with fits.open(specfile) as hdul:
        header = hdul[1].header
        ra_pnt = float(header["RA_PNT"])
        dec_pnt = float(header["DEC_PNT"])
        pa_pnt = float(header["PA_PNT"])
    det_center = [optaxis_x - 1.0, optaxis_y - 1.0]
    quat = attitude.eq_to_quat(ra_pnt, dec_pnt, pa_pnt)
    axis_ra, axis_dec = attitude.det2radecpix(det_center, det_center, pixel_size, focallen, quat, mij)
    xpix, ypix = exp_wcs.all_world2pix(float(np.asarray(axis_ra).ravel()[0]), float(np.asarray(axis_dec).ravel()[0]), 0)
    return float(xpix), float(ypix)


def _load_beta_table(det_prefix: str) -> dict[str, np.ndarray]:
    """Load the near-axis beta PSF parametrization from CALDB."""
    if det_prefix in BETA_CACHE:
        return BETA_CACHE[det_prefix]
    caldb = Path(os.environ["CALDB"])
    for directory in ("data/ep/fxt/cpf/psf", "data/ep/fxt/cpf/eef"):
        for suffix in (".fits", ".fits.gz"):
            path = caldb / directory / f"{det_prefix}_beta{suffix}"
            if path.is_file():
                with fits.open(path) as hdul:
                    data = hdul[1].data
                    bandwidth = np.asarray(data["EMAX"] - data["EMIN"], dtype=np.float64)
                    use = bandwidth < 4.0 * np.median(bandwidth)
                    e_mid = 0.5 * np.asarray(data["EMIN"] + data["EMAX"], dtype=np.float64)[use]
                    order = np.argsort(e_mid)
                    table = {
                        "e_mid": e_mid[order],
                        "A1": np.asarray(data["A1"], dtype=np.float64)[use][order],
                        "R1": np.asarray(data["R1"], dtype=np.float64)[use][order],
                        "ALP1": np.asarray(data["ALP1"], dtype=np.float64)[use][order],
                        "A2": np.asarray(data["A2"], dtype=np.float64)[use][order],
                        "R2": np.asarray(data["R2"], dtype=np.float64)[use][order],
                        "ALP2": np.asarray(data["ALP2"], dtype=np.float64)[use][order],
                    }
                BETA_CACHE[det_prefix] = table
                return table
    raise FileNotFoundError(f"No beta PSF file found for {det_prefix}")


def _dual_beta_cdf(
    radius_arcsec: np.ndarray,
    a1: np.ndarray,
    r1: np.ndarray,
    alp1: np.ndarray,
    a2: np.ndarray,
    r2: np.ndarray,
    alp2: np.ndarray,
) -> np.ndarray:
    """Evaluate the enclosed-flux CDF of the dual-beta PSF."""
    w1 = a1 * (r1 ** 2) / (alp1 - 1.0)
    w2 = a2 * (r2 ** 2) / (alp2 - 1.0)
    u1 = 1.0 + (radius_arcsec / (2.0 * r1)) ** 2
    u2 = 1.0 + (radius_arcsec / (2.0 * r2)) ** 2
    c1 = 1.0 - np.power(u1, 1.0 - alp1)
    c2 = 1.0 - np.power(u2, 1.0 - alp2)
    return (w1 * c1 + w2 * c2) / (w1 + w2)


def _beta_psf_fraction(
    det_prefix: str,
    energy_bins: np.ndarray,
    source_xy: tuple[float, float],
    region_mask: np.ndarray,
    subpixels: int = 5,
) -> np.ndarray:
    """Integrate the near-axis beta PSF over an arbitrary aperture mask."""
    table = _load_beta_table(det_prefix)
    ny, nx = region_mask.shape
    y_index, x_index = np.indices((ny, nx), dtype=np.float64)
    offsets = (np.arange(subpixels, dtype=np.float64) + 0.5) / subpixels - 0.5
    dx, dy = np.meshgrid(offsets, offsets)
    x_eval = x_index[:, :, None, None] + dx[None, None, :, :]
    y_eval = y_index[:, :, None, None] + dy[None, None, :, :]
    weight = np.broadcast_to(
        region_mask[:, :, None, None] / float(subpixels * subpixels),
        x_eval.shape,
    )
    radius_sq_arcsec = (
        (x_eval - source_xy[0]) ** 2 + (y_eval - source_xy[1]) ** 2
    ) * (PIXEL_ARCSEC ** 2)
    active = weight > 0.0
    radius_sq_arcsec = radius_sq_arcsec[active]
    weight = weight[active]
    pixel_area = PIXEL_ARCSEC ** 2
    eef_samples = np.zeros_like(table["e_mid"], dtype=np.float64)
    for idx in range(len(table["e_mid"])):
        a1 = table["A1"][idx]
        r1 = table["R1"][idx]
        alp1 = table["ALP1"][idx]
        a2 = table["A2"][idx]
        r2 = table["R2"][idx]
        alp2 = table["ALP2"][idx]
        psf_val = (
            a1 * np.power(1.0 + radius_sq_arcsec / (4.0 * r1 * r1), -alp1)
            + a2 * np.power(1.0 + radius_sq_arcsec / (4.0 * r2 * r2), -alp2)
        )
        total_flux = (
            4.0 * np.pi * a1 * r1 ** 2 / (alp1 - 1.0)
            + 4.0 * np.pi * a2 * r2 ** 2 / (alp2 - 1.0)
        )
        eef_samples[idx] = np.sum(psf_val * weight) * pixel_area / total_flux
    return np.clip(_interp_correction(table["e_mid"], eef_samples, energy_bins), 0.0, 1.0)


def _region_area_fractions(
    radius_edges: np.ndarray,
    region_mask: np.ndarray,
    source_xy: tuple[float, float],
    subpixels: int = 5,
) -> np.ndarray:
    """Compute aperture coverage as a function of radius around the source."""
    ny, nx = region_mask.shape
    y_index, x_index = np.indices((ny, nx), dtype=np.float64)
    offsets = (np.arange(subpixels, dtype=np.float64) + 0.5) / subpixels - 0.5
    dx, dy = np.meshgrid(offsets, offsets)
    x_eval = x_index[:, :, None, None] + dx[None, None, :, :]
    y_eval = y_index[:, :, None, None] + dy[None, None, :, :]
    weights = region_mask[:, :, None, None] / float(subpixels * subpixels)
    radius = np.sqrt((x_eval - source_xy[0]) ** 2 + (y_eval - source_xy[1]) ** 2)
    bin_index = np.digitize(radius.ravel(), radius_edges) - 1
    valid = (bin_index >= 0) & (bin_index < len(radius_edges) - 1)
    full_weights = np.full(radius.size, 1.0 / float(subpixels * subpixels), dtype=np.float64)
    numerator = np.bincount(
        bin_index[valid],
        weights=weights.ravel()[valid],
        minlength=len(radius_edges) - 1,
    )
    denominator = np.bincount(
        bin_index[valid],
        weights=full_weights[valid],
        minlength=len(radius_edges) - 1,
    )
    fraction = np.zeros(len(radius_edges) - 1, dtype=np.float64)
    good = denominator > 0
    fraction[good] = numerator[good] / denominator[good]
    return np.clip(fraction, 0.0, 1.0)


def _legacy_eef_fraction(
    det_prefix: str,
    filt: str,
    theta_arcmin: float,
    energy_bins: np.ndarray,
    source_xy: tuple[float, float],
    region_mask: np.ndarray,
) -> np.ndarray:
    """Integrate the legacy off-axis EEF curves over an arbitrary aperture."""
    caldb = Path(os.environ["CALDB"])
    eef_path = caldb / "data/ep/fxt/cpf/eef"
    energy_points = np.array([0.277, 0.93, 1.49, 3.0, 4.51, 8.048], dtype=np.float64)
    energy_tags = np.array(["C_K", "Cu_L", "Al_K", "Ag_L", "Ti_K", "Cu_K"])
    eef_samples: list[float] = []
    sample_energies: list[float] = []
    for energy_point, tag in zip(energy_points, energy_tags):
        filename = eef_path / f"{det_prefix}_{_filter_name(filt)}_{tag}_eef.fits"
        if not filename.is_file():
            continue
        radius_list, eef_curve = cal_eefcurve(theta_arcmin, str(filename))
        radius_list = np.asarray(radius_list, dtype=np.float64)
        eef_curve = np.asarray(eef_curve, dtype=np.float64)
        fractions = _region_area_fractions(radius_list, region_mask, source_xy)
        eef_annulus = np.diff(eef_curve)
        captured = float(np.sum(fractions[: len(eef_annulus)] * eef_annulus[: len(fractions)]))
        sample_energies.append(float(energy_point))
        eef_samples.append(max(0.0, min(1.0, captured)))
    if not sample_energies:
        raise FileNotFoundError("No EEF calibration files were found for off-axis PSF correction.")
    return np.clip(
        _interp_correction(np.asarray(sample_energies), np.asarray(eef_samples), energy_bins),
        0.0,
        1.0,
    )


def compute_psf_correction(
    metadata: SpectrumMetadata,
    energy_bins: np.ndarray,
    source_xy: tuple[float, float],
    region_mask: np.ndarray,
    theta_arcmin: float,
) -> np.ndarray:
    """Compute the encircled PSF fraction captured by the region."""
    det_prefix = _detector_prefix(metadata.detnam)
    if theta_arcmin < BETA_EEF_THETA_MAX_ARCMIN:
        return _beta_psf_fraction(det_prefix, energy_bins, source_xy, region_mask)
    return _legacy_eef_fraction(det_prefix, metadata.filt, theta_arcmin, energy_bins, source_xy, region_mask)


def resolve_source_position(
    exp_wcs: WCS,
    region_set: RegionSet,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
) -> SourcePosition:
    """Resolve the PSF anchor position in exposure-map pixel coordinates."""
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


def _write_factorized_arf(
    outfile: str,
    energ_lo: np.ndarray,
    energ_hi: np.ndarray,
    specresp: np.ndarray,
    base_arf: np.ndarray,
    vign_corr: np.ndarray,
    psf_corr: np.ndarray,
    regcov_corr: np.ndarray,
    clobber: bool,
) -> None:
    """Write an OGIP ARF with additive factorization columns."""
    if Path(outfile).exists() and not clobber:
        raise FileExistsError(f"Output ARF already exists: {outfile}")
    total_corr = vign_corr * psf_corr * regcov_corr
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
    hdul = fits.HDUList([primary, spectrum])
    hdul.writeto(outfile, overwrite=clobber)


def generate_arf(
    specfile: str,
    expfile: str,
    regionfile: str,
    outfile: str,
    metadata: SpectrumMetadata,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    clobber: bool = False,
) -> ArfProducts:
    """Generate a factorized ARF for one spectrum and DS9 region file."""
    energ_lo, energ_hi, base_arf = read_base_arf_table(metadata)
    energy_bins = 0.5 * (energ_lo + energ_hi)
    with fits.open(expfile) as hdul:
        exp_data = np.asarray(hdul[0].data, dtype=np.float64)
        exp_wcs = WCS(hdul[0].header)
    image_shape = exp_data.shape
    region_set = load_region_set(regionfile, image_shape=image_shape, wcs=exp_wcs, oversample=5)
    source = resolve_source_position(exp_wcs, region_set, srcx=srcx, srcy=srcy, ra=ra, dec=dec)
    opt_x, opt_y = compute_optical_axis_pixel(specfile, exp_wcs, metadata)
    pixel_scale_deg = float(np.mean(proj_plane_pixel_scales(exp_wcs.celestial)))
    yy, xx = np.indices(image_shape, dtype=np.float64)
    theta_map_arcmin = np.hypot(xx - opt_x, yy - opt_y) * pixel_scale_deg * 60.0
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
    weighted_region = region_mask * exposure_fraction
    weight_sum = float(np.sum(weighted_region))
    vign_table = resolve_vignetting_table(metadata)
    vign_energy = np.asarray(vign_table["ENERGY"], dtype=np.float64)
    vign_samples = []
    for coef0, coef1 in zip(vign_table["COEF0"], vign_table["COEF1"]):
        vign_map = np.clip(vignfunc(theta_map_arcmin, float(coef0), float(coef1)), 0.0, 1.0)
        if weight_sum > 0.0:
            vign_samples.append(float(np.sum(weighted_region * vign_map) / weight_sum))
        else:
            vign_samples.append(0.0)
    vign_corr = np.clip(_interp_correction(vign_energy, np.asarray(vign_samples), energy_bins), 0.0, 1.0)
    theta_source_arcmin = float(np.hypot(source.x - opt_x, source.y - opt_y) * pixel_scale_deg * 60.0)
    psf_corr = compute_psf_correction(
        metadata,
        energy_bins,
        (source.x, source.y),
        region_mask,
        theta_source_arcmin,
    )
    regcov_corr = np.full_like(psf_corr, regcov_value, dtype=np.float64)
    total_corr = vign_corr * psf_corr * regcov_corr
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
        clobber=clobber,
    )
    return ArfProducts(arf_out=outfile, region_set=region_set, source_position=source)
