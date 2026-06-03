"""Helpers for PSF-aware EEF metadata and detection-mask construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from fxtpsf_helpers import available_theta_arcmin, build_mission_psf_context
from fxtpsf_helpers.geometry import infer_optical_axis


FILTER_NAME_MAP = {
    "00": "open",
    "01": "thin",
    "02": "medium",
    "03": "hole",
    "80": "open",
    "81": "thin",
    "82": "medium",
    "83": "hole",
}


def module_to_instrument(module: str) -> str:
    """Convert one FXT module code into the instrument string.

    Parameters
    ----------
    module : str
        Short module code such as ``"a"`` or ``"b"``.

    Returns
    -------
    str
        Instrument string expected by PSF/EEF helpers.
    """
    lowered = str(module).strip().lower()
    if lowered not in {"a", "b"}:
        raise ValueError(f"Unsupported FXT module code: {module!r}")
    return f"fxt{lowered}"


def filter_code_to_name(filter_code: str) -> str:
    """Convert one FXT filter code into the CALDB filter name.

    Parameters
    ----------
    filter_code : str
        Filename/header filter code such as ``"01"``.

    Returns
    -------
    str
        Filter name such as ``"thin"`` or ``"open"``.
    """
    code = str(filter_code).strip().lower()
    if code in FILTER_NAME_MAP:
        return FILTER_NAME_MAP[code]
    if code in {"open", "thin", "medium", "hole"}:
        return code
    raise ValueError(f"Unsupported FXT filter code for EEF selection: {filter_code!r}")


def infer_optaxis_from_image(image_path: str | Path) -> tuple[float, float]:
    """Infer the optical-axis pixel from one image WCS and pointing center.

    Parameters
    ----------
    image_path : str | Path
        FITS image path with celestial WCS and pointing keywords.

    Returns
    -------
    tuple[float, float]
        Optical-axis coordinates in 1-based image pixels.
    """
    with fits.open(image_path) as hdul:
        header = hdul[0].header.copy()
        data = np.asarray(hdul[0].data)
    shape = data.shape
    if "RA_PNT" not in header or "DEC_PNT" not in header:
        return infer_optical_axis(shape, None, None)
    try:
        wcs = WCS(header).celestial
        if wcs is None or not getattr(wcs, "has_celestial", False):
            return infer_optical_axis(shape, None, None)
        x_pix, y_pix = wcs.world_to_pixel_values(float(header["RA_PNT"]), float(header["DEC_PNT"]))
        return float(x_pix + 1.0), float(y_pix + 1.0)
    except Exception:
        return infer_optical_axis(shape, None, None)


def get_theta_max_caldb(
    *,
    mission: str,
    instrument: str,
    filter_name: str,
    emin_keV: float | None,
    emax_keV: float | None,
) -> float:
    """Return the largest calibrated off-axis angle for one PSF context.

    Parameters
    ----------
    mission : str
        Mission identifier.
    instrument : str
        Instrument string such as ``"fxta"``.
    filter_name : str
        Filter name such as ``"thin"``.
    emin_keV : float | None
        Lower energy bound in keV.
    emax_keV : float | None
        Upper energy bound in keV.

    Returns
    -------
    float
        Largest calibrated off-axis angle in arcminutes.
    """
    context = build_mission_psf_context(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
    )
    return float(np.max(available_theta_arcmin(context)))


def build_detection_mask(
    image_path: str | Path,
    expmap_path: str | Path,
    eefmap_path: str | Path,
    out_path: str | Path,
) -> dict[str, float | str | int]:
    """Build one per-OBSID detection-valid mask.

    Parameters
    ----------
    image_path : str | Path
        Detection-band image path.
    expmap_path : str | Path
        Matching exposure-map path.
    eefmap_path : str | Path
        Matching EEF-radius bundle path.
    out_path : str | Path
        Output FITS path for the boolean mask.

    Returns
    -------
    dict[str, float | str | int]
        Summary metadata for logging and downstream bookkeeping.
    """
    with fits.open(image_path) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    with fits.open(expmap_path) as hdul:
        exposure = np.asarray(hdul[0].data, dtype=np.float64)
    with fits.open(eefmap_path) as hdul:
        r90 = np.asarray(hdul["R90"].data, dtype=np.float64)
        eef_header = hdul[0].header.copy()
    mask = (
        np.isfinite(image)
        & np.isfinite(exposure)
        & (exposure > 0.0)
        & np.isfinite(r90)
        & (r90 > 0.0)
    )
    header["MASKTYPE"] = ("DETECT", "Mask valid for stacked source detection")
    header["THMAXCAL"] = (float(eef_header.get("THMAXCAL", 0.0)), "Largest calibrated off-axis angle [arcmin]")
    fits.writeto(out_path, mask.astype(np.uint8), header, overwrite=True)
    return {
        "path": str(out_path),
        "valid_pixels": int(np.count_nonzero(mask)),
        "theta_max_caldb": float(eef_header.get("THMAXCAL", 0.0)),
    }
