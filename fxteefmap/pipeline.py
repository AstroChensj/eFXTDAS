#!/usr/bin/env python3
"""Generate EP/FXT EEF-radius map products for one image footprint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from fxtpsf_helpers import (
    available_theta_arcmin,
    build_mission_psf_context,
    eef_radius,
    infer_optical_axis,
    load_local_eef,
)
from fxteefmap.config import DEFAULT_PIXEL_SCALE_ARCSEC
from fxteefmap.utils.logger import build_cli_logger, emit


def _read_ccd(path: Path) -> CCDData:
    """Read a FITS image into a CCDData object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        try:
            ccd = CCDData.read(path)
        except ValueError:
            ccd = CCDData.read(path, unit=u.dimensionless_unscaled)
    if ccd.data.ndim != 2:
        raise ValueError(f"{path} is not a 2D FITS image.")
    return ccd


def _load_img(path: Path) -> np.ndarray:
    """Load a 2D FITS image as a floating-point array."""
    return np.asarray(_read_ccd(path).data, dtype=np.float64)


def _load_header(path: Path) -> Any | None:
    """Load the primary FITS header from a file."""
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def _save_img(path: Path, array: np.ndarray, header: Any | None = None) -> None:
    """Write a 2D array to a FITS image while preserving metadata."""
    out_header = header.copy() if header is not None else fits.Header()
    ccd = CCDData(np.asarray(array, dtype=np.float32), unit=u.dimensionless_unscaled, meta=out_header)
    ccd.write(path, overwrite=True)


def _load_wcs(path: Path) -> Any | None:
    """Load celestial WCS metadata from a FITS image."""
    wcs = _read_ccd(path).wcs
    if wcs is None or not getattr(wcs, "has_celestial", False):
        return None
    return wcs.celestial


def _infer_pixel_scale_arcsec(wcs: Any | None, fallback: float) -> float:
    """Infer the image pixel scale from WCS metadata."""
    if wcs is None:
        return float(fallback)
    pixel_scales_deg = proj_plane_pixel_scales(wcs)
    if len(pixel_scales_deg) < 2:
        return float(fallback)
    pixel_scale_arcsec = float(np.sqrt(abs(pixel_scales_deg[0] * pixel_scales_deg[1])) * 3600.0)
    return pixel_scale_arcsec if pixel_scale_arcsec > 0 else float(fallback)


def build_eef_radius_map(
    image: np.ndarray,
    pixel_scale_arcsec: float,
    eeffrac: float,
    mission: str,
    instrument: str | None,
    filter_name: str | None,
    emin_keV: float | None,
    emax_keV: float | None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    exposure_map: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an EEF-radius map for one image footprint.

    Parameters
    ----------
    image : np.ndarray
        Input image array used only for shape.
    pixel_scale_arcsec : float
        Image pixel scale in arcsec/pixel.
    eeffrac : float
        Requested encircled-energy fraction in ``[0, 1]``.
    mission : str
        Mission PSF model identifier.
    instrument : str | None
        Mission instrument or detector arm.
    filter_name : str | None
        Mission filter state.
    emin_keV : float | None
        Lower image energy bound in keV.
    emax_keV : float | None
        Upper image energy bound in keV.
    optaxis_x : float | None
        Optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optical-axis y coordinate in 1-based image pixels.
    exposure_map : np.ndarray | None
        Optional exposure map used to zero invalid pixels.

    Returns
    -------
    result : tuple[np.ndarray, dict[str, Any]]
        ``(radius_map, metadata)`` where the radius map is in image pixels.
    """
    context = build_mission_psf_context(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
    )
    opt_x, opt_y = infer_optical_axis(image.shape, optaxis_x, optaxis_y)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    theta_map = np.hypot((xx + 1.0) - opt_x, (yy + 1.0) - opt_y) * float(pixel_scale_arcsec) / 60.0

    theta_grid = available_theta_arcmin(context)
    radius_grid = np.empty_like(theta_grid, dtype=np.float64)
    for idx, theta_arcmin in enumerate(theta_grid):
        radius_pix, frac = load_local_eef(context, float(theta_arcmin))
        radius_grid[idx] = float(eef_radius(radius_pix, frac, eeffrac))
    radius_map = np.interp(theta_map, theta_grid, radius_grid, left=float(radius_grid[0]), right=float(radius_grid[-1])).astype(np.float32)

    if exposure_map is not None:
        radius_map = np.where(np.asarray(exposure_map, dtype=np.float64) > 0.0, radius_map, 0.0).astype(np.float32)

    metadata = {
        "mission": context.mission,
        "instrument": context.meta.get("instrument", instrument or ""),
        "filter": context.meta.get("filter", filter_name or ""),
        "line": context.meta.get("line", ""),
        "energy_keV": context.meta.get("energy_keV", np.nan),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "optaxis_x": float(opt_x),
        "optaxis_y": float(opt_y),
        "eeffrac": float(eeffrac),
    }
    return radius_map, metadata


def build_eef_radius_maps(
    image: np.ndarray,
    pixel_scale_arcsec: float,
    eeffrac_values: list[float] | tuple[float, ...],
    mission: str,
    instrument: str | None,
    filter_name: str | None,
    emin_keV: float | None,
    emax_keV: float | None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    exposure_map: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build multiple EEF-radius maps for one image footprint.

    Parameters
    ----------
    image : np.ndarray
        Input image array used only for shape.
    pixel_scale_arcsec : float
        Image pixel scale in arcsec/pixel.
    eeffrac_values : list[float] | tuple[float, ...]
        Requested encircled-energy fractions in ``[0, 1]``.
    mission : str
        Mission PSF model identifier.
    instrument : str | None
        Mission instrument or detector arm.
    filter_name : str | None
        Mission filter state.
    emin_keV : float | None
        Lower image energy bound in keV.
    emax_keV : float | None
        Upper image energy bound in keV.
    optaxis_x : float | None
        Optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optical-axis y coordinate in 1-based image pixels.
    exposure_map : np.ndarray | None
        Optional exposure map used to zero invalid pixels.

    Returns
    -------
    result : tuple[dict[str, np.ndarray], dict[str, Any]]
        ``(maps, metadata)`` where ``maps`` is keyed by extension name such as
        ``R50`` or ``R90`` and each map is in image pixels.
    """
    fractions = [float(np.clip(value, 0.0, 1.0)) for value in eeffrac_values]
    if not fractions:
        raise ValueError("At least one encircled-energy fraction is required.")

    context = build_mission_psf_context(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
    )
    opt_x, opt_y = infer_optical_axis(image.shape, optaxis_x, optaxis_y)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    theta_map = np.hypot((xx + 1.0) - opt_x, (yy + 1.0) - opt_y) * float(pixel_scale_arcsec) / 60.0

    theta_grid = available_theta_arcmin(context)
    radius_by_frac: dict[float, np.ndarray] = {}
    for frac_value in fractions:
        radius_grid = np.empty_like(theta_grid, dtype=np.float64)
        for idx, theta_arcmin in enumerate(theta_grid):
            radius_pix, frac = load_local_eef(context, float(theta_arcmin))
            radius_grid[idx] = float(eef_radius(radius_pix, frac, frac_value))
        radius_map = np.interp(
            theta_map,
            theta_grid,
            radius_grid,
            left=float(radius_grid[0]),
            right=float(radius_grid[-1]),
        ).astype(np.float32)
        if exposure_map is not None:
            radius_map = np.where(np.asarray(exposure_map, dtype=np.float64) > 0.0, radius_map, 0.0).astype(np.float32)
        radius_by_frac[frac_value] = radius_map

    maps = {f"R{int(round(100.0 * frac_value)):02d}": radius_by_frac[frac_value] for frac_value in fractions}
    metadata = {
        "mission": context.mission,
        "instrument": context.meta.get("instrument", instrument or ""),
        "filter": context.meta.get("filter", filter_name or ""),
        "line": context.meta.get("line", ""),
        "energy_keV": context.meta.get("energy_keV", np.nan),
        "pixel_scale_arcsec": float(pixel_scale_arcsec),
        "optaxis_x": float(opt_x),
        "optaxis_y": float(opt_y),
        "eeffrac_values": tuple(fractions),
    }
    return maps, metadata


def _save_multi_extension_maps(path: Path, maps: dict[str, np.ndarray], header: Any | None, meta: dict[str, Any]) -> None:
    """Write multiple EEF-radius maps into one FITS file."""
    primary_header = header.copy() if header is not None else fits.Header()
    primary_header["MISSION"] = (str(meta["mission"]), "Mission PSF model")
    primary_header["INSTRUME"] = (str(meta["instrument"]), "Instrument or detector arm")
    primary_header["FILTER"] = (str(meta["filter"]), "Filter state")
    primary_header["PSFLINE"] = (str(meta["line"]), "Selected PSF calibration line")
    primary_header["PSFENERG"] = (float(meta["energy_keV"]), "Selected PSF calibration energy [keV]")
    primary_header["PIXSCALE"] = (float(meta["pixel_scale_arcsec"]), "Image pixel scale [arcsec/pixel]")
    primary_header["OPTAXISX"] = (float(meta["optaxis_x"]), "Optical-axis X [1-based image pixel]")
    primary_header["OPTAXISY"] = (float(meta["optaxis_y"]), "Optical-axis Y [1-based image pixel]")
    primary_header["BUNIT"] = ("pixel", "EEF radius unit")
    hdus: list[fits.ImageHDU | fits.PrimaryHDU] = [fits.PrimaryHDU(header=primary_header)]
    for name, array in maps.items():
        ext_header = header.copy() if header is not None else fits.Header()
        hdu = fits.ImageHDU(np.asarray(array, dtype=np.float32), header=ext_header, name=name)
        try:
            frac_value = float(name[1:]) / 100.0
        except ValueError:
            frac_value = np.nan
        hdu.header["EEF_FRAC"] = (frac_value, "Requested encircled-energy fraction")
        hdu.header["BUNIT"] = ("pixel", "EEF radius unit")
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Generate an EP/FXT EEF-radius map for one image footprint.")
    p.add_argument("image", type=Path, help="Input FITS image used for shape and WCS.")
    p.add_argument("--out", type=Path, required=True, help="Output FITS EEF-radius map path.")
    p.add_argument("--expmap", type=Path, default=None, help="Optional exposure map; pixels with exp<=0 are set to 0.")
    p.add_argument("--mission", default="ep-fxt", help="Mission PSF model identifier.")
    p.add_argument("--instrument", default=None, help="Mission instrument or detector arm, e.g. fxta or fxtb.")
    p.add_argument("--filter", dest="filter_name", default=None, help="Mission filter state, e.g. open, medium, thin, hole.")
    p.add_argument("--emin", type=float, default=None, help="Lower image energy bound in keV.")
    p.add_argument("--emax", type=float, default=None, help="Upper image energy bound in keV.")
    p.add_argument("--eeffrac", type=float, default=None, help="Requested encircled-energy fraction in [0,1]; when omitted, write multi-extension output with the standard fractions.")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.50, 0.75, 0.80, 0.90], help="Encircled-energy fractions for multi-extension output.")
    p.add_argument("--optaxis-x", type=float, default=None, help="Optical-axis X position in 1-based image pixels; defaults to image center.")
    p.add_argument("--optaxis-y", type=float, default=None, help="Optical-axis Y position in 1-based image pixels; defaults to image center.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level for CLI and output log file.")
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file path; defaults to <out>.log.")
    return p


def main() -> None:
    """Run the command-line interface."""
    args = _parser().parse_args()
    log_file = args.log_file if args.log_file is not None else args.out.with_suffix(".log")
    logger = build_cli_logger("eFXTDAS.fxteefmap", args.log_level, log_file)
    emit(logger, "info", f"Loading image: {args.image}")
    image = _load_img(args.image)
    exposure = _load_img(args.expmap) if args.expmap is not None else None
    header = _load_header(args.image)
    wcs = _load_wcs(args.image)
    pixel_scale_arcsec = _infer_pixel_scale_arcsec(wcs, DEFAULT_PIXEL_SCALE_ARCSEC)
    emit(logger, "info", f"Using pixel scale = {pixel_scale_arcsec:.4f} arcsec/pixel")
    if args.eeffrac is not None:
        emit(logger, "info", f"Building single EEF-radius map for fraction {float(np.clip(args.eeffrac, 0.0, 1.0)):.3f}")
        radius_map, meta = build_eef_radius_map(
            image=image,
            pixel_scale_arcsec=pixel_scale_arcsec,
            eeffrac=float(np.clip(args.eeffrac, 0.0, 1.0)),
            mission=args.mission,
            instrument=args.instrument,
            filter_name=args.filter_name,
            emin_keV=args.emin,
            emax_keV=args.emax,
            optaxis_x=args.optaxis_x,
            optaxis_y=args.optaxis_y,
            exposure_map=exposure,
        )

        out_header = header.copy() if header is not None else fits.Header()
        out_header["BUNIT"] = ("pixel", "EEF radius unit")
        out_header["EEF_FRAC"] = (float(meta["eeffrac"]), "Requested encircled-energy fraction")
        out_header["MISSION"] = (str(meta["mission"]), "Mission PSF model")
        out_header["INSTRUME"] = (str(meta["instrument"]), "Instrument or detector arm")
        out_header["FILTER"] = (str(meta["filter"]), "Filter state")
        out_header["PSFLINE"] = (str(meta["line"]), "Selected PSF calibration line")
        out_header["PSFENERG"] = (float(meta["energy_keV"]), "Selected PSF calibration energy [keV]")
        out_header["PIXSCALE"] = (float(meta["pixel_scale_arcsec"]), "Image pixel scale [arcsec/pixel]")
        out_header["OPTAXISX"] = (float(meta["optaxis_x"]), "Optical-axis X [1-based image pixel]")
        out_header["OPTAXISY"] = (float(meta["optaxis_y"]), "Optical-axis Y [1-based image pixel]")
        _save_img(args.out, radius_map, header=out_header)
        emit(logger, "info", f"Wrote EEF-radius map: {args.out}")
        return

    emit(logger, "info", f"Building multi-extension EEF-radius map for fractions={args.fractions}")
    maps, meta = build_eef_radius_maps(
        image=image,
        pixel_scale_arcsec=pixel_scale_arcsec,
        eeffrac_values=args.fractions,
        mission=args.mission,
        instrument=args.instrument,
        filter_name=args.filter_name,
        emin_keV=args.emin,
        emax_keV=args.emax,
        optaxis_x=args.optaxis_x,
        optaxis_y=args.optaxis_y,
        exposure_map=exposure,
    )
    _save_multi_extension_maps(args.out, maps, header, meta)
    emit(logger, "info", f"Wrote EEF-radius map bundle: {args.out}")


if __name__ == "__main__":
    main()
