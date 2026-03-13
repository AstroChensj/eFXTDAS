"""Mission-specific PSF and EEF utilities.

This module keeps calibration-specific logic out of the detector core so the
main source-detection code stays mission-agnostic. The public API is generic:
build a mission PSF context once, then query the local EEF/PSF for a source.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


EPS = 1e-12
DEFAULT_EP_PIXEL_SCALE_ARCSEC = 9.6
EP_LINE_ENERGY_KEV = {
    "C_K": 0.277,
    "Ag_L": 2.98,
    "Al_K": 1.49,
    "Cu_L": 0.93,
    "Cu_K": 8.04,
    "Ti_K": 4.51,
}


@dataclass(frozen=True)
class MissionPSFContext:
    """Mission-specific PSF selection state for one image."""

    mission: str
    instrument: str | None
    filter_name: str | None
    energy_keV: float | None
    selected_eef: Path | None
    mean_eef_files: tuple[Path, ...]
    meta: dict[str, str | float]
    default_pixel_scale_arcsec: float


def _read_fits_table_xy(path: Path, xcol: str, ycol: str, ext: int) -> tuple[np.ndarray, np.ndarray]:
    """Read two numeric columns from a FITS binary table extension.

    Parameters
    ----------
    path : Path
        FITS file path.
    xcol : str
        Name of the first column.
    ycol : str
        Name of the second column.
    ext : int
        FITS extension index.

    Returns
    -------
    columns : tuple[np.ndarray, np.ndarray]
        ``(x_values, y_values)`` as ``float64`` arrays.
    """
    data = fits.getdata(path, ext=ext)
    return np.asarray(data[xcol], dtype=np.float64), np.asarray(data[ycol], dtype=np.float64)


def _eef_theta_extensions(path: Path) -> list[tuple[float, int]]:
    """Discover the off-axis EEF extensions available in one FITS file.

    Parameters
    ----------
    path : Path
        EEF FITS file path.

    Returns
    -------
    extensions : list[tuple[float, int]]
        A list of ``(theta_arcmin, ext_index)`` pairs.
    """
    out: list[tuple[float, int]] = []
    with fits.open(path) as hdul:
        for idx, hdu in enumerate(hdul[1:], start=1):
            name = str(hdu.name)
            if "arcmin" not in name:
                continue
            out.append((float(name.replace("arcmin", "")), idx))
    return out


def eef_radius(radius_values: np.ndarray, frac: np.ndarray, target: float) -> float:
    """Interpolate the radius at a target encircled-energy fraction.

    Parameters
    ----------
    radius_values : np.ndarray
        Radius grid in pixels.
    frac : np.ndarray
        Encircled-energy fractions on the same grid.
    target : float
        Requested encircled-energy fraction.

    Returns
    -------
    radius : float
        The interpolated radius in pixels.
    """
    return float(np.interp(float(np.clip(target, 0.0, 1.0)), frac, radius_values))


def build_psf_kernel(radius_pix: np.ndarray, frac: np.ndarray, size: int | None = None) -> np.ndarray:
    """Convert a radial EEF curve into a symmetric 2D PSF kernel.

    Parameters
    ----------
    radius_pix : np.ndarray
        Radius grid in pixels.
    frac : np.ndarray
        Encircled-energy fractions on the same grid.
    size : int | None
        Optional output kernel size.

    Returns
    -------
    kernel : np.ndarray
        A normalized 2D PSF kernel.
    """
    if size is None:
        r90_pix = eef_radius(radius_pix, frac, 0.90)
        radius = max(4, int(math.ceil(3.0 * r90_pix)))
    else:
        radius = max(1, int(size // 2))

    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr_pix = np.sqrt(x * x + y * y)

    rr_mid = np.arange(0.0, rr_pix.max() + 1.0, 1.0)
    eef_mid = np.interp(rr_mid, radius_pix, frac, left=0.0, right=float(frac[-1]))
    deriv = np.gradient(eef_mid, rr_mid, edge_order=1)
    density = deriv / np.maximum(2.0 * math.pi * np.maximum(rr_mid, 0.5), EPS)
    density[0] = density[1] if len(density) > 1 else density[0]
    radial_pdf = np.interp(rr_pix, rr_mid, density, left=density[0], right=0.0)
    radial_pdf = np.clip(radial_pdf, 0.0, None)
    radial_pdf /= np.sum(radial_pdf)
    return radial_pdf


def _select_ep_eef_file(
    instrument: str | None,
    filter_name: str | None,
    emin_keV: float | None,
    emax_keV: float | None,
) -> tuple[Path | None, dict[str, str | float], tuple[Path, ...]]:
    """Select the EP/FXT EEF FITS file that best matches one image setup.

    Parameters
    ----------
    instrument : str | None
        Instrument or detector arm name.
    filter_name : str | None
        Filter state name.
    emin_keV : float | None
        Lower energy bound in keV.
    emax_keV : float | None
        Upper energy bound in keV.

    Returns
    -------
    selection : tuple[Path | None, dict[str, str | float], tuple[Path, ...]]
        ``(selected_file, metadata, mean_files)`` for local EEF queries.
    """
    caldb = os.environ["CALDB"]
    eef_dir = Path(caldb) / "data" / "ep" / "fxt" / "cpf" / "eef"

    entries: list[dict[str, Any]] = []
    for path in sorted(eef_dir.glob("fxt*_eef.fits")):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        line = "_".join(parts[2:-1])
        energy = EP_LINE_ENERGY_KEV.get(line)
        if energy is None:
            continue
        entries.append(
            {
                "path": path,
                "instrument": parts[0].lower(),
                "filter": parts[1].lower(),
                "line": line,
                "energy": float(energy),
            }
        )

    if not entries:
        raise RuntimeError(f"No EP/FXT EEF FITS files found under {eef_dir}")

    target_energy = None
    if emin_keV is not None and emax_keV is not None:
        target_energy = 0.5 * (float(emin_keV) + float(emax_keV))
    elif emin_keV is not None:
        target_energy = float(emin_keV)
    elif emax_keV is not None:
        target_energy = float(emax_keV)

    subset = entries
    if instrument:
        subset = [entry for entry in subset if entry["instrument"] == instrument.lower()]
    if filter_name:
        subset = [entry for entry in subset if entry["filter"] == filter_name.lower()]
    if not subset:
        subset = entries

    if target_energy is not None:
        chosen = min(subset, key=lambda entry: abs(entry["energy"] - target_energy))
        return (
            Path(chosen["path"]),
            {
                "instrument": str(chosen["instrument"]),
                "filter": str(chosen["filter"]),
                "line": str(chosen["line"]),
                "energy_keV": float(chosen["energy"]),
            },
            (),
        )

    return (
        None,
        {
            "instrument": instrument or "mean",
            "filter": filter_name or "mean",
            "line": "mean",
            "energy_keV": float(np.mean([entry["energy"] for entry in subset])),
        },
        tuple(Path(entry["path"]) for entry in subset),
    )


def _load_ep_eef_curve_for_theta(
    eef_file: Path | None,
    theta_arcmin: float,
    mean_files: tuple[Path, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the nearest EP/FXT EEF curve for one off-axis angle.

    Parameters
    ----------
    eef_file : Path | None
        Selected EEF FITS file, or ``None`` for mean-mode selection.
    theta_arcmin : float
        Off-axis angle in arcminutes.
    mean_files : tuple[Path, ...]
        File set used when averaging multiple EEF curves.

    Returns
    -------
    eef_curve : tuple[np.ndarray, np.ndarray]
        ``(radius_pix, eef_fraction)`` for the nearest off-axis calibration.
    """
    def interpolate_one_file(path: Path, target_theta: float) -> tuple[np.ndarray, np.ndarray]:
        theta_grid = _eef_theta_extensions(path)
        if not theta_grid:
            raise RuntimeError(f"No off-axis EEF extensions found in {path}")

        theta_vals = np.asarray([item[0] for item in theta_grid], dtype=np.float64)
        curves = [_read_fits_table_xy(path, "radius_pixel", "EEF", ext) for _theta, ext in theta_grid]

        if len(curves) == 1:
            return curves[0]

        idx_hi = int(np.searchsorted(theta_vals, float(target_theta), side="left"))
        if idx_hi <= 0:
            idx_lo = 0
            idx_hi = 1
        elif idx_hi >= len(theta_vals):
            idx_hi = len(theta_vals) - 1
            idx_lo = idx_hi - 1
        else:
            idx_lo = idx_hi - 1

        theta_lo = float(theta_vals[idx_lo])
        theta_hi = float(theta_vals[idx_hi])
        radius_lo, eef_lo = curves[idx_lo]
        radius_hi, eef_hi = curves[idx_hi]
        radius_common = radius_lo if radius_lo[-1] >= radius_hi[-1] else radius_hi
        eef_lo_common = np.interp(radius_common, radius_lo, eef_lo, left=0.0, right=float(eef_lo[-1]))
        eef_hi_common = np.interp(radius_common, radius_hi, eef_hi, left=0.0, right=float(eef_hi[-1]))

        if abs(theta_hi - theta_lo) < EPS:
            frac = eef_lo_common
        else:
            weight_hi = (float(target_theta) - theta_lo) / (theta_hi - theta_lo)
            weight_lo = 1.0 - weight_hi
            frac = weight_lo * eef_lo_common + weight_hi * eef_hi_common
        frac = np.clip(frac, 0.0, 1.0)
        frac = np.maximum.accumulate(frac)
        return radius_common, frac

    if eef_file is None:
        curves = [interpolate_one_file(path, theta_arcmin) for path in mean_files]
        if not curves:
            raise RuntimeError("No EP/FXT EEF curves available for mean PSF selection.")
        ref_radius = max(curves, key=lambda item: item[0][-1])[0]
        frac_stack = [
            np.interp(ref_radius, radius, eef, left=0.0, right=float(eef[-1]))
            for radius, eef in curves
        ]
        frac_mean = np.clip(np.mean(np.vstack(frac_stack), axis=0), 0.0, 1.0)
        frac_mean = np.maximum.accumulate(frac_mean)
        return ref_radius, frac_mean

    return interpolate_one_file(eef_file, theta_arcmin)


def build_mission_psf_context(
    mission: str,
    instrument: str | None,
    filter_name: str | None,
    emin_keV: float | None,
    emax_keV: float | None,
    psf_eef_path: Path | None = None,
) -> MissionPSFContext:
    """Build a mission-specific PSF context for one image configuration.

    Parameters
    ----------
    mission : str
        Mission identifier, such as ``ep-fxt``.
    instrument : str | None
        Mission instrument or detector arm.
    filter_name : str | None
        Mission filter state.
    emin_keV : float | None
        Lower energy bound in keV.
    emax_keV : float | None
        Upper energy bound in keV.
    psf_eef_path : Path | None
        Optional direct EEF file override.

    Returns
    -------
    context : MissionPSFContext
        A ``MissionPSFContext`` used for subsequent local PSF queries.
    """
    mission = mission.lower()
    if mission != "ep-fxt":
        raise ValueError(f"Unsupported mission: {mission}")

    if psf_eef_path is not None:
        energy_keV = None
        if emin_keV is not None and emax_keV is not None:
            energy_keV = 0.5 * (float(emin_keV) + float(emax_keV))
        elif emin_keV is not None:
            energy_keV = float(emin_keV)
        elif emax_keV is not None:
            energy_keV = float(emax_keV)
        meta: dict[str, str | float] = {
            "instrument": instrument or "custom",
            "filter": filter_name or "custom",
            "line": psf_eef_path.name,
            "energy_keV": energy_keV if energy_keV is not None else math.nan,
        }
        return MissionPSFContext(
            mission=mission,
            instrument=instrument,
            filter_name=filter_name,
            energy_keV=energy_keV,
            selected_eef=psf_eef_path,
            mean_eef_files=(),
            meta=meta,
            default_pixel_scale_arcsec=DEFAULT_EP_PIXEL_SCALE_ARCSEC,
        )

    selected_eef, meta, mean_eef_files = _select_ep_eef_file(instrument, filter_name, emin_keV, emax_keV)
    return MissionPSFContext(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        energy_keV=float(meta["energy_keV"]) if isinstance(meta["energy_keV"], (int, float)) else None,
        selected_eef=selected_eef,
        mean_eef_files=mean_eef_files,
        meta=meta,
        default_pixel_scale_arcsec=DEFAULT_EP_PIXEL_SCALE_ARCSEC,
    )


def load_local_eef(context: MissionPSFContext, theta_arcmin: float) -> tuple[np.ndarray, np.ndarray]:
    """Load the local EEF curve for one source position.

    Parameters
    ----------
    context : MissionPSFContext
        Mission PSF context for the image.
    theta_arcmin : float
        Off-axis angle in arcminutes.

    Returns
    -------
    eef_curve : tuple[np.ndarray, np.ndarray]
        ``(radius_pix, eef_fraction)`` for the local PSF.
    """
    if context.mission != "ep-fxt":
        raise ValueError(f"Unsupported mission: {context.mission}")
    return _load_ep_eef_curve_for_theta(context.selected_eef, theta_arcmin, context.mean_eef_files)


def available_theta_arcmin(context: MissionPSFContext) -> np.ndarray:
    """Return the off-axis calibration grid available for one PSF context.

    Parameters
    ----------
    context : MissionPSFContext
        Mission PSF context for the image.

    Returns
    -------
    theta_grid : np.ndarray
        Sorted off-axis calibration grid in arcminutes.
    """
    if context.mission != "ep-fxt":
        raise ValueError(f"Unsupported mission: {context.mission}")

    theta_values: set[float] = set()
    paths: tuple[Path, ...]
    if context.selected_eef is not None:
        paths = (context.selected_eef,)
    else:
        paths = context.mean_eef_files

    for path in paths:
        for theta_arcmin, _ext in _eef_theta_extensions(path):
            theta_values.add(float(theta_arcmin))

    if not theta_values:
        raise RuntimeError("No off-axis EEF extensions available for the selected PSF context.")
    return np.asarray(sorted(theta_values), dtype=np.float64)


def representative_r90_pix(context: MissionPSFContext) -> float:
    """Return a representative PSF ``r90`` for first-pass background carving.

    Parameters
    ----------
    context : MissionPSFContext
        Mission PSF context for the image.

    Returns
    -------
    r90_pix : float
        A representative ``r90`` radius in pixels.
    """
    radius_pix, frac = load_local_eef(context, 0.0)
    return eef_radius(radius_pix, frac, 0.90)
