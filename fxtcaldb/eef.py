"""EEF calibration selection and interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.interpolate import Rbf

from fxtcaldb.env import CaldbPaths


EPS = 1e-12
EP_LINE_ENERGY_KEV = {
    "C_K": 0.277,
    "Ag_L": 2.98,
    "Al_K": 1.49,
    "Cu_L": 0.93,
    "Cu_K": 8.04,
    "Ti_K": 4.51,
}


@dataclass(frozen=True)
class EEFSelection:
    """Selected EEF calibration files for one mission/image configuration."""

    selected_eef: Path | None
    mean_eef_files: tuple[Path, ...]
    meta: dict[str, str | float]


def _read_fits_table_xy(path: Path, xcol: str, ycol: str, ext: int) -> tuple[np.ndarray, np.ndarray]:
    """Read two numeric columns from one FITS binary-table extension."""
    data = fits.getdata(path, ext=ext)
    return np.asarray(data[xcol], dtype=np.float64), np.asarray(data[ycol], dtype=np.float64)


def _eef_theta_extensions(path: Path) -> list[tuple[float, int]]:
    """Discover the off-axis EEF extensions available in one FITS file."""
    out: list[tuple[float, int]] = []
    with fits.open(path) as hdul:
        for idx, hdu in enumerate(hdul[1:], start=1):
            name = str(hdu.name)
            lower_name = name.lower()
            if "arcmin" not in lower_name:
                continue
            out.append((float(lower_name.replace("arcmin", "")), idx))
    return out


def _interpolate_curve(path: Path, theta_arcmin: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one EEF FITS file to the requested off-axis angle."""
    with fits.open(path) as hdul:
        angles = []
        curves = []
        for hdu in hdul[1:]:
            extname = str(hdu.header["EXTNAME"])
            lower_name = extname.lower()
            if "arcmin" not in lower_name:
                continue
            angles.append(float(lower_name.split("arcmin")[0]))
            curves.append(np.asarray(hdu.data["EEF"], dtype=np.float64))
        if not angles:
            raise RuntimeError(f"No off-axis EEF extensions found in {path}")
        angles_array = np.asarray(angles, dtype=np.float64)
        curves_array = np.asarray(curves, dtype=np.float64)
        if theta_arcmin < angles_array.min() or theta_arcmin > angles_array.max():
            theta_arcmin = float(min(angles_array, key=lambda value: abs(value - theta_arcmin)))
        interp_funcs = [Rbf(angles_array, curves_array[:, idx], function="multiquadric") for idx in range(curves_array.shape[1])]
        interpolated_curve = np.asarray([func(theta_arcmin) for func in interp_funcs], dtype=np.float64)
        radius_pixel = np.asarray(hdul[1].data["radius_pixel"], dtype=np.float64)
    return radius_pixel, interpolated_curve


def select_ep_eef_files(
    instrument: str | None,
    filter_name: str | None,
    emin_keV: float | None,
    emax_keV: float | None,
) -> EEFSelection:
    """Select the EP/FXT EEF file set that best matches one image setup."""
    caldb = Path(CaldbPaths.resolve().root)
    eef_dir = caldb / "data" / "ep" / "fxt" / "cpf" / "eef"
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
        return EEFSelection(
            selected_eef=Path(chosen["path"]),
            mean_eef_files=(),
            meta={
                "instrument": str(chosen["instrument"]),
                "filter": str(chosen["filter"]),
                "line": str(chosen["line"]),
                "energy_keV": float(chosen["energy"]),
            },
        )

    return EEFSelection(
        selected_eef=None,
        mean_eef_files=tuple(Path(entry["path"]) for entry in subset),
        meta={
            "instrument": instrument or "mean",
            "filter": filter_name or "mean",
            "line": "mean",
            "energy_keV": float(np.mean([entry["energy"] for entry in subset])),
        },
    )


def load_eef_curve_for_theta(eef_file: Path | None, theta_arcmin: float, mean_files: tuple[Path, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Load the local EEF curve for one off-axis angle."""
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
        return radius_common, np.maximum.accumulate(frac)

    if eef_file is None:
        curves = [interpolate_one_file(path, theta_arcmin) for path in mean_files]
        if not curves:
            raise RuntimeError("No EP/FXT EEF curves available for mean PSF selection.")
        ref_radius = max(curves, key=lambda item: item[0][-1])[0]
        frac_stack = [
            np.interp(ref_radius, radius, frac, left=0.0, right=float(frac[-1]))
            for radius, frac in curves
        ]
        frac_mean = np.clip(np.mean(np.vstack(frac_stack), axis=0), 0.0, 1.0)
        return ref_radius, np.maximum.accumulate(frac_mean)
    return _interpolate_curve(Path(eef_file), float(theta_arcmin))


def available_theta_arcmin(eef_file: Path | None, mean_files: tuple[Path, ...] = ()) -> np.ndarray:
    """Return the off-axis calibration grid available for one EEF selection."""
    theta_values: set[float] = set()
    paths = (eef_file,) if eef_file is not None else mean_files
    for path in paths:
        for theta_arcmin, _ext in _eef_theta_extensions(Path(path)):
            theta_values.add(float(theta_arcmin))
    if not theta_values:
        raise RuntimeError("No off-axis EEF extensions available for the selected PSF context.")
    return np.asarray(sorted(theta_values), dtype=np.float64)
