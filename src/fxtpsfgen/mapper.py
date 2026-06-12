"""PSF mapper products built from EP-FXT radial EEF and beta calibrations.

This module builds compact, machine-readable PSF products that sit above
``fxtcaldb``. The canonical radial truth model is a sampled scalar cube

``EEF(energy, theta, radius)``

augmented by the near-axis analytical beta model. Observation products keep
their native detector/image geometry. Stacked products embed one or more
observation products together with projected theta/weight maps on a common WCS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from fxtcaldb.env import CaldbPaths
from fxtcaldb.optics import compute_optical_axis_pixel
from fxtcaldb.psf import build_psf_kernel, load_beta_psf_table, load_eef_curve_for_theta
from fxtcaldb.query import ObservationMetadata, read_observation_metadata


BETA_EEF_THETA_MAX_ARCMIN = 3.0
EP_LINE_ENERGY_KEV = {
    "C_K": 0.277,
    "Ag_L": 2.98,
    "Al_K": 1.49,
    "Cu_L": 0.93,
    "Cu_K": 8.04,
    "Ti_K": 4.51,
}
FILTER_NAME_MAP = {
    "00": "open",
    "0": "open",
    "01": "thin",
    "1": "thin",
    "02": "medium",
    "2": "medium",
    "03": "hole",
    "3": "hole",
    "80": "open",
    "81": "thin",
    "82": "medium",
    "83": "hole",
    "open": "open",
    "thin": "thin",
    "medium": "medium",
    "hole": "hole",
}


def _instrument_from_detector(detnam: str) -> str:
    """Convert one FXT detector name into the instrument token.

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
        raise ValueError(f"Unsupported FXT detector name: {detnam!r}")
    return f"fxt{suffix.lower()}"


def _filter_name(filter_value: str | None) -> str:
    """Normalize one FXT filter value to the CALDB family name.

    Parameters
    ----------
    filter_value : str | None
        FITS/CALDB filter value.

    Returns
    -------
    str
        Filter-family token such as ``open`` or ``thin``.
    """
    if filter_value is None:
        raise ValueError("FXT filter value is required to build a PSF mapper.")
    key = str(filter_value).strip().lower()
    if key not in FILTER_NAME_MAP:
        raise ValueError(f"Unsupported FXT filter value: {filter_value!r}")
    return FILTER_NAME_MAP[key]


def _default_energy(emin_keV: float | None, emax_keV: float | None, energy_grid: np.ndarray) -> float:
    """Choose the representative energy used by projected convenience maps.

    Parameters
    ----------
    emin_keV : float | None
        Lower image energy bound in keV.
    emax_keV : float | None
        Upper image energy bound in keV.
    energy_grid : np.ndarray
        Native mapper energy grid in keV.

    Returns
    -------
    float
        Representative energy in keV.
    """
    if emin_keV is not None and emax_keV is not None:
        return 0.5 * (float(emin_keV) + float(emax_keV))
    if emin_keV is not None:
        return float(emin_keV)
    if emax_keV is not None:
        return float(emax_keV)
    return float(np.mean(np.asarray(energy_grid, dtype=np.float64)))


def _interp_curve(x_grid: np.ndarray, values: np.ndarray, x_value: float) -> np.ndarray:
    """Interpolate one stack of curves along the leading axis.

    Parameters
    ----------
    x_grid : np.ndarray
        Monotonic sample grid.
    values : np.ndarray
        Array shaped ``(len(x_grid), n)``.
    x_value : float
        Target coordinate.

    Returns
    -------
    np.ndarray
        Interpolated 1D curve.
    """
    grid = np.asarray(x_grid, dtype=np.float64)
    if grid.size == 1:
        return np.asarray(values[0], dtype=np.float64)
    x = float(np.clip(x_value, float(grid[0]), float(grid[-1])))
    idx_hi = int(np.searchsorted(grid, x, side="left"))
    if idx_hi <= 0:
        return np.asarray(values[0], dtype=np.float64)
    if idx_hi >= len(grid):
        return np.asarray(values[-1], dtype=np.float64)
    idx_lo = idx_hi - 1
    x_lo = float(grid[idx_lo])
    x_hi = float(grid[idx_hi])
    if abs(x_hi - x_lo) < 1e-12:
        return np.asarray(values[idx_lo], dtype=np.float64)
    frac_hi = (x - x_lo) / (x_hi - x_lo)
    frac_lo = 1.0 - frac_hi
    return frac_lo * np.asarray(values[idx_lo], dtype=np.float64) + frac_hi * np.asarray(values[idx_hi], dtype=np.float64)


def _beta_eef_curve(radius_grid: np.ndarray, beta_table: dict[str, np.ndarray], energy_keV: float) -> np.ndarray:
    """Evaluate the analytical dual-beta EEF curve on a radius grid.

    Parameters
    ----------
    radius_grid : np.ndarray
        Radius grid in image pixels.
    beta_table : dict[str, np.ndarray]
        Parsed beta-PSF coefficient table.
    energy_keV : float
        Requested energy in keV.

    Returns
    -------
    np.ndarray
        Monotonic EEF curve on ``radius_grid``.
    """
    radius_arcsec = np.asarray(radius_grid, dtype=np.float64) * 9.6687
    e_mid = np.asarray(beta_table["e_mid"], dtype=np.float64)
    curves = []
    for idx in range(len(e_mid)):
        a1 = float(beta_table["A1"][idx])
        r1 = float(beta_table["R1"][idx])
        alp1 = float(beta_table["ALP1"][idx])
        a2 = float(beta_table["A2"][idx])
        r2 = float(beta_table["R2"][idx])
        alp2 = float(beta_table["ALP2"][idx])
        w1 = a1 * (r1**2) / max(alp1 - 1.0, 1e-12)
        w2 = a2 * (r2**2) / max(alp2 - 1.0, 1e-12)
        u1 = 1.0 + (radius_arcsec / (2.0 * r1)) ** 2
        u2 = 1.0 + (radius_arcsec / (2.0 * r2)) ** 2
        c1 = 1.0 - np.power(u1, 1.0 - alp1)
        c2 = 1.0 - np.power(u2, 1.0 - alp2)
        curve = (w1 * c1 + w2 * c2) / max(w1 + w2, 1e-12)
        curves.append(np.maximum.accumulate(np.clip(curve, 0.0, 1.0)))
    stack = np.vstack(curves)
    curve = _interp_curve(e_mid, stack, float(energy_keV))
    return np.maximum.accumulate(np.clip(curve, 0.0, 1.0))


def _project_theta_map(header: fits.Header, metadata: ObservationMetadata) -> np.ndarray:
    """Project the optical-axis distance map on one image grid.

    Parameters
    ----------
    header : fits.Header
        FITS header carrying celestial WCS.
    metadata : ObservationMetadata
        Observation metadata with pointing information.

    Returns
    -------
    np.ndarray
        Off-axis angle map in arcminutes.
    """
    wcs = WCS(header).celestial
    if not getattr(wcs, "has_celestial", False):
        raise ValueError("Input image must contain celestial WCS to build a PSF mapper.")
    opt_x, opt_y = compute_optical_axis_pixel(metadata, wcs)
    pixel_scale_deg = float(np.mean(proj_plane_pixel_scales(wcs)))
    shape = tuple(int(value) for value in fits.PrimaryHDU(header=header).data.shape) if False else None
    ny = int(header["NAXIS2"])
    nx = int(header["NAXIS1"])
    yy, xx = np.indices((ny, nx), dtype=np.float64)
    return np.hypot((xx + 1.0) - opt_x, (yy + 1.0) - opt_y) * pixel_scale_deg * 60.0


def _collect_eef_library(instrument: str, filter_name: str) -> list[tuple[float, Path]]:
    """List EEF calibration files for one detector/filter family.

    Parameters
    ----------
    instrument : str
        Instrument token such as ``fxta``.
    filter_name : str
        Filter-family token such as ``open``.

    Returns
    -------
    list[tuple[float, Path]]
        Sorted ``(energy_keV, path)`` pairs.
    """
    eef_dir = Path(CaldbPaths.resolve().root) / "data" / "ep" / "fxt" / "cpf" / "eef"
    entries: list[tuple[float, Path]] = []
    pattern = f"{instrument}_{filter_name}_*_eef.fits"
    for path in sorted(eef_dir.glob(pattern)):
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        line = "_".join(parts[2:-1])
        energy = EP_LINE_ENERGY_KEV.get(line)
        if energy is None:
            continue
        entries.append((float(energy), path))
    if not entries:
        raise FileNotFoundError(f"No EEF calibration files found for {instrument}/{filter_name} under {eef_dir}")
    return sorted(entries, key=lambda item: item[0])


def _build_eef_cube(instrument: str, filter_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the sampled EEF cube for one detector/filter family.

    Parameters
    ----------
    instrument : str
        Instrument token such as ``fxta``.
    filter_name : str
        Filter-family token such as ``open``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(energy_grid, theta_grid, radius_grid, eef_cube)``.
    """
    library = _collect_eef_library(instrument, filter_name)
    theta_values: list[float] = []
    radius_values: list[np.ndarray] = []
    for _energy, path in library:
        with fits.open(path) as hdul:
            for hdu in hdul[1:]:
                lower_name = str(hdu.name).lower()
                if "arcmin" not in lower_name:
                    continue
                theta_values.append(float(lower_name.replace("arcmin", "")))
                radius_values.append(np.asarray(hdu.data["radius_pixel"], dtype=np.float64))
    theta_grid = np.asarray(sorted(set(theta_values)), dtype=np.float64)
    radius_grid = np.unique(np.concatenate(radius_values)).astype(np.float64)
    energy_grid = np.asarray([energy for energy, _path in library], dtype=np.float64)
    cube = np.empty((len(energy_grid), len(theta_grid), len(radius_grid)), dtype=np.float64)
    for energy_idx, (_energy, path) in enumerate(library):
        for theta_idx, theta_arcmin in enumerate(theta_grid):
            radius_pix, frac = load_eef_curve_for_theta(path, float(theta_arcmin))
            interp = np.interp(radius_grid, radius_pix, frac, left=0.0, right=float(frac[-1]))
            cube[energy_idx, theta_idx] = np.maximum.accumulate(np.clip(interp, 0.0, 1.0))
    return energy_grid, theta_grid, radius_grid, cube


def _radial_annulus_fractions(
    radius_edges: np.ndarray,
    weight_map: np.ndarray,
    source_xy: tuple[float, float],
    subpixels: int = 5,
) -> np.ndarray:
    """Compute average weights in radial annuli around one source position.

    Parameters
    ----------
    radius_edges : np.ndarray
        Radial edges in image pixels.
    weight_map : np.ndarray
        Pixel-space weight map.
    source_xy : tuple[float, float]
        Source position in zero-based image pixels.
    subpixels : int, optional
        Subpixel rasterization factor.

    Returns
    -------
    np.ndarray
        Average weight in each annulus, normalized to full geometric coverage.
    """
    weight_array = np.asarray(weight_map, dtype=np.float64)
    if not np.any(weight_array > 0.0):
        return np.zeros(len(radius_edges) - 1, dtype=np.float64)
    cropped_weight_map, cropped_source_xy = _crop_weight_map_centered(
        radius_edges=np.asarray(radius_edges, dtype=np.float64),
        weight_map=weight_array,
        source_xy=source_xy,
    )
    return _radial_annulus_fractions_full(
        radius_edges=np.asarray(radius_edges, dtype=np.float64),
        weight_map=cropped_weight_map,
        source_xy=cropped_source_xy,
        subpixels=subpixels,
    )


def _crop_weight_map_centered(
    radius_edges: np.ndarray,
    weight_map: np.ndarray,
    source_xy: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop a weight map around the source center using the occupied annulus.

    Parameters
    ----------
    radius_edges : np.ndarray
        Radial edges in image pixels. The crop extends to the upper edge of the
        farthest annulus that still contains nonzero weight support.
    weight_map : np.ndarray
        Pixel-space weight map.
    source_xy : tuple[float, float]
        Source position in zero-based image pixels.

    Returns
    -------
    tuple[np.ndarray, tuple[float, float]]
        Cropped weight map and crop-local source coordinates.
    """
    ny, nx = weight_map.shape
    source_x = float(source_xy[0])
    source_y = float(source_xy[1])
    positive = np.argwhere(weight_map > 0.0)
    if len(positive) == 0:
        return weight_map, (source_x, source_y)
    dy = positive[:, 0].astype(np.float64) - source_y
    dx = positive[:, 1].astype(np.float64) - source_x
    pixel_half_diagonal = math.sqrt(0.5)
    r_support = float(np.max(np.hypot(dx, dy) + pixel_half_diagonal)) if len(positive) else 0.0
    radius_edges = np.asarray(radius_edges, dtype=np.float64)
    if len(radius_edges) >= 2:
        upper_idx = int(np.searchsorted(radius_edges, r_support, side="right"))
        upper_idx = min(max(upper_idx, 1), len(radius_edges) - 1)
        r_cover = float(radius_edges[upper_idx])
    else:
        r_cover = r_support
    pad_pix = 1.0
    half_width = r_cover + pad_pix
    x_min = max(0, int(math.floor(source_x - half_width)))
    x_max = min(nx - 1, int(math.ceil(source_x + half_width)))
    y_min = max(0, int(math.floor(source_y - half_width)))
    y_max = min(ny - 1, int(math.ceil(source_y + half_width)))
    cropped = np.asarray(weight_map[y_min:y_max + 1, x_min:x_max + 1], dtype=np.float64)
    return cropped, (source_x - float(x_min), source_y - float(y_min))


def _radial_annulus_fractions_full(
    radius_edges: np.ndarray,
    weight_map: np.ndarray,
    source_xy: tuple[float, float],
    subpixels: int = 5,
) -> np.ndarray:
    """Compute average weights in radial annuli on the supplied working window.

    Parameters
    ----------
    radius_edges : np.ndarray
        Radial edges in image pixels.
    weight_map : np.ndarray
        Pixel-space weight map for the already-selected working window.
    source_xy : tuple[float, float]
        Source position in zero-based working-window pixels.
    subpixels : int, optional
        Subpixel rasterization factor.

    Returns
    -------
    np.ndarray
        Average weight in each annulus, normalized to full geometric coverage.
    """
    ny, nx = weight_map.shape
    y_index, x_index = np.indices((ny, nx), dtype=np.float64)
    offsets = (np.arange(subpixels, dtype=np.float64) + 0.5) / subpixels - 0.5
    dx, dy = np.meshgrid(offsets, offsets)
    x_eval = x_index[:, :, None, None] + dx[None, None, :, :]
    y_eval = y_index[:, :, None, None] + dy[None, None, :, :]
    weights = np.broadcast_to(
        np.asarray(weight_map, dtype=np.float64)[:, :, None, None] / float(subpixels * subpixels),
        (ny, nx, subpixels, subpixels),
    )
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


@dataclass(frozen=True)
class ObservationPSFMapper:
    """Per-observation PSF mapper built from radial EEF and beta calibrations."""

    mission: str
    detector: str
    filter_name: str
    energy_grid: np.ndarray
    theta_grid: np.ndarray
    radius_grid: np.ndarray
    eef_cube: np.ndarray
    beta_table: dict[str, np.ndarray]
    theta_map_arcmin: np.ndarray
    header: fits.Header
    default_energy_keV: float

    def eef_curve(self, theta_arcmin: float, energy_keV: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the local EEF curve at one off-axis angle and energy.

        Parameters
        ----------
        theta_arcmin : float
            Off-axis angle in arcminutes.
        energy_keV : float | None, optional
            Requested energy in keV. When omitted, the default mapper energy is
            used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(radius_pix, eef_fraction)`` for the requested location.
        """
        energy = self.default_energy_keV if energy_keV is None else float(energy_keV)
        if float(theta_arcmin) < BETA_EEF_THETA_MAX_ARCMIN:
            curve = _beta_eef_curve(self.radius_grid, self.beta_table, energy)
            return self.radius_grid, curve
        energy_slices = np.vstack([
            _interp_curve(self.theta_grid, self.eef_cube[idx], float(theta_arcmin))
            for idx in range(self.eef_cube.shape[0])
        ])
        curve = _interp_curve(self.energy_grid, energy_slices, energy)
        curve = np.maximum.accumulate(np.clip(curve, 0.0, 1.0))
        return self.radius_grid, curve

    def local_eef_curve(self, x_ima: float, y_ima: float, energy_keV: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the local EEF curve at one native image position.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(radius_pix, eef_fraction)`` for the local PSF.
        """
        x_idx = int(round(float(x_ima) - 1.0))
        y_idx = int(round(float(y_ima) - 1.0))
        if y_idx < 0 or y_idx >= self.theta_map_arcmin.shape[0] or x_idx < 0 or x_idx >= self.theta_map_arcmin.shape[1]:
            raise ValueError("Requested image position lies outside the mapper footprint.")
        return self.eef_curve(float(self.theta_map_arcmin[y_idx, x_idx]), energy_keV=energy_keV)

    def eef_at_radius(self, theta_arcmin: float, radius_pix: float, energy_keV: float | None = None) -> float:
        """Evaluate the EEF at one radius.

        Parameters
        ----------
        theta_arcmin : float
            Off-axis angle in arcminutes.
        radius_pix : float
            Radius in image pixels.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Encircled-energy fraction.
        """
        radius_grid, frac = self.eef_curve(theta_arcmin, energy_keV=energy_keV)
        return float(np.interp(float(radius_pix), radius_grid, frac, left=0.0, right=float(frac[-1])))

    def eef_at_position(self, x_ima: float, y_ima: float, radius_pix: float, energy_keV: float | None = None) -> float:
        """Evaluate the local EEF at one radius and native image position.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        radius_pix : float
            Radius in pixels.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Encircled-energy fraction.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return float(np.interp(float(radius_pix), radius_grid, frac, left=0.0, right=float(frac[-1])))

    def radius_at_eef(self, theta_arcmin: float, frac_value: float, energy_keV: float | None = None) -> float:
        """Return the radius at one encircled-energy fraction.

        Parameters
        ----------
        theta_arcmin : float
            Off-axis angle in arcminutes.
        frac_value : float
            Requested EEF fraction in ``[0, 1]``.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Radius in pixels.
        """
        radius_grid, frac = self.eef_curve(theta_arcmin, energy_keV=energy_keV)
        return float(np.interp(float(np.clip(frac_value, 0.0, 1.0)), frac, radius_grid))

    def radius_at_position(self, x_ima: float, y_ima: float, frac_value: float, energy_keV: float | None = None) -> float:
        """Return the local radius at one EEF fraction and image position.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        frac_value : float
            Requested EEF fraction in ``[0, 1]``.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Radius in pixels.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return float(np.interp(float(np.clip(frac_value, 0.0, 1.0)), frac, radius_grid))

    def kernel(self, theta_arcmin: float, energy_keV: float | None = None, size: int | None = None) -> np.ndarray:
        """Build a symmetric kernel from the local EEF curve.

        Parameters
        ----------
        theta_arcmin : float
            Off-axis angle in arcminutes.
        energy_keV : float | None, optional
            Requested energy in keV.
        size : int | None, optional
            Optional kernel size in pixels.

        Returns
        -------
        np.ndarray
            Normalized 2D kernel.
        """
        radius_grid, frac = self.eef_curve(theta_arcmin, energy_keV=energy_keV)
        return build_psf_kernel(radius_grid, frac, size=size)

    def kernel_at_position(self, x_ima: float, y_ima: float, energy_keV: float | None = None, size: int | None = None) -> np.ndarray:
        """Build the local kernel at one native image position.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        energy_keV : float | None, optional
            Requested energy in keV.
        size : int | None, optional
            Optional kernel size in pixels.

        Returns
        -------
        np.ndarray
            Normalized 2D kernel.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return build_psf_kernel(radius_grid, frac, size=size)

    def capture_fraction(
        self,
        x_ima: float,
        y_ima: float,
        weight_map: np.ndarray,
        energy_keV: float | None = None,
        subpixels: int = 5,
    ) -> float:
        """Integrate the local radial PSF over an arbitrary image-space weight map.

        Parameters
        ----------
        x_ima : float
            Source x coordinate in 1-based image pixels.
        y_ima : float
            Source y coordinate in 1-based image pixels.
        weight_map : np.ndarray
            Pixel-space capture weights, usually a region mask or joint capture map.
        energy_keV : float | None, optional
            Requested energy in keV.
        subpixels : int, optional
            Subpixel rasterization factor.

        Returns
        -------
        float
            Captured PSF fraction in ``[0, 1]``.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        annulus_fraction = _radial_annulus_fractions(
            radius_edges=np.asarray(radius_grid, dtype=np.float64),
            weight_map=np.asarray(weight_map, dtype=np.float64),
            source_xy=(float(x_ima) - 1.0, float(y_ima) - 1.0),
            subpixels=int(subpixels),
        )
        eef_annulus = np.diff(np.asarray(frac, dtype=np.float64))
        length = min(len(annulus_fraction), len(eef_annulus))
        if length <= 0:
            return 0.0
        return float(np.clip(np.sum(annulus_fraction[:length] * eef_annulus[:length]), 0.0, 1.0))

    def radius_map(self, frac_value: float, energy_keV: float | None = None) -> np.ndarray:
        """Project one radius-at-EEF map on the native observation grid.

        Parameters
        ----------
        frac_value : float
            Requested EEF fraction in ``[0, 1]``.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        np.ndarray
            Radius map in image pixels.
        """
        energy = self.default_energy_keV if energy_keV is None else float(energy_keV)
        radius_by_theta = np.asarray(
            [self.radius_at_eef(float(theta), float(frac_value), energy_keV=energy) for theta in self.theta_grid],
            dtype=np.float64,
        )
        return np.interp(
            self.theta_map_arcmin,
            self.theta_grid,
            radius_by_theta,
            left=float(radius_by_theta[0]),
            right=float(radius_by_theta[-1]),
        )

    def write(self, path: str | Path) -> str:
        """Write the mapper to a FITS product.

        Parameters
        ----------
        path : str | Path
            Output FITS path.

        Returns
        -------
        str
            Written file path.
        """
        header = self.header.copy()
        header["PSFTYPE"] = "OBS"
        header["MISSION"] = self.mission
        header["DETECTOR"] = self.detector
        header["FILTER"] = self.filter_name
        header["DEFENERG"] = float(self.default_energy_keV)
        header["THETCUT"] = float(BETA_EEF_THETA_MAX_ARCMIN)
        hdus: list[fits.HDUBase] = [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=np.asarray(self.theta_map_arcmin, dtype=np.float32), header=self.header.copy(), name="THETA_MAP"),
            fits.ImageHDU(data=np.asarray(self.energy_grid, dtype=np.float32), name="ENERGY_GRID"),
            fits.ImageHDU(data=np.asarray(self.theta_grid, dtype=np.float32), name="THETA_GRID"),
            fits.ImageHDU(data=np.asarray(self.radius_grid, dtype=np.float32), name="RADIUS_GRID"),
            fits.ImageHDU(data=np.asarray(self.eef_cube, dtype=np.float32), name="EEF_CUBE"),
        ]
        beta = self.beta_table
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="E_MID", format="E", array=np.asarray(beta["e_mid"], dtype=np.float32)),
                    fits.Column(name="A1", format="E", array=np.asarray(beta["A1"], dtype=np.float32)),
                    fits.Column(name="R1", format="E", array=np.asarray(beta["R1"], dtype=np.float32)),
                    fits.Column(name="ALP1", format="E", array=np.asarray(beta["ALP1"], dtype=np.float32)),
                    fits.Column(name="A2", format="E", array=np.asarray(beta["A2"], dtype=np.float32)),
                    fits.Column(name="R2", format="E", array=np.asarray(beta["R2"], dtype=np.float32)),
                    fits.Column(name="ALP2", format="E", array=np.asarray(beta["ALP2"], dtype=np.float32)),
                ],
                name="BETA",
            )
        )
        for frac_value, extname in ((0.50, "R50"), (0.75, "R75"), (0.80, "R80"), (0.90, "R90")):
            hdus.append(
                fits.ImageHDU(
                    data=np.asarray(self.radius_map(frac_value), dtype=np.float32),
                    header=self.header.copy(),
                    name=extname,
                )
            )
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return str(path)

    @classmethod
    def read(cls, path: str | Path) -> "ObservationPSFMapper":
        """Read one observation mapper product from FITS.

        Parameters
        ----------
        path : str | Path
            Mapper FITS path.

        Returns
        -------
        ObservationPSFMapper
            Loaded observation mapper.
        """
        with fits.open(path) as hdul:
            header = hdul[0].header.copy()
            beta_data = hdul["BETA"].data
            beta_table = {
                "e_mid": np.asarray(beta_data["E_MID"], dtype=np.float64),
                "A1": np.asarray(beta_data["A1"], dtype=np.float64),
                "R1": np.asarray(beta_data["R1"], dtype=np.float64),
                "ALP1": np.asarray(beta_data["ALP1"], dtype=np.float64),
                "A2": np.asarray(beta_data["A2"], dtype=np.float64),
                "R2": np.asarray(beta_data["R2"], dtype=np.float64),
                "ALP2": np.asarray(beta_data["ALP2"], dtype=np.float64),
            }
            return cls(
                mission=str(header.get("MISSION", "ep-fxt")),
                detector=str(header["DETECTOR"]),
                filter_name=str(header["FILTER"]),
                energy_grid=np.asarray(hdul["ENERGY_GRID"].data, dtype=np.float64),
                theta_grid=np.asarray(hdul["THETA_GRID"].data, dtype=np.float64),
                radius_grid=np.asarray(hdul["RADIUS_GRID"].data, dtype=np.float64),
                eef_cube=np.asarray(hdul["EEF_CUBE"].data, dtype=np.float64),
                beta_table=beta_table,
                theta_map_arcmin=np.asarray(hdul["THETA_MAP"].data, dtype=np.float64),
                header=hdul["THETA_MAP"].header.copy(),
                default_energy_keV=float(header["DEFENERG"]),
            )


@dataclass(frozen=True)
class StackedPSFMapper:
    """Stacked PSF mapper that evaluates weighted local EEF curves per source."""

    mission: str
    default_energy_keV: float
    radius_grid: np.ndarray
    header: fits.Header
    components: tuple[dict[str, Any], ...]

    def local_eef_curve(self, x_ima: float, y_ima: float, energy_keV: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the weighted-average local EEF curve at one stacked-image pixel.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(radius_pix, eef_fraction)`` for the weighted local PSF.
        """
        x_idx = int(round(float(x_ima) - 1.0))
        y_idx = int(round(float(y_ima) - 1.0))
        curves: list[np.ndarray] = []
        weights: list[float] = []
        for component in self.components:
            theta_map = component["theta_map"]
            weight_map = component["weight_map"]
            if y_idx < 0 or y_idx >= theta_map.shape[0] or x_idx < 0 or x_idx >= theta_map.shape[1]:
                continue
            weight = float(weight_map[y_idx, x_idx])
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            mapper: ObservationPSFMapper = component["mapper"]
            theta_arcmin = float(theta_map[y_idx, x_idx])
            _, frac = mapper.eef_curve(theta_arcmin, energy_keV=energy_keV)
            if not np.allclose(mapper.radius_grid, self.radius_grid):
                frac = np.interp(self.radius_grid, mapper.radius_grid, frac, left=0.0, right=float(frac[-1]))
            curves.append(np.maximum.accumulate(np.clip(frac, 0.0, 1.0)))
            weights.append(weight)
        if not weights:
            raise ValueError("No valid PSF support exists at the requested stacked-image position.")
        weight_array = np.asarray(weights, dtype=np.float64)
        stack = np.vstack(curves)
        avg = np.average(stack, axis=0, weights=weight_array)
        return self.radius_grid, np.maximum.accumulate(np.clip(avg, 0.0, 1.0))

    def eef_at_position(self, x_ima: float, y_ima: float, radius_pix: float, energy_keV: float | None = None) -> float:
        """Evaluate the weighted local EEF at one radius.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        radius_pix : float
            Radius in pixels.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Encircled-energy fraction.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return float(np.interp(float(radius_pix), radius_grid, frac, left=0.0, right=float(frac[-1])))

    def radius_at_position(self, x_ima: float, y_ima: float, frac_value: float, energy_keV: float | None = None) -> float:
        """Return the weighted local radius at one EEF fraction.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        frac_value : float
            Requested EEF fraction in ``[0, 1]``.
        energy_keV : float | None, optional
            Requested energy in keV.

        Returns
        -------
        float
            Radius in pixels.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return float(np.interp(float(np.clip(frac_value, 0.0, 1.0)), frac, radius_grid))

    def kernel_at_position(self, x_ima: float, y_ima: float, energy_keV: float | None = None, size: int | None = None) -> np.ndarray:
        """Build the weighted local PSF kernel at one stacked-image position.

        Parameters
        ----------
        x_ima : float
            X image coordinate in 1-based pixels.
        y_ima : float
            Y image coordinate in 1-based pixels.
        energy_keV : float | None, optional
            Requested energy in keV.
        size : int | None, optional
            Optional kernel size in pixels.

        Returns
        -------
        np.ndarray
            Normalized 2D kernel.
        """
        radius_grid, frac = self.local_eef_curve(x_ima, y_ima, energy_keV=energy_keV)
        return build_psf_kernel(radius_grid, frac, size=size)

    def write(self, path: str | Path) -> str:
        """Write the stacked mapper product to FITS.

        Parameters
        ----------
        path : str | Path
            Output FITS path.

        Returns
        -------
        str
            Written file path.
        """
        header = self.header.copy()
        header["PSFTYPE"] = "STACK"
        header["MISSION"] = self.mission
        header["DEFENERG"] = float(self.default_energy_keV)
        header["NCOMP"] = len(self.components)
        hdus: list[fits.HDUBase] = [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=np.asarray(self.radius_grid, dtype=np.float32), name="RADIUS_GRID"),
        ]
        comp_rows = []
        for idx, component in enumerate(self.components, start=1):
            mapper: ObservationPSFMapper = component["mapper"]
            suffix = f"{idx:02d}"
            hdus.extend(
                [
                    fits.ImageHDU(data=np.asarray(component["theta_map"], dtype=np.float32), header=self.header.copy(), name=f"THETA_{suffix}"),
                    fits.ImageHDU(data=np.asarray(component["weight_map"], dtype=np.float32), header=self.header.copy(), name=f"WEIGHT_{suffix}"),
                    fits.ImageHDU(data=np.asarray(mapper.energy_grid, dtype=np.float32), name=f"EGRID_{suffix}"),
                    fits.ImageHDU(data=np.asarray(mapper.theta_grid, dtype=np.float32), name=f"TGRID_{suffix}"),
                    fits.ImageHDU(data=np.asarray(mapper.radius_grid, dtype=np.float32), name=f"RGRID_{suffix}"),
                    fits.ImageHDU(data=np.asarray(mapper.eef_cube, dtype=np.float32), name=f"EEF_{suffix}"),
                ]
            )
            beta = mapper.beta_table
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(name="E_MID", format="E", array=np.asarray(beta["e_mid"], dtype=np.float32)),
                        fits.Column(name="A1", format="E", array=np.asarray(beta["A1"], dtype=np.float32)),
                        fits.Column(name="R1", format="E", array=np.asarray(beta["R1"], dtype=np.float32)),
                        fits.Column(name="ALP1", format="E", array=np.asarray(beta["ALP1"], dtype=np.float32)),
                        fits.Column(name="A2", format="E", array=np.asarray(beta["A2"], dtype=np.float32)),
                        fits.Column(name="R2", format="E", array=np.asarray(beta["R2"], dtype=np.float32)),
                        fits.Column(name="ALP2", format="E", array=np.asarray(beta["ALP2"], dtype=np.float32)),
                    ],
                    name=f"BETA_{suffix}",
                )
            )
            comp_rows.append(
                (
                    component["name"],
                    mapper.detector,
                    mapper.filter_name,
                    suffix,
                    float(mapper.default_energy_keV),
                )
            )
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="NAME", format="64A", array=[row[0] for row in comp_rows]),
                    fits.Column(name="DETECTOR", format="8A", array=[row[1] for row in comp_rows]),
                    fits.Column(name="FILTER", format="16A", array=[row[2] for row in comp_rows]),
                    fits.Column(name="SUFFIX", format="2A", array=[row[3] for row in comp_rows]),
                    fits.Column(name="DEFENERG", format="E", array=np.asarray([row[4] for row in comp_rows], dtype=np.float32)),
                ],
                name="COMPONENTS",
            )
        )
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return str(path)

    @classmethod
    def read(cls, path: str | Path) -> "StackedPSFMapper":
        """Read one stacked mapper product from FITS.

        Parameters
        ----------
        path : str | Path
            Stacked mapper FITS path.

        Returns
        -------
        StackedPSFMapper
            Loaded stacked mapper.
        """
        with fits.open(path) as hdul:
            header = hdul[0].header.copy()
            radius_grid = np.asarray(hdul["RADIUS_GRID"].data, dtype=np.float64)
            components = []
            for row in hdul["COMPONENTS"].data:
                suffix = str(row["SUFFIX"]).strip()
                beta = hdul[f"BETA_{suffix}"].data
                mapper = ObservationPSFMapper(
                    mission=str(header.get("MISSION", "ep-fxt")),
                    detector=str(row["DETECTOR"]).strip(),
                    filter_name=str(row["FILTER"]).strip(),
                    energy_grid=np.asarray(hdul[f"EGRID_{suffix}"].data, dtype=np.float64),
                    theta_grid=np.asarray(hdul[f"TGRID_{suffix}"].data, dtype=np.float64),
                    radius_grid=np.asarray(hdul[f"RGRID_{suffix}"].data, dtype=np.float64),
                    eef_cube=np.asarray(hdul[f"EEF_{suffix}"].data, dtype=np.float64),
                    beta_table={
                        "e_mid": np.asarray(beta["E_MID"], dtype=np.float64),
                        "A1": np.asarray(beta["A1"], dtype=np.float64),
                        "R1": np.asarray(beta["R1"], dtype=np.float64),
                        "ALP1": np.asarray(beta["ALP1"], dtype=np.float64),
                        "A2": np.asarray(beta["A2"], dtype=np.float64),
                        "R2": np.asarray(beta["R2"], dtype=np.float64),
                        "ALP2": np.asarray(beta["ALP2"], dtype=np.float64),
                    },
                    theta_map_arcmin=np.asarray(hdul[f"THETA_{suffix}"].data, dtype=np.float64),
                    header=hdul[f"THETA_{suffix}"].header.copy(),
                    default_energy_keV=float(row["DEFENERG"]),
                )
                components.append(
                    {
                        "name": str(row["NAME"]).strip(),
                        "mapper": mapper,
                        "theta_map": np.asarray(hdul[f"THETA_{suffix}"].data, dtype=np.float64),
                        "weight_map": np.asarray(hdul[f"WEIGHT_{suffix}"].data, dtype=np.float64),
                    }
                )
            return cls(
                mission=str(header.get("MISSION", "ep-fxt")),
                default_energy_keV=float(header["DEFENERG"]),
                radius_grid=radius_grid,
                header=header,
                components=tuple(components),
            )


def build_observation_psf_mapper(
    image_path: str | Path,
    expmap_path: str | Path | None = None,
    *,
    metadata: ObservationMetadata | None = None,
    instrument: str | None = None,
    filter_name: str | None = None,
    emin_keV: float | None = None,
    emax_keV: float | None = None,
) -> ObservationPSFMapper:
    """Build one per-observation PSF mapper product.

    Parameters
    ----------
    image_path : str | Path
        Image FITS file used for WCS and native geometry.
    expmap_path : str | Path | None, optional
        Unused placeholder kept for symmetry with higher-level workflows.
    metadata : ObservationMetadata | None, optional
        Optional observation metadata override. When supplied, the mapper uses
        it instead of rereading pointing and detector keywords from
        ``image_path``.
    instrument : str | None, optional
        Optional detector-arm override such as ``fxta``.
    filter_name : str | None, optional
        Optional filter-family override such as ``open``.
    emin_keV : float | None, optional
        Lower image energy bound in keV.
    emax_keV : float | None, optional
        Upper image energy bound in keV.

    Returns
    -------
    ObservationPSFMapper
        Built observation mapper.
    """
    del expmap_path
    metadata = metadata or read_observation_metadata(str(image_path), preferred_ext=0)
    with fits.open(image_path) as hdul:
        header = hdul[0].header.copy()
        data_shape = np.asarray(hdul[0].data).shape
    if len(data_shape) != 2:
        raise ValueError(f"{image_path} is not a 2D FITS image.")
    instrument_name = instrument or _instrument_from_detector(metadata.detnam)
    filter_token = filter_name or _filter_name(metadata.filt)
    energy_grid, theta_grid, radius_grid, eef_cube = _build_eef_cube(instrument_name, filter_token)
    beta_table = load_beta_psf_table(metadata.detnam)
    theta_map_arcmin = _project_theta_map(header, metadata)
    return ObservationPSFMapper(
        mission="ep-fxt",
        detector=instrument_name,
        filter_name=filter_token,
        energy_grid=energy_grid,
        theta_grid=theta_grid,
        radius_grid=radius_grid,
        eef_cube=eef_cube,
        beta_table=beta_table,
        theta_map_arcmin=theta_map_arcmin,
        header=header,
        default_energy_keV=_default_energy(emin_keV, emax_keV, energy_grid),
    )


def build_stacked_psf_mapper(
    observation_products: list[str | Path],
    weight_maps: list[str | Path],
    ref_header: fits.Header,
) -> StackedPSFMapper:
    """Build one stacked PSF mapper from per-observation products and weights.

    Parameters
    ----------
    observation_products : list[str | Path]
        Per-observation mapper product paths.
    weight_maps : list[str | Path]
        Per-observation vignetted weight/exposure map paths.
    ref_header : fits.Header
        Reference stacked-image header with celestial WCS and image shape.

    Returns
    -------
    StackedPSFMapper
        Stacked mapper aligned to the reference WCS.
    """
    if len(observation_products) != len(weight_maps):
        raise ValueError("observation_products and weight_maps must have the same length.")
    ref_wcs = WCS(ref_header).celestial
    ref_shape = (int(ref_header["NAXIS2"]), int(ref_header["NAXIS1"]))
    components = []
    radius_grids = []
    default_energies = []
    for obs_path, weight_path in zip(observation_products, weight_maps):
        mapper = ObservationPSFMapper.read(obs_path)
        radius_grids.append(mapper.radius_grid)
        default_energies.append(mapper.default_energy_keV)
        theta_native_header = mapper.header.copy()
        theta_reproj, theta_foot = reproject_interp((mapper.theta_map_arcmin, WCS(theta_native_header).celestial), ref_wcs, shape_out=ref_shape)
        with fits.open(weight_path) as hdul:
            weight_data = np.asarray(hdul[0].data, dtype=np.float64)
            weight_wcs = WCS(hdul[0].header).celestial
        weight_reproj, weight_foot = reproject_interp((weight_data, weight_wcs), ref_wcs, shape_out=ref_shape)
        valid = np.isfinite(theta_reproj) & (theta_foot > 0) & np.isfinite(weight_reproj) & (weight_foot > 0) & (weight_reproj > 0.0)
        theta_map = np.where(valid, theta_reproj, np.nan)
        weight_map = np.where(valid, weight_reproj, 0.0)
        components.append(
            {
                "name": Path(obs_path).name,
                "mapper": mapper,
                "theta_map": theta_map,
                "weight_map": weight_map,
            }
        )
    common_radius = np.unique(np.concatenate(radius_grids)).astype(np.float64)
    return StackedPSFMapper(
        mission="ep-fxt",
        default_energy_keV=float(np.mean(default_energies)),
        radius_grid=common_radius,
        header=ref_header.copy(),
        components=tuple(components),
    )


def load_psf_product(path: str | Path) -> ObservationPSFMapper | StackedPSFMapper:
    """Load a PSF mapper product from disk.

    Parameters
    ----------
    path : str | Path
        PSF product path.

    Returns
    -------
    ObservationPSFMapper | StackedPSFMapper
        Loaded mapper object.
    """
    psf_type = str(fits.getval(path, "PSFTYPE", ext=0)).strip().upper()
    if psf_type == "OBS":
        return ObservationPSFMapper.read(path)
    if psf_type == "STACK":
        return StackedPSFMapper.read(path)
    raise ValueError(f"Unsupported PSF product type in {path}: {psf_type!r}")
