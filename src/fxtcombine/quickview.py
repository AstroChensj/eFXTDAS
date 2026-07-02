#!/usr/bin/env python3
"""Create quick-view summary figures for ``fxtcombine`` stacked products."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from cycler import cycler
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from tqdm.auto import tqdm

from fxtcombine.utils.logger import build_cli_logger, emit
from fxtpsfgen.mapper import load_psf_product, radius_extension_name


@dataclass(frozen=True)
class QuickViewPaths:
    """Resolved paths to the stacked products used by the quick-view plot."""

    stack_dir: Path
    rate: Path
    bkgmap: Path
    mask: Path
    expmap: Path
    psfprod: Path
    src_reg: Path
    target_src_reg: Path
    target_bkg_reg: Path


@dataclass(frozen=True)
class CircleRegion:
    """One circular region in either image or sky coordinates."""

    x: float | None
    y: float | None
    center: SkyCoord | None
    radius_arcsec: float | None
    radius_pix: float | None


@dataclass(frozen=True)
class AnnulusRegion:
    """One annular region in sky coordinates."""

    center: SkyCoord
    r_in_arcsec: float
    r_out_arcsec: float


def plt_style_setup() -> None:
    """Apply the copied FancyPlots plotting style.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Matplotlib global rcParams are updated in place.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Times New Roman"],
            "font.size": 12.0,
            "axes.labelsize": 16,
            "axes.titlesize": 30,
            "axes.linewidth": 2.5,
            "axes.labelweight": "light",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.size": 6,
            "xtick.major.width": 1.5,
            "xtick.minor.size": 4,
            "xtick.minor.width": 1,
            "xtick.direction": "in",
            "xtick.top": True,
            "xtick.major.pad": 9,
            "ytick.major.size": 6,
            "ytick.major.width": 1,
            "ytick.minor.size": 4,
            "ytick.minor.width": 1.5,
            "ytick.direction": "in",
            "ytick.right": True,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "legend.frameon": False,
            "figure.figsize": [13, 12],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "grid.linewidth": 1,
            "lines.linewidth": 1.5,
            "lines.markersize": 3,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.5,
            "mathtext.fontset": "stix",
            "axes.prop_cycle": cycler(
                color=[
                    "#0C5DA5",
                    "#00B945",
                    "#FF9500",
                    "#FF2C00",
                    "#845B97",
                    "#474747",
                    "#9e9e9e",
                ]
            ),
        }
    )


def resolve_paths(
    stack_dir: Path,
    *,
    rate: Path | None = None,
    bkgmap: Path | None = None,
    mask: Path | None = None,
    expmap: Path | None = None,
    psfprod: Path | None = None,
    src_reg: Path | None = None,
    target_src_reg: Path | None = None,
    target_bkg_reg: Path | None = None,
) -> QuickViewPaths:
    """Resolve standard fxtcombine product paths from one stack directory.

    Parameters
    ----------
    stack_dir : Path
        Directory containing current ``fxtcombine`` stacked products.
    rate, bkgmap, mask, expmap, psfprod, src_reg, target_src_reg, target_bkg_reg : Path | None
        Optional overrides for nonstandard product locations.

    Returns
    -------
    QuickViewPaths
        Resolved path bundle.
    """
    root = Path(stack_dir).expanduser().resolve()
    return QuickViewPaths(
        stack_dir=root,
        rate=(rate or root / "stack_rate.fits").expanduser().resolve(),
        bkgmap=(bkgmap or root / "stack_bkgmap.fits").expanduser().resolve(),
        mask=(mask or root / "stack_mask.fits").expanduser().resolve(),
        expmap=(expmap or root / "stack_expmap.fits").expanduser().resolve(),
        psfprod=(psfprod or root / "stack_psfprod.fits").expanduser().resolve(),
        src_reg=(src_reg or root / "stack_src.reg").expanduser().resolve(),
        target_src_reg=(target_src_reg or root / "target_src.reg").expanduser().resolve(),
        target_bkg_reg=(target_bkg_reg or root / "target_bkg.reg").expanduser().resolve(),
    )


def validate_paths(paths: QuickViewPaths) -> None:
    """Validate that all required quick-view inputs exist.

    Parameters
    ----------
    paths : QuickViewPaths
        Product paths to validate.

    Returns
    -------
    None
        Raises an exception when a required path is missing.
    """
    missing = [
        str(path)
        for path in (
            paths.rate,
            paths.bkgmap,
            paths.mask,
            paths.expmap,
            paths.psfprod,
            paths.src_reg,
            paths.target_src_reg,
            paths.target_bkg_reg,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required fxtcombine quick-view input(s):\n" + "\n".join(missing))


def _load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a primary FITS image as float data and header.

    Parameters
    ----------
    path : Path
        FITS image path.

    Returns
    -------
    tuple[np.ndarray, fits.Header]
        Floating-point image data and copied primary header.
    """
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64), hdul[0].header.copy()


def _split_region_args(line: str) -> list[str]:
    """Split the arguments inside one DS9 region expression.

    Parameters
    ----------
    line : str
        DS9 region expression such as ``circle(1,2,3")``.

    Returns
    -------
    list[str]
        Comma-separated argument strings.
    """
    inside = line.split("(", 1)[1].split(")", 1)[0]
    return [item.strip() for item in inside.split(",")]


def _parse_radius_arcsec(value: str) -> float:
    """Parse a DS9 radius token as arcseconds.

    Parameters
    ----------
    value : str
        Radius token, usually ending in a double quote.

    Returns
    -------
    float
        Radius in arcseconds.
    """
    token = value.strip()
    if token.endswith('"'):
        return float(token[:-1])
    if token.endswith("'"):
        return 60.0 * float(token[:-1])
    if token.lower().endswith("d"):
        return 3600.0 * float(token[:-1])
    return float(token)


def parse_image_circles(reg_path: Path) -> list[CircleRegion]:
    """Parse image-coordinate circle regions from a DS9 file.

    Parameters
    ----------
    reg_path : Path
        DS9 region file path.

    Returns
    -------
    list[CircleRegion]
        Parsed image-coordinate circles with zero-based pixel centers.
    """
    circles: list[CircleRegion] = []
    for raw_line in reg_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower() in {"image", "fk5"} or line.lower().startswith("global"):
            continue
        if line.startswith("-"):
            continue
        if not line.lower().startswith("circle("):
            continue
        x_str, y_str, r_str = _split_region_args(line)[:3]
        circles.append(
            CircleRegion(
                x=float(x_str) - 1.0,
                y=float(y_str) - 1.0,
                center=None,
                radius_arcsec=None,
                radius_pix=float(r_str),
            )
        )
    return circles


def parse_fk5_regions(reg_path: Path) -> tuple[CircleRegion | AnnulusRegion | None, list[CircleRegion]]:
    """Parse FK5 include and exclude regions from a DS9 file.

    Parameters
    ----------
    reg_path : Path
        DS9 FK5 region file path.

    Returns
    -------
    tuple[CircleRegion | AnnulusRegion | None, list[CircleRegion]]
        Main include region and circular exclude regions.
    """
    main_region: CircleRegion | AnnulusRegion | None = None
    excludes: list[CircleRegion] = []
    for raw_line in reg_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower() in {"image", "fk5"} or line.lower().startswith("global"):
            continue
        exclude = line.startswith("-")
        expr = line[1:].strip() if exclude else line
        lower = expr.lower()
        if lower.startswith("circle("):
            ra_str, dec_str, radius_str = _split_region_args(expr)[:3]
            region = CircleRegion(
                x=None,
                y=None,
                center=SkyCoord(float(ra_str), float(dec_str), unit="deg", frame="fk5"),
                radius_arcsec=_parse_radius_arcsec(radius_str),
                radius_pix=None,
            )
            if exclude:
                excludes.append(region)
            else:
                main_region = region
        elif lower.startswith("annulus(") and not exclude:
            ra_str, dec_str, rin_str, rout_str = _split_region_args(expr)[:4]
            main_region = AnnulusRegion(
                center=SkyCoord(float(ra_str), float(dec_str), unit="deg", frame="fk5"),
                r_in_arcsec=_parse_radius_arcsec(rin_str),
                r_out_arcsec=_parse_radius_arcsec(rout_str),
            )
    return main_region, excludes


def _image_norm(data: np.ndarray, percentile: float = 99.5) -> ImageNormalize | None:
    """Build an image normalization from finite values.

    Parameters
    ----------
    data : np.ndarray
        Image data.
    percentile : float
        Percentile interval width.

    Returns
    -------
    ImageNormalize | None
        Matplotlib-compatible normalization, or ``None`` for empty data.
    """
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    vmin, vmax = PercentileInterval(percentile).get_limits(finite)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return ImageNormalize(vmin=vmin, vmax=vmax)


def _shade_invalid_mask(ax: Any, valid_mask: np.ndarray, alpha: float = 0.5) -> None:
    """Overlay invalid analysis pixels in gray.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    valid_mask : np.ndarray
        Boolean valid-pixel mask.
    alpha : float
        Overlay transparency.

    Returns
    -------
    None
        The axes are modified in place.
    """
    invalid = ~np.asarray(valid_mask, dtype=bool)
    overlay = np.where(invalid, 1.0, np.nan)
    ax.imshow(overlay, origin="lower", cmap=ListedColormap(["gray"]), alpha=alpha, interpolation="nearest")


def _pixel_scale_arcsec(wcs: WCS) -> float:
    """Return the scalar pixel scale for one celestial WCS.

    Parameters
    ----------
    wcs : WCS
        Celestial WCS.

    Returns
    -------
    float
        Pixel scale in arcseconds per pixel.
    """
    scales = proj_plane_pixel_scales(wcs.celestial)
    return float(np.sqrt(abs(scales[0] * scales[1])) * 3600.0)


def compute_r90_map(
    psfprod_path: Path,
    shape: tuple[int, int],
    valid_mask: np.ndarray,
    r90_stride: int = 4,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Compute a stacked PSF R90 map from the current PSF product.

    Parameters
    ----------
    psfprod_path : Path
        Current ``stack_psfprod.fits`` product.
    shape : tuple[int, int]
        Output image shape as ``(ny, nx)``.
    valid_mask : np.ndarray
        Boolean mask selecting pixels where R90 should be evaluated.
    r90_stride : int
        Pixel stride for coarse-grid R90 sampling. ``1`` evaluates every valid
        pixel exactly; larger values sample a grid and interpolate.
    logger : logging.Logger | None
        Optional progress logger.

    Returns
    -------
    np.ndarray
        R90 map in image pixels, with invalid pixels set to ``NaN``.
    """
    r90 = np.full(shape, np.nan, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    valid_pixels = np.argwhere(mask)
    if len(valid_pixels) == 0:
        emit(logger, "warning", "No valid pixels were available for R90 calculation.")
        return r90

    cached = _load_cached_radius_map(psfprod_path, 0.90, shape)
    if cached is not None:
        emit(logger, "info", f"Using cached R90 map from {psfprod_path}")
        r90[mask] = cached[mask].astype(np.float32)
        return r90

    mapper = load_psf_product(psfprod_path)
    stride = max(int(r90_stride), 1)
    if stride == 1:
        sample_pixels = valid_pixels
    else:
        sample_mask = mask.copy()
        yy, xx = np.indices(shape)
        sample_mask &= (yy % stride == 0) & (xx % stride == 0)
        sample_pixels = np.argwhere(sample_mask)
        if len(sample_pixels) < 3:
            sample_pixels = valid_pixels

    emit(
        logger,
        "info",
        (
            f"Computing R90 from {len(sample_pixels)} sampled pixel(s) "
            f"(stride={stride}; valid pixels={len(valid_pixels)}) using {psfprod_path}"
        ),
    )
    sample_values: list[float] = []
    sample_points: list[tuple[float, float]] = []
    for y_idx, x_idx in tqdm(sample_pixels, desc="R90 samples", unit="pix"):
        try:
            value = float(mapper.radius_at_position(float(x_idx) + 1.0, float(y_idx) + 1.0, 0.90))
        except ValueError:
            continue
        r90[y_idx, x_idx] = value
        sample_points.append((float(x_idx), float(y_idx)))
        sample_values.append(value)

    if stride == 1:
        return r90
    if not sample_values:
        emit(logger, "warning", "No R90 samples had valid PSF support.")
        return r90

    points = np.asarray(sample_points, dtype=np.float64)
    values = np.asarray(sample_values, dtype=np.float64)
    grid_y, grid_x = np.indices(shape, dtype=np.float64)
    target_points = np.column_stack((grid_x[mask].ravel(), grid_y[mask].ravel()))

    interpolated = None
    if len(values) >= 3:
        try:
            interpolated = griddata(points, values, target_points, method="linear")
        except Exception as exc:
            emit(logger, "warning", f"Linear R90 interpolation failed; using nearest interpolation only: {exc}")
    if interpolated is None:
        interpolated = np.full(len(target_points), np.nan, dtype=np.float64)

    missing = ~np.isfinite(interpolated)
    if np.any(missing):
        nearest = griddata(points, values, target_points[missing], method="nearest")
        interpolated[missing] = nearest

    r90[mask] = interpolated.astype(np.float32)
    return r90


def _load_cached_radius_map(psfprod_path: Path, frac_value: float, shape: tuple[int, int]) -> np.ndarray | None:
    """Load a cached radius-at-EEF map from a PSF product when present.

    Parameters
    ----------
    psfprod_path : Path
        PSF product path.
    frac_value : float
        Requested EEF fraction.
    shape : tuple[int, int]
        Expected output image shape.

    Returns
    -------
    np.ndarray | None
        Cached radius map in pixels, or ``None`` when unavailable.
    """
    path = Path(psfprod_path)
    if not path.exists():
        return None
    extname = radius_extension_name(frac_value)
    try:
        with fits.open(path) as hdul:
            if extname not in hdul:
                return None
            data = np.asarray(hdul[extname].data, dtype=np.float64)
            if data.shape != tuple(shape):
                return None
            unit = str(hdul[extname].header.get("BUNIT", "pixel")).strip().lower()
            if unit not in {"pixel", "pixels", "pix"}:
                return None
            return data
    except Exception:
        return None


def _add_circle(ax: Any, x: float, y: float, radius: float, **kwargs: Any) -> None:
    """Add one circle patch to an axes.

    Parameters
    ----------
    ax : Any
        Matplotlib axes.
    x, y : float
        Zero-based image coordinates.
    radius : float
        Radius in image pixels.
    **kwargs : Any
        Patch style keyword arguments.

    Returns
    -------
    None
        The axes are modified in place.
    """
    ax.add_patch(Circle((x, y), radius, fill=False, **kwargs))


def _sky_circle_to_pixel(region: CircleRegion, wcs: WCS, pixel_scale_arcsec: float) -> tuple[float, float, float]:
    """Convert one FK5 circular region to pixel coordinates.

    Parameters
    ----------
    region : CircleRegion
        Sky-coordinate circular region.
    wcs : WCS
        Image WCS.
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel.

    Returns
    -------
    tuple[float, float, float]
        Zero-based ``x, y, radius`` in image pixels.
    """
    if region.center is None or region.radius_arcsec is None:
        raise ValueError("Circle region lacks sky coordinates.")
    x, y = wcs.celestial.world_to_pixel(region.center)
    return float(np.asarray(x)), float(np.asarray(y)), float(region.radius_arcsec) / pixel_scale_arcsec


def plot_quickview(
    paths: QuickViewPaths,
    out_path: Path,
    *,
    smooth_sigma: float = 2.0,
    r90_stride: int = 4,
    title: str | None = None,
    dpi: int = 100,
    logger: logging.Logger | None = None,
) -> Path:
    """Generate and save the fxtcombine six-panel quick-view figure.

    Parameters
    ----------
    paths : QuickViewPaths
        Resolved product paths.
    out_path : Path
        Output figure path.
    smooth_sigma : float
        Gaussian smoothing sigma in image pixels for the rate-map panels.
    r90_stride : int
        Pixel stride for panel-6 R90 map sampling. ``1`` uses exact per-pixel
        evaluation.
    title : str | None
        Optional figure title shown above the six-panel layout.
    dpi : int
        Output image DPI.
    logger : logging.Logger | None
        Optional progress logger.

    Returns
    -------
    Path
        Resolved output path.
    """
    emit(logger, "info", "Loading stacked products")
    rate_map, rate_header = _load_image(paths.rate)
    bkg_map, _ = _load_image(paths.bkgmap)
    mask_map, _ = _load_image(paths.mask)
    exp_map, _ = _load_image(paths.expmap)
    valid_mask = np.asarray(mask_map, dtype=np.float64) > 0.0
    rate_wcs = WCS(rate_header)
    pixel_scale = _pixel_scale_arcsec(rate_wcs)
    smoothed_rate = gaussian_filter(rate_map, sigma=float(smooth_sigma))

    detected_circles = parse_image_circles(paths.src_reg)
    target_src_region, target_src_excludes = parse_fk5_regions(paths.target_src_reg)
    target_bkg_region, target_bkg_excludes = parse_fk5_regions(paths.target_bkg_reg)
    if not isinstance(target_src_region, CircleRegion):
        raise ValueError(f"{paths.target_src_reg} must contain one FK5 source circle.")
    if not isinstance(target_bkg_region, AnnulusRegion):
        raise ValueError(f"{paths.target_bkg_reg} must contain one FK5 background annulus.")

    rate_norm = _image_norm(smoothed_rate)
    bkg_norm = _image_norm(bkg_map)
    exp_norm = _image_norm(exp_map)

    plt_style_setup()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=30, fontweight="bold")
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()
    text_pe = [pe.withStroke(linewidth=3, foreground="black")]

    emit(logger, "info", "Generating panel 1: smoothed stacked rate map")
    ax1.imshow(smoothed_rate, origin="lower", cmap="magma", norm=rate_norm)
    _shade_invalid_mask(ax1, valid_mask, alpha=0.50)
    for region in detected_circles:
        if region.x is not None and region.y is not None and region.radius_pix is not None:
            _add_circle(ax1, region.x, region.y, region.radius_pix, edgecolor="cyan", linewidth=0.9, alpha=0.85)
    ax1.text(
        0.03,
        0.04,
        f"Nsrc = {len(detected_circles)}",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        color="yellow",
        fontsize=16,
        path_effects=text_pe,
    )
    ax1.set_title(r"$\mathbf{Stacked\ Rate\ Map}$")
    ax1.set_xlabel("X (pix)")
    ax1.set_ylabel("Y (pix)")

    emit(logger, "info", "Generating panel 2: stacked background map")
    im2 = ax2.imshow(bkg_map, origin="lower", cmap="inferno", norm=bkg_norm)
    ax2.set_title(r"$\mathbf{Stacked\ Background\ Map}$")
    ax2.set_xlabel("X (pix)")
    ax2.set_ylabel("Y (pix)")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("Background counts / pixel")

    emit(logger, "info", "Generating panel 3: target zoom and extraction regions")
    ax3.imshow(smoothed_rate, origin="lower", cmap="magma", norm=rate_norm)
    target_x, target_y, src_radius_pix = _sky_circle_to_pixel(target_src_region, rate_wcs, pixel_scale)
    bkg_x, bkg_y = rate_wcs.celestial.world_to_pixel(target_bkg_region.center)
    bkg_x = float(np.asarray(bkg_x))
    bkg_y = float(np.asarray(bkg_y))
    bkg_r_in_pix = target_bkg_region.r_in_arcsec / pixel_scale
    bkg_r_out_pix = target_bkg_region.r_out_arcsec / pixel_scale
    _add_circle(ax3, target_x, target_y, src_radius_pix, edgecolor="cyan", linewidth=3.0)
    _add_circle(ax3, bkg_x, bkg_y, bkg_r_in_pix, edgecolor="crimson", linewidth=2.5)
    _add_circle(ax3, bkg_x, bkg_y, bkg_r_out_pix, edgecolor="crimson", linewidth=2.5)
    zoom_radius_pix = max(bkg_r_out_pix, src_radius_pix)
    for region in target_bkg_excludes:
        ex_x, ex_y, ex_r = _sky_circle_to_pixel(region, rate_wcs, pixel_scale)
        _add_circle(ax3, ex_x, ex_y, ex_r, edgecolor="white", linewidth=1.2, linestyle="--")
        slash_half = ex_r / np.sqrt(2.0)
        ax3.plot([ex_x - slash_half, ex_x + slash_half], [ex_y - slash_half, ex_y + slash_half], color="white", linewidth=1.2)
        zoom_radius_pix = max(zoom_radius_pix, np.hypot(ex_x - target_x, ex_y - target_y) + ex_r)
    for region in target_src_excludes:
        ex_x, ex_y, ex_r = _sky_circle_to_pixel(region, rate_wcs, pixel_scale)
        _add_circle(ax3, ex_x, ex_y, ex_r, edgecolor="yellow", linewidth=1.5, linestyle="--")
        slash_half = ex_r / np.sqrt(2.0)
        ax3.plot([ex_x - slash_half, ex_x + slash_half], [ex_y - slash_half, ex_y + slash_half], color="yellow", linewidth=1.5)
        zoom_radius_pix = max(zoom_radius_pix, np.hypot(ex_x - target_x, ex_y - target_y) + ex_r)
    zoom_radius_pix += 10.0
    ax3.set_xlim(target_x - zoom_radius_pix, target_x + zoom_radius_pix)
    ax3.set_ylim(target_y - zoom_radius_pix, target_y + zoom_radius_pix)
    ax3.set_title(r"$\mathbf{Target\ Zoom\ +\ Extraction\ Regions}$")
    ax3.set_xlabel("X (pix)")
    ax3.set_ylabel("Y (pix)")

    emit(logger, "info", "Generating panel 4: stacked analysis mask")
    im4 = ax4.imshow(valid_mask.astype(float), origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    ax4.set_title(r"$\mathbf{Stacked\ Analysis\ Mask}$")
    ax4.set_xlabel("X (pix)")
    ax4.set_ylabel("Y (pix)")
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("Valid pixel")

    emit(logger, "info", "Generating panel 5: stacked exposure map")
    im5 = ax5.imshow(exp_map, origin="lower", cmap="viridis", norm=exp_norm)
    ax5.set_title(r"$\mathbf{Stacked\ Exposure\ Map}$")
    ax5.set_xlabel("X (pix)")
    ax5.set_ylabel("Y (pix)")
    cb5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.set_label("Exposure (s)")

    emit(logger, "info", "Generating panel 6: stacked PSF R90 map")
    r90_map = compute_r90_map(
        paths.psfprod,
        rate_map.shape,
        valid_mask & np.isfinite(exp_map) & (exp_map > 0.0),
        r90_stride=r90_stride,
        logger=logger,
    )
    im6 = ax6.imshow(r90_map, origin="lower", cmap="cividis", norm=_image_norm(r90_map))
    ax6.set_title(r"$\mathbf{Stacked\ PSF\ Map\ (R90)}$")
    ax6.set_xlabel("X (pix)")
    ax6.set_ylabel("Y (pix)")
    cb6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cb6.set_label("R90 (pix)")

    for ax in axes.ravel():
        ax.tick_params(axis="both", which="both", length=6, width=1, labelsize=14)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    resolved_out = Path(out_path).expanduser().resolve()
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    emit(logger, "info", f"Saving figure to {resolved_out}")
    fig.savefig(resolved_out, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return resolved_out


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``fxtcombine-quickview``.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Create a six-panel quick-view plot from fxtcombine stacked products.")
    parser.add_argument("stack_dir", type=Path, help="Directory containing fxtcombine stacked products.")
    parser.add_argument("--out", type=Path, default=None, help="Output figure path. Default: <stack_dir>/quickview.png")
    parser.add_argument("--smooth-sigma", type=float, default=2.0, help="Gaussian smoothing sigma for stacked rate-map panels. Default: 2")
    parser.add_argument("--r90-stride", type=int, default=4, help="Pixel stride for R90 sampling. Use 1 for exact per-pixel R90. Default: 4")
    parser.add_argument("--title", default=None, help="Optional title shown above the quick-view figure.")
    parser.add_argument("--dpi", type=int, default=100, help="Output figure DPI. Default: 100")
    parser.add_argument("--rate", type=Path, default=None, help="Override stacked rate FITS path.")
    parser.add_argument("--bkgmap", type=Path, default=None, help="Override stacked background map FITS path.")
    parser.add_argument("--mask", type=Path, default=None, help="Override stacked analysis mask FITS path.")
    parser.add_argument("--expmap", type=Path, default=None, help="Override stacked exposure map FITS path.")
    parser.add_argument("--psfprod", type=Path, default=None, help="Override stacked PSF product FITS path.")
    parser.add_argument("--src-reg", type=Path, default=None, help="Override detected-source image region path.")
    parser.add_argument("--target-src-reg", type=Path, default=None, help="Override target source FK5 region path.")
    parser.add_argument("--target-bkg-reg", type=Path, default=None, help="Override target background FK5 region path.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the quick-view command-line interface.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument vector override.

    Returns
    -------
    None
        The quick-view figure is written to disk.
    """
    args = build_parser().parse_args(argv)
    logger = build_cli_logger("eFXTDAS.fxtcombine.quickview", args.log_level, args.log_file)
    paths = resolve_paths(
        args.stack_dir,
        rate=args.rate,
        bkgmap=args.bkgmap,
        mask=args.mask,
        expmap=args.expmap,
        psfprod=args.psfprod,
        src_reg=args.src_reg,
        target_src_reg=args.target_src_reg,
        target_bkg_reg=args.target_bkg_reg,
    )
    validate_paths(paths)
    out_path = args.out if args.out is not None else paths.stack_dir / "quickview.png"
    result = plot_quickview(
        paths,
        out_path,
        smooth_sigma=args.smooth_sigma,
        r90_stride=args.r90_stride,
        title=args.title,
        dpi=args.dpi,
        logger=logger,
    )
    emit(logger, "info", f"Wrote quick-view figure: {result}")


if __name__ == "__main__":
    main()
