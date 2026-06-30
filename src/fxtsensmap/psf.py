"""PSF-radius loading helpers for ``fxtsensmap``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from fxtpsfgen.mapper import ObservationPSFMapper, StackedPSFMapper, load_psf_product, radius_extension_name


def infer_pixel_scale_arcsec(header: fits.Header, fallback: float = 9.6687) -> float:
    """Infer the geometric-mean image pixel scale.

    Parameters
    ----------
    header : fits.Header
        FITS image header.
    fallback : float, optional
        Pixel scale in arcsec/pixel used when WCS metadata is unavailable.

    Returns
    -------
    float
        Pixel scale in arcsec per pixel.
    """
    wcs = WCS(header).celestial
    if not getattr(wcs, "has_celestial", False):
        return float(fallback)
    scales = proj_plane_pixel_scales(wcs)
    if len(scales) < 2:
        return float(fallback)
    scale = float(np.sqrt(abs(scales[0] * scales[1])) * 3600.0)
    return scale if np.isfinite(scale) and scale > 0.0 else float(fallback)


def load_radius_map(
    *,
    psfprod: Path | None,
    psfmap: Path | None,
    eef: float,
    target_header: fits.Header,
    target_shape: tuple[int, int],
    energy_keV: float | None = None,
    block_rows: int = 64,
    nworkers: int = 1,
) -> np.ndarray:
    """Load or compute an aperture radius map in target image pixels.

    Parameters
    ----------
    psfprod : Path | None
        ``fxtpsfgen`` PSF product path.
    psfmap : Path | None
        Official ``fxtpsfmap`` radius-image path.
    eef : float
        Requested encircled-energy fraction.
    target_header : fits.Header
        Header of the sensitivity-map grid.
    target_shape : tuple[int, int]
        Target image shape as ``(ny, nx)``.
    energy_keV : float | None, optional
        Requested PSF energy for computed ``psfprod`` maps.
    block_rows : int, optional
        Row block size for stacked ``psfprod`` computation.
    nworkers : int, optional
        Thread workers for stacked ``psfprod`` computation.

    Returns
    -------
    np.ndarray
        Radius map in target-image pixels.
    """
    if (psfprod is None) == (psfmap is None):
        raise ValueError("Provide exactly one of psfprod or psfmap.")
    if psfprod is not None:
        return load_psfprod_radius_map(
            psfprod,
            eef=eef,
            target_shape=target_shape,
            energy_keV=energy_keV,
            block_rows=block_rows,
            nworkers=nworkers,
        )
    return load_official_psfmap_radius_map(
        psfmap,
        eef=eef,
        target_header=target_header,
        target_shape=target_shape,
    )


def load_psfprod_radius_map(
    path: str | Path,
    *,
    eef: float,
    target_shape: tuple[int, int],
    energy_keV: float | None = None,
    block_rows: int = 64,
    nworkers: int = 1,
) -> np.ndarray:
    """Load a cached ``fxtpsfgen`` radius map or compute it from the mapper.

    Parameters
    ----------
    path : str | Path
        PSF product path.
    eef : float
        Requested encircled-energy fraction.
    target_shape : tuple[int, int]
        Expected output shape.
    energy_keV : float | None, optional
        Requested PSF energy.
    block_rows : int, optional
        Row block size for stacked products.
    nworkers : int, optional
        Thread workers for stacked products.

    Returns
    -------
    np.ndarray
        Radius map in image pixels.
    """
    cached = read_cached_radius_extension(path, eef=eef, target_shape=target_shape)
    if cached is not None:
        return cached
    mapper = load_psf_product(path)
    if isinstance(mapper, StackedPSFMapper):
        radius = mapper.radius_map(eef, energy_keV=energy_keV, block_rows=block_rows, nworkers=nworkers)
    elif isinstance(mapper, ObservationPSFMapper):
        radius = mapper.radius_map(eef, energy_keV=energy_keV)
    else:
        raise TypeError(f"Unsupported PSF product object: {type(mapper)!r}")
    if radius.shape != tuple(target_shape):
        raise ValueError(f"PSF radius map shape {radius.shape} does not match target shape {target_shape}.")
    return np.asarray(radius, dtype=np.float64)


def read_cached_radius_extension(
    path: str | Path,
    *,
    eef: float,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Read a cached Rxx extension from a PSF product if available.

    Parameters
    ----------
    path : str | Path
        PSF product path.
    eef : float
        Requested encircled-energy fraction.
    target_shape : tuple[int, int]
        Expected image shape.

    Returns
    -------
    np.ndarray | None
        Cached radius map in pixels, or ``None``.
    """
    extname = radius_extension_name(eef)
    with fits.open(path) as hdul:
        if extname not in hdul:
            return None
        hdu = hdul[extname]
        data = np.asarray(hdu.data, dtype=np.float64)
        if data.shape != tuple(target_shape):
            raise ValueError(f"Cached {extname} shape {data.shape} does not match target shape {target_shape}.")
        header_eef = hdu.header.get("EEF")
        if header_eef is not None and abs(float(header_eef) - float(eef)) > 1.0e-6:
            raise ValueError(f"Cached {extname} EEF={header_eef} does not match requested EEF={eef}.")
        unit = str(hdu.header.get("BUNIT", "pixel")).strip().lower()
        if unit not in {"pixel", "pixels", "pix"}:
            raise ValueError(f"Cached {extname} has unsupported BUNIT={unit!r}; expected pixels.")
        return data


def load_official_psfmap_radius_map(
    path: str | Path,
    *,
    eef: float,
    target_header: fits.Header,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Load an official ``fxtpsfmap`` radius image.

    Parameters
    ----------
    path : str | Path
        Official PSF map path.
    eef : float
        Requested encircled-energy fraction.
    target_header : fits.Header
        Header of the target image grid.
    target_shape : tuple[int, int]
        Target image shape as ``(ny, nx)``.

    Returns
    -------
    np.ndarray
        Radius map in target-image pixels.
    """
    with fits.open(path) as hdul:
        radius = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    header_eef = header.get("EEF", header.get("ECF"))
    if header_eef is not None and abs(float(header_eef) - float(eef)) > 1.0e-6:
        raise ValueError(f"PSF map EEF={header_eef} does not match requested EEF={eef}.")
    if radius.shape != tuple(target_shape):
        radius = _reproject_radius_map(radius, header, target_header, target_shape)
    unit = str(header.get("BUNIT", "")).strip().lower()
    if unit in {"arcsec", "arcsecond", "arcseconds"}:
        radius = radius / infer_pixel_scale_arcsec(target_header)
    elif unit in {"pixel", "pixels", "pix"}:
        radius = radius
    else:
        raise ValueError(f"Unsupported official PSF map BUNIT={unit!r}; expected arcsec or pixel.")
    return radius


def _reproject_radius_map(
    data: np.ndarray,
    source_header: fits.Header,
    target_header: fits.Header,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject a PSF-radius map onto the target grid.

    Parameters
    ----------
    data : np.ndarray
        Source radius map.
    source_header : fits.Header
        Source WCS header.
    target_header : fits.Header
        Target WCS header.
    target_shape : tuple[int, int]
        Target image shape.

    Returns
    -------
    np.ndarray
        Reprojected radius map.
    """
    source_wcs = WCS(source_header).celestial
    target_wcs = WCS(target_header).celestial
    if not getattr(source_wcs, "has_celestial", False) or not getattr(target_wcs, "has_celestial", False):
        raise ValueError("PSF map shape differs from target and celestial WCS is unavailable for reprojection.")
    reproj, footprint = reproject_interp((data, source_wcs), target_wcs, shape_out=target_shape)
    return np.where(footprint > 0.0, reproj, np.nan)
