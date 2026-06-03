"""I/O helpers for FXT extraction-region generation."""

from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import FITSFixedWarning


def read_ccd(path: Path) -> CCDData:
    """Read a FITS image into a CCDData object.

    Parameters
    ----------
    path : Path
        FITS image path.

    Returns
    -------
    ccd : CCDData
        Image data with metadata and WCS.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        try:
            ccd = CCDData.read(path)
        except ValueError:
            ccd = CCDData.read(path, unit=u.dimensionless_unscaled)
    if ccd.data.ndim != 2:
        raise ValueError(f"{path} is not a 2D FITS image.")
    return ccd


def load_catalog(path: Path) -> Table:
    """Load a source catalog from CSV or FITS.

    Parameters
    ----------
    path : Path
        Catalog file path.

    Returns
    -------
    catalog : Table
        Source catalog table.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return Table.read(path, format="ascii.csv")
    if suffix in {".fits", ".fit", ".fts"}:
        return Table.read(path)
    raise ValueError(f"Unsupported catalog format: {path}")
