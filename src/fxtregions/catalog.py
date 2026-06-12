"""Catalog helpers for FXT extraction-region generation."""

from __future__ import annotations

from astropy.coordinates import SkyCoord
from astropy.table import Table


def find_column(table: Table, candidates: list[str]) -> str:
    """Find the first matching column name in a table.

    Parameters
    ----------
    table : Table
        Input Astropy table.
    candidates : list[str]
        Candidate column names in priority order.

    Returns
    -------
    name : str
        Matching table column name.
    """
    lower_map = {name.lower(): name for name in table.colnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of the columns {candidates} were found")


def match_target(catalog: Table, target: SkyCoord, threshold_arcsec: float) -> tuple[int | None, SkyCoord, float]:
    """Match a target sky position against the source catalog.

    Parameters
    ----------
    catalog : Table
        Source catalog table.
    target : SkyCoord
        User-supplied target coordinate.
    threshold_arcsec : float
        Maximum allowed matching separation.

    Returns
    -------
    match : tuple[int | None, SkyCoord, float]
        ``(row_index, adopted_coord, separation_arcsec)``.
    """
    ra_col = find_column(catalog, ["RA"])
    dec_col = find_column(catalog, ["DEC"])
    cat_coords = SkyCoord(catalog[ra_col], catalog[dec_col], unit="deg")
    sep = target.separation(cat_coords).arcsec
    idx = int(sep.argmin())
    if float(sep[idx]) <= threshold_arcsec:
        return idx, cat_coords[idx], float(sep[idx])
    return None, target, float(sep[idx])


def nearest_catalog_row(catalog: Table, target: SkyCoord) -> Table:
    """Return the nearest catalog row to a target sky coordinate.

    Parameters
    ----------
    catalog : Table
        Source catalog table.
    target : SkyCoord
        User-supplied target coordinate.

    Returns
    -------
    row : Table
        Nearest table row.
    """
    ra_col = find_column(catalog, ["RA"])
    dec_col = find_column(catalog, ["DEC"])
    cat_coords = SkyCoord(catalog[ra_col], catalog[dec_col], unit="deg")
    idx = int(target.separation(cat_coords).arcsec.argmin())
    return catalog[idx]
