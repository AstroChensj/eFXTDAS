"""DS9 region formatting helpers for FXT extraction regions."""

from __future__ import annotations

from pathlib import Path

from astropy.coordinates import SkyCoord


def ds9_circle(coord: SkyCoord, radius_arcsec: float) -> str:
    """Format a DS9 FK5 circle region string."""
    return f"circle({coord.ra.deg:.8f},{coord.dec.deg:.8f},{radius_arcsec:.3f}\")"


def ds9_annulus(coord: SkyCoord, r_in_arcsec: float, r_out_arcsec: float) -> str:
    """Format a DS9 FK5 annulus region string."""
    return f"annulus({coord.ra.deg:.8f},{coord.dec.deg:.8f},{r_in_arcsec:.3f}\",{r_out_arcsec:.3f}\")"


def write_region_file(path: Path, include_region: str, exclude_regions: list[str]) -> None:
    """Write a DS9 region file with one include region and multiple exclusions."""
    with path.open("w") as fh:
        fh.write("# Region file format: DS9 version 4.1\n")
        fh.write("fk5\n")
        fh.write(f"{include_region}\n")
        for reg in exclude_regions:
            fh.write(f"-{reg}\n")
