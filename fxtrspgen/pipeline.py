"""CLI and top-level orchestration for ``fxtrspgen``."""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits

from fxtcaldb.query import read_observation_metadata
from fxtrspgen.arf import generate_arf
from fxtrspgen.rmf import generate_rmf


def _default_response_path(specfile: str, suffix: str) -> str:
    """Build the default response path next to the PHA file."""
    path = Path(specfile)
    return str(path.with_suffix(suffix))


def _update_pha_headers(specfile: str, arf_out: str, rmf_out: str) -> None:
    """Write ``ANCRFILE`` and ``RESPFILE`` into the spectrum extension."""
    with fits.open(specfile, mode="update") as hdul:
        hdul[1].header["ANCRFILE"] = arf_out
        hdul[1].header["RESPFILE"] = rmf_out
        for hdu in hdul:
            hdu.add_checksum()
            hdu.add_datasum()
        hdul.flush()


def run_fxtrspgen(
    specfile: str,
    expfile: str,
    regionfile: str,
    arf_out: str | None = None,
    rmf_out: str | None = None,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    update_pha: bool = False,
    clobber: bool = False,
) -> dict[str, str]:
    """Generate ARF and RMF products for one FXT spectrum.

    Parameters
    ----------
    specfile : str
        Input PHA file.
    expfile : str
        Exposure map used for region rasterization and vignetting weighting.
    regionfile : str
        External DS9 source-region file.
    arf_out : str | None, optional
        Output ARF path. Defaults to ``specfile`` with ``.arf`` suffix.
    rmf_out : str | None, optional
        Output RMF path. Defaults to ``specfile`` with ``.rmf`` suffix.
    srcx, srcy : float | None, optional
        Source position override in DS9/FITS image coordinates.
    ra, dec : float | None, optional
        Source position override in sky coordinates, degrees.
    update_pha : bool, optional
        Update the PHA ``ANCRFILE`` and ``RESPFILE`` headers in place.
    clobber : bool, optional
        Overwrite existing response files.

    Returns
    -------
    dict[str, str]
        Written output paths.
    """
    arf_path = arf_out or _default_response_path(specfile, ".arf")
    rmf_path = rmf_out or _default_response_path(specfile, ".rmf")
    metadata = read_observation_metadata(specfile, preferred_ext=1)
    generate_rmf(specfile, rmf_path, metadata, clobber=clobber)
    generate_arf(
        expfile,
        regionfile,
        arf_path,
        metadata,
        srcx=srcx,
        srcy=srcy,
        ra=ra,
        dec=dec,
        clobber=clobber,
    )
    if update_pha:
        _update_pha_headers(specfile, arf_path, rmf_path)
    return {"arf_out": arf_path, "rmf_out": rmf_path}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxtrspgen`` command-line parser."""
    parser = argparse.ArgumentParser(description="Generate FXT ARF and RMF from an external DS9 source region.")
    parser.add_argument("specfile", help="Input source spectrum (PHA) file")
    parser.add_argument("expfile", help="Exposure map used for weighting and rasterization")
    parser.add_argument("regionfile", help="External DS9 source-region file")
    parser.add_argument("--arf-out", default=None, help="Output ARF path")
    parser.add_argument("--rmf-out", default=None, help="Output RMF path")
    parser.add_argument("--srcx", type=float, default=None, help="Source X override in DS9/FITS image coordinates")
    parser.add_argument("--srcy", type=float, default=None, help="Source Y override in DS9/FITS image coordinates")
    parser.add_argument("--ra", type=float, default=None, help="Source RA override in degrees")
    parser.add_argument("--dec", type=float, default=None, help="Source Dec override in degrees")
    parser.add_argument(
        "--update-pha",
        action="store_true",
        help="Write RESPFILE and ANCRFILE into the input PHA",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Overwrite existing ARF/RMF outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``fxtrspgen`` CLI."""
    args = build_parser().parse_args(argv)
    run_fxtrspgen(
        specfile=args.specfile,
        expfile=args.expfile,
        regionfile=args.regionfile,
        arf_out=args.arf_out,
        rmf_out=args.rmf_out,
        srcx=args.srcx,
        srcy=args.srcy,
        ra=args.ra,
        dec=args.dec,
        update_pha=args.update_pha,
        clobber=args.clobber,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
