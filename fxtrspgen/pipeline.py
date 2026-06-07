"""CLI and top-level orchestration for ``fxtrspgen``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

from astropy.io import fits

from fxtcaldb.query import read_observation_metadata
from fxtrspgen.arf import generate_arf
from fxtrspgen.rmf import generate_rmf
from fxtrspgen.utils.logger import build_cli_logger, emit


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
    psfprod: str | None = None,
    srcx: float | None = None,
    srcy: float | None = None,
    ra: float | None = None,
    dec: float | None = None,
    update_pha: bool = False,
    clobber: bool = False,
    logger: logging.Logger | None = None,
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
    psfprod : str | None, optional
        Existing observation PSF product to load instead of rebuilding one.
    srcx, srcy : float | None, optional
        Source position override in DS9/FITS image coordinates.
    ra, dec : float | None, optional
        Source position override in sky coordinates, degrees.
    update_pha : bool, optional
        Update the PHA ``ANCRFILE`` and ``RESPFILE`` headers in place.
    clobber : bool, optional
        Overwrite existing response files.
    logger : logging.Logger | None, optional
        Optional logger used for workflow messages.

    Returns
    -------
    dict[str, str]
        Written output paths.
    """
    t0 = time.perf_counter()
    arf_path = arf_out or _default_response_path(specfile, ".arf")
    rmf_path = rmf_out or _default_response_path(specfile, ".rmf")
    emit(logger, "info", "================================")
    emit(logger, "info", "**** Welcome to FXTRSPGEN! ****")
    emit(logger, "info", "================================")
    emit(logger, "info", f"  specfile = {specfile}")
    emit(logger, "info", f"  expfile = {expfile}")
    emit(logger, "info", f"  regionfile = {regionfile}")
    emit(logger, "info", f"  arf_out = {arf_path}")
    emit(logger, "info", f"  rmf_out = {rmf_path}")
    emit(logger, "info", f"  psfprod = {psfprod}")
    emit(logger, "info", f"  update_pha = {update_pha}")
    emit(logger, "info", f"  clobber = {clobber}")
    emit(logger, "info", "Reading observation metadata ...")
    stage_start = time.perf_counter()
    metadata = read_observation_metadata(specfile, preferred_ext=1)
    emit(logger, "info", f"Metadata read: {time.perf_counter() - stage_start:.2f}s")
    emit(logger, "info", "Generating RMF ...")
    stage_start = time.perf_counter()
    generate_rmf(specfile, rmf_path, metadata, clobber=clobber, logger=logger)
    emit(logger, "info", f"RMF stage runtime: {time.perf_counter() - stage_start:.2f}s")
    emit(logger, "info", "Generating ARF ...")
    stage_start = time.perf_counter()
    generate_arf(
        expfile,
        regionfile,
        arf_path,
        metadata,
        psfprod=psfprod,
        srcx=srcx,
        srcy=srcy,
        ra=ra,
        dec=dec,
        clobber=clobber,
        logger=logger,
    )
    emit(logger, "info", f"ARF stage runtime: {time.perf_counter() - stage_start:.2f}s")
    if update_pha:
        emit(logger, "info", "Updating PHA headers ...")
        stage_start = time.perf_counter()
        _update_pha_headers(specfile, arf_path, rmf_path)
        emit(logger, "info", f"PHA header update: {time.perf_counter() - stage_start:.2f}s")
    emit(logger, "info", f"FXTRSPGEN total runtime: {time.perf_counter() - t0:.2f}s")
    return {"arf_out": arf_path, "rmf_out": rmf_path}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxtrspgen`` command-line parser."""
    parser = argparse.ArgumentParser(description="Generate FXT ARF and RMF from an external DS9 source region.")
    parser.add_argument("specfile", help="Input source spectrum (PHA) file")
    parser.add_argument("expfile", help="Exposure map used for weighting and rasterization")
    parser.add_argument("regionfile", help="External DS9 source-region file")
    parser.add_argument("--arf-out", default=None, help="Output ARF path")
    parser.add_argument("--rmf-out", default=None, help="Output RMF path")
    parser.add_argument("--psfprod", default=None, help="Existing observation PSF product to reuse")
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
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level for CLI and output log file")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path; defaults to <arf-out>.log")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``fxtrspgen`` CLI."""
    args = build_parser().parse_args(argv)
    arf_path = args.arf_out or _default_response_path(args.specfile, ".arf")
    log_file = args.log_file if args.log_file is not None else Path(arf_path).with_suffix(".log")
    cli_logger = build_cli_logger("eFXTDAS.fxtrspgen", args.log_level, log_file)
    run_fxtrspgen(
        specfile=args.specfile,
        expfile=args.expfile,
        regionfile=args.regionfile,
        arf_out=args.arf_out,
        rmf_out=args.rmf_out,
        psfprod=args.psfprod,
        srcx=args.srcx,
        srcy=args.srcy,
        ra=args.ra,
        dec=args.dec,
        update_pha=args.update_pha,
        clobber=args.clobber,
        logger=cli_logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
