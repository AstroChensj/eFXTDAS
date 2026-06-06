"""RMF generation for ``fxtrspgen``."""

from __future__ import annotations

from pathlib import Path
import shutil

from astropy.io import fits

from fxtcaldb.query import ObservationMetadata
from fxtcaldb.response import resolve_rmf


def _normalize_rmf_headers(outfile: str, specfile: str) -> None:
    """Apply the same OGIP-style header normalization as ``fxtrmfgen``."""
    with fits.open(specfile) as spec_hdul:
        spec_header = spec_hdul[1].header
        detnam = spec_header["DETNAM"]
        filt = spec_header["FILTER"]

    with fits.open(outfile, mode="update") as hdul:
        hdul[1].header["EXTNAME"] = "MATRIX"
        hdul[1].header["TELESCOP"] = "EP"
        hdul[1].header["INSTRUME"] = "FXT"
        hdul[1].header["DETNAM"] = detnam
        hdul[1].header["FILTER"] = filt
        hdul[1].header["TLMIN4"] = 0
        hdul[1].header["TLMAX4"] = 1023
        hdul[1].header["HDUVERS1"] = "1.5.1"
        hdul[1].header["HDUCLAS3"] = "REDIST"
        if len(hdul) > 2:
            hdul[2].header["HDUVERS1"] = "1.5.1"
            hdul[2].header["DETNAM"] = detnam
            hdul[2].header["FILTER"] = filt
        for hdu in hdul:
            hdu.add_checksum()
            hdu.add_datasum()
        hdul.flush()


def generate_rmf(
    specfile: str,
    outfile: str,
    metadata: ObservationMetadata,
    clobber: bool = False,
) -> str:
    """Generate an RMF file chosen from CALDB.

    Parameters
    ----------
    specfile : str
        Input PHA path.
    outfile : str
        Output RMF path.
    metadata : ObservationMetadata
        Spectrum metadata used for CALDB lookup.
    clobber : bool, optional
        Overwrite the output if it exists.

    Returns
    -------
    str
        Written RMF path.
    """
    target = Path(outfile)
    if target.exists() and not clobber:
        raise FileExistsError(f"Output RMF already exists: {outfile}")
    rmf_path, _ = resolve_rmf(metadata)
    shutil.copyfile(rmf_path, outfile)
    _normalize_rmf_headers(outfile, specfile)
    return outfile
