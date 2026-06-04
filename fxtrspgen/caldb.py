"""CALDB lookup helpers for ``fxtrspgen``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtrspgen.runtime import ensure_fxtdas_py_path

ensure_fxtdas_py_path()

from py_getcalf import getcalf  # type: ignore  # noqa: E402


@dataclass(frozen=True)
class SpectrumMetadata:
    """Spectrum header metadata used for CALDB selection."""

    telescope: str
    instrument: str
    detector_code: str
    detnam: str
    filt: str
    datamode: str
    start_date: str
    start_time: str
    stop_date: str
    stop_time: str
    max_grade: int


def _split_isot(value: str) -> tuple[str, str]:
    """Split an ISO timestamp into date and time parts."""
    date_part, time_part = value.split("T", 1)
    return date_part, time_part


def _parse_max_grade(header: fits.Header) -> int:
    """Recover the highest selected grade from ``DSTYP*``/``DSVAL*``."""
    for key in header:
        if not key.startswith("DSTYP"):
            continue
        if str(header[key]).strip().upper() != "GRADE":
            continue
        suffix = key[5:]
        dsval = str(header.get(f"DSVAL{suffix}", "")).strip()
        if ":" in dsval:
            try:
                return int(dsval.split(":")[-1].strip())
            except ValueError:
                continue
    return 12


def read_spectrum_metadata(specfile: str) -> SpectrumMetadata:
    """Read the spectrum metadata needed for response lookup.

    Parameters
    ----------
    specfile : str
        Input PHA path.

    Returns
    -------
    SpectrumMetadata
        Parsed metadata bundle.
    """
    with fits.open(specfile) as hdul:
        header = hdul[1].header
        filt_raw = str(header["FILTER"]).strip()
        start_date, start_time = _split_isot(str(header["DATE-OBS"]).strip())
        stop_date, stop_time = _split_isot(str(header["DATE-END"]).strip())
        detnam = str(header["DETNAM"]).strip()
        return SpectrumMetadata(
            telescope=str(header["TELESCOP"]).strip(),
            instrument=str(header["INSTRUME"]).strip(),
            detector_code=detnam[-1],
            detnam=detnam,
            filt=filt_raw[-1],
            datamode=str(header["DATAMODE"]).strip(),
            start_date=start_date,
            start_time=start_time,
            stop_date=stop_date,
            stop_time=stop_time,
            max_grade=_parse_max_grade(header),
        )


def _lookup_file(
    metadata: SpectrumMetadata,
    codename: str,
    filt: str,
    expr: str | None,
) -> tuple[str, int]:
    """Resolve one CALDB file path and extension number."""
    filepath, extno = getcalf(
        metadata.telescope,
        metadata.instrument,
        metadata.detector_code,
        filt,
        codename,
        metadata.start_date,
        metadata.start_time,
        metadata.stop_date,
        metadata.stop_time,
        expr,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    return str(Path(filepath).resolve()), int(extno)


def resolve_base_arf(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB base ARF file."""
    expr = f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{metadata.max_grade})"
    return _lookup_file(metadata, "SPECRESP", metadata.filt, expr)


def resolve_rmf(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB RMF file."""
    expr = f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{metadata.max_grade})"
    return _lookup_file(metadata, "MATRIX", "None", expr)


def resolve_teldef(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB TELDEF file."""
    return _lookup_file(metadata, "TELDEF", "None", None)


def resolve_vignetting_table(metadata: SpectrumMetadata) -> np.recarray:
    """Resolve and read the CALDB vignetting table."""
    filt = "1" if int(metadata.filt) < 3 else metadata.filt
    filepath, extno = _lookup_file(metadata, "VIGNET", filt, "NONE")
    with fits.open(filepath) as hdul:
        return hdul[extno].data.copy()


def read_base_arf_table(metadata: SpectrumMetadata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the uncorrected ARF on its native energy grid."""
    filepath, extno = resolve_base_arf(metadata)
    with fits.open(filepath) as hdul:
        data = hdul[extno].data
        return (
            np.asarray(data["ENERG_LO"], dtype=np.float64),
            np.asarray(data["ENERG_HI"], dtype=np.float64),
            np.asarray(data["SPECRESP"], dtype=np.float64),
        )
