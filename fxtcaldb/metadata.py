"""Metadata normalization for FXT calibration selection."""

from __future__ import annotations

from dataclasses import dataclass

from astropy.io import fits


def _split_isot(value: str) -> tuple[str, str]:
    """Split an ISO timestamp into date and time components."""
    return value.split("T", 1)


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


def normalize_detector(value: str) -> str:
    """Normalize detector identifiers to the short FXT form."""
    text = str(value).strip().upper()
    if text in {"A", "FXTA"}:
        return "A"
    if text in {"B", "FXTB"}:
        return "B"
    return text


def normalize_filter(value: str | int | None) -> str:
    """Normalize filter identifiers for CALDB matching."""
    if value is None:
        return "NONE"
    text = str(value).strip().upper()
    if not text or text == "-":
        return "NONE"
    if text in {"NONE", "OPEN", "THIN", "MEDIUM", "HOLE"}:
        return text
    try:
        return str(int(text))
    except ValueError:
        return text


@dataclass(frozen=True)
class SpectrumMetadata:
    """Spectrum header metadata needed for response and calibration lookup."""

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


def read_spectrum_metadata(specfile: str) -> SpectrumMetadata:
    """Read and normalize the metadata needed for CALDB selection."""
    with fits.open(specfile) as hdul:
        header = hdul[1].header
        start_date, start_time = _split_isot(str(header["DATE-OBS"]).strip())
        stop_date, stop_time = _split_isot(str(header["DATE-END"]).strip())
        detnam = str(header["DETNAM"]).strip()
        return SpectrumMetadata(
            telescope=str(header["TELESCOP"]).strip(),
            instrument=str(header["INSTRUME"]).strip(),
            detector_code=normalize_detector(detnam),
            detnam=detnam,
            filt=normalize_filter(header["FILTER"]),
            datamode=str(header["DATAMODE"]).strip().upper(),
            start_date=start_date,
            start_time=start_time,
            stop_date=stop_date,
            stop_time=stop_time,
            max_grade=_parse_max_grade(header),
        )
