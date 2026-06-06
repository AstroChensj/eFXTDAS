"""Response-product lookup helpers for EP-FXT."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from fxtcaldb.query import ObservationMetadata, find_calibration_file, require_caldb_metadata


FILTER_CODE_MAP = {
    "OPEN": "0",
    "THIN": "1",
    "MEDIUM": "2",
    "HOLE": "3",
}
SUPPORTED_RESPONSE_GRADES = {0, 4, 12}


def _canonicalize_response_detector(metadata: ObservationMetadata) -> str:
    """Map metadata detector identity to response-family ``DETNAM``.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for response lookup.

    Returns
    -------
    str
        Response-family detector code, ``A`` or ``B``.
    """
    if metadata.detector_code is None:
        raise ValueError("Response lookup requires detector metadata.")
    return str(metadata.detector_code)


def _canonicalize_specresp_filter(metadata: ObservationMetadata) -> str:
    """Map metadata filter identity to ``SPECRESP`` filter conventions.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for ARF lookup.

    Returns
    -------
    str
        ``SPECRESP`` filter code in ``0/1/2/3`` form.
    """
    if metadata.filt is None:
        raise ValueError("SPECRESP lookup requires FILTER metadata.")
    filt = str(metadata.filt).strip().upper()
    if filt in FILTER_CODE_MAP:
        return FILTER_CODE_MAP[filt]
    if filt.isdigit():
        value = str(int(filt))
        if value in {"0", "1", "2", "3"}:
            return value
    raise ValueError(f"Unsupported SPECRESP filter value: {metadata.filt!r}")


def _require_supported_response_grade(metadata: ObservationMetadata) -> int:
    """Validate that the requested grade is indexed by the EP-FXT responses.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for response lookup.

    Returns
    -------
    int
        Supported maximum grade.
    """
    if metadata.max_grade is None:
        raise ValueError("Response lookup requires a GRADE selection.")
    grade = int(metadata.max_grade)
    if grade not in SUPPORTED_RESPONSE_GRADES:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_RESPONSE_GRADES))
        raise ValueError(
            f"Unsupported EP-FXT response grade G0:{grade}. "
            f"Supported indexed grades are: {supported}."
        )
    return grade


def resolve_base_arf(metadata: ObservationMetadata) -> tuple[str, int]:
    """Resolve the CALDB base ARF file with a strict indexed request.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for ARF lookup.

    Returns
    -------
    tuple[str, int]
        Resolved ARF file path and extension number.
    """
    require_caldb_metadata(metadata)
    grade = _require_supported_response_grade(metadata)
    return find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=_canonicalize_response_detector(metadata),
        filt=_canonicalize_specresp_filter(metadata),
        codename="SPECRESP",
        start_date=metadata.start_date,
        start_time=metadata.start_time or "00:00:00",
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time or "00:00:00",
        expr=f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{grade})",
    )


def resolve_rmf(metadata: ObservationMetadata) -> tuple[str, int]:
    """Resolve the CALDB RMF file with a strict indexed request.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for RMF lookup.

    Returns
    -------
    tuple[str, int]
        Resolved RMF file path and extension number.
    """
    require_caldb_metadata(metadata)
    grade = _require_supported_response_grade(metadata)
    return find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=_canonicalize_response_detector(metadata),
        filt="NONE",
        codename="MATRIX",
        start_date=metadata.start_date,
        start_time=metadata.start_time or "00:00:00",
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time or "00:00:00",
        expr=f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{grade})",
    )


def read_base_arf_table(metadata: ObservationMetadata) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the uncorrected ARF on its native energy grid.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for ARF lookup.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(ENERG_LO, ENERG_HI, SPECRESP)`` arrays.
    """
    filepath, extno = resolve_base_arf(metadata)
    with fits.open(filepath) as hdul:
        data = hdul[extno].data
        return (
            np.asarray(data["ENERG_LO"], dtype=np.float64),
            np.asarray(data["ENERG_HI"], dtype=np.float64),
            np.asarray(data["SPECRESP"], dtype=np.float64),
        )
