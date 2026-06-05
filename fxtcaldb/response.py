"""Response-product lookup helpers."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from fxtcaldb.metadata import SpectrumMetadata
from fxtcaldb.query import find_calibration_file


def _lookup_file(metadata: SpectrumMetadata, codename: str, filt: str, expr: str | None) -> tuple[str, int]:
    """Resolve one CALDB file path and extension number."""
    return find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=metadata.detector_code,
        filt=filt,
        codename=codename,
        start_date=metadata.start_date,
        start_time=metadata.start_time,
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time,
        expr=expr or "",
    )


def resolve_base_arf(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB base ARF file."""
    expr = f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{metadata.max_grade})"
    return _lookup_file(metadata, "SPECRESP", metadata.filt, expr)


def resolve_rmf(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB RMF file with a controlled grade-less fallback."""
    expr = f"DATAMODE({metadata.datamode}) .AND. GRADE(G0:{metadata.max_grade})"
    try:
        return _lookup_file(metadata, "MATRIX", "NONE", expr)
    except RuntimeError as exc:
        try:
            return _lookup_file(metadata, "MATRIX", "NONE", f"DATAMODE({metadata.datamode})")
        except RuntimeError:
            raise RuntimeError(
                "RMF lookup failed for "
                f"telescope={metadata.telescope}, instrument={metadata.instrument}, "
                f"detector={metadata.detector_code}, filter=NONE, datamode={metadata.datamode}, "
                f"grade=G0:{metadata.max_grade}, start={metadata.start_date}T{metadata.start_time}, "
                f"stop={metadata.stop_date}T{metadata.stop_time}"
            ) from exc


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
