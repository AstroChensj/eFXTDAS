"""TELDEF lookup and readers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from fxtcaldb.metadata import SpectrumMetadata
from fxtcaldb.query import find_calibration_file


@dataclass(frozen=True)
class TeldefInfo:
    """TELDEF quantities needed by current PSF/response tools."""

    alignment_matrix: np.ndarray
    focal_length: float
    pixel_size: float
    optaxis_x: float
    optaxis_y: float


def resolve_teldef(metadata: SpectrumMetadata) -> tuple[str, int]:
    """Resolve the CALDB TELDEF file."""
    return find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=metadata.detector_code,
        filt="NONE",
        codename="TELDEF",
        start_date=metadata.start_date,
        start_time=metadata.start_time,
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time,
        expr="",
    )


def read_teldef_info(filepath: str) -> TeldefInfo:
    """Read the alignment and optical-axis quantities from one TELDEF file."""
    header = fits.getheader(filepath)
    return TeldefInfo(
        alignment_matrix=np.array(
            [
                [header["ALIGNM11"], header["ALIGNM12"], header["ALIGNM13"]],
                [header["ALIGNM21"], header["ALIGNM22"], header["ALIGNM23"]],
                [header["ALIGNM31"], header["ALIGNM32"], header["ALIGNM33"]],
            ],
            dtype=np.float64,
        ),
        focal_length=float(header["FOCALLEN"]),
        pixel_size=float(header["DET_XSCL"]),
        optaxis_x=float(header["OPTAXISX"]),
        optaxis_y=float(header["OPTAXISY"]),
    )
