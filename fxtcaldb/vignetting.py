"""Vignetting calibration helpers."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from fxtcaldb.metadata import SpectrumMetadata
from fxtcaldb.query import find_calibration_file


def resolve_vignetting_table(metadata: SpectrumMetadata) -> np.recarray:
    """Resolve and read the CALDB vignetting table."""
    filt = "1" if metadata.filt.isdigit() and int(metadata.filt) < 3 else metadata.filt
    filepath, extno = find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=metadata.detector_code,
        filt=filt,
        codename="VIGNET",
        start_date=metadata.start_date,
        start_time=metadata.start_time,
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time,
        expr="",
    )
    with fits.open(filepath) as hdul:
        return hdul[extno].data.copy()
