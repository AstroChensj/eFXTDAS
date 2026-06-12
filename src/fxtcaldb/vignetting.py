"""Vignetting calibration helpers for EP-FXT."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from fxtcaldb.query import ObservationMetadata, find_calibration_file, require_caldb_metadata


def _canonicalize_vignet_detector(metadata: ObservationMetadata) -> str:
    """Map metadata detector identity to ``VIGNET`` detector conventions.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for lookup.

    Returns
    -------
    str
        ``VIGNET`` detector code, ``A`` or ``B``.
    """
    if metadata.detector_code is None:
        raise ValueError("VIGNET lookup requires detector metadata.")
    return str(metadata.detector_code)


def _canonicalize_vignet_filter(metadata: ObservationMetadata) -> str:
    """Map metadata filter identity to the EP-FXT vignetting family.

    Notes
    -----
    ``refs/caldb.indx`` shows that ``CAL_CNAM=VIGNET`` only distinguishes two
    calibration families:

    - normal filters: encoded as ``1`` or ``01``
    - hole filter: encoded as ``3`` or ``03``

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for lookup.

    Returns
    -------
    str
        Canonical ``VIGNET`` filter code, either ``1`` or ``3``.
    """
    if metadata.filt is None:
        raise ValueError("VIGNET lookup requires FILTER metadata.")
    filt = str(metadata.filt).strip().upper()
    if filt in {"OPEN", "THIN", "MEDIUM"}:
        return "1"
    if filt == "HOLE":
        return "3"
    if filt.isdigit():
        value = str(int(filt))
        if value in {"0", "1", "2"}:
            return "1"
        if value == "3":
            return "3"
    raise ValueError(f"Unsupported VIGNET filter value: {metadata.filt!r}")


def resolve_vignetting_table(metadata: ObservationMetadata) -> np.recarray:
    """Resolve and read the CALDB vignetting table.

    Parameters
    ----------
    metadata : ObservationMetadata
        Observation metadata used for lookup.

    Returns
    -------
    np.recarray
        Vignetting table payload from the selected FITS extension.
    """
    require_caldb_metadata(metadata)
    filepath, extno = find_calibration_file(
        telescope=metadata.telescope,
        instrument=metadata.instrument,
        detname=_canonicalize_vignet_detector(metadata),
        filt=_canonicalize_vignet_filter(metadata),
        codename="VIGNET",
        start_date=metadata.start_date,
        start_time=metadata.start_time or "00:00:00",
        stop_date=metadata.stop_date,
        stop_time=metadata.stop_time or "00:00:00",
        expr="",
    )
    with fits.open(filepath) as hdul:
        return hdul[extno].data.copy()
