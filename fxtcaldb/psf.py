"""PSF calibration readers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtcaldb.env import CaldbPaths
from fxtcaldb.metadata import normalize_detector


BETA_CACHE: dict[str, dict[str, np.ndarray]] = {}


def resolve_beta_psf_path(detector: str) -> Path:
    """Resolve the near-axis beta-PSF calibration file for one detector."""
    det = normalize_detector(detector)
    prefix = {"A": "fxta", "B": "fxtb"}.get(det, det.lower())
    caldb = Path(CaldbPaths.resolve().root)
    for directory in ("data/ep/fxt/cpf/psf", "data/ep/fxt/cpf/eef"):
        for suffix in (".fits", ".fits.gz"):
            path = caldb / directory / f"{prefix}_beta{suffix}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"No beta PSF file found for detector {detector}")


def load_beta_psf_table(detector: str) -> dict[str, np.ndarray]:
    """Load the beta-PSF parametrization table for one detector."""
    det = normalize_detector(detector)
    if det in BETA_CACHE:
        return BETA_CACHE[det]
    path = resolve_beta_psf_path(det)
    with fits.open(path) as hdul:
        data = hdul[1].data
        bandwidth = np.asarray(data["EMAX"] - data["EMIN"], dtype=np.float64)
        use = bandwidth < 4.0 * np.median(bandwidth)
        e_mid = 0.5 * np.asarray(data["EMIN"] + data["EMAX"], dtype=np.float64)[use]
        order = np.argsort(e_mid)
        table = {
            "e_mid": e_mid[order],
            "A1": np.asarray(data["A1"], dtype=np.float64)[use][order],
            "R1": np.asarray(data["R1"], dtype=np.float64)[use][order],
            "ALP1": np.asarray(data["ALP1"], dtype=np.float64)[use][order],
            "A2": np.asarray(data["A2"], dtype=np.float64)[use][order],
            "R2": np.asarray(data["R2"], dtype=np.float64)[use][order],
            "ALP2": np.asarray(data["ALP2"], dtype=np.float64)[use][order],
        }
    BETA_CACHE[det] = table
    return table
