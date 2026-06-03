"""Regression tests for background-threshold optimization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtbkgoptrate.pipeline import run_bkgoptrate


def _write_lightcurve(path: Path, rates: list[float], fracexp: list[float] | None = None) -> Path:
    """Write a simple RATE light curve for optimizer tests.

    Parameters
    ----------
    path : Path
        Output FITS path.
    rates : list[float]
        Light-curve rate samples.
    fracexp : list[float] | None, optional
        Fractional exposure values per bin.

    Returns
    -------
    Path
        Written FITS path.
    """
    time = np.arange(len(rates), dtype=np.float64)
    fracexp_array = np.ones(len(rates), dtype=np.float64) if fracexp is None else np.asarray(fracexp, dtype=np.float64)
    rate_array = np.asarray(rates, dtype=np.float64)
    error_array = np.sqrt(np.clip(rate_array, 0.0, None))
    table_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=time),
            fits.Column(name="RATE", format="D", array=rate_array),
            fits.Column(name="ERROR", format="D", array=error_array),
            fits.Column(name="FRACEXP", format="D", array=fracexp_array),
        ],
        name="RATE",
    )
    table_hdu.header["TIMEDEL"] = 1.0
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(path, overwrite=True)
    return path


def test_snr_method_prefers_lowest_threshold_for_monotonic_background(tmp_path: Path) -> None:
    """The legacy S/N optimizer should preserve its current threshold behavior."""
    lc_path = _write_lightcurve(tmp_path / "snr.lc", [1.0, 1.0, 1.0, 10.0, 10.0])
    result = run_bkgoptrate(str(lc_path), method="snr")
    assert result["best_threshold"] == 1.0
    assert result["status"] == "optimal"
    assert result["method"] == "snr"


def test_robust_iqr_method_avoids_minimum_rate_collapse(tmp_path: Path) -> None:
    """The robust method should tolerate low-count outliers without pegging low."""
    lc_path = _write_lightcurve(tmp_path / "robust.lc", [1.0, 2.0, 2.0, 3.0, 12.0])
    result = run_bkgoptrate(str(lc_path), method="robust_iqr")
    assert result["best_threshold"] > 1.0
    assert result["best_threshold"] < 12.0
    assert result["status"] == "optimal"
    assert result["method"] == "robust_iqr"


def test_robust_iqr_method_returns_no_cut_needed_for_zero_iqr(tmp_path: Path) -> None:
    """A flat light curve should keep all bins under the robust method."""
    lc_path = _write_lightcurve(tmp_path / "flat.lc", [2.0, 2.0, 2.0, 2.0])
    result = run_bkgoptrate(str(lc_path), method="robust_iqr")
    assert result["best_threshold"] == 2.0
    assert result["status"] == "no_cut_needed"
    assert np.all(result["kept_mask"])


def test_robust_iqr_method_honors_min_time_ratio(tmp_path: Path) -> None:
    """The robust method should lift the threshold when too few bins survive."""
    lc_path = _write_lightcurve(tmp_path / "min_ratio.lc", [1.0, 5.0, 5.0, 100.0])
    result = run_bkgoptrate(str(lc_path), method="robust_iqr", min_time_ratio=0.8)
    assert result["best_threshold"] == 100.0
    assert result["kept_fraction"] >= 0.8


def test_diagnostic_headers_include_optimizer_method(tmp_path: Path) -> None:
    """Diagnostic FITS headers should persist the chosen optimizer method."""
    lc_path = _write_lightcurve(tmp_path / "diag.lc", [1.0, 2.0, 2.0, 6.0])
    diag_path = tmp_path / "diag_out.fits"
    run_bkgoptrate(str(lc_path), method="robust_iqr", diagnostic_outfile=str(diag_path))
    with fits.open(diag_path) as hdul:
        hdr = hdul["BKGOPT"].header
        assert hdr["OPTMETH"] == "robust_iqr"
        assert "BGOPTCUT" in hdr
        assert "FRACTLFT" in hdr
        assert "OPTSTAT" in hdr
