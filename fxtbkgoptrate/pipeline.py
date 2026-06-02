#!/usr/bin/env python3
"""Optimize a background light-curve threshold and derive flare GTIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtbkgoptrate.utils.logger import build_cli_logger, emit


def _resolve_y_column(columns: fits.ColDefs, ycol: str | None, tsstyle: str) -> tuple[str, str]:
    """Resolve the light-curve value column and its style."""
    available = {col.name.upper(): col.name for col in columns}
    if ycol is not None:
        resolved = available.get(ycol.upper())
        if resolved is None:
            raise KeyError(f"Column {ycol!r} was not found in the light curve.")
        if tsstyle == "auto":
            style = "rate" if resolved.upper() == "RATE" else "count"
        else:
            style = tsstyle
        return resolved, style
    if "RATE" in available:
        return available["RATE"], "rate"
    if "COUNT" in available:
        return available["COUNT"], "count"
    if "COUNTS" in available:
        return available["COUNTS"], "count"
    raise KeyError("No RATE/COUNT/COUNTS column was found in the light curve.")


def load_lightcurve(
    infile: str,
    *,
    xcol: str = "TIME",
    ycol: str | None = None,
    fracexpcol: str = "FRACEXP",
    tsstyle: str = "auto",
    fracexpstyle: str = "auto",
    fracexplower: float = 0.0,
    starttime: float | None = None,
    endtime: float | None = None,
    lowercutoffcount: float | None = None,
) -> dict:
    """Load a FITS light curve into analysis-ready arrays.

    Parameters
    ----------
    infile : str
        Input light-curve FITS path.
    xcol : str, optional
        Time column name.
    ycol : str | None, optional
        Rate/count column name. When omitted, RATE is preferred.
    fracexpcol : str, optional
        Fractional exposure column name.
    tsstyle : {"auto", "rate", "count"}, optional
        How to interpret the Y column.
    fracexpstyle : {"auto", "calc", "threshold", "none"}, optional
        Fractional-exposure handling policy.
    fracexplower : float, optional
        Lower FRACEXP threshold when ``fracexpstyle="threshold"``.
    starttime : float | None, optional
        Optional absolute lower time bound.
    endtime : float | None, optional
        Optional absolute upper time bound.
    lowercutoffcount : float | None, optional
        Optional lower cutoff applied to the raw Y values.

    Returns
    -------
    dict
        Structured arrays and metadata for optimization.
    """
    with fits.open(infile) as hdul:
        hdu = hdul["RATE"] if "RATE" in hdul else hdul[1]
        data = hdu.data
        header = hdu.header.copy()
        primary_header = hdul[0].header.copy()
    columns = hdu.columns
    ycol_resolved, resolved_style = _resolve_y_column(columns, ycol, tsstyle)
    time = np.asarray(data[xcol], dtype=np.float64)
    y = np.asarray(data[ycol_resolved], dtype=np.float64)
    timedel = float(header.get("TIMEDEL") or primary_header.get("TIMEDEL") or 0.0)
    if timedel <= 0.0 and len(time) > 1:
        timedel = float(np.nanmedian(np.diff(time)))
    if timedel <= 0.0:
        timedel = 1.0
    timezero = float(header.get("TIMEZERO") or primary_header.get("TIMEZERO") or 0.0)
    timepixr = float(header.get("TIMEPIXR") or primary_header.get("TIMEPIXR") or 0.5)
    time_center = time + timezero
    time_start = time_center - timepixr * timedel
    time_stop = time_start + timedel

    fracexp_present = fracexpcol in data.names
    if fracexpstyle == "auto":
        fracexpstyle = "calc" if fracexp_present else "none"
    if fracexpstyle == "none" or not fracexp_present:
        fracexp = np.ones_like(time_center, dtype=np.float64)
    else:
        fracexp = np.asarray(data[fracexpcol], dtype=np.float64)
    if fracexpstyle == "threshold":
        fracexp_mask = np.isfinite(fracexp) & (fracexp >= fracexplower)
    else:
        fracexp_mask = np.isfinite(fracexp) & (fracexp > 0.0)
    exposure = timedel * np.clip(fracexp, 0.0, None)

    if resolved_style == "count":
        raw_values = y
        rate = np.divide(y, exposure, out=np.full_like(y, np.nan), where=exposure > 0.0)
    else:
        raw_values = y
        rate = y

    valid = np.isfinite(time_center) & np.isfinite(rate) & np.isfinite(raw_values)
    valid &= rate >= 0.0
    valid &= exposure > 0.0
    valid &= fracexp_mask
    if starttime is not None:
        valid &= time_stop > starttime
    if endtime is not None:
        valid &= time_start < endtime
    if lowercutoffcount is not None:
        valid &= raw_values > lowercutoffcount

    return {
        "infile": infile,
        "header": header,
        "primary_header": primary_header,
        "time_center": time_center,
        "time_start": time_start,
        "time_stop": time_stop,
        "rate": rate,
        "raw_values": raw_values,
        "fracexp": fracexp,
        "exposure": exposure,
        "valid_mask": valid,
        "timedel": timedel,
        "timezero": timezero,
        "timepixr": timepixr,
        "ycol": ycol_resolved,
        "tsstyle": resolved_style,
    }


def find_optimal_rate(
    lc_data: dict,
    *,
    min_time_ratio: float = 0.05,
) -> dict:
    """Find the optimum background threshold for one light curve.

    Parameters
    ----------
    lc_data : dict
        Structured light-curve arrays from :func:`load_lightcurve`.
    min_time_ratio : float, optional
        Minimum retained exposure fraction allowed for candidate thresholds.

    Returns
    -------
    dict
        Optimization result including the best threshold, kept-bin mask, and
        diagnostic trial arrays.
    """
    valid_mask = np.asarray(lc_data["valid_mask"], dtype=bool)
    rate = np.asarray(lc_data["rate"], dtype=np.float64)
    exposure = np.asarray(lc_data["exposure"], dtype=np.float64)
    total_exposure = float(np.sum(exposure[valid_mask]))
    if total_exposure <= 0.0 or not np.any(valid_mask):
        return {
            "status": "no_valid_bin",
            "best_threshold": np.nan,
            "kept_mask": valid_mask.copy(),
            "kept_fraction": 0.0,
            "score": np.nan,
            "trials": [],
        }

    candidate_thresholds = np.unique(rate[valid_mask])
    trials = []
    best_trial = None
    for threshold in candidate_thresholds:
        kept_mask = valid_mask & (rate <= threshold)
        kept_exposure = float(np.sum(exposure[kept_mask]))
        kept_fraction = kept_exposure / total_exposure if total_exposure > 0.0 else 0.0
        if kept_fraction < min_time_ratio:
            continue
        background_sum = float(np.sum(rate[kept_mask] * exposure[kept_mask]))
        score = kept_exposure / np.sqrt(background_sum) if background_sum > 0.0 else np.nan
        if np.isfinite(score):
            max_rate_kept = float(np.max(rate[kept_mask])) if np.any(kept_mask) else np.nan
            trial = {
                "threshold": float(threshold),
                "max_rate_kept": max_rate_kept,
                "score": float(score),
                "n_bin": int(np.sum(kept_mask)),
                "kept_fraction": kept_fraction,
                "kept_mask": kept_mask.copy(),
            }
            trials.append(trial)
            if best_trial is None or trial["score"] > best_trial["score"]:
                best_trial = trial

    if best_trial is None:
        return {
            "status": "min_time_ratio_not_met",
            "best_threshold": np.nan,
            "kept_mask": valid_mask.copy(),
            "kept_fraction": 1.0,
            "score": np.nan,
            "trials": trials,
        }

    status = "optimal"
    if np.all(best_trial["kept_mask"] == valid_mask):
        status = "no_cut_needed"
    return {
        "status": status,
        "best_threshold": best_trial["threshold"],
        "kept_mask": best_trial["kept_mask"],
        "kept_fraction": best_trial["kept_fraction"],
        "score": best_trial["score"],
        "trials": trials,
    }


def write_diagnostic_table(
    outfile: str,
    lc_data: dict,
    result: dict,
) -> str:
    """Write the threshold-search diagnostic table to FITS.

    Parameters
    ----------
    outfile : str
        Output FITS path.
    lc_data : dict
        Structured light-curve arrays from :func:`load_lightcurve`.
    result : dict
        Optimization result from :func:`find_optimal_rate`.

    Returns
    -------
    str
        Written diagnostic FITS path.
    """
    trials = result["trials"]
    if trials:
        threshold = np.asarray([trial["threshold"] for trial in trials], dtype=np.float64)
        max_rate = np.asarray([trial["max_rate_kept"] for trial in trials], dtype=np.float64)
        score = np.asarray([trial["score"] for trial in trials], dtype=np.float64)
        n_bin = np.asarray([trial["n_bin"] for trial in trials], dtype=np.int32)
        kept_fraction = np.asarray([trial["kept_fraction"] for trial in trials], dtype=np.float64)
    else:
        threshold = np.asarray([], dtype=np.float64)
        max_rate = np.asarray([], dtype=np.float64)
        score = np.asarray([], dtype=np.float64)
        n_bin = np.asarray([], dtype=np.int32)
        kept_fraction = np.asarray([], dtype=np.float64)
    table_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="BKGRATECUT", format="D", array=threshold),
            fits.Column(name="BKGRATE", format="D", array=max_rate),
            fits.Column(name="SN_RATIO", format="D", array=score),
            fits.Column(name="N_BIN", format="J", array=n_bin),
            fits.Column(name="FRACTLFT", format="D", array=kept_fraction),
        ],
        name="BKGOPT",
    )
    table_hdu.header["BGOPTCUT"] = (float(result["best_threshold"]) if np.isfinite(result["best_threshold"]) else np.nan, "Optimum background threshold")
    table_hdu.header["FRACTLFT"] = (float(result["kept_fraction"]), "Retained exposure fraction")
    table_hdu.header["OPTSTAT"] = (str(result["status"]), "Threshold optimization status")
    primary_hdu = fits.PrimaryHDU(header=lc_data["primary_header"])
    fits.HDUList([primary_hdu, table_hdu]).writeto(outfile, overwrite=True)
    return outfile


def build_gti_from_mask(
    lc_data: dict,
    kept_mask: np.ndarray,
    outfile: str,
) -> str:
    """Build a GTI FITS file from an accepted-bin mask.

    Parameters
    ----------
    lc_data : dict
        Structured light-curve arrays from :func:`load_lightcurve`.
    kept_mask : numpy.ndarray
        Boolean mask selecting retained bins.
    outfile : str
        Output GTI FITS path.

    Returns
    -------
    str
        Written GTI FITS path.
    """
    kept_mask = np.asarray(kept_mask, dtype=bool) & np.asarray(lc_data["valid_mask"], dtype=bool)
    starts = np.asarray(lc_data["time_start"][kept_mask], dtype=np.float64)
    stops = np.asarray(lc_data["time_stop"][kept_mask], dtype=np.float64)
    merged_start: list[float] = []
    merged_stop: list[float] = []
    if len(starts):
        current_start = float(starts[0])
        current_stop = float(stops[0])
        epsilon = max(float(lc_data["timedel"]) * 1e-6, 1e-6)
        for start, stop in zip(starts[1:], stops[1:]):
            if float(start) <= current_stop + epsilon:
                current_stop = max(current_stop, float(stop))
            else:
                merged_start.append(current_start)
                merged_stop.append(current_stop)
                current_start = float(start)
                current_stop = float(stop)
        merged_start.append(current_start)
        merged_stop.append(current_stop)
    gti_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="1D", unit="s", array=np.asarray(merged_start, dtype=np.float64)),
            fits.Column(name="STOP", format="1D", unit="s", array=np.asarray(merged_stop, dtype=np.float64)),
        ],
        name="GTI",
    )
    gti_hdu.header["MJDREFI"] = lc_data["header"].get("MJDREFI", lc_data["primary_header"].get("MJDREFI"))
    gti_hdu.header["MJDREFF"] = lc_data["header"].get("MJDREFF", lc_data["primary_header"].get("MJDREFF"))
    fits.HDUList([fits.PrimaryHDU(header=lc_data["primary_header"]), gti_hdu]).writeto(outfile, overwrite=True)
    return outfile


def intersect_gtis(
    base_gti_path: str,
    flare_gti_path: str,
    outfile: str,
) -> str:
    """Intersect two GTI FITS files and write the result.

    Parameters
    ----------
    base_gti_path : str
        Base GTI FITS path.
    flare_gti_path : str
        Flare-screening GTI FITS path.
    outfile : str
        Output merged GTI FITS path.

    Returns
    -------
    str
        Written merged GTI FITS path.
    """
    with fits.open(base_gti_path) as base_hdul, fits.open(flare_gti_path) as flare_hdul:
        base_data = base_hdul["GTI"].data if "GTI" in base_hdul else base_hdul[1].data
        flare_data = flare_hdul["GTI"].data if "GTI" in flare_hdul else flare_hdul[1].data
        base_primary = base_hdul[0].header.copy()
        base_header = (base_hdul["GTI"].header if "GTI" in base_hdul else base_hdul[1].header).copy()
    starts: list[float] = []
    stops: list[float] = []
    i = 0
    j = 0
    while i < len(base_data) and j < len(flare_data):
        start = max(float(base_data["START"][i]), float(flare_data["START"][j]))
        stop = min(float(base_data["STOP"][i]), float(flare_data["STOP"][j]))
        if start < stop:
            starts.append(start)
            stops.append(stop)
        if float(base_data["STOP"][i]) <= float(flare_data["STOP"][j]):
            i += 1
        else:
            j += 1
    gti_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="1D", unit="s", array=np.asarray(starts, dtype=np.float64)),
            fits.Column(name="STOP", format="1D", unit="s", array=np.asarray(stops, dtype=np.float64)),
        ],
        name="GTI",
    )
    for key in ("MJDREFI", "MJDREFF", "TIMEZERO", "TIMESYS", "TIMEREF", "TELESCOP", "INSTRUME"):
        if key in base_header:
            gti_hdu.header[key] = base_header[key]
    fits.HDUList([fits.PrimaryHDU(header=base_primary), gti_hdu]).writeto(outfile, overwrite=True)
    return outfile


def run_bkgoptrate(
    infile: str,
    *,
    xcol: str = "TIME",
    ycol: str | None = None,
    fracexpcol: str = "FRACEXP",
    tsstyle: str = "auto",
    fracexpstyle: str = "auto",
    fracexplower: float = 0.0,
    starttime: float | None = None,
    endtime: float | None = None,
    lowercutoffcount: float | None = None,
    min_time_ratio: float = 0.05,
    diagnostic_outfile: str | None = None,
    flare_gti_outfile: str | None = None,
    base_gti_path: str | None = None,
    screened_gti_outfile: str | None = None,
    logger=None,
) -> dict:
    """Run the full background-threshold optimization workflow.

    Parameters
    ----------
    infile : str
        Input light-curve FITS path.
    xcol, ycol, fracexpcol, tsstyle, fracexpstyle, fracexplower, starttime, endtime, lowercutoffcount
        Passed through to :func:`load_lightcurve`.
    min_time_ratio : float, optional
        Minimum retained exposure fraction.
    diagnostic_outfile : str | None, optional
        Optional diagnostic FITS output.
    flare_gti_outfile : str | None, optional
        Optional flare-only GTI output.
    base_gti_path : str | None, optional
        Optional base GTI path used to build a screened GTI.
    screened_gti_outfile : str | None, optional
        Output path for the merged screened GTI.
    logger : logging.Logger | None, optional
        Logger for progress messages.

    Returns
    -------
    dict
        Optimization summary and optional output paths.
    """
    lc_data = load_lightcurve(
        infile,
        xcol=xcol,
        ycol=ycol,
        fracexpcol=fracexpcol,
        tsstyle=tsstyle,
        fracexpstyle=fracexpstyle,
        fracexplower=fracexplower,
        starttime=starttime,
        endtime=endtime,
        lowercutoffcount=lowercutoffcount,
    )
    result = find_optimal_rate(lc_data, min_time_ratio=min_time_ratio)
    emit(logger, "info", f"fxtbkgoptrate status={result['status']} threshold={result['best_threshold']} kept_fraction={result['kept_fraction']:.4f}")
    if diagnostic_outfile is not None:
        write_diagnostic_table(diagnostic_outfile, lc_data, result)
    if flare_gti_outfile is not None:
        build_gti_from_mask(lc_data, result["kept_mask"], flare_gti_outfile)
    screened_gti_path = None
    if base_gti_path is not None and flare_gti_outfile is not None and screened_gti_outfile is not None:
        screened_gti_path = intersect_gtis(base_gti_path, flare_gti_outfile, screened_gti_outfile)
    output = dict(result)
    output.update(
        {
            "diagnostic_outfile": diagnostic_outfile,
            "flare_gti_outfile": flare_gti_outfile,
            "screened_gti_outfile": screened_gti_path,
        }
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``fxtbkgoptrate``."""
    parser = argparse.ArgumentParser(description="Optimize a background light-curve threshold and optionally write flare GTIs.")
    parser.add_argument("infile", help="Input light-curve FITS file.")
    parser.add_argument("--xcol", default="TIME", help="Time column name. Default: TIME")
    parser.add_argument("--ycol", default=None, help="Rate/count column name. Default: auto-detect")
    parser.add_argument("--fracexpcol", default="FRACEXP", help="FRACEXP column name. Default: FRACEXP")
    parser.add_argument("--tsstyle", choices=["auto", "rate", "count"], default="auto", help="Input light-curve style. Default: auto")
    parser.add_argument("--fracexpstyle", choices=["auto", "calc", "threshold", "none"], default="auto", help="FRACEXP handling policy. Default: auto")
    parser.add_argument("--fracexplower", type=float, default=0.0, help="Lower FRACEXP threshold when fracexpstyle=threshold.")
    parser.add_argument("--starttime", type=float, default=None, help="Optional absolute lower time bound.")
    parser.add_argument("--endtime", type=float, default=None, help="Optional absolute upper time bound.")
    parser.add_argument("--lowercutoffcount", type=float, default=None, help="Optional lower cutoff applied to the raw Y values.")
    parser.add_argument("--min-time-ratio", type=float, default=0.05, help="Minimum retained exposure fraction. Default: 0.05")
    parser.add_argument("--diag-out", default=None, help="Optional diagnostic FITS output.")
    parser.add_argument("--flare-gti-out", default=None, help="Optional flare-only GTI output.")
    parser.add_argument("--base-gti", default=None, help="Optional base GTI FITS used to create a screened GTI.")
    parser.add_argument("--screened-gti-out", default=None, help="Optional merged GTI output.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="CLI logging level.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional CLI log file.")
    return parser


def main() -> None:
    """Run the ``fxtbkgoptrate`` CLI."""
    args = build_parser().parse_args()
    logger = build_cli_logger("eFXTDAS.fxtbkgoptrate", args.log_level, args.log_file)
    result = run_bkgoptrate(
        args.infile,
        xcol=args.xcol,
        ycol=args.ycol,
        fracexpcol=args.fracexpcol,
        tsstyle=args.tsstyle,
        fracexpstyle=args.fracexpstyle,
        fracexplower=args.fracexplower,
        starttime=args.starttime,
        endtime=args.endtime,
        lowercutoffcount=args.lowercutoffcount,
        min_time_ratio=args.min_time_ratio,
        diagnostic_outfile=args.diag_out,
        flare_gti_outfile=args.flare_gti_out,
        base_gti_path=args.base_gti,
        screened_gti_outfile=args.screened_gti_out,
        logger=logger,
    )
    threshold = result["best_threshold"]
    if np.isfinite(threshold):
        print(f"{threshold:.8g}")
    else:
        print("nan")


if __name__ == "__main__":
    main()
