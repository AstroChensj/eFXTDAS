"""Helpers for FF-mode FSA flare screening."""

from __future__ import annotations

import os
import shlex

import numpy as np
from astropy.io import fits

from fxtcombine.utils.cmd import finalize_xselect_log, remove_xselect_tmp_files, run_cmd
from fxtcombine.utils.energy import energy_range_to_channel_range
from fxtcombine.utils.logger import emit


def _load_flare_diag_metadata(diag_path: str) -> dict:
    """Load persisted flare-screening summary metadata from one diagnostic FITS.

    Parameters
    ----------
    diag_path : str
        Path to the diagnostic FITS table written by ``fxtbkgoptrate``.

    Returns
    -------
    dict
        Dictionary containing the persisted threshold, kept fraction, and status.
    """
    with fits.open(diag_path) as hdul:
        hdr = hdul[1].header
        return {
            "flare_threshold": hdr.get("BGOPTCUT", np.nan),
            "flare_kept_fraction": hdr.get("FRACTLFT", 1.0),
            "flare_screen_status": hdr.get("OPTSTAT", "optimal"),
        }


def run_fsa_flare_screening(
    fsa_evt_dict: dict,
    fsa_prep: dict,
    *,
    base_gti_path: str,
    grade: str,
    flare_energy_range: tuple[float, float],
    flare_binsize: float,
    flare_min_time_ratio: float,
    obsid_out_dir: str,
    skip_existing: bool,
    obsid_logger=None,
) -> dict:
    """Run FSA-based flare screening and produce screened GTIs.

    Parameters
    ----------
    fsa_evt_dict : dict
        Matched FSAEVT file metadata.
    fsa_prep : dict
        Prepared FSA calibration-chain products up to grade/GTI.
    base_gti_path : str
        Base GTI path used to derive the screened GTI.
    grade : str
        Grade filter passed to xselect.
    flare_energy_range : tuple[float, float]
        Energy range in keV used for the flare-screening light curve.
    flare_binsize : float
        Flare-screening light-curve bin size in seconds.
    flare_min_time_ratio : float
        Minimum retained exposure fraction passed to ``fxtbkgoptrate``.
    obsid_out_dir : str
        OBSID product directory.
    skip_existing : bool
        Whether to reuse existing intermediate products.
    obsid_logger : logging.Logger | None, optional
        Logger used for progress reporting.

    Returns
    -------
    dict
        Flare-screening metadata including GTI paths and optimizer outputs.
    """
    module = fsa_evt_dict["module"]
    obsid = fsa_evt_dict["obsID"]
    datamode = fsa_evt_dict["mode"]
    filt = fsa_evt_dict["filter"]
    pp = fsa_evt_dict["pp"]
    ver = fsa_evt_dict["version"]
    flare_lc_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_fsaevt_{ver}_flare.lc")
    flare_diag_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_flare_diag.fits")
    flare_gti_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_flare.gti")
    screened_gti_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_screened.gti")

    if not (os.path.exists(flare_lc_path) and skip_existing):
        chan_lo, chan_hi = energy_range_to_channel_range(flare_energy_range, module)
        xsl_path = os.path.join(fsa_prep["sub_log_dir"], "fsaevt_flare.xsl")
        remove_xselect_tmp_files(obsid_out_dir)
        with open(xsl_path, "w") as handle:
            handle.writelines(["EP\n"])
            handle.writelines([f"set datadir {obsid_out_dir}\n"])
            handle.writelines([f"read events {os.path.basename(fsa_prep['grade'])}\n"])
            handle.writelines(["yes\n"])
            handle.writelines([f"filter grade {grade}\n"])
            handle.writelines([f"filter time file {base_gti_path}\n"])
            handle.writelines(['select event "status==b0"\n'])
            handle.writelines([f"filter pha_cutoff {chan_lo} {chan_hi}\n"])
            handle.writelines(['filter column "DETX=3:382 DETY=3:382"\n'])
            handle.writelines([f"set binsize {float(flare_binsize)}\n"])
            handle.writelines(["extract curve copyall=yes\n"])
            handle.writelines([f"save curve {os.path.basename(flare_lc_path)} clobberit=yes\n"])
            handle.writelines(["clear all proceed=yes\n", "quit\n", "no\n"])
        emit(obsid_logger, "info", "Running xselect for FSA flare-screening light curve ...")
        flare_log = os.path.join(fsa_prep["sub_log_dir"], "xselect_fsaevt_flare.log")
        run_cmd(f"xselect @{xsl_path}", logger=obsid_logger, logname=flare_log, cwd=obsid_out_dir)
        finalize_xselect_log(obsid_out_dir, flare_log)

    if all(
        os.path.exists(path)
        for path in [flare_lc_path, flare_diag_path, flare_gti_path, screened_gti_path]
    ) and skip_existing:
        flare_meta = _load_flare_diag_metadata(flare_diag_path)
        return {
            "base_gti": base_gti_path,
            "screened_gti": screened_gti_path,
            "flare_gti": flare_gti_path,
            "flare_lc": flare_lc_path,
            "flare_diag": flare_diag_path,
            "flare_screen_applied": True,
            **flare_meta,
        }

    flare_cmd = " ".join(
        [
            "fxtbkgoptrate",
            shlex.quote(flare_lc_path),
            "--min-time-ratio",
            shlex.quote(str(flare_min_time_ratio)),
            "--diag-out",
            shlex.quote(flare_diag_path),
            "--flare-gti-out",
            shlex.quote(flare_gti_path),
            "--base-gti",
            shlex.quote(base_gti_path),
            "--screened-gti-out",
            shlex.quote(screened_gti_path),
        ]
    )
    emit(obsid_logger, "info", "Running fxtbkgoptrate for FSA flare screening ...")
    flare_opt_log = os.path.join(fsa_prep["sub_log_dir"], "fxtbkgoptrate_fsaevt.log")
    run_cmd(flare_cmd, logger=obsid_logger, logname=flare_opt_log, cwd=obsid_out_dir)

    missing_outputs = [
        path
        for path in [flare_diag_path, flare_gti_path, screened_gti_path]
        if not os.path.exists(path)
    ]
    if missing_outputs:
        raise FileNotFoundError(
            "fxtbkgoptrate completed without writing expected outputs: "
            + ", ".join(missing_outputs)
        )

    flare_meta = _load_flare_diag_metadata(flare_diag_path)
    return {
        "base_gti": base_gti_path,
        "screened_gti": screened_gti_path,
        "flare_gti": flare_gti_path,
        "flare_lc": flare_lc_path,
        "flare_diag": flare_diag_path,
        "flare_screen_applied": True,
        **flare_meta,
    }
