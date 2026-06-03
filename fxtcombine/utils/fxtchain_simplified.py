#!/usr/bin/env python3
"""Simplified Stage-1 and spectral-extraction wrappers around FXTDAS."""

from __future__ import annotations

import os
import shutil
import time

from astropy.io import fits

from fxtcombine.utils.cmd import finalize_xselect_log, remove_xselect_tmp_files, run_cmd
from fxtcombine.utils.eefmask import build_detection_mask, filter_code_to_name, infer_optaxis_from_image, module_to_instrument
from fxtcombine.utils.energy import energy_range_suffix, energy_range_to_channel_range
from fxtcombine.utils.flarescreen import run_fsa_flare_screening
from fxtcombine.utils.logger import emit


def _stream_identity(evt_dict: dict) -> tuple[str, str, str, str, str]:
    """Return the identity tuple used to link EVT and FSAEVT files.

    Parameters
    ----------
    evt_dict : dict
        Input event-file metadata dictionary.

    Returns
    -------
    tuple[str, str, str, str, str]
        Tuple of module, obsid, mode, filter, and processing pipeline tag.
    """
    return (
        evt_dict["module"],
        evt_dict["obsID"],
        evt_dict["mode"],
        evt_dict["filter"],
        evt_dict["pp"],
    )


def build_stream_records(obsid_file_dict: dict) -> dict[str, dict]:
    """Build per-stream records from raw input file discovery.

    Parameters
    ----------
    obsid_file_dict : dict
        Raw file-discovery dictionary from :func:`get_input_files`.

    Returns
    -------
    dict[str, dict]
        Mapping keyed by EVT filename prefix. Each record contains the science
        EVT file, optional matching FSAEVT file, and shared auxiliary files.
    """
    fsa_lookup = {
        _stream_identity(fsa_evt_dict): fsa_evt_dict
        for fsa_evt_dict in obsid_file_dict.get("fsaevt", {}).values()
    }
    stream_records: dict[str, dict] = {}
    for evt_prefix, evt_dict in obsid_file_dict.get("evt", {}).items():
        stream_records[evt_prefix] = {
            "stream_key": evt_prefix,
            "evt_prefix": evt_prefix,
            "evt": evt_dict,
            "fsaevt": fsa_lookup.get(_stream_identity(evt_dict)),
            "mkf": next(iter(obsid_file_dict["mkf"].values()))["filePath"],
            "att": next(iter(obsid_file_dict["att"].values()))["filePath"],
        }
    return stream_records


def _prepare_event_chain(
    evt_dict: dict,
    datatype: str,
    *,
    att_fname: str,
    mkf_fname: str,
    expr: str,
    obsid_out_dir: str,
    obsid_log_dir: str,
    skip_existing: bool,
    obsid_logger=None,
) -> dict:
    """Run the calibration chain up to graded events and base GTI."""
    evt_fname = evt_dict["filePath"]
    ver = evt_dict["version"]
    module = evt_dict["module"]
    obsid = evt_dict["obsID"]
    datamode = evt_dict["mode"]
    filt = evt_dict["filter"]
    pp = evt_dict["pp"]
    sub_log_dir = os.path.join(obsid_log_dir, f"{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}")
    os.makedirs(sub_log_dir, exist_ok=True)

    coord_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.coord")
    if not (os.path.exists(coord_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtcoord",
                f"evtfile={evt_fname}",
                f"attfile={att_fname}",
                f"outfile={coord_path}",
                "clobber=yes",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtcoord_{datatype}.log"),
            cwd=sub_log_dir,
        )

    pi_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.pi")
    if not (os.path.exists(pi_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtpical",
                f"evtfile={coord_path}",
                f"outfile={pi_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtpical_{datatype}.log"),
            cwd=sub_log_dir,
        )

    particle_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.particle")
    if not (os.path.exists(particle_path) and skip_existing):
        if datamode.lower() in ["tm", "dm"]:
            x_length, y_length = "35", "35"
        else:
            x_length, y_length = "11", "11"
        run_cmd(
            " ".join([
                "fxtparticleidentify",
                f"evtfile={pi_path}",
                f"xlength={x_length}",
                f"ylength={y_length}",
                f"outfile={particle_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtparticleidentify_{datatype}.log"),
            cwd=sub_log_dir,
        )

    badpix_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.badpix")
    if not (os.path.exists(badpix_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtbadpix",
                f"evtfile={particle_path}",
                f"outfile={badpix_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtbadpix_{datatype}.log"),
            cwd=sub_log_dir,
        )

    grade_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.grade")
    if not (os.path.exists(grade_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtgrade",
                f"evtfile={badpix_path}",
                "pithresh=0",
                f"outfile={grade_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtgrade_{datatype}.log"),
            cwd=sub_log_dir,
        )

    gti_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.gti")
    if not (os.path.exists(gti_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtgtigen",
                f"mkffile={mkf_fname}",
                f"module=fxt{module}",
                f"expr={expr}",
                f"outfile={gti_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxtgtigen_{datatype}.log"),
            cwd=sub_log_dir,
        )

    return {
        "coord": coord_path,
        "pi": pi_path,
        "particle": particle_path,
        "badpix": badpix_path,
        "grade": grade_path,
        "gti": gti_path,
        "sub_log_dir": sub_log_dir,
    }


def _extract_fsa_stage1_products(
    fsa_evt_dict: dict,
    fsa_prep: dict,
    *,
    selected_gti: str,
    grade: str,
    obsid_out_dir: str,
    skip_existing: bool,
    obsid_logger=None,
) -> dict:
    """Generate clean FSA products and the instrumental-background spectrum."""
    module = fsa_evt_dict["module"]
    obsid = fsa_evt_dict["obsID"]
    datamode = fsa_evt_dict["mode"]
    filt = fsa_evt_dict["filter"]
    pp = fsa_evt_dict["pp"]
    ver = fsa_evt_dict["version"]
    datatype = fsa_evt_dict["fileType"]
    sub_log_dir = fsa_prep["sub_log_dir"]
    xsl_path = os.path.join(sub_log_dir, "fsaevt_stage1.xsl")

    clevt_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_cl.fits")
    spec_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.pi")
    lc_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.lc")
    img_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.img")
    instbkg_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_instbkg.pi")

    if not (all(os.path.exists(path) for path in [clevt_path, spec_path, lc_path, img_path]) and skip_existing):
        remove_xselect_tmp_files(obsid_out_dir)
        with open(xsl_path, "w") as handle:
            handle.writelines(["EP\n"])
            handle.writelines([f"set datadir {obsid_out_dir}\n"])
            handle.writelines([f"read events {os.path.basename(fsa_prep['grade'])}\n"])
            handle.writelines(["yes\n"])
            handle.writelines([f"filter grade {grade}\n"])
            handle.writelines([f"filter time file {selected_gti}\n"])
            handle.writelines(['select event "status==b0"\n'])
            handle.writelines(["show status\n", "extract events copyall=yes\n"])
            handle.writelines([f"save events {os.path.basename(clevt_path)} clobberit=yes\n", "no\n"])
            handle.writelines(["extract spectrum copyall=yes\n"])
            handle.writelines([f"save spectrum {os.path.basename(spec_path)} clobberit=yes\n"])
            handle.writelines(["filter pha_cutoff 38 925\n"])
            handle.writelines(['filter column "DETX=3:382 DETY=3:382"\n'])
            handle.writelines(["extract events copyall=yes\n"])
            handle.writelines([f"save events {os.path.basename(clevt_path)} clobberit=yes\n", "no\n"])
            handle.writelines(["set binsize 20\n", "extract curve copyall=yes\n"])
            handle.writelines([f"save curve {os.path.basename(lc_path)} clobberit=yes\n"])
            handle.writelines(["extract image xysize=601 xybinsize=1 xcenter=300 ycenter=300 copyall=yes\n"])
            handle.writelines([f"save image {os.path.basename(img_path)} clobberit=yes\n"])
            handle.writelines(["clear all proceed=yes\n", "quit\n", "no\n"])
        run_cmd(
            f"xselect @{xsl_path}",
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, "xselect_fsaevt_stage1.log"),
            cwd=obsid_out_dir,
        )
        finalize_xselect_log(obsid_out_dir, os.path.join(sub_log_dir, "xselect_fsaevt_stage1.log"))

    if not (os.path.exists(instbkg_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtbkggen",
                f"infile={spec_path}",
                f"outfile={instbkg_path}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, "fxtbkggen_fsaevt.log"),
            cwd=sub_log_dir,
        )
    return {
        "fsa_clevt": clevt_path,
        "fsa_spec": spec_path,
        "fsa_lc": lc_path,
        "fsa_img": img_path,
        "instbkgpi": instbkg_path,
    }


def _extract_evt_stage1_products(
    evt_dict: dict,
    evt_prep: dict,
    *,
    selected_gti: str,
    grade: str,
    image_energy_ranges: list[tuple[float, float]],
    lightcurve_energy_ranges: list[tuple[float, float]],
    mkf_fname: str,
    obsid_out_dir: str,
    skip_existing: bool,
    obsid_logger=None,
) -> dict:
    """Generate clean EVT products for one stream."""
    module = evt_dict["module"]
    obsid = evt_dict["obsID"]
    datamode = evt_dict["mode"]
    filt = evt_dict["filter"]
    pp = evt_dict["pp"]
    ver = evt_dict["version"]
    datatype = evt_dict["fileType"]
    sub_log_dir = evt_prep["sub_log_dir"]
    clevt_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_cl.fits")
    xsl_path = os.path.join(sub_log_dir, "evt_stage1.xsl")

    image_band_channels = {
        energy_range_suffix(energy_range): energy_range_to_channel_range(energy_range, module)
        for energy_range in image_energy_ranges
    }
    lightcurve_band_channels = {
        energy_range_suffix(energy_range): energy_range_to_channel_range(energy_range, module)
        for energy_range in lightcurve_energy_ranges
    }
    image_paths = {
        energy_range_suffix(energy_range): os.path.join(
            obsid_out_dir,
            f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.img",
        )
        for energy_range in image_energy_ranges
    }
    lightcurve_paths = {
        energy_range_suffix(energy_range): os.path.join(
            obsid_out_dir,
            f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.lc",
        )
        for energy_range in lightcurve_energy_ranges
    }
    expected_paths = [clevt_path] + list(image_paths.values()) + list(lightcurve_paths.values())
    if not (all(os.path.exists(path) for path in expected_paths) and skip_existing):
        remove_xselect_tmp_files(obsid_out_dir)
        with open(xsl_path, "w") as handle:
            handle.writelines(["EP\n"])
            handle.writelines([f"set datadir {obsid_out_dir}\n"])
            handle.writelines([f"read events {os.path.basename(evt_prep['grade'])}\n"])
            handle.writelines(["yes\n"])
            handle.writelines([f"filter grade {grade}\n"])
            handle.writelines([f"filter time file {selected_gti}\n"])
            handle.writelines(['select event "status==b0"\n'])
            handle.writelines(["show status\n", "extract events copyall=yes\n"])
            handle.writelines([f"save events {os.path.basename(clevt_path)} clobberit=yes\n", "no\n", "clear all\n", "yes\n"])
            for energy_range in image_energy_ranges:
                suffix = energy_range_suffix(energy_range)
                chan_lo, chan_hi = image_band_channels[suffix]
                handle.writelines([f"set datadir {obsid_out_dir}\n"])
                handle.writelines([f"read events {os.path.basename(clevt_path)}\n"])
                handle.writelines([f"filter pha_cutoff {chan_lo} {chan_hi}\n"])
                handle.writelines(["extract image xysize=601 xybinsize=1 xcenter=300 ycenter=300 copyall=yes\n"])
                handle.writelines([f"save image {os.path.basename(image_paths[suffix])} clobberit=yes\n"])
                handle.writelines(["clear pha_cutoff\n", "clear events\n"])
            for energy_range in lightcurve_energy_ranges:
                suffix = energy_range_suffix(energy_range)
                chan_lo, chan_hi = lightcurve_band_channels[suffix]
                handle.writelines([f"set datadir {obsid_out_dir}\n"])
                handle.writelines([f"read events {os.path.basename(clevt_path)}\n"])
                handle.writelines([f"filter pha_cutoff {chan_lo} {chan_hi}\n"])
                handle.writelines(["extract curve copyall=yes\n"])
                handle.writelines([f"save curve {os.path.basename(lightcurve_paths[suffix])} clobberit=yes\n"])
                handle.writelines(["clear pha_cutoff\n", "clear events\n"])
            handle.writelines(["clear all proceed=yes\n", "quit\n", "no\n"])
        run_cmd(
            f"xselect @{xsl_path}",
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, "xselect_evt_stage1.log"),
            cwd=obsid_out_dir,
        )
        finalize_xselect_log(obsid_out_dir, os.path.join(sub_log_dir, "xselect_evt_stage1.log"))

    exp_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.expo")
    if not (os.path.exists(exp_path) and skip_existing):
        run_cmd(
            " ".join([
                "fxtexpogen",
                f"mkffile={mkf_fname}",
                f"evtfile={clevt_path}",
                "energy=1.5",
                "area_scale=no",
                "edgecovermask=0",
                "use_clustering=yes",
                "ra_threshold=2.0",
                "pa_threshold=0.01",
                f"outfile={exp_path}",
                "clobber=yes",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, "fxtexpogen_evt.log"),
            cwd=sub_log_dir,
        )

    eefmap_paths = {
        energy_range_suffix(energy_range): os.path.join(
            obsid_out_dir,
            f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.eef",
        )
        for energy_range in image_energy_ranges
    }
    first_image_key = energy_range_suffix(image_energy_ranges[0])
    first_lc_key = energy_range_suffix(lightcurve_energy_ranges[0])
    instrument = module_to_instrument(module)
    filter_name = filter_code_to_name(filt)
    for energy_range in image_energy_ranges:
        suffix = energy_range_suffix(energy_range)
        if os.path.exists(eefmap_paths[suffix]) and skip_existing:
            continue
        emin, emax = energy_range
        optaxis_x, optaxis_y = infer_optaxis_from_image(image_paths[suffix])
        run_cmd(
            " ".join([
                "fxteefmap",
                image_paths[suffix],
                "--out", eefmap_paths[suffix],
                "--expmap", exp_path,
                "--mission", "ep-fxt",
                "--instrument", instrument,
                "--filter", filter_name,
                "--emin", f"{emin}",
                "--emax", f"{emax}",
                "--optaxis-x", f"{optaxis_x}",
                "--optaxis-y", f"{optaxis_y}",
            ]),
            logger=obsid_logger,
            logname=os.path.join(sub_log_dir, f"fxteefmap_evt_{suffix}.log"),
            cwd=sub_log_dir,
        )

    detection_mask_path = os.path.join(
        obsid_out_dir,
        f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{first_image_key}.detectmask.fits",
    )
    if not (os.path.exists(detection_mask_path) and skip_existing):
        detection_mask_meta = build_detection_mask(
            image_path=image_paths[first_image_key],
            expmap_path=exp_path,
            eefmap_path=eefmap_paths[first_image_key],
            out_path=detection_mask_path,
        )
        if obsid_logger is not None:
            emit(
                obsid_logger,
                "info",
                f"Per-OBSID detection mask written to {detection_mask_path} "
                f"(valid={detection_mask_meta['valid_pixels']}, theta_max={detection_mask_meta['theta_max_caldb']:.3f} arcmin)",
            )

    return {
        "evt_clevt": clevt_path,
        "image": image_paths[first_image_key],
        "images": image_paths,
        "image_band_channels": image_band_channels,
        "vexpmap": exp_path,
        "eefmap": eefmap_paths[first_image_key],
        "eefmaps": eefmap_paths,
        "detection_mask": detection_mask_path,
        "exp": fits.getval(exp_path, ext=0, keyword="EXPOSURE"),
        "alllc": lightcurve_paths[first_lc_key],
        "lightcurves": lightcurve_paths,
        "lightcurve_band_channels": lightcurve_band_channels,
    }


def fxtchain_obsid(
    obsid_file_dict,
    obsid_out_dir,
    obsid_log_dir,
    expr="DEFAULT",
    grade="0-12",
    image_energy_ranges=None,
    lightcurve_energy_ranges=None,
    flare_screen=True,
    flare_energy_range=(0.5, 10.0),
    flare_binsize=20.0,
    flare_min_time_ratio=0.05,
    skip_existing=True,
    obsid_logger=None,
):
    """Run the coupled per-OBSID preprocessing chain.

    Parameters
    ----------
    obsid_file_dict : dict
        Parsed input-file dictionary for one OBSID.
    obsid_out_dir : str
        Output directory for generated products of this OBSID.
    obsid_log_dir : str
        Directory used for command log files of this OBSID.
    expr : str, optional
        GTI expression passed to ``fxtgtigen``.
    grade : str, optional
        Grade filter passed to xselect.
    image_energy_ranges : list[tuple[float, float]] | None, optional
        Energy ranges in keV used to generate images through xselect.
    lightcurve_energy_ranges : list[tuple[float, float]] | None, optional
        Energy ranges in keV used to generate light curves through xselect.
    flare_screen : bool, optional
        Whether to run FSA-based flare screening for FF-mode science data.
    flare_energy_range : tuple[float, float], optional
        Energy band in keV used for the flare-screening light curve.
    flare_binsize : float, optional
        Flare-screening light-curve bin size in seconds.
    flare_min_time_ratio : float, optional
        Minimum retained exposure fraction accepted by the threshold optimizer.
    skip_existing : bool, optional
        When ``True``, skip substeps whose outputs already exist.
    obsid_logger : logging.Logger | None, optional
        Logger used for progress reporting.

    Returns
    -------
    dict
        Flat per-stream product dictionary keyed by EVT filename prefix.
    """
    if image_energy_ranges is None:
        image_energy_ranges = [(0.3, 10.0)]
    if lightcurve_energy_ranges is None:
        lightcurve_energy_ranges = [(0.1, 12.0)]

    stream_records = build_stream_records(obsid_file_dict)
    obsid_prod_dict: dict[str, dict] = {}
    for stream_key, stream_record in stream_records.items():    # stream_key looks like `fxt_a_00012345_ff_open_pp0_evt_v01``
        evt_dict = stream_record["evt"]
        att_fname = stream_record["att"]
        mkf_fname = stream_record["mkf"]
        module = evt_dict["module"]
        obsid = evt_dict["obsID"]
        datamode = evt_dict["mode"]
        filt = evt_dict["filter"]
        pp = evt_dict["pp"]
        ver = evt_dict["version"]
        emit(obsid_logger, "info", f"*** Processing {evt_dict['filePath']} ***")
        t0 = time.time()

        flare_meta = {
            "base_gti": None,
            "screened_gti": None,
            "flare_gti": None,
            "flare_lc": None,
            "flare_diag": None,
            "flare_threshold": None,
            "flare_kept_fraction": 1.0,
            "flare_screen_applied": False,
            "flare_screen_status": "disabled",
        }
        fsa_products: dict[str, str] = {}
        selected_gti = None

        if flare_screen and datamode.lower() == "ff" and stream_record["fsaevt"] is not None:
            fsa_evt_dict = stream_record["fsaevt"]
            fsa_prep = _prepare_event_chain(
                fsa_evt_dict,
                "fsaevt",
                att_fname=att_fname,
                mkf_fname=mkf_fname,
                expr=expr,
                obsid_out_dir=obsid_out_dir,
                obsid_log_dir=obsid_log_dir,
                skip_existing=skip_existing,
                obsid_logger=obsid_logger,
            )
            flare_meta = run_fsa_flare_screening(
                fsa_evt_dict,
                fsa_prep,
                base_gti_path=fsa_prep["gti"],
                grade=grade,
                flare_energy_range=flare_energy_range,
                flare_binsize=flare_binsize,
                flare_min_time_ratio=flare_min_time_ratio,
                obsid_out_dir=obsid_out_dir,
                skip_existing=skip_existing,
                obsid_logger=obsid_logger,
            )
            selected_gti = flare_meta["screened_gti"]
            fsa_products = _extract_fsa_stage1_products(
                fsa_evt_dict,
                fsa_prep,
                selected_gti=selected_gti,
                grade=grade,
                obsid_out_dir=obsid_out_dir,
                skip_existing=skip_existing,
                obsid_logger=obsid_logger,
            )
        elif datamode.lower() == "ff":
            flare_meta["flare_screen_status"] = "skip_no_fsaevt"
        else:
            flare_meta["flare_screen_status"] = "skip_non_ff"

        evt_prep = _prepare_event_chain(
            evt_dict,
            "evt",
            att_fname=att_fname,
            mkf_fname=mkf_fname,
            expr=expr,
            obsid_out_dir=obsid_out_dir,
            obsid_log_dir=obsid_log_dir,
            skip_existing=skip_existing,
            obsid_logger=obsid_logger,
        )
        if selected_gti is None:
            selected_gti = evt_prep["gti"]
            flare_meta["base_gti"] = evt_prep["gti"]
            flare_meta["screened_gti"] = evt_prep["gti"]
            if flare_meta["flare_threshold"] is None:
                flare_meta["flare_threshold"] = None

        evt_products = _extract_evt_stage1_products(
            evt_dict,
            evt_prep,
            selected_gti=selected_gti,
            grade=grade,
            image_energy_ranges=image_energy_ranges,
            lightcurve_energy_ranges=lightcurve_energy_ranges,
            mkf_fname=mkf_fname,
            obsid_out_dir=obsid_out_dir,
            skip_existing=skip_existing,
            obsid_logger=obsid_logger,
        )
        obsid_prod_dict[stream_key] = {
            "stream_key": stream_key,
            "module": module,
            "obsid": obsid,
            "datamode": datamode,
            "filter": filt,
            "pp": pp,
            "version": ver,
            "evt_file": evt_dict["filePath"],
            "fsaevt_file": stream_record["fsaevt"]["filePath"] if stream_record["fsaevt"] is not None else None,
            **flare_meta,
            **fsa_products,
            **evt_products,
        }
        emit(obsid_logger, "info", f"Finish running using {time.time()-t0} s.")
    return obsid_prod_dict


def fxt_extract_spec(
    obsid_prod_dict,
    src_reg_fname,
    bkg_reg_fname,
    obsid_out_dir,
    obsid_log_dir,
    skip_existing=True,
    obsid_logger=None,
):
    """Extract source/background spectra and responses for one OBSID.

    Parameters
    ----------
    obsid_prod_dict : dict
        Flat per-stream product dictionary produced by :func:`fxtchain_obsid`.
    src_reg_fname : str
        Source region file used for extraction.
    bkg_reg_fname : str
        Background region file used for extraction.
    obsid_out_dir : str
        Output directory for generated products of this OBSID.
    obsid_log_dir : str
        Directory used for command log files of this OBSID.
    skip_existing : bool, optional
        When ``True``, skip substeps whose outputs already exist.
    obsid_logger : logging.Logger | None, optional
        Logger used for progress reporting.

    Returns
    -------
    dict
        Updated per-stream product dictionary including spectra, light curves,
        ARF, and RMF products.
    """
    for stream_key, prod in obsid_prod_dict.items():
        module = prod["module"]
        obsid = prod["obsid"]
        datamode = prod["datamode"]
        filt = prod["filter"]
        pp = prod["pp"]
        ver = prod["version"]
        emit(obsid_logger, "info", f"*** Processing spectral products for {stream_key} ***")
        sub_log_dir = os.path.join(obsid_log_dir, f"{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}")
        os.makedirs(sub_log_dir, exist_ok=True)
        xsl_path = os.path.join(sub_log_dir, "evt_stage4_spec.xsl")

        evt_cl_path = prod["evt_clevt"]
        shutil.copy(src_reg_fname, obsid_out_dir)
        shutil.copy(bkg_reg_fname, obsid_out_dir)
        srcevt_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src_cl.fits")
        srcpi_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src.pi")
        bkgpi_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_bkg.pi")
        srclc_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src.lc")
        bkglc_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_bkg.lc")
        default_image_path = prod["image"]
        default_band_key = next(key for key, path in prod["images"].items() if path == default_image_path)
        lc_chan_lo, lc_chan_hi = prod["image_band_channels"][default_band_key]
        expected_paths = [srcevt_path, srcpi_path, bkgpi_path, srclc_path, bkglc_path]
        if not (all(os.path.exists(path) for path in expected_paths) and skip_existing):
            remove_xselect_tmp_files(obsid_out_dir)
            with open(xsl_path, "w") as handle:
                handle.writelines(["EP\n"])
                handle.writelines([f"set datadir {obsid_out_dir}\n"])
                handle.writelines([f"read events {os.path.basename(evt_cl_path)}\n"])
                handle.writelines(["yes\n"])
                handle.writelines([f"filter region {src_reg_fname}\n"])
                handle.writelines(["extract events copyall=yes\n"])
                handle.writelines([f"save events {os.path.basename(srcevt_path)} clobberit=yes\n", "no\n", "clear all\n", "yes\n"])
                handle.writelines([f"set datadir {obsid_out_dir}\n", f"read events {os.path.basename(evt_cl_path)}\n"])
                handle.writelines([f"filter region {src_reg_fname}\n", "extract spectrum copyall=yes\n"])
                handle.writelines([f"save spectrum {os.path.basename(srcpi_path)} clobberit=yes\n", "clear region\n"])
                handle.writelines([f"filter region {bkg_reg_fname}\n", "extract spectrum copyall=yes\n"])
                handle.writelines([f"save spectrum {os.path.basename(bkgpi_path)} clobberit=yes\n", "clear region\n"])
                handle.writelines([f"filter region {src_reg_fname}\n", f"filter pha_cutoff {lc_chan_lo} {lc_chan_hi}\n", "extract curve copyall=yes\n"])
                handle.writelines([f"save curve {os.path.basename(srclc_path)} clobberit=yes\n", "clear region\n", "clear pha_cutoff\n"])
                handle.writelines([f"filter region {bkg_reg_fname}\n", f"filter pha_cutoff {lc_chan_lo} {lc_chan_hi}\n", "extract curve copyall=yes\n"])
                handle.writelines([f"save curve {os.path.basename(bkglc_path)} clobberit=yes\n", "clear region\n", "clear pha_cutoff\n"])
                handle.writelines(["clear all proceed=yes\n", "quit\n", "no\n"])
            run_cmd(
                f"xselect @{xsl_path}",
                logger=obsid_logger,
                logname=os.path.join(sub_log_dir, "xselect_evt_stage4_spec.log"),
                cwd=obsid_out_dir,
            )
            finalize_xselect_log(obsid_out_dir, os.path.join(sub_log_dir, "xselect_evt_stage4_spec.log"))

        arf_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src.arf")
        if not (os.path.exists(arf_path) and skip_existing):
            run_cmd(
                " ".join([
                    "fxtarfgen",
                    f"specfile={srcpi_path}",
                    f"expfile={prod['vexpmap']}",
                    "extend=0",
                    "psfcor=1",
                    f"outfile={arf_path}",
                    "clobber=yes",
                ]),
                logger=obsid_logger,
                logname=os.path.join(sub_log_dir, "fxtarfgen.log"),
                cwd=sub_log_dir,
            )

        rmf_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src.rmf")
        if not (os.path.exists(rmf_path) and skip_existing):
            run_cmd(
                " ".join([
                    "fxtrmfgen",
                    f"specfile={srcpi_path}",
                    f"outfile={rmf_path}",
                    "clobber=yes",
                ]),
                logger=obsid_logger,
                logname=os.path.join(sub_log_dir, "fxtrmfgen.log"),
                cwd=sub_log_dir,
            )

        grade_path = os.path.join(obsid_out_dir, f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_evt_{ver}_src.grade")
        emit(obsid_logger, "info", "Running fxtplotgrade ...")
        # keep as a logged no-op until the existing plotting workflow is re-enabled
        fits.setval(srcpi_path, ext=1, keyword="BACKFILE", value=os.path.basename(bkgpi_path), comment="bkg")
        fits.setval(srcpi_path, ext=1, keyword="RESPFILE", value=os.path.basename(rmf_path), comment="rmf")
        fits.setval(srcpi_path, ext=1, keyword="ANCRFILE", value=os.path.basename(arf_path), comment="arf")
        fits.setval(bkgpi_path, ext=1, keyword="RESPFILE", value=os.path.basename(rmf_path), comment="rmf")
        fits.setval(bkgpi_path, ext=1, keyword="ANCRFILE", value=os.path.basename(arf_path), comment="arf")
        fits.setval(rmf_path, ext=1, keyword="ANCRFILE", value=os.path.basename(arf_path), comment="arf")
        prod.update(
            {
                "srcclevt": srcevt_path,
                "srcpi": srcpi_path,
                "bkgpi": bkgpi_path,
                "rmf": rmf_path,
                "arf": arf_path,
                "srclc": srclc_path,
                "bkglc": bkglc_path,
                "srcgrade": grade_path,
            }
        )
    return obsid_prod_dict
