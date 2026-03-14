#!/usr/bin/env python3
"""
Simplified version of FXTDAS FXTCHAIN.
"""
from astropy.io import fits
from fxtcombine.utils.energy import energy_range_suffix, energy_range_to_channel_range
from fxtcombine.utils.logger import emit
from fxtcombine.utils.cmd import run_cmd, remove_xselect_tmp_files
import os
import shutil
import time


def fxtchain_obsid(
        obsid_file_dict,datatype_lst,
        obsid_out_dir,obsid_log_dir,
        expr="DEFAULT",grade="0-12",
        image_energy_ranges=None,
        lightcurve_energy_ranges=None,
        skip_existing=True,
        obsid_logger=None,
    ):
    """Run the simplified per-OBSID preprocessing chain.

    Parameters
    ----------
    obsid_file_dict : dict
        Parsed input-file dictionary for one OBSID.
    datatype_lst : list[str]
        Datatypes to process, typically ``["evt"]`` or ``["evt", "fsaevt"]``.
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
    skip_existing : bool, optional
        When ``True``, skip substeps whose outputs already exist.
    obsid_logger : logging.Logger | None, optional
        Logger used for progress reporting.

    Returns
    -------
    dict
        Nested dictionary of generated products keyed by datatype and event
        filename prefix.
    """

    att_fname = next(iter(obsid_file_dict["att"].values()))["filePath"]
    mkf_fname = next(iter(obsid_file_dict["mkf"].values()))["filePath"]
    if image_energy_ranges is None:
        image_energy_ranges = [(0.3, 10.0)]
    if lightcurve_energy_ranges is None:
        lightcurve_energy_ranges = [(0.1, 12.0)]
    # datatype_lst = [datatype for datatype in list(obsid_file_dict.keys()) if datatype not in ["mkf","att","orb"]] # [evt|fsaevt]
    obsid_prod_dict = {}  # datatype [evt|fsaevt] -- module [..a..|..b..] -- product type [clevt|expmap]
    
    for datatype in datatype_lst:

        obsid_prod_dict[datatype] = {}

        for evt_fname_prefix,evt_dict in obsid_file_dict[datatype].items():

            evt_fname = evt_dict["filePath"]
            ver = evt_dict["version"]
            module = evt_dict["module"]
            obsid = evt_dict["obsID"]
            datamode = evt_dict["mode"]
            filt = evt_dict["filter"]
            pp = evt_dict["pp"]
            level = evt_dict["level"]
            image_band_channels = {
                energy_range_suffix(energy_range): energy_range_to_channel_range(energy_range, module)
                for energy_range in image_energy_ranges
            }
            lightcurve_band_channels = {
                energy_range_suffix(energy_range): energy_range_to_channel_range(energy_range, module)
                for energy_range in lightcurve_energy_ranges
            }

            emit(obsid_logger, "info", f"*** Processing {evt_fname} ***")
            t0 = time.time()
            sub_log_dir = os.path.join(obsid_log_dir,f"{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}")
            os.makedirs(sub_log_dir,exist_ok=True)

            #--- run fxtcoord
            evt_coord_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.coord")
            if os.path.exists(evt_coord_fname) and skip_existing: # lazy: skip this step if file already exists
                emit(obsid_logger, "info", f"{evt_coord_fname} already exists.")
            else:
                fxtcoord_cmd = " ".join([
                    "fxtcoord",
                    f"evtfile={evt_fname}",f"attfile={att_fname}",
                    f"outfile={evt_coord_fname}","clobber=yes",
                ])
                emit(obsid_logger, "info", "Running fxtcoord ...")
                fxtcoord_log = os.path.join(sub_log_dir,f"fxtcoord.log")
                run_cmd(fxtcoord_cmd,logger=obsid_logger,logname=fxtcoord_log)

            #--- run fxtpical
            evt_pi_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.pi")
            if os.path.exists(evt_pi_fname) and skip_existing:
                emit(obsid_logger, "info", f"{evt_pi_fname} already exists.")
            else:
                fxtpical_cmd = " ".join([
                    "fxtpical",
                    f"evtfile={evt_coord_fname}",
                    f"outfile={evt_pi_fname}",
                ])
                emit(obsid_logger, "info", "Running fxtpical ...")
                fxtpical_log = os.path.join(sub_log_dir,f"fxtpical.log")
                run_cmd(fxtpical_cmd,logger=obsid_logger,logname=fxtpical_log)

            #--- run fxtparticleidentify
            evt_particle_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.particle")
            if os.path.exists(evt_particle_fname) and skip_existing:
                emit(obsid_logger, "info", f"{evt_particle_fname} already exists.")
            else:
                if datamode in ["TM","DM","tm","dm"]:
                    x_length = "35"
                    y_length = "35"
                else:
                    x_length = "11"
                    y_length = "11"
                fxtparticleidentify_cmd = " ".join([
                    "fxtparticleidentify",
                    f"evtfile={evt_pi_fname}",f"xlength={x_length}",f"ylength={y_length}",
                    f"outfile={evt_particle_fname}",
                ])
                fxtparticleidentify_log = os.path.join(sub_log_dir,f"fxtparticleidentify.log")
                run_cmd(fxtparticleidentify_cmd,logger=obsid_logger,logname=fxtparticleidentify_log)

            #--- run fxtbadpix
            evt_badpix_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.badpix")
            if os.path.exists(evt_badpix_fname) and skip_existing:
                emit(obsid_logger, "info", f"{evt_badpix_fname} already exists.")
            else:
                fxtbadpix_cmd = " ".join([
                    "fxtbadpix",
                    f"evtfile={evt_particle_fname}",
                    f"outfile={evt_badpix_fname}",
                ])
                emit(obsid_logger, "info", "Running fxtbadpix ...")
                fxtbadpix_log = os.path.join(sub_log_dir,f"fxtbadpix.log")
                run_cmd(fxtbadpix_cmd,logger=obsid_logger,logname=fxtbadpix_log)

            #--- run fxtgrade
            evt_grade_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.grade")
            if os.path.exists(evt_grade_fname) and skip_existing:
                emit(obsid_logger, "info", f"{evt_grade_fname} already exists.")
            else:
                fxtgrade_cmd = " ".join([
                    "fxtgrade",
                    f"evtfile={evt_badpix_fname}",f"pithresh=0",
                    f"outfile={evt_grade_fname}",
                ])
                emit(obsid_logger, "info", "Running fxtgrade ...")
                fxtgrade_log = os.path.join(sub_log_dir,f"fxtgrade.log")
                run_cmd(fxtgrade_cmd,logger=obsid_logger,logname=fxtgrade_log)


            #--- run fxtgtigen
            gti_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.gti")
            if os.path.exists(gti_fname) and skip_existing:
                emit(obsid_logger, "info", f"{gti_fname} already exists.")
            else:
                fxtgtigen_cmd = " ".join([
                    "fxtgtigen",
                    f"mkffile={mkf_fname}",f"module=fxt{module}",f"expr={expr}",
                    f"outfile={gti_fname}",
                ])
                emit(obsid_logger, "info", "Running fxtgtigen ...")
                fxtgtigen_log = os.path.join(sub_log_dir,f"fxtgtigen.log")
                run_cmd(fxtgtigen_cmd,logger=obsid_logger,logname=fxtgtigen_log)

            #--- run xselect to get clean events plus requested band-limited images/light curves
            # TODO: remove existing EP_* log to avoid rerunning?
            xsl_fname = os.path.join(sub_log_dir,f"img.xsl")
            evt_cl_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_cl.fits")
            image_fname_map = {
                energy_range_suffix(energy_range): os.path.join(
                    obsid_out_dir,
                    f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.img",
                )
                for energy_range in image_energy_ranges
            }
            lc_fname_map = {
                energy_range_suffix(energy_range): os.path.join(
                    obsid_out_dir,
                    f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.lc",
                )
                for energy_range in lightcurve_energy_ranges
            }
            expected_stage1_products = [evt_cl_fname] + list(image_fname_map.values()) + list(lc_fname_map.values())
            if all(os.path.exists(path) for path in expected_stage1_products) and skip_existing:
                emit(obsid_logger, "info", f"{evt_cl_fname} and all requested image/light-curve products already exist.")
            else:
                remove_xselect_tmp_files()
                with open(xsl_fname,"w") as f:
                    f.writelines([f"EP\n"])
                    f.writelines([f"set datadir {obsid_out_dir}\n"])
                    f.writelines([f"read events {os.path.basename(evt_grade_fname)}\n"])
                    f.writelines([f"yes\n"])
                    ##--- filter clean events
                    # TODO: binsize?
                    f.writelines([f"filter grade {grade}\n"])
                    f.writelines([f"filter time file {os.path.basename(gti_fname)}\n"])
                    f.writelines([f'select event "status==b0"\n'])
                    f.writelines([f"show status\n"])
                    f.writelines([f"extract events copyall=yes\n"])
                    f.writelines([f"save events {evt_cl_fname} clobberit=yes\n"])
                    f.writelines([f"no\n"])
                    f.writelines([f"clear all\n"])
                    f.writelines([f"yes\n"])
                    ##--- get energy-selected images
                    for energy_range in image_energy_ranges:
                        suffix = energy_range_suffix(energy_range)
                        img_fname = image_fname_map[suffix]
                        chan_lo, chan_hi = image_band_channels[suffix]
                        f.writelines([f"set datadir {obsid_out_dir}\n"])
                        f.writelines([f"read events {os.path.basename(evt_cl_fname)}\n"])
                        f.writelines([f"filter pha_cutoff {chan_lo} {chan_hi}\n"])
                        f.writelines([f"extract image xysize=601 xybinsize=1 xcenter=300 ycenter=300 copyall=yes\n"])
                        f.writelines([f"save image {img_fname} clobberit=yes\n"])
                        f.writelines([f"clear pha_cutoff\n"])
                        f.writelines([f"clear events\n"])
                    ##--- get energy-selected light curves
                    for energy_range in lightcurve_energy_ranges:
                        suffix = energy_range_suffix(energy_range)
                        lc_fname = lc_fname_map[suffix]
                        chan_lo, chan_hi = lightcurve_band_channels[suffix]
                        f.writelines([f"set datadir {obsid_out_dir}\n"])
                        f.writelines([f"read events {os.path.basename(evt_cl_fname)}\n"])
                        f.writelines([f"filter pha_cutoff {chan_lo} {chan_hi}\n"])
                        f.writelines([f"extract curve copyall=yes\n"])
                        f.writelines([f"save curve {lc_fname} clobberit=yes\n"])
                        f.writelines([f"clear pha_cutoff\n"])
                        f.writelines([f"clear events\n"])
                    ##--- finish
                    f.writelines([f"clear all proceed=yes\n"])
                    f.writelines([f"quit\n"])
                    f.writelines([f"no\n"])
                xsl_cmd = f"xselect @{xsl_fname}"
                emit(obsid_logger, "info", "Running xselect ...")
                xselect_log = os.path.join(sub_log_dir,f"xselect.log")
                run_cmd(xsl_cmd,logger=obsid_logger,logname=xselect_log)

            #--- run fxtexpogen
            exp_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}.expo")
            if os.path.exists(exp_fname) and skip_existing:
                emit(obsid_logger, "info", f"{exp_fname} already exists.")
            else:
                fxtexpogen_cmd = " ".join([
                    "fxtexpogen",
                    f"mkffile={mkf_fname}",f"evtfile={evt_cl_fname}",
                    f"energy=1.5","area_scale=no","edgecovermask=0",
                    "use_clustering=yes","ra_threshold=2.0","pa_threshold=0.01",
                    f"outfile={exp_fname}","clobber=yes"
                ])
                emit(obsid_logger, "info", "Running fxtexpogen ...")
                fxtexpogen_log = os.path.join(sub_log_dir,f"fxtexpogen.log")
                run_cmd(fxtexpogen_cmd,logger=obsid_logger,logname=fxtexpogen_log)

            #--- run fxteefmap (from eFXTDAS) for each requested image band
            eefmap_fname_map = {
                energy_range_suffix(energy_range): os.path.join(
                    obsid_out_dir,
                    f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_{energy_range_suffix(energy_range)}.eef",
                )
                for energy_range in image_energy_ranges
            }
            for energy_range in image_energy_ranges:
                image_key = energy_range_suffix(energy_range)
                eefmap_fname = eefmap_fname_map[image_key]
                img_fname = image_fname_map[image_key]
                emin, emax = energy_range
                if os.path.exists(eefmap_fname) and skip_existing:
                    emit(obsid_logger, "info", f"{eefmap_fname} already exists.")
                else:
                    fxteefmap_cmd = " ".join([
                        "fxteefmap",
                        f"{img_fname}",
                        "--out", f"{eefmap_fname}",
                        "--expmap", f"{exp_fname}",
                        "--mission", "ep-fxt",
                        "--emin", f"{emin}",
                        "--emax", f"{emax}",
                    ])
                    emit(obsid_logger, "info", f"Running fxteefmap for {image_key} ...")
                    fxteefmap_log = os.path.join(sub_log_dir,f"fxteefmap_{image_key}.log")
                    run_cmd(fxteefmap_cmd,logger=obsid_logger,logname=fxteefmap_log)

            #--- append prod_dict
            obsid_prod_dict[datatype][evt_fname_prefix] = {}
            obsid_prod_dict[datatype][evt_fname_prefix]["clevt"] = evt_cl_fname
            first_image_key = energy_range_suffix(image_energy_ranges[0])
            first_lc_key = energy_range_suffix(lightcurve_energy_ranges[0])
            obsid_prod_dict[datatype][evt_fname_prefix]["image"] = image_fname_map[first_image_key]
            obsid_prod_dict[datatype][evt_fname_prefix]["images"] = image_fname_map
            obsid_prod_dict[datatype][evt_fname_prefix]["image_band_channels"] = image_band_channels
            obsid_prod_dict[datatype][evt_fname_prefix]["vexpmap"] = exp_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["eefmap"] = eefmap_fname_map[first_image_key]
            obsid_prod_dict[datatype][evt_fname_prefix]["eefmaps"] = eefmap_fname_map
            obsid_prod_dict[datatype][evt_fname_prefix]["exp"] = fits.getval(exp_fname,ext=0,keyword="EXPOSURE")
            obsid_prod_dict[datatype][evt_fname_prefix]["alllc"] = lc_fname_map[first_lc_key]
            obsid_prod_dict[datatype][evt_fname_prefix]["lightcurves"] = lc_fname_map
            obsid_prod_dict[datatype][evt_fname_prefix]["lightcurve_band_channels"] = lightcurve_band_channels


            emit(obsid_logger, "info", f"Finish running using {time.time()-t0} s.")


    return obsid_prod_dict


def fxt_extract_spec(
        obsid_file_dict,obsid_prod_dict,datatype_lst,
        src_reg_fname,bkg_reg_fname,
        obsid_out_dir,obsid_log_dir,
        skip_existing=True,
        obsid_logger=None,
    ):
    """Extract source/background spectra and responses for one OBSID.

    Parameters
    ----------
    obsid_file_dict : dict
        Parsed input-file dictionary for one OBSID.
    obsid_prod_dict : dict
        Existing product dictionary produced by :func:`fxtchain_obsid`.
    datatype_lst : list[str]
        Datatypes to process.
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
        Updated product dictionary including spectra, light curves, ARF, and
        RMF products.
    """
    for datatype in datatype_lst:

        for evt_fname_prefix,evt_dict in obsid_file_dict[datatype].items():

            evt_fname = evt_dict["filePath"]
            ver = evt_dict["version"]
            module = evt_dict["module"]
            obsid = evt_dict["obsID"]
            datamode = evt_dict["mode"]
            filt = evt_dict["filter"]
            pp = evt_dict["pp"]
            level = evt_dict["level"]

            emit(obsid_logger, "info", f"*** Processing {evt_fname} ***")
            t0 = time.time()
            sub_log_dir = os.path.join(obsid_log_dir,f"{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}")
            os.makedirs(sub_log_dir,exist_ok=True)

            #--- run xselect to extract src & bkg spectra
            xsl_fname = os.path.join(sub_log_dir,f"spec.xsl")
            evt_cl_fname = obsid_prod_dict[datatype][evt_fname_prefix]["clevt"]
            shutil.copy(src_reg_fname,obsid_out_dir)
            shutil.copy(bkg_reg_fname,obsid_out_dir)
            srcevt_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src_cl.fits")
            srcpi_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src.pi")
            bkgpi_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_bkg.pi")
            srclc_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src.lc")
            bkglc_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_bkg.lc")
            default_image_key = obsid_prod_dict[datatype][evt_fname_prefix]["image"]
            default_band_key = next(
                key for key, path in obsid_prod_dict[datatype][evt_fname_prefix]["images"].items() if path == default_image_key
            )
            lc_chan_lo, lc_chan_hi = obsid_prod_dict[datatype][evt_fname_prefix]["image_band_channels"][default_band_key]
            if os.path.exists(srcevt_fname) and os.path.exists(srcpi_fname) and os.path.exists(bkgpi_fname) and os.path.exists(srclc_fname) and os.path.exists(bkglc_fname) and skip_existing:
                emit(obsid_logger, "info", f"{srcevt_fname} and {srcpi_fname} and {bkgpi_fname} and {srclc_fname} and {bkglc_fname} already exists.")
            else:
                remove_xselect_tmp_files()
                with open(xsl_fname,"w") as f:
                    f.writelines([f"EP\n"])
                    f.writelines([f"set datadir {obsid_out_dir}\n"])
                    f.writelines([f"read events {os.path.basename(evt_cl_fname)}\n"])
                    f.writelines([f"yes\n"])
                    ##--- extract source events
                    f.writelines([f"filter region {src_reg_fname}\n"])  # use relpath would fail to load
                    f.writelines([f"extract events copyall=yes\n"])
                    f.writelines([f"save events {srcevt_fname} clobberit=yes\n"])
                    f.writelines([f"no\n"])
                    f.writelines([f"clear all\n"])
                    f.writelines([f"yes\n"])
                    ##--- extract source spectra
                    f.writelines([f"set datadir {obsid_out_dir}\n"])
                    f.writelines([f"read events {os.path.basename(evt_cl_fname)}\n"])
                    f.writelines([f"filter region {src_reg_fname}\n"])
                    f.writelines([f"extract spectrum copyall=yes\n"])
                    f.writelines([f"save spectrum {srcpi_fname} clobberit=yes\n"])
                    f.writelines([f"clear region\n"])
                    ##--- extract background spectra
                    f.writelines([f"filter region {bkg_reg_fname}\n"])
                    f.writelines([f"extract spectrum copyall=yes\n"])
                    f.writelines([f"save spectrum {bkgpi_fname} clobberit=yes\n"])
                    f.writelines([f"clear region\n"])
                    ##--- extract source light curve
                    f.writelines([f"filter region {src_reg_fname}\n"])
                    f.writelines([f"filter pha_cutoff {lc_chan_lo} {lc_chan_hi}\n"])
                    f.writelines([f"extract curve copyall=yes\n"])
                    f.writelines([f"save curve {srclc_fname} clobberit=yes\n"])
                    f.writelines([f"clear region\n"])
                    f.writelines([f"clear pha_cutoff\n"])
                    ##--- extract background light curve
                    f.writelines([f"filter region {bkg_reg_fname}\n"])
                    f.writelines([f"filter pha_cutoff {lc_chan_lo} {lc_chan_hi}\n"])
                    f.writelines([f"extract curve copyall=yes\n"])
                    f.writelines([f"save curve {bkglc_fname} clobberit=yes\n"])
                    f.writelines([f"clear region\n"])
                    f.writelines([f"clear pha_cutoff\n"])
                    ##--- finish
                    f.writelines([f"clear all proceed=yes\n"])
                    f.writelines([f"quit\n"])
                    f.writelines([f"no\n"])
                xsl_cmd = f"xselect @{xsl_fname}"
                emit(obsid_logger, "info", "Running xselect ...")
                xselect_log = os.path.join(sub_log_dir,f"xselect.log")
                run_cmd(xsl_cmd,logger=obsid_logger,logname=xselect_log)

            #--- run arfgen
            arf_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src.arf")
            if os.path.exists(arf_fname) and skip_existing:
                emit(obsid_logger, "info", f"{arf_fname} already exists.")
            else:
                exp_fname = obsid_prod_dict[datatype][evt_fname_prefix]["vexpmap"]
                fxtarfgen_cmd = " ".join([
                    "fxtarfgen",
                    f"specfile={srcpi_fname}",f"expfile={exp_fname}",
                    "extend=0","psfcor=1",  # important! psfcor should be 1, regardless of pile-up or not!
                    f"outfile={arf_fname}","clobber=yes",
                ])
                emit(obsid_logger, "info", "Running fxtarfgen ...")
                fxtarfgen_log = os.path.join(sub_log_dir,f"fxtarfgen.log")
                run_cmd(fxtarfgen_cmd,logger=obsid_logger,logname=fxtarfgen_log)

            #--- run rmfgen
            rmf_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src.rmf")
            if os.path.exists(rmf_fname) and skip_existing:
                emit(obsid_logger, "info", f"{rmf_fname} already exists.")
            else:
                fxtrmfgen_cmd = " ".join([
                    "fxtrmfgen",
                    f"specfile={srcpi_fname}",
                    f"outfile={rmf_fname}","clobber=yes",
                ])
                emit(obsid_logger, "info", "Running fxtrmfgen ...")
                fxtrmfgen_log = os.path.join(sub_log_dir,f"fxtrmfgen.log")
                run_cmd(fxtrmfgen_cmd,logger=obsid_logger,logname=fxtrmfgen_log)

            #--- sanity check on pile-up
            grade_fname = os.path.join(obsid_out_dir,f"fxt_{module}_{obsid}_{datamode}_{filt}_{pp}_{datatype}_{ver}_src.grade")
            fxtplotgrade_cmd = " ".join([
                "fxtplotgrade",
                f"evtfile={srcevt_fname}",
                f"outfile={grade_fname}","clobber=yes",
            ])
            emit(obsid_logger, "info", "Running fxtplotgrade ...")
            fxtplotgrade_log = os.path.join(sub_log_dir,f"fxtplotgrade.log")
            # run_cmd(fxtplotgrade_cmd,logger=obsid_logger,logname=fxtplotgrade_log)

            ##--- TODO: update keywords? prod_dict?
            #--- update keywords
            fits.setval(srcpi_fname,ext=1,keyword="BACKFILE",value=os.path.basename(bkgpi_fname),comment="bkg")
            fits.setval(srcpi_fname,ext=1,keyword="RESPFILE",value=os.path.basename(rmf_fname),comment="rmf")
            fits.setval(srcpi_fname,ext=1,keyword="ANCRFILE",value=os.path.basename(arf_fname),comment="arf")
            fits.setval(bkgpi_fname,ext=1,keyword="RESPFILE",value=os.path.basename(rmf_fname),comment="rmf")
            fits.setval(bkgpi_fname,ext=1,keyword="ANCRFILE",value=os.path.basename(arf_fname),comment="arf")
            fits.setval(rmf_fname,ext=1,keyword="ANCRFILE",value=os.path.basename(arf_fname),comment="arf")

            #--- update product dict
            obsid_prod_dict[datatype][evt_fname_prefix]["srcclevt"] = srcevt_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["srcpi"] = srcpi_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["bkgpi"] = bkgpi_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["rmf"] = rmf_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["arf"] = arf_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["srclc"] = srclc_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["bkglc"] = bkglc_fname
            obsid_prod_dict[datatype][evt_fname_prefix]["srcgrade"] = grade_fname


    return obsid_prod_dict
