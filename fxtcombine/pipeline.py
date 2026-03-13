#!/usr/bin/env python3
"""
Simple calling FXTDAS tasks.
"""
import argparse
import json
import logging
import numpy as np
import os
from pathlib import Path
import sys

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from fxtcombine.config import (
	FXT_POSITION_ERR90_ARCSEC,
	SRC_EXTRACT_RADIUS,
	BKG_EXTRACT_INNER_RADIUS,
	BKG_EXTRACT_OUTER_RADIUS,
)
from fxtcombine.utils.logger import build_cli_logger, build_file_logger, emit
from fxtcombine.utils.cmd import run_cmd
from fxtcombine.utils.fxtchain_simplified import fxtchain_obsid, fxt_extract_spec
from fxtcombine.utils.fxtprep import get_input_files
from fxtcombine.utils.image import reproject_events_xy_to_refwcs


def fxtcombine_pipeline(
		src_dir,ra=None,dec=None,obsid_lst=None,
		out_dir="./",module="a,b",datamode="ff",datatype="evt",grade="0-12",expr="DEFAULT",skip_existing=False,
		logger: logging.Logger | None = None,
	):
	"""Combine multiple EP-FXT observations into stacked images and spectra.

	Parameters
	----------
	src_dir : str
		Directory containing one subdirectory per OBSID with the expected FXT
		data layout.
	ra : float | None, optional
		Target right ascension in degrees.
	dec : float | None, optional
		Target declination in degrees.
	obsid_lst : str | None, optional
		Comma-separated OBSID list or a file containing one OBSID per line.
	out_dir : str, optional
		Output directory used for stacked products and per-OBSID intermediates.
	module : str, optional
		Comma-separated module selection, for example ``"a,b"``.
	datamode : str, optional
		Comma-separated datamode selection.
	datatype : str, optional
		Comma-separated datatype selection, for example ``"evt"`` or
		``"evt,fsaevt"``.
	grade : str, optional
		Grade filter passed to the xselect stage.
	expr : str, optional
		GTI selection expression passed to ``fxtgtigen``.
	skip_existing : bool, optional
		When ``True``, reuse existing intermediate products where supported.
		When ``False``, rerun all steps.
	logger : logging.Logger | None, optional
		Optional logger used for workflow messages. When omitted, a file logger is
		created under ``out_dir/log``.

	Returns
	-------
	None
		Products are written to disk in ``out_dir``.
	"""
	#--- parse parameter
	src_dir = os.path.abspath(src_dir)
	out_dir = os.path.abspath(out_dir)
	module_lst = module.split(",")
	datamode_lst = datamode.split(",")
	datatype_lst = datatype.split(",")

	#--- define logger
	os.makedirs(out_dir,exist_ok=True)
	log_dir = os.path.join(out_dir,"log")
	os.makedirs(log_dir,exist_ok=True)
	main_logname = os.path.join(log_dir, "fxtcombine.log")
	main_logger = logger if logger is not None else build_file_logger("eFXTDAS.fxtcombine", main_logname)
	emit(main_logger, "info", f"**** Welcome to FXTCOMBINE ! ****")
	emit(main_logger, "info", f"Source directory is: {src_dir}")
	emit(main_logger, "info", f"Source coordinate is: ICRS({ra}, {dec})")


	#--- get obsid list
	emit(main_logger, "info", "Reading OBSID list ...")
	if os.path.exists(obsid_lst):	# input is a file
		emit(main_logger, "info", "Input obsid_lst is a file.")
		with open(obsid_lst,"r") as f:
			lines = f.readlines()
		obsids = [line.strip() for line in lines if line.strip()]
	else:	# input is an array, e.g., `xxxxx,yyyyy,zzzzz`
		emit(main_logger, "info", "Input obsid_lst is a list.")
		obsids = [obsid.strip() for obsid in obsid_lst.split(",") if obsid.strip()]

	obsid_lst = []
	for obsid in obsids:
		obsid_dir = os.path.join(src_dir,obsid)
		if os.path.isdir(obsid_dir):
			obsid_lst.append(obsid)

	emit(main_logger, "info", f"Valid OBSIDs for processing: {obsid_lst}")
	if not obsid_lst:
		raise ValueError("No valid OBSIDs were found for processing.")


	#--- initial iterate: to get clean events & image & exposure map for each obsid
	emit(main_logger, "info", f"**** Stage 1: initial iterate to get clean events & exposure map for each OBSID ****")
	all_file_dict = {}
	all_prod_dict = {}
	for obsid in obsid_lst:
		emit(main_logger, "info", f"Processing {obsid} ...")
		obsid_dir = os.path.join(src_dir,obsid)
		obsid_out_dir = os.path.join(out_dir,obsid,"products")
		os.makedirs(obsid_out_dir,exist_ok=True)
		emit(main_logger, "info", f"Output directory is {obsid_out_dir}")
		##--- define obsid logger
		obsid_log_dir = os.path.join(obsid_out_dir,"log")
		os.makedirs(obsid_log_dir,exist_ok=True)
		obsid_logname = os.path.join(obsid_log_dir, "fxtchain.log")
		obsid_logger = build_file_logger(f"eFXTDAS.fxtcombine.{obsid}.stage1", obsid_logname)
		##--- parse events files
		obsid_file_dict = get_input_files(obsid_dir,datamode,module,datatype)
		all_file_dict[obsid] = obsid_file_dict
		##--- calling sub-modules to run fxtchain for this obsid
		##--- generates: clean events, 0.3-10 keV image, exposure map
		obsid_prod_dict = fxtchain_obsid(
			obsid_file_dict=obsid_file_dict,datatype_lst=datatype_lst,
			obsid_out_dir=obsid_out_dir,obsid_log_dir=obsid_log_dir,
			expr=expr,grade=grade,
			skip_existing=skip_existing,
			obsid_logger=obsid_logger,
		) # TODO: print logger at each stage?
		all_prod_dict[obsid] = obsid_prod_dict


	#--- stack all images and expmaps; evt and fsaevt (if any) stacked separately
	emit(main_logger, "info", "**** Stage 2: stacking all images and exposure maps. EVT and FSAEVT are stacked separately. ****")
	stack_dir = os.path.join(out_dir,"stack")
	os.makedirs(stack_dir,exist_ok=True)
	for datatype in datatype_lst:	# [evt|fsaevt]
		emit(main_logger, "info", f"For {datatype} ...")
		clevt_fname_lst = []
		img_fname_lst = []
		exp_fname_lst = []
		exp_lst = []

		for obsid,obsid_prod_dict in all_prod_dict.items():
			for evt_fname_prefix,evt_prod_dict in obsid_prod_dict[datatype].items():
				clevt_fname_lst.append(evt_prod_dict["clevt"])
				img_fname_lst.append(evt_prod_dict["image"])
				exp_fname_lst.append(evt_prod_dict["vexpmap"])
				exp_lst.append(evt_prod_dict["exp"])
		clevt_fname_lst = np.array(clevt_fname_lst)
		img_fname_lst = np.array(img_fname_lst)
		exp_fname_lst = np.array(exp_fname_lst)
		exp_lst = np.array(exp_lst)
		if len(clevt_fname_lst) == 0:
			raise ValueError(f"No products found for datatype={datatype}. Check your OBSID selection and datatype filters.")

		##--- sort according to expo
		idx_sort = np.argsort(exp_lst)[::-1]
		clevt_fname_lst = clevt_fname_lst[idx_sort]
		img_fname_lst = img_fname_lst[idx_sort]
		exp_fname_lst = exp_fname_lst[idx_sort]
		exp_lst = exp_lst[idx_sort]
		if datatype == "evt":
			exp_tot = np.sum(exp_lst)

		##--- logging all files
		emit(main_logger, "info", f"You have the following clean events files: {clevt_fname_lst}")
		emit(main_logger, "info", f"You have the following images: {img_fname_lst}")
		emit(main_logger, "info", f"You have the following vign-exposure maps: {exp_fname_lst}")
		emit(main_logger, "info", f"Their corresponding exposures are: {exp_lst}")

		##--- choosing the reference frame (with longest exposure)
		with fits.open(img_fname_lst[0]) as hdu:
			refimg = hdu[0]
			refimg_wcs = WCS(refimg.header)
			refimg_shape = refimg.data.shape
			cts_sum = np.zeros(refimg_shape)
		emit(main_logger, "info", f"The reference frame is {img_fname_lst[0]}.")
		emit(main_logger, "info", f"Reference WCS is {refimg_wcs}.")

		with fits.open(exp_fname_lst[0]) as hdu:
			refexp = hdu[0]
			refexp_wcs = WCS(refexp.header)
			refexp_shape = refexp.data.shape
			exp_sum = np.zeros(refexp_shape)
		assert refimg_shape == refexp_shape, f"The image and exposure should have same shape, but now gets {refimg_shape} and {refexp_shape}!"

		# ##--- reproject and stack image
		# main_logger.info(f"Reprojecting and stacking images ...")
		# for img_fname_i in img_fname_lst:
		# 	with fits.open(img_fname_i) as hdu:
		# 		img_i = hdu[0]
		# 		data_i = img_i.data
		# 		wcs_i  = WCS(img_i.header)
		# 	###--- flux/count-conserving reprojection
		# 	data_i_reproj,footprint_i = reproject_exact((data_i,wcs_i),refimg_wcs,shape_out=refimg_shape)
		# 	###--- footprint is 0..1 overlap fraction; use it to ignore empty pixels
		# 	m = np.isfinite(data_i_reproj) & (footprint_i > 0)
		# 	cts_sum[m] += data_i_reproj[m]
		# cts_sum_fname = os.path.join(stack_dir,f"{datatype}_stack_cts.fits")
		# fits.writeto(cts_sum_fname,cts_sum,refimg.header,overwrite=True)
		# main_logger.info(f"Stacked count image written to {cts_sum_fname}")

		##--- reproject and stack image
		emit(main_logger, "info", f"Reprojecting and stacking images ...")
		for i in range(len(clevt_fname_lst)):
			clevt_fname = clevt_fname_lst[i]
			img_fname = img_fname_lst[i]
			with fits.open(clevt_fname) as hdu:
				clevt_data = hdu[1].data
			clevt_x = clevt_data["X"]
			clevt_y = clevt_data["Y"]
			with fits.open(img_fname) as hdu:
				img_wcs = WCS(hdu[0].header)
				img = reproject_events_xy_to_refwcs(
					clevt_x,clevt_y,
					img_wcs,refimg_wcs,
					shape_ref=refimg_shape,	# (ny, nx)
					weight=None,
					method="nearest",  		# "nearest" or "floor"
					event_origin=1.0,
				)
			cts_sum += img
		cts_sum_fname = os.path.join(stack_dir,f"{datatype}_stack_cts.fits")
		fits.writeto(cts_sum_fname,cts_sum,refimg.header,overwrite=True)
		emit(main_logger, "info", f"Stacked count image written to {cts_sum_fname}")

		##--- reproject and stack expmap
		emit(main_logger, "info", f"Reprojecting and stacking exposure maps ...")
		for exp_fname_i in exp_fname_lst:
			with fits.open(exp_fname_i) as hdu:
				exp_i = hdu[0]
				data_i = exp_i.data
				wcs_i  = WCS(exp_i.header)
			###--- flux/count-conserving reprojection
			data_i_reproj,footprint_i = reproject_interp((data_i,wcs_i),refexp_wcs,shape_out=refexp_shape)
			###--- footprint is 0..1 overlap fraction; use it to ignore empty pixels
			m = np.isfinite(data_i_reproj) & (footprint_i > 0)
			exp_sum[m] += data_i_reproj[m]
		exp_sum_fname = os.path.join(stack_dir,f"{datatype}_stack_exp.fits")
		fits.writeto(exp_sum_fname,exp_sum,refexp.header,overwrite=True)
		emit(main_logger, "info", f"Stacked exposure map written to {exp_sum_fname}")

		##--- rate map
		rate_sum = cts_sum / exp_sum
		rate_sum[np.isinf(rate_sum)] = 0
		rate_sum[np.isnan(rate_sum)] = 0
		rate_sum_fname = os.path.join(stack_dir,f"{datatype}_stack_rate.fits")
		fits.writeto(rate_sum_fname,rate_sum,refimg.header,overwrite=True)
		emit(main_logger, "info", f"Stacked rate image written to {rate_sum_fname}")


	#--- source detection on stacked evt products and region generation
	emit(main_logger, "info", "**** Stage 3: source detection and region file creation ****")
	stack_image_fname = os.path.join(stack_dir, "evt_stack_cts.fits")
	stack_expmap_fname = os.path.join(stack_dir, "evt_stack_exp.fits")
	srcdet_src_fname = os.path.join(stack_dir, "stack_src.fits")
	srcdet_reg_fname = os.path.join(stack_dir, "stack_src.reg")
	srcdet_bkg_fname = os.path.join(stack_dir, "stack_bkgmap.fits")
	srcdet_log = os.path.join(stack_dir, "srcdet.log")
	srcdet_cmd = " ".join([
		f'"{sys.executable}"',
		"-m", "fxtsrcdet",
		f'"{stack_image_fname}"',
		"--expmap", f'"{stack_expmap_fname}"',
		"--mission", "ep-fxt",
		"--emin", "0.3",
		"--emax", "10.0",
		"--out", f'"{srcdet_src_fname}"',
		"--regfile", f'"{srcdet_reg_fname}"',
		"--save-bkgmap", f'"{srcdet_bkg_fname}"',
		"--log-file", f'"{srcdet_log}"',
	])
	run_cmd(srcdet_cmd, logger=main_logger, logname=srcdet_log)
	emit(main_logger, "info", f"Stacked source catalog written to {srcdet_src_fname}")
	emit(main_logger, "info", f"Stacked source region file written to {srcdet_reg_fname}")
	emit(main_logger, "info", f"Stacked background map written to {srcdet_bkg_fname}")

	#--- src & bkg region definition via fxtregions
	emit(main_logger, "info", "Generating src & bkg regions with fxtregions ...")
	src_reg_fname = os.path.join(stack_dir, "target_src.reg")
	bkg_reg_fname = os.path.join(stack_dir, "target_bkg.reg")
	fxtregions_log = os.path.join(stack_dir, "fxtregions.log")
	fxtregions_cmd = " ".join([
		f'"{sys.executable}"',
		"-m", "fxtregions",
		f'"{stack_image_fname}"',
		f'"{srcdet_src_fname}"',
		"--bkgmap", f'"{srcdet_bkg_fname}"',
		"--ra", f"{float(ra)}",
		"--dec", f"{float(dec)}",
		"--mission", "ep-fxt",
		"--emin", "0.3",
		"--emax", "10.0",
		"--mode", "manual",
		"--src-radius", f"{SRC_EXTRACT_RADIUS}",
		"--bkg-inner", f"{BKG_EXTRACT_INNER_RADIUS}",
		"--bkg-outer", f"{BKG_EXTRACT_OUTER_RADIUS}",
		"--match-threshold", f"{FXT_POSITION_ERR90_ARCSEC}",
		"--src-regfile", f'"{src_reg_fname}"',
		"--bkg-regfile", f'"{bkg_reg_fname}"',
		"--log-file", f'"{fxtregions_log}"',
	])
	run_cmd(fxtregions_cmd, logger=main_logger, logname=fxtregions_log)
	emit(main_logger, "info", f"Target source region file saved to {src_reg_fname}")
	emit(main_logger, "info", f"Target source background region file saved to {bkg_reg_fname}")


	#--- second iterate: spectral extraction, with contaminating sources excluded
	emit(main_logger, "info", "**** Stage 4: Iterate again on each obsid to extract spectra ****")
	for obsid in obsid_lst:
		emit(main_logger, "info", f"Processing {obsid} ...")
		obsid_dir = os.path.join(src_dir,obsid)
		obsid_out_dir = os.path.join(out_dir,obsid,"products")
		##--- define obsid logger
		obsid_log_dir = os.path.join(obsid_out_dir,"log")
		obsid_logname = os.path.join(obsid_log_dir, "fxtchain.log")
		obsid_logger = build_file_logger(f"eFXTDAS.fxtcombine.{obsid}.stage4", obsid_logname)
		##--- parse events files
		obsid_file_dict = all_file_dict[obsid]
		##--- calling sub-modules to run fxtchain for this obsid
		##--- generates: clean events, 0.3-10 keV image, exposure map
		obsid_prod_dict = all_prod_dict[obsid]
		obsid_prod_dict = fxt_extract_spec(
			obsid_file_dict,obsid_prod_dict,datatype_lst,
			src_reg_fname,bkg_reg_fname,
			obsid_out_dir,obsid_log_dir,
			skip_existing=skip_existing,
			obsid_logger=obsid_logger,
		)
		all_prod_dict[obsid] = obsid_prod_dict	# update

	#--- spectral stacking
	emit(main_logger, "info", "**** Stage 5: spectral stacking ****")
	srcpi_paths = []
	for obsid in obsid_lst:
		obsid_prod_dict = all_prod_dict[obsid]
		for datatype in datatype_lst:
			for evt_fname_prefix, evt_dict in obsid_prod_dict[datatype].items():
				srcpi_paths.append(evt_dict["srcpi"])
	if not srcpi_paths:
		raise ValueError("No source spectra were extracted; cannot run runXstack.")

	##--- input spectra list preparation
	srcpi_fname_lst_file = os.path.join(out_dir,"all_obsid.filelist")
	with open(srcpi_fname_lst_file,"w") as f:
		for srcpi_path in srcpi_paths:
			f.writelines(f"{srcpi_path}\n")

	##--- run Xstack
	stackpi_prefix = os.path.join(stack_dir,f"stack_")
	runXstack_cmd = " ".join([
		"runXstack",
		f"{srcpi_fname_lst_file}",
		"--prefix",f"{stackpi_prefix}",
		"--rsp_weight_method","FLX",
		"--nthreads","20",
		"--same_target",	# activate same-target mode
	])
	runXstack_log = f"{stackpi_prefix}runXstack.log"
	run_cmd(runXstack_cmd,logger=main_logger,logname=runXstack_log)
	

	#--- dump output to json file
	summary_fname = os.path.join(out_dir,"all_obsid.json")
	with open(summary_fname,"w") as f:
		json.dump(all_prod_dict,f,indent=4)


	emit(main_logger, "info", f"**** FXTCOMBINE run successfully! ****")
	emit(main_logger, "info", f"Total exposure: {exp_tot} s")
	emit(main_logger, "info", f"Stacked SRC PI: {os.path.join(stack_dir, 'stack_pi.fits')}")
	emit(main_logger, "info", f"Stacked BKG PI: {os.path.join(stack_dir, 'stack_bkgpi.fits')}")
	emit(main_logger, "info", f"Stacked RMF: {os.path.join(stack_dir, 'stack_rmf.fits')}")
	emit(main_logger, "info", f"Stacked ARF: {os.path.join(stack_dir, 'stack_arf.fits')}")
	emit(main_logger, "info", f"Please check each OBSID product dir for grade plot, and light curve, for sanity check!")
	emit(main_logger, "info", f"Summary of generated files (per OBSID) saved to {summary_fname}")
	emit(main_logger, "info", f"{all_prod_dict}")

	return


def build_parser() -> argparse.ArgumentParser:
	"""Build the command-line parser for ``fxtcombine``.

	Returns
	-------
	argparse.ArgumentParser
		Configured parser for the ``fxtcombine`` CLI.
	"""
	parser = argparse.ArgumentParser(
		description="Combine multi-epoch Einstein Probe FXT observations into stacked images and spectra."
	)
	parser.add_argument("src_dir", help="Source directory containing OBSID subdirectories.")
	parser.add_argument("--ra", type=float, required=True, help="Target right ascension in degrees.")
	parser.add_argument("--dec", type=float, required=True, help="Target declination in degrees.")
	parser.add_argument(
		"--obsid-lst",
		required=True,
		help="Comma-separated OBSID list or a file containing one OBSID per line.",
	)
	parser.add_argument("--out-dir", default="./", help="Output directory. Default: current directory.")
	parser.add_argument("--module", default="a,b", help="Comma-separated module selection. Default: a,b")
	parser.add_argument("--datamode", default="ff", help="Comma-separated datamode selection. Default: ff")
	parser.add_argument(
		"--datatype",
		default="evt",
		help="Comma-separated datatype selection, e.g. evt or evt,fsaevt. Default: evt",
	)
	parser.add_argument("--grade", default="0-12", help="Grade filter passed to xselect. Default: 0-12")
	parser.add_argument("--expr", default="DEFAULT", help="GTI selection expression. Default: DEFAULT")
	parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level for CLI and output log file")
	parser.add_argument("--log-file", type=Path, default=None, help="Optional main log file path; defaults to <out-dir>/log/fxtcombine.log")
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		default=False,
		help="Reuse existing intermediate files when present. Default: rerun all steps.",
	)
	return parser


def main() -> None:
	"""Run the ``fxtcombine`` command-line entry point.

	Returns
	-------
	None
	"""
	args = build_parser().parse_args()
	log_file = args.log_file if args.log_file is not None else Path(args.out_dir) / "log" / "fxtcombine.log"
	cli_logger = build_cli_logger("eFXTDAS.fxtcombine", args.log_level, log_file)
	fxtcombine_pipeline(
		src_dir=args.src_dir,
		ra=args.ra,
		dec=args.dec,
		obsid_lst=args.obsid_lst,
		out_dir=args.out_dir,
		module=args.module,
		datamode=args.datamode,
		datatype=args.datatype,
		grade=args.grade,
		expr=args.expr,
		skip_existing=args.skip_existing,
		logger=cli_logger,
	)


if __name__ == "__main__":
	main()


combine_spec = fxtcombine_pipeline
