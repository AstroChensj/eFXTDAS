#!/usr/bin/env python3
"""
Simple calling FXTDAS tasks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import numpy as np
import os
from pathlib import Path
import shlex
import warnings

from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from reproject import reproject_interp
from astropy.wcs import WCS
from tqdm import tqdm

from fxtcombine.config import (
	FXT_POSITION_ERR90_ARCSEC,
	SRC_EXTRACT_RADIUS,
	BKG_EXTRACT_INNER_RADIUS,
	BKG_EXTRACT_OUTER_RADIUS,
)
from fxtcombine.utils.energy import energy_range_suffix, parse_energy_ranges
from fxtcombine.utils.logger import build_cli_logger, build_file_logger, emit
from fxtcombine.utils.cmd import run_cmd
from fxtcombine.utils.fxtchain_simplified import fxtchain_obsid, fxt_extract_spec
from fxtcombine.utils.fxtprep import get_input_files
from fxtcombine.utils.image import reproject_events_xy_to_refwcs
from fxtcombine.utils.spectrum import stack_instbkg_spectra
from fxtpsfgen.mapper import build_stacked_psf_mapper
from fxtsensmap import DEFAULT_ECF, sigma_to_likemin


def _run_stage1_obsid(
	obsid,
	src_dir,
	out_dir,
	datamode,
	module,
	expr,
	grade,
	image_energy_ranges,
	lightcurve_energy_ranges,
	flare_screen,
	flare_threshold_method,
	flare_energy_range,
	flare_binsize,
	flare_min_time_ratio,
	skip_existing,
):
	"""Run Stage-1 preprocessing for one OBSID in an isolated worker."""
	obsid_dir = os.path.join(src_dir, obsid)
	obsid_out_dir = os.path.join(out_dir, obsid, "products")
	os.makedirs(obsid_out_dir, exist_ok=True)
	obsid_log_dir = os.path.join(obsid_out_dir, "log")
	os.makedirs(obsid_log_dir, exist_ok=True)
	obsid_logname = os.path.join(obsid_log_dir, "fxtchain.log")
	obsid_logger = build_file_logger(f"eFXTDAS.fxtcombine.{obsid}.stage1", obsid_logname)
	emit(obsid_logger, "info", f"**** Stage 1 Worker: {obsid} ****")
	emit(obsid_logger, "info", f"Input OBSID directory: {obsid_dir}")
	emit(obsid_logger, "info", f"Output OBSID products directory: {obsid_out_dir}")
	obsid_file_dict = get_input_files(obsid_dir, datamode, module, "evt,fsaevt")
	obsid_prod_dict = fxtchain_obsid(
		obsid_file_dict=obsid_file_dict,
		obsid_out_dir=obsid_out_dir,
		obsid_log_dir=obsid_log_dir,
		expr=expr,
		grade=grade,
		image_energy_ranges=image_energy_ranges,
		lightcurve_energy_ranges=lightcurve_energy_ranges,
		flare_screen=flare_screen,
		flare_threshold_method=flare_threshold_method,
		flare_energy_range=flare_energy_range,
		flare_binsize=flare_binsize,
		flare_min_time_ratio=flare_min_time_ratio,
		skip_existing=skip_existing,
		obsid_logger=obsid_logger,
	)
	return obsid, obsid_prod_dict


def _resolve_sensmap_likemin(likemin=None, sigma=None):
	"""Resolve fxtcombine sensitivity-threshold inputs.

	Parameters
	----------
	likemin : float | None, optional
		Native detection likelihood threshold.
	sigma : float | None, optional
		One-sided Gaussian-equivalent false-alarm threshold.

	Returns
	-------
	float
		Positive finite detection likelihood threshold.
	"""
	if likemin is not None and sigma is not None:
		raise ValueError("sens_likemin and sens_sigma are mutually exclusive.")
	if sigma is not None:
		return sigma_to_likemin(float(sigma))
	value = 6.0 if likemin is None else float(likemin)
	if not np.isfinite(value) or value <= 0.0:
		raise ValueError("sens_likemin must be a finite positive value.")
	return float(value)


def _build_fxtsensmap_command(bkgmap, expmap, psfprod, out, eef, ecf, likemin=None, jobs=1, mask=None, sigma=None):
	"""Build the command used for stacked sensitivity-map generation.

	Parameters
	----------
	bkgmap : str
		Stacked background-map FITS path.
	expmap : str
		Stacked exposure-map FITS path.
	psfprod : str
		Stacked ``fxtpsfgen`` PSF product path.
	out : str
		Output sensitivity-map FITS path.
	eef : float
		Encircled-energy fraction passed to ``fxtsensmap``.
	ecf : float
		Count-rate to flux conversion passed to ``fxtsensmap``.
	likemin : float | None, optional
		Detection likelihood threshold passed to ``fxtsensmap``.
	jobs : int, optional
		Thread workers forwarded to ``fxtsensmap`` if it must compute a radius
		map from the PSF product.
	mask : str | None, optional
		Optional analysis-mask FITS path.
	sigma : float | None, optional
		One-sided Gaussian-equivalent threshold passed to ``fxtsensmap``.

	Returns
	-------
	str
		Shell command string.
	"""
	parts = [
		"fxtsensmap",
		"--bkgmap", f'"{bkgmap}"',
		"--expmap", f'"{expmap}"',
		"--psfprod", f'"{psfprod}"',
		"--eef", f"{float(eef)}",
		"--ecf", f"{float(ecf)}",
	]
	if sigma is not None:
		if likemin is not None:
			raise ValueError("likemin and sigma are mutually exclusive.")
		parts.extend(["--sigma", f"{float(sigma)}"])
	else:
		parts.extend(["--likemin", f"{_resolve_sensmap_likemin(likemin, sigma=None)}"])
	parts.extend(["--jobs", f"{max(int(jobs), 1)}"])
	if mask is not None:
		parts.extend(["--mask", f'"{mask}"'])
	parts.extend(["--out", f'"{out}"'])
	return " ".join(parts)


def _build_quickview_command(stack_dir, out, dpi=100, log_file=None, title=None):
	"""Build the command used for final quick-view QA generation.

	Parameters
	----------
	stack_dir : str
		Stacked-product directory.
	out : str
		Output quick-view figure path.
	dpi : int, optional
		Output figure DPI passed to ``fxtcombine-quickview``.
	log_file : str | None, optional
		Optional quick-view log file path.
	title : str | None, optional
		Optional quick-view figure title.

	Returns
	-------
	str
		Shell command string.
	"""
	parts = [
		"fxtcombine-quickview",
		f'"{stack_dir}"',
		"--out", f'"{out}"',
		"--dpi", f"{int(dpi)}",
	]
	if log_file is not None:
		parts.extend(["--log-file", f'"{log_file}"'])
	if title is not None:
		parts.extend(["--title", shlex.quote(str(title))])
	return " ".join(parts)


def _run_quickview_stage(stack_dir, out, dpi=100, title=None, logger=None):
	"""Run the final quick-view QA command and report success.

	Parameters
	----------
	stack_dir : str
		Stacked-product directory.
	out : str
		Output quick-view figure path.
	dpi : int, optional
		Output figure DPI.
	title : str | None, optional
		Optional quick-view figure title.
	logger : logging.Logger | None, optional
		Logger used for workflow messages.

	Returns
	-------
	bool
		``True`` when quick-view generation succeeds, otherwise ``False``.
	"""
	quickview_log = os.path.join(stack_dir, "quickview.log")
	quickview_cmd = _build_quickview_command(
		stack_dir,
		out,
		dpi=dpi,
		log_file=quickview_log,
		title=title,
	)
	try:
		run_cmd(quickview_cmd, logger=logger)
	except Exception as exc:
		emit(logger, "warning", f"Quick-view generation failed after science products were written: {exc}")
		return False
	emit(logger, "info", f"Quick-view figure written to {out}")
	return True


def fxtcombine_pipeline(
		src_dir,ra=None,dec=None,obsid_lst=None,
		out_dir="./",stack_dir=None,module="a,b",datamode="ff",grade="0-12",expr="DEFAULT",
		image_energy_ranges="0.3:10.0",lightcurve_energy_ranges="0.1:12.0",
		flare_screen=True,flare_threshold_method="robust_iqr",flare_energy_range="0.5:10.0",flare_binsize=20.0,flare_min_time_ratio=0.05,
		mask_expfrac=0.3,jobs=1,srcdet_scales="1,2,4,8,16",srcdet_background_sigma_grid="4,8,16,32,64",
		make_sensmap=True,sens_eef=0.90,sens_ecf=DEFAULT_ECF,sens_likemin=None,sens_sigma=None,
		make_quickview=True,quickview_out=None,quickview_dpi=100,quickview_title=None,
		summary_json=None,srcpi_filelist=None,skip_existing=False,
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
	stack_dir : str | None, optional
		Output directory used only for combined stacked products. When omitted,
		``<out_dir>/stack`` is used.
	module : str, optional
		Comma-separated module selection, for example ``"a,b"``.
	datamode : str, optional
		Comma-separated datamode selection.
	grade : str, optional
		Grade filter passed to the xselect stage.
	expr : str, optional
		GTI selection expression passed to ``fxtgtigen``.
	image_energy_ranges : str | list[tuple[float, float]], optional
		Comma-separated energy ranges in keV used to generate images during
		Stage 1, for example ``"0.3:10.0,1.0:3.0"``. The first range is used by
		default for later stacked source detection.
	lightcurve_energy_ranges : str | list[tuple[float, float]], optional
		Comma-separated energy ranges in keV used to generate light curves during
		Stage 1.
	flare_screen : bool, optional
		Whether to run automatic FF-mode flare screening from FSAEVT data.
	flare_threshold_method : str, optional
		Threshold method passed to ``fxtbkgoptrate`` for FSA flare screening.
	flare_energy_range : str | tuple[float, float], optional
		Energy range in keV used for the flare-screening light curve.
	flare_binsize : float, optional
		Bin size in seconds used for flare screening.
	flare_min_time_ratio : float, optional
		Minimum retained exposure fraction accepted by the flare-screening
		threshold optimizer.
	mask_expfrac : float, optional
		Minimum stacked exposure, expressed as a fraction of the maximum stacked
		exposure, required for a pixel to remain valid in the stacked mask passed
		to ``fxtsrcdet``.
	jobs : int, optional
		Number of parallel OBSID workers used in Stage 1. Each worker processes
		one OBSID in its own output directory. ``1`` keeps serial execution.
	srcdet_scales : str | list[float], optional
		Wavelet scales in pixels forwarded to ``fxtsrcdet`` for stacked source
		detection.
	srcdet_background_sigma_grid : str | list[float], optional
		Gaussian smoothing scales in pixels forwarded to ``fxtsrcdet`` for its
		adaptive background model.
	make_sensmap : bool, optional
		Whether to generate ``stack_sensmap.fits`` after spectral stacking.
	sens_eef : float, optional
		Encircled-energy fraction used by ``fxtsensmap``. Default is ``0.90``.
	sens_ecf : float, optional
		Count-rate to flux conversion in ``ct s^-1 / (erg cm^-2 s^-1)`` passed
		to ``fxtsensmap``.
	sens_likemin : float | None, optional
		Detection likelihood threshold passed to ``fxtsensmap``. Defaults to
		``6.0`` when neither ``sens_likemin`` nor ``sens_sigma`` is supplied.
	sens_sigma : float | None, optional
		One-sided Gaussian-equivalent false-alarm threshold passed to
		``fxtsensmap`` as ``--sigma``.
	make_quickview : bool, optional
		Whether to generate ``quickview.png`` as a final QA product.
	quickview_out : str | None, optional
		Output quick-view figure path. Defaults to ``<stack_dir>/quickview.png``.
	quickview_dpi : int, optional
		Output quick-view figure DPI.
	quickview_title : str | None, optional
		Optional quick-view figure title.
	summary_json : str | None, optional
		Path of the summary JSON file. When omitted,
		``<stack_dir>/all_obsid.json`` is used.
	srcpi_filelist : str | None, optional
		Path of the source-spectrum file list passed to ``runXstack``. When
		omitted, ``<stack_dir>/all_obsid.filelist`` is used.
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
	stack_dir = os.path.abspath(stack_dir) if stack_dir is not None else os.path.join(out_dir, "stack")
	summary_json = os.path.abspath(summary_json) if summary_json is not None else os.path.join(stack_dir, "all_obsid.json")
	srcpi_filelist = os.path.abspath(srcpi_filelist) if srcpi_filelist is not None else os.path.join(stack_dir, "all_obsid.filelist")
	quickview_out = os.path.abspath(quickview_out) if quickview_out is not None else os.path.join(stack_dir, "quickview.png")
	if isinstance(image_energy_ranges, str):
		image_energy_ranges = parse_energy_ranges(image_energy_ranges, default=[(0.3, 10.0)])
	else:
		image_energy_ranges = list(image_energy_ranges)
	if isinstance(lightcurve_energy_ranges, str):
		lightcurve_energy_ranges = parse_energy_ranges(lightcurve_energy_ranges, default=[(0.1, 12.0)])
	else:
		lightcurve_energy_ranges = list(lightcurve_energy_ranges)
	if isinstance(flare_energy_range, str):
		flare_energy_range = parse_energy_ranges(flare_energy_range, default=[(0.5, 10.0)])[0]
	else:
		flare_energy_range = tuple(flare_energy_range)

	#--- define logger
	os.makedirs(out_dir,exist_ok=True)
	log_dir = os.path.join(out_dir,"log")
	os.makedirs(log_dir,exist_ok=True)
	main_logname = os.path.join(log_dir, "fxtcombine.log")
	main_logger = logger if logger is not None else build_file_logger("eFXTDAS.fxtcombine", main_logname)
	emit(main_logger, "info", "==================================")
	emit(main_logger, "info", f"**** Welcome to FXTCOMBINE ! ****")
	emit(main_logger, "info", "==================================")
	emit(main_logger, "info", f"Source directory is: {src_dir}")
	emit(main_logger, "info", f"Per-OBSID output directory is: {out_dir}")
	emit(main_logger, "info", f"Stacked output directory is: {stack_dir}")
	emit(main_logger, "info", f"Summary JSON path is: {summary_json}")
	emit(main_logger, "info", f"Source-spectrum file list path is: {srcpi_filelist}")
	emit(main_logger, "info", f"Source coordinate is: ICRS({ra}, {dec})")
	emit(main_logger, "info", f"Image energy ranges are: {image_energy_ranges}")
	emit(main_logger, "info", f"Light-curve energy ranges are: {lightcurve_energy_ranges}")
	emit(main_logger, "info", f"Flare screening enabled: {flare_screen}")
	emit(main_logger, "info", f"Flare screening threshold method is: {flare_threshold_method}")
	emit(main_logger, "info", f"Flare screening energy range is: {flare_energy_range}")
	emit(main_logger, "info", f"Flare screening bin size is: {flare_binsize}")
	emit(main_logger, "info", f"Flare screening min time ratio is: {flare_min_time_ratio}")
	emit(main_logger, "info", f"Stacked-mask minimum exposure fraction is: {mask_expfrac}")
	emit(main_logger, "info", f"Stage 1 parallel workers are: {jobs}")
	emit(main_logger, "info", f"fxtsrcdet wavelet scales are: {srcdet_scales}")
	emit(main_logger, "info", f"fxtsrcdet background sigma grid is: {srcdet_background_sigma_grid}")
	emit(main_logger, "info", f"Stacked sensitivity-map generation enabled: {make_sensmap}")
	emit(main_logger, "info", f"Quick-view generation enabled: {make_quickview}")
	if make_quickview:
		emit(main_logger, "info", f"Quick-view output path is: {quickview_out}")
		emit(main_logger, "info", f"Quick-view DPI is: {quickview_dpi}")
		if quickview_title is not None:
			emit(main_logger, "info", f"Quick-view title is: {quickview_title}")
	sens_likemin_resolved = _resolve_sensmap_likemin(sens_likemin, sens_sigma) if make_sensmap else None
	if make_sensmap:
		emit(main_logger, "info", f"fxtsensmap EEF is: {sens_eef}")
		emit(main_logger, "info", f"fxtsensmap ECF is: {sens_ecf}")
		if sens_sigma is not None:
			emit(main_logger, "info", f"fxtsensmap sigma is: {sens_sigma} (one-sided Gaussian equivalent; likemin={sens_likemin_resolved})")
		else:
			emit(main_logger, "info", f"fxtsensmap likemin is: {sens_likemin_resolved}")


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


	#==============================================================================
	#--- initial iterate: to get clean events & image & exposure map for each obsid
	#==============================================================================
	emit(main_logger, "info", f"**** Stage 1: initial iterate to get clean events & exposure map for each OBSID ****")
	all_prod_dict = {}
	jobs = max(int(jobs), 1)
	##--- normal mode: serialized running, one by one
	if jobs == 1 or len(obsid_lst) == 1:
		for obsid in obsid_lst:
			emit(main_logger, "info", f"Processing {obsid} serially ...")
			obsid_result = _run_stage1_obsid(
				obsid=obsid,
				src_dir=src_dir,
				out_dir=out_dir,
				datamode=datamode,
				module=module,
				expr=expr,
				grade=grade,
				image_energy_ranges=image_energy_ranges,
				lightcurve_energy_ranges=lightcurve_energy_ranges,
				flare_screen=flare_screen,
				flare_threshold_method=flare_threshold_method,
				flare_energy_range=flare_energy_range,
				flare_binsize=flare_binsize,
				flare_min_time_ratio=flare_min_time_ratio,
				skip_existing=skip_existing,
			)
			obsid, obsid_prod_dict = obsid_result
			all_prod_dict[obsid] = obsid_prod_dict
	##--- parallel mode: multiple OBSIDs processed simultaneously to speed up this stage
	else:
		max_workers = min(jobs, len(obsid_lst))
		emit(main_logger, "info", f"Running Stage 1 in parallel with {max_workers} OBSID worker(s).")
		with ProcessPoolExecutor(max_workers=max_workers) as executor:
			future_map = {
				executor.submit(
					_run_stage1_obsid,
					obsid,
					src_dir,
					out_dir,
					datamode,
					module,
					expr,
					grade,
					image_energy_ranges,
					lightcurve_energy_ranges,
					flare_screen,
					flare_threshold_method,
					flare_energy_range,
					flare_binsize,
					flare_min_time_ratio,
					skip_existing,
				): obsid
				for obsid in obsid_lst
			}
			for future in tqdm(as_completed(future_map), total=len(future_map), desc="Stage 1 OBSIDs"):
				obsid = future_map[future]
				emit(main_logger, "info", f"Collecting Stage 1 results for {obsid} ...")
				obsid_name, obsid_prod_dict = future.result()
				all_prod_dict[obsid_name] = obsid_prod_dict
				emit(main_logger, "info", f"Finished Stage 1 for {obsid_name}.")


	#=================================================
	#--- stack all images, expmaps and PSF products
	#=================================================
	emit(main_logger, "info", "**** Stage 2: stacking evt images, exposure maps, and PSF products ****")
	os.makedirs(stack_dir,exist_ok=True)
	detection_image_suffix = energy_range_suffix(image_energy_ranges[0])
	detect_emin, detect_emax = image_energy_ranges[0]
	emit(main_logger, "info", "Stacking EVT products across all streams ...")
	clevt_fname_lst = []
	exp_fname_lst = []
	exp_lst = []
	detect_mask_fname_lst = []
	img_fname_map = { energy_range_suffix(energy_range): [] for energy_range in image_energy_ranges }
	image_band_channels_map = {}

	for obsid, obsid_prod_dict in all_prod_dict.items():
		for stream_key, prod in obsid_prod_dict.items():
			clevt_fname_lst.append(prod["evt_clevt"])
			for image_key in img_fname_map:
				img_fname_map[image_key].append(prod["images"][image_key])
			if not image_band_channels_map:
				image_band_channels_map = dict(prod["image_band_channels"])
			exp_fname_lst.append(prod["vexpmap"])
			exp_lst.append(prod["exp"])
			detect_mask_fname_lst.append(prod["detection_mask"])

	clevt_fname_lst = np.array(clevt_fname_lst)
	exp_fname_lst = np.array(exp_fname_lst)
	exp_lst = np.array(exp_lst)
	detect_mask_fname_lst = np.array(detect_mask_fname_lst)
	if len(clevt_fname_lst) == 0:
		raise ValueError("No EVT products were produced during Stage 1. Check your OBSID selection and inputs.")

	##--- sort according to expo
	idx_sort = np.argsort(exp_lst)[::-1]
	clevt_fname_lst = clevt_fname_lst[idx_sort]
	exp_fname_lst = exp_fname_lst[idx_sort]
	exp_lst = exp_lst[idx_sort]
	detect_mask_fname_lst = detect_mask_fname_lst[idx_sort]
	for image_key, img_list in img_fname_map.items():
		img_fname_map[image_key] = np.array(img_list)[idx_sort]
	exp_tot = np.sum(exp_lst)

	##--- logging all files
	emit(main_logger, "info", f"You have the following clean events files: {clevt_fname_lst}")
	for image_key, img_fname_lst in img_fname_map.items():
		emit(main_logger, "info", f"You have the following images for {image_key}: {img_fname_lst}")
	emit(main_logger, "info", f"You have the following vign-exposure maps: {exp_fname_lst}")
	emit(main_logger, "info", f"You have the following per-OBSID detection masks: {detect_mask_fname_lst}")
	emit(main_logger, "info", f"Their corresponding exposures are: {exp_lst}")

	##--- choose a common reference frame from the first requested image band
	refimg_fname_lst = img_fname_map[detection_image_suffix]
	with warnings.catch_warnings():	# to suppress common warnings, so output log is cleaner and readable
		warnings.simplefilter("ignore", VerifyWarning)
		warnings.simplefilter("ignore", FITSFixedWarning)
		with fits.open(refimg_fname_lst[0]) as hdu:
			refimg = hdu[0]
			refimg_wcs = WCS(refimg.header)
			refimg_shape = refimg.data.shape
	emit(main_logger, "info", f"The reference frame is {refimg_fname_lst[0]}.")
	emit(main_logger, "info", f"Reference WCS is {refimg_wcs}.")

	with warnings.catch_warnings():	# to suppress common warnings, so output log is cleaner and readable
		warnings.simplefilter("ignore", VerifyWarning)
		warnings.simplefilter("ignore", FITSFixedWarning)
		with fits.open(exp_fname_lst[0]) as hdu:
			refexp = hdu[0]
			refexp_wcs = WCS(refexp.header)
			refexp_shape = refexp.data.shape
			exp_sum = np.zeros(refexp_shape)
	assert refimg_shape == refexp_shape, f"The image and exposure should have same shape, but now gets {refimg_shape} and {refexp_shape}!"

	##--- reproject and stack images for each requested image band
	cts_sum_map = {}
	for energy_range in image_energy_ranges:
		image_key = energy_range_suffix(energy_range)
		chan_lo, chan_hi = image_band_channels_map[image_key]
		cts_sum = np.zeros(refimg_shape)
		emit(main_logger, "info", f"Reprojecting and stacking images for {image_key} ...")
		for i in range(len(clevt_fname_lst)):
			clevt_fname = clevt_fname_lst[i]
			img_fname = img_fname_map[image_key][i]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", VerifyWarning)
				with fits.open(clevt_fname) as hdu:
					clevt_data = hdu[1].data
			mask_channel = (clevt_data["CHANNEL"] >= chan_lo) & (clevt_data["CHANNEL"] <= chan_hi)
			clevt_x = clevt_data["X"][mask_channel]
			clevt_y = clevt_data["Y"][mask_channel]
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", VerifyWarning)
				warnings.simplefilter("ignore", FITSFixedWarning)
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
		cts_sum_fname = os.path.join(stack_dir,f"{image_key}_stack_cts.fits")
		fits.writeto(cts_sum_fname,cts_sum,refimg.header,overwrite=True)
		emit(main_logger, "info", f"Stacked count image written to {cts_sum_fname}")
		cts_sum_map[image_key] = cts_sum.copy()
		if image_key == detection_image_suffix:	# the first channel range is used for source detection and later spectral extraction; save a copy as default
			default_cts_sum_fname = cts_sum_fname
			legacy_cts_sum_fname = os.path.join(stack_dir,"stack_cts.fits")
			fits.writeto(legacy_cts_sum_fname,cts_sum,refimg.header,overwrite=True)
			emit(main_logger, "info", f"Default stacked count image also written to {legacy_cts_sum_fname}")

	##--- reproject and stack expmap
	emit(main_logger, "info", f"Reprojecting and stacking exposure maps ...")
	for exp_fname_i in exp_fname_lst:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", VerifyWarning)
			warnings.simplefilter("ignore", FITSFixedWarning)
			with fits.open(exp_fname_i) as hdu:
				exp_i = hdu[0]
				data_i = exp_i.data
				wcs_i  = WCS(exp_i.header)
		###--- flux/count-conserving reprojection
		data_i_reproj,footprint_i = reproject_interp((data_i,wcs_i),refexp_wcs,shape_out=refexp_shape)
		###--- footprint is 0..1 overlap fraction; use it to ignore empty pixels
		m = np.isfinite(data_i_reproj) & (footprint_i > 0)
		exp_sum[m] += data_i_reproj[m]
	exp_sum_fname = os.path.join(stack_dir,"stack_expmap.fits")
	fits.writeto(exp_sum_fname,exp_sum,refexp.header,overwrite=True)
	emit(main_logger, "info", f"Stacked exposure map written to {exp_sum_fname}")
	
	##--- build one stacked PSF product for the default detection band
	psfprod_fname_lst = []
	weight_fname_lst = []
	for obsid in obsid_lst:
		obsid_prod_dict = all_prod_dict[obsid]
		for _stream_key, prod in obsid_prod_dict.items():
			psfprod_path = prod.get("psfprod")
			if psfprod_path is None:
				continue
			psfprod_fname_lst.append(psfprod_path)
			weight_fname_lst.append(prod["vexpmap"])
	if not psfprod_fname_lst:
		raise ValueError("No per-observation PSF products were produced during Stage 1.")
	with fits.open(default_cts_sum_fname) as hdu:
		ref_header = hdu[0].header.copy()
	stack_psfprod_default_fname = os.path.join(stack_dir, "stack_psfprod.fits")
	stacked_psf = build_stacked_psf_mapper(psfprod_fname_lst, weight_fname_lst, ref_header)
	stacked_psf.write(stack_psfprod_default_fname)
	emit(main_logger, "info", f"Stacked PSF product written to {stack_psfprod_default_fname}")
	
	##--- obtain stacked rate image for each image band
	finite_exp = exp_sum[np.isfinite(exp_sum)]
	max_exp = float(np.max(finite_exp)) if finite_exp.size else 0.0
	exp_cut = float(mask_expfrac) * max_exp
	valid_exp = np.isfinite(exp_sum) & (exp_sum >= exp_cut)
	for image_key, cts_sum in cts_sum_map.items():
		rate_sum = np.zeros_like(cts_sum, dtype=np.float64)
		valid = valid_exp & np.isfinite(cts_sum) & (~np.isinf(cts_sum))
		rate_sum[valid] = cts_sum[valid] / exp_sum[valid]
		rate_sum_fname = os.path.join(stack_dir,f"{image_key}_stack_rate.fits")
		fits.writeto(rate_sum_fname,rate_sum,refimg.header,overwrite=True)
		emit(main_logger, "info", f"Stacked rate image written to {rate_sum_fname}")
		if image_key == detection_image_suffix:
			legacy_rate_sum_fname = os.path.join(stack_dir,"stack_rate.fits")
			fits.writeto(legacy_rate_sum_fname,rate_sum,refimg.header,overwrite=True)
			emit(main_logger, "info", f"Default stacked rate image also written to {legacy_rate_sum_fname}")
	
	stack_expmap_default_fname = exp_sum_fname
	stack_image_default_fname = default_cts_sum_fname
	##--- generate stacked analysis mask for the default detection band
	stack_mask_fname = os.path.join(stack_dir, "stack_mask.fits")
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", VerifyWarning)
		warnings.simplefilter("ignore", FITSFixedWarning)
		with fits.open(stack_image_default_fname) as hdu:
			stack_image_data = np.asarray(hdu[0].data, dtype=np.float64)
			stack_image_header = hdu[0].header.copy()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", VerifyWarning)
		warnings.simplefilter("ignore", FITSFixedWarning)
		with fits.open(stack_expmap_default_fname) as hdu:
			stack_exp_data = np.asarray(hdu[0].data, dtype=np.float64)
	mask_support = np.zeros(refimg_shape, dtype=np.float64)
	for detect_mask_fname_i in detect_mask_fname_lst:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", VerifyWarning)
			warnings.simplefilter("ignore", FITSFixedWarning)
			with fits.open(detect_mask_fname_i) as hdu:
				mask_data_i = np.asarray(hdu[0].data, dtype=np.float64)
				mask_wcs_i = WCS(hdu[0].header)
		mask_reproj_i, mask_footprint_i = reproject_interp((mask_data_i, mask_wcs_i), refimg_wcs, shape_out=refimg_shape)
		valid_i = np.isfinite(mask_reproj_i) & (mask_footprint_i > 0) & (mask_reproj_i > 0.0)
		mask_support[valid_i] += mask_reproj_i[valid_i]
	finite_exp = stack_exp_data[np.isfinite(stack_exp_data)]
	max_exp = float(np.max(finite_exp)) if finite_exp.size else 0.0
	exp_cut = float(mask_expfrac) * max_exp
	stack_mask = (
		np.isfinite(stack_image_data)
		& np.isfinite(stack_exp_data)
		& (stack_exp_data >= exp_cut)
		& (mask_support > 0.0)
	)
	fits.writeto(stack_mask_fname, stack_mask.astype(np.uint8), stack_image_header, overwrite=True)
	emit(
		main_logger,
		"info",
		f"Stacked analysis mask written to {stack_mask_fname} with threshold {exp_cut:.6g} ({mask_expfrac:.3f} of max exposure {max_exp:.6g})",
	)
	emit(main_logger, "info", f"Mask valid pixels: {int(np.count_nonzero(stack_mask))} / {int(stack_mask.size)}")
	emit(main_logger, "info", f"Stacked mask support pixels: {int(np.count_nonzero(mask_support > 0.0))} / {int(mask_support.size)}")


	#=============================================
	#--- source detection and region file creation
	#=============================================
	emit(main_logger, "info", "**** Stage 3: source detection and region file creation ****")
	stack_image_fname = stack_image_default_fname
	stack_expmap_fname = stack_expmap_default_fname
	srcdet_src_fname = os.path.join(stack_dir, "stack_src.fits")
	srcdet_reg_fname = os.path.join(stack_dir, "stack_src.reg")
	srcdet_bkg_fname = os.path.join(stack_dir, "stack_bkgmap.fits")
	srcdet_log = os.path.join(stack_dir, "srcdet.log")
	srcdet_cmd = " ".join([
		"fxtsrcdet",
		f'"{stack_image_fname}"',
		"--expmap", f'"{stack_expmap_fname}"',
		"--mask", f'"{stack_mask_fname}"',
		"--psfprod", f'"{stack_psfprod_default_fname}"',
		"--mission", "ep-fxt",
		"--emin", f"{detect_emin}",
		"--emax", f"{detect_emax}",
		"--scales", f'"{srcdet_scales}"',
		"--background-sigma-grid", f'"{srcdet_background_sigma_grid}"',
		"--out", f'"{srcdet_src_fname}"',
		"--regfile", f'"{srcdet_reg_fname}"',
		"--save-bkgmap", f'"{srcdet_bkg_fname}"',
		"--log-file", f'"{srcdet_log}"',
	])
	run_cmd(srcdet_cmd, logger=main_logger, logname=srcdet_log)
	emit(main_logger, "info", f"Stacked source catalog written to {srcdet_src_fname}")
	emit(main_logger, "info", f"Stacked source region file written to {srcdet_reg_fname}")
	emit(main_logger, "info", f"Stacked background map written to {srcdet_bkg_fname}")

	emit(main_logger, "info", "Generating src & bkg regions with fxtregions ...")
	src_reg_fname = os.path.join(stack_dir, "target_src.reg")
	bkg_reg_fname = os.path.join(stack_dir, "target_bkg.reg")
	fxtregions_log = os.path.join(stack_dir, "fxtregions.log")
	fxtregions_cmd = " ".join([
		"fxtregions",
		f'"{stack_image_fname}"',
		f'"{srcdet_src_fname}"',
		"--bkgmap", f'"{srcdet_bkg_fname}"',
		"--ra", f"{float(ra)}",
		"--dec", f"{float(dec)}",
		"--mission", "ep-fxt",
		"--emin", f"{detect_emin}",
		"--emax", f"{detect_emax}",
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


	#============================================================================
	#--- second iterate: spectral extraction, with contaminating sources excluded
	#============================================================================
	emit(main_logger, "info", "**** Stage 4: Iterate again on each obsid to extract spectra ****")
	for obsid in obsid_lst:
		emit(main_logger, "info", f"Processing {obsid} ...")
		obsid_out_dir = os.path.join(out_dir,obsid,"products")
		obsid_log_dir = os.path.join(obsid_out_dir,"log")
		obsid_logname = os.path.join(obsid_log_dir, "fxtchain.log")
		obsid_logger = build_file_logger(f"eFXTDAS.fxtcombine.{obsid}.stage4", obsid_logname)
		obsid_prod_dict = all_prod_dict[obsid]
		obsid_prod_dict = fxt_extract_spec(
			obsid_prod_dict,
			src_reg_fname,bkg_reg_fname,
			obsid_out_dir,obsid_log_dir,
			skip_existing=skip_existing,
			obsid_logger=obsid_logger,
		)
		all_prod_dict[obsid] = obsid_prod_dict
	

	#=====================
	#--- spectral stacking
	#=====================
	emit(main_logger, "info", "**** Stage 5: spectral stacking ****")
	srcpi_paths = []
	instbkg_paths = []
	for obsid in obsid_lst:
		obsid_prod_dict = all_prod_dict[obsid]
		for stream_key, prod in obsid_prod_dict.items():
			srcpi_paths.append(prod["srcpi"])
			instbkgpi = prod.get("instbkgpi")
			if instbkgpi:
				instbkg_paths.append(instbkgpi)
	if not srcpi_paths:
		raise ValueError("No source spectra were extracted; cannot run runXstack.")

	srcpi_fname_lst_file = srcpi_filelist
	with open(srcpi_fname_lst_file,"w") as f:
		for srcpi_path in srcpi_paths:
			f.writelines(f"{srcpi_path}\n")

	stackpi_prefix = os.path.join(stack_dir,f"stack_")
	runXstack_cmd = " ".join([
		"runXstack",
		f"{srcpi_fname_lst_file}",
		"--prefix",f"{stackpi_prefix}",
		"--rsp_weight_method","FLX",
		"--nthreads","20",
		"--same_target",
	])
	run_cmd(runXstack_cmd,logger=main_logger)

	if instbkg_paths:
		stack_instbkg_fname = os.path.join(stack_dir, "stack_instbkgpi.fits")
		stack_instbkg_spectra(instbkg_paths, stack_instbkg_fname, logger=main_logger)
	else:
		emit(main_logger, "warning", "No per-OBSID instrumental background spectra were found to stack.")
	

	#===============================
	#--- sensitivity-map generation
	#===============================
	stack_sensmap_fname = os.path.join(stack_dir, "stack_sensmap.fits")
	if make_sensmap:
		emit(main_logger, "info", "**** Stage 6: sensitivity map generation ****")
		fxtsensmap_log = os.path.join(stack_dir, "fxtsensmap.log")
		fxtsensmap_cmd = _build_fxtsensmap_command(
			srcdet_bkg_fname,
			stack_expmap_default_fname,
			stack_psfprod_default_fname,
			stack_sensmap_fname,
			sens_eef,
			sens_ecf,
			sens_likemin_resolved if sens_sigma is None else None,
			jobs=jobs,
			mask=stack_mask_fname,
			sigma=sens_sigma,
		)
		run_cmd(fxtsensmap_cmd, logger=main_logger, logname=fxtsensmap_log)
		emit(main_logger, "info", f"Stacked sensitivity map written to {stack_sensmap_fname}")
	else:
		emit(main_logger, "info", "Skipping stacked sensitivity-map generation.")

	#===============================
	#--- quick-view QA generation
	#===============================
	quickview_success = False
	if make_quickview:
		emit(main_logger, "info", "**** Stage 7: quick-view QA generation ****")
		quickview_success = _run_quickview_stage(
			stack_dir,
			quickview_out,
			dpi=quickview_dpi,
			title=quickview_title,
			logger=main_logger,
		)
	else:
		emit(main_logger, "info", "Skipping quick-view QA generation.")


	#--- dump output to json file
	summary_fname = summary_json
	with open(summary_fname,"w") as f:
		json.dump(all_prod_dict,f,indent=4)


	emit(main_logger, "info", f"**** FXTCOMBINE run successfully! ****")
	emit(main_logger, "info", f"Total exposure: {exp_tot} s")
	emit(main_logger, "info", f"Stacked SRC PI: {os.path.join(stack_dir, 'stack_pi.fits')}")
	emit(main_logger, "info", f"Stacked BKG PI: {os.path.join(stack_dir, 'stack_bkgpi.fits')}")
	emit(main_logger, "info", f"Stacked RMF: {os.path.join(stack_dir, 'stack_rmf.fits')}")
	emit(main_logger, "info", f"Stacked ARF: {os.path.join(stack_dir, 'stack_arf.fits')}")
	if instbkg_paths:
		emit(main_logger, "info", f"Stacked instrumental background PI: {os.path.join(stack_dir, 'stack_instbkgpi.fits')}")
	if make_sensmap:
		emit(main_logger, "info", f"Stacked sensitivity map: {stack_sensmap_fname}")
	if quickview_success:
		emit(main_logger, "info", f"Quickview figure: {quickview_out}")
	elif make_quickview:
		emit(main_logger, "warning", f"Quickview figure was not generated successfully: {quickview_out}")
	else:
		emit(main_logger, "info", "Quickview figure generation disabled.")
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
	parser.add_argument("--stack-dir", default=None, help="Optional stacked-product directory. Default: <out-dir>/stack")
	parser.add_argument("--summary-json", default=None, help="Optional summary JSON path. Default: <stack-dir>/all_obsid.json")
	parser.add_argument("--srcpi-filelist", default=None, help="Optional runXstack source-spectrum file list path. Default: <stack-dir>/all_obsid.filelist")
	parser.add_argument("--module", default="a,b", help="Comma-separated module selection. Default: a,b")
	parser.add_argument("--datamode", default="ff", help="Comma-separated datamode selection. Default: ff")
	parser.add_argument("--grade", default="0-12", help="Grade filter passed to xselect. Default: 0-12")
	parser.add_argument("--expr", default="DEFAULT", help="GTI selection expression. Default: DEFAULT")
	parser.add_argument(
		"--image-energy-ranges",
		default="0.3:10.0,10.0:12.0",
		help="Comma-separated energy ranges in keV used to generate Stage-1 images, e.g. 0.3:10.0,1.0:3.0. The first range is used for stacked source detection by default.",
	)
	parser.add_argument(
		"--lightcurve-energy-ranges",
		default="0.1:12.0",
		help="Comma-separated energy ranges in keV used to generate Stage-1 light curves, e.g. 0.1:12.0,1.0:3.0.",
	)
	parser.add_argument(
		"--disable-flare-screen",
		action="store_true",
		default=False,
		help="Disable automatic FF-mode FSA-based flare screening during Stage 1.",
	)
	parser.add_argument(
		"--flare-threshold-method",
		choices=["robust_iqr", "snr"],
		default="robust_iqr",
		help="Threshold method passed to fxtbkgoptrate for FSA flare screening. Default: robust_iqr",
	)
	parser.add_argument(
		"--flare-energy-range",
		default="0.5:10.0",
		help="Energy range in keV used for the flare-screening light curve. Default: 0.5:10.0",
	)
	parser.add_argument(
		"--flare-binsize",
		type=float,
		default=20.0,
		help="Flare-screening light-curve bin size in seconds. Default: 20",
	)
	parser.add_argument(
		"--flare-min-time-ratio",
		type=float,
		default=0.05,
		help="Minimum retained exposure fraction accepted by the flare-screening optimizer. Default: 0.05",
	)
	parser.add_argument(
		"--mask-expfrac",
		type=float,
		default=0.3,
		help="Minimum stacked exposure fraction, relative to the stacked exposure maximum, required to keep a pixel in the stacked analysis mask passed to fxtsrcdet. Default: 0.3",
	)
	parser.add_argument(
		"--jobs",
		type=int,
		default=1,
		help="Number of parallel OBSID workers used in Stage 1. Default: 1",
	)
	parser.add_argument(
		"--srcdet-scales",
		default="1,2,4,8,16",
		help="Wavelet scales in pixels forwarded to fxtsrcdet for stacked source detection. Default: '1,2,4,8,16'",
	)
	parser.add_argument(
		"--srcdet-background-sigma-grid",
		default="4,8,16,32,64",
		help="Gaussian smoothing scales in pixels forwarded to fxtsrcdet for adaptive background modeling. Default: '4,8,16,32,64'",
	)
	parser.add_argument(
		"--disable-sensmap",
		action="store_true",
		default=False,
		help="Disable automatic stacked sensitivity-map generation after spectral stacking.",
	)
	parser.add_argument(
		"--sens-eef",
		type=float,
		default=0.90,
		help="Encircled-energy fraction passed to fxtsensmap. Default: 0.90",
	)
	parser.add_argument(
		"--sens-ecf",
		type=float,
		default=DEFAULT_ECF,
		help=f"Count-rate to flux conversion passed to fxtsensmap in ct/s per erg/cm2/s. Default: {DEFAULT_ECF:.5g}",
	)
	sens_threshold_group = parser.add_mutually_exclusive_group()
	sens_threshold_group.add_argument(
		"--sens-likemin",
		type=float,
		default=None,
		help="Detection likelihood threshold passed to fxtsensmap. Default: 6.0",
	)
	sens_threshold_group.add_argument(
		"--sens-sigma",
		type=float,
		default=None,
		help="One-sided Gaussian-equivalent false-alarm threshold passed to fxtsensmap as --sigma.",
	)
	parser.add_argument(
		"--disable-quickview",
		action="store_true",
		default=False,
		help="Disable automatic quick-view QA figure generation after stacked products are written.",
	)
	parser.add_argument(
		"--quickview-out",
		default=None,
		help="Optional quick-view figure path. Default: <stack-dir>/quickview.png",
	)
	parser.add_argument(
		"--quickview-dpi",
		type=int,
		default=100,
		help="Quick-view output figure DPI. Default: 100",
	)
	parser.add_argument(
		"--quickview-title",
		default=None,
		help="Optional title shown above the quick-view QA figure.",
	)
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
		stack_dir=args.stack_dir,
		summary_json=args.summary_json,
		srcpi_filelist=args.srcpi_filelist,
		module=args.module,
		datamode=args.datamode,
		grade=args.grade,
		expr=args.expr,
		image_energy_ranges=args.image_energy_ranges,
		lightcurve_energy_ranges=args.lightcurve_energy_ranges,
		flare_screen=not args.disable_flare_screen,
		flare_threshold_method=args.flare_threshold_method,
		flare_energy_range=args.flare_energy_range,
		flare_binsize=args.flare_binsize,
		flare_min_time_ratio=args.flare_min_time_ratio,
		mask_expfrac=args.mask_expfrac,
		jobs=args.jobs,
		srcdet_scales=args.srcdet_scales,
		srcdet_background_sigma_grid=args.srcdet_background_sigma_grid,
		make_sensmap=not args.disable_sensmap,
		sens_eef=args.sens_eef,
		sens_ecf=args.sens_ecf,
		sens_likemin=args.sens_likemin,
		sens_sigma=args.sens_sigma,
		make_quickview=not args.disable_quickview,
		quickview_out=args.quickview_out,
		quickview_dpi=args.quickview_dpi,
		quickview_title=args.quickview_title,
		skip_existing=args.skip_existing,
		logger=cli_logger,
	)


if __name__ == "__main__":
	main()
