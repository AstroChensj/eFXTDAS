"""Spectrum helpers for fxtcombine."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from fxtcombine.config import INSTBKG_BACKSCAL_RELSTD_WARN
from fxtcombine.utils.logger import emit


def stack_instbkg_spectra(instbkg_paths, outfile, logger=None):
	"""Stack instrumental-background PHAs by summing counts and exposure.

	Parameters
	----------
	instbkg_paths : list[str]
		Input instrumental-background PHA paths.
	outfile : str
		Output stacked instrumental-background PHA path.
	logger : logging.Logger | None, optional
		Optional logger used for messages.

	Returns
	-------
	dict[str, float]
		Summary statistics including mean/std of ``BACKSCAL`` and summed exposure.
	"""
	if not instbkg_paths:
		raise ValueError("No instrumental background spectra were provided for stacking.")
	with fits.open(instbkg_paths[0]) as hdul:
		base_hdul = fits.HDUList([hdu.copy() for hdu in hdul])
		channel = np.asarray(base_hdul["SPECTRUM"].data["CHANNEL"])
		counts_sum = np.asarray(base_hdul["SPECTRUM"].data["COUNTS"], dtype=np.float64)
		has_stat_err = "STAT_ERR" in base_hdul["SPECTRUM"].columns.names
		if has_stat_err:
			stat_err_sq = np.square(np.asarray(base_hdul["SPECTRUM"].data["STAT_ERR"], dtype=np.float64))
		else:
			stat_err_sq = None
		exposure_sum = float(base_hdul["SPECTRUM"].header.get("EXPOSURE", 0.0))
		backscal_values = [float(base_hdul["SPECTRUM"].header.get("BACKSCAL", np.nan))]

	for path in instbkg_paths[1:]:
		with fits.open(path) as hdul:
			spec = hdul["SPECTRUM"]
			this_channel = np.asarray(spec.data["CHANNEL"])
			if not np.array_equal(this_channel, channel):
				raise ValueError(f"Instrumental background spectrum channel grid mismatch: {path}")
			counts_sum += np.asarray(spec.data["COUNTS"], dtype=np.float64)
			exposure_sum += float(spec.header.get("EXPOSURE", 0.0))
			backscal_values.append(float(spec.header.get("BACKSCAL", np.nan)))
			if has_stat_err and "STAT_ERR" in spec.columns.names:
				stat_err_sq += np.square(np.asarray(spec.data["STAT_ERR"], dtype=np.float64))
			elif has_stat_err:
				stat_err_sq = None
				has_stat_err = False

	backscal_arr = np.asarray(backscal_values, dtype=np.float64)
	backscal_mean = float(np.nanmean(backscal_arr))
	backscal_std = float(np.nanstd(backscal_arr))
	rel_std = backscal_std / backscal_mean if np.isfinite(backscal_mean) and backscal_mean != 0 else np.inf

	base_hdul["SPECTRUM"].data["COUNTS"] = counts_sum
	if has_stat_err and stat_err_sq is not None:
		base_hdul["SPECTRUM"].data["STAT_ERR"] = np.sqrt(stat_err_sq)
	base_hdul["SPECTRUM"].header["EXPOSURE"] = exposure_sum
	base_hdul["SPECTRUM"].header["BACKSCAL"] = backscal_mean
	base_hdul["SPECTRUM"].header["HDUCLAS2"] = "BKG"
	base_hdul["SPECTRUM"].header["HISTORY"] = f"Stacked {len(instbkg_paths)} instrumental background spectra."
	base_hdul["SPECTRUM"].header["HISTORY"] = f"BACKSCAL mean={backscal_mean:.8g}, std={backscal_std:.8g}."
	base_hdul.writeto(outfile, overwrite=True)

	emit(logger, "info", f"Stacked instrumental background spectrum written to {outfile}")
	emit(logger, "info", f"Instrumental background BACKSCAL mean/std: {backscal_mean:.8g} / {backscal_std:.8g}")
	if rel_std > INSTBKG_BACKSCAL_RELSTD_WARN:
		emit(
			logger,
			"warning",
			f"Instrumental background BACKSCAL relative scatter is {rel_std:.3f}; stacked instrumental background spectrum may not be reliable.",
		)
	return {"backscal_mean": backscal_mean, "backscal_std": backscal_std, "exposure_sum": exposure_sum}
