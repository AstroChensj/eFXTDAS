"""Helpers for parsing energy-band selections and mapping them to PI channels."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_energy_ranges(range_spec: str | None, *, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
	"""Parse a comma-separated list of inclusive energy ranges in keV."""
	if range_spec is None:
		return list(default)
	text = str(range_spec).strip()
	if not text:
		return list(default)
	energy_ranges = []
	for item in text.split(","):
		piece = item.strip()
		if not piece:
			continue
		if ":" not in piece:
			raise ValueError(f"Invalid energy range '{piece}'. Use lo:hi syntax in keV.")
		emin_text, emax_text = piece.split(":", 1)
		emin = float(emin_text)
		emax = float(emax_text)
		if emin < 0 or emax <= 0 or emin > emax:
			raise ValueError(f"Invalid energy range '{piece}'. Require 0 <= emin <= emax.")
		energy_ranges.append((emin, emax))
	if not energy_ranges:
		raise ValueError("At least one valid energy range is required.")
	return energy_ranges


def energy_range_suffix(energy_range: tuple[float, float]) -> str:
	"""Return a stable filename suffix for one energy range in keV."""
	emin, emax = energy_range
	return f"e{int(round(1000.0 * emin)):05d}_{int(round(1000.0 * emax)):05d}"


@lru_cache(maxsize=8)
def _load_e2pi_edges(module: str) -> tuple[np.ndarray, np.ndarray]:
	"""Load E2PI energy-bin edges for one EP-FXT module."""
	caldb = os.environ.get("CALDB")
	if not caldb:
		raise RuntimeError("CALDB is not set; cannot resolve EP-FXT E2PI calibration.")
	module = str(module).strip().lower()
	instrument_dir = Path(caldb).expanduser() / "data" / "ep" / "fxt" / "bcf" / "instrument"
	candidates = sorted(instrument_dir.glob(f"fxt_{module}_energy_con_pi*.fits"))
	if not candidates:
		raise FileNotFoundError(f"Could not find E2PI calibration file for module '{module}' under {instrument_dir}")
	with fits.open(candidates[-1]) as hdul:
		data = hdul[1].data
		elow = np.asarray(data["ELOW"], dtype=np.float64)
		ehigh = np.asarray(data["EHIGH"], dtype=np.float64)
	return elow, ehigh


def energy_range_to_channel_range(energy_range: tuple[float, float], module: str) -> tuple[int, int]:
	"""Convert one energy range in keV into an inclusive PI-channel range."""
	emin, emax = energy_range
	elow, ehigh = _load_e2pi_edges(module)
	mask = (ehigh > float(emin)) & (elow < float(emax))
	if not np.any(mask):
		raise ValueError(f"Energy range {energy_range} keV does not overlap the calibrated PI grid for module '{module}'.")
	indices = np.nonzero(mask)[0]
	return int(indices[0]), int(indices[-1])
