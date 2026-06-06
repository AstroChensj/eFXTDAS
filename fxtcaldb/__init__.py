"""Shared EP-FXT calibration, optics, and PSF helpers."""

from fxtcaldb.optics import TeldefInfo, compute_optical_axis_pixel, read_teldef_info, resolve_teldef
from fxtcaldb.psf import (
    EEFSelection,
    build_psf_kernel,
    load_beta_psf_table,
    load_eef_curve_for_theta,
    resolve_beta_psf_path,
    select_ep_eef_files,
)
from fxtcaldb.query import ObservationMetadata, read_observation_metadata
from fxtcaldb.response import read_base_arf_table, resolve_base_arf, resolve_rmf
from fxtcaldb.vignetting import resolve_vignetting_table

__all__ = [
    "EEFSelection",
    "ObservationMetadata",
    "TeldefInfo",
    "build_psf_kernel",
    "compute_optical_axis_pixel",
    "load_beta_psf_table",
    "load_eef_curve_for_theta",
    "read_base_arf_table",
    "read_observation_metadata",
    "read_teldef_info",
    "resolve_base_arf",
    "resolve_beta_psf_path",
    "resolve_rmf",
    "resolve_teldef",
    "resolve_vignetting_table",
    "select_ep_eef_files",
]
