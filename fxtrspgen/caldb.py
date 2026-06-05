"""Compatibility wrappers around the shared ``fxtcaldb`` package."""

from __future__ import annotations

from fxtcaldb.metadata import SpectrumMetadata, read_spectrum_metadata
from fxtcaldb.response import read_base_arf_table, resolve_base_arf, resolve_rmf
from fxtcaldb.teldef import resolve_teldef
from fxtcaldb.vignetting import resolve_vignetting_table

__all__ = [
    "SpectrumMetadata",
    "read_base_arf_table",
    "read_spectrum_metadata",
    "resolve_base_arf",
    "resolve_rmf",
    "resolve_teldef",
    "resolve_vignetting_table",
]
