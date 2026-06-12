"""Command-line entry point for building EP-FXT PSF mapper products."""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy.io import fits

from fxtpsfgen.mapper import build_observation_psf_mapper, build_stacked_psf_mapper


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``fxtpsfgen``.

    Returns
    -------
    argparse.ArgumentParser
        Configured CLI parser.
    """
    parser = argparse.ArgumentParser(description="Build EP-FXT PSF mapper products.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    obs_parser = subparsers.add_parser("build-obs", help="Build one per-observation PSF product")
    obs_parser.add_argument("image", type=Path, help="Input image FITS path")
    obs_parser.add_argument("--expmap", type=Path, default=None, help="Optional matching weight/exposure map")
    obs_parser.add_argument("--instrument", type=str, default=None, help="Optional detector-arm override such as fxta")
    obs_parser.add_argument("--filter", type=str, default=None, help="Optional filter-family override such as open")
    obs_parser.add_argument("--emin", type=float, default=None, help="Lower image energy bound in keV")
    obs_parser.add_argument("--emax", type=float, default=None, help="Upper image energy bound in keV")
    obs_parser.add_argument("--out", type=Path, required=True, help="Output PSF product FITS path")

    stack_parser = subparsers.add_parser("stack", help="Build one stacked PSF product")
    stack_parser.add_argument("--obs-psf", type=Path, action="append", required=True, help="Per-observation PSF product path; repeat once per component")
    stack_parser.add_argument("--weightmap", type=Path, action="append", required=True, help="Matching vignetted weight/exposure map; repeat once per component")
    stack_parser.add_argument("--ref-image", type=Path, required=True, help="Reference stacked image FITS path")
    stack_parser.add_argument("--out", type=Path, required=True, help="Output stacked PSF product FITS path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``fxtpsfgen`` CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument vector override.

    Returns
    -------
    int
        Shell return code.
    """
    args = build_parser().parse_args(argv)
    if args.command == "build-obs":
        mapper = build_observation_psf_mapper(
            args.image,
            expmap_path=args.expmap,
            instrument=args.instrument,
            filter_name=args.filter,
            emin_keV=args.emin,
            emax_keV=args.emax,
        )
        mapper.write(args.out)
        return 0
    with fits.open(args.ref_image) as hdul:
        ref_header = hdul[0].header.copy()
    mapper = build_stacked_psf_mapper(args.obs_psf, args.weightmap, ref_header)
    mapper.write(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
