"""Command-line interface for APER-mode FXT sensitivity maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtsensmap.psf import load_radius_map
from fxtsensmap.sensitivity import DEFAULT_ECF, compute_sensitivity_map


def load_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load a primary FITS image as a float array.

    Parameters
    ----------
    path : Path
        FITS image path.

    Returns
    -------
    tuple[np.ndarray, fits.Header]
        Image data and copied header.
    """
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"{path} does not contain a 2D primary image.")
    return data, header


def write_sensitivity_map(
    path: Path,
    sensitivity: np.ndarray,
    header: fits.Header,
    *,
    eef: float,
    ecf: float,
    likemin: float,
    mode: str,
    mask_path: Path | None = None,
) -> None:
    """Write a sensitivity map to FITS.

    Parameters
    ----------
    path : Path
        Output FITS path.
    sensitivity : np.ndarray
        Flux sensitivity map.
    header : fits.Header
        Reference image header.
    eef : float
        Encircled-energy fraction.
    ecf : float
        Count-rate to flux conversion.
    likemin : float
        Detection likelihood threshold.
    mode : str
        Sensitivity algorithm mode.
    mask_path : Path | None, optional
        Analysis-mask FITS path used to restrict valid pixels.

    Returns
    -------
    None
    """
    out_header = header.copy()
    out_header["BUNIT"] = ("erg cm-2 s-1", "Flux sensitivity")
    out_header["SENSMODE"] = (str(mode).upper(), "Sensitivity-map algorithm")
    out_header["EEF"] = (float(eef), "Aperture encircled-energy fraction")
    out_header["ECF"] = (float(ecf), "ct/s per erg/cm2/s")
    out_header["LIKEMIN"] = (float(likemin), "Detection likelihood threshold")
    out_header["MASKED"] = (mask_path is not None, "Analysis mask applied")
    if mask_path is not None:
        out_header["MASKFILE"] = (str(mask_path), "Analysis mask file")
    fits.writeto(path, np.asarray(sensitivity, dtype=np.float32), out_header, overwrite=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fxtsensmap`` command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Generate EP-FXT APER-mode sensitivity maps.")
    parser.add_argument("--bkgmap", type=Path, required=True, help="Background map FITS image in expected counts per pixel")
    parser.add_argument("--expmap", type=Path, required=True, help="Exposure map FITS image in seconds")
    parser.add_argument("--mask", type=Path, default=None, help="Optional analysis mask FITS image; non-zero finite pixels are valid")
    psf_group = parser.add_mutually_exclusive_group(required=True)
    psf_group.add_argument("--psfprod", type=Path, default=None, help="fxtpsfgen PSF product")
    psf_group.add_argument("--psfmap", type=Path, default=None, help="Official fxtpsfmap radius image")
    parser.add_argument("--out", type=Path, required=True, help="Output sensitivity-map FITS path")
    parser.add_argument("--eef", type=float, required=True, help="Encircled-energy fraction used for the aperture")
    parser.add_argument("--ecf", type=float, default=DEFAULT_ECF, help=f"ECF in ct/s per erg/cm2/s. Default: {DEFAULT_ECF:.5g}")
    parser.add_argument("--likemin", type=float, default=6.0, help="Detection likelihood threshold")
    parser.add_argument("--mode", choices=("aper",), default="aper", help="Sensitivity calculation mode")
    parser.add_argument("--energy-kev", type=float, default=None, help="Optional PSF energy for computed fxtpsfgen radius maps")
    parser.add_argument("--block-rows", type=int, default=64, help="Rows per block for stacked psfprod radius-map computation")
    parser.add_argument("--jobs", type=int, default=1, help="Thread workers for stacked psfprod radius-map computation")
    return parser


def run_fxtsensmap(
    *,
    bkgmap_path: Path,
    expmap_path: Path,
    out_path: Path,
    eef: float,
    psfprod: Path | None = None,
    psfmap: Path | None = None,
    mask_path: Path | None = None,
    ecf: float = DEFAULT_ECF,
    likemin: float = 6.0,
    mode: str = "aper",
    energy_keV: float | None = None,
    block_rows: int = 64,
    nworkers: int = 1,
) -> Path:
    """Run the sensitivity-map workflow.

    Parameters
    ----------
    bkgmap_path : Path
        Background map FITS path.
    expmap_path : Path
        Exposure map FITS path.
    out_path : Path
        Output sensitivity map path.
    eef : float
        Encircled-energy fraction.
    psfprod : Path | None, optional
        ``fxtpsfgen`` PSF product path.
    psfmap : Path | None, optional
        Official ``fxtpsfmap`` radius image path.
    mask_path : Path | None, optional
        Analysis-mask FITS path. Non-zero finite pixels are valid.
    ecf : float, optional
        Count-rate to flux conversion.
    likemin : float, optional
        Detection likelihood threshold.
    mode : str, optional
        Sensitivity mode. Only ``aper`` is implemented.
    energy_keV : float | None, optional
        Optional PSF energy for computed PSF-product radius maps.
    block_rows : int, optional
        Row block size for stacked PSF products.
    nworkers : int, optional
        Thread workers for stacked PSF products.

    Returns
    -------
    Path
        Output path.
    """
    if mode.lower() != "aper":
        raise ValueError("Only mode='aper' is implemented.")
    bkgmap, bkg_header = load_image(bkgmap_path)
    expmap, _exp_header = load_image(expmap_path)
    if bkgmap.shape != expmap.shape:
        raise ValueError(f"Background map shape {bkgmap.shape} does not match exposure map shape {expmap.shape}.")
    if mask_path is not None:
        mask_data, _mask_header = load_image(mask_path)
        if mask_data.shape != bkgmap.shape:
            raise ValueError(f"Mask shape {mask_data.shape} does not match background map shape {bkgmap.shape}.")
        valid_mask = np.isfinite(mask_data) & (mask_data != 0.0)
    else:
        valid_mask = None
    radius_map = load_radius_map(
        psfprod=psfprod,
        psfmap=psfmap,
        eef=eef,
        target_header=bkg_header,
        target_shape=bkgmap.shape,
        energy_keV=energy_keV,
        block_rows=block_rows,
        nworkers=nworkers,
    )
    sensitivity = compute_sensitivity_map(
        bkgmap,
        expmap,
        radius_map,
        eef=eef,
        ecf=ecf,
        likemin=likemin,
        valid_mask=valid_mask,
    )
    write_sensitivity_map(out_path, sensitivity, bkg_header, eef=eef, ecf=ecf, likemin=likemin, mode=mode, mask_path=mask_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    """Run the ``fxtsensmap`` CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument vector.

    Returns
    -------
    int
        Shell return code.
    """
    args = build_parser().parse_args(argv)
    run_fxtsensmap(
        bkgmap_path=args.bkgmap,
        expmap_path=args.expmap,
        out_path=args.out,
        eef=args.eef,
        psfprod=args.psfprod,
        psfmap=args.psfmap,
        mask_path=args.mask,
        ecf=args.ecf,
        likemin=args.likemin,
        mode=args.mode,
        energy_keV=args.energy_kev,
        block_rows=args.block_rows,
        nworkers=args.jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
