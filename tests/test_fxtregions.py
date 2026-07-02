"""Tests for source and background region writing in ``fxtregions``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from fxtregions import pipeline


class _FakePSFMapper:
    """Minimal PSF mapper stub for region-building tests."""

    def __init__(self, sigma_pix: float = 3.0, size: int = 51) -> None:
        """Build a normalized Gaussian-like kernel.

        Parameters
        ----------
        sigma_pix : float, optional
            Gaussian width in pixels.
        size : int, optional
            Kernel side length in pixels.

        Returns
        -------
        None
        """
        yy, xx = np.indices((size, size), dtype=np.float64)
        cy = (size - 1) / 2.0
        cx = (size - 1) / 2.0
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        kernel = np.exp(-0.5 * rr2 / (sigma_pix * sigma_pix))
        self._kernel = kernel / kernel.sum()

    def kernel_at_position(self, x_ima: float, y_ima: float) -> np.ndarray:
        """Return a fixed local kernel regardless of position.

        Parameters
        ----------
        x_ima : float
            Image x coordinate in 1-based pixels.
        y_ima : float
            Image y coordinate in 1-based pixels.

        Returns
        -------
        np.ndarray
            Normalized local kernel image.
        """
        return self._kernel.copy()


def _write_image(path: Path, ra0: float = 10.0, dec0: float = 20.0) -> WCS:
    """Write a simple celestial image with one-arcsecond pixels.

    Parameters
    ----------
    path : Path
        Destination FITS path.
    ra0 : float, optional
        Reference right ascension in degrees.
    dec0 : float, optional
        Reference declination in degrees.

    Returns
    -------
    WCS
        Celestial WCS used for the image.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50.0, 50.0]
    wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0], dtype=np.float64)
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = wcs.to_header()
    fits.PrimaryHDU(data=np.zeros((100, 100), dtype=np.float32), header=header).writeto(path, overwrite=True)
    return wcs.celestial


def _write_catalog(path: Path, rows: list[tuple[float, float, str, float, float, float]]) -> None:
    """Write a minimal source catalog FITS table for region matching.

    Parameters
    ----------
    path : Path
        Destination catalog path.
    rows : list[tuple[float, float, str, float, float, float]]
        Row tuples of ``(ra, dec, source_type, ml_cts_0, ml_bkg_0, ext)``.

    Returns
    -------
    None
    """
    table = Table(
        rows=rows,
        names=["RA", "DEC", "SOURCE_TYPE", "ML_CTS_0", "ML_BKG_0", "EXT"],
    )
    table.write(path, overwrite=True)


def test_build_regions_writes_source_exclusions_when_overlap_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Written source regions should include computed contaminant exclusions.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest workspace.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    image_path = tmp_path / "image.fits"
    wcs = _write_image(image_path)
    target_ra = 10.0
    target_dec = 20.0
    neighbor_ra, neighbor_dec = wcs.pixel_to_world_values(38.0, 49.0)
    catalog_path = tmp_path / "catalog.fits"
    _write_catalog(
        catalog_path,
        [
            (target_ra, target_dec, "point", 1000.0, 3600.0, 0.0),
            (float(neighbor_ra), float(neighbor_dec), "point", 1000.0, 3600.0, 0.0),
        ],
    )
    src_regfile = tmp_path / "source.reg"
    bkg_regfile = tmp_path / "background.reg"

    monkeypatch.setattr(pipeline, "build_observation_psf_mapper", lambda *args, **kwargs: _FakePSFMapper())

    info = pipeline.build_regions(
        image_path=image_path,
        catalog_path=catalog_path,
        ra_deg=target_ra,
        dec_deg=target_dec,
        mission="ep-fxt",
        instrument="fxta",
        filter_name="thin",
        emin_keV=0.3,
        emax_keV=10.0,
        mode="manual",
        src_radius_arcsec=10.0,
        bkg_inner_arcsec=30.0,
        bkg_outer_arcsec=60.0,
        src_regfile=src_regfile,
        bkg_regfile=bkg_regfile,
        logger=None,
    )

    assert info["source_excludes"]
    source_text = src_regfile.read_text(encoding="utf-8")
    background_text = bkg_regfile.read_text(encoding="utf-8")
    assert f"-{info['source_excludes'][0]}" in source_text
    assert f"-{info['background_excludes'][0]}" in background_text


def test_build_regions_leaves_source_region_plain_when_no_overlap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Written source regions should stay plain when no source overlap exists.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest workspace.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
    """
    image_path = tmp_path / "image_far.fits"
    wcs = _write_image(image_path)
    target_ra = 10.0
    target_dec = 20.0
    neighbor_ra, neighbor_dec = wcs.pixel_to_world_values(10.0, 49.0)
    catalog_path = tmp_path / "catalog_far.fits"
    _write_catalog(
        catalog_path,
        [
            (target_ra, target_dec, "point", 1000.0, 3600.0, 0.0),
            (float(neighbor_ra), float(neighbor_dec), "point", 1000.0, 3600.0, 0.0),
        ],
    )
    src_regfile = tmp_path / "source_far.reg"

    monkeypatch.setattr(pipeline, "build_observation_psf_mapper", lambda *args, **kwargs: _FakePSFMapper())

    info = pipeline.build_regions(
        image_path=image_path,
        catalog_path=catalog_path,
        ra_deg=target_ra,
        dec_deg=target_dec,
        mission="ep-fxt",
        instrument="fxta",
        filter_name="thin",
        emin_keV=0.3,
        emax_keV=10.0,
        mode="manual",
        src_radius_arcsec=10.0,
        bkg_inner_arcsec=30.0,
        bkg_outer_arcsec=60.0,
        src_regfile=src_regfile,
        logger=None,
    )

    assert info["source_excludes"] == []
    source_lines = src_regfile.read_text(encoding="utf-8").splitlines()
    assert all(not line.startswith("-") for line in source_lines[2:])
