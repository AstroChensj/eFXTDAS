"""Tests for APER-mode sensitivity-map generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from fxtsensmap.pipeline import main as fxtsensmap_main, run_fxtsensmap
from fxtsensmap.psf import load_official_psfmap_radius_map
from fxtsensmap.sensitivity import compute_sensitivity_map, poisson_source_counts_for_likelihood


def _header(shape: tuple[int, int] = (5, 5), pixel_scale_arcsec: float = 1.0) -> fits.Header:
    """Create a simple celestial WCS header for synthetic maps."""
    ny, nx = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    scale_deg = float(pixel_scale_arcsec) / 3600.0
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    header = wcs.to_header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    return header


def _write_image(path: Path, data: np.ndarray, header: fits.Header | None = None) -> Path:
    """Write a synthetic primary image."""
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header or _header(data.shape)).writeto(path, overwrite=True)
    return path


def _write_cached_psfprod(path: Path, radius_pix: np.ndarray, eef: float = 0.90) -> Path:
    """Write a minimal PSF product with one cached Rxx extension."""
    hdu = fits.ImageHDU(data=np.asarray(radius_pix, dtype=np.float32), header=_header(radius_pix.shape), name="R90")
    hdu.header["BUNIT"] = "pixel"
    hdu.header["EEF"] = eef
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"PSFTYPE": "STACK"})), hdu]).writeto(path, overwrite=True)
    return path


def test_poisson_threshold_increases_with_background() -> None:
    """Higher aperture background should require more source counts."""
    low = poisson_source_counts_for_likelihood(0.5, 6.0)
    high = poisson_source_counts_for_likelihood(20.0, 6.0)
    assert low > 0.0
    assert high > low


def test_compute_sensitivity_map_improves_with_exposure() -> None:
    """Flux sensitivity should decrease when exposure increases."""
    bkg = np.ones((5, 5), dtype=np.float64)
    exp = np.full((5, 5), 100.0, dtype=np.float64)
    exp[:, 3:] = 200.0
    radius = np.full((5, 5), 1.5, dtype=np.float64)
    sens = compute_sensitivity_map(bkg, exp, radius, eef=0.90, ecf=1.0e11, likemin=6.0)
    assert np.all(np.isfinite(sens))
    assert sens[2, 3] < sens[2, 1]


def test_compute_sensitivity_map_applies_valid_mask() -> None:
    """Masked center pixels should stay NaN and aperture sums should use valid pixels."""
    bkg = np.ones((5, 5), dtype=np.float64)
    exp = np.full((5, 5), 100.0, dtype=np.float64)
    radius = np.full((5, 5), 1.5, dtype=np.float64)
    valid_mask = np.ones((5, 5), dtype=bool)
    valid_mask[2, 2] = False
    valid_mask[0, :] = False
    valid_mask[:, 0] = False

    sens = compute_sensitivity_map(bkg, exp, radius, eef=0.90, ecf=1.0e11, likemin=6.0, valid_mask=valid_mask)

    assert np.isnan(sens[2, 2])
    assert np.isnan(sens[0, 3])
    assert np.isfinite(sens[3, 3])


def test_fxtsensmap_uses_cached_psfprod_radius_map(tmp_path: Path) -> None:
    """CLI workflow should read cached R90 from fxtpsfgen products."""
    shape = (5, 5)
    header = _header(shape)
    bkg = _write_image(tmp_path / "bkg.fits", np.ones(shape), header)
    exp = _write_image(tmp_path / "exp.fits", np.full(shape, 100.0), header)
    psfprod = _write_cached_psfprod(tmp_path / "stack_psfprod.fits", np.full(shape, 1.5, dtype=np.float32))
    out = tmp_path / "sens.fits"

    result = run_fxtsensmap(
        bkgmap_path=bkg,
        expmap_path=exp,
        out_path=out,
        eef=0.90,
        psfprod=psfprod,
        ecf=1.0e11,
        likemin=6.0,
    )

    assert result == out
    with fits.open(out) as hdul:
        assert hdul[0].data.shape == shape
        assert np.all(np.isfinite(hdul[0].data))
        assert hdul[0].header["SENSMODE"] == "APER"
        assert hdul[0].header["EEF"] == pytest.approx(0.90)


def test_official_psfmap_arcsec_units_convert_to_pixels(tmp_path: Path) -> None:
    """Official fxtpsfmap radius images should validate EEF and convert units."""
    shape = (5, 5)
    header = _header(shape, pixel_scale_arcsec=2.0)
    psfmap = tmp_path / "psfmap.fits"
    map_header = header.copy()
    map_header["BUNIT"] = "arcsec"
    map_header["ECF"] = 0.90
    fits.PrimaryHDU(data=np.full(shape, 4.0, dtype=np.float32), header=map_header).writeto(psfmap, overwrite=True)

    radius = load_official_psfmap_radius_map(psfmap, eef=0.90, target_header=header, target_shape=shape)

    assert np.allclose(radius, 2.0)
    with pytest.raises(ValueError, match="does not match requested EEF"):
        load_official_psfmap_radius_map(psfmap, eef=0.80, target_header=header, target_shape=shape)


def test_fxtsensmap_cli_applies_mask(tmp_path: Path) -> None:
    """The console entry point should write NaN outside the supplied mask."""
    shape = (5, 5)
    header = _header(shape)
    bkg = _write_image(tmp_path / "bkg.fits", np.ones(shape), header)
    exp = _write_image(tmp_path / "exp.fits", np.full(shape, 100.0), header)
    mask_data = np.ones(shape, dtype=np.float32)
    mask_data[2, 2] = 0.0
    mask = _write_image(tmp_path / "mask.fits", mask_data, header)
    psfprod = _write_cached_psfprod(tmp_path / "stack_psfprod.fits", np.full(shape, 1.5, dtype=np.float32))
    out = tmp_path / "sens_masked.fits"

    status = fxtsensmap_main(
        [
            "--bkgmap",
            str(bkg),
            "--expmap",
            str(exp),
            "--mask",
            str(mask),
            "--psfprod",
            str(psfprod),
            "--eef",
            "0.90",
            "--ecf",
            "1e11",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    with fits.open(out) as hdul:
        assert np.isnan(hdul[0].data[2, 2])
        assert np.isfinite(hdul[0].data[2, 3])
        assert hdul[0].header["MASKED"]
        assert hdul[0].header["MASKFILE"] == str(mask)


def test_fxtsensmap_rejects_mask_shape_mismatch(tmp_path: Path) -> None:
    """Mask shape must match the science maps."""
    header = _header((5, 5))
    bkg = _write_image(tmp_path / "bkg.fits", np.ones((5, 5)), header)
    exp = _write_image(tmp_path / "exp.fits", np.full((5, 5), 100.0), header)
    mask = _write_image(tmp_path / "mask.fits", np.ones((4, 5)), _header((4, 5)))
    psfprod = _write_cached_psfprod(tmp_path / "stack_psfprod.fits", np.full((5, 5), 1.5, dtype=np.float32))

    with pytest.raises(ValueError, match="Mask shape"):
        run_fxtsensmap(
            bkgmap_path=bkg,
            expmap_path=exp,
            mask_path=mask,
            out_path=tmp_path / "sens_bad_mask.fits",
            eef=0.90,
            psfprod=psfprod,
            ecf=1.0e11,
        )


def test_fxtsensmap_cli_writes_output(tmp_path: Path) -> None:
    """The console entry point should write a sensitivity FITS map."""
    shape = (5, 5)
    header = _header(shape)
    bkg = _write_image(tmp_path / "bkg.fits", np.ones(shape), header)
    exp = _write_image(tmp_path / "exp.fits", np.full(shape, 100.0), header)
    psfprod = _write_cached_psfprod(tmp_path / "stack_psfprod.fits", np.full(shape, 1.5, dtype=np.float32))
    out = tmp_path / "sens_cli.fits"

    status = fxtsensmap_main(
        [
            "--bkgmap",
            str(bkg),
            "--expmap",
            str(exp),
            "--psfprod",
            str(psfprod),
            "--eef",
            "0.90",
            "--ecf",
            "1e11",
            "--out",
            str(out),
        ]
    )

    assert status == 0
    assert out.exists()
