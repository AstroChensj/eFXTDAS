"""Tests for the fxtcombine quick-view plotting CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtcombine import quickview


class _FakePSFMapper:
    """Minimal PSF mapper used to test R90 sampling.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Instances record queried positions in ``calls``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def radius_at_position(self, x_ima: float, y_ima: float, frac_value: float) -> float:
        """Return a deterministic R90-like value.

        Parameters
        ----------
        x_ima : float
            One-based image x coordinate.
        y_ima : float
            One-based image y coordinate.
        frac_value : float
            Requested EEF fraction.

        Returns
        -------
        float
            Synthetic radius value.
        """
        self.calls.append((x_ima, y_ima))
        return float(x_ima + y_ima + frac_value)


def _write_image(path: Path, data: np.ndarray) -> None:
    """Write one small celestial-WCS FITS image.

    Parameters
    ----------
    path : Path
        Destination FITS path.
    data : np.ndarray
        Image data.

    Returns
    -------
    None
    """
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = data.shape[1]
    header["NAXIS2"] = data.shape[0]
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CRPIX1"] = 5.0
    header["CRPIX2"] = 5.0
    header["CDELT1"] = -0.001
    header["CDELT2"] = 0.001
    fits.writeto(path, np.asarray(data, dtype=np.float32), header, overwrite=True)


def test_resolve_paths_uses_current_stack_names(tmp_path: Path) -> None:
    """Standard path resolution should derive current fxtcombine names.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest workspace.

    Returns
    -------
    None
    """
    paths = quickview.resolve_paths(tmp_path)
    assert paths.rate == (tmp_path / "stack_rate.fits").resolve()
    assert paths.bkgmap == (tmp_path / "stack_bkgmap.fits").resolve()
    assert paths.mask == (tmp_path / "stack_mask.fits").resolve()
    assert paths.expmap == (tmp_path / "stack_exp.fits").resolve()
    assert paths.psfprod == (tmp_path / "stack_psfprod.fits").resolve()
    assert paths.src_reg == (tmp_path / "stack_src.reg").resolve()
    assert paths.target_src_reg == (tmp_path / "target_src.reg").resolve()
    assert paths.target_bkg_reg == (tmp_path / "target_bkg.reg").resolve()


def test_parse_regions_handles_image_and_fk5_exclusions(tmp_path: Path) -> None:
    """Region parsing should distinguish image detections and FK5 exclusions.

    Parameters
    ----------
    tmp_path : Path
        Temporary pytest workspace.

    Returns
    -------
    None
    """
    src_reg = tmp_path / "stack_src.reg"
    src_reg.write_text("image\ncircle(10,20,5)\n-circle(1,2,3)\n", encoding="utf-8")
    detections = quickview.parse_image_circles(src_reg)
    assert len(detections) == 1
    assert detections[0].x == 9.0
    assert detections[0].y == 19.0
    assert detections[0].radius_pix == 5.0

    target_src = tmp_path / "target_src.reg"
    target_src.write_text(
        "# Region file format: DS9 version 4.1\n"
        "fk5\n"
        "circle(10.0,20.0,60.000\")\n"
        "-circle(10.01,20.01,12.500\")\n",
        encoding="utf-8",
    )
    main_src, src_excludes = quickview.parse_fk5_regions(target_src)
    assert isinstance(main_src, quickview.CircleRegion)
    assert main_src.radius_arcsec == 60.0
    assert len(src_excludes) == 1
    assert src_excludes[0].radius_arcsec == 12.5

    target_bkg = tmp_path / "target_bkg.reg"
    target_bkg.write_text(
        "fk5\n"
        "annulus(10.0,20.0,90.000\",300.000\")\n"
        "-circle(9.99,20.01,20.000\")\n",
        encoding="utf-8",
    )
    main_bkg, bkg_excludes = quickview.parse_fk5_regions(target_bkg)
    assert isinstance(main_bkg, quickview.AnnulusRegion)
    assert main_bkg.r_in_arcsec == 90.0
    assert main_bkg.r_out_arcsec == 300.0
    assert len(bkg_excludes) == 1
    assert bkg_excludes[0].radius_arcsec == 20.0


def test_compute_r90_map_supports_exact_and_coarse_sampling(tmp_path: Path, monkeypatch) -> None:
    """R90 calculation should support exact and coarse-grid modes.

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
    fake_mapper = _FakePSFMapper()
    monkeypatch.setattr(quickview, "load_psf_product", lambda path: fake_mapper)
    valid_mask = np.ones((5, 5), dtype=bool)

    exact = quickview.compute_r90_map(tmp_path / "stack_psfprod.fits", valid_mask.shape, valid_mask, r90_stride=1)
    assert np.all(np.isfinite(exact))
    assert len(fake_mapper.calls) == 25

    fake_mapper.calls.clear()
    coarse = quickview.compute_r90_map(tmp_path / "stack_psfprod.fits", valid_mask.shape, valid_mask, r90_stride=2)
    assert np.all(np.isfinite(coarse))
    assert 0 < len(fake_mapper.calls) < 25


def test_compute_r90_map_prefers_cached_extension(tmp_path: Path, monkeypatch) -> None:
    """R90 calculation should read cached PSF-product maps when present."""
    cached = np.arange(25, dtype=np.float32).reshape(5, 5)
    r90_hdu = fits.ImageHDU(data=cached, name="R90")
    r90_hdu.header["BUNIT"] = "pixel"
    r90_hdu.header["EEF"] = 0.90
    psfprod = tmp_path / "stack_psfprod.fits"
    fits.HDUList([fits.PrimaryHDU(header=fits.Header({"PSFTYPE": "STACK"})), r90_hdu]).writeto(psfprod, overwrite=True)
    monkeypatch.setattr(
        quickview,
        "load_psf_product",
        lambda path: (_ for _ in ()).throw(AssertionError("cached R90 should avoid mapper load")),
    )
    valid_mask = np.ones((5, 5), dtype=bool)

    r90 = quickview.compute_r90_map(psfprod, valid_mask.shape, valid_mask, r90_stride=4)

    assert np.allclose(r90, cached)


def test_quickview_cli_writes_figure_with_standard_stack_dir(tmp_path: Path, monkeypatch) -> None:
    """The CLI should infer paths from stack_dir and write a quick-view image.

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
    shape = (10, 10)
    _write_image(tmp_path / "stack_rate.fits", np.arange(100, dtype=float).reshape(shape))
    _write_image(tmp_path / "stack_bkgmap.fits", np.ones(shape))
    _write_image(tmp_path / "stack_mask.fits", np.ones(shape))
    _write_image(tmp_path / "stack_exp.fits", np.full(shape, 100.0))
    fits.writeto(tmp_path / "stack_psfprod.fits", np.zeros((1, 1), dtype=np.float32), overwrite=True)
    (tmp_path / "stack_src.reg").write_text("image\ncircle(5,5,2)\n", encoding="utf-8")
    (tmp_path / "target_src.reg").write_text(
        "fk5\ncircle(10.0,20.0,20.000\")\n-circle(10.001,20.001,5.000\")\n",
        encoding="utf-8",
    )
    (tmp_path / "target_bkg.reg").write_text(
        "fk5\nannulus(10.0,20.0,30.000\",80.000\")\n-circle(9.999,20.001,6.000\")\n",
        encoding="utf-8",
    )

    def _fake_r90(psfprod_path, shape, valid_mask, r90_stride=4, logger=None):
        """Return a deterministic R90 map for CLI smoke testing.

        Parameters
        ----------
        psfprod_path : Path
            Ignored synthetic PSF product path.
        shape : tuple[int, int]
            Requested output map shape.
        valid_mask : np.ndarray
            Boolean mask selecting valid pixels.
        r90_stride : int
            Ignored R90 sampling stride.
        logger : logging.Logger | None
            Ignored logger.

        Returns
        -------
        np.ndarray
            Synthetic R90 map.
        """
        return np.where(valid_mask, 4.0, np.nan).astype(np.float32)

    monkeypatch.setattr(quickview, "compute_r90_map", _fake_r90)
    out_path = tmp_path / "quickview.png"
    quickview.main([str(tmp_path), "--out", str(out_path), "--title", "Quick View Test", "--log-level", "ERROR"])
    assert out_path.exists()
    assert out_path.stat().st_size > 0
