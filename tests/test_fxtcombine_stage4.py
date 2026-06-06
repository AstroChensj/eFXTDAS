"""Tests for Stage-4 spectral extraction response generation."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from fxtcombine.utils import fxtchain_simplified


def _write_spectrum(path: Path) -> None:
    """Write a minimal OGIP-like spectrum with one table extension.

    Parameters
    ----------
    path : Path
        Destination spectrum path.

    Returns
    -------
    None
    """
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="1I", array=[1]),
            fits.Column(name="COUNTS", format="1J", array=[10]),
        ],
        name="SPECTRUM",
    )
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path, overwrite=True)


def _write_response(path: Path, extname: str) -> None:
    """Write a minimal FITS response-like file with one table extension.

    Parameters
    ----------
    path : Path
        Destination FITS path.
    extname : str
        Extension name written into the response file.

    Returns
    -------
    None
    """
    response = fits.BinTableHDU.from_columns([], name=extname)
    fits.HDUList([fits.PrimaryHDU(), response]).writeto(path, overwrite=True)


def test_fxt_extract_spec_uses_fxtrspgen_and_preserves_product_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Stage 4 should call ``fxtrspgen`` and keep filenames/bookkeeping stable.

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
    obsid_out_dir = tmp_path / "products"
    obsid_out_dir.mkdir()
    obsid_log_dir = obsid_out_dir / "log"
    obsid_log_dir.mkdir()
    src_reg = tmp_path / "target_src.reg"
    bkg_reg = tmp_path / "target_bkg.reg"
    src_reg.write_text("image\ncircle(10,10,5)\n", encoding="utf-8")
    bkg_reg.write_text("image\nannulus(10,10,10,20)\n", encoding="utf-8")
    evt_path = obsid_out_dir / "evt_cl.fits"
    vexpmap_path = obsid_out_dir / "evt_vexp.fits"
    fits.HDUList([fits.PrimaryHDU()]).writeto(evt_path, overwrite=True)
    fits.HDUList([fits.PrimaryHDU()]).writeto(vexpmap_path, overwrite=True)

    prod = {
        "module": "a",
        "obsid": "0001",
        "datamode": "ff",
        "filter": "01",
        "pp": "po",
        "version": "v01",
        "evt_clevt": str(evt_path),
        "vexpmap": str(vexpmap_path),
        "image": str(obsid_out_dir / "default_image.fits"),
        "images": {"band0": str(obsid_out_dir / "default_image.fits")},
        "image_band_channels": {"band0": (0, 1023)},
    }
    commands: list[dict[str, str | None]] = []

    def _fake_run_cmd(cmd: str, logger=None, logname=None, cwd=None):
        commands.append({"cmd": cmd, "logname": logname, "cwd": cwd})
        if cmd.startswith("xselect @"):
            _write_spectrum(obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src.pi")
            _write_spectrum(obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_bkg.pi")
            fits.HDUList([fits.PrimaryHDU()]).writeto(
                obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src_cl.fits",
                overwrite=True,
            )
            fits.HDUList([fits.PrimaryHDU()]).writeto(
                obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src.lc",
                overwrite=True,
            )
            fits.HDUList([fits.PrimaryHDU()]).writeto(
                obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_bkg.lc",
                overwrite=True,
            )
            return None
        if cmd.startswith("fxtrspgen "):
            srcpi = obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src.pi"
            arf = obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src.arf"
            rmf = obsid_out_dir / "fxt_a_0001_ff_01_po_evt_v01_src.rmf"
            _write_response(arf, "SPECRESP")
            _write_response(rmf, "MATRIX")
            with fits.open(srcpi, mode="update") as hdul:
                hdul[1].header["ANCRFILE"] = str(arf)
                hdul[1].header["RESPFILE"] = str(rmf)
                hdul.flush()
            return None
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(fxtchain_simplified, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(fxtchain_simplified, "finalize_xselect_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(fxtchain_simplified, "remove_xselect_tmp_files", lambda *args, **kwargs: None)

    result = fxtchain_simplified.fxt_extract_spec(
        {"stream0": prod},
        str(src_reg),
        str(bkg_reg),
        str(obsid_out_dir),
        str(obsid_log_dir),
        skip_existing=False,
        obsid_logger=None,
    )

    stream = result["stream0"]
    srcpi_path = Path(stream["srcpi"])
    bkgpi_path = Path(stream["bkgpi"])
    arf_path = Path(stream["arf"])
    rmf_path = Path(stream["rmf"])
    assert srcpi_path.name == "fxt_a_0001_ff_01_po_evt_v01_src.pi"
    assert bkgpi_path.name == "fxt_a_0001_ff_01_po_evt_v01_bkg.pi"
    assert arf_path.name == "fxt_a_0001_ff_01_po_evt_v01_src.arf"
    assert rmf_path.name == "fxt_a_0001_ff_01_po_evt_v01_src.rmf"

    fxtrspgen_calls = [entry for entry in commands if str(entry["cmd"]).startswith("fxtrspgen ")]
    assert len(fxtrspgen_calls) == 1
    assert "--arf-out" in str(fxtrspgen_calls[0]["cmd"])
    assert "--rmf-out" in str(fxtrspgen_calls[0]["cmd"])
    assert "--update-pha" in str(fxtrspgen_calls[0]["cmd"])
    assert not any("fxtarfgen" in str(entry["cmd"]) for entry in commands)
    assert not any("fxtrmfgen" in str(entry["cmd"]) for entry in commands)
    assert str(fxtrspgen_calls[0]["logname"]).endswith("fxtrspgen.log")

    with fits.open(srcpi_path) as hdul:
        assert hdul[1].header["BACKFILE"] == bkgpi_path.name
        assert hdul[1].header["ANCRFILE"] == str(arf_path)
        assert hdul[1].header["RESPFILE"] == str(rmf_path)
    with fits.open(bkgpi_path) as hdul:
        assert hdul[1].header["ANCRFILE"] == arf_path.name
        assert hdul[1].header["RESPFILE"] == rmf_path.name
    with fits.open(rmf_path) as hdul:
        assert hdul[1].header["ANCRFILE"] == arf_path.name
