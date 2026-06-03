"""Tests for FSA flare-screen command construction."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits

from fxtcombine.utils import flarescreen


def test_run_fsa_flare_screening_passes_method_to_bkgoptrate(tmp_path: Path, monkeypatch) -> None:
    """The flare-screen helper should forward the chosen threshold method."""
    obsid_out_dir = tmp_path / "products"
    obsid_out_dir.mkdir()
    sub_log_dir = tmp_path / "logs"
    sub_log_dir.mkdir()
    flare_lc_path = obsid_out_dir / "fxt_a_0001_ff_po_pp_fsaevt_v01_flare.lc"
    flare_lc_path.write_text("placeholder", encoding="utf-8")
    base_gti_path = tmp_path / "base.gti"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="START", format="1D", array=[0.0]),
                    fits.Column(name="STOP", format="1D", array=[10.0]),
                ],
                name="GTI",
            ),
        ]
    ).writeto(base_gti_path, overwrite=True)

    captured: dict[str, str] = {}

    def _fake_run_cmd(cmd: str, logger=None, logname=None, cwd=None) -> None:
        captured["cmd"] = cmd
        diag_path = obsid_out_dir / "fxt_a_0001_ff_po_pp_evt_v01_flare_diag.fits"
        flare_gti_path = obsid_out_dir / "fxt_a_0001_ff_po_pp_evt_v01_flare.gti"
        screened_gti_path = obsid_out_dir / "fxt_a_0001_ff_po_pp_evt_v01_screened.gti"
        bkgopt_hdu = fits.BinTableHDU.from_columns([], name="BKGOPT")
        bkgopt_hdu.header["BGOPTCUT"] = 3.0
        bkgopt_hdu.header["FRACTLFT"] = 0.75
        bkgopt_hdu.header["OPTSTAT"] = "optimal"
        fits.HDUList([fits.PrimaryHDU(), bkgopt_hdu]).writeto(diag_path, overwrite=True)
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([], name="GTI")]).writeto(flare_gti_path, overwrite=True)
        fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([], name="GTI")]).writeto(screened_gti_path, overwrite=True)

    monkeypatch.setattr(flarescreen, "run_cmd", _fake_run_cmd)

    result = flarescreen.run_fsa_flare_screening(
        {
            "module": "a",
            "obsID": "0001",
            "mode": "ff",
            "filter": "po",
            "pp": "pp",
            "version": "v01",
        },
        {"grade": "grade.fits", "sub_log_dir": str(sub_log_dir)},
        base_gti_path=str(base_gti_path),
        grade="0-12",
        flare_threshold_method="robust_iqr",
        flare_energy_range=(0.5, 10.0),
        flare_binsize=20.0,
        flare_min_time_ratio=0.05,
        obsid_out_dir=str(obsid_out_dir),
        skip_existing=True,
        obsid_logger=None,
    )

    assert "--method robust_iqr" in captured["cmd"]
    assert result["flare_screen_status"] == "optimal"
