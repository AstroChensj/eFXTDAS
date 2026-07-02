"""Tests for CALDB CBD parsing and lookup behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from fxtcaldb.query import ObservationMetadata, _evaluate_expr_conditions, _parse_cbd_entries
from fxtcaldb.response import resolve_rmf


def _write_caldb_config(path: Path) -> None:
    """Write a minimal CALDB config that points to one EP-FXT index.

    Parameters
    ----------
    path : Path
        Destination config path.

    Returns
    -------
    None
    """
    path.write_text("EP FXT dummy data/ep/fxt caldb.indx dummy data/ep/fxt\n", encoding="utf-8")


def _write_caldb_index(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal CALDB index FITS file.

    Parameters
    ----------
    path : Path
        Destination CALDB index path.
    rows : list[dict[str, object]]
        Row dictionaries for the FITS table.

    Returns
    -------
    None
    """
    columns = [
        fits.Column(name="TELESCOP", format="8A", array=[row["TELESCOP"] for row in rows]),
        fits.Column(name="INSTRUME", format="8A", array=[row["INSTRUME"] for row in rows]),
        fits.Column(name="DETNAM", format="8A", array=[row["DETNAM"] for row in rows]),
        fits.Column(name="FILTER", format="8A", array=[row["FILTER"] for row in rows]),
        fits.Column(name="CAL_CNAM", format="16A", array=[row["CAL_CNAM"] for row in rows]),
        fits.Column(name="CAL_QUAL", format="J", array=np.array([row["CAL_QUAL"] for row in rows], dtype=np.int32)),
        fits.Column(name="REF_TIME", format="D", array=np.array([row["REF_TIME"] for row in rows], dtype=np.float64)),
        fits.Column(name="CAL_FILE", format="64A", array=[row["CAL_FILE"] for row in rows]),
        fits.Column(name="CAL_DIR", format="64A", array=[row["CAL_DIR"] for row in rows]),
        fits.Column(name="CAL_XNO", format="J", array=np.array([row["CAL_XNO"] for row in rows], dtype=np.int32)),
        fits.Column(name="CAL_CBD", format="256A", array=[row["CAL_CBD"] for row in rows]),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def _write_rmf(path: Path) -> None:
    """Write a minimal RMF file for lookup tests.

    Parameters
    ----------
    path : Path
        Destination RMF path.

    Returns
    -------
    None
    """
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", array=np.array([0.5], dtype=np.float32)),
            fits.Column(name="ENERG_HI", format="E", array=np.array([1.0], dtype=np.float32)),
            fits.Column(name="N_GRP", format="I", array=np.array([1], dtype=np.int16)),
            fits.Column(name="F_CHAN", format="PI()", array=np.array([[0]], dtype=np.int16)),
            fits.Column(name="N_CHAN", format="PI()", array=np.array([[1]], dtype=np.int16)),
            fits.Column(name="MATRIX", format="PE()", array=np.array([[1.0]], dtype=np.float32)),
        ],
        name="MATRIX",
    )
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=np.array([0], dtype=np.int32)),
            fits.Column(name="E_MIN", format="E", array=np.array([0.5], dtype=np.float32)),
            fits.Column(name="E_MAX", format="E", array=np.array([1.0], dtype=np.float32)),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), matrix, ebounds]).writeto(path, overwrite=True)


def test_parse_cbd_entries_keeps_fxT_keys_with_unit_suffixes() -> None:
    """CBD parsing should not fold unit suffixes into later key names."""
    cbd = "DATAMODE(FF) THETA(0)arcmin PHI(0)deg ENERG(0.1-12.0)keV GRADE(G0:12) FILTER(0)"
    parsed = _parse_cbd_entries(cbd)
    assert set(parsed) == {"DATAMODE", "THETA", "PHI", "ENERG", "GRADE", "FILTER"}
    assert parsed["DATAMODE"][0].sval == "FF"
    assert parsed["THETA"][0].min_val == 0.0
    assert parsed["PHI"][0].min_val == 0.0
    assert parsed["ENERG"][0].min_val == 0.1
    assert parsed["ENERG"][0].max_val == 12.0
    assert parsed["GRADE"][0].sval == "G0:12"
    assert parsed["FILTER"][0].min_val == 0.0


def test_evaluate_expr_conditions_matches_mode_and_grade() -> None:
    """CBD expressions should match the intended FXT mode/grade rows."""
    cbd = "DATAMODE(FF) THETA(0)arcmin PHI(0)deg ENERG(0.1-12.0)keV GRADE(G0:12) FILTER(0)"
    assert _evaluate_expr_conditions("DATAMODE(FF) .AND. GRADE(G0:12)", cbd) is True
    assert _evaluate_expr_conditions("DATAMODE(FF) .AND. GRADE(G0:4)", cbd) is False
    assert _evaluate_expr_conditions("DATAMODE(LW) .AND. GRADE(G0:12)", cbd) is False


def test_resolve_rmf_matches_unit_suffixed_cbd_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """RMF lookup should succeed when CAL_CBD includes trailing unit suffixes."""
    caldb_root = tmp_path / "caldb"
    rmf_dir = caldb_root / "data/ep/fxt/cpf/rmf"
    index_dir = caldb_root / "data/ep/fxt"
    rmf_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    rmf_path = rmf_dir / "ep_fxt_b_ff_g12_20201101_v1.rmf"
    _write_rmf(rmf_path)
    _write_caldb_config(caldb_root / "caldb.config")
    _write_caldb_index(
        index_dir / "caldb.indx",
        [
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "B",
                "FILTER": "NONE",
                "CAL_CNAM": "MATRIX",
                "CAL_QUAL": 0,
                "REF_TIME": 59154.0,
                "CAL_FILE": "ep_fxt_b_ff_g12_20201101_v1.rmf",
                "CAL_DIR": "data/ep/fxt/cpf/rmf",
                "CAL_XNO": 1,
                "CAL_CBD": "DATAMODE(FF) THETA(0)arcmin PHI(0)deg ENERG(0.1-12.0)keV GRADE(G0:12) FILTER(0) NONE NONE NONE",
            }
        ],
    )
    monkeypatch.setenv("CALDB", str(caldb_root))
    monkeypatch.setenv("CALDBCONFIG", str(caldb_root / "caldb.config"))
    metadata = ObservationMetadata(
        telescope="EP",
        instrument="FXT",
        detector_code="B",
        detnam="FXTB",
        filt="NONE",
        datamode="FF",
        start_date="2026-05-21",
        start_time="17:31:42.918",
        stop_date="2026-05-21",
        stop_time="17:56:20.875",
        max_grade=12,
        ra_pnt=None,
        dec_pnt=None,
        pa_pnt=None,
    )
    resolved_path, extno = resolve_rmf(metadata)
    assert Path(resolved_path) == rmf_path
    assert extno == 1
