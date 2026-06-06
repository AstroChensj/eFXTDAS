"""Tests for standalone ``fxtrspgen`` response generation."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from fxtcaldb.optics import compute_optical_axis_pixel
from fxtcaldb.query import read_observation_metadata
from fxtcaldb.response import resolve_base_arf
from fxtcaldb.query import find_calibration_file
from fxtpsfgen.mapper import ObservationPSFMapper, StackedPSFMapper, build_observation_psf_mapper, build_stacked_psf_mapper
from fxtrspgen.arf import resolve_source_position
from fxtrspgen.pipeline import run_fxtrspgen
from fxtrspgen.regions import UnsupportedRegionError, load_region_set
from fxtrspgen.rmf import generate_rmf


def _build_simple_wcs(nx: int = 100, ny: int = 100) -> WCS:
    """Create a simple celestial WCS for synthetic image tests."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    wcs.wcs.cdelt = [-0.002685739662709, 0.002685739662709]
    return wcs


def _write_exposure(path: Path, shape: tuple[int, int] = (100, 100), value: float = 1.0) -> Path:
    """Write a synthetic exposure map with a simple WCS."""
    data = np.full(shape, value, dtype=np.float32)
    data[:, :5] = 0.0
    header = _build_simple_wcs(shape[1], shape[0]).to_header()
    header["TELESCOP"] = "EP"
    header["INSTRUME"] = "FXT"
    header["DETNAM"] = "FXTA"
    header["DATAMODE"] = "FF"
    header["DATE-OBS"] = "2026-01-01T00:00:00"
    header["DATE-END"] = "2026-01-01T00:10:00"
    header["RA_PNT"] = 10.0
    header["DEC_PNT"] = 20.0
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return path


def _write_spectrum(path: Path, ancrfile: str | None = None, respfile: str | None = None) -> Path:
    """Write a minimal FXT spectrum for response-generation tests."""
    primary = fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.float32))
    primary.header.update(_build_simple_wcs(8, 8).to_header())
    counts = np.zeros(16, dtype=np.int32)
    spectrum = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=np.arange(len(counts), dtype=np.int32)),
            fits.Column(name="COUNTS", format="J", array=counts),
        ],
        name="SPECTRUM",
    )
    spectrum.header["TELESCOP"] = "EP"
    spectrum.header["INSTRUME"] = "FXT"
    spectrum.header["DETNAM"] = "FXTA"
    spectrum.header["FILTER"] = "01"
    spectrum.header["DATAMODE"] = "FF"
    spectrum.header["DATE-OBS"] = "2026-01-01T00:00:00"
    spectrum.header["DATE-END"] = "2026-01-01T00:10:00"
    spectrum.header["RA_PNT"] = 10.0
    spectrum.header["DEC_PNT"] = 20.0
    spectrum.header["PA_PNT"] = 0.0
    spectrum.header["DSTYP1"] = "GRADE"
    spectrum.header["DSVAL1"] = "G0:12"
    if ancrfile is not None:
        spectrum.header["ANCRFILE"] = ancrfile
    if respfile is not None:
        spectrum.header["RESPFILE"] = respfile
    gti = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="START", format="D", array=np.array([0.0])),
            fits.Column(name="STOP", format="D", array=np.array([600.0])),
        ],
        name="GTI",
    )
    fits.HDUList([primary, spectrum, gti]).writeto(path, overwrite=True)
    return path


def _write_base_arf(path: Path, values: np.ndarray) -> None:
    """Write a tiny CALDB base ARF file."""
    energies_lo = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    energies_hi = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", array=energies_lo),
            fits.Column(name="ENERG_HI", format="E", array=energies_hi),
            fits.Column(name="SPECRESP", format="D", array=values.astype(np.float64)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def _write_rmf(path: Path, matrix_value: float) -> None:
    """Write a minimal CALDB RMF file."""
    matrix = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", array=np.array([0.5], dtype=np.float32)),
            fits.Column(name="ENERG_HI", format="E", array=np.array([1.0], dtype=np.float32)),
            fits.Column(name="N_GRP", format="I", array=np.array([1], dtype=np.int16)),
            fits.Column(name="F_CHAN", format="PI()", array=np.array([[0]], dtype=np.int16)),
            fits.Column(name="N_CHAN", format="PI()", array=np.array([[1]], dtype=np.int16)),
            fits.Column(name="MATRIX", format="PE()", array=np.array([[matrix_value]], dtype=np.float32)),
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


def _write_vign(path: Path) -> None:
    """Write a synthetic vignetting calibration table."""
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERGY", format="E", array=np.array([0.75, 3.0], dtype=np.float32)),
            fits.Column(name="COEF0", format="E", array=np.array([10.0, 10.0], dtype=np.float32)),
            fits.Column(name="COEF1", format="E", array=np.array([1.0, 1.0], dtype=np.float32)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def _write_beta(path: Path) -> None:
    """Write a synthetic beta-PSF calibration file."""
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="BAND", format="J", array=np.array([1, 2], dtype=np.int32)),
            fits.Column(name="EMIN", format="E", array=np.array([0.5, 2.0], dtype=np.float32)),
            fits.Column(name="EMAX", format="E", array=np.array([1.0, 4.0], dtype=np.float32)),
            fits.Column(name="A1", format="E", array=np.array([1.0, 1.0], dtype=np.float32)),
            fits.Column(name="R1", format="E", array=np.array([5.0, 5.0], dtype=np.float32)),
            fits.Column(name="ALP1", format="E", array=np.array([2.0, 2.0], dtype=np.float32)),
            fits.Column(name="A2", format="E", array=np.array([0.0, 0.0], dtype=np.float32)),
            fits.Column(name="R2", format="E", array=np.array([1.0, 1.0], dtype=np.float32)),
            fits.Column(name="ALP2", format="E", array=np.array([2.0, 2.0], dtype=np.float32)),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def _write_eef(path: Path) -> None:
    """Write a minimal EEF calibration FITS file."""
    radius = np.array([0.0, 2.0, 4.0, 8.0], dtype=np.float32)
    frac_near = np.array([0.0, 0.4, 0.7, 1.0], dtype=np.float32)
    frac_far = np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float32)
    hdus = [fits.PrimaryHDU()]
    for name, frac in (("0arcmin", frac_near), ("10arcmin", frac_far)):
        hdus.append(
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="radius_pixel", format="E", array=radius),
                    fits.Column(name="EEF", format="E", array=frac),
                ],
                name=name,
            )
        )
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_teldef(path: Path) -> None:
    """Write a minimal TELDEF file."""
    header = fits.Header()
    # Match the legacy FXTDAS attitude convention used by ``det2radecpix``:
    # the detector boresight vector is carried along the detector -Z axis and
    # must be rotated onto the sky-pointing +X direction for the on-axis
    # optical axis to land at ``RA_PNT``/``DEC_PNT`` in synthetic tests.
    align = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    for row in range(3):
        for col in range(3):
            header[f"ALIGNM{row + 1}{col + 1}"] = float(align[row, col])
    header["FOCALLEN"] = 3000.0
    header["DET_XSCL"] = 0.075
    header["OPTAXISX"] = 50.5
    header["OPTAXISY"] = 50.5
    fits.PrimaryHDU(header=header).writeto(path, overwrite=True)


def _write_caldb_index(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a minimal CALDB index FITS file."""
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
        fits.Column(name="CAL_CBD", format="64A", array=[row["CAL_CBD"] for row in rows]),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


@pytest.fixture
def fake_caldb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a synthetic CALDB tree for response-generation tests."""
    caldb = tmp_path / "caldb"
    index_dir = caldb / "data/ep/fxt/index"
    arf_dir = caldb / "data/ep/fxt/cpf/arf"
    rmf_dir = caldb / "data/ep/fxt/cpf/rmf"
    vign_dir = caldb / "data/ep/fxt/cpf/vignetting"
    psf_dir = caldb / "data/ep/fxt/cpf/psf"
    eef_dir = caldb / "data/ep/fxt/cpf/eef"
    teldef_dir = caldb / "data/ep/fxt/bcf/teldef"
    for directory in (index_dir, arf_dir, rmf_dir, vign_dir, psf_dir, eef_dir, teldef_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_base_arf(arf_dir / "base_arf.fits", np.array([100.0, 80.0, 50.0]))
    _write_rmf(rmf_dir / "rmf_grade12.fits", 2.0)
    _write_rmf(rmf_dir / "rmf_grade4.fits", 4.0)
    _write_vign(vign_dir / "vign.fits")
    _write_beta(psf_dir / "fxta_beta.fits")
    _write_eef(eef_dir / "fxta_thin_C_K_eef.fits")
    _write_teldef(teldef_dir / "teldef.fits")
    _write_caldb_index(
        index_dir / "caldb.indx",
        [
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "A",
                "FILTER": "1",
                "CAL_CNAM": "SPECRESP",
                "CAL_QUAL": 0,
                "REF_TIME": 60000.0,
                "CAL_FILE": "base_arf.fits",
                "CAL_DIR": "data/ep/fxt/cpf/arf",
                "CAL_XNO": 1,
                "CAL_CBD": "DATAMODE(FF) GRADE(G0:0-12)",
            },
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "A",
                "FILTER": "NONE",
                "CAL_CNAM": "MATRIX",
                "CAL_QUAL": 0,
                "REF_TIME": 60000.0,
                "CAL_FILE": "rmf_grade12.fits",
                "CAL_DIR": "data/ep/fxt/cpf/rmf",
                "CAL_XNO": 1,
                "CAL_CBD": "DATAMODE(FF) GRADE(G0:0-12)",
            },
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "A",
                "FILTER": "NONE",
                "CAL_CNAM": "MATRIX",
                "CAL_QUAL": 0,
                "REF_TIME": 60000.0,
                "CAL_FILE": "rmf_grade4.fits",
                "CAL_DIR": "data/ep/fxt/cpf/rmf",
                "CAL_XNO": 1,
                "CAL_CBD": "DATAMODE(FF) GRADE(G0:4)",
            },
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "A",
                "FILTER": "1",
                "CAL_CNAM": "VIGNET",
                "CAL_QUAL": 0,
                "REF_TIME": 60000.0,
                "CAL_FILE": "vign.fits",
                "CAL_DIR": "data/ep/fxt/cpf/vignetting",
                "CAL_XNO": 1,
                "CAL_CBD": "NONE",
            },
            {
                "TELESCOP": "EP",
                "INSTRUME": "FXT",
                "DETNAM": "A",
                "FILTER": "NONE",
                "CAL_CNAM": "TELDEF",
                "CAL_QUAL": 0,
                "REF_TIME": 60000.0,
                "CAL_FILE": "teldef.fits",
                "CAL_DIR": "data/ep/fxt/bcf/teldef",
                "CAL_XNO": 0,
                "CAL_CBD": "NONE",
            },
        ],
    )
    config = caldb / "caldb.config"
    config.write_text("EP FXT cfg data/ep/fxt/index caldb.indx root data/ep/fxt\n", encoding="utf-8")
    monkeypatch.setenv("CALDB", str(caldb))
    monkeypatch.setenv("CALDBCONFIG", str(config))
    return caldb


def test_region_loader_supports_common_shapes_and_exclusions(tmp_path: Path) -> None:
    """Circle, annulus, ellipse, box, polygon, and exclusions should rasterize."""
    region_path = tmp_path / "shapes.reg"
    region_path.write_text(
        "# Region file format: DS9 version 4.1\n"
        "image\n"
        "circle(20,20,5)\n"
        "-circle(20,20,2)\n"
        "annulus(40,20,2,5)\n"
        "ellipse(20,40,8,4,30)\n"
        "box(40,40,10,6,0)\n"
        "polygon(60,10,70,10,65,20)\n",
        encoding="utf-8",
    )
    region_set = load_region_set(str(region_path), image_shape=(100, 100), wcs=None, oversample=4)
    assert region_set.mask.sum() > 0
    assert np.allclose(region_set.first_positive_center_xy, (19.0, 19.0))
    assert region_set.mask[19, 19] < 1.0


def test_region_loader_rejects_unsupported_shapes(tmp_path: Path) -> None:
    """Unsupported DS9 primitives should fail loudly."""
    region_path = tmp_path / "bad.reg"
    region_path.write_text(
        "# Region file format: DS9 version 4.1\nimage\nline(1,1,10,10)\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedRegionError):
        load_region_set(str(region_path), image_shape=(32, 32), wcs=None)


def test_resolve_source_position_variants(tmp_path: Path) -> None:
    """Source-center resolution should honor explicit overrides before fallback."""
    expfile = _write_exposure(tmp_path / "exp.fits")
    region_path = tmp_path / "src.reg"
    region_path.write_text(
        "# Region file format: DS9 version 4.1\nimage\ncircle(50,50,6)\n",
        encoding="utf-8",
    )
    with fits.open(expfile) as hdul:
        wcs = WCS(hdul[0].header)
    region_set = load_region_set(str(region_path), image_shape=(100, 100), wcs=None)
    pix_source = resolve_source_position(wcs, region_set, srcx=51.0, srcy=52.0)
    assert (pix_source.x, pix_source.y, pix_source.origin) == (50.0, 51.0, "srcx/srcy")
    world_source = resolve_source_position(wcs, region_set, ra=10.0, dec=20.0)
    assert np.allclose((world_source.x, world_source.y), (49.5, 49.5))
    assert world_source.origin == "ra/dec"
    fallback = resolve_source_position(wcs, region_set)
    assert np.allclose((fallback.x, fallback.y), (49.0, 49.0))
    assert fallback.origin == "region-center"


def test_rmf_selection_uses_grade_specific_caldb_row(tmp_path: Path, fake_caldb: Path) -> None:
    """RMF selection should match the grade-derived legacy CALDB policy."""
    specfile = _write_spectrum(tmp_path / "src.pi")
    metadata = read_observation_metadata(str(specfile), preferred_ext=1)
    outfile = tmp_path / "src.rmf"
    generate_rmf(str(specfile), str(outfile), metadata, clobber=True)
    with fits.open(outfile) as hdul:
        assert hdul[1].header["EXTNAME"] == "MATRIX"
        assert hdul[1].header["HDUCLAS3"] == "REDIST"
        assert hdul[1].data["MATRIX"][0][0] == pytest.approx(2.0)


def test_base_arf_selection_matches_grade_range_rows(tmp_path: Path, fake_caldb: Path) -> None:
    """Base ARF lookup should accept CALDB grade ranges such as ``G0:0-12``."""
    specfile = _write_spectrum(tmp_path / "src.pi")
    metadata = read_observation_metadata(str(specfile), preferred_ext=1)
    filepath, extno = resolve_base_arf(metadata)
    assert Path(filepath).name == "base_arf.fits"
    assert extno == 1


def test_rmf_selection_rejects_unsupported_grade(tmp_path: Path, fake_caldb: Path) -> None:
    """RMF lookup should fail immediately for unsupported EP-FXT grades."""
    specfile = _write_spectrum(tmp_path / "src_grade8.pi")
    with fits.open(specfile, mode="update") as hdul:
        hdul[1].header["DSVAL1"] = "G0:20"
        hdul.flush()
    metadata = read_observation_metadata(str(specfile), preferred_ext=1)
    outfile = tmp_path / "src_grade8.rmf"
    with pytest.raises(ValueError, match="Unsupported EP-FXT response grade"):
        generate_rmf(str(specfile), str(outfile), metadata, clobber=True)


def test_base_arf_selection_rejects_unsupported_grade(
    tmp_path: Path,
    fake_caldb: Path,
) -> None:
    """Base ARF lookup should fail immediately for unsupported EP-FXT grades."""
    specfile = _write_spectrum(tmp_path / "src_grade8.pi")
    with fits.open(specfile, mode="update") as hdul:
        hdul[1].header["DSVAL1"] = "G0:8"
        hdul.flush()
    metadata = read_observation_metadata(str(specfile), preferred_ext=1)
    with pytest.raises(ValueError, match="Unsupported EP-FXT response grade"):
        resolve_base_arf(metadata)


def test_caldb_lookup_error_includes_stage_paths_and_candidates(fake_caldb: Path) -> None:
    """Lookup failures should report stage, CALDB paths, and candidate rows."""
    with pytest.raises(RuntimeError, match="stage=cbd") as excinfo:
        find_calibration_file(
            telescope="EP",
            instrument="FXT",
            detname="A",
            filt="1",
            codename="SPECRESP",
            start_date="2026-01-01",
            start_time="00:00:00",
            stop_date="2026-01-01",
            stop_time="00:10:00",
            expr="DATAMODE(BAD) .AND. GRADE(G0:12)",
        )
    message = str(excinfo.value)
    assert "CALDBINDEX=" in message
    assert "row_counts=(" in message
    assert "metadata_candidates=[" in message
    assert "base_arf.fits" in message
    assert "GRADE(G0:0-12)" in message


def test_fxtpsfgen_uses_shared_fxtcaldb_backend(
    fake_caldb: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation PSF mappers should resolve EEF files through the shared CALDB layer."""
    image_path = _write_exposure(tmp_path / "mapper_img.fits", shape=(20, 20), value=1.0)
    monkeypatch.setattr("fxtpsfgen.mapper.compute_optical_axis_pixel", lambda *args, **kwargs: (10.5, 10.5))
    mapper = build_observation_psf_mapper(
        image_path=image_path,
        instrument="fxta",
        filter_name="thin",
        emin_keV=0.5,
        emax_keV=1.0,
    )
    radius_pix, frac = mapper.eef_curve(5.0, energy_keV=0.75)
    assert np.all(np.diff(radius_pix) >= 0.0)
    assert np.all(np.diff(frac) >= -1e-8)
    radius_map = mapper.radius_map(0.90, energy_keV=0.75)
    assert radius_map.shape == (20, 20)
    assert np.all(np.isfinite(mapper.theta_map_arcmin))
    assert float(np.nanmax(mapper.theta_map_arcmin)) > 0.0


def test_observation_metadata_supports_pointing_projection(tmp_path: Path, fake_caldb: Path) -> None:
    """General observation metadata should carry pointing needed for optaxis projection."""
    specfile = _write_spectrum(tmp_path / "src.pi")
    metadata = read_observation_metadata(str(specfile), preferred_ext=1)
    assert metadata.telescope == "EP"
    assert metadata.instrument == "FXT"
    assert metadata.detector_code == "A"
    assert metadata.ra_pnt == pytest.approx(10.0)
    assert metadata.dec_pnt == pytest.approx(20.0)
    assert metadata.pa_pnt == pytest.approx(0.0)


def test_optaxis_projection_fails_when_required_pointing_is_missing(tmp_path: Path, fake_caldb: Path) -> None:
    """Exposure-like products missing ``PA_PNT`` should fail clearly for optaxis projection."""
    expfile = _write_exposure(tmp_path / "exp_missing_pa.fits")
    metadata = read_observation_metadata(str(expfile), preferred_ext=0)
    with fits.open(expfile) as hdul:
        wcs = WCS(hdul[0].header).celestial
    with pytest.raises(ValueError, match="PA_PNT"):
        compute_optical_axis_pixel(metadata, wcs)


def test_run_fxtrspgen_writes_factorized_outputs_and_optional_headers(
    tmp_path: Path,
    fake_caldb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new task should write diagnostic ARF columns and leave PHA untouched by default."""
    specfile = _write_spectrum(tmp_path / "src.pi")
    expfile = _write_exposure(tmp_path / "exp.fits")
    region_path = tmp_path / "src.reg"
    region_path.write_text(
        "# Region file format: DS9 version 4.1\nimage\ncircle(12,50,10)\n-circle(12,50,3)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("fxtrspgen.arf.compute_optical_axis_pixel", lambda *args, **kwargs: (70.5, 50.5))
    outputs = run_fxtrspgen(str(specfile), str(expfile), str(region_path), clobber=True)
    with fits.open(outputs["arf_out"]) as hdul:
        data = hdul[1].data
        assert {"SPECRESP", "BASE_ARF", "VIGN_CORR", "PSF_CORR", "REGCOV_CORR", "TOT_CORR"} <= set(data.names)
        assert np.allclose(data["SPECRESP"], data["BASE_ARF"] * data["TOT_CORR"])
        assert not np.allclose(
            data["TOT_CORR"],
            data["VIGN_CORR"] * data["PSF_CORR"] * data["REGCOV_CORR"],
            rtol=0.0,
            atol=1e-6,
        )
        assert hdul[1].header["FXTRSPNT"] == "Joint total is non-separable"
    with fits.open(outputs["rmf_out"]) as hdul:
        assert hdul[1].header["EXTNAME"] == "MATRIX"
        assert hdul[1].header["TLMIN4"] == 0
        assert hdul[1].header["TLMAX4"] == 1023
    with fits.open(specfile) as hdul:
        assert "ANCRFILE" not in hdul[1].header
        assert "RESPFILE" not in hdul[1].header
    updated_outputs = run_fxtrspgen(
        str(specfile),
        str(expfile),
        str(region_path),
        arf_out=str(tmp_path / "updated.arf"),
        rmf_out=str(tmp_path / "updated.rmf"),
        update_pha=True,
        clobber=True,
    )
    with fits.open(specfile) as hdul:
        assert hdul[1].header["ANCRFILE"] == updated_outputs["arf_out"]
        assert hdul[1].header["RESPFILE"] == updated_outputs["rmf_out"]


def test_repo_fixture_runs_end_to_end_with_synthetic_caldb(
    tmp_path: Path,
    fake_caldb: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One existing repo spectrum/exposure pair should produce OGIP-readable outputs."""
    repo_root = Path(__file__).resolve().parent.parent
    specfile = repo_root / "test2/11900512000/products/fxt_a_11900512000_ff_01_po_evt_6ca_src.pi"
    expfile = repo_root / "test2/11900512000/products/fxt_a_11900512000_ff_01_po_evt_6ca.expo"
    regionfile = tmp_path / "repo.reg"
    with fits.open(specfile) as hdul:
        original_arf = hdul[1].header["ANCRFILE"]
        original_rmf = hdul[1].header["RESPFILE"]
        regionfile.write_text(
            "# Region file format: DS9 version 4.1\n"
            "fk5\n"
            f"circle({hdul[1].header['RA_PNT']:.8f},{hdul[1].header['DEC_PNT']:.8f},60.0\")\n",
            encoding="utf-8",
        )
    monkeypatch.setattr("fxtrspgen.arf.compute_optical_axis_pixel", lambda *args, **kwargs: (299.5, 299.5))
    outputs = run_fxtrspgen(
        str(specfile),
        str(expfile),
        str(regionfile),
        arf_out=str(tmp_path / "repo.arf"),
        rmf_out=str(tmp_path / "repo.rmf"),
        clobber=True,
    )
    with fits.open(outputs["arf_out"]) as hdul:
        assert hdul[1].header["EXTNAME"] == "SPECRESP"
        assert "BASE_ARF" in hdul[1].data.names
    with fits.open(specfile) as hdul:
        assert hdul[1].header["ANCRFILE"] == original_arf
        assert hdul[1].header["RESPFILE"] == original_rmf


def test_observation_psf_mapper_builds_and_roundtrips(fake_caldb: Path, tmp_path: Path) -> None:
    """Observation PSF products should build, serialize, and answer local queries."""
    image_path = _write_exposure(tmp_path / "image.fits")
    with fits.open(image_path, mode="update") as hdul:
        hdul[0].header["FILTER"] = "01"
        hdul[0].header["PA_PNT"] = 0.0
    mapper = build_observation_psf_mapper(
        image_path,
        instrument="fxta",
        filter_name="thin",
        emin_keV=0.3,
        emax_keV=10.0,
    )
    out_path = tmp_path / "obs.psfprod.fits"
    mapper.write(out_path)
    loaded = ObservationPSFMapper.read(out_path)
    radius, frac = loaded.local_eef_curve(50.0, 50.0)
    assert radius.ndim == 1
    assert frac.ndim == 1
    assert np.all(np.diff(frac) >= -1e-8)
    assert loaded.radius_at_position(50.0, 50.0, 0.90) > 0.0
    assert loaded.kernel_at_position(50.0, 50.0).sum() == pytest.approx(1.0)


def test_stacked_psf_mapper_builds_and_answers_weighted_queries(fake_caldb: Path, tmp_path: Path) -> None:
    """Stacked PSF products should combine per-observation mappers on one WCS."""
    image_a = _write_exposure(tmp_path / "image_a.fits")
    image_b = _write_exposure(tmp_path / "image_b.fits")
    for path, ra_pnt in ((image_a, 10.0), (image_b, 10.05)):
        with fits.open(path, mode="update") as hdul:
            hdul[0].header["FILTER"] = "01"
            hdul[0].header["PA_PNT"] = 0.0
            hdul[0].header["RA_PNT"] = ra_pnt
    obs_a = build_observation_psf_mapper(image_a, instrument="fxta", filter_name="thin", emin_keV=0.3, emax_keV=10.0)
    obs_b = build_observation_psf_mapper(image_b, instrument="fxta", filter_name="thin", emin_keV=0.3, emax_keV=10.0)
    obs_a_path = tmp_path / "obs_a.psfprod.fits"
    obs_b_path = tmp_path / "obs_b.psfprod.fits"
    obs_a.write(obs_a_path)
    obs_b.write(obs_b_path)

    weight_a = tmp_path / "weight_a.fits"
    weight_b = tmp_path / "weight_b.fits"
    fits.PrimaryHDU(data=np.ones((100, 100), dtype=np.float32), header=fits.getheader(image_a)).writeto(weight_a, overwrite=True)
    fits.PrimaryHDU(data=np.full((100, 100), 0.5, dtype=np.float32), header=fits.getheader(image_b)).writeto(weight_b, overwrite=True)

    ref_header = fits.getheader(image_a)
    stacked = build_stacked_psf_mapper([obs_a_path, obs_b_path], [weight_a, weight_b], ref_header)
    stacked_path = tmp_path / "stack.psfprod.fits"
    stacked.write(stacked_path)
    loaded = StackedPSFMapper.read(stacked_path)
    radius, frac = loaded.local_eef_curve(50.0, 50.0)
    assert radius.ndim == 1
    assert frac.ndim == 1
    assert np.all(np.diff(frac) >= -1e-8)
    assert loaded.radius_at_position(50.0, 50.0, 0.90) > 0.0
    assert loaded.kernel_at_position(50.0, 50.0).sum() == pytest.approx(1.0)
