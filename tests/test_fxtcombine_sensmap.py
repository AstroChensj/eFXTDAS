"""Tests for fxtcombine sensitivity-map workflow integration."""

from __future__ import annotations

import sys

import pytest

from fxtcombine import pipeline
from fxtsensmap import DEFAULT_ECF


def _parse_fxtcombine_args(extra_args: list[str] | None = None):
    """Parse a minimal ``fxtcombine`` command line.

    Parameters
    ----------
    extra_args : list[str] | None, optional
        Additional command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    argv = [
        "input",
        "--ra",
        "1.0",
        "--dec",
        "2.0",
        "--obsid-lst",
        "0001",
    ]
    if extra_args:
        argv.extend(extra_args)
    return pipeline.build_parser().parse_args(argv)


def test_fxtcombine_parser_enables_sensmap_by_default() -> None:
    """The stacked sensitivity map should be generated unless disabled.

    Returns
    -------
    None
    """
    args = _parse_fxtcombine_args()

    assert args.disable_sensmap is False
    assert args.sens_eef == pytest.approx(0.90)
    assert args.sens_ecf == pytest.approx(DEFAULT_ECF)
    assert args.sens_likemin == pytest.approx(6.0)


def test_fxtcombine_parser_accepts_sensmap_overrides() -> None:
    """Sensitivity-map generation should expose the planned CLI knobs.

    Returns
    -------
    None
    """
    args = _parse_fxtcombine_args(
        [
            "--disable-sensmap",
            "--sens-eef",
            "0.75",
            "--sens-ecf",
            "123.5",
            "--sens-likemin",
            "8.0",
        ]
    )

    assert args.disable_sensmap is True
    assert args.sens_eef == pytest.approx(0.75)
    assert args.sens_ecf == pytest.approx(123.5)
    assert args.sens_likemin == pytest.approx(8.0)


def test_main_forwards_sensmap_options(monkeypatch, tmp_path) -> None:
    """The CLI entry point should pass sensitivity-map options to the pipeline.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to intercept the pipeline call.
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    None
    """
    captured = {}

    def _fake_fxtcombine_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "build_cli_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "fxtcombine_pipeline", _fake_fxtcombine_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fxtcombine",
            str(tmp_path),
            "--ra",
            "1.0",
            "--dec",
            "2.0",
            "--obsid-lst",
            "0001",
            "--disable-sensmap",
            "--sens-eef",
            "0.75",
            "--sens-ecf",
            "123.5",
            "--sens-likemin",
            "8.0",
        ],
    )

    pipeline.main()

    assert captured["make_sensmap"] is False
    assert captured["sens_eef"] == pytest.approx(0.75)
    assert captured["sens_ecf"] == pytest.approx(123.5)
    assert captured["sens_likemin"] == pytest.approx(8.0)


def test_build_fxtsensmap_command() -> None:
    """The Stage-6 command should target the stacked products.

    Returns
    -------
    None
    """
    cmd = pipeline._build_fxtsensmap_command(
        "/tmp/stack bkg.fits",
        "/tmp/stack exp.fits",
        "/tmp/stack psfprod.fits",
        "/tmp/stack sensmap.fits",
        0.9,
        DEFAULT_ECF,
        6.0,
        jobs=4,
    )

    assert cmd == (
        'fxtsensmap --bkgmap "/tmp/stack bkg.fits" '
        '--expmap "/tmp/stack exp.fits" '
        '--psfprod "/tmp/stack psfprod.fits" '
        "--eef 0.9 "
        f"--ecf {float(DEFAULT_ECF)} "
        "--likemin 6.0 "
        "--jobs 4 "
        '--out "/tmp/stack sensmap.fits"'
    )
