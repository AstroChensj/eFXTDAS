from __future__ import annotations

import io
import logging

import numpy as np

from fxtsrcdet.detect import ScaleResult, combine_scales


class CountingPSFMapper:
    """Minimal PSF mapper stub that counts local R90 queries."""

    def __init__(self, radius_pix: float = 4.0) -> None:
        self.radius_pix = float(radius_pix)
        self.calls: list[tuple[float, float, float]] = []

    def radius_at_position(self, x_ima: float, y_ima: float, frac_value: float, energy_keV: float | None = None) -> float:
        self.calls.append((float(x_ima), float(y_ima), float(frac_value)))
        return self.radius_pix


def _build_scale_result(peak_yx: tuple[int, int], scale: float) -> ScaleResult:
    """Create one simple scale result with a single compact source island."""

    shape = (9, 9)
    source_mask = np.zeros(shape, dtype=bool)
    peak_mask = np.zeros(shape, dtype=bool)
    correlation = np.zeros(shape, dtype=np.float64)
    background = np.full(shape, 0.1, dtype=np.float64)
    significance = np.ones(shape, dtype=np.float64)
    py, px = peak_yx

    source_mask[py, px] = True
    source_mask[max(py - 1, 0), px] = True
    source_mask[min(py + 1, shape[0] - 1), px] = True
    peak_mask[py, px] = True
    correlation[py, px] = 12.0 + float(scale)
    significance[py, px] = 1e-8

    return ScaleResult(
        scale=float(scale),
        correlation=correlation,
        background=background,
        significance=significance,
        source_mask=source_mask,
        peak_mask=peak_mask,
    )


def test_combine_scales_caches_one_psf_lookup_per_candidate_and_logs_timings() -> None:
    image = np.zeros((9, 9), dtype=np.float64)
    image[4, 4] = 20.0
    image[3, 4] = 5.0
    image[5, 4] = 4.0
    per_scale = [
        _build_scale_result((4, 4), scale=2.0),
        _build_scale_result((4, 4), scale=4.0),
    ]
    psf_mapper = CountingPSFMapper(radius_pix=3.5)

    logger = logging.getLogger("test.fxtsrcdet.stage1")
    logger.handlers.clear()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    rows, agg_mask, best_sig = combine_scales(
        image=image,
        per_scale=per_scale,
        ellsigma=3.0,
        z_thresh=4.0,
        psf_mapper=psf_mapper,
        logger=logger,
    )

    assert len(psf_mapper.calls) == 2
    assert len(rows) == 1
    assert rows[0].psf_r90_pix == 3.5
    assert rows[0].cluster_link_radius_pix > 0.0
    assert agg_mask.shape == image.shape
    assert best_sig.shape == image.shape

    log_text = stream.getvalue()
    assert "combine_scales candidate extraction:" in log_text
    assert "combine_scales PSF r90 cache:" in log_text
    assert "combine_scales clustering:" in log_text
    assert "combine_scales total runtime:" in log_text

