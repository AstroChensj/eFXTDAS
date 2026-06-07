from __future__ import annotations

import numpy as np

from fxtpsfgen.mapper import _radial_annulus_fractions, _radial_annulus_fractions_full


def _full_frame_reference(
    radius_edges: np.ndarray,
    weight_map: np.ndarray,
    source_xy: tuple[float, float],
    subpixels: int,
) -> np.ndarray:
    """Run the legacy-style full-frame radial annulus integration."""

    return _radial_annulus_fractions_full(
        radius_edges=np.asarray(radius_edges, dtype=np.float64),
        weight_map=np.asarray(weight_map, dtype=np.float64),
        source_xy=(float(source_xy[0]), float(source_xy[1])),
        subpixels=int(subpixels),
    )


def test_radial_annulus_fractions_matches_full_frame_for_compact_source() -> None:
    radius_edges = np.array([0.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    weight_map = np.zeros((60, 60), dtype=np.float64)
    weight_map[28:33, 29:32] = 1.0
    source_xy = (30.0, 30.0)

    optimized = _radial_annulus_fractions(radius_edges, weight_map, source_xy, subpixels=3)
    reference = _full_frame_reference(radius_edges, weight_map, source_xy, subpixels=3)

    assert np.allclose(optimized, reference, rtol=0.0, atol=1e-10)


def test_radial_annulus_fractions_matches_full_frame_for_extended_support() -> None:
    radius_edges = np.array([0.0, 4.0, 8.0, 16.0, 24.0, 40.0], dtype=np.float64)
    weight_map = np.zeros((120, 120), dtype=np.float64)
    yy, xx = np.indices(weight_map.shape, dtype=np.float64)
    rr = np.sqrt((xx - 60.0) ** 2 + (yy - 60.0) ** 2)
    weight_map[(rr <= 18.0)] = 1.0
    weight_map[(rr > 18.0) & (rr <= 28.0)] = 0.35
    source_xy = (60.0, 60.0)

    optimized = _radial_annulus_fractions(radius_edges, weight_map, source_xy, subpixels=3)
    reference = _full_frame_reference(radius_edges, weight_map, source_xy, subpixels=3)

    assert np.allclose(optimized, reference, rtol=0.0, atol=1e-10)


def test_radial_annulus_fractions_matches_full_frame_near_image_edge() -> None:
    radius_edges = np.array([0.0, 2.0, 6.0, 12.0, 20.0], dtype=np.float64)
    weight_map = np.zeros((80, 80), dtype=np.float64)
    weight_map[1:8, 2:10] = 0.7
    source_xy = (4.0, 4.0)

    optimized = _radial_annulus_fractions(radius_edges, weight_map, source_xy, subpixels=3)
    reference = _full_frame_reference(radius_edges, weight_map, source_xy, subpixels=3)

    assert np.allclose(optimized, reference, rtol=0.0, atol=1e-10)


def test_radial_annulus_fractions_matches_full_frame_for_asymmetric_support() -> None:
    radius_edges = np.array([0.0, 2.0, 5.0, 10.0, 18.0], dtype=np.float64)
    weight_map = np.zeros((90, 90), dtype=np.float64)
    weight_map[40:44, 43:50] = 1.0
    weight_map[44:47, 47:52] = 0.5
    weight_map[38:41, 49:54] = 0.25
    source_xy = (45.0, 42.0)

    optimized = _radial_annulus_fractions(radius_edges, weight_map, source_xy, subpixels=3)
    reference = _full_frame_reference(radius_edges, weight_map, source_xy, subpixels=3)

    assert np.allclose(optimized, reference, rtol=0.0, atol=1e-10)


def test_radial_annulus_fractions_zero_weight_map_returns_zeros() -> None:
    radius_edges = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    weight_map = np.zeros((20, 20), dtype=np.float64)
    result = _radial_annulus_fractions(radius_edges, weight_map, (10.0, 10.0), subpixels=3)
    assert np.allclose(result, np.zeros(len(radius_edges) - 1, dtype=np.float64))
