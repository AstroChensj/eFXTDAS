"""PSF-related image geometry helpers."""

from __future__ import annotations


def infer_optical_axis(shape: tuple[int, int], optaxis_x: float | None, optaxis_y: float | None) -> tuple[float, float]:
    """Determine the optical-axis position for PSF selection.

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape ``(ny, nx)``.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based pixels.

    Returns
    -------
    optical_axis : tuple[float, float]
        Optical-axis position in 1-based pixels.
    """
    if optaxis_x is not None and optaxis_y is not None:
        return float(optaxis_x), float(optaxis_y)
    h, w = shape
    return 0.5 * (w + 1.0), 0.5 * (h + 1.0)
