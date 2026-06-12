from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

from fxtsrcdet.config import MIN_GAUSSIAN_SIGMA_PIX


def fft_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D image with a 2D kernel using FFTs.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    kernel : np.ndarray
        Convolution kernel array.

    Returns
    -------
    convolved : np.ndarray
        A same-sized convolution result aligned to the input image.
    """
    return signal.fftconvolve(image, kernel, mode="same")


def smooth_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth an image with a Gaussian kernel."""
    return ndimage.gaussian_filter(image, sigma=max(float(sigma), MIN_GAUSSIAN_SIGMA_PIX), mode="constant", cval=0.0)


def stamp_bounds(shape: tuple[int, int], x0: float, y0: float, radius: float) -> tuple[int, int, int, int]:
    """Compute the integer bounds of a local square stamp."""
    h, w = shape
    x_min = max(0, int(math.floor(x0 - radius)))
    x_max = min(w - 1, int(math.ceil(x0 + radius)))
    y_min = max(0, int(math.floor(y0 - radius)))
    y_max = min(h - 1, int(math.ceil(y0 + radius)))
    return y_min, y_max, x_min, x_max


def extract_stamp(image: np.ndarray, x0: float, y0: float, radius: float) -> tuple[np.ndarray, int, int]:
    """Extract a local image cutout around one position."""
    y_min, y_max, x_min, x_max = stamp_bounds(image.shape, x0, y0, radius)
    return image[y_min:y_max + 1, x_min:x_max + 1], y_min, x_min


def embed_kernel(kernel: np.ndarray, shape: tuple[int, int], x0: float, y0: float, x_min: int, y_min: int) -> np.ndarray:
    """Place a PSF kernel into the coordinate frame of a local fit stamp."""
    out = np.zeros(shape, dtype=np.float64)
    ky, kx = kernel.shape
    cy = (ky - 1) / 2.0
    cx = (kx - 1) / 2.0
    yy, xx = np.indices(shape, dtype=np.float64)
    xx_full = xx + x_min
    yy_full = yy + y_min
    kx_idx = np.rint(xx_full - x0 + cx).astype(int)
    ky_idx = np.rint(yy_full - y0 + cy).astype(int)
    good = (
        (kx_idx >= 0)
        & (kx_idx < kx)
        & (ky_idx >= 0)
        & (ky_idx < ky)
    )
    out[good] = kernel[ky_idx[good], kx_idx[good]]
    return out


def shifted_template(template: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift and renormalize a source template inside a local fit stamp.

    Parameters
    ----------
    template : np.ndarray
        Base source template.
    dx : float
        Shift along image x in pixels.
    dy : float
        Shift along image y in pixels.

    Returns
    -------
    shifted : np.ndarray
        Shifted, clipped, normalized template.
    """
    shifted = ndimage.shift(
        template,
        shift=(float(dy), float(dx)),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted = np.clip(shifted, 0.0, None)
    norm = float(np.sum(shifted))
    if norm <= 0.0:
        return template
    return shifted / norm
