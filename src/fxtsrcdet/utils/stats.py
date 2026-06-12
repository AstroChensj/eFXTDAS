from __future__ import annotations

import numpy as np
from scipy import special


def inverse_normal_survival(p: float) -> float:
    """Convert a one-sided Gaussian tail probability into a z-score.

    Parameters
    ----------
    p : float
        Survival probability for a standard normal variate.

    Returns
    -------
    z : float
        The z value such that ``P(Z > z) = p`` for ``Z ~ N(0, 1)``.
        E.g., ``p = 1e-6`` corresponds to ``z ≈ 4.753``.
    """
    p = float(np.clip(p, 1e-300, 1.0 - 1e-16))
    return float(np.sqrt(2.0) * special.erfcinv(2.0 * p))


def gaussian_sf(z: np.ndarray) -> np.ndarray:
    """Evaluate the Gaussian survival function element-wise.

    Parameters
    ----------
    z : np.ndarray
        Array of standard-normal z values.

    Returns
    -------
    probability : np.ndarray
        An array containing ``P(Z > z)`` for each element.
        E.g., ``z = 4.753`` corresponds to ``P(Z > z) ≈ 1e-6``.
    """
    return special.ndtr(-z)
