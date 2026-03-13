from __future__ import annotations

import math

import numpy as np
from scipy import optimize, signal, special

from fxtsrcdet.config import (
    AMPLITUDE_FIT_UPPER_PAD,
    AMPLITUDE_FIT_UPPER_SCALE,
    AMPLITUDE_FIT_XATOL,
    EPS,
    FIT_EXTENDED_BETA,
    FIT_EXTENDED_MAX_SHIFT_PIX,
    FIT_POINT_MAX_SHIFT_PIX,
)
from fxtsrcdet.utils.imageops import shifted_template
from fxtsrcdet.models import beta_model_kernel


def cash_stat(data: np.ndarray, model: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    """Evaluate the Cash statistic for image counts and a positive model.

    Parameters
    ----------
    data : np.ndarray
        Observed counts image or stamp.
    model : np.ndarray
        Expected counts model on the same grid.
    valid_mask : np.ndarray | None
        Optional boolean mask selecting valid fit pixels.

    Returns
    -------
    statistic : float
        Cash statistic value. Smaller value means better fit. Usually > 0.
    """
    if valid_mask is not None:
        data = data[valid_mask]
        model = model[valid_mask]
    model = np.maximum(model, EPS)
    return float(2.0 * np.sum(model - data * np.log(model)))


def cash_delta_to_like(delta_c: float, dof: float) -> float:
    """Convert a Cash-statistic improvement into an eSASS-like likelihood.

    Parameters
    ----------
    delta_c : float
        Cash-statistic improvement relative to the null model.
    dof : float
        Effective number of additional free parameters.

    Returns
    -------
    likelihood : float
        ``-ln(P)`` from the incomplete gamma survival probability.
    """
    delta_c = max(float(delta_c), 0.0)
    dof = max(float(dof), 1e-6)
    p_value = float(special.gammaincc(0.5 * dof, 0.5 * delta_c))
    return float(-math.log(max(p_value, 1e-300)))


def fit_amplitude_cash(
    data: np.ndarray,
    background: np.ndarray,
    template: np.ndarray,
    exposure: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Fit a non-negative source amplitude on top of a background map.

    Parameters
    ----------
    data : np.ndarray
        Observed counts image or stamp.
    background : np.ndarray
        Background model on the same grid.
    template : np.ndarray
        Source template shape.
    exposure : np.ndarray | None
        Optional local exposure map used to truncate the template.
    valid_mask : np.ndarray | None
        Optional boolean mask selecting valid fit pixels.

    Returns
    -------
    fit_result : tuple[float, float, np.ndarray]
        ``(amplitude, cash_statistic, full_model)``.
    """
    template = np.clip(template, 0.0, None)
    if exposure is not None:
        exp_norm = exposure / max(float(np.max(exposure)), EPS)
        template = template * np.clip(exp_norm, 0.0, None)
    norm = float(np.sum(template))
    if norm <= 0.0:
        model = np.maximum(background, EPS)
        return 0.0, cash_stat(data, model, valid_mask=valid_mask), model
    template = template / norm

    def objective(amp: float) -> float:
        model = background + max(float(amp), 0.0) * template
        return cash_stat(data, model, valid_mask=valid_mask)

    if valid_mask is not None:
        data_fit = data[valid_mask]
        bkg_fit = background[valid_mask]
    else:
        data_fit = data
        bkg_fit = background
    upper = max(float(np.sum(np.clip(data_fit - bkg_fit, 0.0, None))), 1.0)
    result = optimize.minimize_scalar(
        objective,
        bounds=(0.0, upper * AMPLITUDE_FIT_UPPER_SCALE + AMPLITUDE_FIT_UPPER_PAD),
        method="bounded",
        options={"xatol": AMPLITUDE_FIT_XATOL},
    )
    amp = max(float(result.x), 0.0)
    model = np.maximum(background + amp * template, EPS)
    return amp, cash_stat(data, model, valid_mask=valid_mask), model


def fit_point_position_cash(
    data: np.ndarray,
    background: np.ndarray,
    template: np.ndarray,
    exposure: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    max_shift: float = FIT_POINT_MAX_SHIFT_PIX,
) -> tuple[float, float, float, float, np.ndarray]:
    """Fit local source position and amplitude for a point-source model.

    Parameters
    ----------
    data : np.ndarray
        Local image stamp.
    background : np.ndarray
        Local background stamp.
    template : np.ndarray
        Point-source PSF template.
    exposure : np.ndarray | None
        Optional local exposure stamp.
    valid_mask : np.ndarray | None
        Optional boolean mask selecting valid fit pixels.
    max_shift : float
        Maximum allowed fit shift in pixels per axis.

    Returns
    -------
    fit_result : tuple[float, float, float, float, np.ndarray]
        ``(dx, dy, amplitude, cash_statistic, model_image)``.

    Notes
    -----
    Currently performs a brute-force grid search over shifts of the template 
    center, for the sake of speed. More sophisticated optimization could be 
    implemented in the future if needed. 
    """
    best: tuple[float, float, float, float, np.ndarray] | None = None
    shifts = np.array([-max_shift, 0.0, max_shift], dtype=np.float64)
    for dy in shifts:
        for dx in shifts:
            shifted = shifted_template(template, dx, dy)
            amp, cstat, model = fit_amplitude_cash(data, background, shifted, exposure=exposure, valid_mask=valid_mask)
            candidate = (float(dx), float(dy), float(amp), float(cstat), model)
            if best is None or candidate[3] < best[3]:
                best = candidate
    assert best is not None
    return best


def fit_extended_position_cash(
    data: np.ndarray,
    background: np.ndarray,
    point_template: np.ndarray,
    exposure: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    beta: float = FIT_EXTENDED_BETA,
    max_shift: float = FIT_EXTENDED_MAX_SHIFT_PIX,
    rc_grid: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Fit local source position, extent, and amplitude for an extended model.

    Parameters
    ----------
    data : np.ndarray
        Local image stamp.
    background : np.ndarray
        Local background stamp.
    point_template : np.ndarray
        Point-source PSF template.
    exposure : np.ndarray | None
        Optional local exposure stamp.
    valid_mask : np.ndarray | None
        Optional boolean mask selecting valid fit pixels.
    beta : float
        Beta-model slope parameter. Defaults to ``FIT_EXTENDED_BETA``, which
        is currently set to the standard conservative cluster-like value.
    max_shift : float
        Maximum allowed fit shift in pixels per axis.
    rc_grid : np.ndarray | None
        Candidate beta-model core radii in pixels. When omitted, the helper
        uses its internal default trial grid.

    Returns
    -------
    fit_result : tuple[float, float, float, float, float, np.ndarray, np.ndarray]
        ``(dx, dy, rc_pix, amplitude, cash_statistic, model_image, source_template)``.
    """
    def build_ext_template(dx: float, dy: float, rc_pix: float) -> np.ndarray:
        shifted = shifted_template(point_template, dx, dy)
        beta_kernel = beta_model_kernel(point_template.shape, rc_pix, beta=beta)
        ext_kernel = signal.fftconvolve(shifted, beta_kernel, mode="same")
        ext_kernel = np.clip(ext_kernel, 0.0, None)
        return ext_kernel / max(float(np.sum(ext_kernel)), EPS)

    best: tuple[float, float, float, float, float, np.ndarray, np.ndarray] | None = None
    shifts = np.array([-max_shift, 0.0, max_shift], dtype=np.float64)
    if rc_grid is None:
        rc_grid = np.array([0.5, 1.5, 3.0, 6.0], dtype=np.float64)
    else:
        rc_grid = np.asarray(rc_grid, dtype=np.float64)
    for dy in shifts:
        for dx in shifts:
            for rc_pix in rc_grid:
                ext_template = build_ext_template(float(dx), float(dy), float(rc_pix))
                amp, cstat, model = fit_amplitude_cash(
                    data,
                    background,
                    ext_template,
                    exposure=exposure,
                    valid_mask=valid_mask,
                )
                candidate = (float(dx), float(dy), float(rc_pix), float(amp), float(cstat), model, ext_template)
                if best is None or candidate[4] < best[4]:
                    best = candidate
    assert best is not None
    return best


def fit_group_amplitudes_cash(
    data: np.ndarray,
    background: np.ndarray,
    templates: list[np.ndarray],
    exposure: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit non-negative amplitudes for a local group of point-source templates.

    Parameters
    ----------
    data : np.ndarray
        Local image stamp.
    background : np.ndarray
        Local background stamp.
    templates : list[np.ndarray]
        Source templates for all grouped sources.
    exposure : np.ndarray | None
        Optional local exposure stamp.
    valid_mask : np.ndarray | None
        Optional boolean mask selecting valid fit pixels.

    Returns
    -------
    fit_result : tuple[np.ndarray, float, np.ndarray]
        ``(amplitudes, cash_statistic, full_model)``.
    """
    if len(templates) == 0:
        model = np.maximum(background, EPS)
        return np.zeros(0, dtype=np.float64), cash_stat(data, model, valid_mask=valid_mask), model

    norm_templates: list[np.ndarray] = []
    for template in templates:
        temp = np.clip(np.asarray(template, dtype=np.float64), 0.0, None)
        if exposure is not None:
            exp_norm = exposure / max(float(np.max(exposure)), EPS)
            temp = temp * np.clip(exp_norm, 0.0, None)
        tsum = float(np.sum(temp))
        if tsum <= 0.0:
            temp = np.zeros_like(background, dtype=np.float64)
        else:
            temp = temp / tsum
        norm_templates.append(temp)

    def objective(amps: np.ndarray) -> float:
        model = np.array(background, dtype=np.float64, copy=True)
        for amp, template in zip(amps, norm_templates, strict=False):
            model += max(float(amp), 0.0) * template
        return cash_stat(data, model, valid_mask=valid_mask)

    if valid_mask is not None:
        resid = np.clip(data[valid_mask] - background[valid_mask], 0.0, None)
    else:
        resid = np.clip(data - background, 0.0, None)
    total_excess = max(float(np.sum(resid)), 1.0)
    x0 = np.full(len(norm_templates), total_excess / max(len(norm_templates), 1), dtype=np.float64)
    bounds = [(0.0, total_excess * 2.0 + 1.0)] * len(norm_templates)
    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-8},
    )
    amps = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, None)
    model = np.array(background, dtype=np.float64, copy=True)
    for amp, template in zip(amps, norm_templates, strict=False):
        model += float(amp) * template
    model = np.maximum(model, EPS)
    return amps, cash_stat(data, model, valid_mask=valid_mask), model
