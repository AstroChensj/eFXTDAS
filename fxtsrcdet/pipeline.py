#!/usr/bin/env python3
"""
A lightweight, pure-Python source detection algorithm for X-ray images,
which includes the following steps:

- Robust initial source detection with wavelet filtering;
- Source removal and background map creation;
- PSF-aware source identification against local background (detected or
  bkg fluctuation) and characterization (point source or extended).

This wavelet detection follows the high-level algorithm described in the Chandra
Detect Reference Manual (CIAO 3.4, 2006):
- multi-scale Mexican-hat correlation,
- significance thresholding,
- iterative source cleansing for background estimation,
- cross-scale source pixel merging.

For a more pedagogical explanation of how a Mexican-hat filter highlights
source-like excesses above local background, see the user documentation for
``fxtsrcdet``.

Why PSF information can still help
----------------------------------
The wavelet-detected source footprint is a statistical detection region: it
marks pixels that look source-like above background. That is useful for finding
sources, but its size depends on thresholds, source brightness, and the chosen
wavelet scales. A PSF-based region answers a different question: how large and
what shape should a point source physically appear at this detector location?
For Chandra this matters because the PSF changes strongly across the field. PSF
information therefore helps define more stable source apertures for photometry
and source characterization, even though the core wavelet detection step does
not fundamentally require the PSF.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fxtpsf_helpers import build_mission_psf_context, load_radius_map_bundle
from fxtsrcdet.config import BACKGROUND_CARVE_MIN_COUNTS, BACKGROUND_CARVE_MIN_SUPPORT_SCALES
from fxtsrcdet.background import create_background_map
from fxtsrcdet.catalog import classify_sources_with_psf, finalize_catalog_columns, prune_nearby_sources
from fxtsrcdet.detect import detect_sources
from fxtsrcdet.utils.io import (
    load_header,
    load_pipeline_inputs,
    save_img,
    write_ds9_regions,
    write_ds9_sky_regions,
    write_sources_fits,
)
from fxtsrcdet.utils.logger import build_cli_logger, emit
from fxtsrcdet.utils.wcs import augment_rows_with_wcs, infer_pixel_scale_arcsec


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the full source-detection pipeline.

    Parameters
    ----------
    mission : str
        Mission PSF model identifier.
    instrument : str | None
        Mission instrument or detector arm.
    filter_name : str | None
        Mission filter state.
    emin_keV : float | None
        Lower energy bound in keV.
    emax_keV : float | None
        Upper energy bound in keV.
    background_sigma_grid : tuple[float, ...]
        Gaussian smoothing scales in pixels available to the adaptive
        background model.
    scales : tuple[float, ...]
        Wavelet scales in pixels.
    sigthresh : float
        Detection significance threshold.
    bkgsigthresh : float
        Background-cleansing significance threshold.
    maxiter : int
        Maximum cleansing iterations per scale.
    iterstop : float
        Minimum fractional update needed to continue cleansing.
    expthresh : float
        Minimum relative exposure allowed in the analysis.
    ellsigma : float
        Ellipse size scale factor for output regions.
    min_det_like : float
        Minimum detection likelihood for a non-background source.
    min_ext_like : float
        Minimum extent likelihood for an extended classification.
    ecf : float
        Energy conversion factor used to convert count rate into flux.
    show_progress : bool
        Whether to show a progress bar during catalog fitting.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based pixels.
    include_background : bool
        Whether to keep rows classified as background in the returned catalog.
        The default ``False`` produces a science-style catalog that excludes
        background classifications.
    prune_sources : bool
        Whether to suppress nearby duplicate detections after catalog fitting.
        The default ``True`` produces a science-style catalog.
    debug_columns : bool
        Whether CSV output should include internal/debug columns.
    eefmap : Path | None
        Optional precomputed EEF-radius map product from ``fxteefmap``.
    analysis_mask : Path | None
        Optional user-supplied boolean analysis mask FITS image. Non-zero
        pixels are treated as globally valid for detection, background
        estimation, and fitting.
    logger : logging.Logger | None
        Optional logger used for stage-by-stage pipeline messages. If omitted,
        the pipeline falls back to plain ``print``.
    """

    mission: str = "ep-fxt"
    instrument: str | None = None
    filter_name: str | None = None
    emin_keV: float | None = None
    emax_keV: float | None = None
    background_sigma_grid: tuple[float, ...] = (4.0, 8.0, 16.0, 32.0, 64.0)
    scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    sigthresh: float = 1e-6
    bkgsigthresh: float = 1e-3
    maxiter: int = 2
    iterstop: float = 1e-4
    expthresh: float = 0.1
    ellsigma: float = 3.0
    min_det_like: float = 6.0
    min_ext_like: float = 6.0
    ecf: float = 0.0
    show_progress: bool = False
    optaxis_x: float | None = None
    optaxis_y: float | None = None
    include_background: bool = False
    prune_sources: bool = True
    debug_columns: bool = False
    eefmap: Path | None = None
    analysis_mask: Path | None = None
    logger: logging.Logger | None = None


def _science_candidates(rows: list[Any]) -> list[Any]:
    """Filter provisional wavelet candidates for carving and PSF-aware fitting."""
    kept = []
    for row in rows:
        if len(getattr(row, "support_scales", [])) < BACKGROUND_CARVE_MIN_SUPPORT_SCALES:
            continue
        if float(getattr(row, "counts", 0.0)) < BACKGROUND_CARVE_MIN_COUNTS:
            continue
        kept.append(row)
    return kept


def fxtsrcdet_pipeline(
    image: np.ndarray | str | Path,
    exposure: np.ndarray | str | Path | None = None,
    analysis_mask: np.ndarray | str | Path | None = None,
    wcs: Any | None = None,
    mission: str = "ep-fxt",
    instrument: str | None = None,
    filter_name: str | None = None,
    emin_keV: float | None = None,
    emax_keV: float | None = None,
    background_sigma_grid: Iterable[float] = (4.0, 8.0, 16.0, 32.0, 64.0),
    scales: Iterable[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    sigthresh: float = 1e-6,
    bkgsigthresh: float = 1e-3,
    maxiter: int = 2,
    iterstop: float = 1e-4,
    expthresh: float = 0.1,
    ellsigma: float = 3.0,
    min_det_like: float = 6.0,
    min_ext_like: float = 6.0,
    ecf: float = 0.0,
    show_progress: bool = False,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    include_background: bool = False,
    prune_sources: bool = True,
    debug_columns: bool = False,
    eefmap: str | Path | None = None,
    logger: logging.Logger | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run the full source-detection and catalog-preparation workflow.

    Parameters
    ----------
    image : np.ndarray | str | Path
        Input counts image array or image-file path.
    exposure : np.ndarray | str | Path | None
        Optional exposure map array or file path matched to ``image``.
    analysis_mask : np.ndarray | str | Path | None
        Optional user-supplied boolean analysis mask array or file path matched
        to ``image``.
    wcs : Any | None
        Optional celestial WCS for sky-coordinate outputs.
    mission : str
        Mission PSF model identifier.
    instrument : str | None
        Mission instrument or detector arm.
    filter_name : str | None
        Mission filter state.
    emin_keV : float | None
        Lower image energy bound in keV.
    emax_keV : float | None
        Upper image energy bound in keV.
    background_sigma_grid : Iterable[float]
        Gaussian smoothing scales in pixels available to the adaptive
        background model.
    scales : Iterable[float]
        Wavelet scales in pixels.
    sigthresh : float
        Detection significance threshold.
    bkgsigthresh : float
        Background-cleansing significance threshold.
    maxiter : int
        Maximum cleansing iterations per scale.
    iterstop : float
        Minimum fractional update needed to continue cleansing.
    expthresh : float
        Minimum relative exposure allowed in the analysis.
    ellsigma : float
        Ellipse size scale factor for output regions.
    min_det_like : float
        Minimum detection likelihood for a non-background source.
    min_ext_like : float
        Minimum extent likelihood for an extended classification.
    ecf : float
        Energy conversion factor used to convert count rate into flux.
    show_progress : bool
        Whether to show progress bars during catalog fitting.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based pixels.
    include_background : bool
        Whether to keep rows classified as background in the returned catalog.
        The default ``False`` produces a science-style catalog.
    prune_sources : bool
        Whether to suppress nearby duplicate detections after catalog fitting.
        The default ``True`` produces a science-style catalog.
    debug_columns : bool
        Whether the output writer should include debug columns when the returned
        rows are later serialized.
    eefmap : str | Path | None
        Optional precomputed EEF-radius map product from ``fxteefmap``.
    logger : logging.Logger | None
        Optional logger used for stage-by-stage status messages. If omitted,
        messages are printed directly.
    config : PipelineConfig | None
        Optional pipeline configuration object. When provided, it overrides the
        individual pipeline keyword arguments above.

    Returns
    -------
    pipeline_result : dict[str, Any]
        Dictionary containing the full pipeline products. Keys are:

        ``rows`` : list[CatalogRow]
            Final catalog rows after optional background filtering and optional
            duplicate pruning. These rows contain both the public science catalog
            fields and, internally, the extra diagnostics needed for debug output.

        ``per_scale`` : list[ScaleResult]
            Per-wavelet-scale intermediate detection products. Each ``ScaleResult``
            stores the wavelet correlation image, the iterative background estimate,
            the significance map, and the scale-specific source and peak masks.
            This is primarily useful for debugging and notebook review. See examples 
            below for usage.

        ``agg_mask`` : np.ndarray
            Boolean aggregate source mask formed from the accepted wavelet support
            across scales.

        ``best_sig`` : np.ndarray
            Per-pixel map of the best significance retained across all tested wavelet
            scales.

        ``background_map`` : np.ndarray
            Final exposure-aware, source-masked background map used by the catalog
            fitting stage.

        ``psf_context`` : MissionPSFContext
            Mission-specific PSF/EEF lookup context used for local PSF radii and
            fitting templates.

        ``pixel_scale_arcsec`` : float
            Image pixel scale in arcsec per pixel, inferred from WCS when available
            and otherwise taken from the mission PSF defaults.

    Notes
    -----
    The workflow is staged as follows:

    1. Load the image, optional exposure map, and optional WCS.
    2. Build the mission-specific PSF context and optional precomputed EEF-radius map bundle.
    3. Run multi-scale wavelet detection to create coarse source candidates.
    4. Build a source-masked, exposure-aware background map.
    5. Refit the candidates with PSF-aware single-source and grouped-source likelihood models. The latter helps deblend close pairs.
    6. Augment the fitted rows with WCS-derived sky coordinates and finalize the catalog columns.
    7. Optionally drop background-classified rows and prune nearby duplicates.

    Examples
    --------
    Run the full pipeline on FITS inputs and inspect the final catalog rows:

    >>> from fxtsrcdet.pipeline import fxtsrcdet_pipeline
    >>> result = fxtsrcdet_pipeline(
    ...     image="test/evt_stack_cts.fits",
    ...     exposure="test/evt_stack_exp.fits",
    ...     mission="ep-fxt",
    ...     instrument="fxta",
    ...     filter_name="open",
    ...     emin_keV=0.3,
    ...     emax_keV=10.0,
    ... )
    >>> rows = result["rows"]
    >>> len(rows) >= 1
    True
    >>> rows[0].id_src >= 1
    True

    Keep all classified rows for debugging, including background-classified candidates:

    >>> debug_result = fxtsrcdet_pipeline(
    ...     image="test/evt_stack_cts.fits",
    ...     exposure="test/evt_stack_exp.fits",
    ...     include_background=True,
    ...     prune_sources=False,
    ... )
    >>> len(debug_result["rows"]) >= len(rows)
    True

    Access intermediate products for review plots or notebook diagnostics:

    >>> per_scale = result["per_scale"]
    >>> background_map = result["background_map"]
    >>> agg_mask = result["agg_mask"]
    >>> per_scale[0].scale > 0
    True
    >>> background_map.shape == agg_mask.shape
    True

    Plot one wavelet scale detection for visual inspection:

    >>> import matplotlib.pyplot as plt
    >>> scale_result = result["per_scale"][0]   # using the first scale as an example
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    >>> axes[0].imshow(scale_result.background, origin="lower", cmap="gray")
    <...>
    >>> axes[0].set_title(f"Background @ scale={scale_result.scale:g}")
    Text(...)
    >>> axes[1].imshow(scale_result.correlation, origin="lower", cmap="magma")
    <...>
    >>> axes[1].set_title("Wavelet correlation")
    Text(...)
    >>> axes[2].imshow(scale_result.source_mask, origin="lower", cmap="gray")
    <...>
    >>> axes[2].set_title("Source mask")
    Text(...)
    """
    cfg = config or PipelineConfig(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
        background_sigma_grid=tuple(float(sigma) for sigma in background_sigma_grid),
        scales=tuple(float(scale) for scale in scales),
        sigthresh=sigthresh,
        bkgsigthresh=bkgsigthresh,
        maxiter=maxiter,
        iterstop=iterstop,
        expthresh=expthresh,
        ellsigma=ellsigma,
        min_det_like=min_det_like,
        min_ext_like=min_ext_like,
        ecf=ecf,
        show_progress=show_progress,
        optaxis_x=optaxis_x,
        optaxis_y=optaxis_y,
        include_background=include_background,
        prune_sources=prune_sources,
        debug_columns=debug_columns,
        eefmap=None if eefmap is None else Path(eefmap),
        analysis_mask=None if analysis_mask is None else Path(analysis_mask),
        logger=logger,
    )
    active_logger = cfg.logger
    emit(active_logger, "info", "================================")
    emit(active_logger, "info", "**** Welcome to FXTSRCDET! ****")
    emit(active_logger, "info", "================================")
    emit(active_logger, "info", "**** Input Checklist ****")
    emit(active_logger, "info", f"  image = {image}")
    emit(active_logger, "info", f"  exposure = {exposure if exposure is not None else 'None'}")
    emit(active_logger, "info", f"  analysis mask = {analysis_mask if analysis_mask is not None else cfg.analysis_mask if cfg.analysis_mask is not None else 'None'}")
    emit(active_logger, "info", f"  wcs supplied = {'yes' if wcs is not None else 'no'}")
    emit(active_logger, "info", f"  mission = {cfg.mission}")
    emit(active_logger, "info", f"  instrument = {cfg.instrument if cfg.instrument is not None else 'None'}")
    emit(active_logger, "info", f"  filter = {cfg.filter_name if cfg.filter_name is not None else 'None'}")
    emit(active_logger, "info", f"  energy band = {cfg.emin_keV} to {cfg.emax_keV} keV")
    emit(active_logger, "info", f"  background sigma grid = {', '.join(f'{sigma:g}' for sigma in cfg.background_sigma_grid)}")
    emit(active_logger, "info", f"  scales = {', '.join(f'{scale:g}' for scale in cfg.scales)}")
    emit(active_logger, "info", f"  eef map = {cfg.eefmap if cfg.eefmap is not None else 'None'}")
    emit(active_logger, "info", f"  include background rows = {cfg.include_background}")
    emit(active_logger, "info", f"  prune nearby sources = {cfg.prune_sources}")
    emit(active_logger, "info", f"  debug columns = {cfg.debug_columns}")

    mask_input = analysis_mask if analysis_mask is not None else cfg.analysis_mask
    image_data, exposure_data, analysis_mask_data, wcs = load_pipeline_inputs(image, exposure, mask_input, wcs)
    emit(active_logger, "info", f"Loaded image with shape={image_data.shape}")
    if exposure_data is None:
        emit(active_logger, "warning", "No exposure map supplied; exposure-aware masking and photometry will be less reliable")
    if analysis_mask_data is not None:
        emit(active_logger, "info", f"Loaded analysis mask with {int(np.count_nonzero(analysis_mask_data))} valid pixel(s)")
    if wcs is None:
        emit(active_logger, "warning", "No celestial WCS available; sky coordinates and sky regions will be unavailable")

    psf_context = build_mission_psf_context(
        mission=cfg.mission,
        instrument=cfg.instrument,
        filter_name=cfg.filter_name,
        emin_keV=cfg.emin_keV,
        emax_keV=cfg.emax_keV,
    )
    eef_radius_maps = load_radius_map_bundle(cfg.eefmap) if cfg.eefmap is not None else None
    if eef_radius_maps is None:
        emit(active_logger, "info", "No precomputed EEF map provided; falling back to per-source local EEF lookup")
    else:
        emit(active_logger, "info", f"Loaded precomputed EEF map bundle from {cfg.eefmap}")
    pixel_scale_arcsec = infer_pixel_scale_arcsec(wcs, psf_context.default_pixel_scale_arcsec)
    emit(active_logger, "info", f"Using pixel scale = {pixel_scale_arcsec:.4f} arcsec/pixel")

    #--- create a coarse candidate source list (``rows``) with wavelet detection
    emit(active_logger, "info", "**** Stage 1: Multi-Scale Wavelet Detection ****")
    emit(active_logger, "info", "Running wavelet detections ...")
    rows, per_scale, agg_mask, best_sig = detect_sources(
        image=image_data,
        exposure=exposure_data,
        analysis_mask=analysis_mask_data,
        scales=cfg.scales,
        sigthresh=cfg.sigthresh,
        bkgsigthresh=cfg.bkgsigthresh,
        maxiter=cfg.maxiter,
        iterstop=cfg.iterstop,
        expthresh=cfg.expthresh,
        ellsigma=cfg.ellsigma,
        psf_context=psf_context,
        pixel_scale_arcsec=pixel_scale_arcsec,
        optaxis_x=cfg.optaxis_x,
        optaxis_y=cfg.optaxis_y,
        eef_radius_maps=eef_radius_maps,
    )
    emit(active_logger, "info", f"Wavelet detection produced {len(rows)} provisional candidate(s) across {len(per_scale)} scale(s)")
    emit(active_logger, "info", f"Source counts: raw wavelet candidates = {len(rows)}")
    if len(rows) == 0:
        emit(active_logger, "warning", "No wavelet candidates were found; downstream catalog will be empty")

    #--- keep only science-style candidates for both background carving and fitting
    fit_rows = _science_candidates(rows)
    emit(
        active_logger,
        "info",
        f"Science candidate selection for carving/fitting: {len(rows)} -> {len(fit_rows)} "
        f"(support_scales >= {BACKGROUND_CARVE_MIN_SUPPORT_SCALES} AND counts >= {BACKGROUND_CARVE_MIN_COUNTS:g})",
    )
    emit(active_logger, "info", f"Source counts: science candidates after filtering = {len(fit_rows)}")
    if len(fit_rows) == 0:
        emit(active_logger, "warning", "No candidates passed the science-candidate selection; downstream catalog will be empty")

    #--- carve out candidates and smooth to create background map
    emit(active_logger, "info", "**** Stage 2: Source-Masked Background Map ****")
    emit(active_logger, "info", "Generating background map ...")
    background_map = create_background_map(
        image_data,
        fit_rows,
        psf_context=psf_context,
        pixel_scale_arcsec=pixel_scale_arcsec,
        exposure_map=exposure_data,
        analysis_mask=analysis_mask_data,
        optaxis_x=cfg.optaxis_x,
        optaxis_y=cfg.optaxis_y,
        sigma_grid=cfg.background_sigma_grid,
        eef_radius_maps=eef_radius_maps,
    )
    valid_background = background_map[np.isfinite(background_map) & (background_map > 0.0)]
    if valid_background.size:
        emit(
            active_logger,
            "info",
            "Background map summary: "
            f"min={float(np.min(valid_background)):.4g}, "
            f"median={float(np.median(valid_background)):.4g}, "
            f"max={float(np.max(valid_background)):.4g}",
        )
    else:
        emit(active_logger, "warning", "Background map contains no positive valid pixels")

    #--- Reassess candidates detection and extendedness with PSF-aware apertures and local bkg estimates
    emit(active_logger, "info", "**** Stage 3: PSF-Aware Source Fitting and Classification ****")
    emit(active_logger, "info", "Fitting ...")
    rows = classify_sources_with_psf(
        rows=fit_rows,
        image=image_data,
        pixel_scale_arcsec=pixel_scale_arcsec,
        min_det_like=cfg.min_det_like,
        min_ext_like=cfg.min_ext_like,
        psf_context=psf_context,
        background_map=background_map,
        exposure_map=exposure_data,
        analysis_mask=analysis_mask_data,
        optaxis_x=cfg.optaxis_x,
        optaxis_y=cfg.optaxis_y,
        show_progress=cfg.show_progress,
        eef_radius_maps=eef_radius_maps,
    )
    type_counts: dict[str, int] = {}
    for row in rows:
        type_counts[row.source_type] = type_counts.get(row.source_type, 0) + 1
    emit(
        active_logger,
        "info",
        "Classification summary before filtering: "
        + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())),
    )
    emit(active_logger, "info", f"Source counts: after PSF-aware fitting = {len(rows)}")

    #--- drop insignificant or spurious sources
    if not cfg.include_background:
        emit(active_logger, "info", "Dropping insignificant detections as required  ...")
        before = len(rows)
        rows = [row for row in rows if row.source_type != "background"]
        emit(active_logger, "info", f"Dropped background-classified rows: {before} -> {len(rows)}")
        emit(active_logger, "info", f"Source counts: after background rejection = {len(rows)}")
        for idx, row in enumerate(rows, start=1):
            row.id = idx
            row.id_src = idx
    else:
        emit(active_logger, "info", "Keeping background-classified rows in output catalog")
        emit(active_logger, "info", f"Source counts: after background rejection = {len(rows)}")
    if cfg.emin_keV is not None:
        for row in rows:
            row.emin_keV = float(cfg.emin_keV)
    if cfg.emax_keV is not None:
        for row in rows:
            row.emax_keV = float(cfg.emax_keV)

    #--- add physical ra dec to sources
    emit(active_logger, "info", "Add physical ra dec to sources ...")
    rows = augment_rows_with_wcs(rows, wcs)

    #--- finalize catalog columns
    rows = finalize_catalog_columns(
        rows=rows,
        exposure=exposure_data,
        analysis_mask=analysis_mask_data,
        pixel_scale_arcsec=pixel_scale_arcsec,
        ecf=cfg.ecf,
    )
    
    #--- optionally prune sources
    if cfg.prune_sources:
        emit(active_logger, "info", "Pruning nearby duplicate sources as required ...")
        before = len(rows)
        rows = prune_nearby_sources(rows)
        emit(active_logger, "info", f"Pruned nearby duplicate sources: {before} -> {len(rows)}")
        emit(active_logger, "info", f"Source counts: after duplicate pruning = {len(rows)}")
    else:
        emit(active_logger, "info", "Skipping nearby-source pruning")
        emit(active_logger, "info", f"Source counts: after duplicate pruning = {len(rows)}")

    n_point = sum(row.source_type == "point" for row in rows)
    n_ext = sum(row.source_type == "extended" for row in rows)
    n_bkg = sum(row.source_type == "background" for row in rows)
    emit(
        active_logger,
        "info",
        f"Final catalog summary: total={len(rows)}, point={n_point}, extended={n_ext}, background={n_bkg}",
    )
    if n_ext == 0:
        emit(active_logger, "warning", "No extended sources were classified in the final catalog")
    if cfg.include_background and cfg.prune_sources is False:
        emit(active_logger, "warning", "Debug mode is active: background rows are included and nearby-source pruning is disabled")

    return {
        "rows": rows,                               # final clean catalog rows
        "per_scale": per_scale,                     # wavelet detection results at each scale
        "agg_mask": agg_mask,                       # a rough aggregated source mask at initial wavelet stage
        "best_sig": best_sig,                       # best smallest significance value seen at each pixel across all scales
        "background_map": background_map,           # source-carved and smoothed background map
        "analysis_mask": analysis_mask_data,        # user-supplied global analysis-validity mask
        "psf_context": psf_context,                 # mission-specific PSF context
        "pixel_scale_arcsec": pixel_scale_arcsec,   # arcsec/pixel
    }


def _parse_scales(raw: str) -> list[float]:
    """Parse a user-provided scale list."""
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise ValueError("At least one scale is required.")
    return vals


def _parse_background_sigma_grid(raw: str) -> list[float]:
    """Parse a user-provided background sigma grid."""
    vals = [float(x) for x in raw.replace(",", " ").split()]
    if not vals:
        raise ValueError("At least one background smoothing scale is required.")
    if any(val <= 0 for val in vals):
        raise ValueError("All background smoothing scales must be > 0.")
    return vals


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the detector."""
    p = argparse.ArgumentParser(description="wavdetect-style multi-scale X-ray source detection in pure Python")
    #--- basic parameters
    p.add_argument("image", type=Path, help="Input FITS image")
    p.add_argument("--expmap", type=Path, default=None, help="Exposure map FITS image")
    p.add_argument("--mask", type=Path, default=None, help="Optional user-supplied boolean analysis mask FITS image; non-zero pixels are treated as globally valid")
    #--- psf-related parameters
    p.add_argument("--eefmap", type=Path, default=None, help="Optional precomputed EEF-radius map product from ``fxteefmap`` to compute PSF-aware source apertures")
    p.add_argument("--mission", type=str, default="ep-fxt", help="Mission name to construct spatial-dependent PSF if ``eefmap`` not provided, defaults to ep-fxt")
    p.add_argument("--instrument", type=str, default=None, help="Mission instrument name to construct spatial-dependent PSF if ``eefmap`` not provided, e.g. fxta or fxtb for EP/FXT")
    p.add_argument("--filter", type=str, default=None, help="Mission filter name to construct spatial-dependent PSF if ``eefmap`` not provided, e.g. open, medium, thin, hole")
    p.add_argument("--emin", type=float, default=None, help="Lower image energy bound in keV to construct spatial-dependent PSF if ``eefmap`` not provided")
    p.add_argument("--emax", type=float, default=None, help="Upper image energy bound in keV to construct spatial-dependent PSF if ``eefmap`` not provided")
    p.add_argument("--optaxis-x", type=float, default=None, help="Optical-axis X position in 1-based image pixels; defaults to image center")
    p.add_argument("--optaxis-y", type=float, default=None, help="Optical-axis Y position in 1-based image pixels; defaults to image center")
    p.add_argument("--background-sigma-grid", type=str, default="4,8,16,32,64", help="Gaussian smoothing scales in pixels available to the adaptive background model, e.g. '4,8,16,32,64'")
    #--- wavelet detection parameters
    p.add_argument("--scales", type=str, default="1,2,4,8,16", help="Wavelet scales in pixels, e.g. '1,2,4,8,16'")
    p.add_argument("--sigthresh", type=float, default=1e-6, help="Detection significance threshold at wavelet stage")
    p.add_argument("--bkgsigthresh", type=float, default=1e-3, help="Background cleansing significance threshold at wavelet stage")
    p.add_argument("--maxiter", type=int, default=2, help="Max background-cleaning iterations at wavelet stage")
    p.add_argument("--iterstop", type=float, default=1e-4, help="Min fractional pixel change to continue at wavelet stage")
    p.add_argument("--expthresh", type=float, default=0.1, help="Minimum relative exposure to consider valid pixel")
    p.add_argument("--ellsigma", type=float, default=3.0, help="Ellipse scale factor to estimate source regions at wavelet stage (this region is not the final region to appear in output catalog)")
    #--- psf-aware source detection parameters & final catalog parameters
    p.add_argument("--min-det-like", type=float, default=6.0, help="Minimum ermldet-like detection likelihood for a source to be considered a non-background source")
    p.add_argument("--min-ext-like", type=float, default=6.0, help="Minimum extent likelihood threshold for a source to be classified as extended rather than point-like")
    p.add_argument("--ecf", type=float, default=0.0, help="Energy conversion factor to convert count rate to flux; 0 disables flux conversion")
    #--- output parameters
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level for CLI and output log file")
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file path; defaults to <out>.log")
    p.add_argument("--no-progress", action="store_true", help="Disable the catalog-fitting progress bar")
    p.add_argument("--out", type=Path, default=Path("sources.fits"), help="Output source table FITS")
    p.add_argument("--regfile", type=Path, default=Path("sources.reg"), help="Output DS9 region file")
    p.add_argument("--sky-regfile", type=Path, default=None, help="Optional DS9 sky region file (requires FITS WCS)")
    p.add_argument("--save-mask", type=Path, default=None, help="Optional aggregate source mask FITS output")
    p.add_argument("--save-significance", type=Path, default=None, help="Optional best-significance FITS map output")
    p.add_argument("--save-bkgmap", type=Path, default=None, help="Optional carved-and-smoothed background FITS map output")
    #--- debug parameters
    p.add_argument("--debug-columns", action="store_true", help="Include internal/debug columns in the output CSV")
    p.add_argument("--include-background", action="store_true", help="Keep rows classified as background in the output catalog, for debug only")
    p.add_argument("--no-prune-sources", action="store_true", help="Disable nearby-source duplicate pruning in the final catalog, for debug only")
    return p


def main() -> None:
    """Run the full command-line detection and catalog workflow."""
    args = build_parser().parse_args()
    image_header = load_header(args.image)
    log_file = args.log_file if args.log_file is not None else args.out.with_suffix(".log")
    cli_logger = build_cli_logger("eFXTDAS.fxtsrcdet", args.log_level, log_file)
    config = PipelineConfig(
        mission=args.mission,
        instrument=args.instrument,
        filter_name=args.filter,
        emin_keV=args.emin,
        emax_keV=args.emax,
        background_sigma_grid=tuple(_parse_background_sigma_grid(args.background_sigma_grid)),
        scales=tuple(_parse_scales(args.scales)),
        sigthresh=args.sigthresh,
        bkgsigthresh=args.bkgsigthresh,
        maxiter=args.maxiter,
        iterstop=args.iterstop,
        expthresh=args.expthresh,
        ellsigma=args.ellsigma,
        min_det_like=args.min_det_like,
        min_ext_like=args.min_ext_like,
        ecf=args.ecf,
        show_progress=not args.no_progress,
        optaxis_x=args.optaxis_x,
        optaxis_y=args.optaxis_y,
        eefmap=args.eefmap,
        analysis_mask=args.mask,
        include_background=args.include_background,
        prune_sources=not args.no_prune_sources,
        debug_columns=args.debug_columns,
        logger=cli_logger,
    )
    result = fxtsrcdet_pipeline(image=args.image, exposure=args.expmap, analysis_mask=args.mask, config=config)
    rows = result["rows"]
    write_sources_fits(args.out, rows, debug_columns=config.debug_columns)
    write_ds9_regions(args.regfile, rows)
    if args.sky_regfile:
        if not any(np.isfinite(row.ra) and np.isfinite(row.dec) for row in rows):
            raise RuntimeError("Sky-region output requires a FITS image with celestial WCS.")
        write_ds9_sky_regions(args.sky_regfile, rows)
    if args.save_mask:
        save_img(args.save_mask, result["agg_mask"].astype(np.uint8), header=image_header)
    if args.save_significance:
        save_img(args.save_significance, result["best_sig"], header=image_header)
    if args.save_bkgmap:
        save_img(args.save_bkgmap, result["background_map"], header=image_header)
    emit(cli_logger, "info", f"Detected {len(rows)} source(s).")
    emit(cli_logger, "info", f"Wrote source list: {args.out}")
    emit(cli_logger, "info", f"Wrote region file: {args.regfile}")
    if args.sky_regfile:
        emit(cli_logger, "info", f"Wrote sky region file: {args.sky_regfile}")


if __name__ == "__main__":
    main()


run_pipeline = fxtsrcdet_pipeline
