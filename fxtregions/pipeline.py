#!/usr/bin/env python3
"""Build source and background DS9 regions for FXT spectral extraction."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord

from fxtpsf_helpers import build_mission_psf_context, infer_optical_axis
from fxtregions.auto import (
    auto_background_annulus_pix,
    auto_source_radius_pix,
    confusing_source_radius_pix,
)
from fxtregions.catalog import find_column, match_target, nearest_catalog_row
from fxtregions.config import (
    DEFAULT_BKG_INNER_RADIUS_DEG,
    DEFAULT_BKG_OUTER_RADIUS_DEG,
    DEFAULT_SOURCE_RADIUS_DEG,
    FXT_POSITION_ERR90_ARCSEC,
    MAX_CONF_TO_BACK_RATIO,
    MIN_EXCLUDE_DIST_ARCSEC,
)
from fxtregions.utils.ds9 import ds9_annulus, ds9_circle, write_region_file
from fxtregions.utils.io import load_catalog, read_ccd
from fxtregions.utils.logger import build_cli_logger, emit
from fxtregions.measure import estimate_net_counts, kernel_surface_profile, ml_bkg_to_bkg_per_pix, sample_bkg_from_map
from fxtregions.models import TargetContext, source_kernel
from fxtregions.utils.wcs import infer_pixel_scale_arcsec


def build_regions(
    image_path: Path,
    catalog_path: Path,
    ra_deg: float,
    dec_deg: float,
    bkgmap_path: Path | None = None,
    mission: str = "ep-fxt",
    instrument: str | None = None,
    filter_name: str | None = None,
    emin_keV: float | None = None,
    emax_keV: float | None = None,
    mode: str = "auto",
    src_radius_arcsec: float | None = None,
    bkg_inner_arcsec: float | None = None,
    bkg_outer_arcsec: float | None = None,
    match_threshold_arcsec: float = FXT_POSITION_ERR90_ARCSEC,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    src_regfile: Path | None = None,
    bkg_regfile: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Build source and background extraction regions.

    Parameters
    ----------
    image_path : Path
        Counts image FITS path.
    catalog_path : Path
        Source catalog path.
    ra_deg : float
        Requested target right ascension in degrees.
    dec_deg : float
        Requested target declination in degrees.
    bkgmap_path : Path | None
        Optional background map FITS path. When omitted, auto mode uses the
        scalar ``ML_BKG_0`` value from the matched catalog row.
    mission : str
        Mission identifier.
    instrument : str | None
        Instrument override.
    filter_name : str | None
        Filter override.
    emin_keV : float | None
        Lower energy bound override.
    emax_keV : float | None
        Upper energy bound override.
    mode : str
        Region-sizing mode: ``auto`` or ``manual``.
    src_radius_arcsec : float | None
        Manual source radius in arcseconds.
    bkg_inner_arcsec : float | None
        Manual background annulus inner radius in arcseconds.
    bkg_outer_arcsec : float | None
        Manual background annulus outer radius in arcseconds.
    match_threshold_arcsec : float
        Maximum catalog-match separation in arcseconds.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based image pixels.
    src_regfile : Path | None
        Optional output path for the source region file. When supplied, the
        region file is written inside this function.
    bkg_regfile : Path | None
        Optional output path for the background region file. When supplied, the
        region file is written inside this function.
    logger : logging.Logger | None
        Optional logger used for progress and warning messages. When omitted,
        messages are printed directly.

    Returns
    -------
    region_info : dict[str, Any]
        Dictionary describing the matched source, final radii, and exclusion regions.
    """
    emit(logger, "info", "=================================")
    emit(logger, "info", "**** Welcome to FXTREGIONS! ****")
    emit(logger, "info", "=================================")
    emit(logger, "info", "**** Input Checklist ****")
    emit(logger, "info", f"Image file: {image_path}")
    emit(logger, "info", f"Catalog file: {catalog_path}")
    emit(logger, "info", f"Background map: {bkgmap_path}")
    emit(logger, "info", f"Target coordinate: ICRS({ra_deg}, {dec_deg})")
    emit(logger, "info", f"Mission / instrument / filter: {mission}, {instrument}, {filter_name}")
    emit(logger, "info", f"Energy band: {emin_keV}, {emax_keV} keV")
    emit(logger, "info", f"Requested mode: {mode}")
    emit(logger, "info", f"Match threshold: {match_threshold_arcsec} arcsec")
    emit(logger, "info", f"Output source region: {src_regfile}")
    emit(logger, "info", f"Output background region: {bkg_regfile}")
    # emit(logger, "info", "**** Region Building ****")

    #--- read image and wcs
    emit(logger, "info", f"Loading image: {image_path}")
    image_ccd = read_ccd(image_path)
    bkg_ccd = None if bkgmap_path is None else read_ccd(bkgmap_path)
    emit(logger, "info", f"Loading catalog: {catalog_path}")
    if bkg_ccd is not None and image_ccd.data.shape != bkg_ccd.data.shape:
        raise ValueError("Image and background map shapes do not match.")
    wcs = image_ccd.wcs
    if wcs is None or not getattr(wcs, "has_celestial", False):
        raise ValueError("Input image must contain celestial WCS.")
    wcs = wcs.celestial
    pixel_scale = infer_pixel_scale_arcsec(wcs)
    catalog = load_catalog(catalog_path)

    #--- match target to catalog
    emit(logger, "info", "Matching target to catalog ...")
    target = SkyCoord(ra_deg, dec_deg, unit="deg")
    match_idx, adopted_coord, match_sep = match_target(catalog, target, match_threshold_arcsec)
    matched_row = None if match_idx is None else catalog[match_idx]
    matched = matched_row is not None
    effective_mode = mode if matched else "manual"
    if matched:
        emit(
            logger,
            "info",
            f"Matched target to catalog row {match_idx} at separation {match_sep:.3f} arcsec; using catalog coordinates.",
        )
    else:
        emit(
            logger,
            "warning",
            (
                f"Target did not match any catalog source within {match_threshold_arcsec:.1f} arcsec; "
                "keeping the input coordinates."
            ),
        )
        if mode == "auto":
            emit(
                logger,
                "warning",
                "Auto mode requested for an undetected target; falling back to manual mode with configured default radii.",
            )

    #--- read psf
    psf_context = build_mission_psf_context(
        mission=mission,
        instrument=instrument,
        filter_name=filter_name,
        emin_keV=emin_keV,
        emax_keV=emax_keV,
    )
    x_cen, y_cen = wcs.world_to_pixel(adopted_coord)
    x_cen = float(np.asarray(x_cen))
    y_cen = float(np.asarray(y_cen))
    opt_x, opt_y = infer_optical_axis(image_ccd.data.shape, optaxis_x, optaxis_y)
    theta_arcmin = math.hypot((x_cen + 1.0) - opt_x, (y_cen + 1.0) - opt_y) * pixel_scale / 60.0

    #--- resolve required catalog schema explicitly
    ra_col = find_column(catalog, ["RA"])
    dec_col = find_column(catalog, ["DEC"])
    source_type_col = find_column(catalog, ["SOURCE_TYPE"])
    ml_cts_col = find_column(catalog, ["ML_CTS_0"])
    ml_bkg_col = find_column(catalog, ["ML_BKG_0"])
    ext_col = find_column(catalog, ["EXT"])

    #--- determine source type, construct source model (psf for point source, or gaussian-convolved psf for extended)
    emit(logger, "info", "Determining source type and local PSF ...")
    source_type = "point" if not matched else str(matched_row[source_type_col]).lower()
    fitted_extent_sigma_pix = 0.0 if not matched else float(matched_row[ext_col]) / pixel_scale
    kernel, rr, cum = source_kernel(psf_context, theta_arcmin, source_type, fitted_extent_sigma_pix)
    psf_r90_pix = np.interp(0.90, cum, rr)
    psf_r99_pix = np.interp(0.99, cum, rr)

    #--- determine local background
    emit(logger, "info", "Fetching local background ...")
    bkg_image = None if bkg_ccd is None else np.asarray(bkg_ccd.data, dtype=np.float64)
    ##--- if user supplies bkg map, use it to estimate local bkg
    if bkg_image is not None:
        bkg_per_pix = sample_bkg_from_map(bkg_image, x_cen, y_cen)
        background_mode = "map"
    ##--- if no bkg map supplied, but target source detected in the catalog, use ML_BKG_0
    elif matched:
        bkg_per_pix = ml_bkg_to_bkg_per_pix(float(matched_row[ml_bkg_col]), pixel_scale)
        background_mode = "catalog"
    ##--- if no bkg map, and target source undetected, use ML_BKG_0 from nearest source
    else:
        nearest_row = nearest_catalog_row(catalog, target)
        bkg_per_pix = ml_bkg_to_bkg_per_pix(float(nearest_row[ml_bkg_col]), pixel_scale)
        background_mode = "nearest-catalog"
        emit(
            logger,
            "warning",
            "Target is undetected and no background map was supplied; using ML_BKG_0 from the nearest detected catalog source.",
        )

    #--- determine target brightness (used only when AUTO mode activated + target source detected)
    if matched: # detected
        net_counts = float(matched_row[ml_cts_col])
    else:
        net_counts = None

    #--- aperture photometry: not used in region creature, but for later inspection
    aperture_fraction = float(np.interp(psf_r90_pix, rr, cum, left=0.0, right=1.0))
    forced_net_counts = estimate_net_counts(
        np.asarray(image_ccd.data, dtype=np.float64),
        bkg_per_pix,
        x_cen,
        y_cen,
        psf_r90_pix,
        aperture_fraction=aperture_fraction,
    )   # aperture loss already corrected

    target_ctx = TargetContext(
        requested_coord=target,
        adopted_coord=adopted_coord,
        matched=matched,
        matched_index=match_idx,
        match_separation_arcsec=match_sep,
        requested_mode=mode,
        effective_mode=effective_mode,
        source_type=source_type,
        fitted_extent_sigma_pix=fitted_extent_sigma_pix,
        background_mode=background_mode,
        bkg_per_pix=float(bkg_per_pix),
        net_counts=None if net_counts is None else float(net_counts),
        forced_net_counts=float(forced_net_counts),
    )

    #--- determine source extraction radius, and background annulus inner and outer radii
    emit(logger, "info", "**** Stage 1: determine TARGET SOURCE, and TARGET BACKGROUND regions ****")
    if target_ctx.effective_mode == "manual":  # MANUAL mode (user supplies MANUAL mode; or AUTO mode but target undetected)
        src_radius_arcsec_eff = (
            float(src_radius_arcsec)
            if src_radius_arcsec is not None
            else DEFAULT_SOURCE_RADIUS_DEG * 3600.0
        )
        bkg_inner_arcsec_eff = (
            float(bkg_inner_arcsec)
            if bkg_inner_arcsec is not None
            else DEFAULT_BKG_INNER_RADIUS_DEG * 3600.0
        )
        bkg_outer_arcsec_eff = (
            float(bkg_outer_arcsec)
            if bkg_outer_arcsec is not None
            else DEFAULT_BKG_OUTER_RADIUS_DEG * 3600.0
        )
        src_radius_pix = src_radius_arcsec_eff / pixel_scale
        bkg_inner_pix = bkg_inner_arcsec_eff / pixel_scale
        bkg_outer_pix = bkg_outer_arcsec_eff / pixel_scale
        emit(
            logger,
            "info",
            (
                f"Using manual radii: src={src_radius_arcsec_eff:.3f}\" "
                f"bkg=({bkg_inner_arcsec_eff:.3f}\", {bkg_outer_arcsec_eff:.3f}\")"
            ),
        )
    else:   # AUTO mode
        src_radius_pix = auto_source_radius_pix(net_counts, bkg_per_pix, rr, cum, psf_r99_pix, pixel_scale)
        prof_r, prof_sb = kernel_surface_profile(kernel)
        bkg_inner_pix, bkg_outer_pix = auto_background_annulus_pix(
            net_counts,
            bkg_per_pix,
            prof_r,
            prof_sb,
            src_radius_pix,
            psf_r99_pix,
            pixel_scale,
        )
        emit(
            logger,
            "info",
            (
                f"Auto radii from source/background model: src={src_radius_pix * pixel_scale:.3f}\" "
                f"bkg=({bkg_inner_pix * pixel_scale:.3f}\", {bkg_outer_pix * pixel_scale:.3f}\")"
            ),
        )

    if bkg_inner_pix <= src_radius_pix:
        raise ValueError(
            "Background annulus inner radius must be larger than the source extraction radius."
        )

    #--- determine confusing source exclusion size in source extraction circle and background annulus
    emit(logger, "info", "**** Stage 2: determine CONFUSING SOURCE exclusion size in TARGET SOURCE and TARGET BACKGROUND regions ****")
    source_excludes: list[str] = []
    background_excludes: list[str] = []
    warned_undetected_src_exclusion = False
    warned_blended_neighbor = False
    target_prof_r, target_prof_sb = kernel_surface_profile(kernel)  # psf profile for point source; extended profile for extended sources
    for idx, row in enumerate(catalog):
        if match_idx is not None and idx == match_idx:
            continue
        ##--- read neighbor source properties and profile
        neighbor_coord = SkyCoord(float(row[ra_col]), float(row[dec_col]), unit="deg")
        nx, ny = wcs.world_to_pixel(neighbor_coord)
        nx = float(np.asarray(nx))
        ny = float(np.asarray(ny))
        neighbor_theta = math.hypot((nx + 1.0) - opt_x, (ny + 1.0) - opt_y) * pixel_scale / 60.0
        nkernel, nrr, ncum = source_kernel(
            psf_context,
            neighbor_theta,
            str(row[source_type_col]).lower(),
            float(row[ext_col]) / pixel_scale,
        )   # cumulative surface brightness profile
        neighbor_prof_r, neighbor_prof_sb = kernel_surface_profile(nkernel)   # surface brightness profile (1/pixel)
        default_psf_r99 = float(np.interp(0.99, ncum, nrr)) # psf R90 for point source; extended profile R90 for extended sources
        conf_counts = max(float(row[ml_cts_col]), 0.0)
        
        ##--- rare case, but not impossible: 
        # if there are multiple detections around target position, blending / fragmentation occurs
        # based on current instrument psf, we cannot infer whether it is a nearby source, or simply fragment of target source
        # we do not mark them as confusing source; they are absorbed into the final spectrum
        sep_pix = math.hypot(nx - x_cen, ny - y_cen)
        if sep_pix <= MIN_EXCLUDE_DIST_ARCSEC / pixel_scale and not warned_blended_neighbor:
            emit(
                logger,
                "warning",
                (
                    f"A nearby catalog source lies within {MIN_EXCLUDE_DIST_ARCSEC:.1f} arcsec of the target. "
                    "It is treated as a fragment of target source, so no separate contaminant exclusion radius is derived for that neighbour."
                ),
            )
            warned_blended_neighbor = True
        
        ##--- for background region file: we carve out this source with radius determined by local bkg
        back_exc_pix = confusing_source_radius_pix(
            sep_pix,        # separation to target source in pixel
            conf_counts,    # confusing source counts
            neighbor_prof_r,     # confusing source surface brightness profile radius array
            neighbor_prof_sb,    # confusing source surface brightness profile array
            threshold_sb=MAX_CONF_TO_BACK_RATIO * max(bkg_per_pix, 1e-12),  # from which exclusion radius is determined
            pixel_scale_arcsec=pixel_scale,
            conf_r99_pix=default_psf_r99,   # maximum exclusion radius to clip on
        )
        
        ##--- for source region file: we carve out this source with radius determined by brightness of target and this source
        ###--- if target source detected in the catalog, we have its brightness profile
        if matched:
            src_exc_pix = confusing_source_radius_pix(
                sep_pix,
                conf_counts,
                neighbor_prof_r,
                neighbor_prof_sb,
                threshold_sb=0.0,
                pixel_scale_arcsec=pixel_scale,
                conf_r99_pix=default_psf_r99,
                target_counts=float(target_ctx.net_counts),           # target source net counts
                target_profile_r=target_prof_r,     # target source surface brightness profile radius array
                target_profile_sb=target_prof_sb,   # target source surface brightness profile array
            )
        ###--- otherwise if undetected, we cannot know its brightness profile; instead we use the exclusion radius from bkg region
        else:
            src_exc_pix = back_exc_pix
            if src_exc_pix > 0.0 and sep_pix <= (src_radius_pix + src_exc_pix) and not warned_undetected_src_exclusion:
                emit(
                    logger,
                    "warning",
                    (
                        "Target is undetected; using background-style contaminant exclusion "
                        "inside the source region because ML_CTS_0 is unavailable for the target."
                    ),
                )
                warned_undetected_src_exclusion = True
        
        ##--- only append bkg exclusion circles inside bkg annulus
        if back_exc_pix > 0.0 and sep_pix <= (bkg_outer_pix + back_exc_pix):
            background_excludes.append(ds9_circle(neighbor_coord, back_exc_pix * pixel_scale))
        ##--- only append src exclusion circles inside source region
        if src_exc_pix > 0.0 and sep_pix <= (src_radius_pix + src_exc_pix):
            source_excludes.append(ds9_circle(neighbor_coord, src_exc_pix * pixel_scale))
        ##--- raise error if the exclusion fully covers the source region
        if src_exc_pix > 0.0 and sep_pix <= (src_exc_pix - src_radius_pix):
            raise ValueError("Contaminant exclusion fully covers the target source region; effective source region size is zero.")

    src_region = ds9_circle(adopted_coord, src_radius_pix * pixel_scale)
    bkg_region = ds9_annulus(adopted_coord, bkg_inner_pix * pixel_scale, bkg_outer_pix * pixel_scale)

    region_info = {
        "matched_index": match_idx,
        "match_separation_arcsec": target_ctx.match_separation_arcsec,
        "adopted_coord": target_ctx.adopted_coord,
        "source_radius_arcsec": float(src_radius_pix * pixel_scale),
        "background_inner_arcsec": float(bkg_inner_pix * pixel_scale),
        "background_outer_arcsec": float(bkg_outer_pix * pixel_scale),
        "source_region": src_region,
        "background_region": bkg_region,
        "source_excludes": source_excludes,
        "background_excludes": background_excludes,
        "local_background_counts_per_pixel": target_ctx.bkg_per_pix,
        "local_source_counts": target_ctx.net_counts,
        "forced_net_counts": target_ctx.forced_net_counts,
        "background_mode": target_ctx.background_mode,
        "requested_mode": target_ctx.requested_mode,
        "effective_mode": target_ctx.effective_mode,
        "matched": target_ctx.matched,
        "target_context": target_ctx,
    }
    if src_regfile is not None:
        write_region_file(src_regfile, region_info["source_region"], [])
    if bkg_regfile is not None:
        write_region_file(bkg_regfile, region_info["background_region"], region_info["background_excludes"])

    emit(logger, "info", f"Matched source index: {region_info['matched_index']}")
    emit(logger, "info", f"Match separation: {region_info['match_separation_arcsec']:.3f} arcsec")
    emit(logger, "info", f"Mode: requested={region_info['requested_mode']} effective={region_info['effective_mode']}")
    emit(logger, "info", f"Background mode: {region_info['background_mode']}")
    if region_info["local_source_counts"] is None:
        emit(logger, "info", "Local source counts: unavailable for undetected target")
    else:
        emit(logger, "info", f"Local source counts: {region_info['local_source_counts']:.3f}")
    emit(logger, "info", f"Forced net counts: {region_info['forced_net_counts']:.3f}")
    emit(logger, "info", f"Local background: {region_info['local_background_counts_per_pixel']:.6f} count/pixel")
    emit(logger, "info", f"Source radius: {region_info['source_radius_arcsec']:.3f} arcsec")
    emit(
        logger,
        "info",
        (
            "Background annulus: "
            f"{region_info['background_inner_arcsec']:.3f} - {region_info['background_outer_arcsec']:.3f} arcsec"
        ),
    )
    if src_regfile is not None:
        emit(logger, "info", f"Wrote source region: {src_regfile}")
    if bkg_regfile is not None:
        emit(logger, "info", f"Wrote background region: {bkg_regfile}")

    return region_info


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Parameters
    ----------
    None

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured command-line parser.
    """
    p = argparse.ArgumentParser(description="Create source and background DS9 regions for FXT spectral extraction")
    p.add_argument("image", type=Path, help="Input counts image FITS file")
    p.add_argument("catalog", type=Path, help="Input source catalog (CSV or FITS)")
    p.add_argument("--bkgmap", type=Path, default=None, help="Optional background map FITS file")
    p.add_argument("--ra", type=float, required=True, help="Target right ascension in degrees")
    p.add_argument("--dec", type=float, required=True, help="Target declination in degrees")
    p.add_argument("--mission", type=str, default="ep-fxt", help="Mission identifier")
    p.add_argument("--instrument", type=str, default=None, help="Mission instrument or detector arm")
    p.add_argument("--filter", type=str, default=None, help="Mission filter state")
    p.add_argument("--emin", type=float, default=None, help="Lower energy bound in keV")
    p.add_argument("--emax", type=float, default=None, help="Upper energy bound in keV")
    p.add_argument("--mode", choices=["auto", "manual"], default="auto", help="Region sizing mode")
    p.add_argument("--src-radius", type=float, default=None, help="Manual source radius in arcsec")
    p.add_argument("--bkg-inner", type=float, default=None, help="Manual background annulus inner radius in arcsec")
    p.add_argument("--bkg-outer", type=float, default=None, help="Manual background annulus outer radius in arcsec")
    p.add_argument("--match-threshold", type=float, default=FXT_POSITION_ERR90_ARCSEC, help="Catalog matching threshold in arcsec")
    p.add_argument("--optaxis-x", type=float, default=None, help="Optical-axis X position in 1-based image pixels")
    p.add_argument("--optaxis-y", type=float, default=None, help="Optical-axis Y position in 1-based image pixels")
    p.add_argument("--src-regfile", type=Path, default=Path("source.reg"), help="Output source DS9 region file")
    p.add_argument("--bkg-regfile", type=Path, default=Path("background.reg"), help="Output background DS9 region file")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level for CLI and output log file")
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file path; defaults to fxtregions.log beside the output regions")
    return p


def main() -> None:
    """Run the region builder CLI.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = build_parser().parse_args()
    log_file = args.log_file if args.log_file is not None else args.src_regfile.with_name("fxtregions.log")
    logger = build_cli_logger("eFXTDAS.fxtregions", args.log_level, log_file)
    info = build_regions(
        image_path=args.image,
        catalog_path=args.catalog,
        ra_deg=args.ra,
        dec_deg=args.dec,
        bkgmap_path=args.bkgmap,
        mission=args.mission,
        instrument=args.instrument,
        filter_name=args.filter,
        emin_keV=args.emin,
        emax_keV=args.emax,
        mode=args.mode,
        src_radius_arcsec=args.src_radius,
        bkg_inner_arcsec=args.bkg_inner,
        bkg_outer_arcsec=args.bkg_outer,
        match_threshold_arcsec=args.match_threshold,
        optaxis_x=args.optaxis_x,
        optaxis_y=args.optaxis_y,
        src_regfile=args.src_regfile,
        bkg_regfile=args.bkg_regfile,
        logger=logger,
    )
    _ = info


if __name__ == "__main__":
    main()
