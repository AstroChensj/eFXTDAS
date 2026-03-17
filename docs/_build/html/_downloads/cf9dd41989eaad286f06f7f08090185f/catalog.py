from __future__ import annotations

import math

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy import signal
from tqdm.auto import tqdm

from fxtsrcdet.config import (
    CONTAM_MIN_RADIUS_PIX,
    CONTAM_R90_FACTOR,
    CONTAM_SCALE_FACTOR,
    DEFAULT_EXTENT_SIGMA_PIX,
    EDGE_BACKGROUND_DET_LIKE_FACTOR,
    EDGE_BACKGROUND_DET_LIKE_FLOOR,
    EDGE_BACKGROUND_MAX_MASKFRAC,
    EDGE_BACKGROUND_MAX_NPIX,
    EDGE_BACKGROUND_MAX_SUPPORT_SCALES,
    EXTENT_CLASSIFY_MAX_R90_FACTOR,
    EXTENT_CLASSIFY_MAX_SIGMA_PIX,
    EXTENT_CLASSIFY_MIN_MASKFRAC,
    EXTENT_CLASSIFY_MIN_SIGMA_PIX,
    EXTENT_CLASSIFY_MIN_STRETCH,
    EXTENT_FIT_MIN_NPIX,
    EXTENT_FIT_MIN_SUPPORT_SCALES,
    EXTENT_LIKE_DOF,
    EXT_RC_GRID_BASE_PIX,
    EXT_RC_GRID_R50_FACTORS,
    EXT_RC_GRID_R90_FACTOR,
    EXT_RC_MAX_MIN_PIX,
    EXT_RC_MAX_R90_FACTOR,
    EXT_RC_MIN_PIX,
    FIT_STAMP_MIN_RADIUS_PIX,
    FIT_STAMP_R90_FACTOR,
    FIT_STAMP_SCALE_FACTOR,
    GROUP_STAMP_MARGIN_MIN_PIX,
    GROUP_STAMP_MARGIN_R90_FACTOR,
    GROUP_STAMP_MARGIN_SCALE_FACTOR,
    GROUP_LINK_MIN_PSF_R90_PIX,
    GROUP_LINK_MIN_RADIUS_PIX,
    GROUP_LINK_R90_FACTOR,
    GROUP_STAMP_MIN_RADIUS_PIX,
    LOCAL_POINT_MAX_SHIFT_PIX,
    MORPH_EXTENT_DET_LIKE_FACTOR,
    MORPH_EXTENT_DET_LIKE_FLOOR,
    MORPH_EXTENT_MAX_R50_FACTOR,
    MORPH_EXTENT_MIN_MASKFRAC,
    MORPH_EXTENT_MIN_NPIX,
    MORPH_EXTENT_MIN_R80_FACTOR,
    MORPH_EXTENT_MIN_STRETCH,
    MORPH_EXTENT_MIN_SUPPORT_SCALES,
    POINT_SOURCE_DET_LIKE_DOF,
    PRUNE_MIN_PSF_R90_PIX,
    PRUNE_SUPPRESS_MIN_RADIUS_PIX,
    PRUNE_SUPPRESS_R90_FACTOR,
    PROFILE_MIN_RADIUS_PIX,
    PROFILE_R90_FACTOR,
    PROFILE_SCALE_FACTOR,
    SINGLE_SCALE_DET_LIKE_FACTOR,
    SINGLE_SCALE_DET_LIKE_FLOOR,
)
from fxtpsf_helpers import MissionPSFContext, build_psf_kernel, eef_radius, infer_optical_axis, load_local_eef, sample_radius_map
from fxtsrcdet.detect import EPS
from fxtsrcdet.background import create_background_map
from fxtsrcdet.fit import (
    cash_delta_to_like,
    cash_stat,
    fit_amplitude_cash,
    fit_extended_position_cash,
    fit_group_amplitudes_cash,
    fit_point_position_cash,
)
from fxtsrcdet.utils.imageops import embed_kernel, extract_stamp
from fxtsrcdet.utils.measure import (
    exposure_at_position,
    fraction_at_radius,
    mask_fraction,
    radius_at_fraction,
    template_radius_at_fraction,
)
from fxtsrcdet.models import CatalogRow, DetectionCandidate, FitMeasurement, beta_model_kernel


def _detection_rank(row: DetectionCandidate) -> tuple[float, float, float]:
    """Rank candidates by wavelet strength and counts for tie-breaking."""
    return (
        float(row.wavelet_peak_score),
        float(row.net_counts),
        float(row.counts),
    )


def classify_sources_with_psf(
    rows: list[DetectionCandidate],
    image: np.ndarray,
    pixel_scale_arcsec: float,
    min_det_like: float,
    min_ext_like: float,
    psf_context: MissionPSFContext,
    background_map: np.ndarray | None = None,
    exposure_map: np.ndarray | None = None,
    analysis_mask: np.ndarray | None = None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    show_progress: bool = False,
    eef_radius_maps: dict | None = None,
) -> list[CatalogRow]:
    """Measure PSF-aware source properties and assign final source classes.

    Parameters
    ----------
    rows : list[DetectionCandidate]
        Detection-stage candidates returned by :func:`fxtsrcdet.detect.combine_scales`.
        Each candidate provides the provisional source position, wavelet scale support,
        and detection-shape diagnostics used to seed the catalog fit.
    image : np.ndarray
        Counts image in counts per pixel.
    pixel_scale_arcsec : float
        Image pixel scale in arcsec per pixel.
    min_det_like : float
        Minimum detection likelihood for a source to remain in the science catalog.
        Candidates below this threshold are classified as background.
    min_ext_like : float
        Minimum extent likelihood required for a source to be classified as extended.
    psf_context : MissionPSFContext
        Mission-specific PSF/EEF calibration context.
    background_map : np.ndarray | None, optional
        Precomputed background map in counts per pixel. If omitted, a background map is
        constructed from the detection candidates before fitting.
    exposure_map : np.ndarray | None, optional
        Exposure map aligned with ``image``. Pixels with no valid exposure are excluded
        from local likelihood fits and radial-profile measurements.
    analysis_mask : np.ndarray | None, optional
        Optional boolean mask selecting globally valid pixels for background
        estimation and local fits.
    optaxis_x : float | None, optional
        Optical-axis x coordinate in 1-based image pixels. If omitted, it is inferred
        from the image geometry.
    optaxis_y : float | None, optional
        Optical-axis y coordinate in 1-based image pixels. If omitted, it is inferred
        from the image geometry.
    show_progress : bool, optional
        If ``True``, display progress bars for the single-source and grouped-fit stages.
    eef_radius_maps : dict | None, optional
        Optional dictionary of precomputed EEF-radius maps, typically sampled from a
        multi-extension ``fxteefmap`` product. When provided, local ``r50/r75/r80/r90``
        values are read from these maps instead of being inferred only from the CALDB
        EEF lookup.

    Returns
    -------
    list[CatalogRow]
        PSF-aware catalog rows. Each row contains the fitted image and sky position,
        detection and extent likelihoods, PSF radii, source type, and the eSASS-like
        catalog columns used by later reporting and region-generation steps.

    Notes
    -----
    The algorithm proceeds in stages. First, each detection candidate is measured with
    an isolated local point-source fit to obtain a provisional position shift, a local
    detection likelihood, and empirical radial sizes such as ``meas_r50_pix``. An
    isolated extended beta-model fit is also attempted so that obviously broad sources
    have a first-pass extent measurement.

    The candidates are then grouped with :func:`build_source_groups` when their PSF-sized
    neighborhoods overlap. Within each group, all point-source amplitudes are fit
    jointly so that blended sources are deblended before their final detection
    likelihoods are assigned. The per-source grouped ``DET_LIKE`` is conditional: the
    code compares the full grouped model against a null model in which the target member
    has been removed while the background and other group members are kept fixed.

    For grouped sources, the code also performs a conditional grouped extent test. The
    grouped point-source model is held fixed for all other members, and the target
    source is replaced with an extended beta-model template. The resulting likelihood
    improvement defines the grouped ``EXT_LIKE``. Empirical radii are recomputed from
    the residual profile ``data - background - other_group_members`` so that morphology
    diagnostics are tied to the deblended source rather than to the blended image stamp.

    Final source classes are assigned as ``background``, ``point``, or ``extended`` by
    combining the model-based likelihoods with morphology and mask-coverage checks.
    ``DET_LIKE`` alone is not sufficient for an ``extended`` label: the measured source
    size must also be broader than the local PSF and must be supported by adequate valid
    exposure coverage.
    """
    if background_map is None:
        background_map = create_background_map(
            image,
            rows,
            psf_context,
            pixel_scale_arcsec,
            exposure_map=exposure_map,
            analysis_mask=analysis_mask,
            optaxis_x=optaxis_x,
            optaxis_y=optaxis_y,
            eef_radius_maps=eef_radius_maps,
        )
    
    #--- calculate PSF R90 at each source location
    opt_x, opt_y = infer_optical_axis(image.shape, optaxis_x, optaxis_y)
    prelim_psf_r90: list[float] = []
    for row in rows:
        x_ima = float(row.x)
        y_ima = float(row.y)
        local_r90 = sample_radius_map(eef_radius_maps, "R90", x_ima, y_ima)
        if local_r90 is None:
            theta_arcmin = math.hypot(x_ima - opt_x, y_ima - opt_y) * pixel_scale_arcsec / 60.0
            radius_pix, frac = load_local_eef(psf_context, theta_arcmin)
            local_r90 = float(eef_radius(radius_pix, frac, 0.90))
        prelim_psf_r90.append(float(local_r90))

    #--- group nearby candidates for joint fitting
    groups = build_source_groups(rows, prelim_psf_r90)
    row_to_group: dict[int, int] = {}   # mapping from row index to group id
    for group_id, members in enumerate(groups):
        for idx in members:
            row_to_group[idx] = group_id
    
    #--- STAGE 1: fit single source in a local stamp
    # this gives a cheap, source-centered initialization for later joint fitting
    measurements: list[FitMeasurement] = []
    row_iter = tqdm(list(enumerate(rows)), desc="Single source fitting", unit="src", disable=not show_progress)
    for row_index, row in row_iter:
        ##--- calculate local PSF R50, R75, R80, R90
        x0 = float(row.x - 1.0)
        y0 = float(row.y - 1.0)
        theta_arcmin = math.hypot((x0 + 1.0) - opt_x, (y0 + 1.0) - opt_y) * pixel_scale_arcsec / 60.0
        radius_pix, frac = load_local_eef(psf_context, theta_arcmin)
        psf_r50_pix_nom = sample_radius_map(eef_radius_maps, "R50", row.x, row.y)
        psf_r75_pix_nom = sample_radius_map(eef_radius_maps, "R75", row.x, row.y)
        psf_r80_pix_nom = sample_radius_map(eef_radius_maps, "R80", row.x, row.y)
        psf_r90_pix_nom = sample_radius_map(eef_radius_maps, "R90", row.x, row.y)
        if psf_r50_pix_nom is None:
            psf_r50_pix_nom = eef_radius(radius_pix, frac, 0.50)
        if psf_r75_pix_nom is None:
            psf_r75_pix_nom = eef_radius(radius_pix, frac, 0.75)
        if psf_r80_pix_nom is None:
            psf_r80_pix_nom = eef_radius(radius_pix, frac, 0.80)
        if psf_r90_pix_nom is None:
            psf_r90_pix_nom = eef_radius(radius_pix, frac, 0.90)

        ##--- use a moderate fit stamp for the likelihood fit, and a separate
        ## tighter profile radius for radial size measurements
        fit_stamp_radius = max(
            FIT_STAMP_R90_FACTOR * psf_r90_pix_nom,
            FIT_STAMP_SCALE_FACTOR * float(row.scale),
            FIT_STAMP_MIN_RADIUS_PIX,
        )
        profile_radius = max(
            PROFILE_R90_FACTOR * psf_r90_pix_nom,
            PROFILE_SCALE_FACTOR * float(row.scale),
            PROFILE_MIN_RADIUS_PIX,
        )
        data_stamp, y_min, x_min = extract_stamp(image, x0, y0, fit_stamp_radius)
        bkg_stamp = background_map[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]]
        if exposure_map is not None:
            exp_stamp = exposure_map[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]]
            valid_mask = exp_stamp > 0.0
        else:
            exp_stamp = None
            valid_mask = np.ones_like(data_stamp, dtype=bool)
        if analysis_mask is not None:
            mask_stamp = np.asarray(analysis_mask[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]], dtype=bool)
            valid_mask &= mask_stamp
        if not np.any(valid_mask):
            if analysis_mask is not None and np.any(mask_stamp):
                valid_mask = mask_stamp.copy()
            else:
                valid_mask = np.ones_like(data_stamp, dtype=bool)

        group_id = row_to_group.get(row_index, -1)
        group_members = groups[group_id] if group_id >= 0 else [row_index]

        ##--- before starting fitting: we need to worry about strong neighbors 
        # inside the stamp but not a group member; the polluted pixels from these 
        # sources in the stamp should be masked
        neighbor_mask = np.zeros_like(valid_mask, dtype=bool)
        row_rank = _detection_rank(row)
        has_strong_neighbor = False
        for other_index, other in enumerate(rows):
            if other is row:    # no worries if it's the same source,
                continue
            if other_index in group_members:    # no worries if it's a group member, since we'll fit them together later
                continue
            other_rank = _detection_rank(other)
            if other_rank <= row_rank:  # no worries if the neighbor is weaker, since it won't pollute the fit as much
                continue
            ox = float(other.x - 1.0)
            oy = float(other.y - 1.0)
            if math.hypot(ox - x0, oy - y0) > fit_stamp_radius: # no worries if it's outside the stamp, since it won't pollute the fit
                continue
            ###--- the rest case is what we really worry about; take care of it
            other_theta = math.hypot((ox + 1.0) - opt_x, (oy + 1.0) - opt_y) * pixel_scale_arcsec / 60.0
            other_psf_r90 = sample_radius_map(eef_radius_maps, "R90", other.x, other.y)
            if other_psf_r90 is None:
                other_radius_pix, other_frac = load_local_eef(psf_context, other_theta)
                other_psf_r90 = eef_radius(other_radius_pix, other_frac, 0.90)
            contam_radius = max(
                CONTAM_R90_FACTOR * float(other_psf_r90),
                CONTAM_SCALE_FACTOR * float(other.scale),
                CONTAM_MIN_RADIUS_PIX,
            )
            yy, xx = np.indices(data_stamp.shape, dtype=np.float64)
            xx_full = xx + x_min
            yy_full = yy + y_min
            ###--- mask all poluted pixels within the contamination radius of the stronger neighbor
            neighbor_mask |= (xx_full - ox) ** 2 + (yy_full - oy) ** 2 <= contam_radius * contam_radius
            has_strong_neighbor = True
        if np.any(neighbor_mask):
            valid_mask = valid_mask & (~neighbor_mask)
            if not np.any(valid_mask):
                if exp_stamp is not None:
                    valid_mask = exp_stamp > 0.0
                    if analysis_mask is not None:
                        valid_mask &= mask_stamp
                elif analysis_mask is not None:
                    valid_mask = mask_stamp.copy()
                else:
                    valid_mask = np.ones_like(data_stamp, dtype=bool)

        ##--- build local PSF image for fitting
        psf_kernel_nom = build_psf_kernel(radius_pix, frac) # build empirical psf kernel from eef curve
        local_psf = embed_kernel(psf_kernel_nom, data_stamp.shape, x0, y0, x_min, y_min)    # creates an array with the same shape as data_stamp
        if local_psf.sum() <= 0:
            local_psf = np.ones_like(data_stamp, dtype=np.float64)
        local_psf /= np.sum(local_psf)

        ##--- null hypothesis: bkg
        c_null = cash_stat(data_stamp, np.maximum(bkg_stamp, EPS), valid_mask=valid_mask)

        ##--- point source fit
        dx_pt, dy_pt, amp_pt, c_point, _ = fit_point_position_cash(
            data_stamp,
            bkg_stamp,
            local_psf,
            exposure=exp_stamp,
            valid_mask=valid_mask,
            max_shift=LOCAL_POINT_MAX_SHIFT_PIX,
        )
        det_like = cash_delta_to_like(c_null - c_point, dof=POINT_SOURCE_DET_LIKE_DOF)

        ##--- extended source fit if necessary
        need_extent_fit = (
            int(row.npix) >= EXTENT_FIT_MIN_NPIX
            and len(row.support_scales) >= EXTENT_FIT_MIN_SUPPORT_SCALES
            and det_like >= min_det_like
            # and len(group_members) == 1 # TODO: relax this requirement?
        )
        if need_extent_fit:
            ###--- try different core radius in beta model
            rc_grid = np.unique(
                np.clip(
                    np.array(
                        [
                            *EXT_RC_GRID_BASE_PIX,
                            *(factor * psf_r50_pix_nom for factor in EXT_RC_GRID_R50_FACTORS),
                            EXT_RC_GRID_R90_FACTOR * psf_r90_pix_nom,
                        ],
                        dtype=np.float64,
                    ),
                    EXT_RC_MIN_PIX,
                    max(EXT_RC_MAX_R90_FACTOR * psf_r90_pix_nom, EXT_RC_MAX_MIN_PIX),
                )
            )
            dx_ext, dy_ext, best_sigma, best_ext_amp, best_ext_cash, _, ext_template = fit_extended_position_cash(
                data_stamp,
                bkg_stamp,
                local_psf,
                exposure=exp_stamp,
                valid_mask=valid_mask,
                max_shift=LOCAL_POINT_MAX_SHIFT_PIX,
                rc_grid=rc_grid,
            )
            ext_like = cash_delta_to_like(c_point - best_ext_cash, dof=EXTENT_LIKE_DOF)
            ext_r75_pix = template_radius_at_fraction(ext_template, 0.75)
        else:
            dx_ext, dy_ext = dx_pt, dy_pt
            best_sigma = DEFAULT_EXTENT_SIGMA_PIX
            best_ext_amp = amp_pt
            best_ext_cash = c_point
            ext_like = 0.0
            ext_r75_pix = eef_radius(radius_pix, frac, 0.75)
        
        ##--- measures radial net-count profile around the fitted source center, for calculation of r50, r90, etc.
        ## NOTE: this will be overwritten by later joint fitting
        ## which will calculate the profile more cleanly by properly deblending closeby sources
        bkg_per_pix = float(np.mean(bkg_stamp[valid_mask])) if np.any(valid_mask) else float(np.mean(bkg_stamp))
        fit_x0 = x0 + dx_pt
        fit_y0 = y0 + dy_pt
        yy_loc, xx_loc = np.indices(data_stamp.shape, dtype=np.float64)
        xx_full = xx_loc + x_min    # x coordinate grid in full image
        yy_full = yy_loc + y_min    # y coordinate grid in full image
        rr = np.sqrt((xx_full - fit_x0) ** 2 + (yy_full - fit_y0) ** 2)
        prof_sel = valid_mask & (rr <= profile_radius)
        if np.any(prof_sel):
            rr_prof = rr[prof_sel]  # 1d array for distance
            vals_prof = (data_stamp - bkg_stamp)[prof_sel]  # 1d array for net counts
            order = np.argsort(rr_prof) # sort by distance to best-fit center
            rr = rr_prof[order]
            cum = np.cumsum(vals_prof[order])   # cumulative net counts at pixel i from nearest to farthest
            cum = np.maximum(cum, 0.0)
        else:
            rr = np.array([], dtype=np.float64)
            cum = np.array([], dtype=np.float64)
        
        ##--- append single measurement; there will be len(rows) measurements in total
        measurements.append(
            FitMeasurement(
                candidate=row,
                det_bkg=float(np.sum(bkg_stamp)),
                bkg_per_pix=bkg_per_pix,
                det_like=float(det_like),   # contemporary det_like based on single source fit; not to appear in final catalog
                ext_like=float(ext_like),
                best_sigma=float(best_sigma),
                r50_meas=float(radius_at_fraction(rr, cum, 0.50)),
                r80_meas=float(radius_at_fraction(rr, cum, 0.80)),
                r90_meas=float(radius_at_fraction(rr, cum, 0.90)),
                point_amp=float(amp_pt),
                ext_amp=float(best_ext_amp),
                dx_pt=float(dx_pt),
                dy_pt=float(dy_pt),
                dx_ext=float(dx_ext),
                dy_ext=float(dy_ext),
                theta_arcmin=float(theta_arcmin),
                psf_r50_pix_nom=float(psf_r50_pix_nom),
                psf_r75_pix_nom=float(psf_r75_pix_nom),
                psf_r80_pix_nom=float(psf_r80_pix_nom),
                psf_r90_pix_nom=float(psf_r90_pix_nom),
                ext_r75_pix=float(ext_r75_pix),
                stamp_radius_pix=float(fit_stamp_radius),
                ml_eff=float(fraction_at_radius(radius_pix, frac, fit_stamp_radius)),
                fit_maskfrac=float(np.mean(valid_mask)),
                has_strong_neighbor=bool(has_strong_neighbor),
                group_id=int(group_id),
                group_size=int(len(group_members)),
            )
        )

    #--- STAGE 2: joint fitting all member sources in a group
    # this is especially needed to deblend closeby sources
    # without this step, closeby sources (within single fit radius)
    # may steal counts (thereby DET_LIKE) from each other in the single source fit
    # leading to biased DET_LIKE and EXT_LIKE
    grouped_measurements = measurements ## inheriting fitting results from step 1: single source fit
    if groups:
        group_iter = tqdm(
            list(enumerate(groups)),
            desc="Group joint fitting",
            unit="grp",
            disable=not show_progress,
        )
        for group_id, member_indices in group_iter:
            if len(member_indices) <= 1:
                continue
            member_items = [grouped_measurements[idx] for idx in member_indices]

            ##--- determine a compact local fitting stamp for the fitted group
            member_x = np.array([float(item.candidate.x - 1.0 + item.dx_pt) for item in member_items], dtype=np.float64)
            member_y = np.array([float(item.candidate.y - 1.0 + item.dy_pt) for item in member_items], dtype=np.float64)
            max_psf_r90 = max(float(item.psf_r90_pix_nom) for item in member_items)
            max_scale = max(float(item.candidate.scale) for item in member_items)
            cx = float(np.mean(member_x))   # group x center
            cy = float(np.mean(member_y))   # group y center
            ###--- stamp radius is group_extent + margin
            group_extent = float(np.max(np.hypot(member_x - cx, member_y - cy))) if len(member_items) else 0.0
            margin = max(
                GROUP_STAMP_MARGIN_R90_FACTOR * max_psf_r90,
                GROUP_STAMP_MARGIN_SCALE_FACTOR * max_scale,
                GROUP_STAMP_MARGIN_MIN_PIX,
            )
            stamp_radius = max(group_extent + margin, GROUP_STAMP_MIN_RADIUS_PIX)
            data_stamp, y_min, x_min = extract_stamp(image, cx, cy, stamp_radius)
            bkg_stamp = background_map[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]]
            if exposure_map is not None:
                exp_stamp = exposure_map[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]]
                valid_mask = exp_stamp > 0.0
            else:
                exp_stamp = None
                valid_mask = np.ones_like(data_stamp, dtype=bool)
            if analysis_mask is not None:
                mask_stamp = np.asarray(analysis_mask[y_min:y_min + data_stamp.shape[0], x_min:x_min + data_stamp.shape[1]], dtype=bool)
                valid_mask &= mask_stamp
            if not np.any(valid_mask):
                if analysis_mask is not None and np.any(mask_stamp):
                    valid_mask = mask_stamp.copy()
                else:
                    valid_mask = np.ones_like(data_stamp, dtype=bool)

            ##--- mask stronger out-of-group neighbors falling accidentally inside the grouped fit stamp
            neighbor_mask = np.zeros_like(valid_mask, dtype=bool)
            ###--- higher wavelet_peak_score wins in the ranking, then net_counts, then counts for tie-breaking
            group_rank = max((_detection_rank(item.candidate) for item in member_items))
            has_outgroup_strong_neighbor = False
            yy, xx = np.indices(data_stamp.shape, dtype=np.float64)
            xx_full = xx + x_min
            yy_full = yy + y_min
            for other_index, other in enumerate(rows):
                if other_index in member_indices:
                    continue
                other_rank = _detection_rank(other)
                if other_rank <= group_rank:
                    continue
                ox = float(other.x - 1.0)
                oy = float(other.y - 1.0)
                if math.hypot(ox - cx, oy - cy) > stamp_radius:
                    continue
                other_theta = math.hypot((ox + 1.0) - opt_x, (oy + 1.0) - opt_y) * pixel_scale_arcsec / 60.0
                other_psf_r90 = sample_radius_map(eef_radius_maps, "R90", other.x, other.y)
                if other_psf_r90 is None:
                    other_radius_pix, other_frac = load_local_eef(psf_context, other_theta)
                    other_psf_r90 = eef_radius(other_radius_pix, other_frac, 0.90)
                contam_radius = max(
                    CONTAM_R90_FACTOR * float(other_psf_r90),
                    CONTAM_SCALE_FACTOR * float(other.scale),
                    CONTAM_MIN_RADIUS_PIX,
                )
                neighbor_mask |= (xx_full - ox) ** 2 + (yy_full - oy) ** 2 <= contam_radius * contam_radius
                has_outgroup_strong_neighbor = True
            if np.any(neighbor_mask):
                valid_mask = valid_mask & (~neighbor_mask)
                if not np.any(valid_mask):
                    if exp_stamp is not None:
                        valid_mask = exp_stamp > 0.0
                        if analysis_mask is not None:
                            valid_mask &= mask_stamp
                    elif analysis_mask is not None:
                        valid_mask = mask_stamp.copy()
                    else:
                        valid_mask = np.ones_like(data_stamp, dtype=bool)

            ##--- create PSF templates for all group members
            templates: list[np.ndarray] = []
            for item in member_items:
                row = item.candidate
                row_x = float(row.x - 1.0 + item.dx_pt)
                row_y = float(row.y - 1.0 + item.dy_pt)
                theta_arcmin = math.hypot((row_x + 1.0) - opt_x, (row_y + 1.0) - opt_y) * pixel_scale_arcsec / 60.0
                radius_pix, frac = load_local_eef(psf_context, theta_arcmin)
                psf_kernel = build_psf_kernel(radius_pix, frac)
                local_psf = embed_kernel(psf_kernel, data_stamp.shape, row_x, row_y, x_min, y_min)
                if local_psf.sum() <= 0.0:
                    local_psf = np.ones_like(data_stamp, dtype=np.float64)
                local_psf /= np.sum(local_psf)
                templates.append(local_psf)

            ##--- STEP 1:
            # bkg + all group member free as point sources
            amps, full_group_cash, _ = fit_group_amplitudes_cash(
                data_stamp,
                bkg_stamp,
                templates,
                exposure=exp_stamp,
                valid_mask=valid_mask,
            )

            ##--- STEP 2:
            # freeing one member while keeping others fixed
            # in order to calculate DET_LIKE, and radial profile for each member
            for local_idx, item in enumerate(member_items):
                row = item.candidate
                other_templates = [templates[idx] for idx in range(len(templates)) if idx != local_idx]
                other_amps = [amps[idx] for idx in range(len(templates)) if idx != local_idx]

                ###--- null model: 
                # bkg + all except this member from the same group fixed at STEP 1 best-fit models
                null_model = np.array(bkg_stamp, dtype=np.float64, copy=True)
                for amp, template in zip(other_amps, other_templates, strict=False):
                    null_model += float(amp) * template
                without_cash = cash_stat(data_stamp, np.maximum(null_model, EPS), valid_mask=valid_mask)

                ###--- calculate refined DET_LIKE for this member as the improvement of cash by adding this member as a point source to the null model (without this source), with all other group members fixed at earlier joint fit best models
                cond_like = cash_delta_to_like(without_cash - full_group_cash, dof=EXTENT_LIKE_DOF)
                item.det_like = float(cond_like)
                item.point_amp = float(amps[local_idx])
                item.group_size = int(len(member_items))
                item.group_id = int(group_id)
                item.group_stamp_radius_pix = float(stamp_radius)
                item.has_strong_neighbor = bool(has_outgroup_strong_neighbor)
                item.det_bkg = float(np.sum(null_model[valid_mask])) if np.any(valid_mask) else float(np.sum(null_model))

                ###--- recompute empirical morphology from the residual profile:
                ### data - background - other_group_members
                fit_x0 = float(row.x - 1.0 + item.dx_pt)
                fit_y0 = float(row.y - 1.0 + item.dy_pt)
                rr_member = np.sqrt((xx_full - fit_x0) ** 2 + (yy_full - fit_y0) ** 2)
                profile_radius = max(
                    PROFILE_R90_FACTOR * item.psf_r90_pix_nom,
                    PROFILE_SCALE_FACTOR * float(row.scale),
                    PROFILE_MIN_RADIUS_PIX,
                )
                prof_sel = valid_mask & (rr_member <= profile_radius)
                if np.any(prof_sel):
                    rr_prof = rr_member[prof_sel]
                    vals_prof = (data_stamp - null_model)[prof_sel]
                    order = np.argsort(rr_prof)
                    rr_prof = rr_prof[order]
                    cum_prof = np.cumsum(vals_prof[order])
                    cum_prof = np.maximum(cum_prof, 0.0)
                    item.r50_meas = float(radius_at_fraction(rr_prof, cum_prof, 0.50))
                    item.r80_meas = float(radius_at_fraction(rr_prof, cum_prof, 0.80))
                    item.r90_meas = float(radius_at_fraction(rr_prof, cum_prof, 0.90))

            ##--- STEP 3:
            # Replacing each member by extended model (if supported) one by one
            # using greedy algorithm, to derive EXT_LIKE for each source. Compared
            # to single source fit in STAGE 1, this tries to resolve blending issues.
            #
            # Start from the all-point grouped model. Rank members by a
            # composite "extended and bright" score, then test them one by one.
            # When one member is accepted as extended, keep that extended model
            # fixed in the current group model before testing the next member.
            #
            # This is more robust than allowing every member to compare against
            # the original all-point null model, which lets weak point sources
            # absorb residual flux from a nearby bright extended source.
            component_models = [
                float(amp) * template for amp, template in zip(amps, templates, strict=False)
            ]
            current_full_model = np.array(bkg_stamp, dtype=np.float64, copy=True)
            for component in component_models:
                current_full_model += component

            ##--- scoring each member by its extentedness and brightness ...
            candidate_scores: list[tuple[float, int]] = []
            for local_idx, item in enumerate(member_items):
                r50_ratio = max(float(item.r50_meas) / max(float(item.psf_r50_pix_nom), EPS), 1.0)
                r80_ratio = max(float(item.r80_meas) / max(float(item.psf_r80_pix_nom), EPS), 1.0)
                extent_proxy = max(r50_ratio - 1.0, 0.0) + 0.5 * max(r80_ratio - 1.0, 0.0)
                brightness_proxy = math.log1p(max(float(item.point_amp), 0.0))
                candidate_scores.append((brightness_proxy * extent_proxy, local_idx))

            ##--- ... and start fitting extended model from the most extended & brighest one
            for _, local_idx in sorted(candidate_scores, key=lambda item: item[0], reverse=True):
                item = member_items[local_idx]
                ###--- null model:
                # Current accepted group model, with this member removed.
                # Other members could either be point source or extended 
                # depending on whether they have been accepted as extended 
                # in the previous iterations of the greedy loop
                null_model = np.array(current_full_model - component_models[local_idx], dtype=np.float64, copy=True)

                ###--- test model 1: 
                # Current accepted group model, with this member represented as point source
                current_cash = float(cash_stat(data_stamp, np.maximum(current_full_model, EPS), valid_mask=valid_mask))
                
                ###--- test model 2: 
                # Current accepted group model, with this member replaced by extended model
                rc_grid = np.unique(
                    np.clip(
                        np.array(
                            [
                                *EXT_RC_GRID_BASE_PIX,
                                *(factor * item.psf_r50_pix_nom for factor in EXT_RC_GRID_R50_FACTORS),
                                EXT_RC_GRID_R90_FACTOR * item.psf_r90_pix_nom,
                            ],
                            dtype=np.float64,
                        ),
                        EXT_RC_MIN_PIX,
                        max(EXT_RC_MAX_R90_FACTOR * item.psf_r90_pix_nom, EXT_RC_MAX_MIN_PIX),
                    )
                )
                best_group_ext_cash = current_cash
                best_group_ext_sigma = DEFAULT_EXTENT_SIGMA_PIX
                best_group_ext_amp = float(item.point_amp)
                best_group_ext_r75 = float(item.psf_r75_pix_nom)
                best_component_model = np.array(component_models[local_idx], dtype=np.float64, copy=True)

                for rc_pix in rc_grid:
                    beta_kernel = beta_model_kernel(templates[local_idx].shape, float(rc_pix))
                    ext_template = signal.fftconvolve(templates[local_idx], beta_kernel, mode="same")
                    ext_template = np.clip(ext_template, 0.0, None)
                    ext_template /= max(float(np.sum(ext_template)), EPS)
                    tmp_amp_ext, tmp_group_ext_cash, _ = fit_amplitude_cash(
                        data_stamp,
                        null_model,
                        ext_template,
                        exposure=exp_stamp,
                        valid_mask=valid_mask,
                    )
                    if tmp_group_ext_cash < best_group_ext_cash:
                        best_group_ext_cash = float(tmp_group_ext_cash)
                        best_group_ext_sigma = float(rc_pix)
                        best_group_ext_amp = float(tmp_amp_ext)
                        best_group_ext_r75 = float(template_radius_at_fraction(ext_template, 0.75))
                        best_component_model = best_group_ext_amp * ext_template

                trial_ext_like = float(cash_delta_to_like(current_cash - best_group_ext_cash, dof=EXTENT_LIKE_DOF))
                ####--- only when improving significantly current fit will we consider it as next extended source
                if trial_ext_like >= float(min_ext_like):
                    item.ext_like = trial_ext_like
                    item.best_sigma = float(best_group_ext_sigma)
                    item.ext_amp = float(best_group_ext_amp)
                    item.ext_r75_pix = float(best_group_ext_r75)
                    component_models[local_idx] = best_component_model
                    current_full_model = np.array(null_model + best_component_model, dtype=np.float64, copy=True)
                ####--- other with leave EXT_LIKE as 0 even when it was
                # (like falsely) classified as extended in previous single source fit
                else:
                    item.ext_like = 0.0
                    item.best_sigma = DEFAULT_EXTENT_SIGMA_PIX
                    item.ext_amp = float(item.point_amp)
                    item.ext_r75_pix = float(item.psf_r75_pix_nom)

    #--- build final catalog rows with measurements and classifications
    catalog_rows: list[CatalogRow] = []
    for item in grouped_measurements:
        row = item.candidate
        psf_r50_pix = item.psf_r50_pix_nom
        stretch = max(item.r50_meas / max(psf_r50_pix, EPS), 1.0)
        pixel_area_arcmin2 = max((pixel_scale_arcsec / 60.0) ** 2, EPS)
        single_scale = len(row.support_scales) <= 1
        single_scale_floor = max(SINGLE_SCALE_DET_LIKE_FACTOR * float(min_det_like), SINGLE_SCALE_DET_LIKE_FLOOR)
        local_maskfrac = float(item.fit_maskfrac)
        has_strong_neighbor = bool(item.has_strong_neighbor)
        ##--- EXT_LIKE-based extent metric
        extent_supported = (
            item.ext_like >= min_ext_like
            and item.best_sigma > EXTENT_CLASSIFY_MIN_SIGMA_PIX
            and item.best_sigma < max(EXTENT_CLASSIFY_MAX_R90_FACTOR * item.psf_r90_pix_nom, EXTENT_CLASSIFY_MAX_SIGMA_PIX)
            and stretch >= EXTENT_CLASSIFY_MIN_STRETCH
            and int(row.npix) >= EXTENT_FIT_MIN_NPIX
            and len(row.support_scales) >= EXTENT_FIT_MIN_SUPPORT_SCALES
            and local_maskfrac >= EXTENT_CLASSIFY_MIN_MASKFRAC
            # and not has_strong_neighbor
            # and int(item.group_size) == 1
        )
        ##--- empirical morphology-based extent metric
        morph_extent_supported = (
            stretch >= MORPH_EXTENT_MIN_STRETCH
            and item.r80_meas >= MORPH_EXTENT_MIN_R80_FACTOR * max(item.psf_r80_pix_nom, EPS)
            and int(row.npix) >= MORPH_EXTENT_MIN_NPIX
            and len(row.support_scales) >= MORPH_EXTENT_MIN_SUPPORT_SCALES
            and item.det_like >= max(MORPH_EXTENT_DET_LIKE_FACTOR * min_det_like, MORPH_EXTENT_DET_LIKE_FLOOR)
            and local_maskfrac >= MORPH_EXTENT_MIN_MASKFRAC
            and item.r50_meas <= MORPH_EXTENT_MAX_R50_FACTOR * max(item.psf_r50_pix_nom, EPS)
            # and not has_strong_neighbor
            # and int(item.group_size) == 1
        )
        ##--- flags for likely spurious sources near the image edge with insufficient exposure
        edge_like_background = (
            local_maskfrac < EDGE_BACKGROUND_MAX_MASKFRAC
            and item.det_like < max(EDGE_BACKGROUND_DET_LIKE_FACTOR * min_det_like, EDGE_BACKGROUND_DET_LIKE_FLOOR)
            and float(row.npix) <= EDGE_BACKGROUND_MAX_NPIX
            and len(row.support_scales) <= EDGE_BACKGROUND_MAX_SUPPORT_SCALES
        )
        ##--- flags for likely spurious "large" sources with very weak detection likelihood
        # diffuse_weak_point = (
        #     stretch >= 4.0
        #     and item.det_like < max(MORPH_EXTENT_DET_LIKE_FACTOR * min_det_like, MORPH_EXTENT_DET_LIKE_FLOOR)
        #     and not extent_supported
        #     and not morph_extent_supported
        # )
        if item.det_like < min_det_like or (single_scale and item.det_like < single_scale_floor) or edge_like_background:
            src_type = "background"
        elif extent_supported or morph_extent_supported:
            src_type = "extended"
        else:
            src_type = "point"
        
        ##--- populate catalog row
        out = CatalogRow.from_candidate(row)
        ###-- public science catalog columns
        out.source_type = src_type
        out.det_like = float(item.det_like)
        out.ext_like = float(max(item.ext_like, min_ext_like) if morph_extent_supported else item.ext_like)
        out.ml_cts = float(item.ext_amp if src_type == "extended" else item.point_amp)
        out.ml_bkg = float(item.bkg_per_pix / pixel_area_arcmin2)
        out.ext = float(item.best_sigma * pixel_scale_arcsec) if src_type == "extended" else 0.0
        out.ml_eff = float(item.ml_eff)
        ###--- debug columns: detection and morphology
        out.bkg_counts = item.det_bkg
        out.scale = float(row.scale)
        out.support_scales = list(row.support_scales)
        out.wavelet_peak_score = float(row.wavelet_peak_score)
        out.min_significance = float(row.min_significance)
        out.npix = int(row.npix)
        out.counts = float(row.counts)
        out.net_counts = float(row.net_counts)
        out.major = float(row.major)
        out.minor = float(row.minor)
        out.theta_deg = float(row.theta_deg)
        ###--- debug columns: grouping, psf, and fit diagnostics
        out.group_id = int(item.group_id)
        out.group_size = int(item.group_size)
        out.group_stamp_radius_pix = float(item.group_stamp_radius_pix)
        out.theta_arcmin = float(item.theta_arcmin)
        out.psf_r50_pix = float(item.psf_r50_pix_nom)
        out.psf_r75_pix = float(item.psf_r75_pix_nom)
        out.psf_r80_pix = float(item.psf_r80_pix_nom)
        out.psf_r90_pix = float(item.psf_r90_pix_nom)
        out.psf_instrument = str(psf_context.meta["instrument"])
        out.psf_filter = str(psf_context.meta["filter"])
        out.psf_line = str(psf_context.meta["line"])
        out.psf_energy_keV = float(psf_context.meta["energy_keV"]) if isinstance(psf_context.meta["energy_keV"], (int, float)) else math.nan
        out.ml_radius_pix = float(item.stamp_radius_pix)
        out.extent_ratio = float(stretch)
        out.fitted_extent_sigma_pix = float(item.best_sigma)
        out.meas_r50_pix = float(item.r50_meas)
        out.meas_r80_pix = float(item.r80_meas)
        out.meas_r90_pix = float(item.r90_meas)
        ###--- debug columns: final catalog-region diagnostics
        out.catalog_shape = "circle"
        out.catalog_radius_pix = float(item.ext_r75_pix if src_type == "extended" else item.psf_r75_pix_nom)
        if src_type == "extended":
            out.x_ima = float(row.x + item.dx_ext)
            out.y_ima = float(row.y + item.dy_ext)
        else:
            out.x_ima = float(row.x + item.dx_pt)
            out.y_ima = float(row.y + item.dy_pt)
        catalog_rows.append(out)

    #--- sort by DET_LIKE and net_counts
    catalog_rows.sort(key=lambda row: (row.det_like, row.net_counts), reverse=True)
    for idx, row in enumerate(catalog_rows, start=1):
        row.id = idx
        row.id_src = idx
    return catalog_rows


def build_source_groups(rows: list[DetectionCandidate], psf_r90_values: list[float]) -> list[list[int]]:
    """Group nearby candidates for joint point-source fitting.

    Parameters
    ----------
    rows : list[DetectionCandidate]
        Candidate source rows.
    psf_r90_values : list[float]
        Local PSF r90 values in pixels for each row.

    Returns
    -------
    groups : list[list[int]]
        Lists of candidate-list indices belonging to each local source group.
        For example, ``groups = [[0], [1, 2, 3], [4], [5, 6]]`` means
        candidates at indices 1, 2, and 3 are grouped together, while
        candidates at indices 0 and 4 remain isolated, and indices 5 and 6
        form another group.

    Notes
    -----
    This function uses a union-find (disjoint-set) structure to merge nearby
    candidates efficiently. Two candidates are linked when their separation is
    smaller than a generous proximity radius controlled by
    ``GROUP_LINK_R90_FACTOR``, ``GROUP_LINK_MIN_PSF_R90_PIX``, and
    ``GROUP_LINK_MIN_RADIUS_PIX``.

    where ``r90_i`` and ``r90_j`` are the local PSF ``r90`` values supplied in
    ``psf_r90_values``. This is intentionally broader than a deblending radius:
    the goal is to place potentially related nearby candidates into the same
    local fit group so they can be modeled jointly.

    Examples
    --------
    Start with four independent candidates:

    ``parent = [0, 1, 2, 3]``

    If candidate 1 and candidate 2 are close enough, ``union(1, 2)`` changes
    the structure to:

    ``parent = [0, 1, 1, 3]``

    If candidate 2 and candidate 3 are also close enough, ``union(2, 3)``
    changes it to:

    ``parent = [0, 1, 1, 1]``

    At that point, candidates 1, 2, and 3 belong to the same local source
    group and can be fit jointly.
    """
    nrow = len(rows)
    if nrow == 0:
        return []
    parent = list(range(nrow))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(i: int, j: int) -> None:
        ri = find(i)    # r for root
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    #--- union nearby sources based on generous linking radius
    #--- and store the linking in ``parent``
    for i, row_i in enumerate(rows):
        xi = float(row_i.x)
        yi = float(row_i.y)
        ri = max(float(psf_r90_values[i]), GROUP_LINK_MIN_PSF_R90_PIX)
        for j in range(i + 1, nrow):
            row_j = rows[j]
            rj = max(float(psf_r90_values[j]), GROUP_LINK_MIN_PSF_R90_PIX)
            ##--- calculate the distance
            dist = math.hypot(float(row_j.x) - xi, float(row_j.y) - yi) # euclidean distance
            ##--- a generous linking radius to ensure all potentially related sources are grouped together for the joint fit; this is not a deblending radius
            link_radius = max(GROUP_LINK_R90_FACTOR * max(ri, rj), GROUP_LINK_MIN_RADIUS_PIX)
            if dist <= link_radius:
                union(i, j)

    #--- return the groups of indices based on the union-find structure
    groups_dict: dict[int, list[int]] = {}  # {root_idx: [member_indices]}
    for idx in range(nrow):
        groups_dict.setdefault(find(idx), []).append(idx)
    groups = list(groups_dict.values())
    groups.sort(key=lambda members: min(members))
    return groups


def estimate_position_error_pix(row: CatalogRow) -> float:
    """Approximate the 1-sigma centroid error in pixels."""
    counts = max(float(row.get("ML_CTS", row.get("net_counts", 0.0))), 1.0)
    psf_scale = max(float(row.get("psf_r50_pix", 1.0)), 0.5)
    return max(psf_scale / math.sqrt(counts), 0.05)


def nearest_neighbor_distance_arcsec(rows: list[CatalogRow], pixel_scale_arcsec: float) -> list[float]:
    """Compute nearest-neighbor separations for all catalog rows."""
    if len(rows) <= 1:
        return [math.nan] * len(rows)
    if all(np.isfinite(row.ra) and np.isfinite(row.dec) for row in rows):
        coords = SkyCoord(ra=[row.ra for row in rows] * u.deg, dec=[row.dec for row in rows] * u.deg)
        out = []
        for idx, coord in enumerate(coords):
            sep = coord.separation(coords).arcsec
            sep[idx] = np.inf
            out.append(float(np.min(sep)))
        return out
    xy = np.array([(row.x_ima, row.y_ima) for row in rows], dtype=np.float64)
    out = []
    for idx, pos in enumerate(xy):
        dist_pix = np.sqrt(np.sum((xy - pos) ** 2, axis=1))
        dist_pix[idx] = np.inf
        out.append(float(np.min(dist_pix) * pixel_scale_arcsec))
    return out


def prune_nearby_sources(rows: list[CatalogRow]) -> list[CatalogRow]:
    """Suppress weaker detections inside the PSF footprint of stronger sources.

    Parameters
    ----------
    rows : list[dict]
        Final source rows after catalog columns are populated.

    Returns
    -------
    pruned_rows : list[dict]
        Source rows after duplicate suppression.
    """
    kept: list[CatalogRow] = []
    ordered = sorted(rows, key=lambda row: float(row.det_like), reverse=True)
    #--- current source being tested
    for row in ordered:
        x = float(row.x_ima)
        y = float(row.y_ima)
        row_psf = max(float(row.psf_r90_pix), PRUNE_MIN_PSF_R90_PIX)
        row_single = len(row.support_scales) <= 1
        row_group = int(row.group_id)
        suppressed = False
        ##--- check if the current source lives next to a strong source
        for strong in kept:
            strong_group = int(strong.group_id)
            ##--- no worries if the contaminant is another member from the same group (already joint-fitted)
            if row_group >= 0 and strong_group == row_group:
                continue
            dx = x - float(strong.x_ima)
            dy = y - float(strong.y_ima)
            dist = math.hypot(dx, dy)
            strong_psf = max(float(strong.psf_r90_pix), PRUNE_MIN_PSF_R90_PIX)
            suppress_radius = max(PRUNE_SUPPRESS_R90_FACTOR * strong_psf, PRUNE_SUPPRESS_MIN_RADIUS_PIX)
            ##--- no worries if the contaminant is distant enough
            if dist > suppress_radius:
                continue
            ##--- if contaminant strong enough: then current source is likely spurious, need to prune out
            strong_multi = len(strong.support_scales) > 1
            if row_single and strong_multi:
                suppressed = True
                break
        if not suppressed:
            kept.append(row)
    #--- reorder and rename target
    kept.sort(key=lambda row: int(row.id_src if row.id_src else row.id))
    for idx, row in enumerate(kept, start=1):
        row.id_src = idx
        row.id = idx
    return kept


def finalize_catalog_columns(
    rows: list[CatalogRow],
    exposure: np.ndarray | None,
    analysis_mask: np.ndarray | None,
    pixel_scale_arcsec: float,
    ecf: float,
) -> list[CatalogRow]:
    """Populate the final eSASS-like catalog columns."""
    if not rows:
        return rows
    nn_arcsec = nearest_neighbor_distance_arcsec(rows, pixel_scale_arcsec)
    for idx, row in enumerate(rows):
        ml_cts = float(row.ml_cts if row.ml_cts else row.net_counts)
        bkg_counts = float(row.bkg_counts)
        cts_err = math.sqrt(max(ml_cts + bkg_counts, 1.0))
        pos_err_pix = estimate_position_error_pix(row)
        ext_arcsec = float(row.ext)
        ext_err = ext_arcsec / math.sqrt(max(ml_cts, 1.0)) if ext_arcsec > 0 else 0.0
        exp_val = exposure_at_position(exposure, float(row.x_ima), float(row.y_ima))
        rate = ml_cts / exp_val if np.isfinite(exp_val) and exp_val > 0 else math.nan
        rate_err = cts_err / exp_val if np.isfinite(exp_val) and exp_val > 0 else math.nan
        if ecf > 0 and np.isfinite(rate):
            flux = rate * ecf
            flux_err = rate_err * ecf if np.isfinite(rate_err) else math.nan
        else:
            flux = 0.0
            flux_err = 0.0
        ra_err = pos_err_pix * pixel_scale_arcsec
        dec_err = pos_err_pix * pixel_scale_arcsec
        if np.isfinite(row.ra) and np.isfinite(row.dec):
            gal = SkyCoord(ra=row.ra * u.deg, dec=row.dec * u.deg).galactic
            row.lii = float(gal.l.deg)
            row.bii = float(gal.b.deg)
        else:
            row.lii = math.nan
            row.bii = math.nan
        ##--- populate the final catalog columns with values
        row.id_src = int(row.id_src if row.id_src else (row.id if row.id else idx + 1))
        row.id_band = 0
        row.id_cluster = 0
        row.ml_cts = ml_cts
        row.ml_cts_err = cts_err
        row.ml_cts_lowerr = cts_err
        row.ml_cts_uperr = cts_err
        row.x_ima = float(row.x_ima)
        row.x_ima_err = pos_err_pix
        row.x_ima_lowerr = pos_err_pix
        row.x_ima_uperr = pos_err_pix
        row.y_ima = float(row.y_ima)
        row.y_ima_err = pos_err_pix
        row.y_ima_lowerr = pos_err_pix
        row.y_ima_uperr = pos_err_pix
        row.ext_err = ext_err
        row.ext_lowerr = ext_err
        row.ext_uperr = ext_err
        row.det_like = float(row.det_like)
        row.ext_like = float(row.ext_like)
        row.ml_bkg = float(row.ml_bkg)
        row.ml_exp = exp_val
        row.ml_flux = flux
        row.ml_flux_err = flux_err
        row.ml_flux_lowerr = flux_err
        row.ml_flux_uperr = flux_err
        row.ml_rate = rate
        row.ml_rate_err = rate_err
        row.ml_rate_lowerr = rate_err
        row.ml_rate_uperr = rate_err
        row.ra_lowerr = ra_err
        row.ra_uperr = ra_err
        row.dec_lowerr = dec_err
        row.dec_uperr = dec_err
        row.radec_err = math.hypot(ra_err, dec_err)
        row.ml_radius = float(row.ml_radius_pix * pixel_scale_arcsec)
        row.catalog_radius_arcsec = float(row.catalog_radius_pix * pixel_scale_arcsec)
        if exposure is not None:
            effective_valid_mask = exposure > 0.0
            if analysis_mask is not None:
                effective_valid_mask &= np.asarray(analysis_mask, dtype=bool)
        else:
            effective_valid_mask = None if analysis_mask is None else np.asarray(analysis_mask, dtype=bool)
        row.maskfrac = mask_fraction(
            exposure,
            float(row.x_ima),
            float(row.y_ima),
            float(row.ml_radius_pix),
            valid_mask=effective_valid_mask,
        )
        row.dist_nn = float(nn_arcsec[idx])
    return rows
