from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage

from fxtpsf_helpers import MissionPSFContext, eef_radius, load_local_eef, sample_radius_map
from fxtsrcdet.config import (
    CLUSTER_MIN_RADIUS_PIX,
    CLUSTER_R90_FACTOR,
    EPS,
    LOCAL_MAX_FILTER_SIZE,
    MEXICAN_HAT_TRUNCATE,
    PEAK_TRIM_MIN_RADIUS_PIX,
    PEAK_TRIM_SCALE_FACTOR,
    SINGLE_SCALE_FRAGMENT_MAX_NPIX,
    SINGLE_SCALE_PEAK_SCORE_MARGIN,
)
from fxtsrcdet.utils.imageops import fft_convolve2d
from fxtsrcdet.models import DetectionCandidate
from fxtsrcdet.utils.measure import ellipse_from_pixels
from fxtsrcdet.utils.stats import gaussian_sf, inverse_normal_survival


@dataclass
class ScaleResult:
    scale: float
    correlation: np.ndarray
    background: np.ndarray
    significance: np.ndarray
    source_mask: np.ndarray
    peak_mask: np.ndarray


def detect_sources(
    image: np.ndarray,
    exposure: np.ndarray | None = None,
    scales: Iterable[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    sigthresh: float = 1e-6,
    bkgsigthresh: float = 1e-3,
    maxiter: int = 2,
    iterstop: float = 1e-4,
    expthresh: float = 0.1,
    ellsigma: float = 3.0,
    psf_context: MissionPSFContext | None = None,
    pixel_scale_arcsec: float | None = None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    eef_radius_maps: dict | None = None,
) -> tuple[list[DetectionCandidate], list[ScaleResult], np.ndarray, np.ndarray]:
    """Run the multi-scale wavelet detection stage on an image.

    Parameters
    ----------
    image : np.ndarray
        Input counts image.
    exposure : np.ndarray | None
        Optional exposure map matched to ``image``.
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
    psf_context : MissionPSFContext | None
        Optional mission PSF context used to adapt the cross-scale clustering
        radius to the local PSF size.
    pixel_scale_arcsec : float | None
        Pixel scale in arcsec/pixel, required when ``psf_context`` is used.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based image pixels.
    eef_radius_maps : dict | None
        Optional precomputed EEF-radius map product from ``fxteefmap``.

    Returns
    -------
    detection_result : tuple[list[DetectionCandidate], list[ScaleResult], np.ndarray, np.ndarray]
        ``(rows, per_scale_results, aggregate_source_mask, best_sig)``.
        - ``rows`` is the final list of source candidates after merging multi-scale 
          detections.
        - ``per_scale_results`` is a list of :class:`ScaleResult` objects containing 
          the wavelet results at each scale.
        - ``aggregate_source_mask`` is the boolean union of source masks over all scales. 
          It is more aggressive than the final ``rows`` representation and may contain 
          false source-like fragments that are later rejected.
        - ``best_sig`` is the smallest significance value seen at each 
          pixel across all scales.

    Notes
    -----
    The basic algorithm of this function:

    - Apply multi-scale (i.e., different "width") Mexican Hat filter to the image.
    - For each scale, estimate background map with iterative negative annular 
      filtering. Then identify source-like pixels above detection threshold.
    - Cluster nearby source-like pixels within radius adapted to the local PSF size.
      Merge all candidates in that cluster as one physical indistinguishable source.
      Keep only source that is detected on multiple scales to reject weak one-scale 
      fragments.
      - The intuition of "multi-scale" classification: real sources should be detected 
        at multiple scales (point source at lower scales, while extended sources at 
        larger scales), while noise fluctuations are more likely to be detected at 
        only one scale. This is a common technique in wavelet-based source detection 
        to improve reliability.

    Examples
    --------
    Display the final wavelet-detected sources on top of the input image:

    >>> rows, per_scale, agg_mask, best_sig = detect_sources(image, exposure, scales)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=5)
    >>> for row in rows:
    ...     e = Ellipse(
    ...         (row.x - 1, row.y - 1),
    ...         width=2 * row.major,
    ...         height=2 * row.minor,
    ...         angle=row.theta_deg,
    ...         edgecolor="cyan",
    ...         facecolor="none",
    ...         lw=1.2,
    ...     )
    ...     ax.add_patch(e)
    ...     ax.text(row.x - 1, row.y - 1, str(row.id), color="yellow", fontsize=8)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    data = image.astype(np.float64, copy=False)
    if exposure is None:
        exp = np.ones_like(data)
    else:
        exp = exposure.astype(np.float64, copy=False)
        if exp.shape != data.shape:
            raise ValueError("Exposure map shape must match image shape.")

    if not (0 < sigthresh < 1):
        raise ValueError("sigthresh must be in (0,1).")
    if not (0 < bkgsigthresh < 1):
        raise ValueError("bkgsigthresh must be in (0,1).")
    if bkgsigthresh < sigthresh:
        raise ValueError("bkgsigthresh should be >= sigthresh.")

    z_thresh = inverse_normal_survival(sigthresh)

    #--- iterate to detect sources at all scales
    per_scale: list[ScaleResult] = []   # wavelet results for each scale
    for scale in scales:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("All scales must be > 0.")

        ##--- estimate bkg with iterative negative annular filtering
        kernel = mexican_hat_kernel(scale)
        bkg = iterative_background(
            image=data,
            exposure=exp,
            kernel=kernel,
            bkgsigthresh=bkgsigthresh,
            maxiter=maxiter,
            iterstop=iterstop,
            expthresh=expthresh,
        )

        ##--- source candidates identification
        corr = fft_convolve2d(data, kernel)
        mean_c = fft_convolve2d(bkg, kernel)
        var_c = fft_convolve2d(bkg, kernel * kernel)
        sigma_c = np.sqrt(np.maximum(var_c, EPS))

        z = (corr - mean_c) / sigma_c
        sig = gaussian_sf(z)

        ###--- reject false detections on invalid pixels with low exposure
        rel_exp = exp / max(float(exp.max()), EPS)
        src = (z >= z_thresh) & (rel_exp >= expthresh)  # NOTE: a physically-single source can be split into multiple source pixels with discontinuity, e.g., "1 0 1" case can appear

        per_scale.append(
            ScaleResult(
                scale=scale,
                correlation=corr,
                background=bkg,
                significance=sig,
                source_mask=src,
                peak_mask=local_maxima(src, corr),
            )
        )

    #--- condense multi-scale results into final candidates
    rows, agg_mask, best_sig = combine_scales(
        data,
        per_scale,
        ellsigma,
        z_thresh,
        psf_context=psf_context,
        pixel_scale_arcsec=pixel_scale_arcsec,
        optaxis_x=optaxis_x,
        optaxis_y=optaxis_y,
        eef_radius_maps=eef_radius_maps,
    )
    return rows, per_scale, agg_mask, best_sig


def mexican_hat_kernel(scale: float, truncate: float = MEXICAN_HAT_TRUNCATE) -> np.ndarray:
    """Build an isotropic 2D Mexican-hat wavelet kernel.

    Parameters
    ----------
    scale : float
        Wavelet scale in pixels.
    truncate : float
        Kernel half-width in units of ``scale``.

    Returns
    -------
    kernel : np.ndarray
        A finite-support wavelet kernel with near-zero numerical sum.
    """
    radius = max(2, int(math.ceil(truncate * scale)))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r2 = x.astype(np.float64) ** 2 + y.astype(np.float64) ** 2
    s2 = scale * scale
    core = (2.0 - r2 / s2) * np.exp(-r2 / (2.0 * s2))
    core -= core.mean()
    return core


def estimate_background(
    image: np.ndarray,
    exposure: np.ndarray,
    kernel: np.ndarray,
    expthresh: float,
) -> np.ndarray:
    """Estimate the local background from the negative annulus of a wavelet.

    Parameters
    ----------
    image : np.ndarray
        Input counts image.
    exposure : np.ndarray
        Exposure map matched to ``image``.
    kernel : np.ndarray
        Wavelet kernel for the current detection scale.
    expthresh : float
        Minimum relative exposure used to mask invalid low-exposure pixels in
        the returned background image.

    Returns
    -------
    background : np.ndarray
        Local background estimate in counts/pixel on the image grid.

    Notes
    -----
    The basic algorithm is:

    1. Keep only the negative annulus of the Mexican-hat wavelet.
    2. Convolve the counts image with that annulus to measure weighted local
       annulus counts around each pixel.
    3. Convolve the exposure map with the same annulus to measure the local
       effective exposure in that annulus.
    4. Divide the two to estimate a local background rate.
    5. Multiply by the exposure map to convert the local rate back to expected
       background counts per pixel on the image grid.
    6. Zero the background in pixels whose relative exposure falls below
       ``expthresh``.

    Only the negative part of the wavelet is used here because it samples the
    local surroundings of each pixel and therefore acts like a weighted local
    background aperture.

    The exposure normalization is necessary when the exposure is spatially
    non-uniform because of vignetting, chip edges, stacked-image boundaries, or
    invalid pixels outside the field of view. Without the exposure convolution,
    edge or low-exposure regions would appear to have artificially low
    background simply because fewer valid pixels contribute to the annulus.
    """
    nw = np.where(kernel < 0.0, -kernel, 0.0)   # keeping only the negative annulus ring (now positive)
    if not np.any(nw > 0):
        return np.clip(image, 0.0, None)
    num = fft_convolve2d(image, nw)     # each pixel now is the averaged value of the annulus around it
    den = fft_convolve2d(exposure, nw)  # and do the same to the expmap
    norm_background = num / np.maximum(den, EPS)
    bkg = exposure * norm_background
    rel = exposure / max(float(exposure.max()), EPS)
    bkg[rel < expthresh] = 0.0  # bkg at invalid pixels will be flagged 0 
    return np.clip(bkg, 0.0, None)


def iterative_background(
    image: np.ndarray,
    exposure: np.ndarray,
    kernel: np.ndarray,
    bkgsigthresh: float,
    maxiter: int,
    iterstop: float,
    expthresh: float,
) -> np.ndarray:
    """Iteratively suppress likely source pixels while refining the background.

    Parameters
    ----------
    image : np.ndarray
        Input counts image.
    exposure : np.ndarray
        Exposure map matched to ``image``.
    kernel : np.ndarray
        Wavelet kernel for the current detection scale.
    bkgsigthresh : float
        Loose significance threshold used to flag likely source pixels during
        background cleansing.
    maxiter : int
        Maximum number of cleansing iterations.
    iterstop : float
        Stop once the fraction of newly cleaned pixels drops below this value.
    expthresh : float
        Minimum relative exposure allowed when marking source-like pixels.

    Returns
    -------
    background : np.ndarray
        Refined local background estimate in counts/pixel.

    Notes
    -----
    A one-pass annulus-based background estimate is often biased high around
    real sources because source counts leak into the local background aperture.
    This routine reduces that bias by iterating:

    1. Estimate a background from the current working image.
    2. Compute the wavelet correlation expected from that background.
    3. Mark pixels whose wavelet response is unlikely under background alone.
    4. Replace those pixels with the current background estimate.
    5. Repeat until the number of newly cleaned pixels becomes small or
       ``maxiter`` is reached.

    The final background is then recomputed once more from the cleaned image.
    """
    cleaned = image.astype(np.float64, copy=True)
    n_pixels = image.size
    z_bkg = inverse_normal_survival(bkgsigthresh)

    for _ in range(maxiter):
        #--- estimate bkg with negative annulus on current image
        bkg = estimate_background(cleaned, exposure, kernel, expthresh)
        mean_c = fft_convolve2d(bkg, kernel) # to keep consistent treatment with cleaned image, we convolve bkg with the same filter
        var_c = fft_convolve2d(bkg, kernel * kernel)
        sigma_c = np.sqrt(np.maximum(var_c, EPS))

        #--- mark likely source pixels
        corr = fft_convolve2d(cleaned, kernel)
        z = (corr - mean_c) / sigma_c
        src = z >= z_bkg

        #--- keep only reliable sources with valid exposure
        rel = exposure / max(float(exposure.max()), EPS)
        src &= rel >= expthresh

        #--- replace those source-like pixels with bkg
        # np.count_nonzero measures how many pixels are being newly cleaned in this iteration
        n_new = int(np.count_nonzero(src & (cleaned != bkg)))
        frac = n_new / n_pixels
        cleaned[src] = bkg[src]
        if frac < iterstop:
            break

    return estimate_background(cleaned, exposure, kernel, expthresh)


def local_maxima(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Find local maxima inside a boolean support mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask defining which pixels are eligible to be considered as
        source peaks.
    values : np.ndarray
        Image of values to maximize locally, typically the wavelet correlation
        map at one detection scale.

    Returns
    -------
    peak_mask : np.ndarray
        Boolean mask marking pixels that are equal to the maximum value in
        their local neighborhood, whose size is controlled by
        ``LOCAL_MAX_FILTER_SIZE``, and also lie inside ``mask``.

    Notes
    -----
    A local maximum filter is applied to ``values`` using
    ``LOCAL_MAX_FILTER_SIZE``. A pixel is kept when:

    - it is inside ``mask``
    - its value is equal to the local neighborhood maximum

    This picks out candidate peak pixels inside each thresholded source-like
    region before later clustering and merging across scales.

    Several cases where a connected ``mask`` component may fail to contribute a
    useful peak pixel in later reconstruction:

    - the component is a weak thresholded fragment attached to the shoulder of
      a brighter nearby source, so its largest correlation value still lies in
      the brighter source's neighborhood rather than inside the fragment
    - the component is only one or two low-significance pixels created by
      thresholding noise, so it is not a stable peak even if it is a connected
      component in ``mask``
    - the component is a broad, nearly flat plateau, so several adjacent pixels
      may all satisfy the local-maximum test and no single pixel stands out as
      a unique peak
    - two bright pixels separated by a sub-threshold gap can appear as
      ``1 0 1`` in ``mask``; this yields two connected components, each treated
      independently by later labeling

    Example
    -------
    For the 2D array

    ``[[0, 1, 0, 0],``
    `` [1, 4, 2, 0],``
    `` [0, 2, 3, 1],``
    `` [0, 0, 1, 0]]``

    the pixel with value ``4`` is a local maximum because it is the largest
    value in its local neighborhood. The pixel with value ``3`` is not a
    local maximum because the value ``4`` lies within its neighborhood.
    """
    local_max = ndimage.maximum_filter(values, size=LOCAL_MAX_FILTER_SIZE, mode="nearest")
    return mask & (values >= local_max)


def cluster_peak_candidates(
    candidates: list[DetectionCandidate],
    radius: float = CLUSTER_MIN_RADIUS_PIX,
    psf_context: MissionPSFContext | None = None,
    pixel_scale_arcsec: float | None = None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    eef_radius_maps: dict | None = None,
) -> list[list[DetectionCandidate]]:
    """Group nearby peak candidates from different wavelet scales.

    Parameters
    ----------
    candidates : list[DetectionCandidate]
        Per-scale provisional candidates to be merged across scales.
    radius : float
        Minimum clustering radius in pixels when no PSF information is
        available.
    psf_context : MissionPSFContext | None
        Optional mission PSF context used to adapt the clustering radius to the
        local PSF size.
    pixel_scale_arcsec : float | None
        Image pixel scale in arcsec/pixel. Required when ``psf_context`` is
        used.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based image pixels.
    eef_radius_maps : dict | None
        Optional precomputed EEF-radius map product from ``fxteefmap``.

    Returns
    -------
    clusters : list[list[DetectionCandidate]]
        Candidate groups merged by spatial proximity. Each inner list contains
        provisional detections believed to represent the same physical source.

    Notes
    -----
    Candidates are processed from highest to lowest ``z_peak`` so the strongest
    peak anchors each cluster first. Without PSF information, a fixed radius is
    used. When PSF information is available, the link radius is adapted to the
    local ``r90`` scale so broader off-axis PSFs can merge more permissively
    than sharp on-axis PSFs.
    """
    #--- determine cluster radius
    clusters: list[list[DetectionCandidate]] = []
    dynamic_radius = (
        psf_context is not None
        and pixel_scale_arcsec is not None
        and pixel_scale_arcsec > 0.0
    )
    if dynamic_radius:
        if optaxis_x is None or optaxis_y is None:
            peak_x = np.array([cand.peak_x for cand in candidates], dtype=np.float64)
            peak_y = np.array([cand.peak_y for cand in candidates], dtype=np.float64)
            optaxis_x = float(0.5 * (np.nanmax(peak_x) + np.nanmin(peak_x))) if len(peak_x) else 0.0
            optaxis_y = float(0.5 * (np.nanmax(peak_y) + np.nanmin(peak_y))) if len(peak_y) else 0.0

    def local_cluster_radius(cand: DetectionCandidate) -> float:
        if not dynamic_radius:
            return float(radius)
        local_r90 = sample_radius_map(eef_radius_maps, "R90", cand.peak_x, cand.peak_y)
        if local_r90 is None:
            theta_arcmin = math.hypot(float(cand.peak_x) - float(optaxis_x), float(cand.peak_y) - float(optaxis_y)) * float(pixel_scale_arcsec) / 60.0
            radius_pix, frac = load_local_eef(psf_context, theta_arcmin)
            local_r90 = float(eef_radius(radius_pix, frac, 0.90))
        return max(CLUSTER_R90_FACTOR * float(local_r90), float(radius))

    #--- for each candidate ...
    for cand in sorted(candidates, key=lambda r: r.z_peak, reverse=True):   # highest-z first
        cand_radius = local_cluster_radius(cand)
        ##--- ... we search through existing clusters ...
        for cluster in clusters:
            cx = float(np.mean([item.peak_x for item in cluster]))
            cy = float(np.mean([item.peak_y for item in cluster]))
            dx = cand.peak_x - cx
            dy = cand.peak_y - cy
            cluster_radius = max(local_cluster_radius(item) for item in cluster)
            link_radius = max(cand_radius, cluster_radius)
            ###--- ... if it is close enough to any cluster center, we add it to that cluster and stop searching ...
            if dx * dx + dy * dy <= link_radius * link_radius:
                cluster.append(cand)
                break
        ##--- ... if this is the first candidate (highest-z), we create a new cluster for it
        else:
            clusters.append([cand])
    return clusters


def combine_scales(
    image: np.ndarray,
    per_scale: list[ScaleResult],
    ellsigma: float,
    z_thresh: float,
    psf_context: MissionPSFContext | None = None,
    pixel_scale_arcsec: float | None = None,
    optaxis_x: float | None = None,
    optaxis_y: float | None = None,
    eef_radius_maps: dict | None = None,
) -> tuple[list[DetectionCandidate], np.ndarray, np.ndarray]:
    """Merge per-scale wavelet detections into final source candidates.

    Parameters
    ----------
    image : np.ndarray
        Input counts image. Pixel values are used as weights when measuring the
        centroid and ellipse of each candidate.
    per_scale : list[ScaleResult]
        Per-scale wavelet detection results produced by :func:`detect_sources`.
        Each element contains the source mask, peak mask, background estimate,
        and significance map for one wavelet scale.
    ellsigma : float
        Ellipse size scale factor passed to :func:`ellipse_from_pixels`.
    z_thresh : float
        Detection threshold in Gaussian ``z`` units, used here to suppress weak
        one-scale fragments.
    psf_context : MissionPSFContext | None
        Optional mission PSF context used to adapt the cross-scale clustering
        radius to the local PSF size.
    pixel_scale_arcsec : float | None
        Pixel scale in arcsec/pixel, required when ``psf_context`` is used.
    optaxis_x : float | None
        Optional optical-axis x coordinate in 1-based image pixels.
    optaxis_y : float | None
        Optional optical-axis y coordinate in 1-based image pixels.
    eef_radius_maps : dict | None
        Optional precomputed EEF-radius map product from ``fxteefmap``.

    Returns
    -------
    merged_result : tuple[list[DetectionCandidate], np.ndarray, np.ndarray]
        ``(rows, agg_mask, best_sig)`` where ``rows`` is the final candidate
        list after cross-scale clustering, ``agg_mask`` is the boolean union of
        source-mask pixels over all scales, and ``best_sig`` is the smallest
        significance value seen at each pixel across all scales.

    Notes
    -----
    This function works in three stages.

    1. Build one provisional candidate from each connected source-like
       component at each scale.

       Connected components are labeled from ``result.source_mask``. A single
       physical source can still be split into multiple components if the mask
       is discontinuous, for example a ``1 0 1`` pattern. Local peak pixels are
       taken from ``result.peak_mask``. If a component has multiple peaks, the
       strongest correlation peak is chosen. The component is optionally
       trimmed to a local neighborhood around the peak so one candidate does
       not inherit a large irregular source island.

    2. Convert each provisional component into a :class:`DetectionCandidate`.

       Centroid and ellipse come from the weighted source pixels. Counts and
       net counts are measured on that local pixel set. The per-scale peak
       significance becomes ``wavelet_peak_score`` / ``z_peak``.

    3. Merge nearby candidates into clusters and keep only clusters detected
       on multiple scales.

       Candidates are grouped only by spatial proximity. The strongest member
       of each cluster is kept as the representative row. Support scales from
       all cluster members are merged into one list. Weak one-scale fragments
       are rejected before returning the final rows.

    ``agg_mask`` is more aggressive than the final ``rows`` representation. It
    is not simply the union of the ellipse areas stored in ``rows`` and may
    contain false source-like fragments that are later rejected.

    Examples
    --------
    Display the returned wavelet ellipses on top of the input image:

    >>> rows, agg_mask, best_sig = combine_scales(image, per_scale, ellsigma, z_thresh)
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> ax.imshow(image, origin="lower", cmap="gray", vmin=0, vmax=5)
    >>> for row in rows:
    ...     e = Ellipse(
    ...         (row.x - 1, row.y - 1),
    ...         width=2 * row.major,
    ...         height=2 * row.minor,
    ...         angle=row.theta_deg,
    ...         edgecolor="cyan",
    ...         facecolor="none",
    ...         lw=1.2,
    ...     )
    ...     ax.add_patch(e)
    ...     ax.text(row.x - 1, row.y - 1, str(row.id), color="yellow", fontsize=8)

    """
    if not per_scale:
        shape = image.shape
        return [], np.zeros(shape, dtype=bool), np.ones(shape, dtype=np.float64)

    best_sig = np.ones_like(image, dtype=np.float64)    # smaller number means more significant detection
    agg_mask = np.zeros_like(image, dtype=bool)
    candidates: list[DetectionCandidate] = []

    #--- construct candidate source catalog at each scale
    # with certain heuristics to prevent the source from inheriting artifacts from neighboring structures
    # with fitted ellipse parameters for each source
    for result in per_scale:
        best_sig = np.minimum(best_sig, result.significance)
        agg_mask |= result.source_mask  # NOTE: a physically-single source can be split into multiple source pixels with discontinuity, e.g., "1 0 1" case can appear
        labeled, nlab = ndimage.label(result.source_mask)   # label connected source regions with unique integers, and nlab is the number of such regions
        peak_pixels = np.argwhere(result.peak_mask) # e.g., array([[106,399], [118,321], [132,314]])
        if len(peak_pixels) == 0 or nlab <= 0:
            continue
        peaks_by_label: dict[int, list[tuple[int, int]]] = {}   # key is int, value is list of (int, int)
        # e.g., {1: [(106, 399)], 2: [(118, 321)], 4: [(132, 314)]}
        for py, px in peak_pixels:
            label = int(labeled[py, px])
            if label <= 0 or label > nlab:
                continue
            peaks_by_label.setdefault(label, []).append((int(py), int(px)))
        for label, comp_peaks in peaks_by_label.items():
            pixels = np.argwhere(labeled == label)  # pixel [x,y] belonging to this label (source)
            if len(pixels) == 0:
                continue
            if len(comp_peaks) == 1:
                py, px = comp_peaks[0]
            else:
                peak_vals = [float(result.correlation[py, px]) for py, px in comp_peaks]
                py, px = comp_peaks[int(np.argmax(peak_vals))]
            peak_radius = max(PEAK_TRIM_SCALE_FACTOR * float(result.scale), PEAK_TRIM_MIN_RADIUS_PIX)
            # a label (source) can contain large number of pixels, and may include:
            # - extended wings
            # - bridges toward nearby peaks
            # - irregular low-significance outskirts
            # we need to prevent the source from inheriting these artifacts by keeping only the centralized pixels
            if len(pixels) > 1: # if this label (source) has >1 pixels
                rr2 = (pixels[:, 1] - px) ** 2 + (pixels[:, 0] - py) ** 2   # distance of each component pixel to the peak pixel
                local_pixels = pixels[rr2 <= peak_radius * peak_radius]
                if len(local_pixels) >= 3:
                    pixels = local_pixels
            if len(pixels) == 0:
                continue
            weights = image[pixels[:, 0], pixels[:, 1]] # image value of each component pixel
            xc, yc, major, minor, theta = ellipse_from_pixels(pixels, weights, ellsigma)    # compute the weighted centroid and ellipse parameters of this source
            local_counts = float(np.sum(weights))
            local_bkg = float(np.sum(result.background[pixels[:, 0], pixels[:, 1]]))
            local_z = float(inverse_normal_survival(float(result.significance[py, px])))
            # Use the actual wavelet peak as the representative source center.
            # For large or asymmetric connected components (e.g., two bright 
            # sources close to each other), the weighted centroid (xc,yc) can 
            # drift far from the dominant peak (e.g., at the middle point between
            # the two sources, which is an empty field without any signal) and 
            # seed later PSF fitting at the wrong position.
            x_seed = float(px + 1.0)
            y_seed = float(py + 1.0)
            candidates.append(
                DetectionCandidate(
                    peak_x=x_seed,
                    peak_y=y_seed,
                    x=x_seed,
                    y=y_seed,
                    major=float(max(major, 0.5)),
                    minor=float(max(minor, 0.5)),
                    theta_deg=float(theta),
                    npix=int(len(pixels)),
                    counts=local_counts,
                    net_counts=local_counts - local_bkg,
                    scale=float(result.scale),
                    support_scales=[float(result.scale)],
                    min_significance=float(result.significance[py, px]),
                    wavelet_peak_score=float(local_z),
                    z_peak=float(local_z),
                )
            )
    
    #--- merge nearby candidates into cluster based on distance (no scale info used yet)
    #--- and keep only clusters detected on multiple scales
    clusters = cluster_peak_candidates(
        candidates,
        psf_context=psf_context,
        pixel_scale_arcsec=pixel_scale_arcsec,
        optaxis_x=optaxis_x,
        optaxis_y=optaxis_y,
        eef_radius_maps=eef_radius_maps,
    )
    rows: list[DetectionCandidate] = []
    for cluster in clusters:
        cluster = sorted(cluster, key=lambda row: row.z_peak, reverse=True) # sort by z
        best = cluster[0]
        best.support_scales = sorted({item.scale for item in cluster})
        best.counts = float(max(item.counts for item in cluster))
        best.net_counts = float(max(item.net_counts for item in cluster))
        best.min_significance = float(min(item.min_significance for item in cluster))
        best.wavelet_peak_score = float(max(item.wavelet_peak_score for item in cluster))
        nscale = len(best.support_scales)
        if nscale == 1:
            # Most false candidates are tiny one-scale fragments. Reject them here
            # before they contaminate the background map and later ML fitting.
            if int(best.npix) <= SINGLE_SCALE_FRAGMENT_MAX_NPIX:
                continue
            if float(best.wavelet_peak_score) < (z_thresh + SINGLE_SCALE_PEAK_SCORE_MARGIN):
                continue
        rows.append(best)

    #--- sort by wavelet_peak_score and counts, and assign final IDs
    rows.sort(key=lambda row: (row.wavelet_peak_score, row.net_counts, row.counts), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row.id = idx
    return rows, agg_mask, best_sig
