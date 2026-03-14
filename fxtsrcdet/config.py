"""Internal tuning constants for ``fxtsrcdet``.

These constants collect non-user-facing heuristic values that affect source
detection, background estimation, PSF fitting, and final classification.
They are intentionally separate from the public CLI / ``PipelineConfig``
parameters, which remain the primary user-facing controls.
"""

# Numerical floor used to avoid division-by-zero and log(0) in statistics and
# fitting code.
EPS = 1e-12

# Mexican-hat kernel support is truncated at this many scale lengths.
MEXICAN_HAT_TRUNCATE = 5.0

# Local maxima in the wavelet-correlation image are identified with a 3x3
# neighborhood filter.
LOCAL_MAX_FILTER_SIZE = 3

# Minimum clustering radius in pixels when PSF-based clustering is not used.
CLUSTER_MIN_RADIUS_PIX = 8.0

# When PSF information is available, the clustering link radius is at least
# this fraction of the local PSF r90 size.
CLUSTER_R90_FACTOR = 0.75

# Large connected source islands are trimmed to a neighborhood around the
# dominant peak before ellipse measurement.
PEAK_TRIM_SCALE_FACTOR = 2.5
PEAK_TRIM_MIN_RADIUS_PIX = 4.0

# One-scale wavelet fragments are often noise-like; these thresholds suppress
# tiny or only-marginally-significant candidates.
SINGLE_SCALE_FRAGMENT_MAX_NPIX = 4
SINGLE_SCALE_PEAK_SCORE_MARGIN = 1.0

# Gaussian smoothing in helper routines should never collapse below this width.
MIN_GAUSSIAN_SIGMA_PIX = 0.5

# Background-map source carving radius is tied to the local PSF size and the
# detected wavelet scale, with a hard floor for compact sources.
BACKGROUND_CARVE_R90_FACTOR = 0.8
BACKGROUND_CARVE_SCALE_FACTOR = 1.5
BACKGROUND_CARVE_MIN_RADIUS_PIX = 3.5
BACKGROUND_CARVE_MIN_SUPPORT_SCALES = 2
BACKGROUND_CARVE_MIN_COUNTS = 4.0

# The adaptive background smoother never evaluates sigmas below this floor,
# even if the user supplies a smaller background-smoothing scale.
BACKGROUND_SIGMA_FLOOR_PIX = 4.0

# Each background pixel chooses the smallest smoothing scale that achieves this
# approximate effective number of source-free background counts.
BACKGROUND_TARGET_COUNTS = 100.0

# Pixels with extremely low source-free support at the broadest smoothing scale
# are forced to zero in the final background map.
BACKGROUND_MIN_SUPPORT_WEIGHT = 0.1

# To suppress rare pathological background-rate spikes near carved holes or
# sharp exposure edges, the local rate model is clipped relative to the upper
# tail of source-free rate samples.
BACKGROUND_RATE_CAP_PERCENTILE = 99.9
BACKGROUND_RATE_CAP_FACTOR = 3.0

# Single-source fit stamps are larger than the morphology-profile radius so the
# likelihood fit has enough context, while the radial profile stays local.
FIT_STAMP_R90_FACTOR = 1.75
FIT_STAMP_SCALE_FACTOR = 1.75
FIT_STAMP_MIN_RADIUS_PIX = 6.0
PROFILE_R90_FACTOR = 1.25
PROFILE_SCALE_FACTOR = 1.25
PROFILE_MIN_RADIUS_PIX = 5.0

# Stronger out-of-group neighbors are masked during local fits with a radius
# tied to their PSF and detection scale.
CONTAM_R90_FACTOR = 1.0
CONTAM_SCALE_FACTOR = 1.5
CONTAM_MIN_RADIUS_PIX = 4.0

# Local point/extended single-source fits only search a small grid of centroid
# shifts, and the point-source detection likelihood is evaluated with this
# effective number of free parameters.
LOCAL_POINT_MAX_SHIFT_PIX = 1.0
POINT_SOURCE_DET_LIKE_DOF = 3.0
EXTENT_LIKE_DOF = 1.0
DEFAULT_EXTENT_SIGMA_PIX = 0.5

# Candidate extended-model core radii are clipped to this range relative to
# the local PSF size.
EXT_RC_GRID_BASE_PIX = (0.5, 1.5, 3.0, 6.0)
EXT_RC_GRID_R50_FACTORS = (0.4, 0.7, 1.0)
EXT_RC_GRID_R90_FACTOR = 0.4
EXT_RC_MIN_PIX = 0.5
EXT_RC_MAX_R90_FACTOR = 1.2
EXT_RC_MAX_MIN_PIX = 6.0

# Grouped fit stamps must include all members plus a margin tied to the largest
# local PSF / wavelet scale.
GROUP_STAMP_MARGIN_R90_FACTOR = 1.5
GROUP_STAMP_MARGIN_SCALE_FACTOR = 1.5
GROUP_STAMP_MARGIN_MIN_PIX = 6.0
GROUP_STAMP_MIN_RADIUS_PIX = 10.0

# Nearby detections are grouped conservatively before joint fitting so that
# potentially blended neighbors are modeled together.
GROUP_LINK_R90_FACTOR = 1.25
GROUP_LINK_MIN_PSF_R90_PIX = 4.0
GROUP_LINK_MIN_RADIUS_PIX = 8.0

# Morphology and classification heuristics.
EXTENT_FIT_MIN_NPIX = 5
EXTENT_FIT_MIN_SUPPORT_SCALES = 2
EXTENT_CLASSIFY_MIN_SIGMA_PIX = 1.0
EXTENT_CLASSIFY_MAX_R90_FACTOR = 1.5
EXTENT_CLASSIFY_MAX_SIGMA_PIX = 12.0
EXTENT_CLASSIFY_MIN_STRETCH = 1.25
EXTENT_CLASSIFY_MIN_MASKFRAC = 0.8

MORPH_EXTENT_MIN_STRETCH = 2.5
MORPH_EXTENT_MIN_R80_FACTOR = 1.8
MORPH_EXTENT_MIN_NPIX = 20
MORPH_EXTENT_MIN_SUPPORT_SCALES = 3
MORPH_EXTENT_MIN_MASKFRAC = 0.9
MORPH_EXTENT_MAX_R50_FACTOR = 8.0

# Single-scale detections are penalized unless they are especially significant.
SINGLE_SCALE_DET_LIKE_FACTOR = 4.5
SINGLE_SCALE_DET_LIKE_FLOOR = 28.0

# Weak edge-like detections are more likely to be background fragments.
EDGE_BACKGROUND_MAX_MASKFRAC = 0.75
EDGE_BACKGROUND_DET_LIKE_FACTOR = 3.0
EDGE_BACKGROUND_DET_LIKE_FLOOR = 20.0
EDGE_BACKGROUND_MAX_NPIX = 8
EDGE_BACKGROUND_MAX_SUPPORT_SCALES = 2

# Extent-like morphology also requires a stronger DET_LIKE threshold than the
# basic science-catalog detection cut.
MORPH_EXTENT_DET_LIKE_FACTOR = 3.0
MORPH_EXTENT_DET_LIKE_FLOOR = 20.0

# Final duplicate pruning suppresses weak one-scale sources that fall within
# the PSF footprint of a stronger multi-scale neighbor.
PRUNE_MIN_PSF_R90_PIX = 4.0
PRUNE_SUPPRESS_R90_FACTOR = 1.35
PRUNE_SUPPRESS_MIN_RADIUS_PIX = 8.0

# The 1D amplitude optimizer searches a bounded interval expanded beyond the
# naive excess-count estimate.
AMPLITUDE_FIT_UPPER_SCALE = 2.0
AMPLITUDE_FIT_UPPER_PAD = 1.0
AMPLITUDE_FIT_XATOL = 1e-3

# Standalone fitting defaults used by ``fit.py`` helpers.
FIT_POINT_MAX_SHIFT_PIX = 2.0
FIT_EXTENDED_BETA = 2.0 / 3.0
FIT_EXTENDED_MAX_SHIFT_PIX = 2.0
