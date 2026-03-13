"""Configuration constants for FXT extraction-region generation."""

# Representative EP-FXT source-position accuracy at 90% confidence from the
# mission handbook, considering PSF shape and typical source brightness. 
# Use this for target-to-catalog cross matching rather than the PSF half-power 
# diameter.
FXT_POSITION_ERR90_ARCSEC = 8.6

# Default source extraction radius, in degrees, used when auto mode falls back
# to manual mode for an undetected target.
DEFAULT_SOURCE_RADIUS_DEG = 0.00944444

# Default background annulus inner radius, in degrees, used in the same manual
# fallback branch.
DEFAULT_BKG_INNER_RADIUS_DEG = 0.0233333

# Default background annulus outer radius, in degrees, used in the same manual
# fallback branch.
DEFAULT_BKG_OUTER_RADIUS_DEG = 0.118

# Minimum allowed automatically generated source radius.
MIN_SOURCE_RADIUS_ARCSEC = 15.0

# Smallest contaminant exclusion radius allowed in either source or
# background-region carving.
MIN_EXCLUDE_RADIUS_ARCSEC = 5.0

# Neighbors closer than this are treated as blended rather than carved as
# independent confusing sources.
MIN_EXCLUDE_DIST_ARCSEC = 10.0

# Initial guess for the background annulus inner radius relative to the source
# radius in auto mode.
INITIAL_SRC_TO_BKG_INNER_RATIO = 2.0

# Target source-wing to local-background surface-brightness ratio at the inner
# edge of the background annulus.
MAX_SRC_TO_BKG_RATIO = 0.05

# Desired background annulus area relative to the source extraction area.
BACK_TO_SRC_AREA_RATIO = 10.0

# Contaminant-to-target surface-brightness threshold used when carving
# confusing sources out of the source extraction region.
MAX_CONF_TO_SRC_RATIO = 0.20

# Contaminant-to-background surface-brightness threshold used when carving
# confusing sources out of the background annulus.
MAX_CONF_TO_BACK_RATIO = 0.10

# Upper cap on the auto background inner radius in units of the local r99.
MAX_BACK_R1_TO_R99_RATIO = 3.0

# Maximum allowed width of the background annulus.
MAX_BACK_ANNULUS_WIDTH_ARCSEC = 120.0
