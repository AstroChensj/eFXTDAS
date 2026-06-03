"""Background light-curve threshold optimization utilities."""

from fxtbkgoptrate.pipeline import (
    build_gti_from_mask,
    find_robust_iqr_rate,
    find_optimal_rate,
    intersect_gtis,
    load_lightcurve,
    run_bkgoptrate,
    write_diagnostic_table,
)

__all__ = [
    "build_gti_from_mask",
    "find_robust_iqr_rate",
    "find_optimal_rate",
    "intersect_gtis",
    "load_lightcurve",
    "run_bkgoptrate",
    "write_diagnostic_table",
]
