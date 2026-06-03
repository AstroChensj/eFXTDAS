# `fxtbkgoptrate`

## What It Does

`fxtbkgoptrate` optimizes a background-rate threshold from one FITS light curve and can convert the accepted low-background bins into flare GTIs.

It supports two methods:

- `snr`
  - modeled after the XMM SAS `bkgoptrate` idea
  - evaluates candidate rate thresholds from the observed light-curve bins
  - keeps bins with `rate <= threshold`
  - maximizes a simple S/N proxy based on retained exposure and background rate
- `robust_iqr`
  - computes a robust cutoff from the valid light-curve rate distribution
  - uses `Q3 + 1.5 * IQR`
  - is intended for sparse low-count cases where the S/N scan can peg at the minimum sampled rate
  - if that cutoff keeps too little exposure, the final threshold is lifted just enough to satisfy `--min-time-ratio`

Both methods can optionally write:

- a diagnostic FITS table
- a flare-only GTI
- a screened GTI created by intersecting the flare GTI with an existing base GTI

`fxtcombine` uses this task internally for `FF`-mode FSA flare screening by calling the installed `fxtbkgoptrate` CLI entry point with `--method robust_iqr` by default and then reading the persisted summary metadata back from the diagnostic FITS.

## Basic Usage

### Command-Line Usage

```bash
fxtbkgoptrate flare.lc \
  --method snr \
  --diag-out flare_diag.fits \
  --flare-gti-out flare.gti \
  --base-gti base.gti \
  --screened-gti-out screened.gti
```

### Python Usage

```python
from fxtbkgoptrate import run_bkgoptrate

result = run_bkgoptrate(
    "flare.lc",
    method="snr",
    min_time_ratio=0.05,
    diagnostic_outfile="flare_diag.fits",
    flare_gti_outfile="flare.gti",
    base_gti_path="base.gti",
    screened_gti_outfile="screened.gti",
)

print(result["best_threshold"], result["status"])
```

## Main Inputs and Outputs

Common inputs:

- `infile`: FITS light curve, typically containing `TIME` and `RATE` or `COUNT`
- `--method`: threshold method, `snr` or `robust_iqr`
- `--ycol`: optional explicit rate/count column override
- `--min-time-ratio`: minimum retained exposure fraction allowed for the chosen threshold

Optional outputs:

- `--diag-out`: diagnostic FITS table of threshold trials
- `--flare-gti-out`: GTI built from accepted low-background bins
- `--base-gti` plus `--screened-gti-out`: intersect flare GTI with an existing GTI

The CLI prints the chosen threshold to stdout. If no finite optimum is found, it prints `nan`.

The diagnostic FITS header is also the persisted machine-readable summary used by downstream tasks such as `fxtcombine`. In particular:

- `BGOPTCUT`: chosen threshold
- `FRACTLFT`: retained exposure fraction
- `OPTSTAT`: optimizer status such as `optimal` or `no_cut_needed`
- `OPTMETH`: optimization method used to derive the threshold

## Notes

- If the light curve contains `FRACEXP`, it is used automatically unless disabled.
- If the light curve contains `RATE`, that is preferred by default; otherwise `COUNT` or `COUNTS` is used.
- Standalone `fxtbkgoptrate` keeps `snr` as its default for backward compatibility.
- `fxtcombine` defaults to `robust_iqr` for FSA flare screening.
- If no extra cut is needed, the returned status is `no_cut_needed`.
- If no valid optimization is possible, the task returns a fallback status rather than crashing by default.
