# `fxtbkgoptrate`

## What It Does

`fxtbkgoptrate` optimizes a background-rate threshold from one FITS light curve and can convert the accepted low-background bins into flare GTIs.

It is modeled after the XMM SAS `bkgoptrate` idea:

- evaluate candidate rate thresholds from the observed light-curve bins
- keep bins with `rate <= threshold`
- maximize a simple S/N proxy based on retained exposure and background rate
- optionally write:
  - a diagnostic FITS table
  - a flare-only GTI
  - a screened GTI created by intersecting the flare GTI with an existing base GTI

`fxtcombine` uses this task internally for `FF`-mode FSA flare screening.

## Basic Usage

### Command-Line Usage

```bash
fxtbkgoptrate flare.lc \
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
- `--ycol`: optional explicit rate/count column override
- `--min-time-ratio`: minimum retained exposure fraction allowed for the chosen threshold

Optional outputs:

- `--diag-out`: diagnostic FITS table of threshold trials
- `--flare-gti-out`: GTI built from accepted low-background bins
- `--base-gti` plus `--screened-gti-out`: intersect flare GTI with an existing GTI

The CLI prints the chosen threshold to stdout. If no finite optimum is found, it prints `nan`.

## Notes

- If the light curve contains `FRACEXP`, it is used automatically unless disabled.
- If the light curve contains `RATE`, that is preferred by default; otherwise `COUNT` or `COUNTS` is used.
- If no extra cut is needed, the returned status is `no_cut_needed`.
- If no valid optimization is possible, the task returns a fallback status rather than crashing by default.
