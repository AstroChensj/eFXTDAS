# `fxtrspgen`

## What It Does

`fxtrspgen` is the current FXT response generator for one extracted spectrum
when the source aperture is defined by an external DS9 region.

It builds both the ARF and the RMF while taking that DS9 source-region file as
the authoritative aperture definition.

Compared with `fxtarfgen` and `fxtrmfgen`, `fxtrspgen`:

- accepts an external DS9 source region directly
- supports source-side exclusion composition
- owns both ARF and RMF generation in one task
- writes decomposed per-energy correction columns into the output ARF
- does not mutate the input PHA unless asked to
- is the response path used by `fxtcombine` Stage 4

## Basic Usage

```bash
fxtrspgen source.pi source.expo source.reg
```

With explicit output names:

```bash
fxtrspgen source.pi source.expo source.reg \
  --arf-out source_rsp.arf \
  --rmf-out source_rsp.rmf
```

With an explicit PSF anchor position:

```bash
fxtrspgen source.pi source.expo source.reg \
  --srcx 301.5 \
  --srcy 298.5
```

To also update the spectrum headers:

```bash
fxtrspgen source.pi source.expo source.reg \
  --update-pha
```

This is the mode used by `fxtcombine`, which keeps the existing `..._src.arf`
and `..._src.rmf` filenames while updating the source PHA `ANCRFILE` and
`RESPFILE` headers in place.

## Inputs

- required:
  - `specfile`
    - authoritative source of detector, filter, datamode, observation dates, and grade selection
  - `expfile`
    - authoritative image-space grid and WCS for rasterizing the source region
  - `regionfile`
    - external DS9 source-region file
- optional source-position override:
  - `--srcx/--srcy`
  - `--ra/--dec`

If no source position override is given, `fxtrspgen` uses the center of the first positive DS9 region component as the PSF anchor point.

## Supported Source Shapes

`fxtrspgen` supports these DS9 source-region primitives in v1:

- circle
- annulus
- ellipse
- box
- polygon
- subtraction / exclusion composition

Unsupported shapes fail explicitly instead of being ignored.

## Response Model

The output ARF keeps the standard OGIP `SPECRESP` column and factorizes it as:

```text
SPECRESP = BASE_ARF * VIGN_CORR * PSF_CORR * REGCOV_CORR
```

The additional per-energy columns are:

- `BASE_ARF`
  - uncorrected CALDB spectral response on the output energy grid
- `VIGN_CORR`
  - region-weighted vignetting factor across the supplied aperture footprint
- `PSF_CORR`
  - PSF fraction captured by the supplied aperture for the chosen source center
- `REGCOV_CORR`
  - explicit region-covering completeness factor derived from the exposure-map footprint
- `TOT_CORR`
  - `VIGN_CORR * PSF_CORR * REGCOV_CORR`

`SPECRESP` remains the standard response column for downstream OGIP tools. The extra columns are additive diagnostics and provenance.

## RMF Behavior

`fxtrspgen` resolves the RMF through the same CALDB-selection policy currently used by `fxtrmfgen`:

- telescope, instrument, detector, filter, datamode, and observation date come from the input spectrum
- the grade selection is recovered from `DSTYP*` / `DSVAL*` when present
- the output RMF is normalized with the same OGIP-style header edits as the legacy task

## Notes

- By default `fxtrspgen` writes the ARF and RMF only.
- `--update-pha` should be used only when the new response pair is intended to become the authoritative response for that exact PHA file.
- `fxtcombine` uses `--update-pha` so the source PHA headers are owned by the
  canonical response-generation task rather than by later manual header edits.
