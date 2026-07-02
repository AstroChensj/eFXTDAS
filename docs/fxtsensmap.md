# `fxtsensmap`

## What It Does

`fxtsensmap` generates an EP-FXT aperture-mode sensitivity map from a background map, an exposure map, and local PSF aperture radii.

For each output pixel, the task:

- reads the expected background counts in a circular aperture
- finds the minimum aperture counts required to pass a Poisson background-only false-alarm threshold
- converts the required source counts into a flux limit using the local exposure, requested EEF, and ECF

The current implementation supports one public mode:

- `aper`
  - aperture-count sensitivity using a circular PSF radius at the requested encircled-energy fraction
  - designed for full-image maps and for stacked products from `fxtcombine`

`fxtcombine` can call `fxtsensmap` automatically after spectral stacking to write `stack_sensmap.fits`.

## Basic Usage

### Command-Line Usage

Using a `fxtpsfgen` PSF product:

```bash
fxtsensmap \
  --bkgmap stack_bkgmap.fits \
  --expmap stack_expmap.fits \
  --mask stack_mask.fits \
  --psfprod stack_psfprod.fits \
  --eef 0.90 \
  --out stack_sensmap.fits
```

Using an official `fxtpsfmap` radius image:

```bash
fxtsensmap \
  --bkgmap stack_bkgmap.fits \
  --expmap stack_expmap.fits \
  --psfmap r90_psfmap.fits \
  --eef 0.90 \
  --out sensmap.fits
```

Using a conventional one-sided Gaussian-equivalent threshold:

```bash
fxtsensmap \
  --bkgmap stack_bkgmap.fits \
  --expmap stack_expmap.fits \
  --mask stack_mask.fits \
  --psfprod stack_psfprod.fits \
  --eef 0.90 \
  --sigma 3.0 \
  --out stack_sensmap_3sigma.fits
```

### Python Usage

```python
from pathlib import Path

from fxtsensmap.pipeline import run_fxtsensmap

run_fxtsensmap(
    bkgmap_path=Path("stack_bkgmap.fits"),
    expmap_path=Path("stack_expmap.fits"),
    mask_path=Path("stack_mask.fits"),
    psfprod=Path("stack_psfprod.fits"),
    eef=0.90,
    out_path=Path("stack_sensmap.fits"),
)
```

## Inputs

Required science inputs:

- `--bkgmap`
  - FITS image of expected background counts per pixel
  - must have the same shape as `--expmap`
- `--expmap`
  - FITS image of exposure time in seconds
  - pixels with non-positive exposure are written as `NaN`
- `--eef`
  - encircled-energy fraction defining the aperture radius and the aperture-to-total source-count correction
- exactly one PSF-radius input:
  - `--psfprod`: `fxtpsfgen` observation or stacked PSF product
  - `--psfmap`: official `fxtpsfmap` radius image
- `--out`
  - output sensitivity-map FITS path

Optional inputs and controls:

- `--mask`
  - optional analysis mask
  - finite non-zero pixels are valid; masked pixels are written as `NaN`
  - when used with `fxtcombine`, this is normally `stack_mask.fits`
- `--ecf`
  - count-rate to flux conversion in `ct/s per erg/cm2/s`
  - default: `1.3787e11`
- `--likemin`
  - native detection-likelihood threshold
  - default: `6.0`
- `--sigma`
  - one-sided Gaussian-equivalent false-alarm threshold
  - mutually exclusive with `--likemin`
  - converted internally as `likemin = -ln(norm.sf(sigma))`
- `--energy-kev`
  - optional representative PSF energy when an uncached radius map must be computed from a `fxtpsfgen` product
- `--block-rows`
  - row block size for stacked `psfprod` radius-map computation
- `--jobs`
  - thread workers for stacked `psfprod` radius-map computation

## PSF Radius Inputs

With `--psfprod`, `fxtsensmap` first looks for a cached radius extension matching the requested EEF, such as `R50`, `R75`, `R80`, or `R90`. These maps are written in image pixels by current `fxtpsfgen` products. If the requested EEF is not cached, `fxtsensmap` computes the radius map from the stored PSF/EEF information. For stacked PSF products, this computation is block-wise and can use `--jobs`.

With `--psfmap`, `fxtsensmap` treats the primary image as the radius map. The map must have `BUNIT` set to `arcsec`, `arcsecond`, `arcseconds`, `pixel`, `pixels`, or `pix`. Arcsecond maps are converted to target-image pixels using the background-map WCS. If the PSF map shape differs from the background-map shape, it is reprojected onto the background-map WCS. If the PSF map records an `EEF` keyword, or the official fraction keyword `ECF`, it must match `--eef`.

## Outputs

The output is a primary-image FITS file with:

- data: flux sensitivity in `erg cm-2 s-1`
- invalid, masked, zero-exposure, or unsupported pixels: `NaN`
- header metadata:
  - `BUNIT = erg cm-2 s-1`
  - `SENSMODE`
  - `EEF`
  - `ECF`
  - `LIKEMIN`
  - `SIGMA` and `SIGDEF` when `--sigma` is used
  - `MASKED` and `MASKFILE` when `--mask` is used

## Algorithm

For a pixel `(x, y)`, let:

- `A_xy` be the circular aperture defined by the local PSF radius at the requested EEF
- `B_xy` be the sum of the background map inside `A_xy`
- `T_xy` be the exposure-map value at `(x, y)`
- `q = exp(-likemin)` be the accepted background-only tail probability

`fxtsensmap` finds the smallest integer `N_min` such that:

```text
P(N >= N_min | background = B_xy) <= q
```

where `N` is Poisson distributed. The aperture-contained source counts required for detection are then:

```text
S_ap = max(N_min - B_xy, 0)
```

The output flux limit is:

```text
F_lim = S_ap / (T_xy * ECF * EEF)
```

This is an exact low-count Poisson threshold for the background-only false-alarm probability. It is not the same as a simple `B + 3 * sqrt(B)` Gaussian approximation, and `--sigma` is only a way to choose the equivalent one-sided false-alarm probability.

## Notes

- The current mode is map-based and computes a full sensitivity image. Per-source sensitivity can be read from the output map at the source location.
- The background map is interpreted as expected counts per pixel, not count rate.
- The exposure map is interpreted as seconds.
- The default ECF is a practical package default. For publication-grade flux limits, use an ECF derived for the intended detector combination, energy band, spectral model, and response assumptions.
