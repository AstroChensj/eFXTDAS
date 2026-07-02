# `fxtpsfgen`

## What It Does

`fxtpsfgen` builds reusable PSF support products for EP-FXT analysis.

It has two public modes:

- `build-obs`
  - build one per-observation PSF product from an image footprint and optional
    exposure map
- `stack`
  - combine multiple per-observation PSF products onto one stacked reference
    image using matching weight maps

These PSF products are the preferred public inputs for `fxtsrcdet`,
`fxtsensmap`, and the stacked-imaging stages of `fxtcombine`.

## Basic Usage

### Command-Line Usage

Per-observation PSF product:

```bash
fxtpsfgen build-obs image.fits \
  --expmap expmap.fits \
  --instrument fxta \
  --filter thin \
  --emin 0.3 \
  --emax 10.0 \
  --out obs.psfprod.fits
```

Stacked PSF product:

```bash
fxtpsfgen stack \
  --obs-psf obs_a.psfprod.fits \
  --obs-psf obs_b.psfprod.fits \
  --weightmap exp_a.fits \
  --weightmap exp_b.fits \
  --ref-image stack_cts.fits \
  --out stack_psfprod.fits
```

### Python Usage

```python
from astropy.io import fits

from fxtpsfgen.mapper import build_observation_psf_mapper, build_stacked_psf_mapper

obs_mapper = build_observation_psf_mapper(
    "image.fits",
    expmap_path="expmap.fits",
    instrument="fxta",
    filter_name="thin",
    emin_keV=0.3,
    emax_keV=10.0,
)
obs_mapper.write("obs.psfprod.fits")

with fits.open("stack_cts.fits") as hdul:
    ref_header = hdul[0].header.copy()

stack_mapper = build_stacked_psf_mapper(
    ["obs_a.psfprod.fits", "obs_b.psfprod.fits"],
    ["exp_a.fits", "exp_b.fits"],
    ref_header,
)
stack_mapper.write("stack_psfprod.fits")
```

## Inputs

- `build-obs`
  - required positional image FITS path
  - optional `--expmap` matching weight/exposure map
  - optional `--instrument`, `--filter`, `--emin`, `--emax`
    - these override or supply the metadata needed to select the PSF context
  - required `--out` output PSF-product path
- `stack`
  - repeated `--obs-psf` inputs, one per component observation
  - repeated `--weightmap` inputs, one per component observation
  - required `--ref-image` stacked image whose primary-header WCS defines the
    output frame
  - required `--out` output stacked PSF-product path

The `stack` inputs must stay aligned by order: the first `--obs-psf` is paired
with the first `--weightmap`, and so on.

## Outputs

- observation mode writes one `*.psfprod.fits` product describing the local PSF
  behavior across that observation footprint
- stack mode writes one stacked `*.psfprod.fits` product on the chosen stacked
  image frame
- current products also cache standard radius-at-EEF maps such as `R50`,
  `R75`, `R80`, and `R90`

These products are consumed directly by:

- `fxtsrcdet --psfprod`
- `fxtsensmap --psfprod`
- `fxtcombine` stacked source-detection stages
- any downstream workflow that needs a local PSF/EEF description on one image
  footprint

## Notes

- `fxtcombine` uses `fxtpsfgen build-obs` per observation and then combines the
  resulting products into `stack_psfprod.fits`.
- `fxtsensmap` uses the cached radius maps directly when the requested EEF is
  present, and otherwise computes the requested radius map from the stored
  PSF/EEF information.
- `fxtpsfgen` replaces the need to expose EEF-map bundles as the main public
  PSF-analysis interface.
