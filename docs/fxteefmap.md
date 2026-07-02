# `fxteefmap`

```{admonition} Legacy Internal Task
:class: note
`fxteefmap` is no longer part of the main installed public CLI surface or the
main user-facing task list.

For supported public PSF-product workflows, use `fxtpsfgen` instead.
```

## What It Does

`fxteefmap` creates an image-sized map of local EEF radii for FXT images.

The standard output is a multi-extension FITS product containing per-pixel maps such as:

- `R50`
- `R75`
- `R80`
- `R90`

Each map stores the radius in image pixels corresponding to the requested encircled-energy fraction at that image position.

## Basic Usage

### Command-Line Usage

```bash
fxteefmap img.fits \
  --expmap expmap.fits \
  --mission ep-fxt \
  --instrument fxta \
  --filter open \
  --emin 0.3 \
  --emax 10.0 \
  --out eef_maps.fits
```

### Single-Fraction Output

```bash
fxteefmap img.fits \
  --expmap expmap.fits \
  --mission ep-fxt \
  --instrument fxta \
  --filter open \
  --emin 0.3 \
  --emax 10.0 \
  --eeffrac 0.75 \
  --out r75_map.fits
```

### Python Usage

```python
from fxteefmap import build_eef_radius_maps

maps, meta = build_eef_radius_maps(
    image=image,
    pixel_scale_arcsec=9.6,
    eeffrac_values=(0.50, 0.75, 0.80, 0.90),
    mission="ep-fxt",
    instrument="fxta",
    filter_name="open",
    emin_keV=0.3,
    emax_keV=10.0,
)
```

### Inputs

- required:
  - input image FITS
    - positional argument: `image`
    - used for image shape, WCS, and the target output footprint
- required output target:
  - `--out`
    - output EEF-radius FITS product
- optional calibration / context inputs:
  - exposure map FITS: `--expmap`
    - if supplied, pixels with `exp <= 0` are forced to zero in the output
      radius maps
  - mission / instrument / filter / energy metadata:
    - `--mission`
    - `--instrument`
    - `--filter`
    - `--emin`
    - `--emax`
    - used to resolve the PSF / EEF calibration line
  - optional optical-axis override:
    - `--optaxis-x`
    - `--optaxis-y`
- EEF-output controls:
  - `--eeffrac`
    - request a single EEF-radius map
  - `--fractions`
    - request a multi-extension bundle of several EEF fractions
    - defaults to `0.50 0.75 0.80 0.90`
- logging controls:
  - `--log-level`
  - `--log-file`

### Environment Overrides for Internal Constants

`fxteefmap` supports package-scoped environment variables for overriding
internal constants defined in `fxteefmap/config.py`.

Current example:

```bash
export FXTEEFMAP_DEFAULT_PIXEL_SCALE_ARCSEC=9.6
```

This is used only for the internal fallback path when the input image WCS does
not provide a usable pixel scale.

### Outputs

For a standard multi-extension run such as:

```bash
fxteefmap img.fits \
  --expmap expmap.fits \
  --mission ep-fxt \
  --instrument fxta \
  --emin 0.3 \
  --emax 10.0 \
  --out eef_maps.fits
```

the output tree is conceptually:

```text
<working-directory>/
|-- img.fits
|-- expmap.fits
|-- eef_maps.fits
|-- eef_maps.log
```

For single-fraction mode:

```bash
fxteefmap img.fits \
  --eeffrac 0.75 \
  --out r75_map.fits
```

the tree is:

```text
<working-directory>/
|-- img.fits
|-- r75_map.fits
|-- r75_map.log
```

The products mean:

- `eef_maps.fits`
  - multi-extension FITS bundle
  - primary HDU stores metadata such as mission, instrument, selected PSF line,
    optical axis, and inferred pixel scale
  - image extensions such as `R50`, `R75`, `R80`, and `R90` store per-pixel EEF
    radii in image pixels
- `r75_map.fits`
  - single-image FITS output when `--eeffrac` is used
  - stores one requested encircled-energy-fraction radius map
- `<out>.log`
  - CLI log file
  - by default this is written beside the requested output file

Each output image extension carries or inherits:

- the input image WCS where available
- `BUNIT = pixel`
- the requested `EEF_FRAC` in single-map mode or per extension in
  multi-extension mode
- PSF calibration metadata such as mission, instrument, filter, and optical axis

### Visualization

The output maps can be opened directly in SAOImage DS9.

Each image extension carries:

- WCS copied from the input image
- `EEF_FRAC`
- `BUNIT = pixel`

That makes it practical to inspect spatial PSF broadening across the field.

## Detailed Algorithm and How It Works

1. Load the input image and infer the pixel scale.
2. Build the mission PSF context for the requested instrument / filter / energy band.
3. Compute the optical-axis geometry.
4. For each requested encircled-energy fraction:
   - evaluate local off-axis angle over the image
   - interpolate the local EEF calibration in off-axis angle
   - interpolate the radius corresponding to the requested EEF fraction
5. Write the result as a FITS image or multi-extension FITS product.

The current implementation uses the EP/FXT EEF calibration and interpolates in off-axis angle rather than snapping to the nearest tabulated calibration extension. This keeps the output radius map spatially continuous.

## Tunable Parameters and Heuristic Constants

### User-Facing Parameters

- default output fractions:
  - `0.50, 0.75, 0.80, 0.90`
- single-map mode:
  - `--eeffrac`
- multi-extension mode:
  - `--fractions`

### Pixel Scale Handling

- image pixel scale is inferred from WCS when available
- if WCS is missing or unusable, the current fallback in the CLI path is:
  - `9.6 arcsec/pixel`

### Off-Axis Interpolation

- local EEF radii are interpolated in off-axis angle rather than snapped to the nearest tabulated calibration extension
- endpoint extrapolation uses the nearest available calibration curve

### Outside-FOV Handling

- if an exposure map is supplied, zero-exposure pixels are written as zero in the radius maps

### Default Fraction Choice

The default `R50/R75/R80/R90` set is chosen because:

- `R50` is useful for PSF-core and extent checks
- `R75` is useful for public source/cell radii
- `R80` is a useful intermediate profile scale
- `R90` is useful for background carving and fit-support sizing

## Suggested Future Additions

- optional theta map output alongside the radius maps
- a validation page comparing `fxteefmap` radii with `fxtpsfgen` / `fxteefgen` at selected positions
