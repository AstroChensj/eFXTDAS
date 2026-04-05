# `fxtregions`

## What It Does

`fxtregions` builds source and background extraction regions for downstream analysis tools such as `xselect`.

It is intended as an `srctool`-inspired region builder for FXT data. The tool:

- matches an input sky position against a detected source catalog
- adopts the catalog position when the target is detected
- chooses source and background radii in either `manual` or `auto` mode
- carves out contaminating neighboring sources from both source and background regions
- writes DS9/XSELECT-readable region files

Unlike eSASS `srctool`, the current implementation does not use catalog flags such as `AUTO_EXTRACT` or `AUTO_EXCLUDE`.

```{admonition} Warning on current source region
:class: warning
`fxtarfgen` does not recognize complex shape in source region, so at the moment, the exclusion region in source is disabled!

It means that nearby confusing source around target could contaminate source counts. 
```

## Basic Usage

### Command-Line Usage

Auto mode with catalog-driven source sizing:

```bash
fxtregions \
  image.fits \
  sources.fits \
  --ra 9.25937 \
  --dec 9.16681 \
  --mission ep-fxt \
  --instrument fxta \
  --filter open \
  --emin 0.3 \
  --emax 10.0 \
  --mode auto \
  --src-regfile source.reg \
  --bkg-regfile background.reg
```

Auto mode with an optional background map:

```bash
fxtregions \
  image.fits \
  sources.fits \
  --bkgmap bkgmap.fits \
  --ra 9.25937 \
  --dec 9.16681 \
  --mission ep-fxt \
  --instrument fxta \
  --filter open \
  --emin 0.3 \
  --emax 10.0 \
  --mode auto \
  --src-regfile source.reg \
  --bkg-regfile background.reg
```

Manual mode:

```bash
fxtregions \
  image.fits \
  sources.fits \
  --ra 9.25937 \
  --dec 9.16681 \
  --mode manual \
  --src-radius 34.0 \
  --bkg-inner 84.0 \
  --bkg-outer 425.0 \
  --src-regfile source.reg \
  --bkg-regfile background.reg
```

### Python Usage

```python
from pathlib import Path
from fxtregions.pipeline import build_regions

info = build_regions(
    image_path=Path("image.fits"),
    catalog_path=Path("sources.fits"),
    ra_deg=9.25937,
    dec_deg=9.16681,
    mission="ep-fxt",
    instrument="fxta",
    filter_name="open",
    emin_keV=0.3,
    emax_keV=10.0,
    mode="auto",
)

print(info["source_region"])
print(info["background_region"])
print(info["source_radius_arcsec"])
print(info["background_inner_arcsec"], info["background_outer_arcsec"])
```

### Inputs

- required:
  - counts image FITS with celestial WCS
    - positional argument: `image.fits`
    - used to define the pixel scale, source position on the detector, and
      local geometry for region construction
  - detected source catalog from `fxtsrcdet`
    - positional argument: `sources.fits`
    - provides the matched target row and neighboring sources used for
      contaminant exclusion
  - target sky coordinate:
    - `--ra`
    - `--dec`
- optional calibration / context inputs:
  - background map FITS: `--bkgmap`
    - if supplied, local background is sampled directly from this map
    - if omitted, `fxtregions` falls back to catalog `ML_BKG_0`
  - mission / instrument / filter / energy metadata:
    - `--mission`
    - `--instrument`
    - `--filter`
    - `--emin`
    - `--emax`
    - these are used to build the local PSF / source-kernel model
  - optional optical-axis override:
    - `--optaxis-x`
    - `--optaxis-y`
- region-sizing controls:
  - `--mode auto`
    - derive source and background radii from the matched-source brightness and
      local background
  - `--mode manual`
    - use explicit user-supplied radii
  - manual-radius parameters:
    - `--src-radius`
    - `--bkg-inner`
    - `--bkg-outer`
  - matching threshold:
    - `--match-threshold`
    - maximum target-to-catalog separation used to decide whether the target is
      considered detected
- output / logging controls:
  - `--src-regfile`
    - output source region file
  - `--bkg-regfile`
    - output background region file
  - `--log-level`
  - `--log-file`

### Environment Overrides for Internal Constants

`fxtregions` supports package-scoped environment variables for overriding
internal constants defined in `fxtregions/config.py`.

Examples:

```bash
export FXTREGIONS_DEFAULT_SOURCE_RADIUS_DEG=0.012
export FXTREGIONS_MAX_CONF_TO_BACK_RATIO=0.15
export FXTREGIONS_MAX_BACK_ANNULUS_WIDTH_ARCSEC=150
```

Notes:

- this mechanism is intended for global internal-default tuning
- explicit CLI radius and matching arguments still take precedence when supplied
- invalid override values raise an error during import rather than being ignored

### Outputs

For a run such as:

```bash
fxtregions \
  image.fits \
  sources.fits \
  --bkgmap bkgmap.fits \
  --ra 9.25937 \
  --dec 9.16681 \
  --mode auto \
  --src-regfile source.reg \
  --bkg-regfile background.reg
```

the output tree is conceptually:

```text
<working-directory>/
|-- image.fits
|-- sources.fits
|-- bkgmap.fits
|-- source.reg
|-- background.reg
|-- fxtregions.log
```

If no background map is supplied and the default log naming is used, the
minimal tree is:

```text
<working-directory>/
|-- image.fits
|-- sources.fits
|-- source.reg
|-- background.reg
|-- fxtregions.log
```

The products mean:

- `source.reg`
  - DS9/XSELECT-readable source extraction region
  - centered on the adopted target coordinate
  - in current practice, this is written as a plain source aperture without
    carved source exclusions, because complex source-region geometry can confuse
    downstream `fxtarfgen`
- `background.reg`
  - DS9/XSELECT-readable background annulus region
  - includes contaminating-source exclusion regions where needed
- `fxtregions.log`
  - CLI log file
  - by default this is written beside the requested output region files

The Python API also returns a richer summary dictionary from
`build_regions(...)`:

```text
region_info
|-- context
|-- source_region
|-- background_region
|-- source_excludes
|-- background_excludes
|-- source_radius_arcsec
|-- background_inner_arcsec
|-- background_outer_arcsec
|-- pixel_scale_arcsec
|-- theta_arcmin
|-- psf_r90_arcsec
|-- psf_r99_arcsec
```

where:

- `context`
  - matched/adopted target information, effective mode, local background mode,
    and matched-source metadata
- `source_region`
  - DS9 region expression for the source aperture
- `background_region`
  - DS9 region expression for the background annulus
- `source_excludes`
  - contaminant exclusions derived for the source region logic
  - currently useful mainly for diagnostics, since the written source region is
    intentionally kept simple for ARF compatibility
- `background_excludes`
  - contaminant exclusions applied to the background region
- `source_radius_arcsec`
  - final source extraction radius
- `background_inner_arcsec`
  - final background annulus inner radius
- `background_outer_arcsec`
  - final background annulus outer radius
- `pixel_scale_arcsec`
  - inferred image pixel scale
- `theta_arcmin`
  - target off-axis angle in arcminutes
- `psf_r90_arcsec`, `psf_r99_arcsec`
  - local PSF reference radii used when choosing automatic apertures

### Visualization

Typical review workflow:

1. Open the counts image in DS9.
2. Load the source region file.
3. Load the background region file.
4. Check:
   - whether the adopted source center is correct
   - whether the source circle is sensible
   - whether the background annulus avoids the source wings
   - whether contaminating neighbors are excluded appropriately

## Detailed Algorithm and How It Works

### Region Construction Logic

```{mermaid}
flowchart TD
    A[Input RA, Dec, image, catalog] --> B{Target detected in catalog?}
    B -- Yes --> C[Adopt catalog coordinate]
    B -- No --> D[Keep input coordinate]

    C --> E{Requested mode}
    D --> E
    E -- MANUAL --> F[Use user radii]
    E -- AUTO --> E2{Target detected?}
    E2 -- Yes --> G[Use ML_CTS_0 and local background]
    E2 -- No --> H[Fallback to manual defaults]

    F --> I[Construct source circle and background annulus]
    G --> I
    H --> I

    I --> J[Carve exclusion regions from background annulus]
    J --> K{Target detected?}
    K -- Yes --> L[Build source-region exclusions from contaminant vs target brightness]
    K -- No --> M[Use background-style exclusions in source region and warn]

    L --> N{Any exclusion fully covers source region?}
    M --> N
    N -- Yes --> O[Raise error]
    N -- No --> P[Write source and background region files]
```

### Source and Background Extraction Region Construction

1. Match the input coordinate against the detected source catalog.
2. If a match is found within the configured threshold:
   - adopt the catalog position
   - treat the target as detected
3. Otherwise:
   - keep the original input coordinate
   - treat the target as undetected
4. Determine the effective mode:
   - matched target: keep requested `manual` or `auto`
   - unmatched target + requested `auto`: fall back to `manual`
5. Determine the local background:
   - if a background map is supplied, sample it at the target position
   - else if the target is detected, use catalog `ML_BKG_0`
   - else use the nearest detected source `ML_BKG_0` and emit a warning
6. Determine the target brightness:
   - if detected, use catalog `ML_CTS_0`
   - if undetected, do not invent a catalog brightness
7. Determine source and background radii:
   - manual mode uses user-supplied radii
   - unmatched targets in effective manual mode use config defaults
   - matched auto mode uses `ML_CTS_0` and local background following the same broad policy as eSASS `srctool`

### Exclusion Region in Background Region

For every neighboring catalog source:

1. Build a contaminant source model from the local PSF and, when available, the catalog extent.
2. Estimate the contaminant brightness from `ML_CTS_0`.
3. Determine the exclusion radius by comparing the contaminant surface-brightness profile to the local background level.
4. Append that exclusion circle to the background region if it overlaps the background annulus.

This step is applied regardless of whether the target itself is detected or undetected.

### Exclusion Region in Source Region

For every neighboring catalog source:

1. If the target is detected:
   - determine the contaminant exclusion radius by comparing the contaminant surface brightness against the target source surface brightness, following the same broad policy as `srctool`
2. If the target is undetected:
   - the target `ML_CTS_0` is unavailable
   - reuse the background-style contaminant exclusion radius
   - emit a warning through the logger
3. If a single contaminant exclusion fully covers the target source region:
   - abort with an error because the effective source region would be empty

### Final Output

After the source/background geometry and both exclusion sets are determined, the tool writes:

- one source region file
- one background region file

Both are written in DS9/XSELECT-readable syntax.

### Notes on the Current `srctool`-Like Policy

Current auto-mode logic follows the same broad policy as `srctool`:

- source extraction radius is chosen to maximize nominal SNR
- the background annulus starts outside the source region
- the inner background radius is pushed outward until the target source wing level is sufficiently low
- contaminating sources are carved out using surface-brightness thresholds

The implementation is intentionally simpler than eSASS and uses local PSF-derived source kernels rather than the full `srctool` machinery.

## Tunable Parameters and Heuristic Constants

The following constants are defined in [`fxtregions/config.py`](../fxtregions/config.py):

### Matching

- `FXT_POSITION_ERR90_ARCSEC = 8.6`
  - Representative EP-FXT source-position accuracy at 90% confidence.
  - Used as the default target-to-catalog matching threshold.

### Default Manual Fallback Radii

- `DEFAULT_SOURCE_RADIUS_DEG = 0.00944444`
  - Default source radius when auto mode falls back to effective manual mode for an undetected target.
- `DEFAULT_BKG_INNER_RADIUS_DEG = 0.0233333`
  - Default background annulus inner radius in the same fallback case.
- `DEFAULT_BKG_OUTER_RADIUS_DEG = 0.118`
  - Default background annulus outer radius in the same fallback case.

### Source Radius Floors

- `MIN_SOURCE_RADIUS_ARCSEC = 15.0`
  - Lower limit for automatically chosen source radii.

### Contaminant Exclusion Geometry

- `MIN_EXCLUDE_RADIUS_ARCSEC = 5.0`
  - Minimum contaminant exclusion radius.
- `MIN_EXCLUDE_DIST_ARCSEC = 10.0`
  - Neighboring sources closer than this are treated as blended rather than carved out as a separate contaminant.

### Source / Background Radius Relations

- `INITIAL_SRC_TO_BKG_INNER_RATIO = 2.0`
  - Initial background inner-radius guess relative to the source radius.
- `BACK_TO_SRC_AREA_RATIO = 10.0`
  - Target ratio of background annulus area to source region area.
- `MAX_BACK_R1_TO_R99_RATIO = 3.0`
  - Upper cap linking the background inner radius to the local `r99`.
- `MAX_BACK_ANNULUS_WIDTH_ARCSEC = 120.0`
  - Maximum allowed width of the background annulus.

### Surface-Brightness Thresholds

- `MAX_SRC_TO_BKG_RATIO = 0.05`
  - Target source wing level at the inner edge of the background annulus, as a fraction of the local background.
- `MAX_CONF_TO_SRC_RATIO = 0.20`
  - Contaminant exclusion threshold inside the source region, relative to the target source surface brightness.
- `MAX_CONF_TO_BACK_RATIO = 0.10`
  - Contaminant exclusion threshold inside the background annulus, relative to the local background.

## Suggested Future Additions

- a worked crowded-field example with DS9 screenshots
- optional output of the intermediate adopted target state
- optional diagnostic plot of source/background radii and contaminant exclusions
