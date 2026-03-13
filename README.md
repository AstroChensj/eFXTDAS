# eFXTDAS

Analysis tools extending official FXTDAS.

This repository contains a set of analysis utilities for Einstein Probe FXT data products, centered on a pure-Python `wavdetect`-style source detection workflow with mission-aware PSF fitting for catalog construction.

## What it does

- Multi-scale Mexican-hat (Marr) wavelet correlation
- Iterative source cleansing for background estimation
- Significance thresholding (Gaussian approximation to correlation statistics)
- Cross-scale peak clustering to suppress single-scale noise detections
- Erbackmap-like carved-and-smoothed background map creation
- Ermldet-like catalog scoring with EP/FXT PSF and Cash-statistic `det_like` / `ext_like`
- Source table output (`.csv`) and DS9 region output (`.reg`)
- Standard catalog CSV by default, with optional internal/debug columns

The implementation follows the algorithmic outline in the Chandra Detect Reference Manual (`2006cxc_ciaoDetect_DOC.pdf`), but is not a bitwise reimplementation of CIAO `wavdetect`.

## Package Layout

- `fxtpsf_helpers/`: mission-aware PSF and EEF utilities
  - `mission.py`: public mission-agnostic PSF interface
  - `ep_fxt.py`: EP/FXT-specific EEF selection and interpolation
  - `models.py`: shared PSF data models
- `fxteefmap/`: EEF-radius map generator
  - `pipeline.py`: map-generation logic
- `fxtsrcdet/`: source-detection and catalog pipeline
  - `pipeline.py`: top-level orchestration and public Python API
  - `detect.py`: wavelet detection stage
  - `background.py`: exposure-aware background-map creation
  - `fit.py`: Cash-statistic fitting utilities
  - `catalog.py`: grouping, PSF-aware classification, pruning, and catalog derivation
  - `models.py`: detector/catalog data models
  - `utils/`: FITS/WCS, logging, image, and other support helpers
- `fxtregions/`: source/background region construction
  - `pipeline.py`: region-building logic
  - `models.py`: region data models
- `archive/`: fallback copies of older implementations

## Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `astropy` for FITS I/O and sky-coordinate output when the input FITS image contains celestial WCS

## Quick start

Run on FITS image (with optional exposure map):

```bash
fxtsrcdet img.fits --expmap expmap.fits --scales "1 2 4 8 16"
```

Add debug/internal columns to the FITS catalog only when needed:

```bash
fxtsrcdet img.fits --debug-columns
```

Run the same full workflow from Python:

```python
from pathlib import Path
import fxtsrcdet as wd

result = wd.fxtsrcdet_pipeline(
    image=Path("img.fits"),
    exposure=Path("expmap.fits"),
    mission="ep-fxt",
    instrument="fxta",
    filter_name="open",
    emin_keV=0.3,
    emax_keV=10.0,
)

rows = result["rows"]
bkg_map = result["background_map"]
```

You can also pass a configuration object to the same API:

```python
result = wd.fxtsrcdet_pipeline(
    image=Path("img.fits"),
    exposure=Path("expmap.fits"),
    config=wd.PipelineConfig(mission="ep-fxt"),
)

rows = result["rows"]
best_sig = result["best_sig"]
```

Main outputs:

- `sources.fits`: source catalog
- `sources.reg`: DS9 image regions

If the input is a FITS image with WCS and `astropy` is installed, the catalog also includes sky-coordinate and angular-size columns such as:

- `RA`, `DEC`
- `major_arcsec`, `minor_arcsec`
- `radius_arcsec`

The final catalog starts with an eSASS-like column block, including:

- `ID_SRC`, `ID_BAND`, `ID_CLUSTER`, `SOURCE_TYPE`
- `ML_CTS_0`, `ML_CTS_ERR_0`, `ML_CTS_LOWERR_0`, `ML_CTS_UPERR_0` in counts
- `X_IMA`, `Y_IMA` and their error columns in pixels
- `EXT` and its error columns in arcsec
- `DET_LIKE_0`, `EXT_LIKE`
- `ML_BKG_0` in counts / arcmin^2
- `ML_EXP_0` in seconds
- `ML_RATE_0` and its error columns in counts / sec
- `ML_FLUX_0` and its error columns in `erg cm^-2 s^-1`, using user-supplied `--ecf`
- `RA`, `DEC` in degrees with error columns in arcsec
- `LII`, `BII` in degrees
- `ML_RADIUS` in arcsec
- `MASKFRAC`, `ML_EFF_0`, `DIST_NN`

When `--debug-columns` is used, the catalog also includes internal diagnostics such as:

- `extent_ratio`
- `wavelet_peak_score`
- `theta_arcmin`
- `psf_r50_pix`, `psf_r75_pix`, `psf_r80_pix`, `psf_r90_pix`
- `psf_instrument`, `psf_filter`, `psf_line`, `psf_energy_keV`
- `meas_r50_pix`, `meas_r80_pix`, `meas_r90_pix`
- `group_id`, `group_size`, `group_stamp_radius_pix`

Optional DS9 sky regions:

```bash
fxtsrcdet img.fits --sky-regfile sources_fk5.reg
```

Optional maps:

```bash
fxtsrcdet img.fits --save-mask srcmask.fits --save-significance bestsig.fits --save-bkgmap bkgmap.fits
```

## Key parameters

- `--scales`: wavelet scales in pixels (e.g. `"1 2 4 8 16"`)
- `--sigthresh`: source detection threshold
- `--bkgsigthresh`: threshold used when cleansing pixels for background estimation
- `--maxiter`: maximum cleansing iterations
- `--iterstop`: stop when newly cleansed pixel fraction drops below this
- `--expthresh`: minimum relative exposure for analysis
- `--ellsigma`: output ellipse scale multiplier
- `--mission`: mission PSF model for catalog scoring, currently `ep-fxt`
- `--instrument`: mission instrument or detector arm, e.g. `fxta` or `fxtb`
- `--filter`: mission filter state, e.g. `open`, `medium`, `thin`, `hole`
- `--emin`, `--emax`: optional image energy bounds in keV
- `--ecf`: optional energy conversion factor for converting `counts/sec` to flux
- `--optaxis-x`, `--optaxis-y`: optical-axis position in 1-based image pixels; defaults to image center
- `--min-det-like`: minimum detection likelihood for non-background classification
- `--min-ext-like`: minimum extent likelihood threshold for `source_type=extended`
- `--sky-regfile`: DS9 `fk5` region output using FITS WCS and `astropy`
- `--save-bkgmap`: write the carved-and-smoothed background map used by the catalog stage

## Notes

- Input image values are expected to be counts (not count-rate).
- This implementation uses a Gaussian approximation for detection significance in correlation space; CIAO `wavdetect` uses more detailed threshold estimation.
- Final source candidates are selected from clustered wavelet peaks across scales, which is a pragmatic approximation to CIAO's cross-scale reconstruction rather than an exact reimplementation.
- The background map is created by replacing source regions with local background estimates and smoothing the carved image, in the spirit of `erbackmap`.
- FITS outputs for derived maps reuse the input FITS header, so saved background maps retain the original WCS.
- `det_like` and `ext_like` are now based on local Cash-statistic fits of background-only, point-source, and extended-source models. This is much closer to eSASS than the previous aperture heuristic, but it is still an approximation rather than a faithful reimplementation of `ermldet`.
- Mission-specific PSF code lives in `fxtpsf_helpers/`. The current implementation supports `ep-fxt` and can be extended for future missions there without changing the detector core.
- The EP/FXT PSF/EEF selector uses calibration curves matched by instrument, filter, and nearest calibration-line energy. If instrument/filter are not specified, it uses the mean of all available curves.
- EP EEF FITS tables contain multiple off-axis extensions named like `0.00arcmin`, `5.66arcmin`, etc. The code computes each source's `theta_arcmin` and selects the nearest extension for that source.
- The EEF FITS radius column is `radius_pixel`, so EP PSF radii are used directly in pixels rather than converted from arcsec.
- `wavelet_peak_score` is only an internal wavelet-ranking quantity. `det_like` is the catalog-stage detection statistic that should be used for filtering and ranking final sources.
- Sky ellipse sizes are locally approximated from WCS pixel scales, so they are most reliable when distortion across a source region is small.
- Use this code as a transparent research baseline and extend/tune as needed for mission-specific calibration.

## Naming

The toolkit is named `eFXTDAS`, short for "analysis tools extending official FXTDAS".
The local directory may still be named `srcdet` in an existing checkout, but the intended project name is `eFXTDAS`.
