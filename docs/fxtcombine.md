# `fxtcombine`

## What It Does

`fxtcombine` combines multiple EP-FXT observations of the same target into stacked imaging products, a stacked source spectrum, and, in `FF` mode, a stacked instrumental-background spectrum derived from `fsaevt` when available.

The current workflow:

- scans multiple OBSID directories and builds one coupled per-stream workflow from each `evt`
- for `FF` mode with matching `fsaevt`, processes `fsaevt` first, derives a flare-screened GTI with `fxtbkgoptrate`, then reuses that GTI for both `fsaevt` and `evt`
- for `evt`, creates clean events, multi-band images/light curves, exposure maps, and EEF bundles
- for `fsaevt`, creates the cleaned FSA products needed by `fxtbkggen` and predicts one instrumental-background spectrum per OBSID
- reprojects and stacks `evt` images, exposure maps, and EEF bundles onto a common reference frame
- runs `fxtsrcdet` on the stacked `evt` image
- runs `fxtregions` to generate source and background extraction regions
- re-enters each OBSID to extract source/background spectra and response products from `evt`
- calls external `runXstack` to build the final stacked source spectrum, background spectrum, RMF, and ARF
- sums per-OBSID `fsaevt` instrumental-background spectra into a stacked `stack_instbkgpi.fits`

This makes `fxtcombine` the top-level orchestration task in the current `eFXTDAS` toolkit.

## Basic Usage

### Command-Line Usage

```bash
fxtcombine /data/epfxt \
  --obsid-lst 02001234567,02001234568 \
  --ra 9.25937 \
  --dec 9.16681 \
  --image-energy-ranges "0.3:10.0,10.0:12.0" \
  --lightcurve-energy-ranges "0.1:12.0,10.0:12.0" \
  --jobs 4 \
  --stack-dir combine_stack \
  --out-dir combine_out
```

Reuse already existing intermediates:

```bash
fxtcombine /data/epfxt \
  --obsid-lst obsids.txt \
  --ra 9.25937 \
  --dec 9.16681 \
  --image-energy-ranges "0.3:10.0,10.0:12.0" \
  --lightcurve-energy-ranges "0.1:12.0,10.0:12.0" \
  --out-dir combine_out \
  --skip-existing
```

### Python Usage

```python
from fxtcombine.pipeline import fxtcombine_pipeline

fxtcombine_pipeline(
    src_dir="/data/epfxt",
    obsid_lst="02001234567,02001234568",
    ra=9.25937,
    dec=9.16681,
    out_dir="combine_out",
    stack_dir="combine_stack",
    module="a,b",
    datamode="ff",
    image_energy_ranges="0.3:10.0,10.0:12.0",
    lightcurve_energy_ranges="0.1:12.0,10.0:12.0",
    jobs=4,
    skip_existing=False,
)
```

### Inputs

- required path-like inputs:
  - source directory: `src_dir`
    - one subdirectory per OBSID
    - each OBSID directory is expected to follow the standard FXT archive
      layout used by `get_input_files()`
  - OBSID selection: `--obsid-lst`
    - either a comma-separated OBSID list or a file containing one OBSID per
      line
    - only OBSIDs that both appear in `obsid_lst` and exist under `src_dir`
      are processed
  - output directory: `--out-dir`
    - root directory for per-OBSID products, summary files, and the main log
  - stacked output directory: `--stack-dir`
    - optional directory used only for stacked combined products
    - default: `<out-dir>/stack`
- required target inputs:
  - `--ra`
  - `--dec`
- optional event-selection inputs:
  - `--module`
  - `--datamode`
  - `--grade`
  - `--expr`
- Stage-1 image and light-curve controls:
  - `--image-energy-ranges`
    - comma-separated energy ranges in keV such as `0.3:10.0,10.0:12.0`
    - each range is converted internally to the corresponding PI/channel range
      for the relevant FXT module and then applied through `xselect`
    - one image is generated per range
    - the first requested image band is used by default for later stacked source
      detection, and its `emin`/`emax` are also passed to `fxteefmap`,
      `fxtsrcdet`, and `fxtregions`
  - `--lightcurve-energy-ranges`
    - comma-separated energy ranges in keV such as `0.1:12.0,10.0:12.0`
    - each range is converted internally to the corresponding PI/channel range
      for the relevant FXT module and then applied through `xselect`
    - one whole-field light curve is generated per range
  - `--mask-expfrac`
    - minimum stacked exposure fraction, relative to the maximum of the
      stacked exposure map, required to keep a pixel in the stacked analysis
      mask
    - pixels are also rejected if the default stacked count image or stacked
      exposure map contains `NaN` or `Inf`
    - the generated `stack_mask.fits` is passed directly to `fxtsrcdet`
- workflow controls:
  - `--disable-flare-screen`
    - disable the default `FF`-mode FSA-based flare screening
  - `--flare-energy-range`
    - energy range in keV used to build the FSA flare-screening light curve
  - `--flare-binsize`
    - flare-screening light-curve bin size in seconds
  - `--flare-min-time-ratio`
    - minimum retained exposure fraction accepted by `fxtbkgoptrate`
  - `--jobs`
    - number of parallel OBSID workers used in Stage 1
    - each worker owns one OBSID and writes only inside that OBSID's own
      `products/` and `products/log/` directory
    - default: `1`
  - `--srcdet-scales`
    - wavelet scales in pixels forwarded to `fxtsrcdet` for stacked source
      detection
    - default: `1,2,4,8,16`
- `--srcdet-background-sigma-grid`
    - Gaussian smoothing scales in pixels forwarded to `fxtsrcdet` for its
      adaptive background model
    - default: `4,8,16,32,64`
  - `--summary-json`
    - optional summary JSON path
    - default: `<stack_dir>/all_obsid.json`
  - `--srcpi-filelist`
    - optional `runXstack` source-spectrum file list path
    - default: `<stack_dir>/all_obsid.filelist`
  - `--skip-existing`
  - `--log-level`
  - `--log-file`

### Environment Overrides for Internal Constants

`fxtcombine` supports package-scoped environment variables for overriding
internal constants defined in `fxtcombine/config.py`.

Examples:

```bash
export FXTCOMBINE_SRC_EXTRACT_RADIUS=75
export FXTCOMBINE_BKG_EXTRACT_OUTER_RADIUS=360
export FXTCOMBINE_INSTBKG_BACKSCAL_RELSTD_WARN=0.2
```

Notes:

- these overrides affect internal defaults used when `fxtcombine` calls other
  package tasks or evaluates stacked instrumental-background consistency
- explicit CLI and Python arguments still remain the main user-facing controls
- invalid override values raise an error during import rather than being ignored

### Outputs

`fxtcombine` writes a directory tree under `out_dir` that contains:

- per-OBSID intermediate products under `<out-dir>/<OBSID>/products/`
- stacked imaging / region / spectral products under `stack_dir`
- summary files such as `all_obsid.json` and `all_obsid.filelist` under `stack_dir`
- main and per-step log files

The full output layout, including the relationship between `src_dir`,
`obsid_lst`, and `out_dir`, is shown in the next section:

- `Output Data Structure`

## Output Data Structure

The most useful way to understand `fxtcombine` outputs is as a directory tree
linked to the four main path-like inputs:

- `src_dir`: input archive with one subdirectory per OBSID
- `obsid_lst`: which OBSID subdirectories under `src_dir` are actually used
- `out_dir`: where all combined products are written
- `stack_dir`: where the stacked combined science products are written
- `summary_json`: summary JSON path, defaulting to `<stack_dir>/all_obsid.json`
- `srcpi_filelist`: `runXstack` file list path, defaulting to `<stack_dir>/all_obsid.filelist`

### Input/Output Relationship

If the user runs:

```bash
fxtcombine <src_dir> \
  --obsid-lst 11900458112,11900465408 \
  --out-dir <out_dir> \
  --stack-dir <stack_dir> \
  --image-energy-ranges 0.3:10.0,1.0:3.0 \
  --lightcurve-energy-ranges 0.1:12.0,1.0:3.0
```

then the layout is conceptually:

```text
<src_dir>/
|-- 11900458112/
|   |-- ... original FXT archive content ...
|-- 11900465408/
|   |-- ... original FXT archive content ...
|-- ...

<out_dir>/
|-- log/
|   |-- fxtcombine.log
|-- 11900458112/
|   |-- products/
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.coord
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.particle
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.badpix
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.grade
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.gti
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_screened.gti
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_cl.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>.expo
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e00300_10000.eef
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e01000_03000.eef
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e00300_10000.img
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e01000_03000.img
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e00100_12000.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_e01000_03000.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_src_cl.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_src.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_bkg.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_src.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_bkg.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_src.arf
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_src.rmf
|   |   |-- target_src.reg
|   |   |-- target_bkg.reg
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.coord
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.particle
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.badpix
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.grade
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.gti
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>_flare.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_flare_diag.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>_flare.gti
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>_cl.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>.img
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>_instbkgpi.fits
|   |   |-- log/
|   |   |   |-- fxtchain.log
|   |   |   |-- <module>_<obsid>_<mode>_<filter>_<pp>_evt_<ver>/
|   |   |   |   |-- evt_stage1.xsl
|   |   |   |   |-- evt_stage4_spec.xsl
|   |   |   |   |-- fxtcoord_evt.log
|   |   |   |   |-- fxtpical_evt.log
|   |   |   |   |-- fxtparticleidentify_evt.log
|   |   |   |   |-- fxtbadpix_evt.log
|   |   |   |   |-- fxtgrade_evt.log
|   |   |   |   |-- fxtgtigen_evt.log
|   |   |   |   |-- xselect_evt_stage1.log
|   |   |   |   |-- fxtexpogen_evt.log
|   |   |   |   |-- fxteefmap_evt_e00300_10000.log
|   |   |   |   |-- fxteefmap_evt_e01000_03000.log
|   |   |   |   |-- xselect_evt_stage4_spec.log
|   |   |   |   |-- fxtarfgen.log
|   |   |   |   |-- fxtrmfgen.log
|   |   |   |-- <module>_<obsid>_<mode>_<filter>_<pp>_fsaevt_<ver>/
|   |   |   |   |-- fsaevt_flare.xsl
|   |   |   |   |-- fsaevt_stage1.xsl
|   |   |   |   |-- fxtcoord_fsaevt.log
|   |   |   |   |-- fxtpical_fsaevt.log
|   |   |   |   |-- fxtparticleidentify_fsaevt.log
|   |   |   |   |-- fxtbadpix_fsaevt.log
|   |   |   |   |-- fxtgrade_fsaevt.log
|   |   |   |   |-- fxtgtigen_fsaevt.log
|   |   |   |   |-- xselect_fsaevt_flare.log
|   |   |   |   |-- xselect_fsaevt_stage1.log
|   |   |   |   |-- fxtbkggen_fsaevt.log
|-- 11900465408/
|   |-- products/
|   |   |-- ... same product pattern as above ...

<stack_dir>/
|-- all_obsid.filelist
|-- all_obsid.json
|   |-- stack_exp.fits
|   |-- stack_cts.fits
|   |-- stack_rate.fits
|   |-- stack_eef.fits
|   |-- stack_mask.fits
|   |-- e00300_10000_stack_cts.fits
|   |-- e00300_10000_stack_rate.fits
|   |-- e00300_10000_stack_eef.fits
|   |-- e01000_03000_stack_cts.fits
|   |-- e01000_03000_stack_rate.fits
|   |-- e01000_03000_stack_eef.fits
|   |-- stack_src.fits
|   |-- stack_src.reg
|   |-- stack_bkgmap.fits
|   |-- target_src.reg
|   |-- target_bkg.reg
|   |-- srcdet.log
|   |-- fxtregions.log
|   |-- stack_pi.fits
|   |-- stack_bkgpi.fits
|   |-- stack_rmf.fits
|   |-- stack_arf.fits
|   |-- stack_instbkgpi.fits
|   |-- stack_runXstack.log
```

### Meaning of the Tree

- `src_dir/<obsid>/`
  - original input archive for each observation
- `out_dir/<obsid>/products/`
  - all per-OBSID intermediate and extracted products
  - this includes the per-band `.img`, `.lc`, and `.eef` products plus
    per-OBSID step logs under `products/log/`
  - in `FF` mode with matching `fsaevt`, this also includes the flare-screening
    light curve, flare GTIs, cleaned FSA products, and one predicted
    instrumental-background spectrum from `fxtbkggen`
  - when Stage 4 runs, the copied extraction regions
    `target_src.reg` and `target_bkg.reg` are also placed here for the
    per-OBSID spectral extraction
- `stack_dir/`
  - products derived from combining all valid OBSIDs together
  - `stack_instbkgpi.fits` appears here when `fsaevt` is requested
- `stack_dir/all_obsid.json`
  - machine-readable summary of the per-OBSID product paths
- `stack_dir/all_obsid.filelist`
  - source-spectrum file list passed to `runXstack`

### Output Logs

The most useful logs for debugging are:

- `out_dir/log/fxtcombine.log`
  - top-level orchestration log for the whole run
- `out_dir/<obsid>/products/log/fxtchain.log`
  - high-level per-OBSID log showing which Stage-1 or Stage-4 substep failed
- `out_dir/<obsid>/products/log/<module>_<obsid>_<mode>_<filter>_<pp>_<filetype>_<ver>/`
  - detailed task logs for one event stream
  - this is the first place to inspect if an individual FXTDAS task failed
- `stack_dir/srcdet.log`
  - stacked `fxtsrcdet` output
- `stack_dir/fxtregions.log`
  - stacked `fxtregions` output
- `stack_dir/stack_runXstack.log`
  - `runXstack` log for stacked spectral combination

Recommended debugging order:

1. start with `out_dir/log/fxtcombine.log` to identify the failing OBSID or stacked step
2. if the failure is per-OBSID, check `products/log/fxtchain.log`
3. then open the corresponding detailed task log in the `evt` or `fsaevt` subdirectory
4. if the failure is in stacked detection or stacking, inspect the relevant log under `stack_dir`

### Multi-Band Image and Light-Curve Products

When multiple energy ranges are requested:

- each Stage-1 image band gets its own file:
  - `..._e00300_10000.img`
  - `..._e01000_03000.img`
- each Stage-1 image band also gets its own EEF bundle:
  - `..._e00300_10000.eef`
  - `..._e01000_03000.eef`
- each Stage-1 light-curve band gets its own file:
  - `..._e00100_12000.lc`
  - `..._e01000_03000.lc`

For stacking:

- the first requested image band is treated as the default detection band
- that default band is written twice:
  - legacy names:
    - `stack_cts.fits`
    - `stack_rate.fits`
  - explicit band-labelled names:
    - `e00300_10000_stack_cts.fits`
    - `e00300_10000_stack_rate.fits`
    - `e00300_10000_stack_eef.fits`
- additional image bands are written only with explicit band labels
- one stacked analysis mask is built from the default stacked count image and
  stacked exposure map:
  - `stack_mask.fits`

### The Summary JSON Structure

By default, `all_obsid.json` stores one flat per-stream record under each OBSID:

```text
all_obsid.json
|-- <obsid>
|   |-- <stream_key>
|   |   |-- evt_clevt
|   |   |-- image
|   |   |-- images
|   |   |-- vexpmap
|   |   |-- eefmaps
|   |   |-- exp
|   |   |-- lightcurves
|   |   |-- screened_gti
|   |   |-- flare_lc
|   |   |-- flare_diag
|   |   |-- srcclevt
|   |   |-- srcpi
|   |   |-- bkgpi
|   |   |-- arf
|   |   |-- rmf
|   |   |-- fsa_spec
|   |   |-- fsa_lc
|   |   |-- fsa_img
|   |   |-- instbkgpi
```

Important conventions:

- `image` is always the first requested image band
- `images` stores all requested image bands
- `image_band_channels` stores the per-module PI/channel ranges derived from the requested image energy bands
- `alllc` is always the first requested light-curve band
- `lightcurves` stores all requested light-curve bands
- `lightcurve_band_channels` stores the per-module PI/channel ranges derived from the requested light-curve energy bands
- `srcclevt`, `srcpi`, `bkgpi`, `srclc`, `bkglc`, `arf`, and `rmf` appear only after Stage 4
- `flare_lc`, `flare_diag`, `flare_gti`, and `screened_gti` appear when `FF` flare screening runs
- `fsa_spec`, `fsa_lc`, `fsa_img`, and `instbkgpi` appear only when matching `fsaevt` was available

Example: collect all light-curve filenames from `all_obsid.json`:

```python
import json
from pathlib import Path

summary_path = Path("combine_stack/all_obsid.json")

with summary_path.open() as f:
    data = json.load(f)

all_lc = []

for obsid, obsid_products in data.items():
    for stream_key, prod in obsid_products.items():
        for band_key, lc_path in prod.get("lightcurves", {}).items():
            all_lc.append(
                {
                    "obsid": obsid,
                    "stream_key": stream_key,
                    "band": band_key,
                    "path": lc_path,
                }
            )

for row in all_lc:
    print(row["obsid"], row["band"], row["path"])
```

If only the default first-band light curve is needed for each event file:

```python
import json

with open("combine_stack/all_obsid.json") as f:
    data = json.load(f)

default_lc = []

for obsid, obsid_products in data.items():
    for stream_key, prod in obsid_products.items():
        if "alllc" in prod:
            default_lc.append(prod["alllc"])

print(default_lc)
```

## Detailed Algorithm and How It Works

### 1. Per-OBSID Preprocessing

For each selected OBSID and each coupled event stream:

1. if `FF` with matching `fsaevt`, run the FSA chain first through `fxtgtigen`
2. build an FSA flare-screening light curve and run `fxtbkgoptrate`
3. intersect the flare GTI with the base GTI to create a screened GTI
4. apply the screened GTI to `fsaevt` and generate cleaned FSA products plus `instbkg`
5. run the `evt` chain through `fxtgtigen`
6. apply the screened GTI, or the base GTI when flare screening is skipped
7. run `xselect` for cleaned `evt`, one image per requested image band, and one whole-field light curve per requested light-curve band
8. run `fxtexpogen` to produce a vignetted exposure map
9. run `fxteefmap` on each requested image energy band to produce one per-OBSID EEF bundle per band
10. for `fsaevt`, run `fxtbkggen` on the cleaned FSA spectrum to predict one instrumental-background spectrum for the image area

This stage creates the per-OBSID products that are later stacked.
It can optionally be parallelized over OBSIDs with `--jobs`, but each OBSID is
still processed serially inside its own worker so different observations do not
touch the same output files.

### 2. Image and Exposure Stacking

The longest-exposure image is used as the reference frame.

Current stacking behavior:

- each clean event list is projected onto the reference WCS and binned into the reference image grid
- each exposure map is reprojected onto the same reference frame and summed
- a stacked rate map is computed as `counts / exposure`

Only `evt` enters this stage. `fsaevt` does not go through image/exposure/EEF stacking.

### 3. Detection and Region Building

After the stacked `evt` products are written:

1. `fxtcombine` uses the first requested stacked image band and the corresponding stacked EEF bundle
2. `fxtcombine` builds `stack_mask.fits` from the default stacked count image and stacked exposure map
3. `fxtcombine` calls `fxtsrcdet`
4. `fxtsrcdet` writes the stacked source catalog and source region file
5. `fxtcombine` calls `fxtregions`
6. `fxtregions` writes the source and background extraction region files

The current `fxtcombine` usage of `fxtregions` is in `manual` mode, with fixed source and background radii:

- source radius: `60 arcsec`
- background annulus: `90-300 arcsec`

Target-to-catalog matching currently uses the EP-FXT representative position accuracy at 90% confidence:

- `FXT_POSITION_ERR90_ARCSEC = 8.6`

### 4. Per-OBSID Spectral Extraction

Only `evt` enters this stage. After the stacked source/background regions are created, `fxtcombine` loops over each OBSID again and:

- extracts source-filtered events
- extracts source and background PI spectra
- extracts source and background light curves
- runs `fxtarfgen`
- runs `fxtrmfgen`
- updates OGIP header keywords so the products are internally linked

### 5. Spectral Stacking

For `evt`, `fxtcombine` prepares a file list of all extracted source spectra and calls:

- `runXstack`

with same-target mode enabled.

For `fsaevt`, `fxtcombine` does not call `runXstack`. Instead it sums the per-OBSID instrumental-background spectra from `fxtbkggen` into `stack_instbkgpi.fits`, sums their `EXPOSURE`, averages `BACKSCAL`, and reports the standard deviation of `BACKSCAL`. If the relative scatter of `BACKSCAL` is large, a warning is emitted that the stacked instrumental-background spectrum may not be reliable.

So this stage can produce:

- the final stacked source spectrum and associated response products from `evt`
- the final stacked instrumental-background spectrum from `fsaevt`

## Logging and Output Layout

The default main log file is:

- `<out-dir>/log/fxtcombine.log`

Per-OBSID logs are written under:

- `<out-dir>/<OBSID>/products/log/`

Command-specific logs such as `xselect.log`, `fxtarfgen.log`, and `fxtbkggen.log` are written beside the stage that generated them.

## Tunable Parameters and Constants

Important current constants are defined in [`fxtcombine/config.py`](../fxtcombine/config.py):

- `FXT_POSITION_ERR90_ARCSEC = 8.6`
  - representative source-position accuracy at 90% confidence
  - used for target-to-catalog matching via `fxtregions`
- `SRC_EXTRACT_RADIUS = 60`
  - source extraction radius in arcsec
- `BKG_EXTRACT_INNER_RADIUS = 90`
  - background annulus inner radius in arcsec
- `BKG_EXTRACT_OUTER_RADIUS = 300`
  - background annulus outer radius in arcsec

The main user-facing pipeline controls are:

- `--module`
- `--datamode`
- `--grade`
- `--expr`
- `--disable-flare-screen`
- `--flare-energy-range`
- `--flare-binsize`
- `--flare-min-time-ratio`
- `--jobs`
- `--srcdet-scales`
- `--srcdet-background-sigma-grid`
- `--skip-existing`

## FAQ

1. Why does `fxtcombine` ask for energy ranges instead of channel ranges now?

   - Energy ranges are easier to interpret scientifically. Internally, `fxtcombine` converts each requested energy band to the corresponding PI/channel range using the FXT E2PI calibration table, and `xselect` still filters events in PI/channel space.

2. Does a range like `0.3:10.0` include both endpoints?

   - The requested energy interval is converted to the set of PI/channel bins whose calibrated energy intervals overlap that band. The final event filtering is then done on inclusive channel bounds inside `xselect`.

3. Why does only the first image energy band drive `fxtsrcdet` by default?

   - `fxtcombine` can generate many stacked images, but it still needs one default detection image, one default stacked EEF bundle, and one default stacked mask for the stacked source-detection step. The current convention is to use the first requested image energy range for that purpose. All requested image bands still get their own stacked counts, rate, and EEF products.

4. What exactly does `--mask-expfrac` do?

   - It defines the minimum acceptable exposure, relative to the maximum of the stacked exposure map, for a pixel to remain valid in `stack_mask.fits`. Pixels are also rejected if the stacked counts image or stacked exposure map has `NaN` or `Inf`. The same threshold is applied when building stacked per-band rate maps so low-exposure edge pixels do not dominate the visual appearance.

5. If different observations have slightly different WCS rotation, is the stacked image pixel scale wrong?

   - No. The stacked products are explicitly defined on the reference image grid, so the pixel scale of the stacked image is, by construction, the pixel scale of that reference image. Rotation differences between epochs do not make that inconsistent. The main tradeoff is not a wrong pixel scale, but the fidelity of the reprojection and binning scheme onto a single chosen grid.

6. Why is there both `stack_cts.fits` and `e00300_10000_stack_cts.fits`?

   - The first requested image energy band is written twice:
     - once with a generic legacy name such as `stack_cts.fits` or `stack_rate.fits`
     - once with its explicit band-labelled name such as `e00300_10000_stack_cts.fits`
   - This keeps backward compatibility while making all multi-band products explicit.

7. What does `fsaevt` do inside `fxtcombine`?

   - In `FF` mode, matching `fsaevt` is used first to build the flare-screening light curve and derive a screened GTI through `fxtbkgoptrate`. That same screened GTI is then reused for both the final cleaned `fsaevt` and the final cleaned `evt`. `fsaevt` also provides one per-OBSID instrumental-background spectrum with `fxtbkggen`, and those per-OBSID products are then stacked into `stack_instbkgpi.fits`.

8. Why does the `fsaevt` path use `DETX=3:382 DETY=3:382`?

   - That detector-region cut matches the official `fxtchain` implementation for the default cleaned FSA spectrum. The later module-dependent mapping from FSA to IMG instrumental background is handled inside `fxtbkggen` through its CALDB products, so `fxtcombine` does not apply an extra A/B-specific geometric correction itself.

## Current Limitations

- `fxtcombine` assumes all selected observations belong to the same astrophysical target.
- The stacked-image detection/region step is currently fixed to the `fxtsrcdet` + `fxtregions` workflow.
- The current region-building step uses fixed manual radii rather than the `auto` sizing mode in `fxtregions`.
- Spectral stacking relies on external `runXstack`; there is no local in-package replacement.
- The counts-image stacking step currently bins reprojected events onto the reference grid rather than performing a full flux-conserving image reprojection.

## Suggested Future Additions

- examples using a real multi-epoch test dataset
- documentation of expected FXTDAS external dependencies
- a diagram of the per-OBSID versus stacked-product directory tree
- a validation note comparing stacked extraction against single-epoch extraction
