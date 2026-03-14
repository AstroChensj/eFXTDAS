# `fxtcombine`

## What It Does

`fxtcombine` combines multiple EP-FXT observations of the same target into a stacked imaging product and a stacked X-ray spectrum.

The current workflow:

- scans multiple OBSID directories and selects matching event products
- runs a simplified per-OBSID preprocessing chain to create clean events, images, and exposure maps
- reprojects and stacks images and vignetted exposure maps onto a common reference frame
- runs `fxtsrcdet` on the stacked image
- runs `fxtregions` to generate source and background extraction regions
- re-enters each OBSID to extract source/background spectra and response products
- calls external `runXstack` to build the final stacked source spectrum, background spectrum, RMF, and ARF

This makes `fxtcombine` the top-level orchestration task in the current `eFXTDAS` toolkit.

## Basic Usage

### Command-Line Usage

```bash
fxtcombine /data/epfxt \
  --obsid-lst 02001234567,02001234568 \
  --ra 9.25937 \
  --dec 9.16681 \
  --image-energy-ranges "0.3:10.0,10.0:12.0" \
  --lightcurve-energy-ranges "0.3:10.0,10.0:12.0" \
  --out-dir combine_out
```

Reuse already existing intermediates:

```bash
fxtcombine /data/epfxt \
  --obsid-lst obsids.txt \
  --ra 9.25937 \
  --dec 9.16681 \
  --image-energy-ranges "0.3:10.0,10.0:12.0" \
  --lightcurve-energy-ranges "0.3:10.0,10.0:12.0" \
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
    module="a,b",
    datamode="ff",
    datatype="evt",
    image_energy_ranges="0.3:10.0,10.0:12.0",
    lightcurve_energy_ranges="0.3:10.0,10.0:12.0",
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
    - root directory for all per-OBSID and stacked products
- required target inputs:
  - `--ra`
  - `--dec`
- optional event-selection inputs:
  - `--module`
  - `--datamode`
  - `--datatype`
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
- workflow controls:
  - `--skip-existing`
  - `--log-level`
  - `--log-file`

### Outputs

`fxtcombine` writes a directory tree under `out_dir` that contains:

- per-OBSID intermediate products under `<out-dir>/<OBSID>/products/`
- stacked imaging / region / spectral products under `<out-dir>/stack/`
- summary files such as `all_obsid.json` and `all_obsid.filelist`
- main and per-step log files

The full output layout, including the relationship between `src_dir`,
`obsid_lst`, and `out_dir`, is shown in the next section:

- [Output Data Structure](#output-data-structure)

## Output Data Structure

The most useful way to understand `fxtcombine` outputs is as a directory tree
linked to the three main path-like inputs:

- `src_dir`: input archive with one subdirectory per OBSID
- `obsid_lst`: which OBSID subdirectories under `src_dir` are actually used
- `out_dir`: where all combined products are written

### Input/Output Relationship

If the user runs:

```bash
fxtcombine <src_dir> \
  --obsid-lst 11900458112,11900465408 \
  --out-dir <out_dir> \
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
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.coord
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.particle
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.badpix
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.grade
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.gti
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_cl.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>.expo
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e00300_10000.eef
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e01000_03000.eef
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e00300_10000.img
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e01000_03000.img
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e00100_12000.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_e01000_03000.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_src.fits
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_src.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_bkg.pi
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_src.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_bkg.lc
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_src.arf
|   |   |-- fxt_<module>_<obsid>_<mode>_<filter>_<pp>_<datatype>_<ver>_src.rmf
|   |   |-- log/
|   |   |   |-- fxtchain.log
|   |   |   |-- <step-specific logs>
|-- 11900465408/
|   |-- products/
|   |   |-- ... same product pattern as above ...
|-- stack/
|   |-- stack_exp.fits
|   |-- stack_cts.fits
|   |-- stack_rate.fits
|   |-- stack_eef.fits
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
|   |-- fxteefmap.log
|   |-- srcdet.log
|   |-- fxtregions.log
|   |-- stack_pi.fits
|   |-- stack_bkgpi.fits
|   |-- stack_rmf.fits
|   |-- stack_arf.fits
|   |-- stack_runXstack.log
|-- all_obsid.filelist
|-- all_obsid.json
```

### Meaning of the Tree

- `src_dir/<obsid>/`
  - original input archive for each observation
- `out_dir/<obsid>/products/`
  - all per-OBSID intermediate and extracted products
- `out_dir/stack/`
  - products derived from combining all valid OBSIDs together
- `out_dir/all_obsid.json`
  - machine-readable summary of the per-OBSID product paths

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

### The Summary JSON Structure

`all_obsid.json` stores the same information as a nested dictionary:

```text
all_obsid.json
|-- <obsid>
|   |-- <datatype>
|   |   |-- <evt_fname_prefix>
|   |   |   |-- clevt
|   |   |   |-- image
|   |   |   |-- images
|   |   |   |   |-- e00300_10000
|   |   |   |   |-- e01000_03000
|   |   |   |-- image_band_channels
|   |   |   |   |-- e00300_10000
|   |   |   |   |-- e01000_03000
|   |   |   |-- vexpmap
|   |   |   |-- eefmap
|   |   |   |-- eefmaps
|   |   |   |   |-- e00300_10000
|   |   |   |   |-- e01000_03000
|   |   |   |-- exp
|   |   |   |-- alllc
|   |   |   |-- lightcurves
|   |   |   |   |-- e00100_12000
|   |   |   |   |-- e01000_03000
|   |   |   |-- lightcurve_band_channels
|   |   |   |   |-- e00100_12000
|   |   |   |   |-- e01000_03000
|   |   |   |-- srcpi
|   |   |   |-- bkgpi
|   |   |   |-- srclc
|   |   |   |-- bkglc
|   |   |   |-- arf
|   |   |   |-- rmf
```

Important conventions:

- `image` is always the first requested image band
- `images` stores all requested image bands
- `image_band_channels` stores the per-module PI/channel ranges derived from the requested image energy bands
- `alllc` is always the first requested light-curve band
- `lightcurves` stores all requested light-curve bands
- `lightcurve_band_channels` stores the per-module PI/channel ranges derived from the requested light-curve energy bands
- `srcpi`, `bkgpi`, `arf`, and `rmf` appear only after Stage 4

## Detailed Algorithm and How It Works

### 1. Per-OBSID Preprocessing

For each selected OBSID and each selected event file:

1. run `fxtcoord`
2. run `fxtpical`
3. run `fxtparticleidentify`
4. run `fxtbadpix`
5. run `fxtgrade`
6. run `fxtgtigen`
7. run `xselect` to produce:
   - clean events
   - one image per requested image energy band
   - one whole-field light curve per requested light-curve energy band
8. run `fxtexpogen` to produce a vignetted exposure map
9. run `fxteefmap` on each requested image energy band to produce one per-OBSID EEF bundle per band

This stage creates the per-OBSID products that are later stacked.

### 2. Image and Exposure Stacking

The longest-exposure image is used as the reference frame.

Current stacking behavior:

- each clean event list is projected onto the reference WCS and binned into the reference image grid
- each exposure map is reprojected onto the same reference frame and summed
- a stacked rate map is computed as `counts / exposure`

The stacked outputs are written separately for each requested datatype such as `evt` or `fsaevt`.

### 3. Detection and Region Building

After the stacked `evt` products are written:

1. `fxtcombine` uses the first requested stacked image band and the corresponding stacked EEF bundle
2. `fxtcombine` calls `fxtsrcdet`
3. `fxtsrcdet` writes the stacked source catalog and source region file
4. `fxtcombine` calls `fxtregions`
5. `fxtregions` writes the source and background extraction region files

The current `fxtcombine` usage of `fxtregions` is in `manual` mode, with fixed source and background radii:

- source radius: `60 arcsec`
- background annulus: `90-300 arcsec`

Target-to-catalog matching currently uses the EP-FXT representative position accuracy at 90% confidence:

- `FXT_POSITION_ERR90_ARCSEC = 8.6`

### 4. Per-OBSID Spectral Extraction

After the stacked source/background regions are created, `fxtcombine` loops over each OBSID again and:

- extracts source-filtered events
- extracts source and background PI spectra
- extracts source and background light curves
- runs `fxtarfgen`
- runs `fxtrmfgen`
- updates OGIP header keywords so the products are internally linked

### 5. Spectral Stacking

Finally, `fxtcombine` prepares a file list of all extracted source spectra and calls:

- `runXstack`

with same-target mode enabled.

This stage produces the final stacked source spectrum and associated response products.

## Logging and Output Layout

The default main log file is:

- `<out-dir>/log/fxtcombine.log`

Per-OBSID logs are written under:

- `<out-dir>/<OBSID>/products/log/`

Command-specific logs such as `xselect.log`, `fxtarfgen.log`, and `runXstack.log` are written beside the stage that generated them.

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
- `--datatype`
- `--grade`
- `--expr`
- `--skip-existing`

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
