# eFXTDAS

`eFXTDAS` is a small analysis toolkit that extends the official [FXTDAS](https://epfxt.ihep.ac.cn/analysis) workflow for Einstein Probe FXT data.

The official FXTDAS tasks provide the core mission pipeline: event calibration, screening, coordinate correction, exposure-map generation, and standard low-level products. For science analysis, users still commonly need extra workflow glue: multi-OBSID stacking, source detection, source/background region construction, PSF-aware diagnostics, (correct) response generation, sensitivity maps, and compact QA figures. That is where `eFXTDAS` helps.

For normal multi-OBSID science analysis, start with `fxtcombine`: it is the top-level task that orchestrates screening, stacking, PSF products, source detection, region generation, response generation, sensitivity maps, and the final quick-view figure.

The lower-level command-line tools are exposed so individual stages can be inspected, tuned, or rerun manually.

## What You Get from `fxtcombine`

![eFXTDAS summary figure](docs/figs/readme_summary_2x3.png)

The six-panel quick-view figure summarizes the main stacked products from `fxtcombine`: smoothed stacked counts, stacked background map, target zoom with source/background extraction regions, stacked exposure map, stacked PSF R90 map, and stacked sensitivity map.

- The stacked counts image is labeled with detected sources (out to `75%` EEF radius). 
  - By default the catalog generated with `fxtsrcdet` keeps only sources with detection likelihood over `6` (as per [eROSITA simulation](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..78S/abstract), this roughly corresponds to a false detection rate of 14\%).
  - Note that we have grayed out the masked region with insufficient exposure near the image edge; sources in those regions are dropped. This is a conservative approach, and is because the EP-FXT *vignetting correction is not perfect*, so the rate near the edge will be erroneously high and thus leading to many false positives.
- The stacked background map is created after carving out wavelet-detected sources from the image. Per-pixel smoothing is adopted.
- The target zoom-in shows the target region (cyan), background region (crimson), and nearby contamination sources (white) to be carved out. This is inspired from `eSASS`-`srctool`.
- The mask map is defined so that invalid pixels are those with stacked exposure smaller than `30%` of maximum exposure, and quick-view panels gray out those invalid pixels where relevant.
- The stacked PSF product carries the local PSF/EEF support used by `fxtsrcdet` for source fitting and by downstream region/response workflows.
- The stacked sensitivity map shows the flux limit from `fxtsensmap` when that optional product is present.

`fxtcombine` also writes stacked spectra, responses, masks, source catalogs, regions, sensitivity maps, logs, and machine-readable summaries.

## Quick Start: Top-Level Workflow

Use `fxtcombine` for the full multi-OBSID workflow.

```bash
cd /path/to/your/observations
fxtcombine ./ \
  --obsid-lst obsids.txt \
  --ra 9.25937 \
  --dec 9.16681 \
  --image-energy-ranges "0.3:10.0,10.0:12.0" \
  --lightcurve-energy-ranges "0.1:12.0,10.0:12.0" \
  --jobs 4 \
  --out-dir combine_out \
  --stack-dir combine_stack
```

The source directory should contain one subdirectory per OBSID. `obsids.txt`
lists the OBSID directories to process:

```text
/path/to/your/observations/
|-- obsids.txt
|-- 02001234567/
|   |-- ... original FXT archive content ...
|-- 02001234568/
|   |-- ... original FXT archive content ...
|-- 02001234569/
|   |-- ... original FXT archive content ...
```

Example `obsids.txt`:

```text
02001234567
02001234568
02001234569
```

Key parameters:

- `/path/to/your/observations`: Source directory containing OBSID subdirectories.
- `--obsid-lst`: comma-separated OBSIDs or a file with one OBSID per line
- `--image-energy-ranges`: stacked detection defaults to the first image band
- `--lightcurve-energy-ranges`: whole-field diagnostic light curves
- `--disable-flare-screen`: disable the default `FF`-mode FSA-based flare screening
- `--flare-threshold-method`: FSA flare-threshold method, default `robust_iqr`
- `--jobs`: Stage-1 OBSID parallelism
- `--stack-dir`: where stacked science products go

Most useful outputs:

- `quickview.png`
- `stack_cts.fits`
- `stack_bkgmap.fits`
- `stack_mask.fits`
- `stack_psfprod.fits`
- `stack_sensmap.fits`
- `stack_src.fits` / `stack_src.reg`
- `target_src.reg` / `target_bkg.reg`
- `stack_pi.fits`, `stack_bkgpi.fits`, `stack_arf.fits`, `stack_rmf.fits`

See: [docs/fxtcombine.md](docs/fxtcombine.md)

For full information on the packages, please check the [docs](https://efxtdas.readthedocs.io/en/latest/).

## Installation and Prerequisites

`eFXTDAS` is not a replacement for official FXTDAS. It is a Python-layer extension around that environment.

You should assume:

- official FXTDAS / HEASoft command-line tools are already installed and runnable
- `eFXTDAS` is installed into the **same environment**
- `CALDB` is available for workflows that need mission calibration

Typical install:

```bash
cd ~  # or any other your favorite path
git clone https://github.com/AstroChensj/eFXTDAS.git
cd eFXTDAS
python -m pip install -e .
```

This installs the Python packages and CLI entry points:

- `fxtcombine`
- `fxtcombine-quickview`
- `fxtbkgoptrate`
- `fxtsrcdet`
- `fxtregions`
- `fxtrspgen`
- `fxtpsfgen`
- `fxtsensmap`

Because `fxtcombine` will stack spectra and responses from multiple observations, [Xstack](https://github.com/AstroChensj/Xstack) software is also needed:

```bash
cd ~
git clone git@github.com:AstroChensj/Xstack.git
cd Xstack
python -m pip install -e .
```

## Task Hierarchy

For normal analysis, start with `fxtcombine`. The other commands are exposed so individual stages can be inspected, tuned, or rerun manually.

| Workflow order | Task | Role in `fxtcombine` | Use directly when... |
| --- | --- | --- | --- |
| Top level | `fxtcombine` | orchestrates the full multi-OBSID workflow | this is the default starting point |
| 1 | `fxtbkgoptrate` | optimizes FSA flare/background screening in `FF` mode | tuning one light curve or flare threshold |
| 2 | `fxtpsfgen` | builds per-OBSID and stacked PSF products | checking or rebuilding PSF support products |
| 3 | `fxtsrcdet` | detects sources and builds the stacked background map | tuning source detection/background modeling |
| 4 | `fxtregions` | builds target source/background extraction regions | adjusting extraction geometry |
| 5 | `fxtrspgen` | builds per-OBSID ARF/RMF products for extracted spectra | manually generating responses |
| 6 | `fxtsensmap` | builds the final stacked sensitivity/flux-limit map | generating standalone sensitivity maps |
| 7 | `fxtcombine-quickview` | builds the final six-panel QA figure | regenerating the summary plot |

## How the Workflow Fits Together

For most science use cases, the intended sequence is:

1. Run `fxtcombine` on all OBSIDs of the same target field.
2. Inspect the stacked diagnostics:
   - `quickview.png`
   - `stack_cts.fits`
   - `stack_bkgmap.fits`
   - `stack_mask.fits`
   - `stack_psfprod.fits`
   - `stack_src.fits` / `stack_src.reg`
   - `stack_sensmap.fits`
   - `target_src.reg` / `target_bkg.reg`
   - `stack_pi.fits`, `stack_bkgpi.fits`, `stack_arf.fits`, `stack_rmf.fits`
3. In most cases the default settings should be enough. However, in case you find source detection or extraction regions need tuning, rerun the relevant lower-level task:
   - `fxtsrcdet` directly on the stacked counts image
   - `fxtregions` on an updated source catalog
   - manually extract source / bkg PI spectra with the updated region files, using `xselect` tool.
   - `fxtsensmap` when sensitivity-map assumptions need changing. It does not affect the final extracted spectra and responses, though.
4. Use the final stacked spectra, regions, sensitivity map, and response products for downstream science analysis.

Important current behavior:

- `fxtcombine` uses energy ranges in keV, not direct PI/channel ranges, for Stage-1 image and light-curve generation.
- In `FF` mode, `fxtcombine` automatically uses matching `fsaevt` when available to derive a flare-screened GTI, then reuses that screened GTI for both `fsaevt` and `evt`.
- In that `FF`/`fsaevt` flare-screening path, `fxtcombine` calls the installed `fxtbkgoptrate` CLI task through its normal task wrapper rather than importing the optimizer directly in-process.
- `fxtcombine` reads the persisted flare-screening summary from the `fxtbkgoptrate` diagnostic FITS headers, including `BGOPTCUT`, `FRACTLFT`, `OPTSTAT`, and `OPTMETH`.
- `fxtcombine` generates `stack_mask.fits` and `stack_psfprod.fits`, then passes both to `fxtsrcdet`.
- `fxtcombine` now generates per-OBSID spectral responses through `fxtrspgen`, using the DS9 source region directly for the ARF/RMF pair.

> [!IMPORTANT]
> The input parameters in `eFXTDAS` fall into two classes:
> - user-facing parameters that you set through the CLI or Python API
> - internal heuristic defaults that are not exposed directly
>
> Examples of the second class include the source-carving radius used in background-map creation, PSF-fitting radius, and some internal smoothing or threshold constants. These defaults were chosen heuristically and are intended to be reasonable starting points, but users should still try different values when tuning a workflow is needed.
>
> Internal defaults can be overridden globally through environment variables. For example:
> `export FXTSRCDET_BACKGROUND_CARVE_R90_FACTOR=1.5`
>
> Package-specific internal parameters are documented in:
> [docs/fxtcombine.md](docs/fxtcombine.md),
> [docs/fxtsrcdet.md](docs/fxtsrcdet.md),
> [docs/fxtregions.md](docs/fxtregions.md).


## Advanced: Run Individual Tasks

These lower-level tasks are normally called by `fxtcombine`, but they can also be run directly to inspect or tune individual stages.

### 1. `fxtbkgoptrate`

Use this when you want to optimize a flare/background threshold on one FITS light curve directly.

```bash
fxtbkgoptrate flare.lc \
  --method snr \
  --diag-out flare_diag.fits \
  --flare-gti-out flare.gti \
  --base-gti base.gti \
  --screened-gti-out screened.gti
```

Key parameters:

- `infile`: FITS light curve, usually with `RATE` or `COUNT`
- `--method`: threshold method, `snr` or `robust_iqr`; standalone default remains `snr`
- `--diag-out`: writes the threshold-trial table and chosen threshold metadata
- `--flare-gti-out`: writes the GTI built from accepted low-background bins
- `--base-gti` and `--screened-gti-out`: intersect the flare GTI with an existing GTI

Python API is available through `run_bkgoptrate`.

When `fxtcombine` uses `fxtbkgoptrate` internally, it invokes the same CLI entry point shown above with `--method robust_iqr` by default, then reads the persisted summary from the diagnostic FITS rather than relying on an in-memory Python return value.
`robust_iqr` uses the valid-bin rate distribution and sets the threshold to `Q3 + 1.5 * IQR`, with the usual retained-exposure floor still enforced by `--min-time-ratio`.

### 2. `fxtpsfgen`

Use this when you need observation or stacked PSF support products such as `stack_psfprod.fits`.

Per-observation PSF product:

```bash
fxtpsfgen build-obs evt_image.fits \
  --expmap evt_vexp.fits \
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

Key parameters:

- `build-obs`: build one per-observation PSF product from an image footprint and optional exposure map
- `stack`: combine multiple observation PSF products onto one stacked reference image
- `--obs-psf` and `--weightmap`: repeat once per stacked component and keep them aligned by order

See: [docs/fxtpsfgen.md](docs/fxtpsfgen.md)

### 3. `fxtsrcdet`

Use this for source detection and background-map generation on one image.

```bash
fxtsrcdet stack_cts.fits \
  --expmap stack_exp.fits \
  --mask stack_mask.fits \
  --psfprod stack_psfprod.fits \
  --mission ep-fxt \
  --emin 0.3 \
  --emax 10.0 \
  --out sources.fits \
  --regfile sources.reg \
  --save-bkgmap bkgmap.fits
```

Key parameters:

- `image`: counts image
- `--expmap`: exposure map
- `--mask`: global analysis-validity mask
- `--psfprod`: stacked or observation PSF product, usually from `fxtpsfgen`
- `--scales`: wavelet scales
- `--background-sigma-grid`: adaptive background smoothing grid
- `--save-bkgmap`: most useful intermediate diagnostic

Python API is available through `fxtsrcdet_pipeline`.

See: [docs/fxtsrcdet.md](docs/fxtsrcdet.md)

### 4. `fxtregions`

Use this after `fxtsrcdet` to build source and background extraction regions for one target.

```bash
fxtregions \
  stack_cts.fits \
  sources.fits \
  --bkgmap bkgmap.fits \
  --ra 9.25937 \
  --dec 9.16681 \
  --mission ep-fxt \
  --emin 0.3 \
  --emax 10.0 \
  --mode auto \
  --src-regfile source.reg \
  --bkg-regfile background.reg
```

Key parameters:

- `image` and `catalog`: stacked image plus `fxtsrcdet` catalog
- `--ra`, `--dec`: target sky position
- `--bkgmap`: recommended for better local background sampling
- `--mode`: `auto` or `manual`
- `--src-radius`, `--bkg-inner`, `--bkg-outer`: used in manual mode

Python users can call `fxtregions.pipeline.build_regions(...)`.

See: [docs/fxtregions.md](docs/fxtregions.md)

### 5. `fxtrspgen`

Use this when you need standalone ARF/RMF generation from a DS9 source region.

```bash
fxtrspgen source.pi source.expo source.reg \
  --arf-out source.arf \
  --rmf-out source.rmf \
  --update-pha
```

Key parameters:

- positional inputs: source PHA, exposure map, and DS9 source region
- `--arf-out`, `--rmf-out`: explicit output response names
- `--update-pha`: write `ANCRFILE` and `RESPFILE` into the source PHA

Python users can call `fxtrspgen.run_fxtrspgen(...)`.

See: [docs/fxtrspgen.md](docs/fxtrspgen.md)

### 6. `fxtsensmap`

Use this when you need a standalone flux-limit map from an existing background map, exposure map, and PSF product.

```bash
fxtsensmap \
  --bkgmap stack_bkgmap.fits \
  --expmap stack_expmap.fits \
  --mask stack_mask.fits \
  --psfprod stack_psfprod.fits \
  --eef 0.90 \
  --out stack_sensmap.fits
```

Key parameters:

- `--bkgmap`: expected background counts per pixel
- `--expmap`: exposure map in seconds
- `--mask`: optional analysis mask; non-zero finite pixels are valid
- `--psfprod` or `--psfmap`: local aperture-radius source
- `--likemin` or `--sigma`: false-alarm threshold definition
- `--ecf`: count-rate to flux conversion

See: [docs/fxtsensmap.md](docs/fxtsensmap.md)

### 7. `fxtcombine-quickview`

Use this when you want to regenerate the six-panel QA figure from an existing `fxtcombine` stack directory.

```bash
fxtcombine-quickview combine_stack \
  --out combine_stack/quickview.png \
  --title "Target stack"
```

Key parameters:

- `stack_dir`: directory containing the stacked `fxtcombine` products
- `--out`: output PNG path
- `--title`: optional figure title
- `--sensmap`: optional override for the sensitivity-map FITS path

See: [docs/fxtcombine.md](docs/fxtcombine.md)

## Repo Layout

The most relevant top-level directories are:

| Path | Meaning |
| --- | --- |
| `src/fxtcombine/` | multi-OBSID stacking workflow |
| `src/fxtsrcdet/` | source detection and catalog construction |
| `src/fxtregions/` | source/background region construction |
| `src/fxtpsfgen/` | observation and stacked PSF-product generation |
| `src/fxtsensmap/` | aperture-mode sensitivity-map generation |
| `src/fxtcaldb/` | shared calibration, optics, and PSF / EEF support code |
| `fxtdas-bin/` | local copies of official FXTDAS task scripts for inspection/reference |
| `fxtdas-py/` | local Python support code from the FXTDAS environment |
| `docs/` | detailed package documentation |
| `test*` | sample products, experiments, and debugging outputs |

## Notes for AI Assistants

If a user asks for help with this repo:

- start from `fxtcombine` when the request is about multi-OBSID science products
- start from `fxtbkgoptrate` when the request is about flare/background thresholding
- start from `fxtpsfgen` when the request is about PSF products or local PSF support
- start from `fxtsrcdet` when the request is about source counts maps, background maps, or source catalogs
- start from `fxtregions` when the request is about extraction-region geometry
- start from `fxtrspgen` when the request is about response generation
- start from `fxtsensmap` when the request is about flux-limit or sensitivity maps
- start from `fxtcombine-quickview` when the request is about the six-panel QA figure

Most useful diagnostics to inspect first:

- `quickview.png`
- `stack_cts.fits`
- `stack_bkgmap.fits`
- `stack_mask.fits`
- `stack_psfprod.fits`
- `stack_src.fits` / `stack_src.reg`
- `stack_sensmap.fits`
- `target_src.reg` / `target_bkg.reg`
- `stack_pi.fits`, `stack_bkgpi.fits`, `stack_arf.fits`, `stack_rmf.fits`

Detailed package references:

- [docs/fxtcombine.md](docs/fxtcombine.md)
- [docs/fxtbkgoptrate.md](docs/fxtbkgoptrate.md)
- [docs/fxtpsfgen.md](docs/fxtpsfgen.md)
- [docs/fxtsrcdet.md](docs/fxtsrcdet.md)
- [docs/fxtregions.md](docs/fxtregions.md)
- [docs/fxtrspgen.md](docs/fxtrspgen.md)
- [docs/fxtsensmap.md](docs/fxtsensmap.md)
