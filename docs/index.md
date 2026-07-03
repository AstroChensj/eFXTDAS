# eFXTDAS Documentation

`eFXTDAS` is a small toolkit of analysis utilities extending the official FXTDAS workflow.

This documentation is organized by task:

```{toctree}
:maxdepth: 2
:caption: Tasks

fxtcombine
fxtbkgoptrate
fxtsrcdet
fxtregions
fxtrspgen
fxtpsfgen
fxtsensmap
```

```{toctree}
:maxdepth: 1
:caption: Project

acknowledgements
```

```{toctree}
:maxdepth: 1
:caption: Legacy/Internal
:hidden:

fxteefmap
```

## Scope

The current toolkit contains seven user-facing tasks:

- `fxtcombine`: multi-epoch stacking and spectral combination
- `fxtbkgoptrate`: optimum-threshold flare/background screening on one light curve
- `fxtsrcdet`: wavelet-style source detection and catalog construction
- `fxtregions`: source/background region construction for downstream analysis
- `fxtrspgen`: standalone ARF/RMF generation from external DS9 source regions
- `fxtpsfgen`: observation and stacked PSF-product generation
- `fxtsensmap`: aperture-mode flux-limit map generation from background, exposure, and PSF products

## Notes for Further Expansion

These pages are intended as a living template. As the code evolves, the most useful additions will be:

- concrete examples using the test dataset under `test/`
- algorithm diagrams for grouped fitting and background construction
- references to relevant CIAO / eSASS / FXTDAS concepts where the implementation intentionally approximates mission tools
