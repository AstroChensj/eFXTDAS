# eFXTDAS Documentation

`eFXTDAS` is a small toolkit of analysis utilities extending the official FXTDAS workflow.

This documentation is organized by task:

```{toctree}
:maxdepth: 2
:caption: Tasks

fxtsrcdet
fxteefmap
fxtregions
fxtcombine
```

## Scope

The current toolkit contains four user-facing tasks:

- `fxtsrcdet`: wavelet-style source detection and catalog construction
- `fxteefmap`: image-sized EEF-radius map generation
- `fxtregions`: source/background region construction for downstream analysis
- `fxtcombine`: multi-epoch stacking and spectral combination

## Notes for Further Expansion

These pages are intended as a living template. As the code evolves, the most useful additions will be:

- concrete examples using the test dataset under `test/`
- algorithm diagrams for grouped fitting and background construction
- references to relevant CIAO / eSASS / FXTDAS concepts where the implementation intentionally approximates mission tools
