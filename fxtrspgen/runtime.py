"""Runtime helpers for legacy FXTDAS Python modules."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_fxtdas_py_path() -> Path:
    """Add the local ``fxtdas-py`` directory to ``sys.path``.

    Returns
    -------
    Path
        Absolute path to the inserted directory.
    """
    root = Path(__file__).resolve().parent.parent
    fxtdas_py = root / "fxtdas-py"
    path_str = str(fxtdas_py)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return fxtdas_py
