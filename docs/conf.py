"""Sphinx configuration for eFXTDAS."""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "eFXTDAS"
author = "Chen SJ"
copyright = "2026, Chen SJ"
release = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

if importlib.util.find_spec("sphinxcontrib.mermaid") is not None:
    extensions.append("sphinxcontrib.mermaid")

autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
numfig = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
