"""Sphinx configuration for the project documentation."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

project = "Planar Instance Generation"
author = "Planar Instance Generation contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "description": "Reproducible near-planar network instance generation",
    "fixed_sidebar": True,
    "github_button": False,
}
