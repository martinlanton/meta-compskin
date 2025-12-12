"""Sphinx configuration file for meta-compskin documentation."""

import sys
from pathlib import Path

sys.path.insert(0, Path("../src").resolve())

project = "metacompskin"
copyright = "2024, Meta Platforms, Inc."
author = "Meta Platforms, Inc."
version = "0.1.0"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # Source code links
    "sphinx.ext.mathjax",  # LaTeX math support (CRITICAL!)
    "sphinx.ext.intersphinx",  # Links to other docs
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# MathJax configuration for LaTeX
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Intersphinx links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"  # Better than alabaster for technical docs
html_static_path = ["_static"]
