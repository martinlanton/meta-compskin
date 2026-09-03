"""Sphinx configuration file for meta-compskin documentation.

Narrative pages are written in Markdown (MyST); the API reference under
``api/`` is written in reStructuredText because it is made of autodoc
directives. See ``docs/README.md`` for the layout and build instructions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / ".." / "src").resolve()))

from metacompskin import __version__  # noqa: E402

project = "metacompskin"
copyright = "2024, Meta Platforms, Inc."
author = "Meta Platforms, Inc."
version = __version__
release = __version__

extensions = [
    "myst_parser",  # Markdown narrative pages
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # Source code links
    "sphinx.ext.mathjax",  # LaTeX math support
    "sphinx.ext.intersphinx",  # Links to other docs
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",  # ::: fences for admonitions inside Markdown
    "deflist",  # definition lists
    "dollarmath",  # $...$ and $$...$$ maths
    "fieldlist",
]
myst_heading_anchors = 3

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
# Render "Attributes:" as :ivar: fields; otherwise dataclass fields are
# documented twice (once by napoleon, once by autodoc's undoc-members).
napoleon_use_ivar = True

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
    "private-members": False,  # Don't document private/internal methods
    "special-members": False,  # Don't document special methods like __init__
}

# Files that are not part of the rendered site.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
