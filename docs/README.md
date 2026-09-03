# Documentation sources

This folder is the Sphinx source tree for the metacompskin documentation. The
rendered site is published to GitHub Pages by `.github/workflows/docs.yml` on
every push to `main`.

## Layout

| Folder | Format | Audience |
|--------|--------|----------|
| `index.md` | Markdown | Landing page with reading paths for riggers and for TDs / engineers. |
| `getting_started/` | Markdown | Installation and a first end-to-end run. |
| `concepts/` | Markdown | What the method does and why, from a plain-language overview down to the equations, plus the file formats. |
| `user_guide/` | Markdown | Task-oriented pages: preparing data, compressing, evaluating, integrating in a pipeline, Maya rig workflow, troubleshooting. |
| `api/` | reStructuredText | Auto-generated API reference (`autodoc` directives, one page per module). |
| `developer/` | Markdown | Architecture and development workflow for contributors. |

Two formats are used on purpose: narrative pages are Markdown (rendered by
MyST, readable on GitHub as-is), and the API reference is reStructuredText
because it consists of Sphinx `automodule` directives.

## Building

```bash
pip install -e ".[dev]"     # Sphinx, the RTD theme and MyST are in the dev extra
cd docs
make html                   # output in docs/_build/html/index.html
```

Or from the repository root: `python scripts/build_docs.py`.

Warnings are treated as things to fix: the site should build with zero
warnings. Autodoc imports the package, so a broken import in `src/` also
breaks the docs build.

## Adding a page

1. Create the Markdown file in the matching folder.
2. Add it to the `toctree` in `index.md`.
3. Cross-reference other pages with standard Markdown links to the `.md`
   file, for example `[Quick start](../getting_started/quickstart.md)`.
   MyST rewrites them to the rendered pages and GitHub follows them too.

## Adding a module to the API reference

Create `api/<module>.rst` following the existing files and add it to the
toctree in `api/index.rst`. Only public members with Google-style docstrings
are rendered; `_`-prefixed members are excluded by configuration.
