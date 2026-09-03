# /update-docs

Run this command after any change to the public API (new classes/methods, changed signatures, renamed modules, updated docstrings) or to the documentation sources.

## 1. Rebuild the documentation

```bash
cd docs
make clean
make html
```

Alternative methods:
```bash
python scripts/build_docs.py                 # from the repository root, cleans first
sphinx-build -b html docs docs/_build/html   # manual
```

The API pages are hand-maintained `automodule` stubs in `docs/api/`; do **not** run `sphinx-apidoc`, it generates a parallel set of pages with different names.

## 2. Verify the build

1. Build must complete with **no errors and no warnings**
2. Open `docs/_build/html/index.html` in browser (`open docs/_build/html/index.html` on Mac)
3. Navigate to the API Reference section — verify all public modules, classes, and methods appear
4. Confirm docstrings render correctly (sections, code blocks, math equations)
5. Confirm private methods (`_`-prefixed) do **not** appear (`"private-members": False` in `docs/conf.py`)
6. If a narrative page changed, check its cross-links resolve (Sphinx warns on broken ones)

**Common issues:**
- Module missing from API: add `docs/api/<module>.rst` and list it in `docs/api/index.rst`
- Private methods appearing: verify `"private-members": False` in `docs/conf.py` autodoc settings
- Duplicate object description warnings: `napoleon_use_ivar = True` must stay set in `conf.py`
- Docutils "substitution reference" errors: a docstring contains `|...|`; use `‖...‖` or `abs(...)` instead
- "Unexpected indentation" errors: ASCII art or matrices in a docstring need a `::` literal block
- Math not rendering: use `` :math:`M_j = I + \sum c_k N_{k,j}` `` in docstrings, `$...$` in Markdown pages
- Code examples unformatted: check indentation in docstring examples

## 3. API updates checklist

- [ ] Google-style docstrings updated: correct `Args:`, `Returns:`, `Raises:`, `References:` sections
- [ ] Type hints added/updated on all parameters and return values
- [ ] `Example:` added for complex methods
- [ ] Build completed without errors or warnings
- [ ] HTML renders correctly in browser
- [ ] API reference shows new/changed methods correctly
- [ ] Private methods absent from API docs
- [ ] Narrative pages that mention the changed API updated (`grep -r <name> docs/`)
- [ ] Code and documentation changes committed together

## 4. Documentation standards

### Public API only
- Public classes, methods, and functions: **must** have full Google-style docstrings
- Private methods (`_`-prefixed): docstrings and type hints for developers, but excluded from Sphinx output

### Required docstring components
All public API elements must include:
1. **Summary** — one-line description ending with a period
2. **Extended description** — algorithm context, paper background (optional)
3. **Args** — all parameters with descriptions; use `name: description` format
4. **Returns** — return type and description
5. **Raises** — all exceptions that may be raised (if applicable)
6. **Example** — usage examples for complex methods (encouraged)
7. **Note/Warning** — important caveats (when applicable)
8. **References** — paper section and equation numbers for all math implementations

### Type hints required
```python
# Good
def load_model(path: Path, normalize: bool = True) -> BlendshapeModelData:

# Bad
def load_model(path, normalize=True):
```

### Documentation structure
```
docs/
├── conf.py                    Sphinx configuration (MyST + autodoc)
├── README.md                  layout and build notes (not part of the site)
├── index.md                   landing page and toctrees
├── getting_started/           installation.md, quickstart.md
├── concepts/                  overview, blendshapes_to_skinning, how_the_solver_works, data_formats
├── user_guide/                preparing_data, compressing, evaluating_results,
│                              pipeline_integration, maya_rig_workflow, troubleshooting
├── api/                       index.rst + one automodule .rst per module
├── developer/                 architecture.md, development.md
└── _build/                    generated HTML (gitignored)
```

Narrative pages are Markdown (MyST); API pages are reStructuredText. New narrative pages go in the matching folder and in the toctree in `docs/index.md`.

### Automatic deployment
Documentation is deployed to GitHub Pages on merge to `main` via `.github/workflows/docs.yml`.
