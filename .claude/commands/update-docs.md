# /update-docs

Run this command after any change to the public API (new classes/methods, changed signatures, renamed modules, updated docstrings).

## 1. Rebuild API Documentation

```bash
cd docs
sphinx-apidoc -f -o api ../src/metacompskin
make clean
make html
```

Alternative methods:
```bash
python scripts/build_docs.py          # if build script exists
sphinx-build -b html . _build/html    # manual
```

## 2. Verify the Build

1. Build must complete with **no errors and no warnings**
2. Open `docs/_build/html/index.html` in browser (`open docs/_build/html/index.html` on Mac)
3. Navigate to the API Reference section — verify all public modules, classes, and methods appear
4. Confirm docstrings render correctly (sections, code blocks, math equations)
5. Confirm private methods (`_`-prefixed) do **not** appear (`"private-members": False` in `docs/conf.py`)

**Common issues:**
- Module missing from API: check it's imported in `__init__.py` and listed in `docs/api/index.rst`
- Private methods appearing: verify `"private-members": False` in `docs/conf.py` autodoc settings
- Type hints not rendering: ensure `sphinx.ext.autodoc` and `napoleon` are enabled in `conf.py`
- Math not rendering: use `` :math:`M_j = I + \sum c_k N_{k,j}` `` with `sphinx.ext.mathjax`
- Code examples unformatted: check indentation in docstring examples

## 3. API Updates Checklist

- [ ] Google-style docstrings updated: correct `Args:`, `Returns:`, `Raises:`, `References:` sections
- [ ] Type hints added/updated on all parameters and return values
- [ ] `Example:` added for complex methods
- [ ] Build completed without errors or warnings
- [ ] HTML renders correctly in browser
- [ ] API reference shows new/changed methods correctly
- [ ] Private methods absent from API docs
- [ ] Code and documentation changes committed together

## 4. Documentation Standards

### Public API Only
- Public classes, methods, and functions: **must** have full Google-style docstrings
- Private methods (`_`-prefixed): docstrings and type hints for developers, but excluded from Sphinx output

### Required Docstring Components
All public API elements must include:
1. **Summary** — one-line description ending with a period
2. **Extended description** — algorithm context, paper background (optional)
3. **Args** — all parameters with descriptions; use `name: description` format
4. **Returns** — return type and description
5. **Raises** — all exceptions that may be raised (if applicable)
6. **Example** — usage examples for complex methods (encouraged)
7. **Note/Warning** — important caveats (when applicable)
8. **References** — paper section and equation numbers for all math implementations

### Type Hints Required
```python
# Good
def load_model(path: Path, normalize: bool = True) -> BlendshapeModelData:

# Bad
def load_model(path, normalize=True):
```

### Documentation Structure
```
docs/
├── conf.py                       # Sphinx configuration
├── index.rst                     # Main page
├── installation.rst
├── quickstart.rst
├── api/
│   ├── index.rst
│   ├── model_data.rst
│   ├── model_fit.rst
│   ├── animation_generator.rst
│   └── maya_loader.rst
├── guides/
└── _build/                       # Generated HTML (gitignored)
```

### Automatic Deployment
Documentation is deployed to GitHub Pages on merge to `main` via `.github/workflows/docs.yml`.
Enable in repository Settings → Pages → Source: `gh-pages` branch.