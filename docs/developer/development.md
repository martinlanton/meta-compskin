# Development

How to set up, test, lint, document and contribute. The project conventions in
full are in `CLAUDE.md` at the repository root; this page is the short version.

## Setup

```bash
git clone https://github.com/martinlanton/meta-compskin.git
cd meta-compskin
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Install a CUDA build of PyTorch first if you have a GPU
([Installation](../getting_started/installation.md#gpu-support)); the
regression tests at 10 000 iterations are impractical on CPU.

## Everyday commands

```bash
pytest                                   # fast tests (coverage report in htmlcov/)
pytest -k short_iter -v                  # the 600-iteration regression only
pytest tests/test_default_output.py -v   # includes the slow 10 000-iteration test

ruff format .                            # format
ruff check --fix .                       # lint and auto-fix
mypy src/ --ignore-missing-imports       # type check

cd docs && make html                     # build the documentation
python scripts/build_docs.py             # same, from the root, with a clean first
```

Before opening a pull request: tests green, `ruff format` and `ruff check`
clean, `mypy` clean, docs building without warnings if you touched public
docstrings or the docs.

## Conventions

- **Tests first.** Write the failing test, then the minimal code, then
  refactor. Tests are Arrange-Act-Assert and named for what they verify.
- **Google-style docstrings** on every public class, method and function.
  Ruff's `pydocstyle` rules enforce the format; Sphinx renders them, so
  `Args:`, `Returns:`, `Raises:` and `References:` (paper section and equation)
  are expected, and maths uses `` :math:`...` ``.
- **Tensor shapes in comments**, using the $N, S, P, F, K$ letters.
- **Small functions**, explicit device placement, fixed seeds, vectorised
  operations.
- **Commit messages** follow Conventional Commits:
  `feat(model_fit): ...`, `fix(maya_exporter): ...`, `docs: ...`.
- **Branches**: `feature/`, `bugfix/`, `refactor/`, `docs/` plus a short
  description.

## Regression data

The bit-exact tests compare against files under `tests/test_data/`:

- `source_models/`: the four sample heads and a test animation.
- `windows/`, `macos/`: expected compressed output and expected vertex
  positions, generated on that platform. `SETUP.md` in each folder records the
  environment.

## End-to-end tests

`tests/test_end_to_end_compression.py` and `tests/test_end_to_end_maya.py`
run the whole workflow on the `proteus` head (23 735 vertices, 287 shapes):

- **Compression** (any platform): a short run, checked against error bounds
  rather than exact values.
- **Maya** (skipped without mayapy): the head is built with a blendShape node,
  `compress_and_build_rig` exports, compresses and rigs it, and the scene is
  saved to `output/end_to_end/proteus_rig.mb`. In that scene the original
  head, offset in X, plays each blendshape on the same frame as the rig, so
  you can scrub the timeline and compare the two.

Both default to 600 iterations per phase; set `COMPSKIN_E2E_ITERATIONS` for a
longer, cleaner compression. The modules get the head only through the
`head_model` and `head_shape_names` fixtures in `conftest.py`, so they run
unchanged on another model.

If a deliberate change to the solver alters the output, regenerate the
expected files on the same platform they were created on, and say so in the
commit message. Do not loosen tolerances to make a regression pass.

## Documentation

- Narrative pages: Markdown under `docs/`, one folder per section. Add new
  pages to the toctree in `docs/index.md`.
- API pages: reStructuredText under `docs/api/`, one per module. Add new
  modules to `docs/api/index.rst`.
- The build must be warning-free. Autodoc imports the package, so docstring
  syntax errors show up as Sphinx warnings pointing at the source line.
- `README.md` at the root is a landing page and should stay short; put detail
  here.

## Contributing

Pull requests are welcome. Fork, branch from `main`, add tests for new
behaviour, update the docs for API changes, and make sure the checks above
pass. Contributions require Meta's Contributor License Agreement; see
`CONTRIBUTING.md`. Bugs go to GitHub issues with a reproducible description.
