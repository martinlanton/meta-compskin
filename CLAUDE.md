# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Compressed Skinning for Facial Blendshapes** — a Meta Reality Labs Research implementation of the SIGGRAPH 2024 paper (`paper/compressed_skinning_for_facial_blendshapes.md`). Converts dense facial blendshape animation data into a sparse linear blend skinning (LBS) representation, achieving ~90% sparsity with 5-7× memory savings and 2-3× speedup over Dem Bones. Targets low-spec mobile platforms (e.g., Snapdragon 652).

## Commands

```bash
# Install
pip install -e ".[dev]"        # development install
pip install -e ".[viz]"        # add visualization support (matplotlib)

# Test
pytest                         # all tests
pytest tests/test_default_output.py::TestCompskin::test_default_output_short_iter -v  # single test

# Lint & format
ruff format .                  # format
ruff check .                   # lint
ruff check --fix .             # auto-fix lint issues
mypy src/ --ignore-missing-imports  # type checking

# Docs (API pages are hand-maintained in docs/api/; do not run sphinx-apidoc)
cd docs && make html
```

## Architecture

**Source layout:** `src/metacompskin/`

### Core Pipeline

**`model_data.py` — `BlendshapeModelData`**
Frozen dataclass. Loads `.npz` files containing blendshape deltas, rest vertices/faces, and rig logic metadata. Validates all array shapes on construction. The `alpha` regularization parameter is model-specific (see `constants.py`).

**`model_fit.py` — `SkinCompressor`**
The optimization engine. Decomposes the blendshape delta matrix **A** (shape `3S×N`, where S=blendshapes, N=vertices) into **B·C**:
- **B** (`3S×4P`): sparse transformation matrices (~90% zeros), P≈40 bones
- **C** (`4P×N`): sparse LBS weights (K≈8 non-zero influences per vertex)

Two-phase training using proximal gradient descent + Adam over ~20k iterations. Laplacian regularization enforces spatial smoothness. CUDA support gives ~60× speedup over CPU.

**`animation_generator.py` — `AnimationFrameGenerator`**
Runtime frame synthesis from compressed output. Applies rig logic (72 controls → S blendshapes), evaluates skinning transforms via **M_j = I + Σ c_k N_k,j** (Equation 7 from the paper), and writes OBJ files.

**`rig/riglogic.py`**
Rig control evaluation: maps input animation controls to blendshape activation weights, handling inbetween and corrective shapes.

**`maya_loader.py`**
Loads blendshape data directly from Maya rig files.

### Data Flow

```
Input .npz (blendshapes)
  → BlendshapeModelData   (load + validate)
  → SkinCompressor        (optimize, ~20k iterations)
  → Output .npz           (compressed weights B, C + metadata)
  → AnimationFrameGenerator (runtime frame synthesis)
  → OBJ files             (animation sequence)
```

### Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `P` | 40 | Number of virtual bones |
| `K` | 8 | Max LBS influences per vertex |
| `L` | 6000 | Non-zeros in transformation matrices |
| `alpha` | model-specific | Laplacian regularization strength |

## Testing

Tests live in `tests/test_default_output.py`. They load pre-computed expected outputs from `tests/test_data/` and compare against fresh compression runs. The short-iteration test (`600` iterations) is fast; the full test (`10000` iterations) is slow and should only be run when the code is PR-ready. Both compare all keys in the output NPZ against stored expected values.

### TDD (Test-Driven Development)

Follow the **Three Laws** strictly:
1. Write no production code until you have a failing test
2. Write no more test code than sufficient to fail
3. Write no more production code than sufficient to pass

**Cycle**: Red → Green → Refactor. Always show the failing test first, then minimal passing code, then cleanup.

**FIRST principles**: Tests must be **F**ast, **I**ndependent, **R**epeatable, **S**elf-validating, **T**imely (written before production code).

**Structure**: Arrange-Act-Assert, every test.
```python
def test_weight_normalization():
    weights = torch.tensor([0.3, 0.5, 0.2])       # Arrange
    normalized = normalize_weights(weights)         # Act
    assert torch.allclose(normalized.sum(), torch.tensor(1.0))  # Assert
```

Test names must describe what is being tested. Tests should target the highest-level API possible. Never disable a failing test — fix it.

## Tooling

### Ruff

Ruff is the formatter and linter (replaces black, flake8, isort). Configuration in `pyproject.toml` under `[tool.ruff]`. Run `ruff format .` then `ruff check --fix .` before committing.

### Sphinx

API docs use `sphinxcontrib-napoleon` (Google-style docstrings), `sphinx-rtd-theme`, and `myst-parser`. Configuration in `docs/conf.py`. Rebuild when modifying public API, method signatures, or docstrings. View output at `docs/_build/html/index.html`.

## Code Quality Principles

### Core Rules
- **Incremental progress**: small changes that compile and pass tests
- **Clear intent over clever code**: be boring and obvious; if you need to explain it, simplify it
- **Never** disable or skip failing tests — fix them
- Stop after 3 failed attempts and reassess

### Naming
- Intention-revealing, pronounceable, searchable names
- Nouns for classes (`SkinCompressor`), verbs for methods (`compute_skinning_weights`)
- No single-letter names except loop variables in short methods; no noise words (`data` vs `info`)

### Functions
- Small (5–20 lines), do one thing, one level of abstraction per function
- ≤2 arguments preferred, avoid 3+; no side effects
- Command Query Separation: functions either DO something or ANSWER something, not both
- Extract till you drop: if a block needs a comment to explain it, extract it into a named function

### Comments
- Prefer self-documenting code; comments explain **why**, not what
- Good: intent, non-obvious constraints, warnings, TODOs, docstrings
- Bad: redundant descriptions of what the code does, commented-out code, noise

### Error Handling
- Use exceptions, not return codes; provide context in exception messages
- Never return `None` for error cases; never pass `None` as an argument

### Python Style
- Type hints on all function signatures
- Google-style docstrings on all public classes/methods/functions (enforced by Ruff `pydocstyle`)
- f-strings for formatting; `@dataclass` for data containers; `@property` for getters
- Avoid mutable default arguments; prefer standard library over reinventing

### Scientific Computing
- Vectorize: avoid explicit loops, use PyTorch/NumPy tensor operations (95%+ of operations)
- Always explicit about device placement (CPU/GPU)
- Set random seeds for reproducibility
- Handle numerical edge cases (division by zero, log of zero, etc.)

### Tensor Shape Conventions

| Symbol | Meaning |
|--------|---------|
| `N` | Vertices |
| `S` | Blendshapes |
| `P` | Bones |
| `F` | Faces |
| `K` | Max influences per vertex |

Always annotate tensor shapes in docstrings and inline comments (e.g., `# shape: (S, N, 3)`).

## Documentation

### Google-Style Docstrings

**Required** on all public classes, methods, and functions. Sections in order:

1. Short summary (one line, ends with period)
2. Extended description (optional, separated by blank line)
3. `Args:` — format: `name: description` or `name (type): description`
4. `Returns:`
5. `Raises:` (if applicable)
6. `Attributes:` (classes)
7. `Note:` / `Warning:` (optional)
8. `Example:` (encouraged for complex methods)
9. `References:` — paper sections and equations for mathematical implementations

**Key format rules:**
- First line: short summary ending with a period
- Blank line between summary and extended description
- Section headers end with colon (`Args:`, `Returns:`, etc.)
- Section content indented 4 spaces
- Args format: `name: description` (type in hint, not repeated here unless clarifying)
- Full sentences with proper capitalization and periods throughout

Math equations use LaTeX notation: `` :math:`M_j = I + \sum c_k N_{k,j}` ``

### Mathematical Implementations

When implementing paper equations, document the full mathematical formulation, define all symbols, and reference the paper section. Use `torch.einsum` for readability:

```python
def equation_7_skinning_transforms(
    blend_weights: torch.Tensor,    # (S,) — c_k
    delta_transforms: torch.Tensor, # (S, P, 3, 4) — N_{k,j}
    n_shapes: int,
    n_bones: int,
) -> torch.Tensor:                  # (P, 3, 4) — M_j
    """Compute skinning transformations (Equation 7): M_j = I + Σ c_k * N_{k,j}.

    Args:
        blend_weights: Blendshape coefficients c_k, shape (S,).
        delta_transforms: Sparse transformation deltas N_{k,j}, shape (S, P, 3, 4).
        n_shapes: Number of blendshapes S.
        n_bones: Number of proxy bones P.

    Returns:
        Skinning transformations M_j, shape (P, 3, 4).

    References:
        Paper Section 3, Equation 7.
    """
    identity = torch.zeros(n_bones, 3, 4, device=blend_weights.device)
    identity[:, :, :3] = torch.eye(3, device=blend_weights.device)
    weighted_deltas = torch.einsum('s,spij->pij', blend_weights, delta_transforms)
    return identity + weighted_deltas
```

### API Docs (Sphinx)

```bash
cd docs && sphinx-apidoc -f -o api ../src/metacompskin && make html
# View: open docs/_build/html/index.html
```

Private methods (`_`-prefixed) must NOT appear in API docs (`"private-members": False` in `docs/conf.py`). Run `/update-docs` for the full rebuild and verification checklist.

## Git Workflow

### Commit Messages (Conventional Commits)

```
<type>(<scope>): <short summary>

<detailed description>

Refs: Section X, Equation Y
Closes #XXX
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

### Branch Naming

`feature/`, `bugfix/`, `hotfix/`, `refactor/`, `docs/` + short-description

## Performance Standards

Per Table 1 in the paper:
- **MAE** (Mean Absolute Error): < 0.05 mm
- **MXE** (Maximum Error): < 10 mm
- **Memory**: 5-7× reduction vs. dense methods
- **Speed**: 2-3× improvement vs. dense methods
- **Sparsity**: ~90% zeros in transformation matrices

## Agent Responsibilities

### Before Making Changes
- [ ] Review relevant paper sections and existing architecture
- [ ] Study current code patterns and identify affected components
- [ ] Plan changes before implementing; verify assumptions in code, not memory

### During Implementation
- [ ] Write tests first (TDD: Red → Green → Refactor)
- [ ] Show failing tests before implementation, then minimal code to pass, then refactor
- [ ] Keep functions small with meaningful names and Google-style docstrings
- [ ] Run `ruff format .` and `ruff check --fix .` regularly
- [ ] Ensure all existing tests pass at every step; ask for help if they block implementation

### After Implementation
Run `/pre-commit-checks` for the full checklist. Minimum steps:
- [ ] `pytest` — all tests pass
- [ ] `ruff format .` and `ruff check .` — clean
- [ ] `mypy src/ --ignore-missing-imports` — type checking clean
- [ ] Run `/update-docs` if any public API was modified