# /pre-commit-checks

Run this command before committing. Execute all steps in order and fix any failures before proceeding.

## 1. Format, Lint, Type Check

```bash
ruff format .                        # format all files
ruff check --fix .                   # lint + auto-fix
ruff check .                         # verify no remaining issues
mypy src/ --ignore-missing-imports   # type checking
```

## 2. Tests

```bash
pytest                               # all tests
```

Do **not** run `test_default_output` with full iterations unless the code is PR-ready (slow). Use the short-iteration test during development:
```bash
pytest tests/test_default_output.py::TestCompskin::test_default_output_short_iter -v
```

## 3. Code Review Checklist

### Clean Code
- [ ] Names are intention-revealing, pronounceable, searchable; no noise words
- [ ] Functions are 5–20 lines, do one thing, one abstraction level
- [ ] No more than 3 function arguments
- [ ] No side effects; Command Query Separation respected
- [ ] No magic numbers — use named constants
- [ ] Comments explain **why**, not what; no commented-out code; no redundant comments
- [ ] Error handling uses exceptions with context; no `None` returned/passed for errors
- [ ] Appropriate abstraction — no premature abstractions, no clever tricks

### Design
- [ ] SOLID principles followed
- [ ] DRY (Don't Repeat Yourself) — no duplicated logic
- [ ] YAGNI (You Aren't Gonna Need It) — no speculative abstractions
- [ ] Composition favored over inheritance
- [ ] Interfaces/protocols used over concrete dependencies where it enables testing

### Testing
- [ ] All tests pass
- [ ] New code has tests written before implementation (TDD)
- [ ] Tests follow Arrange-Act-Assert structure
- [ ] Test names describe what is being tested
- [ ] Edge cases covered
- [ ] Minimum 80% coverage for new code
- [ ] Tests are independent (no shared mutable state between tests)

### Documentation
- [ ] All public classes, methods, functions have Google-style docstrings
- [ ] All docstring sections present: `Args:`, `Returns:`, `Raises:` (if applicable), `References:` (for math)
- [ ] Type hints present and complete on all public signatures
- [ ] `Example:` provided for complex methods
- [ ] Paper section/equation referenced for all mathematical implementations
- [ ] Run `/update-docs` if any public API was modified

### Code Quality (Ruff)
- [ ] `ruff format --check .` — no formatting issues
- [ ] `ruff check .` — no linting errors
- [ ] Imports properly sorted (standard library → third-party → local)
- [ ] No unused imports or variables
- [ ] Docstrings follow Google convention (enforced by `pydocstyle`)

### Scientific Computing / Performance
- [ ] Vectorized operations used (95%+ via PyTorch/NumPy, no explicit loops)
- [ ] Device placement explicit (no implicit CPU/GPU assumptions)
- [ ] Tensor shapes annotated in docstrings and inline (`# shape: (S, N, 3)`)
- [ ] Memory usage considered for large models
- [ ] Numerical edge cases handled (division by zero, log of zero, etc.)

### Python Style
- [ ] PEP 8 compliant
- [ ] f-strings used for string formatting
- [ ] `@dataclass` used for data containers
- [ ] `@property` used for getters; no explicit get/set pairs
- [ ] No mutable default arguments

## 4. Pre-commit Hooks (optional)

Install pre-commit to automate format and lint checks on every commit:
```bash
pip install pre-commit
pre-commit install
```

Configuration in `.pre-commit-config.yaml` runs `ruff` and `ruff-format` automatically.