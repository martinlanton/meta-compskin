# Design — `iterations` per stage, seed control, experiment script (revision 3)

Read [spec.md](spec.md) first. Line numbers refer to commit `87dcd6d`.

## 1. Architecture overview

```
                 ┌──────────────────────────────────────────────────────────┐
                 │ SkinCompressor.__init__                                  │
   seed ───────► │  torch.manual_seed(seed)                                 │
   iterations ─► │  self.stage_iterations = _as_stage_iterations(iterations)│
                 │  self.schedule = build_training_schedule(stages, K, L)   │
                 └──────────────────────────┬───────────────────────────────┘
                                            │   (user may reassign self.schedule here)
                 ┌──────────────────────────▼───────────────────────────────┐
                 │ SkinCompressor.run(output)                               │
                 │  _check_schedule_fits_model()   (non-empty, K < P, L ≤ 6SP)
                 │  A, Laplacian, TR, B_rt ~ randn, W ~ randn (unchanged)   │
                 │  for index, phase in enumerate(self.schedule):           │
                 │      print(phase header)                                 │
                 │      self.train(B_rt, TR, A, W, phase)                   │
                 │  Wn, BX, B = final evaluation (unchanged)                │
                 │  self.reconstruction_error = ReconstructionError(...)    │
                 │  np.savez(...)                          (unchanged)      │
                 └──────────────────────────────────────────────────────────┘

   cli.py ──parse──► kwargs ──► SkinCompressor         (--iterations INT[,INT...], --seed)
   maya_pipeline.CompressionSettings ──compression_command──► cli flags
   scripts/compare_schedules.py ──► SkinCompressor × (variants × seeds) ──► results.csv + summary
```

Files that change: `model_fit.py`, `cli.py`, `maya_pipeline.py`, tests, docs,
`pyproject.toml` (pytest path), the new script.

## 2. Data model (`src/metacompskin/model_fit.py`)

Place both dataclasses above `class SkinCompressor`, after the module
constants. Add `_DEFAULT_SEED = 12345` next to `_DEFAULT_MAX_INFLUENCES`
(`model_fit.py:33-34`). Add
`from dataclasses import dataclass` and `from collections.abc import Sequence`
to the imports.

```python
@dataclass(frozen=True)
class TrainingPhase:
    """One stage of the optimisation with fixed sparsity budgets.

    Attributes:
        iterations: Adam steps in this phase.
        max_influences: K, weights kept per vertex by the projection.
        total_nnz_B_rt: L, delta coefficients kept globally by the projection.
        normalize_weights: Whether the forward pass divides each vertex's
            weights by their sum (partition of unity).

    Raises:
        ValueError: If any count is below one.

    References:
        Section 4 of the paper, projections (1)–(3); spec.md §7.
    """

    iterations: int
    max_influences: int
    total_nnz_B_rt: int  # noqa: N815 (matches SkinCompressor)
    normalize_weights: bool

    def __post_init__(self) -> None:
        for name in ("iterations", "max_influences", "total_nnz_B_rt"):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )


@dataclass(frozen=True)
class ReconstructionError:
    """Final fit error of a compression run, in model units.

    Attributes:
        max_abs: Largest per-axis absolute error over all vertices and shapes (MXE).
        mean_abs: Mean per-axis absolute error (MAE).

    References:
        Paper Table 1 ("MXE", "MAE"); docs/user_guide/evaluating_results.md.
    """

    max_abs: float
    mean_abs: float
```

## 3. Schedule construction

Public, pure (no torch, no model):

```python
def build_training_schedule(
    stage_iterations: Sequence[int],
    max_influences: int,
    total_nnz_B_rt: int,  # noqa: N803
) -> tuple[TrainingPhase, ...]:
    """Builds the phases run() executes, one stage per entry of stage_iterations.

    Stage i runs ``stage_iterations[i]`` steps and keeps
    ``max_influences * 2 ** (N - 1 - i)`` weights per vertex, N being the
    number of stages, so the budget halves each stage down to
    ``max_influences``. The delta budget is constant. The first stage is
    preceded by a warm-up of the same length with weight normalisation off;
    every stage itself runs normalised. One entry is the classic two-phase
    schedule.

    Args:
        stage_iterations: Steps per stage; non-empty.
        max_influences: K reached in the final stage.
        total_nnz_B_rt: L used in every phase.

    Returns:
        ``len(stage_iterations) + 1`` phases.

    Raises:
        ValueError: If ``stage_iterations`` is empty or an entry is below one.

    References:
        Zhu & Gupta 2017 (gradual magnitude pruning); research.md §4-5.
    """
    if not stage_iterations:
        raise ValueError("iterations must contain at least one stage")
    n_stages = len(stage_iterations)
    stages = [
        TrainingPhase(
            steps,
            max_influences * 2 ** (n_stages - 1 - i),
            total_nnz_B_rt,
            normalize_weights=True,
        )
        for i, steps in enumerate(stage_iterations)
    ]
    warm_up = TrainingPhase(
        stages[0].iterations,
        stages[0].max_influences,
        total_nnz_B_rt,
        normalize_weights=False,
    )
    return (warm_up, *stages)
```

A private helper normalises the constructor argument:

```python
def _as_stage_iterations(iterations: int | Sequence[int]) -> tuple[int, ...]:
    """(10000,) for an int, tuple(iterations) for a sequence.

    Raises:
        ValueError: If the sequence is empty (entries below one are rejected
            by TrainingPhase).
    """
```

Anything else (annealing L, a differently sized warm-up, non-geometric
steps) is done by constructing `TrainingPhase` objects directly and
assigning them to `compressor.schedule`.

## 4. `SkinCompressor` changes

### 4.1 Constructor

Signature after the change (keep the existing `# noqa: PLR0913, PLR0917`):

```python
def __init__(
    self,
    model_data: BlendshapeModelData,
    iterations: int | Sequence[int] = 10000,
    rest_joint_matrices: np.ndarray | list | None = None,
    number_of_bones: int | None = None,
    max_influences: int = _DEFAULT_MAX_INFLUENCES,
    total_nnz_B_rt: int = 6000,
    init_weight: float = 1e-3,
    power: int = 2,
    seed: int = _DEFAULT_SEED,
):
```

Body: everything up to and including `self.power = power`
(`model_fit.py:188-226`) is unchanged (`self.iterations = iterations` at
line 189 keeps the value as given). Then:

```python
self.stage_iterations = _as_stage_iterations(iterations)
self.schedule: Sequence[TrainingPhase] = build_training_schedule(
    self.stage_iterations, max_influences, total_nnz_B_rt
)

self.seed = seed
torch.manual_seed(self.seed)
```

(`self.seed = 12345` at line 228 becomes `self.seed = seed`; the
`manual_seed` call stays where it is.) After `self.loss_list` /
`self.abserr_list` (`model_fit.py:242-243`) add
`self.reconstruction_error: ReconstructionError | None = None`.

The schedule code is pure Python; it neither touches torch nor changes the
RNG stream.

Docstring: widen `iterations` in `Args:` ("int: steps for a single stage,
today's two-phase run; sequence: steps per stage, the influence budget
halving each stage from K·2^(N−1) down to K; the unnormalised warm-up
takes the first entry's length"); add `seed`; extend `Raises:`; add
`stage_iterations`, `schedule` (say it can be reassigned before `run()`),
`reconstruction_error` to the class `Attributes:` (`model_fit.py:89-112`;
`seed` is already listed — update its text). Add an `Example:` showing a
sequence and one showing a hand-built schedule:

```python
>>> compressor = SkinCompressor(model_data, iterations=(5000, 5000, 10000))
>>> compressor.schedule = (
...     TrainingPhase(2000, 32, 6000, normalize_weights=False),
...     TrainingPhase(8000, 8, 6000, normalize_weights=True),
... )
```

### 4.2 `run()`

Call `self._check_schedule_fits_model()` as the **first statement** of
`run()` so a bad schedule fails in milliseconds. Replace `model_fit.py:364-365`
with:

```python
for index, phase in enumerate(self.schedule, start=1):
    print(
        f"phase {index}/{len(self.schedule)}: iterations={phase.iterations} "
        f"K={phase.max_influences} L={phase.total_nnz_B_rt} "
        f"normalize_weights={phase.normalize_weights}"
    )
    self.train(B_rt=B_rt, TR=TR, A=A, W=W, phase=phase)
```

After `print(f"meanDelta {meanDelta}")` (`model_fit.py:382`):

```python
self.reconstruction_error = ReconstructionError(
    max_abs=float(maxDelta), mean_abs=float(meanDelta)
)
```

```python
def _check_schedule_fits_model(self) -> None:
    """Rejects a schedule the projections cannot apply to this model.

    Raises:
        ValueError: If the schedule is empty, or a phase keeps K >= P weights
            per vertex, or more than 6·S·P delta coefficients.
    """
    if not self.schedule:
        raise ValueError("schedule must contain at least one TrainingPhase")
    n_coefficients = 6 * self.model_data.n_blendshapes * self.number_of_bones
    for index, phase in enumerate(self.schedule, start=1):
        if phase.max_influences >= self.number_of_bones:
            raise ValueError(
                f"phase {index}: max_influences={phase.max_influences} must be "
                f"smaller than number_of_bones={self.number_of_bones}"
            )
        if phase.total_nnz_B_rt > n_coefficients:
            raise ValueError(
                f"phase {index}: total_nnz_B_rt={phase.total_nnz_B_rt} exceeds "
                f"6*S*P={n_coefficients}"
            )
```

Update the `run()` docstring: workflow items 7–8 become "run every phase of
`self.schedule`"; "Two-Phase Training Strategy" becomes "Training schedule"
pointing at `build_training_schedule`; add `reconstruction_error` to Side
Effects; add `Raises: ValueError` for the schedule check.

### 4.3 `train()`

```python
def train(
    self,
    B_rt: torch.Tensor,
    TR: torch.Tensor,
    A: torch.Tensor,
    W: torch.Tensor,
    phase: TrainingPhase,
) -> None:
```

Inside, exactly four substitutions and nothing else:

| Today (`model_fit.py`) | After |
|---|---|
| `for i in range(self.iterations):` (600) | `for i in range(phase.iterations):` |
| `W_n = W / W.sum(dim=0) if normalizeW else W` (601) | `... if phase.normalize_weights else W` |
| `torch.topk(W, self.max_influences + 1, dim=0)` (622) | `torch.topk(W, phase.max_influences + 1, dim=0)` |
| `torch.topk(B_decider.flatten(), self.total_nnz_B_rt)` (629) | `torch.topk(B_decider.flatten(), phase.total_nnz_B_rt)` |

Adam construction (596-597), loss (608, 615), projections and logging stay
textually identical. Docstring: replace `normalizeW` with `phase` in
`Args:`; the "Note" about two phases now points at `build_training_schedule`.

### 4.4 Bit-exactness argument

With defaults, `self.schedule` is
`(TrainingPhase(10000, 8, 6000, False), TrainingPhase(10000, 8, 6000, True))`.
`run()` then executes the same tensor ops, in the same order, with the same
integers as `train(normalizeW=False)` followed by `train(normalizeW=True)`
today. The seed value and call site are unchanged and no torch RNG call is
added. Hence the output arrays are bit-identical on the same platform, which
AC2 verifies against the stored expected data.

## 5. CLI (`src/metacompskin/cli.py`)

```python
def _int_list(text: str) -> tuple[int, ...]:
    """Parses "5000,5000,10000" into (5000, 5000, 10000) for argparse.

    Raises:
        argparse.ArgumentTypeError: If any item is not an integer.
    """
    try:
        return tuple(int(item) for item in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from e
```

Change `--iterations` (`cli.py:39`) to
`parser.add_argument("--iterations", type=_int_list, default=(10000,), help="steps per stage, e.g. 5000,5000,10000")`;
`_compressor_settings` passes the tuple through unchanged (a one-entry
tuple is today's run). After `--alpha` (`cli.py:45`) add
`parser.add_argument("--seed", type=int, default=None)` and `"seed"` to the
option tuple (`cli.py:97-103`). Update the module docstring example
(`cli.py:8-9`).

## 6. Maya pipeline (`src/metacompskin/maya_pipeline.py`)

`CompressionSettings.iterations` (`maya_pipeline.py:96`) becomes
`int | tuple[int, ...] = 10000`; add `seed: int | None = None` after `alpha`
with `Attributes:` entries. `compression_command` (`maya_pipeline.py:399-407`)
formats the iteration value with a private helper
`_format_ints(value)` returning `str(value)` for an int and `",".join(...)`
for a tuple, so `["--iterations", "10000"]` is unchanged for the default. Add
`"--seed": settings.seed,` after `"--alpha"` in the options dict.
`compress_and_build_rig` (`maya_pipeline.py:123-137`): widen `iterations` to
`int | tuple[int, ...]`, add `seed: int | None = None` after `alpha`, pass
both through, document them.

## 7. Experiment script (`scripts/compare_schedules.py`)

Standalone module; `main(argv) -> Path` returns the CSV path. Everything but
`run_variant` and `main` is pure and unit-tested.

```python
@dataclass(frozen=True)
class Variant:
    name: str
    schedule: tuple[TrainingPhase, ...]      # already sized for total_iterations

    @property
    def phases(self) -> int: return len(self.schedule)
    @property
    def influence_stages(self) -> str        # e.g. "32,16,8" (K of the normalised phases)
    @property
    def nnz_stages(self) -> str

@dataclass(frozen=True)
class RunResult:
    variant: str; seed: int; phases: int; iterations_per_phase: int; total_steps: int
    influence_stages: str; nnz_stages: str
    max_abs: float; mean_abs: float; seconds: float; device: str; torch_version: str

@dataclass(frozen=True)
class VariantSummary:
    variant: str; runs: int
    max_abs_mean: float; max_abs_std: float; max_abs_min: float; max_abs_max: float
    mean_abs_mean: float; mean_abs_std: float

def iterations_per_phase(total_iterations: int, phases: int) -> int     # total // phases; ValueError if < 1
def build_variants(total_iterations: int, max_influences: int, total_nnz: int,
                   n_coefficients: int, anneal_stages: int) -> list[Variant]
def summarize(results: list[RunResult]) -> list[VariantSummary]         # numpy; std ddof=1 when n > 1 else 0.0
def format_summary(summaries: list[VariantSummary]) -> str              # aligned text table
def write_csv(results: list[RunResult], path: Path) -> None             # csv.DictWriter over dataclasses.asdict
def run_variant(model_data, variant, seed, settings: dict, output: Path) -> RunResult
def build_parser() -> argparse.ArgumentParser
def main(argv=None) -> Path
```

`build_variants` (N = `anneal_stages`, cap = `n_coefficients`):

```
baseline          build_training_schedule((iterations_per_phase(total, 2),), K, L)
anneal_k          T = iterations_per_phase(total, N + 1); build_training_schedule((T,) * N, K, L)
baseline_control  T as above; (TrainingPhase(T, K, L, False),) + (TrainingPhase(T, K, L, True),) * N
anneal_kl         T as above; K_i = K * 2 ** (N - 1 - i); L_i = min(cap, L * 2 ** (N - 1 - i));
                  (TrainingPhase(T, K_0, L_0, False),) + tuple(TrainingPhase(T, K_i, L_i, True) for i)
```

`run_variant` constructs
`SkinCompressor(model_data, seed=seed, **settings)`
(settings: `number_of_bones`, `max_influences`, `total_nnz_B_rt`,
`init_weight`, `power`, optional `rest_joint_matrices` read with
`metacompskin.cli._stored_joint_matrices`), then sets
`compressor.schedule = variant.schedule`, calls `run(output)` under
`time.perf_counter()`, and reads `compressor.reconstruction_error`. `alpha` is
given to `BlendshapeModelData.from_npz(path, alpha=...)` once in `main`, as
`cli.py:71` does.

Loop order in `main`: for each seed, for each variant, so partial results
stay balanced if interrupted. Append each result to the CSV as it finishes
(header written once). Print the summary at the end and write it to
`OUT_DIR/summary.txt`. `--variants` filters by name; unknown names are an
error listing the known ones. `--dry-run` prints one line per (variant,
seed) with phases, iterations per phase and the K/L stages, then exits.

Fairness: every variant gets `total_iterations` steps in total. Print a
warning when `total_iterations` is not divisible by a variant's phase count.

The script is model-agnostic; the private companion repository wraps it
with `scripts/run_schedule_experiment.py`, which builds a model NPZ from its
Maya OBJ fixtures (cached) and forwards the remaining flags here, plus a
`slow`-marked end-to-end test. Wall time scales with
`4 variants × len(seeds) × total_iterations` and the hardware's steps/second; measure that once (`--dry-run` shows the full run
matrix without spending compute) before committing to a large sweep.

## 8. Testing strategy

| Area | File | What |
|---|---|---|
| Seed | `tests/test_skin_compressor.py` | default; constructor value; same seed → identical `weights`/`shapeXform`; different seed → different |
| Seed CLI | `tests/test_cli.py` | `--seed` forwarded |
| Seed pipeline | `tests/test_maya_pipeline.py` | every-setting command includes `--seed 5`; defaults unchanged |
| Metrics | `tests/test_skin_compressor.py` | `reconstruction_error` set after run, finite, `max_abs >= mean_abs > 0`; `None` before |
| Phase/schedule (pure) | `tests/test_training_schedule.py` (new) | `TrainingPhase` validation; `build_training_schedule` for one and four stages, per-stage lengths; empty rejected |
| Compressor + schedule | `tests/test_skin_compressor.py` | default 2-phase; `iterations=(100,100,300)` shape; `iterations=()`/`0` rejected; run with 3 stages, K=2, P=10; run-time bound errors; assigned schedule honoured |
| Anneal CLI | `tests/test_cli.py` | `--iterations 100,100,300 --max-influences 2` → ≤2 non-zeros; `--iterations 300` unchanged; `3,x` exits 2 |
| Anneal pipeline | `tests/test_maya_pipeline.py` | tuple emitted as `5000,5000,10000`; int default unchanged |
| Regression | `tests/test_default_output_macos.py -k short_iter` | bit-exact defaults (AC2) |
| Smoke | `tests/test_pipeline_smoke.py` | unchanged, must stay green |
| Script | `tests/test_compare_schedules.py` (new) | `build_variants`, `iterations_per_phase`, `summarize`, `format_summary`, `write_csv`, `build_parser` defaults |

Grid-model parameters (`tests/conftest.py`): S=3, N=25; use
`number_of_bones=10` for run tests, so K < 10 and L ≤ 180. With
`max_influences=2, iterations=(100, 100, 300)` the K stages are 8, 4, 2; with
four entries the warm-up K is 16 ≥ 10 and must be rejected at `run()`.

## 9. Documentation changes

| File | Change |
|---|---|
| `docs/user_guide/compressing.md` | settings table: `seed`, widened `iterations`; "attributes you can change before run": `schedule`; CLI option list (line 30-31); new section "Annealing the influence budget" after "Choosing settings" (sequence `iterations`, K per stage, warm-up rule, N+1 phases, total steps, N ≤ log2(P/K)+1, make the last stage the longest, example command, hand-built schedule example); Reproducibility (150-156) around `seed`; "After the run" mentions `reconstruction_error`; "Batch processing" points to `scripts/compare_schedules.py` |
| `docs/concepts/how_the_solver_works.md` | "Two phases" (74-84) → "Training phases" with the table from spec §7.1 and a note on a fresh Adam per phase; init section (86-96) mentions `seed`; "Reading the log" shows the phase header; settings table (130-137): `iterations` row rewritten, `seed` row added |
| `docs/user_guide/evaluating_results.md` | after the headline numbers: `compressor.reconstruction_error` |
| `CLAUDE.md` | Key Parameters table: `seed` row; `iterations` note |
| `tests/test_data/*/SETUP.md` | seed row wording |
| `src/metacompskin/cli.py` module docstring | example with new flags |

Build with `cd docs && make clean && make html`; zero warnings. The Skin
Compressor API page must show `TrainingPhase`, `ReconstructionError`,
`build_training_schedule` and not `_check_schedule_fits_model`.

## 10. Error messages

- `"iterations must contain at least one stage"`
- `"schedule must contain at least one TrainingPhase"`
- `"phase 1: max_influences=16 must be smaller than number_of_bones=10"`
- `"phase 2: total_nnz_B_rt=500 exceeds 6*S*P=180"`
- `"iterations must be at least 1, got 0"` (from `TrainingPhase`)

## 11. Performance

Per-iteration cost is unchanged: both `topk` calls scan the full tensors
whatever K and L are, so their cost is flat in the sparsity budget —
confirm this on the target device before relying on it (research.md §9).
Total cost is phases × `iterations`; the script equalises it.

## 12. Alternatives considered

- **Per-stage K/L lists (`influence_schedule=(32,16,8)`)** — revision 1.
  Rejected: needed length matching, broadcasting and final-budget conflict
  rules to support flexibility nobody asked for.
- **Integer `anneal_stages` beside a scalar `iterations`** — revision 2.
  Rejected: the two collapse into one sequence whose length is the stage
  count and whose entries are the per-stage steps; that also removes any
  need for a separate per-stage-iterations parameter.
- **A new `stage_iterations` parameter next to the old `iterations`** —
  rejected: two parameters for one thing, with conflict rules; widening
  `iterations` is backward compatible and needs neither.
- **A `schedule=` constructor argument** in addition to the sequence `iterations` —
  rejected: two construction paths for one thing; attribute assignment is
  the codebase's existing override idiom (`alpha`).
- **A public `anneal_factor` / `anneal_nnz` knob** — rejected for now
  (YAGNI); trivially added later if the experiment motivates it.
- **N counts phases instead of stages** — viable; kept stages so N=1 is
  today's run without a special case. One-line change in the builder.
- **Three extra keyword arguments on `train()`** — rejected: eight
  parameters and no printable phase object.
- **Validating L ≤ 6SP at construction** — rejected: breaks an existing
  construct-only test and the assigned schedule is only known at `run()`.
- **Experiment code inside the package** — rejected: no runtime need.
