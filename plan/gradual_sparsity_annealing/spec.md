# Spec — Sparsity annealing (`iterations` per stage), seed control, and the schedule-vs-seed experiment

Status: planned, not started. Revision 3, 2026-09-08, against commit `87dcd6d`
on `main`. Revision 1 used explicit per-stage K/L lists, revision 2 a single
integer `anneal_stages`; both were replaced by letting `iterations` be a
sequence — one entry per stage (see §11).
Companion documents: [design.md](design.md) (how), [tasks.md](tasks.md)
(ordered checklist), [research.md](research.md) (why, with references).

## 1. Summary

Three additions to `metacompskin`, delivered as three pull requests:

1. **Seed control.** `SkinCompressor` accepts a `seed` argument (default
   12345, today's hard-coded value), exposed on the CLI (`--seed`) and in the
   Maya pipeline. A small companion change keeps the final MXE/MAE on the
   compressor object (`reconstruction_error`) instead of only printing them.
2. **Sparsity annealing.** `iterations` may now be a sequence, one entry per
   stage: `iterations=(5000, 5000, 5000, 10000)` runs four stages of those
   lengths with the per-vertex influence budget K·2^(N−1), …, 2K, K — for
   K=8: 64, 32, 16, 8. An int (or a one-entry sequence) is today's run, bit
   for bit. Power users can assign any schedule of `TrainingPhase` objects to
   `compressor.schedule` before `run()`.
3. **Experiment script.** `scripts/compare_schedules.py` runs the baseline,
   a phase-count control, and the annealed variants over several seeds at
   equal total step counts, writes a CSV, and prints mean ± std of MXE and
   MAE per variant, so the effect of annealing can be judged against
   seed-to-seed noise.

## 2. Background and motivation

### 2.1 What the solver does today

`SkinCompressor.run()` (`src/metacompskin/model_fit.py:249`) initialises the
deltas `B_rt` and weights `W` from Gaussian noise, then calls `train()` twice
(`model_fit.py:364-365`): first with weight normalisation off, then on. Every
iteration of `train()` applies two projections after the Adam step
(`model_fit.py:621-634`):

- keep the K largest weights per vertex, zero the rest, clamp negatives;
- keep the L largest `|B_rt|` coefficients globally, zero the rest.

Both budgets are fixed for the whole run and applied **from iteration 1**,
when `B_rt` is noise and `W` is essentially zero (`1e-8 * randn`). Each
vertex therefore commits to its K joints before those joints mean anything,
and the commitment is practically irreversible (research.md §3).

The random seed is hard-coded to 12345 (`model_fit.py:228-229`).

### 2.2 Why this might help

On many deployment targets, K and P are fixed by the platform (a GPU
skinning shader budget, a joint-count limit) and are not free tuning knobs.
That leaves L as the main capacity dial for a fixed K, P — and because the
lock-in effect above applies to both projections, a capacity sweep over L
can behave non-monotonically: quality improving as L grows, then
*degrading* past some point, rather than plateauing. That shape is the
signature of the optimiser settling into a worse basin as capacity
changes, not of a capacity limit being reached (research.md §3) — and a
single-seed sweep cannot tell the two apart, since seed alone already
determines which basin is found.

### 2.3 Hypothesis

Delaying the K commitment until the joints have self-organised (gradual
magnitude pruning, research.md §4) should give a better and more seed-stable
final assignment at the same shipped K, with no change to the runtime
format. Whether it does cannot be decided from one run per configuration;
the seed spread must be measured too.

## 3. Goals

- G1. The seed is a first-class, documented setting wherever settings are set
  (constructor, CLI, `CompressionSettings`, `compress_and_build_rig`).
- G2. Annealing is reachable with **one argument**: `iterations` as a
  sequence, whose length is the number of stages and whose entries are the
  steps per stage — on the constructor, the CLI and the Maya pipeline.
- G3. The default path is numerically identical: the platform regression
  tests (`tests/test_default_output*.py`, `tests/test_vertex_positions.py`)
  pass **without regenerating expected data**.
- G4. A repeatable experiment answers: "does annealing beat the baseline by
  more than the seed noise, at equal total iterations?"
- G5. Documentation and API docs updated; tooling green (`ruff`, `mypy`,
  `pytest`, Sphinx without warnings).

## 4. Out of scope

- Annealing L in the public API (testable through the explicit schedule; a
  public knob is a follow-up once the experiment says it helps).
- Changing the loss: `power` other than 2 in the experiment, alpha retuning,
  residual-target Laplacian.
- A tunable halving factor, linear/cubic ramps, an independently sized
  warm-up on the public API (the explicit schedule allows all of these),
  regrowth of pruned entries.
- Annealing P (fixed by the platform).
- Cross-device determinism; CUDA non-determinism.
- Plotting.
- Any change to the output NPZ format or to `AnimationFrameGenerator`,
  `maya_rig_builder`, `maya_exporter`.

## 5. Users and use cases

| Who | Wants |
|---|---|
| TD tuning a head model | `--iterations 5000,5000,10000 --seed 7` from the shell, keep the best NPZ. |
| Pipeline engineer calling from Maya | `compress_and_build_rig(..., seed=…, iterations=(…))`. |
| Researcher | Run `scripts/compare_schedules.py` on a model and read one table; hand-build unusual schedules with `TrainingPhase`. |
| Maintainer | Regression suites unchanged; new behaviour behind explicit arguments. |

## 6. User-facing behaviour

### 6.1 Python API

New public names in `metacompskin.model_fit`:

```python
@dataclass(frozen=True)
class TrainingPhase:
    iterations: int  # Adam steps in this phase
    max_influences: int  # K applied by the per-vertex projection
    total_nnz_B_rt: int  # L applied by the global projection
    normalize_weights: bool  # partition of unity inside the forward pass


def build_training_schedule(
    stage_iterations: Sequence[int],
    max_influences: int,
    total_nnz_B_rt: int,
) -> tuple[TrainingPhase, ...]: ...


@dataclass(frozen=True)
class ReconstructionError:
    max_abs: float  # MXE, what run() prints as maxDelta
    mean_abs: float  # MAE, what run() prints as meanDelta
```

`SkinCompressor.__init__` gains one keyword argument (`seed`) and widens the
type of one (`iterations`); nothing else in the signature changes:

```python
SkinCompressor(
    model_data,
    iterations=10000,  # int | Sequence[int]: one entry per stage
    rest_joint_matrices=None,
    number_of_bones=None,
    max_influences=8,
    total_nnz_B_rt=6000,
    init_weight=1e-3,
    power=2,
    seed=12345,  # new
)
```

New attributes:

- `seed: int`.
- `stage_iterations: tuple[int, ...]` — `iterations` normalised to a tuple
  (`10000` → `(10000,)`).
- `schedule: tuple[TrainingPhase, ...]` — built at construction from
  `stage_iterations`, `max_influences`, `total_nnz_B_rt`. It may be
  **reassigned before `run()`**, exactly like `alpha` today
  (`docs/user_guide/compressing.md` "Attributes you can change after
  construction and before run"). `run()` executes whatever is there.
- `reconstruction_error: ReconstructionError | None` — `None` until `run()`
  has finished.

`self.iterations` keeps the value as given (int or the sequence) for
backward compatibility; nothing reads it after construction.

`SkinCompressor.train` changes signature from
`train(B_rt, TR, A, W, normalizeW=False)` to
`train(B_rt, TR, A, W, phase: TrainingPhase)`. `run()` is the supported
entry point; nothing in the repository calls `train` directly except `run()`.

### 6.2 Command line

```
python -m metacompskin model.npz out.npz --iterations 5000,5000,5000,10000 --seed 7
```

- `--iterations INT[,INT...]` — one entry per stage; a single integer is
  today's behaviour. Malformed values are an argparse error (exit 2).
- `--seed INT` (default: compressor default 12345).

### 6.3 Maya pipeline

`CompressionSettings.iterations` becomes `int | tuple[int, ...] = 10000`;
`compression_command` emits `--iterations 10000` for an int (unchanged) and
`--iterations 5000,5000,10000` for a tuple. `CompressionSettings` gains
`seed: int | None = None`, forwarded as `--seed` when set.
`compress_and_build_rig` accepts the same `iterations` type and a `seed`
keyword.

### 6.4 Progress log

`run()` prints one header line before each phase:

```
phase 1/5: iterations=5000 K=64 L=6000 normalize_weights=False
00000(0.123) 1.23456e-02 4.56789e-01 850 47824
...
```

The per-200-iteration lines are unchanged. With default settings the header
reads `phase 1/2` and `phase 2/2`.

### 6.5 Experiment script

```
python scripts/compare_schedules.py MODEL.npz OUT_DIR \
    [--seeds S1,S2,...] [--total-iterations N] [--anneal-stages N] \
    [--number-of-bones P] [--max-influences K] [--total-nnz-b-rt L] \
    [--alpha A] [--power P] [--init-weight W] \
    [--variants baseline,baseline_control,anneal_k,anneal_kl] \
    [--ignore-joint-matrices] [--dry-run]
```

`--anneal-stages` (N) is a script-only convenience: the script builds the
`iterations` sequences and explicit schedules itself so that every variant
gets the same total step count. Variants (all end at the same K and L;
cap = 6·S·P; T = `total_iterations // phases`):

| Variant | How it is built | Phases | Purpose |
|---|---|---|---|
| `baseline` | `iterations=(T,)` | 2 | today's behaviour |
| `baseline_control` | explicit schedule: N+1 phases all at (K, L), first unnormalised | N+1 | control: same phase count and Adam resets as the annealed runs, no annealing |
| `anneal_k` | `iterations=(T,) * N` | N+1 | anneal weights only (the public feature) |
| `anneal_kl` | explicit schedule: K as `anneal_k`, L_i = min(cap, L·2^(N−1−i)) | N+1 | anneal both |

One NPZ per run goes to `OUT_DIR/<variant>/seed_<seed>.npz`.
`OUT_DIR/results.csv` has one row per run: `variant, seed, phases,
iterations_per_phase, total_steps, influence_stages, nnz_stages, max_abs,
mean_abs, seconds, device, torch_version`. At the end the script prints, per
variant: n, MXE mean ± std (min, max), MAE mean ± std, and writes the same to
`OUT_DIR/summary.txt`. `--dry-run` prints the run matrix and per-phase
iteration counts without compressing anything.

### 6.6 Documentation

- `docs/user_guide/compressing.md`: settings table rows for `seed` and the
  widened `iterations`; `schedule` in the "attributes you can change" table; CLI
  option list; new section "Annealing the influence budget"; Reproducibility
  section rewritten around `seed`; "After the run" mentions
  `reconstruction_error`; pointer to the experiment script.
- `docs/concepts/how_the_solver_works.md`: "Two phases" becomes "Training
  phases" and explains the schedule; initialisation mentions `seed`; log
  section shows the phase header; settings table rows.
- `docs/user_guide/evaluating_results.md`: mention `reconstruction_error`.
- `CLAUDE.md` Key Parameters table: add `seed`; note `iterations` may be a
  sequence.
- `tests/test_data/{macos,windows}/SETUP.md`: seed row says it is the default
  of the `seed` argument.
- API pages are automodule stubs; new public names appear automatically.

## 7. Semantics in detail

### 7.1 Schedule built from `iterations`

Let `stage_iterations` = (T_0, …, T_(N−1)) be `iterations` as a tuple,
N its length, K = `max_influences`, L = `total_nnz_B_rt`. Stage i uses
K_i = K·2^(N−1−i) and runs T_i steps. The schedule has **N + 1 phases**:

| Phase | steps | K | L | normalize_weights |
|---|---|---|---|---|
| 0 (warm-up) | T_0 | K_0 = K·2^(N−1) | L | False |
| 1 | T_0 | K_0 | L | True |
| 2 | T_1 | K_1 = K·2^(N−2) | L | True |
| … | … | … | L | True |
| N | T_(N−1) | K_(N−1) = K | L | True |

`iterations=10000` (N = 1) gives exactly today's two phases of 10 000 steps.
`iterations=(5000, 5000, 5000, 10000)` with K = 8 gives phases
(5000@64 warm-up, 5000@64, 5000@32, 5000@16, 10000@8), 30 000 steps in all.

Why the first stage runs twice: the unnormalised warm-up is today's phase 1
(weights grow freely to discover ownership) and belongs at the loosest K,
where ownership is being decided. Giving it T_0 steps is what keeps an int
`iterations` identical to the current behaviour with no special case. A
warm-up of a different length is an explicit-schedule matter.

### 7.2 Validation

At construction: `iterations` must be an int ≥ 1 or a non-empty sequence of
ints ≥ 1, else `ValueError` (the per-entry check comes from `TrainingPhase`).

At the start of `run()`, for whatever `self.schedule` holds:

- non-empty sequence of `TrainingPhase`;
- every phase `max_influences < P` (the projection uses `topk(K + 1)`);
- every phase `total_nnz_B_rt <= 6·S·P`.

Each violation raises `ValueError` naming the phase, the value and the limit.
Today these surface as an opaque `torch.topk` error. Bounds are checked at
`run()` and not at construction because
`tests/test_skin_compressor.py::test_settings_are_taken_from_the_constructor`
constructs L above the cap without running, and because a user-assigned
schedule is only known at `run()`.

The largest usable N is set by K·2^(N−1) < P: at the defaults (K = 8,
P = 100) that gives N ≤ 4.

### 7.3 What does not change

- Numerics of every projection, the loss, Adam settings, the init, the seed
  default, output NPZ keys/shapes, `loss_list` / `abserr_list` sampling.
- `max_influences` and `total_nnz_B_rt` keep their types and defaults; an
  int `iterations` keeps its meaning.
- A fresh Adam optimizer per phase (already the case).

## 8. Acceptance criteria

- AC1. `SkinCompressor(model_data)` has `seed == 12345`,
  `stage_iterations == (10000,)`, `len(schedule) == 2`, `schedule[0].normalize_weights is False`,
  `schedule[1].normalize_weights is True`, both phases at K=8, L=6000,
  10 000 iterations; `reconstruction_error is None`.
- AC2. On macOS (`tests/test_data/macos/`), `pytest -k short_iter -v` passes
  after every PR **without any change under `tests/test_data/`**. On the
  Windows/CUDA machine the same holds for `tests/test_default_output.py`.
- AC3. Same `seed` and settings → identical `weights` and `shapeXform`;
  different seeds → different arrays.
- AC4. `iterations=(100, 100, 300), max_influences=2, number_of_bones=10` on
  the test grid model: `schedule` has 4 phases with (steps, K)
  ((100, 8), (100, 8), (100, 4), (300, 2)); after `run()` at most 2 non-zero
  weights per vertex, non-negative, summing to 1.
- AC5. `iterations=()` and `iterations=0` raise `ValueError` at construction;
  `iterations=(100,) * 4, max_influences=2, number_of_bones=10` (K_0 = 16 ≥ 10)
  and `total_nnz_B_rt=500` on the grid (cap 180) raise `ValueError` at
  `run()` before any training.
- AC6. A schedule assigned to `compressor.schedule` before `run()` is what
  `run()` executes (verified through the non-zero count of the output).
- AC7. CLI: `--seed` and a comma-separated `--iterations` accepted and
  forwarded; `--iterations 300` still works; `--iterations 3,x` exits 2.
- AC8. `compression_command` emits `--iterations 10000` for an int and
  `--iterations 5000,5000,10000` for a tuple, `--seed` exactly when set, and
  `test_defaults_only_pass_the_iteration_count` passes unchanged.
- AC9. After `run()`, `reconstruction_error` equals the printed `maxDelta` /
  `meanDelta`.
- AC10. `scripts/compare_schedules.py --dry-run` prints the run matrix; a real
  run produces `results.csv` with the columns in §6.5 and one NPZ per run;
  its pure helpers are unit-tested.
- AC11. `ruff format --check .`, `ruff check .`,
  `mypy src/ --ignore-missing-imports` clean; `cd docs && make html` with no
  warnings; every new public name has a Google-style docstring.
- AC12. Experiment executed against the real-data fixtures in the private
  companion repository (tasks.md §4), results and interpretation recorded
  there as `experiments/schedule_experiment/results.md` plus the CSV.

## 9. Constraints

- TDD as in `CLAUDE.md`: failing test first, minimal code, refactor.
- Keep `ruff` and `mypy` on their latest releases; fix code, never pin back.
- Full test suite >10 minutes on the development Mac; use the `-k` selections
  in tasks.md §0.
- `docs/api/` stubs are hand-maintained; never run `sphinx-apidoc`.
- No change to expected regression data. If a default-path test fails, the
  change is wrong, not the data.
- Anything that needs real (non-sample) model data — the experiment itself,
  its end-to-end smoke test — lives in the private companion repository
  `meta-compskin_private_tests` and uses the fixtures there. This repository
  stays generic: no model-specific dimensions, budgets or results.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Refactoring `train()` changes floating-point op order | Read K/L/normalize/iterations from `phase` and nothing else; keep every tensor expression textually identical; run `-k short_iter` after each WS2 step. |
| Torch RNG consumed before `B_rt`/`W` init | Schedule construction is pure Python. |
| Warm-up K ≥ P for long `iterations` sequences | Run-time check with a clear message; docs state N ≤ log2(P/K)+1. |
| Warm-up length tied to stage 0 surprises someone | Documented next to the parameter, with the explicit-schedule escape hatch. |
| Experiment compares unequal compute | Equal total steps per variant; `baseline_control` isolates the phase-count effect. |
| Five seeds too few for a small effect | Report min/max/std; research.md §8 decision rule; `--seeds` takes more. |
| `test_runs_the_package_cli_with_every_setting` asserts the exact command list | Update the expected list in the same commit (test first). |
| Assignable `schedule` lets `max_influences` disagree with the final phase | Documented: the schedule is authoritative for `run()`; the ints are what the default schedule is built from. Nothing after training reads them. |

## 11. Decisions made (and what revisions 2–3 changed)

- **`iterations` as a sequence (revision 3).** Revision 2 had an integer
  `anneal_stages` next to a scalar `iterations`. The owner pointed out the
  two collapse into one array: its length is the stage count and its entries
  are the steps per stage, which also gives per-stage iteration control for
  free. Reusing the existing `iterations` name (widened to
  `int | Sequence[int]`) means no new parameter, no conflict rules and full
  backward compatibility.
- **Warm-up length = first stage length.** The only way to keep an int
  `iterations` bit-identical to today without a special case; an
  independently sized warm-up is an explicit-schedule matter.
- **One integer instead of per-stage K/L lists (revision 2).** Revision 1 exposed
  `influence_schedule` / `nnz_schedule` tuples plus rules to derive the final
  budgets, broadcast a missing list and reject conflicts. Review judged that
  over-engineered for the hypothesis: the geometric staircase is the whole
  idea and `max_influences` / `total_nnz_B_rt` keep their meaning and defaults.
- **Explicit schedules through attribute assignment**, not a constructor
  argument: mirrors how `alpha` is overridden today, keeps one construction
  path, and gives experiments (and future ramps) full control with
  `TrainingPhase` objects — including per-phase iteration counts.
- **L is not annealed by the public knob.** The lock-in argument is strongest
  for W; L annealing is unproven and is one of the arms of the experiment.
  Baking it into the convenience path before the data is in would risk
  shipping a harmful default.
- **N stages → N+1 phases.** Keeps N=1 identical to today without a special
  case. Alternative (N counts phases) is a one-line change in the builder;
  flagged for the owner to overrule.
- **Halving factor fixed at 2**, as requested. A finer tail (e.g. 32, 16, 12,
  8) is a follow-up if the final phase is seen to struggle.
- **Warm-up at K·2^(N−1), not unconstrained**: the paper argues sparsity from
  step 1 helps the solver adapt, and K=32 is the regime its HD experiment
  validated (paper lines 626-633).
- **Total steps = T_0 + Σ T_i.** The owner sizes the final stage directly;
  the docs recommend making it the longest, since it follows the harshest cut.
- **Bounds checked at `run()`, `iterations` shape at construction.**
- **`ReconstructionError` stored, not returned**: `run()` stays a command.
- **Experiment helpers in `scripts/`, unit-tested** via `pythonpath = ["src", "scripts"]`.
- New dataclasses importable from `metacompskin.model_fit`; not added to the
  lazy re-exports in `metacompskin/__init__.py`.

## 12. Deliverables

| PR | Branch | Contents |
|---|---|---|
| 1 | `feature/seed-parameter` | seed argument (API, CLI, pipeline), `ReconstructionError`, docs, tests |
| 2 | `feature/sparsity-annealing` | `TrainingPhase`, `build_training_schedule`, sequence `iterations`, assignable `schedule`, CLI + pipeline, docs, tests |
| 3 | `feature/schedule-experiment` | `scripts/compare_schedules.py`, its tests; the real-data driver, slow test and results live in `meta-compskin_private_tests` |

Commit messages follow Conventional Commits (`feat(model_fit): …`,
`docs: …`, `test: …`), see `docs/developer/development.md`.
