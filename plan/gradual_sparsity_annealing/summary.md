# Summary — sparsity annealing, seed control, schedule experiment

One-page digest of [research.md](research.md), [spec.md](spec.md),
[design.md](design.md) and [tasks.md](tasks.md). Read those for detail; this
file is the record of what the plan was, why, and what landed.

## Status

Implemented in commit `9dd5186` (three planned PRs: seed, annealing,
experiment script). The `spec.md` header still says "planned, not started"
and the `tasks.md` boxes are unticked; both are stale. Not verifiable from the
repo: the experiment run itself and its recorded result, since the data it
ran on is not here.

## Reasoning

- The solver applies two projections after every Adam step: keep the top K
  weights per vertex, and keep the top L delta coefficients globally. Both
  run from iteration 1, when the joints are noise.
- A zeroed weight almost never re-enters. One Adam step is about the learning
  rate (1e-3), while the K+1-th weight in a column is soon about 0.1. So each
  vertex commits to its K joints on noise, and the assignment is locked in
  within the first steps.
- Consequence: the seed picks a basin, and single-seed hyper-parameter sweeps
  (for example over L) are confounded with it. A quality drop at high L looks
  like a worse basin, not a capacity limit.
- Hypothesis: gradual magnitude pruning (Zhu & Gupta 2017, Han et al. 2015,
  Gale et al. 2019) delays the K commitment until the joints carry
  information. That should give a better and more seed-stable result at the
  same shipped K, with no change to the runtime format.
- `power` and `alpha` are held fixed in the experiment so that only the
  schedule varies.

## Spec

Three additions, defaults bit-identical to before:

1. **Seed.** `SkinCompressor(seed=12345)`, `--seed`, the Maya pipeline
   settings, and `reconstruction_error` (MXE and MAE kept on the object).
2. **Annealing.** `iterations` may be a sequence, one entry per stage. Stage i
   keeps `K * 2**(N-1-i)` influences, for example 64, 32, 16, 8 for K=8 and
   four stages. A warm-up phase with weight normalisation off precedes the
   first stage. An int is today's two-phase run. New public names:
   `TrainingPhase`, `build_training_schedule`, `ReconstructionError`, and a
   `schedule` attribute that can be reassigned before `run()` (the only way
   to anneal L). The CLI takes `--iterations 5000,5000,10000`.
3. **Experiment.** `scripts/compare_schedules.py` runs the variants
   `baseline`, `baseline_control` (same phase count, no annealing),
   `anneal_k` and `anneal_kl` over several seeds at equal total steps. It
   writes a CSV and prints mean ± std of MXE and MAE per variant.

Out of scope: a public L-annealing knob, a tunable halving factor, regrowth
of pruned entries, annealing P, any change to the NPZ format.

## Design

- The schedule is a tuple of frozen `TrainingPhase(iterations,
  max_influences, total_nnz_B_rt, normalize_weights)`. `train()` takes one
  phase instead of a `normalizeW` flag.
- `run()` prints a `phase i/n: iterations=… K=… L=…` header per phase.
- Bit-exactness for the default path: one stage gives phases `(T, K, L,
  False)` and `(T, K, L, True)`, the same as the old hard-coded two calls.
  The pinned regression outputs pass without regeneration.
- The experiment script keeps its pure logic (`build_variants`, `summarize`,
  `write_csv`) unit-testable, appends to the CSV after each run, and loops
  seeds in the outer loop so partial results stay balanced.

Rejected: per-stage K/L lists, a separate `anneal_stages` integer, a second
`stage_iterations` parameter, a `schedule=` constructor argument (attribute
assignment is the existing idiom, as with `alpha`), and putting the
experiment code inside the package.

## What landed

`TrainingPhase`, `build_training_schedule` and the `schedule` attribute are in
`model_fit.py`. The CLI accepts `--seed` and comma-separated `--iterations`.
`scripts/compare_schedules.py` and `tests/test_compare_schedules.py` exist, as
do `tests/test_training_schedule.py` and the annealing sections of
`docs/user_guide/compressing.md` and `docs/concepts/how_the_solver_works.md`.
Later work builds on it: the geodesic candidate filter warms up at the mask
width M, and the root-influence plan extends `build_training_schedule` with
`start_influences`.
