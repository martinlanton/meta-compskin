# Tasks — ordered checklist (revision 3)

Work through these in order. Each task is one red → green → refactor cycle
or one non-code step, sized for a single commit. Tick the box when the
"Done when" line holds. Read [spec.md](spec.md) and [design.md](design.md)
before starting; [research.md](research.md) explains the why.

## 0. Before you start

- [ ] **0.1 Environment.** `pip install -e ".[dev]"` in a fresh venv
  (Python ≥ 3.10; the regression data was made with 3.13 / torch 2.11 on
  macOS, see `tests/test_data/macos/SETUP.md`). Run `pre-commit install`.
- [ ] **0.2 Learn the fast test loops.** The whole suite takes >10 min on a
  laptop. Use:
  - `pytest -k "not default_output and not vertex_positions" -q` — the fast
    grid-model tests, about a minute.
  - `pytest -k short_iter -v` — the bit-exact regression at 600 iterations, a
    few minutes on CPU. This proves you did not change the numerics. It runs
    only on the platform whose expected data you have (macOS or Windows/CUDA).
  - `ruff format . && ruff check --fix . && mypy src/ --ignore-missing-imports`
- [ ] **0.3 Baseline.** On `main`, run both pytest commands and confirm green.
  If `short_iter` is not green *before* you change anything, stop and ask.
- [ ] **0.4 Read the code you will touch.** `src/metacompskin/model_fit.py`
  lines 64-250 (class + constructor), 249-400 (`run`), 513-652 (`train`);
  `src/metacompskin/cli.py`; `src/metacompskin/maya_pipeline.py` lines 75-137
  and 385-421; `tests/conftest.py`; `tests/test_skin_compressor.py`;
  `tests/test_cli.py`; `tests/test_maya_pipeline.py` lines 73-120.
- [ ] **0.5 Read the paper bits.** `paper/compressed_skinning_for_facial_blendshapes.md`
  lines 139-148 (initialisation), 440-497 (Eq. 9 and the projections),
  505-517 (Section 4.1), 626-633 (the K=32 experiment).

Conventions for every code task below:

- Write the test first, run it, **see it fail for the right reason**, then
  write the minimum code, run again, see it pass, then tidy.
- Tests are Arrange-Act-Assert; names say what is verified.
- Every new public function/class/dataclass has a Google-style docstring
  (`Args:`, `Returns:`, `Raises:`, `References:` for maths).
- Before each commit: `ruff format .`, `ruff check .`,
  `mypy src/ --ignore-missing-imports`, the fast pytest selection. Before each
  PR: `/pre-commit-checks` in full and `pytest -k short_iter -v`.

---

## 1. PR 1 — seed parameter (`feature/seed-parameter`)

- [ ] **1.0 Branch.** `git checkout -b feature/seed-parameter main`.

- [ ] **1.1 Test: seed default and constructor value.**
  File `tests/test_skin_compressor.py`.
  - In `test_defaults` add `assert compressor.seed == 12345`.
  - Add
    ```python
    def test_seed_is_taken_from_the_constructor(self, grid_model_data):
        compressor = SkinCompressor(model_data=grid_model_data, seed=7)

        assert compressor.seed == 7
    ```
  Done when: `pytest tests/test_skin_compressor.py -q` fails the new test with
  `TypeError: ... unexpected keyword argument 'seed'`.

- [ ] **1.2 Implement the seed argument.**
  File `src/metacompskin/model_fit.py`.
  - Add `_DEFAULT_SEED = 12345` under `_DEFAULT_MAX_INFLUENCES` (line 34).
  - Add `seed: int = _DEFAULT_SEED` as the last constructor parameter (after
    `power`, line 144).
  - Replace `self.seed = 12345` (line 228) with `self.seed = seed`. Leave
    `torch.manual_seed(self.seed)` where it is.
  - Docstring: `Args:` entry `seed: Torch random seed for the initial deltas and weights (default 12345).`;
    class `Attributes:` line 107 already lists `seed` — update its text.
  Done when: green.

- [ ] **1.3 Test: seed determines the output.**
  Same file, same class; mirror the existing run-based tests.
  ```python
  def test_same_seed_reproduces_the_output(self, grid_model_data, tmp_path):
      settings = dict(iterations=300, number_of_bones=10, total_nnz_B_rt=100, seed=3)

      SkinCompressor(model_data=grid_model_data, **settings).run(tmp_path / "a.npz")
      SkinCompressor(model_data=grid_model_data, **settings).run(tmp_path / "b.npz")

      first, second = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
      np.testing.assert_array_equal(first["weights"], second["weights"])
      np.testing.assert_array_equal(first["shapeXform"], second["shapeXform"])


  def test_different_seeds_change_the_output(self, grid_model_data, tmp_path):
      settings = dict(iterations=300, number_of_bones=10, total_nnz_B_rt=100)

      SkinCompressor(model_data=grid_model_data, seed=3, **settings).run(
          tmp_path / "a.npz"
      )
      SkinCompressor(model_data=grid_model_data, seed=4, **settings).run(
          tmp_path / "b.npz"
      )

      first, second = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
      assert not np.array_equal(first["weights"], second["weights"])
  ```
  These pass immediately (they pin behaviour the feature relies on); keep
  them. Done when: both green.

- [ ] **1.4 Test: CLI `--seed`.**
  File `tests/test_cli.py`.
  ```python
  def test_cli_forwards_the_seed(grid_model_npz, tmp_path):
      def compress(name, seed):
          output = tmp_path / f"{name}.npz"
          main(
              [
                  str(grid_model_npz),
                  str(output),
                  *_FAST,
                  "--number-of-bones",
                  "10",
                  "--seed",
                  str(seed),
              ]
          )
          return np.load(output)["weights"]

      first, second, third = compress("a", 3), compress("b", 3), compress("c", 4)

      np.testing.assert_array_equal(first, second)
      assert not np.array_equal(first, third)
  ```
  Done when: fails with `SystemExit: 2` ("unrecognized arguments: --seed").

- [ ] **1.5 Implement CLI `--seed`.**
  File `src/metacompskin/cli.py`.
  - After `--alpha` (line 45): `parser.add_argument("--seed", type=int, default=None)`.
  - In `_compressor_settings`, add `"seed"` to the option tuple (line 97-103).
  - Module docstring example (line 8-9): add `--seed 7`.
  Done when: `pytest tests/test_cli.py -q` green.

- [ ] **1.6 Test: pipeline forwards the seed.**
  File `tests/test_maya_pipeline.py`, `TestCompressionCommand`.
  In `test_runs_the_package_cli_with_every_setting` add `seed=5` to the
  `CompressionSettings(...)` and insert `"--seed", "5",` in the expected list
  right after `"--alpha", "20.0",` (before `"--ignore-joint-matrices"`).
  Done when: fails with `TypeError: ... unexpected keyword argument 'seed'`.

- [ ] **1.7 Implement pipeline seed.**
  File `src/metacompskin/maya_pipeline.py`.
  - `CompressionSettings`: add `seed: int | None = None` after `alpha`
    (line 102) and an `Attributes:` line.
  - `compression_command`: add `"--seed": settings.seed,` after
    `"--alpha": settings.alpha,` (line 414).
  - `compress_and_build_rig`: add `seed: int | None = None` after `alpha`
    (line 134); pass `seed=seed` into `CompressionSettings` (line 181-190);
    add an `Args:` line.
  Done when: `pytest tests/test_maya_pipeline.py -q` green, including
  `test_defaults_only_pass_the_iteration_count` untouched.

- [ ] **1.8 Test: reconstruction error is kept on the compressor.**
  File `tests/test_skin_compressor.py`.
  ```python
  def test_run_records_the_reconstruction_error(self, grid_model_data, tmp_path):
      compressor = SkinCompressor(
          model_data=grid_model_data,
          iterations=300,
          number_of_bones=10,
          total_nnz_B_rt=100,
      )

      compressor.run(tmp_path / "compressed.npz")

      error = compressor.reconstruction_error
      assert error is not None
      assert np.isfinite(error.max_abs) and np.isfinite(error.mean_abs)
      assert error.max_abs >= error.mean_abs > 0
  ```
  Also add `assert compressor.reconstruction_error is None` to `test_defaults`.
  Done when: fails with `AttributeError`.

- [ ] **1.9 Implement `ReconstructionError`.**
  File `src/metacompskin/model_fit.py`.
  - Add `from dataclasses import dataclass` to the imports.
  - Add the `ReconstructionError` dataclass from design.md §2 above the class.
  - In `__init__`, next to `self.loss_list` (line 242):
    `self.reconstruction_error: ReconstructionError | None = None`.
  - In `run()`, right after `print(f"meanDelta {meanDelta}")` (line 382):
    ```python
    self.reconstruction_error = ReconstructionError(
        max_abs=float(maxDelta), mean_abs=float(meanDelta)
    )
    ```
  - Docstrings: class `Attributes:` and `run()` Side Effects.
  Done when: 1.8 green.

- [ ] **1.10 Docs for PR 1.**
  - `docs/user_guide/compressing.md`: settings table (line 40-49) add
    `| seed | 12345 | Torch seed for the random initial deltas and weights. Change it to explore other local minima. |`;
    CLI option list (line 30-31) add `--seed`; rewrite "Reproducibility"
    (line 150-156): the seed is an argument; same seed + same platform =
    identical file; different seeds = different but equally valid solutions
    worth comparing; "After the run" (line 172-175): add
    "`compressor.reconstruction_error` holds the same two numbers as
    `max_abs` and `mean_abs`."
  - `docs/concepts/how_the_solver_works.md` line 93-96: the seed is the
    `seed` argument, default 12345.
  - `docs/user_guide/evaluating_results.md` after the headline table: one
    sentence on `reconstruction_error`.
  - `tests/test_data/macos/SETUP.md` and `tests/test_data/windows/SETUP.md`:
    seed row → `12345 (default of the seed argument of SkinCompressor)`.
  - `CLAUDE.md` Key Parameters table: add `| seed | 12345 | Torch seed for the random initialisation |`.
  - `cd docs && make clean && make html` → no warnings; open
    `docs/_build/html/api/model_fit.html` and check `ReconstructionError` is
    listed.
  Done when: build clean, pages read correctly.

- [ ] **1.11 Checks and commits.**
  - `ruff format . && ruff check . && mypy src/ --ignore-missing-imports`
  - `pytest -k "not default_output and not vertex_positions" -q`
  - `pytest -k short_iter -v` — must pass with no change under `tests/test_data/`.
  - Commits, e.g.:
    - `feat(model_fit): expose the random seed as a constructor setting`
    - `feat(cli): forward --seed to the compressor and the Maya pipeline`
    - `feat(model_fit): keep the final reconstruction error on the compressor`
    - `docs: describe the seed setting and reconstruction_error`
  - Open PR 1 against `main`. Description: what/why, AC1–AC3, AC8, AC9 status.

---

## 2. PR 2 — sparsity annealing (`feature/sparsity-annealing`)

Branch from `main` after PR 1 is merged (or stack on
`feature/seed-parameter` and rebase later).

- [ ] **2.0 Branch.** `git checkout -b feature/sparsity-annealing main`.

- [ ] **2.1 Test: `TrainingPhase`.**
  New file `tests/test_training_schedule.py`.
  ```python
  import pytest
  from metacompskin.model_fit import TrainingPhase, build_training_schedule


  class TestTrainingPhase:
      def test_holds_its_values(self):
          phase = TrainingPhase(
              iterations=10, max_influences=8, total_nnz_B_rt=6000, normalize_weights=True
          )

          assert (phase.iterations, phase.max_influences, phase.total_nnz_B_rt) == (
              10,
              8,
              6000,
          )
          assert phase.normalize_weights is True

      @pytest.mark.parametrize(
          "field", ["iterations", "max_influences", "total_nnz_B_rt"]
      )
      def test_rejects_counts_below_one(self, field):
          values = dict(
              iterations=10,
              max_influences=8,
              total_nnz_B_rt=6000,
              normalize_weights=False,
          )
          values[field] = 0

          with pytest.raises(ValueError, match=field):
              TrainingPhase(**values)
  ```
  Done when: fails with `ImportError`.

- [ ] **2.2 Implement `TrainingPhase`.** `src/metacompskin/model_fit.py`,
  design.md §2 verbatim. Done when: 2.1 green.

- [ ] **2.3 Test: `build_training_schedule`.** Same file, new class.
  ```python
  class TestBuildTrainingSchedule:
      def _phases(self, schedule):
          return [
              (p.iterations, p.max_influences, p.total_nnz_B_rt, p.normalize_weights)
              for p in schedule
          ]

      def test_one_stage_gives_the_classic_two_phases(self):
          schedule = build_training_schedule((100,), 8, 6000)

          assert self._phases(schedule) == [(100, 8, 6000, False), (100, 8, 6000, True)]

      def test_four_stages_halve_the_influences_after_a_warm_up(self):
          schedule = build_training_schedule((10, 20, 30, 40), 8, 6000)

          assert self._phases(schedule) == [
              (10, 64, 6000, False),
              (10, 64, 6000, True),
              (20, 32, 6000, True),
              (30, 16, 6000, True),
              (40, 8, 6000, True),
          ]

      def test_rejects_no_stages(self):
          with pytest.raises(ValueError, match="at least one stage"):
              build_training_schedule((), 8, 6000)

      def test_rejects_a_stage_without_iterations(self):
          with pytest.raises(ValueError, match="iterations"):
              build_training_schedule((100, 0), 8, 6000)
  ```
  Done when: fails with `ImportError` on `build_training_schedule`.

- [ ] **2.4 Implement `build_training_schedule`** — design.md §3 verbatim.
  Done when: 2.3 green.

- [ ] **2.5 Test: compressor builds and exposes the schedule.**
  `tests/test_skin_compressor.py`, new class `TestSkinCompressorSchedule`.
  ```python
  def _phases(self, compressor):
      return [(p.iterations, p.max_influences) for p in compressor.schedule]


  def test_default_schedule_is_two_phases_at_the_default_budgets(self, grid_model_data):
      compressor = SkinCompressor(model_data=grid_model_data)

      assert compressor.stage_iterations == (10000,)
      assert [
          (p.max_influences, p.total_nnz_B_rt, p.normalize_weights)
          for p in compressor.schedule
      ] == [(8, 6000, False), (8, 6000, True)]
      assert all(p.iterations == 10000 for p in compressor.schedule)


  def test_a_sequence_of_iterations_halves_the_influences_per_stage(
      self, grid_model_data
  ):
      compressor = SkinCompressor(
          model_data=grid_model_data, iterations=(100, 100, 300), max_influences=2
      )

      assert compressor.stage_iterations == (100, 100, 300)
      assert self._phases(compressor) == [(100, 8), (100, 8), (100, 4), (300, 2)]
      assert compressor.max_influences == 2


  def test_an_int_and_a_one_entry_sequence_build_the_same_schedule(self, grid_model_data):
      by_int = SkinCompressor(model_data=grid_model_data, iterations=300)
      by_tuple = SkinCompressor(model_data=grid_model_data, iterations=(300,))

      assert by_int.schedule == by_tuple.schedule


  @pytest.mark.parametrize("iterations", [(), 0, (100, 0)])
  def test_invalid_iterations_are_rejected(self, grid_model_data, iterations):
      with pytest.raises(ValueError, match="iterations"):
          SkinCompressor(model_data=grid_model_data, iterations=iterations)
  ```
  Done when: fails with `AttributeError: stage_iterations` / `schedule`.

- [ ] **2.6 Implement constructor changes.** design.md §3 (`_as_stage_iterations`)
  and §4.1: widened `iterations`, `self.stage_iterations`, `self.schedule`,
  docstring including the two `Example:` snippets. `test_defaults` and
  `test_settings_are_taken_from_the_constructor` must stay green untouched.
  Done when: 2.5 green and the whole file green.

- [ ] **2.7 Test: run applies the schedule and checks bounds.**
  Same class. Grid model: S=3, `number_of_bones=10` → K < 10, L ≤ 180.
  ```python
  def test_run_with_staged_iterations_ends_at_max_influences(
      self, grid_model_data, tmp_path
  ):
      compressor = SkinCompressor(
          model_data=grid_model_data,
          iterations=(100, 100, 300),
          number_of_bones=10,
          max_influences=2,
          total_nnz_B_rt=100,
      )

      compressor.run(tmp_path / "compressed.npz")

      weights = np.load(tmp_path / "compressed.npz")["weights"]
      assert ((weights != 0).sum(axis=1) <= 2).all()
      assert (weights >= 0).all()
      np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)


  def test_run_executes_a_schedule_assigned_before_it(self, grid_model_data, tmp_path):
      compressor = SkinCompressor(
          model_data=grid_model_data,
          iterations=300,
          number_of_bones=10,
          total_nnz_B_rt=100,
      )
      compressor.schedule = (
          TrainingPhase(300, 5, 100, normalize_weights=False),
          TrainingPhase(300, 3, 100, normalize_weights=True),
      )

      compressor.run(tmp_path / "compressed.npz")

      weights = np.load(tmp_path / "compressed.npz")["weights"]
      assert ((weights != 0).sum(axis=1) <= 3).all()


  def test_run_rejects_a_warm_up_not_below_the_bone_count(
      self, grid_model_data, tmp_path
  ):
      compressor = SkinCompressor(
          model_data=grid_model_data,
          iterations=(100,) * 4,
          number_of_bones=10,
          max_influences=2,
          total_nnz_B_rt=100,  # warm-up K = 16
      )

      with pytest.raises(ValueError, match="number_of_bones"):
          compressor.run(tmp_path / "compressed.npz")


  def test_run_rejects_a_coefficient_budget_above_the_model_size(
      self, grid_model_data, tmp_path
  ):
      compressor = SkinCompressor(
          model_data=grid_model_data, number_of_bones=10, total_nnz_B_rt=500
      )

      with pytest.raises(ValueError, match="6\\*S\\*P"):
          compressor.run(tmp_path / "compressed.npz")


  def test_run_rejects_an_empty_schedule(self, grid_model_data, tmp_path):
      compressor = SkinCompressor(
          model_data=grid_model_data, number_of_bones=10, total_nnz_B_rt=100
      )
      compressor.schedule = ()

      with pytest.raises(ValueError, match="at least one"):
          compressor.run(tmp_path / "compressed.npz")
  ```
  (Import `TrainingPhase` at the top of the test file.) Done when: the
  first two fail inside `train` (`TypeError` about `phase`/`normalizeW`),
  the others fail with a `torch` error or no error.

- [ ] **2.8 Implement `run()` loop, `_check_schedule_fits_model`, `train(phase)`.**
  design.md §4.2–4.3. Make **only** the four substitutions listed in §4.3
  inside `train`; do not reorder or reformat any tensor expression. Update
  `run()` and `train()` docstrings. Done when: 2.7 green;
  `pytest tests/test_pipeline_smoke.py -q` green.

- [ ] **2.9 Bit-exactness gate.** `pytest -k short_iter -v` on the platform
  with expected data. Done when: green with `git status` showing nothing
  under `tests/test_data/`. If it fails: `git diff main -- src/metacompskin/model_fit.py`
  and check `train()`; only the four substitutions may differ. Never
  regenerate the expected files for this PR.

- [ ] **2.10 Test: CLI staged `--iterations`.** `tests/test_cli.py`.
  ```python
  def test_cli_accepts_iterations_per_stage(grid_model_npz, tmp_path):
      output = tmp_path / "compressed.npz"

      main(
          [
              str(grid_model_npz),
              str(output),
              "--iterations",
              "100,100,300",
              "--total-nnz-b-rt",
              "100",
              "--number-of-bones",
              "10",
              "--max-influences",
              "2",
          ]
      )

      weights = np.load(output)["weights"]
      assert ((weights != 0).sum(axis=1) <= 2).all()


  def test_cli_rejects_malformed_iterations(grid_model_npz, tmp_path):
      with pytest.raises(SystemExit):
          main([str(grid_model_npz), str(tmp_path / "o.npz"), "--iterations", "3,x"])
  ```
  The existing tests keep passing `--iterations 300` through `_FAST`; they
  must stay green. Done when: the first fails with `SystemExit: 2`
  ("invalid int value: '100,100,300'").

- [ ] **2.11 Implement CLI parsing.** design.md §5: `_int_list`, the
  `--iterations` type change, `--seed` already done in PR 1. Done when:
  `pytest tests/test_cli.py -q` green.

- [ ] **2.12 Test: pipeline formats staged iterations.**
  `tests/test_maya_pipeline.py`: in `test_runs_the_package_cli_with_every_setting`
  set `iterations=(500, 500, 1000)` and expect `"--iterations", "500,500,1000"`
  at the start of the option list. `test_defaults_only_pass_the_iteration_count`
  must keep expecting `["--iterations", "10000"]`. Done when: fails on the
  formatted value.

- [ ] **2.13 Implement pipeline formatting.** design.md §6: widen the
  field and the `compress_and_build_rig` parameter, add `_format_ints`.
  Done when: `pytest tests/test_maya_pipeline.py -q` green.

- [ ] **2.14 Docs for PR 2.** design.md §9:
  - `compressing.md`: rewrite the `iterations` row:
    `| iterations | 10 000 | Steps per stage. An int is one stage (the classic two-phase run). A sequence such as (5000, 5000, 10000) runs one stage per entry, halving the influence budget each stage from $K \cdot 2^{N-1}$ down to $K$; the unnormalised warm-up takes the first entry's length. Needs $K \cdot 2^{N-1} < P$. Make the last stage the longest. |`;
    `schedule` row in the "attributes you can change" table with the
    hand-built example from design.md §4.1; CLI option list; new section
    "Annealing the influence budget" (what a sequence does, N+1 phases,
    total steps `T_0 + ΣT_i`, when to try it — worst-case error, speckled weight
    maps, results that vary between seeds — and an example command);
    "Choosing settings" bullet pointing there.
  - `how_the_solver_works.md`: replace "Two phases" with "Training phases"
    (table from spec §7.1; note a fresh Adam per phase); "Reading the log"
    add the `phase i/n:` header; settings table: `iterations` row rewritten.
  - `CLAUDE.md` Key Parameters: note on `iterations`.
  - `cd docs && make clean && make html` → no warnings; API page shows
    `TrainingPhase` and `build_training_schedule`, no private names.
  Done when: build clean.

- [ ] **2.15 Checks and commits.**
  - full `/pre-commit-checks`; `pytest -k short_iter -v` again.
  - Commits, e.g.:
    - `feat(model_fit): add TrainingPhase and build_training_schedule`
    - `feat(model_fit): anneal the influence budget over staged iterations`
    - `feat(cli): accept comma-separated --iterations and forward it from Maya`
    - `docs: describe influence annealing`
  - Open PR 2. Description: AC1, AC2, AC4–AC8, AC11 status, and the line
    "default path bit-identical; no regression data touched".

---

## 3. PR 3 — experiment script (`feature/schedule-experiment`)

- [ ] **3.0 Branch** from `main` after PR 2 is merged.

- [ ] **3.1 pytest can import scripts.** `pyproject.toml`
  `[tool.pytest.ini_options]`: `pythonpath = ["src", "scripts"]`.
  Done when: `pytest -q -k nothing` still runs.

- [ ] **3.2 Test: pure helpers.** New `tests/test_compare_schedules.py`.
  ```python
  import numpy as np
  import pytest
  from compare_schedules import (
      RunResult,
      build_parser,
      build_variants,
      format_summary,
      iterations_per_phase,
      summarize,
      write_csv,
  )


  def test_iterations_per_phase_splits_the_total():
      assert iterations_per_phase(40000, 2) == 20000
      assert iterations_per_phase(40000, 4) == 10000


  def test_iterations_per_phase_rejects_totals_below_one_per_phase():
      with pytest.raises(ValueError):
          iterations_per_phase(3, 4)


  def test_build_variants_gives_every_variant_the_same_total_steps():
      variants = {v.name: v for v in build_variants(40000, 8, 10000, 25000, 3)}

      assert set(variants) == {"baseline", "baseline_control", "anneal_k", "anneal_kl"}
      assert all(
          sum(p.iterations for p in v.schedule) == 40000 for v in variants.values()
      )
      assert variants["baseline"].phases == 2
      assert variants["anneal_k"].phases == 4


  def test_build_variants_stages():
      variants = {v.name: v for v in build_variants(40000, 8, 10000, 25000, 3)}

      assert variants["anneal_k"].influence_stages == "32,16,8"
      assert variants["anneal_k"].nnz_stages == "10000,10000,10000"
      assert (
          variants["anneal_kl"].nnz_stages == "25000,20000,10000"
      )  # 4L capped at n_coefficients
      assert variants["baseline_control"].influence_stages == "8,8,8"


  def _result(variant, seed, max_abs, mean_abs):
      return RunResult(
          variant, seed, 2, 10, 20, "8", "100", max_abs, mean_abs, 1.0, "cpu", "x"
      )


  def test_summarize_reports_mean_std_min_max_per_variant():
      results = [
          _result("a", 1, 1.0, 0.1),
          _result("a", 2, 3.0, 0.3),
          _result("b", 1, 5.0, 0.5),
      ]

      summary = {s.variant: s for s in summarize(results)}

      assert summary["a"].runs == 2
      assert summary["a"].max_abs_mean == pytest.approx(2.0)
      assert summary["a"].max_abs_std == pytest.approx(np.std([1.0, 3.0], ddof=1))
      assert (summary["a"].max_abs_min, summary["a"].max_abs_max) == (1.0, 3.0)
      assert summary["b"].max_abs_std == 0.0


  def test_format_summary_has_one_line_per_variant():
      lines = format_summary(
          summarize([_result("a", 1, 1.0, 0.1), _result("b", 1, 2.0, 0.2)])
      ).splitlines()

      assert any(line.startswith("a") for line in lines)
      assert any(line.startswith("b") for line in lines)


  def test_write_csv_round_trips(tmp_path):
      path = tmp_path / "results.csv"

      write_csv([_result("a", 1, 1.0, 0.1)], path)

      text = path.read_text()
      assert text.splitlines()[0].startswith("variant,seed,phases")
      assert "a,1,2,10,20,8,100,1.0,0.1" in text


  def test_parser_defaults():
      args = build_parser().parse_args(["m.npz", "out"])

      assert args.seeds == (1, 2, 3, 4, 5)
      assert args.total_iterations == 40000
      assert args.anneal_stages == 3  # script-only convenience for the experiment
      assert args.variants is None and args.dry_run is False
  ```
  Done when: fails with `ModuleNotFoundError: compare_schedules`.

- [ ] **3.3 Implement the helpers.** New `scripts/compare_schedules.py`:
  module docstring (purpose, usage, fairness rule), dataclasses `Variant`,
  `RunResult`, `VariantSummary`, and `iterations_per_phase`,
  `build_variants`, `summarize`, `format_summary`, `write_csv`,
  `build_parser` per design.md §7. A local `_int_list("1,2,3")` parses
  `--seeds`. Done when: 3.2 green, `ruff check scripts/` clean.

- [ ] **3.4 Implement `run_variant` and `main`.** design.md §7: loop seeds ×
  variants, set `compressor.schedule = variant.schedule` before `run()`,
  append to CSV as each run finishes, write `summary.txt`, print the table,
  honour `--dry-run`, `--variants`, `--ignore-joint-matrices`. Done when:
  `python scripts/compare_schedules.py tests/test_data/source_models/aura.npz /tmp/x --dry-run`
  prints 4 variants × 5 seeds with per-phase iteration counts and K/L stages
  and creates no files.

- [ ] **3.5 Smoke run on the sample head.** Small on CPU:
  `python scripts/compare_schedules.py tests/test_data/source_models/aura.npz out_smoke --seeds 1,2 --total-iterations 1200 --number-of-bones 40 --total-nnz-b-rt 6000 --variants baseline,anneal_kl`
  (cap = 6·267·40 = 64 080; K stages 32, 16, 8 with P=40; L stages 24 000,
  12 000, 6 000). Done when: `out_smoke/results.csv` has 4 rows, the summary
  prints, every `max_abs` is finite.

- [ ] **3.6 Docs.** `docs/user_guide/compressing.md` "Batch processing": a
  paragraph "Comparing schedules and seeds" with the command from spec §6.5
  and two sentences on reading the table (research.md §8). Done when: docs
  build clean.

- [ ] **3.7 Checks and commit.** `ruff format . && ruff check .`,
  `mypy src/ --ignore-missing-imports`, and also
  `mypy scripts/compare_schedules.py --ignore-missing-imports` (fix what it
  reports even though CI does not gate on it), fast pytest.
  `feat(scripts): add the schedule-versus-seed comparison experiment`. Open PR 3.

---

## 4. Run the experiment and record the result

Real-data work lives in the private companion repository
`meta-compskin_private_tests` (a sibling checkout of this one), never here:
this repository only ships the small sample heads under
`tests/test_data/source_models/`, which are too small to be informative.
That repository provides:

- `scripts/run_schedule_experiment.py` — loads its Maya OBJ fixtures through
  mayapy once (cached as `<out-dir>/model.npz`), then runs this repository's
  `scripts/compare_schedules.py` on that model, forwarding every other flag.
- `tests/test_schedule_experiment.py` — a `slow`-marked end-to-end smoke
  test of the same path at a tiny step count (`pytest -m slow`).

Run the steps below from that repository. Fill in `<P>`, `<K>`, `<L>`,
`<ALPHA>` with the number-of-bones, max-influences, total-nnz-b-rt and alpha
the fixture model is normally compressed with (or omit them for the package
defaults).

- [ ] **4.1 Dry run.**
  `python scripts/run_schedule_experiment.py --number-of-bones <P> --max-influences <K> --total-nnz-b-rt <L> --alpha <ALPHA> --power 2 --seeds 1,2,3,4,5 --total-iterations 40000 --anneal-stages 3 --dry-run`
  (the first call builds the model cache through mayapy, a few minutes).
  Check: 4 variants printed; `baseline` gets half the total steps per phase,
  the other three a quarter each; `anneal_k`'s K stages are `4*<K>`, `2*<K>`,
  `<K>`; `anneal_kl`'s L stages are `min(cap, 4*<L>)`, `min(cap, 2*<L>)`,
  `<L>` where `cap = 6 * n_blendshapes * <P>`. If the model file carries
  `rest_joint_matrices` for a different joint count, add
  `--ignore-joint-matrices` (affects joint placement metadata only, not the
  fit — `docs/user_guide/compressing.md` "Custom joints").
- [ ] **4.2 Full run.** Same command without `--dry-run`. Wall time scales
  with total steps and the hardware's steps/second (research.md §9); the
  dry run's printed step counts let you estimate it from a short timed run
  first if that matters. Keep the terminal output.
- [ ] **4.3 Write `experiments/schedule_experiment/results.md`** in the
  private repository, next to the `results.csv` and `summary.txt` the run
  produced there. Include: the exact command, torch version and device, the
  summary table, and the verdict using research.md §8:
  1. seed spread of `baseline` (std and max−min of `max_abs`);
  2. `baseline_control` vs `baseline` (phase-count effect alone);
  3. `anneal_k` and `anneal_kl` vs `baseline_control` (annealing effect);
  4. recommendation: adopt a three-stage `iterations` sequence as the
     default setting for this model, keep best-of-N seeds only, or
     neither — with the numbers.
- [ ] **4.4 Follow-ups (only if warranted, each its own ticket):** public
  L annealing; longer final phase; finer tail (32, 16, 12, 8); `power=4` on
  the winning schedule; residual-Laplacian regulariser; more seeds.

## Definition of done (whole plan)

All acceptance criteria in spec.md §8 hold; three PRs merged; the
experiment's `results.md` committed in the private repository;
`/update-docs` checklist completed for PR 1 and PR 2; nothing under
`tests/test_data/` modified.

## If you get stuck

- `short_iter` fails after PR 2 work → you changed numerics; compare `train()`
  with `git diff main -- src/metacompskin/model_fit.py`; only the four
  substitutions in design.md §4.3 may differ.
- `torch.topk` complains about `k` → a K ≥ P or L > 6·S·P slipped past the
  check; the check must run first thing in `run()` (design.md §4.2).
- A new test passes immediately when it should fail → you implemented
  before testing, or the test is not exercising the new behaviour.
- Sphinx warns "duplicate object description" → `napoleon_use_ivar` in
  `docs/conf.py` must stay `True`.
- Three failed attempts at anything → stop, write down what you tried, ask.
