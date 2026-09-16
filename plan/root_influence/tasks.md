# Tasks — R0 warm-up budget and R1 root influence

Revision 2, against commit `12ac666`. *Changes from Revision 1 (self-review
Q9):* the existing tests coupled to "P equals the number of matrices" are
now named per task instead of being discovered during implementation, and
the idle-joint report excludes the root. Read [spec.md](spec.md) and
[design.md](design.md) first; [research.md](research.md) has the reasoning.
One commit per task; each follows the repo's TDD law (failing test, minimal
code, refactor). Conventional-commit types are suggested per task.

## Before you start

- Environment: `pip install -e ".[dev]"` (or the repo's `.venv`); ruff and
  mypy at their newest release (`ruff>=0.16.6`).
- Fast test loop (the full suite exceeds ten minutes on a Mac):

  ```
  pytest -q --no-cov -k "not full_iter and (short_iter or not default_output)"
  ```

  Run it once before touching anything and keep it green after every task.
  The pinned anonymous-path guard is
  `tests/test_default_output_macos.py::TestCompskin::test_default_output_short_iter`
  (or the Windows twin); it must pass unchanged through every task.
- Read: `src/metacompskin/model_fit.py` lines 224–360 (schedule, candidate
  resolution), 467–626 (constructor), 627–800 (`run`), 975–1121 (`train`);
  `plan/research_output.md` §2 and §5.0–5.1; paper §4 (projections);
  `tests/test_skin_compressor.py` (`TestCandidateJoints` shows the grid-model
  patterns and the `_FAST` settings); `tests/conftest.py::make_grid_model`.
- Branch: `feature/root-influence`.

## Tasks

### 1. R0: the warm-up runs at the mask width for a single stage — `feat(model_fit)`

- [ ] In `tests/test_training_schedule.py` replace
      `test_start_influences_is_ignored_by_a_single_stage` with
      `test_start_influences_sets_the_warm_up_even_for_a_single_stage`
      expecting `[(10, 16, 6000, False), (10, 8, 6000, True)]`; keep the
      multi-stage tests as they are. Run: fails.
- [ ] In `tests/test_skin_compressor.py::TestCandidateJoints` add
      `test_single_stage_warm_up_uses_the_candidate_count`: with 40 identity
      joint matrices and `iterations=300`, `[p.max_influences for p in
      compressor.schedule] == [16, 8]`. Run: fails.
- [ ] Change the warm-up construction in `build_training_schedule`
      (`model_fit.py:270-276`) to use `start_influences` when given; update
      the docstring sentence "A single stage keeps max_influences throughout
      whatever the ceiling." Run: both pass.
- [ ] Done when: the two tests pass, `test_caps_the_annealing_ceiling` still
      gives `[6, 6, 3, 2]`, the anonymous schedule tests are untouched, and
      the fast loop is green.

### 2. Extract the projections from `train` (pure refactor) — `refactor(model_fit)`

- [ ] Add `_project_weights(W, phase)` and `_project_transforms(B_rt, phase)`
      per `design.md` §2.5 with the existing steps in the existing order, and
      call them from the `no_grad` block (`model_fit.py:1089-1104`). No new
      behaviour.
- [ ] Add a unit test in `tests/test_skin_compressor.py` (`TestProjections`)
      that builds a compressor on the grid model, calls `_project_weights`
      on a hand-made `(P, N)` tensor with a `TrainingPhase(1, 2, 100, True)`
      and asserts at most two positive entries per column and no negatives;
      same for `_project_transforms` keeping exactly `L` largest magnitudes.
- [ ] Done when: the pinned short-iteration regression test passes
      bit-for-bit and the fast loop is green.

### 3. `root_joint` resolution — `feat(model_fit)`

- [ ] Tests (`TestRootJoint`) for `_resolve_root_joint` through the
      constructor: default with joint matrices appends (`root_joint == P`,
      `number_of_bones == P + 1`); default without matrices gives `None` and
      `100`; `"none"` with matrices gives `None` and `P`; `"new"` without
      matrices gives `100` and `101`; `3` with matrices gives `3` and `P`;
      `P`, `-1`, `True`, `"head"` raise `ValueError` naming `root_joint`.
      Run: fail (unknown argument).
- [ ] Update the three existing tests in `TestSkinCompressorSettings`
      (`tests/test_skin_compressor.py:64-92`) that assert
      `number_of_bones == 7` for seven matrices: with the new default the
      count is `8`; keep one variant with `root_joint="none"` expecting `7`.
      The conflict check (`number_of_bones=5` versus seven matrices) is
      unchanged: `number_of_bones`, when given, must equal the number of
      matrices, and the synthetic root is added afterwards; say so in the
      docstring.
- [ ] Implement `_resolve_root_joint`, the constructor argument, docstring
      (`Args`, `Raises`) and attributes; pass the number of geodesic joints
      (the caller's $P$) to `_resolve_candidate_joints`. Add a test that the
      default `candidate_joints_per_vertex` is still `min(2K, P - 1)` over
      the caller's joints when a root is appended.
- [ ] Done when: tests pass; `SkinCompressor(...).run` still works for the
      anonymous path (fast loop green).

### 4. Root transforms are zero and the rest transform is written — `feat(model_fit)`

- [ ] Tests: after `run` with a synthetic root, `restXform` has shape
      `(P + 1, 3, 4)` and its last row is `np.eye(3, 4)`; with a designated
      root, `restXform` equals the caller's matrices; in both cases every
      `shapeXform` block of the root (columns `4r:4r+4`, all shapes) is
      exactly zero. Run: fail.
- [ ] Implement `_build_transform_mask`, apply it at initialisation and in
      `_project_transforms`; implement `_rest_transforms_for_output` and use
      it in `np.savez`. Update the `run` docstring's workflow list and output
      description.
- [ ] Done when: tests pass; fast loop green.

### 5. The root is a universal candidate — `feat(model_fit)`

- [ ] Test: with joint positions on the grid, `M = 3`, a synthetic root, after
      `run` the compressor's `candidate_mask` has an all-true root row and
      exactly three true entries per column elsewhere; with a designated root
      its row is all true. Run: fail.
- [ ] Extend `_build_candidate_mask` per `design.md` §2.4.
- [ ] Update `test_run_only_weights_joints_near_each_vertex`
      (`tests/test_skin_compressor.py:348-368`) and
      `test_cli_passes_the_candidate_joint_count` (`tests/test_cli.py:38-66`):
      both compare the `(P, N)` geodesic mask against `weights.T`, which now
      has `P + 1` rows; assert on the non-root rows (`weights.T[:-1]`) so the
      locality guarantee is still tested with the root present.
- [ ] Done when: tests pass.

### 6. The root weight has a floor — `feat(model_fit)`

- [ ] Tests on `_project_weights`: a column whose non-root entries are all
      negative ends with the root at exactly `_ROOT_WEIGHT_FLOOR` and every
      other entry zero, so the column sum is positive; a column with two
      strong joints and the root at the floor keeps the two joints and the
      root at the floor (the cut dropped it, the floor restored it); the
      initial `W` built by `run` has the root at the floor (check
      `compressor.candidate_mask`-independent path by inspecting a
      one-iteration run's saved weights for a positive root share on a
      vertex whose other candidates are empty). Run: fail.
- [ ] Implement the floor in `_project_weights` and the root initialisation
      in `run`.
- [ ] Done when: tests pass; fast loop green.

### 7. Final weights and validation — `feat(model_fit)`

- [ ] Tests on `_final_weights` with hand-made tensors (`design.md` §4):
      floor-level root beside live joints is dropped; floor-level root alone
      normalises to one; a root above the floor is kept; every column sums to
      one. Tests on `_validate_weights`: `NaN`, a negative entry, a row
      summing to 0.9, and `K + 1` non-zeros each raise `RuntimeError`
      naming the failed check. Run: fail.
- [ ] Implement both, call them in `run` before `np.savez` (validation runs
      on both paths); add `RuntimeError` to the `run` docstring's `Raises`.
- [ ] Done when: tests pass; the smoke tests and the pinned short test pass.

### 8. Integration: a static region lands on the root, and without a root the run refuses to save — `test(model_fit)`

- [ ] Build in the test a grid model with `alpha=0.0` whose two rightmost
      columns of vertices have zero deltas in every shape, six joints placed
      on the moving side, `max_influences=2`, `candidate_joints_per_vertex=3`,
      `total_nnz_B_rt=100`, schedule warm-up 600 + stage 300 (`iterations=600`
      then override `compressor.schedule` for the shorter stage, or use
      `iterations=(600,)` and accept 600 + 600).
- [ ] Test A (root, default): `run` succeeds, the saved weights pass the
      LBS invariants, and every vertex in the rightmost column has weight one
      on the root.
- [ ] Test B (`root_joint="none"`): `run` raises `RuntimeError` from
      `_validate_weights` (the reproduction of the collapse). If the collapse
      does not reproduce at this budget, lengthen the warm-up before
      weakening the assertion; do not skip it.
- [ ] Done when: both pass in under about thirty seconds combined on CPU.

### 9. Reports: empty columns in the progress line, root usage in the coverage report — `feat(model_fit)`

- [ ] Tests with `capsys`: the progress line ends with the empty-column
      count (zero with a root); with a root the log contains
      `vertices on the root: <n> full, <m> partial, largest delta on root
      <x>` (exact wording per the implementation, asserted by prefix).
- [ ] Implement `_report_root_usage(Wn, A)` and the extra progress number.
      `_report_joint_coverage` must skip the root in its idle-joint list (an
      idle root is expected when nothing is static), which means it needs
      the root index: make it an instance method. Add a test that a root
      driving no vertex is not listed.
- [ ] Done when: tests pass; `test_run_reports_joints_driving_no_vertex`
      (`tests/test_skin_compressor.py:370-384`) still reports exactly `[9]`
      with the root appended as joint 10.

### 10. CLI — `feat(cli)`

- [ ] Tests in `tests/test_cli.py`: `--root-joint none` with stored joint
      matrices gives `weights.shape[1] == P`; the default gives `P + 1`;
      `--root-joint 0` gives `P` with a zero `shapeXform` block for joint 0;
      `--root-joint head` fails parsing.
- [ ] Implement `_root_joint_arg` and the flag; add `"root_joint"` to
      `_compressor_settings`.
- [ ] Done when: tests pass.

### 11. Maya pipeline settings — `feat(maya_pipeline)`

- [ ] Tests in `tests/test_maya_pipeline.py`: `compression_command` emits
      `--root-joint 3` / `--root-joint none`; `compress_and_build_rig`'s
      name resolution (a pure helper `resolve_root_joint_name(root_joint,
      joints)`) maps `"head_JNT"` to its index in `joints`, passes ints and
      sentinels through, and raises `ValueError` for a name not in `joints`
      or when `joints` is `None`.
- [ ] Implement `CompressionSettings.root_joint`, the command, the helper
      and the `compress_and_build_rig` argument and docstring.
- [ ] Under `mayapy`, run `tests/test_maya_pipeline_in_maya.py` and
      `tests/test_maya_rig_builder_in_maya.py`. The pipeline test asserts
      `weights.shape[1] == 10` for ten exported joints
      (`tests/test_maya_pipeline_in_maya.py:73`); with the default root it
      is `11`, and the rig gains one static joint. Update the assertion and
      check the built rig has `P + 1` joints with the last one static across
      frames.
- [ ] Done when: tests pass on both interpreters.

### 12. Experiment driver passthrough — `feat(scripts)`

Justification: the R1-versus-none comparison on the private data (E2/E3
re-runs, `plan/research_output.md` §6.2) goes through
`scripts/compare_schedules.py`, which builds `SkinCompressor` itself.

- [ ] Test in `tests/test_compare_schedules.py`: the parser accepts
      `--root-joint none` and the settings carry `root_joint`.
- [ ] Implement with the same converter as the CLI.
- [ ] Done when: test passes.

### 13. Documentation — `docs`

- [ ] `docs/user_guide/compressing.md`: `root_joint` row in the settings
      table; `iterations` row notes the warm-up at $M$ with the filter; the
      custom-joints section gains a "Root joint" paragraph (what it is, the
      default, designating the head, what the rig builder does with it).
- [ ] `docs/concepts/how_the_solver_works.md`: projections gain the root
      floor and transform mask; training phases gain the $K_0 = M$ warm-up;
      the log-reading section documents the empty-column number and the
      root usage lines.
- [ ] `docs/concepts/data_formats.md`: the root is a bone with zero
      `shapeXform` blocks; `P` may be one larger than the matrices given.
- [ ] `docs/user_guide/troubleshooting.md`: rewrite the `nan` entry (empty
      weight columns after a long warm-up; the root fixes it; `RuntimeError`
      instead of a saved `nan`).
- [ ] `docs/user_guide/maya_rig_workflow.md` and
      `docs/user_guide/evaluating_results.md`: the static root joint and the
      new report lines.
- [ ] `CLAUDE.md`: add `root_joint` to the key-parameters table; note the
      warm-up budget in the `iterations` row.
- [ ] Run `/update-docs`; the API page is an `automodule`, so a rebuild is
      enough. Done when: `cd docs && make html` is clean and the pages read
      correctly.

## Pre-PR checklist

- [ ] `/pre-commit-checks`: `ruff format .`, `ruff check .`,
      `mypy src/ --ignore-missing-imports`.
- [ ] Full suite including the 10 000-iteration tests; the anonymous-path
      expected outputs must pass unchanged.
- [ ] Private repository (`../meta-compskin_private_tests`): update
      `tests/test_joint_matrices.py` (`number_of_bones == 29`,
      `restXform.shape == (29, 3, 4)`, or pass `root_joint="none"` where the
      old numbers are wanted) and re-run its fast tests. Regenerate no
      expected files for the anonymous path.
- [ ] On the private data: one single-stage run at the budget that
      previously diverged, with the default root; confirm it completes and
      read the root-usage lines. Record the result in
      `plan/experiment_results.md`.
- [ ] Conventional-commit messages; PR description links `plan/root_influence/`
      and `plan/research_output.md` §2.
