# Spec — warm-up at $K_0 = M$ (R0) and a zero-transform root influence (R1)

Revision 2, written against commit `12ac666`. *Changes from Revision 1:*
self-review Q9 found tests coupled to "P equals the number of matrices"
(three constructor tests, one CLI test, one in-Maya pipeline test) and an
idle-joint report that would list the root; `tasks.md` now names each, and
§4.3 states that the root is excluded from the idle list. Companion documents:
[design.md](design.md) (how), [tasks.md](tasks.md) (ordered TDD checklist),
[research.md](research.md) (why, and what was rejected). Background:
`plan/research_output.md` §2 and §5.0–5.1, `plan/additional_context.md`,
`plan/experiment_results.md`.

## 1. Summary

Two changes to `SkinCompressor` (`src/metacompskin/model_fit.py`), both
active only when the caller opts in or supplies joint matrices:

- **R0.** With the geodesic candidate filter on, the un-normalised warm-up
  runs at the mask width $M$ instead of $K$, for single-stage schedules as
  it already does for annealed ones. One line in `build_training_schedule`.
- **R1.** A *root influence*: one joint whose delta transforms are held at
  zero in every shape, which every vertex may use, and whose weight can
  never be pruned below a small floor. Static vertices and vertices that no
  candidate joint can explain end up on it instead of on an all-zero weight
  column, which is what currently turns into `NaN`. The root is either one
  of the caller's joints (`root_joint=<index>`, the head) or a synthetic
  identity joint appended as bone $P$ (`root_joint="new"`, the default when
  joint matrices are given).

Plus two guards that apply everywhere: `run` validates the weights before
saving (finite, non-negative, rows sum to one, at most $K$ non-zeros) and
raises instead of writing an invalid archive, and the progress line reports
how many weight columns are empty.

## 2. Background and motivation

Problem #6 of `plan/specs.md`: long runs on the joint path collapse to
`NaN`. The mechanism is confirmed (`plan/research_output.md` §2,
`plan/experiment_results.md`):

1. The un-normalised warm-up (`build_training_schedule`, line 270–276,
   `normalize_weights=False`) uses $W$ directly in the forward pass. For a
   vertex with zero deltas, or whose $M$ candidates cannot reproduce its
   deltas, every positive weight adds error, so Adam walks all of them to
   the clamp (line 1096).
2. The top-$K$ cut (line 1092–1095) zeroes the $M - K$ smallest candidates
   every step, small positive ones included, and a zeroed entry does not
   re-enter (`plan/gradual_sparsity_annealing/research.md` §3). A
   single-stage schedule warms up at $K$ with $M = 2K$ candidates, so half
   of every column is discarded each step and only $K$ survivors need to
   decay. Annealed schedules warm up at $K_0 = M$, where the cut keeps every
   positive candidate; they survived where single-stage runs of equal
   length died.
3. The first normalised iteration computes `W / W.sum(dim=0)` (line 1069)
   on an all-zero column, the `NaN` enters the shared $B_{rt}$ gradient and
   every parameter is `NaN` within one step. `run` then saves it (line 762,
   792).

Linear blend skinning has no way to say "this vertex does not move" except
weight on a joint whose transform is the identity in every frame. The
solver has no such joint. R1 adds one; R0 removes the cut from the
warm-up. Problem #3 (static vertices scattered over many joints) has the
same root cause and is addressed by the same change, though this plan only
claims the `NaN` fix and the "static vertices land on one joint" behaviour
that follows from it; the locality work (R2, R3) is separate.

## 3. Current behaviour (file:line at `12ac666`)

| What | Where |
|---|---|
| Warm-up budget is the first stage's budget; a single stage keeps $K$ "whatever the ceiling" | `model_fit.py:224-277`, warm-up built at 270–276; test `tests/test_training_schedule.py::test_start_influences_is_ignored_by_a_single_stage` pins it |
| Candidate count $M$ resolved against $P$ | `model_fit.py:316-358` |
| Joint matrices set $P$ and are echoed as `restXform` | `model_fit.py:546-582`, `790-796` |
| $W$ initialised as $10^{-8}$ noise, masked | `model_fit.py:744-752` |
| Projection block: mask, top-$K$ cut, clamp, top-$L$ on $B_{rt}$ | `model_fit.py:1089-1104` |
| Final weights `W / W.sum(dim=0)`, saved without validation | `model_fit.py:762`, `792-799` |
| Coverage report with joint matrices | `model_fit.py:846-861` |
| CLI flags, settings passthrough | `cli.py:60-98`, `132-155` |
| Maya pipeline settings and command | `maya_pipeline.py:76-110`, `129-215`, `430-456` |
| Rig builder makes one joint per weight column under a weightless `_root` | `maya_rig_builder.py:351-440`, `587-620`, `660-690` |
| Private tests assert `number_of_bones == 28` and `restXform.shape == (28, 3, 4)` with the 28-joint fixture | `../meta-compskin_private_tests/tests/test_joint_matrices.py:76-88` |

## 4. Proposed change

### 4.1 R0

`build_training_schedule(..., start_influences=M)` builds the warm-up at
`start_influences` when it is given, otherwise at the first stage's budget
as today. Annealed schedules are unchanged (their first stage already
equals the ceiling); the anonymous path is unchanged (`start_influences`
is `None` there). The single-stage joint path changes from phases
`(K, K)` to `(M, K)`.

### 4.2 R1

New constructor argument `root_joint: int | str | None = None`:

| Value | Meaning |
|---|---|
| `None` (default) | With `rest_joint_matrices`: same as `"new"`. Without: no root. |
| `"new"` | Append a synthetic root as bone $P$ with an identity rest transform; $P$ becomes $P + 1$. Allowed without joint matrices (anonymous opt-in). |
| `int` in `[0, P)` | That supplied joint is the root; $P$ unchanged; its `restXform` row is the caller's matrix. |
| `"none"` | No root, today's behaviour. |

Anything else (a `bool`, an out-of-range index, another string) raises
`ValueError`.

Solver semantics of the root, index $r$:

- Its $B_{rt}$ block is zero at initialisation and re-zeroed after every
  Adam step, before the top-$L$ projection, so it never costs $L$ budget
  and its `shapeXform` blocks are exactly zero.
- Its mask row is all true: every vertex has its $M$ geodesic candidates
  plus the root.
- Its weight starts at a floor $\varepsilon_r$ (module constant, $10^{-6}$)
  and, after every projection, is reset to at least $\varepsilon_r$. The
  cut may drop it; the floor restores it. A column's sum is therefore
  always positive and the division at line 1069 and line 762 cannot
  produce `NaN`.
- In the un-normalised warm-up the root has exactly zero gradient, so it
  sits at the floor while useless weights decay; at the first
  normalisation a vertex whose other weights all reached zero gets weight
  one on the root. In normalised phases the root receives the dilution
  gradient and can grow from the floor when the vertex's other joints are
  weak; it cannot displace a strong joint, because a floor-level entry
  cannot beat the $(K{+}1)$-th cutoff in one Adam step.

Save-time reconciliation, so the output respects $K$ exactly: a root entry
still at the floor in a column that has other non-zero weights is a
leftover of the floor, not a solved weight; it is zeroed before the final
normalisation. Every other root entry survived the last cut, so its column
already has at most $K$ non-zeros. See `design.md` §4 for the argument.

### 4.3 Guards (both paths)

- `run` validates the final weights (finite, $\ge 0$, row sums within
  $10^{-5}$ of one, at most $K$ non-zeros per vertex) and raises
  `RuntimeError` naming the first failing check instead of saving.
- The progress line gains the number of all-zero weight columns, the
  diagnostic `plan/research_output.md` §6 (E1) asked for. With a root it
  is always zero; with `root_joint="none"` it shows the collapse coming.
- With a root, the coverage report adds the number of vertices carried
  entirely by the root, the number carrying partial root weight, and the
  largest delta magnitude among the former (the residual the rigger pays
  for a vertex no joint could explain). The idle-joint list skips the root:
  a root that drives no vertex means nothing was static, which is not a
  rigger's concern.

### 4.4 Plumbing

- CLI: `--root-joint {auto,none,new,<int>}`; `auto` and an omitted flag
  leave the compressor's default.
- Maya pipeline: `CompressionSettings.root_joint` and
  `compress_and_build_rig(root_joint=...)`; a string that is neither
  `"none"` nor `"new"` is a joint name resolved against `joints`.
- `scripts/compare_schedules.py`: `--root-joint` passthrough, needed for
  the R1-versus-none experiment on the private data.
- Output format unchanged: the root is an ordinary bone with zero
  `shapeXform` blocks and an identity (synthetic) or caller (designated)
  `restXform` row. `AnimationFrameGenerator` and the Maya rig builder need
  no change; the builder places the synthetic root at the rest-space origin
  (the head centre) as a static joint.

## 5. Scope

In scope: everything in §4, tests, and the documentation pages that
describe phases, settings, the output format and the log
(`docs/user_guide/compressing.md`, `docs/concepts/how_the_solver_works.md`,
`docs/concepts/data_formats.md`, `docs/user_guide/troubleshooting.md`,
`docs/user_guide/maya_rig_workflow.md`, `docs/user_guide/evaluating_results.md`,
`CLAUDE.md` parameter table).

Out of scope, deliberately:

- Non-negative initial $W$ and normalised-from-step-one schedules (E3).
- The distance penalty, proximity initialisation, mask policy (R3), the
  hierarchical basis (R2), the weight Laplacian (R4).
- Binding the rig builder's `_root` node with the root's weights (user
  decision: the synthetic root is an ordinary static joint for now).
- A builder-side threshold snap of near-root vertices onto the root.
- Updating the private repository's tests. They assert $P = 28$ on the
  joint fixture; with the new default they will see $29$. The pre-PR
  checklist in `tasks.md` lists the change, but it lands in that
  repository.
- Any change to the anonymous path's default numerics. Its pinned
  regression outputs must pass unchanged.

## 6. Open questions

None blocking. Two soft choices are made in `design.md` and can be revisited
with data: the floor value ($10^{-6}$) and reporting thresholds in the
coverage report.

## 7. Self-review

1. **Goals.** Stop the `NaN` collapse on the joint path by construction
   (R1), remove the warm-up cut that accelerates it (R0), never save an
   invalid archive, and give static or unexplainable vertices a legal home
   on one joint. Every task in `tasks.md` maps to one of these.
2. **Constraints.** Hard: the anonymous path's default numerics are
   bit-identical (pinned tests); output weights satisfy the LBS
   constraints with at most $K$ non-zeros; the root's transforms are
   exactly zero in every shape; with a root, no all-zero weight column can
   exist. Soft: the floor value, the report wording, the CLI spelling, the
   choice to append the synthetic root last.
3. **Alternatives.** Six were weighed (`research.md` §3): an epsilon in
   the denominator (ships zero rows, rejected), a displacement threshold
   before the solve (misses unexplainable vertices, kept as a later
   builder-side snap), an exact sparse-simplex projection (changes every
   iteration's numerics, later), running the top-$K$ cut on non-root rows
   only (yields $K{+}1$ outputs that need lossy dropping, rejected in favour
   of "root competes, floor restores"), $K{-}1$ real joints plus root
   (capacity cost, kept as fallback), and rescuing dead columns onto the
   nearest moving joint (adds error, rejected).
4. **Literature.** `plan/research_output.md` §4.5–4.6 and §10 cover the
   simplex-projection, proximal-Adam and dead-unit literature; nothing
   further is needed for a change this small.
5. **Maintainability** improves: the projection block is extracted into
   two small methods with unit tests; the output gets a validator.
   **Extensibility** improves: R2/R3 need a root index and the extracted
   projections. **Testability** improves: projection, reconciliation and
   validation are testable on hand-made tensors. **Reliability** improves:
   invalid archives can no longer be written.
6. **Performance.** Neutral. R0 changes the cut's $k$, and `topk` scans
   the full tensor whatever $k$ is; R1 adds one row to $W$, a clamp and a
   multiply per step, and for a synthetic root one bone's worth of
   parameters.
7. **Quality.** A correctness bug that invalidates long runs is removed
   without touching the anonymous path, and the code gets more testable.
8. **Patterns.** "Zero it after every step" is reused from the candidate
   mask; it fits the solver's projection design and costs nothing. The
   `int | str | None` argument mirrors `candidate_joints_per_vertex`'s
   "None means automatic, a sentinel turns it off"; an enum would be
   cleaner for the two sentinels but would be the only enum in the public
   API, so strings are kept and `bool` is rejected explicitly.
9. **Scope.** The `compare_schedules.py` flag and the empty-column counter
   are adjacent diagnostics needed for the follow-up experiments and cost a
   few lines each; both are justified in `tasks.md`. Nothing else outside
   `model_fit.py`, its callers, tests and docs is touched. The revision-2
   pass checked every test that asserts a bone count or compares against
   the geodesic mask; each is named in the task that changes its premise,
   so no test is silently weakened or discovered mid-implementation.
