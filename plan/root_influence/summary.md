# Summary — warm-up at M (R0) and a zero-transform root influence (R1)

One-page digest of [research.md](research.md), [spec.md](spec.md),
[design.md](design.md) and [tasks.md](tasks.md). Read those for detail.

## Status

Implemented in commit `550c6c6` (tasks 1–13). The `tasks.md` boxes are
unticked, so that file is stale. Not verified here: the plan's pre-PR
checklist, and that the warm-up change (R0) behaves exactly as specified. A
quick look at that commit's diff would settle both.

## Reasoning

- Long runs on the joint path (caller-supplied joints) collapse to `NaN`. The
  un-normalised warm-up drives the weights of a static or unexplainable
  vertex to zero. The top-K cut then zeroes small positive candidates every
  step and they do not re-enter. The first normalised step divides by an
  all-zero column sum, and the `NaN` spreads to every parameter in one step.
  The archive is saved anyway.
- Annealed schedules survived longer only because their warm-up runs at
  K0 = M, where the cut keeps every candidate.
- The deeper cause is representational. A vertex that does not move is fit
  exactly by zero weights, but linear blend skinning cannot ship a zero row.
  The only legal way to say "does not move" is weight one on a joint whose
  transform is the identity in every frame, and the solver had no such joint.
- Alternatives rejected:
  - an epsilon in the denominator (ships zero rows);
  - a displacement threshold before the solve (misses unexplainable
    vertices; it stays a possible later builder-side snap);
  - rescuing dead columns onto the nearest joint (adds error);
  - exact sparse-simplex projection or sparsemax (changes every pinned
    output);
  - a top-K cut on non-root rows only (yields K+1 entries that need lossy
    dropping);
  - K−1 real joints plus a root (kept as a fallback);
  - a root exempt from the cut but with no floor (Adam outgrows it and the
    collapse returns).
- The floor is 1e-6: well below one Adam step (about 1e-3) and still
  representable next to sums of order one in float32 (1e-8 is not).

## Spec

- **R0.** With the geodesic candidate filter on, the warm-up runs at the
  mask width M even for a single-stage schedule. It is one change in
  `build_training_schedule` via `start_influences`. The anonymous path is
  unchanged.
- **R1.** New `root_joint` argument:

  | Value | Meaning |
  |---|---|
  | `None` | `"new"` with joint matrices, no root without |
  | `"new"` | append a synthetic identity bone as bone P |
  | `int` | that supplied joint is the root |
  | `"none"` | today's behaviour |

  The root's transform blocks are held at exactly zero. Every vertex may use
  it. Its weight is floored at 1e-6 after every projection, so no weight
  column can be all zero.
- **Guards.** `run` validates the final weights (finite, non-negative, rows
  sum to one, at most K non-zeros) and raises instead of saving. The
  progress line reports the number of empty columns. The coverage report
  adds root usage and leaves the root out of the idle-joint list.
- **Plumbing.** `--root-joint` on the CLI, `CompressionSettings.root_joint`
  in the Maya pipeline, and a passthrough in `scripts/compare_schedules.py`.
  The output format is unchanged.
- **Out of scope.** Non-negative initial weights (E3), the hierarchical
  basis (R2), locality on the weight side (R3), the weight Laplacian (R4),
  binding the rig builder's `_root` node, and a builder-side threshold snap.

## Design

- The projection block of `train` is extracted into `_project_weights`
  (mask, top-K cut, clamp, root floor) and `_project_transforms` (zero the
  root block, top-L). Both are testable on hand-made tensors.
- During the un-normalised warm-up the root has exactly zero gradient, so it
  sits at the floor while useless weights decay. At the first normalisation a
  vertex whose other weights are all zero gets weight one on the root. In
  normalised phases the root can grow only where the other joints are weak.
- Save-time reconciliation: a root entry still at the floor in a column with
  other live weights is a floor leftover and is zeroed before the final
  normalisation. Every saved column therefore has at most K non-zeros and
  none is empty.
- The root is an ordinary bone with zero `shapeXform` blocks, so
  `AnimationFrameGenerator` and the Maya rig builder need no change.
- Test coupling to "P equals the number of matrices" was found in advance and
  named per task (constructor tests, a CLI test, an in-Maya pipeline test).

## What landed

`_resolve_root_joint`, `_ROOT_WEIGHT_FLOOR`, `_project_weights`,
`_project_transforms`, `_report_root_usage`, weight validation, the CLI and
Maya pipeline options, and `TestRootJoint` in `tests/test_skin_compressor.py`
are all in place. The user guide and the solver-concepts page document the
option.

## Still open

R2, R3 and R4, plus reallocation of idle joints, remain only in
`plan/research_output.md`.
