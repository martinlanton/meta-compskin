# Design — R0 warm-up budget and R1 root influence

Revision 1, against commit `12ac666`. Read [spec.md](spec.md) first;
[tasks.md](tasks.md) sequences the work; [research.md](research.md) records
the reasoning.

## 1. Data flow

```
SkinCompressor.__init__
  rest_joint_matrices (P,4,4) ──► rest_joint_matrices_3x4 (P,3,4)   [unchanged]
  root_joint ─────────────────► _resolve_root_joint() ──► self.root_joint: int | None
                                                         self.number_of_bones: P or P+1
  candidate_joints_per_vertex ► _resolve_candidate_joints(…, n_geodesic_joints=P)  [P excludes a synthetic root]
  iterations ─────────────────► build_training_schedule(…, start_influences=M)
                                  warm-up budget = M when given  (R0)

SkinCompressor.run
  W  = 1e-8·randn (P',N) · mask ; W[root] = floor
  B_rt = init_weight·randn · transform_mask          (root block zero)
  candidate_mask = geodesic mask over the caller's joints, root row all true
  for phase in schedule: train(...)
      forward: W_n = W / W.sum(0)  (normalised phases)  — sum ≥ floor > 0
      step: Adam
      _project_weights(W, phase):   mask → top-K cut → clamp ≥ 0 → root ≥ floor
      _project_transforms(B_rt, phase): root block → 0 → top-L
      progress line: … + empty_columns(W)
  Wn = _final_weights(W): drop floor-level root entries where others are alive, normalise
  _validate_weights(Wn, K)  → RuntimeError on failure
  report: error, coverage (+ root usage)
  save: weights Wnᵀ, restXform = _rest_transforms_for_output(), shapeXform = B (root blocks zero)
```

Consumers (`AnimationFrameGenerator`, `maya_rig_builder`) are unchanged: the
root is an ordinary bone whose delta transforms are zero, so Eq. 7 gives the
identity and the builder creates a joint that never moves.

## 2. `model_fit.py`

### 2.1 Constants and helpers (module level)

```python
_ROOT_WEIGHT_FLOOR = 1e-6  # smallest weight the root influence can hold
_ROOT_JOINT_NEW = "new"
_ROOT_JOINT_NONE = "none"


def _resolve_root_joint(
    requested: int | str | None, has_joint_matrices: bool, number_of_bones: int
) -> tuple[int | None, int]:
    """Turns the root_joint argument into (root index or None, bone count).

    None means "new" with joint matrices and no root without; "new" appends
    a synthetic root as the last bone; an index designates a supplied bone;
    "none" turns the root off. Bools and out-of-range values raise.
    """
```

Returns `(None, P)`, `(P, P + 1)` for a new root, or `(index, P)`.

### 2.2 `build_training_schedule` (R0)

```python
warm_up = TrainingPhase(
    stages[0].iterations,
    start_influences if start_influences is not None else stages[0].max_influences,
    total_nnz_B_rt,
    normalize_weights=False,
)
```

Docstring: "The warm-up runs at `start_influences` when given, even for a
single stage, so the top-$K$ cut keeps every candidate while ownership is
decided." `_annealed_influence_budgets` is unchanged.

### 2.3 Constructor

- New argument `root_joint: int | str | None = None` after
  `candidate_joints_per_vertex`. Docstring per spec §4.2, with a `Raises`
  entry.
- After the joint-matrix block: `self.root_joint, self.number_of_bones =
  _resolve_root_joint(root_joint, has_joint_matrices, self.number_of_bones)`.
- `_resolve_candidate_joints` receives the number of *geodesic* joints: the
  caller's $P$ (or the anonymous bone count when no matrices are given),
  never counting a synthetic root. Its validation `K <= M < P` therefore
  keeps its meaning.
- `build_training_schedule(..., start_influences=self.candidate_joints_per_vertex)`
  unchanged; R0 lives inside it.
- New attribute `self.transform_mask: torch.Tensor | None` built in `run`.

### 2.4 `run`

```python
self.candidate_mask = self._build_candidate_mask()      # root row all true
self.transform_mask = self._build_transform_mask()      # (1,1,P',1,1), 0 at the root
B_rt = (init_weight * randn(6, S, P', 1, 1)) ...        # then B_rt *= transform_mask
W = 1e-8 * randn(P', N); W *= candidate_mask; W[root] = _ROOT_WEIGHT_FLOOR
...
Wn = self._final_weights(W)
self._validate_weights(Wn)
...
self._report_joint_coverage(Wn)
if self.root_joint is not None:
    self._report_root_usage(Wn, A)
np.savez(..., restXform=self._rest_transforms_for_output(), ...)
```

`_rest_transforms_for_output()` returns the caller's $(P,3,4)$ matrices
with an identity row appended for a synthetic root, or $(P',3,4)$ identities
without matrices.

`_build_candidate_mask()`: unchanged geodesic mask over the caller's joints,
then `mask[root] = True` (designated) or a true row appended (synthetic).
Without a filter it stays `None`; the root needs no mask row then.

### 2.5 `train`

The projection block (lines 1089–1104) becomes two methods called inside
the existing `no_grad` block; `train` otherwise unchanged:

```python
def _project_weights(self, W: torch.Tensor, phase: TrainingPhase) -> None:
    """Applies the mask, the top-K cut, the sign clamp and the root floor in place."""
    if self.candidate_mask is not None:
        W *= self.candidate_mask
    cutoff = torch.topk(W, phase.max_influences + 1, dim=0).values[-1, :]
    W.copy_((cutoff < W) * W)
    W.clamp_(min=0)
    if self.root_joint is not None:
        W[self.root_joint].clamp_(min=_ROOT_WEIGHT_FLOOR)


def _project_transforms(self, B_rt: torch.Tensor, phase: TrainingPhase) -> None:
    """Zeroes the root's block and keeps the L largest coefficients in place."""
    if self.transform_mask is not None:
        B_rt *= self.transform_mask
    decider = B_rt.abs()
    cutoff = torch.topk(decider.flatten(), phase.total_nnz_B_rt).values[-1]
    B_rt.copy_((decider >= cutoff) * B_rt)
```

Order matters and is the same as today for the existing steps, so the
anonymous path's numerics are unchanged (the extraction is a pure refactor,
verified by the pinned short-iteration test before any behaviour lands).

Progress line: append `(W.sum(dim=0) == 0).sum()` as a fifth number,
documented in the log-reading section of the concepts page.

### 2.6 Final weights and validation

```python
def _final_weights(self, W: torch.Tensor) -> torch.Tensor:
    """Normalises the weights, dropping floor-level root entries first."""
    W = W.detach().clone()
    if self.root_joint is not None:
        others_alive = (W.sum(dim=0) - W[self.root_joint]) > 0
        at_floor = W[self.root_joint] <= _ROOT_WEIGHT_FLOOR
        W[self.root_joint, at_floor & others_alive] = 0
    return W / W.sum(dim=0)


def _validate_weights(self, Wn: torch.Tensor) -> None:
    """Raises RuntimeError unless Wn is finite, non-negative, normalised and K-sparse."""
```

## 3. Callers

- `cli.py`: `--root-joint` with a converter `_root_joint_arg(text) -> int | str`
  (`auto` → omitted, `none`/`new` → the string, digits → `int`, else
  `ArgumentTypeError`); `_compressor_settings` adds `"root_joint"`.
- `maya_pipeline.py`: `CompressionSettings.root_joint: int | str | None = None`;
  `compression_command` emits `--root-joint <value>`; `compress_and_build_rig`
  takes `root_joint` and resolves a joint name to its index in `joints`
  (`ValueError` if `joints` is missing or the name is not in it).
- `scripts/compare_schedules.py`: `--root-joint` parsed the same way and
  forwarded in `settings`.

## 4. Why the output respects $K$

After the last projection every column has at most $K$ non-root or root
entries above the cutoff plus, possibly, the root reset to the floor. Three
cases for the root entry $w_r$ of a column:

1. $w_r > \varepsilon_r$: it survived the cut, so the column holds at most
   $K$ entries including it. Kept.
2. $w_r = \varepsilon_r$ and some other entry is non-zero: it was reset by
   the floor after the cut dropped it; the column holds at most $K$ other
   entries. Zeroed by `_final_weights`.
3. $w_r = \varepsilon_r$ and every other entry is zero: the vertex has no
   influence but the root. Kept; it normalises to one.

Hence every saved column has at most $K$ non-zeros, sums to one, and none
is empty. `_validate_weights` checks exactly this and the smoke test's
`test_weights_satisfy_lbs_constraints` already asserts it end to end.

## 5. File-by-file

| File | Change |
|---|---|
| `src/metacompskin/model_fit.py` | R0 in `build_training_schedule`; `_resolve_root_joint`; constructor argument and attributes; `_build_transform_mask`, root row in `_build_candidate_mask`; `_project_weights`, `_project_transforms` extracted from `train`; `_final_weights`, `_validate_weights`, `_rest_transforms_for_output`, `_report_root_usage`; progress line |
| `src/metacompskin/cli.py` | `--root-joint` |
| `src/metacompskin/maya_pipeline.py` | `CompressionSettings.root_joint`, `compression_command`, `compress_and_build_rig` name resolution |
| `scripts/compare_schedules.py` | `--root-joint` passthrough |
| `tests/test_training_schedule.py` | single-stage warm-up test flipped to $M$ |
| `tests/test_skin_compressor.py` | `TestRootJoint` (resolution, outputs, projections, reconciliation, validation, static-region integration, report) |
| `tests/test_cli.py`, `tests/test_maya_pipeline.py`, `tests/test_compare_schedules.py` | flag and settings tests |
| `docs/user_guide/compressing.md` | settings row, custom-joints section, iterations row |
| `docs/concepts/how_the_solver_works.md` | projections and phases: root, floor, warm-up at $M$, log column |
| `docs/concepts/data_formats.md` | the root as a zero-block bone |
| `docs/user_guide/troubleshooting.md` | the `nan` entry |
| `docs/user_guide/maya_rig_workflow.md`, `docs/user_guide/evaluating_results.md` | static root joint, new report lines |
| `CLAUDE.md` | parameter table |

## 6. Compatibility

- Anonymous path: `root_joint=None` resolves to no root; `start_influences`
  is `None`; the projections are extracted without reordering. The pinned
  regression tests (600 and 10 000 iterations) must pass unchanged; the
  refactor task runs the short one before any behaviour task.
- Joint path: numerics change (R0 and R1 both). No public test pins them;
  the private repository's joint tests assert $P = 28$ and will need
  `root_joint="none"` or $29$.
- NPZ format: unchanged keys and shapes semantics; $P$ may be one larger.
