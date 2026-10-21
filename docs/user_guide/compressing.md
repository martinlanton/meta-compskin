# Compressing

`SkinCompressor` takes a `BlendshapeModelData` and writes the compressed file.
This page covers running it, every setting, custom joints, and practical
advice on getting good results. The theory behind the settings is in
[How the solver works](../concepts/how_the_solver_works.md).

## Minimal run

```python
from metacompskin import BlendshapeModelData, SkinCompressor

model_data = BlendshapeModelData.from_npz("exports/head.npz")
compressor = SkinCompressor(model_data=model_data, iterations=10000)
compressor.run(output_location="exports/head_compressed.npz")
```

- The output directory must already exist.
- The constructor prints the model summary and the `alpha` in use; `run`
  prints progress every 200 iterations and the final errors.
- The device is chosen automatically: CUDA if `torch.cuda.is_available()`,
  otherwise CPU.

From a shell, the same run is:

```bash
python -m metacompskin exports/head.npz exports/head_compressed.npz --iterations 10000
```

Every constructor setting below has a matching option (`--iterations` also
takes a comma-separated list, `--number-of-bones`,
`--max-influences`, `--total-nnz-b-rt`, `--init-weight`, `--power`, `--alpha`,
`--seed`, `--warm-up`, `--weight-init`). Joint matrices stored in the model
file by the exporter are used unless you pass `--ignore-joint-matrices`; the
joint hierarchy stored with them is used only with `--joint-hierarchy`. This is
the command the Maya pipeline runs in a subprocess ([Maya rig workflow](maya_rig_workflow.md#42-one-call-from-maya)).

## Settings

Constructor arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_data` | required | The input model. |
| `iterations` | 10 000 | Steps per stage. An int is one stage (the classic two-phase run). A sequence such as `(5000, 5000, 10000)` runs one stage per entry, halving the influence budget each stage from $K \cdot 2^{N-1}$ down to $K$; the unnormalised warm-up takes the first entry's length when `warm_up_enabled=True`. Needs $K \cdot 2^{N-1} < P$. Make the last stage the longest. |
| `rest_joint_matrices` | `None` | `(P, 4, 4)` joint rest matrices. When given, $P$ becomes the number of matrices. |
| `number_of_bones` | 100 | $P$. Only needed without `rest_joint_matrices`; if both are given they must agree. |
| `max_influences` | 8 | $K$, non-zero weights per vertex. Must be less than $P$. |
| `candidate_joints_per_vertex` | `min(K+2, P-1)` with joint matrices | $M$, the nearest joints (along the mesh surface) a vertex may take its weights from, so each joint drives the region around it. `0` turns the filter off. Needs `rest_joint_matrices`; $K \le M < P$. Also the ceiling of an annealed budget. See [Custom joints](#custom-joints). |
| `weight_init` | `"distance"` with joint matrices, `"noise"` without | Where the skin weights start. `"distance"` weights each vertex to its $K$ nearest joints by distance and puts vertices no shape moves on the root; `"noise"` starts from tiny random values. See [Starting weights](#starting-weights). |
| `distance_penalty` | 0 (off) | With joint matrices, pulls each joint's skin toward the joint by penalising weight on joints farther than a vertex's nearest one. Scale-free. See [Keeping joints local](#keeping-joints-local). |
| `smoothing` | `"reconstruction"` | What `alpha` smooths: the skinned result (the paper's term) or, with `"residual"`, the error, which keeps fine detail the target has. See [Smoothing the error instead of the result](#smoothing-the-error-instead-of-the-result). |
| `still_vertices` | `None` | `"static"` or a distance in model units: skin that does not move (or moves less than that) is pinned to the root and left out of the solve, which makes the run faster. Needs the root influence. See [Skin that does not move](#skin-that-does-not-move). |
| `total_nnz_B_rt` | 6000 | $L$, non-zero delta coefficients across the whole model. Six coefficients make one $(k, j)$ block. |
| `power` | 2 | Exponent $p$ of the error norm. |
| `init_weight` | 1e-3 | Scale of the random initial deltas. Rarely worth touching. |
| `joint_parents` | `None` | With `rest_joint_matrices`, the parent index of each joint (-1 for none). The motion is then solved through that hierarchy. Optional; see [Joint hierarchy](#joint-hierarchy). |
| `deltas_about_joints` | `False` | With `joint_parents`, solve each joint's own delta about the joint itself rather than the mesh centre. See [Joint hierarchy](#joint-hierarchy). |
| `seed` | 12345 | Torch seed for the random initial deltas and weights. Change it to explore other local minima. |
| `warm_up_enabled` | `True` | Whether to include an initial unnormalised warm-up phase. Set to `False` to skip the warm-up and run only normalised training stages; useful for research comparing the impact of the warm-up decay process. |

Attributes you can change after construction and before `run`:

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `alpha` | from `model_data.alpha` | Laplacian smoothness weight. |
| `schedule` | built from `iterations`, `max_influences`, `total_nnz_B_rt` | The `TrainingPhase` objects `run` executes. Reassign for a hand-built schedule (different stage lengths, an annealed $L$, more than one warm-up). |

```python
compressor = SkinCompressor(
    model_data=model_data, iterations=15000, number_of_bones=60, total_nnz_B_rt=9000
)
compressor.alpha = 20.0
compressor.run("exports/head_60j.npz")
```

`alpha` is looked up by model name in `metacompskin.constants`
(the sample heads and anything unknown get 10). Your
own heads get 10 unless you pass `alpha=` to `from_npz` or set it on the
compressor. Lower values suit dense meshes, higher values sparse ones.

## Choosing settings

Start with the defaults; they are the paper's settings and give good results
on human heads of 6 000 to 25 000 vertices with 250 to 320 shapes.

**Fewer shapes than the defaults assume.** $L$ cannot exceed the number of
coefficients, $6 S P$. With 3 shapes and 100 joints that is 1800, so the default
6000 is meaningless and `topk` will fail. Cap it:

```python
budget = min(6000, int(0.8 * 6 * model_data.n_blendshapes * 100))
compressor = SkinCompressor(model_data=model_data, total_nnz_B_rt=budget)
```

**Runtime budget is tight.** Reduce $P$ before reducing $L$; the number of
joints sets the per-frame skinning cost and the number of matrices uploaded,
while $L$ only sets the cost of the sparse sum. Below about 20 joints the error
climbs steeply.

**Worst-case error matters more than average.** Raise `power` to 12 and give
the solver more capacity (more joints, larger $L$, more iterations). Expect
less smooth weights and a much longer solve.

**Result looks blurred or loses wrinkles.** Lower `alpha`, or try
`smoothing="residual"`, which smooths the error instead of the result. If it
looks noisy or the weight map is speckled, raise `alpha`.

**Results vary a lot between seeds, or worst-case error is worse than
expected at your $K$.** Try annealing the influence budget — see
[Annealing the influence budget](#annealing-the-influence-budget) below.

**Smoke-testing a pipeline.** `iterations=600` runs in about a minute on CPU
and produces a valid file with a few times the final error.

## Annealing the influence budget

`max_influences` ($K$) is fixed by the runtime (a GPU skinning shader budget),
so it cannot be raised to give the solver more capacity. Passing `iterations`
as a sequence instead of an int gives the solver that capacity *during
training only*, and hands back a result at the same $K$ you shipped with.

```python
compressor = SkinCompressor(
    model_data=model_data, iterations=(5000, 5000, 5000, 10000), max_influences=8
)
compressor.run("exports/head_annealed.npz")
```

Each entry is one stage: the influence budget starts at
$K \cdot 2^{N-1}$ ($64$ here, for $N = 4$ stages) and halves every stage down
to $K$ ($8$), so weights are free to explore more joints while the solver is
still deciding which ones matter, and only commit to the final $K$ once that
decision is informed. The schedule runs $N + 1$ phases (an unnormalised
warm-up at the loosest budget, then one normalised phase per stage), so the
example above trains for $5000 \times 2 + 5000 + 5000 + 10000 = 30\,000$
steps. Needs $K \cdot 2^{N-1} < P$; make the last stage the longest, since it
follows the harshest cut.

With custom joints the ceiling is the candidate count $M$ instead of
$K \cdot 2^{N-1}$, and the budget shrinks geometrically from $M$ to $K$: two
stages at the defaults give $16, 8$, three give $16, 11, 8$. A single stage
keeps $K$ throughout in both cases.

For anything the sequence form cannot express — a differently sized warm-up,
an annealed $L$, a non-geometric progression — assign `TrainingPhase` objects
to `schedule` directly before calling `run`:

```python
from metacompskin.model_fit import TrainingPhase

compressor = SkinCompressor(model_data=model_data)  # ships K=8, L=6000
compressor.schedule = (
    TrainingPhase(
        iterations=2000,
        max_influences=32,
        total_nnz_B_rt=24000,
        normalize_weights=False,
    ),
    TrainingPhase(
        iterations=8000, max_influences=32, total_nnz_B_rt=24000, normalize_weights=True
    ),
    TrainingPhase(
        iterations=8000, max_influences=16, total_nnz_B_rt=12000, normalize_weights=True
    ),
    TrainingPhase(
        iterations=20000, max_influences=8, total_nnz_B_rt=6000, normalize_weights=True
    ),
)
compressor.run("exports/head_custom_schedule.npz")
```

## Custom joints

By default the joints are 100 anonymous handles with identity rest transforms.
If your rig already has facial joints, or you want a specific joint count and
placement for organisational reasons, pass their rest matrices:

```python
import json
import numpy as np

with open("matrices.json", encoding="utf-8") as f:
    joint_matrices = np.array(json.load(f)).reshape(-1, 4, 4)  # (P, 4, 4)

compressor = SkinCompressor(
    model_data=model_data,
    iterations=10000,
    rest_joint_matrices=joint_matrices,
)
compressor.run("exports/head_compressed.npz")
```

The matrices come straight from `MayaBlendshapeExporter(joints=[...])` as the
`rest_joint_matrices` key, or from any source that produces column-vector
$4 \times 4$ homogeneous matrices.

What custom joints change:

- They set $P$. At least `max_influences + 1` are required.
- They are echoed into `restXform` in the output, so a rig builder can place
  joints where you expect them.
- **Each joint drives the skin around it.** A vertex may only take weight
  from its $M$ nearest joints, measured along the mesh surface so that an
  upper-lip vertex is far from a lower-lip joint even though the two are
  close in space. $M$ is `candidate_joints_per_vertex`, `min(K+2, P-1)` by
  default, so 10 at the default $K$: two spare candidates per vertex, which
  on real rigs gave markedly more local joints than a wider mask at the same
  error. This is what makes the joints usable as tweakers: the solved motion
  goes on the tweaker control's parent, and an animator moving the control
  on top of it moves the skin under that joint and nothing else. Pass
  `candidate_joints_per_vertex=0` to solve without the filter (where a joint
  could drive any part of the face).
- **The weights start near the joints.** See [Starting
  weights](#starting-weights) below.

What they do not change: the deltas are still computed as if every joint sat
at the origin with identity orientation; joint placement in the rig is
handled by the bind pose, and an offset added in the rig rotates the skin
about the joint's own pivot. See [Maya rig workflow](maya_rig_workflow.md).

The filter narrows the solver's search, so expect a somewhat higher error
than the same $P$ without it; a two-stage schedule such as
`iterations=(5000, 10000)` lets all $M$ candidates compete before the cut to
$K$ and usually recovers part of that. After the run the log reports how many
vertices each joint drives and lists joints driving none; an idle joint is
too far from the mesh or redundant with a neighbour.

### Starting weights

Which joints a vertex ends up on is largely decided in the first few solver
steps (see [How the solver works](../concepts/how_the_solver_works.md)), so
where the weights start matters. With joint matrices, `weight_init="distance"`
(the default) starts every vertex on its $K$ nearest joints along the surface,
the nearest with the largest weight and the weight falling to zero at the
next joint out as $(1 - d/d_{K+1})^2$, and starts vertices that no shape moves
fully on the root. The
solver is then free to change all of it: static skin stays on the root except
along its border with moving skin, where the smoothness term blends the two.
On real rigs this start gave both more local joints and a slightly lower
error than the random start.

`weight_init="noise"` (`--weight-init noise`) is the original behaviour, and
the only one without joint matrices: tiny random values, so the first steps
pick the joints. Use it to compare against earlier results.

### Joint hierarchy

Facial rigs are usually hierarchical: lip and chin tweakers sit under the jaw,
eyelid joints under an eye joint. The solver can use that hierarchy. It is
optional and off by default: on the rigs tested so far it did not make joints
more local and cost a little accuracy, so turn it on to try it, not as a
matter of course.

```python
compressor = SkinCompressor(
    model_data=model_data,
    rest_joint_matrices=joint_matrices,
    joint_parents=[-1, 0, 0, 1, ...],  # parent index per joint, -1 for none
)
```

The exporter writes `joint_parents` for you; pass `--joint-hierarchy` on the
command line or `use_joint_hierarchy=True` in the Maya pipeline to use it.

What changes:

- **A child moves with its parent.** Each joint's motion is the sum of its own
  delta and its ancestors'. A lip vertex weighted only to the nearest lip
  joint still opens with the jaw, so its weights can stay on the joints around
  it instead of reaching for the jaw.
- **The delta budget goes further.** `total_nnz_B_rt` counts each joint's own
  delta, so motion shared by a whole region is stored once, on the parent.

Optionally, `deltas_about_joints=True` (`--deltas-about-joints`) solves each
joint's own delta about the joint's rest position instead of about the mesh
centre, so a joint's rotation is directly its local rotation. It does not
change what can be represented, but it changes which coefficients the delta
budget keeps, and with a tight budget that can cost accuracy, so it is off by
default.

What stays the same: what the solver can represent (any result without the
hierarchy can be written with it), and the meaning of `shapeXform`, which holds
each joint's complete motion so runtimes and the rig builder work unchanged.
The file additionally stores the hierarchy and each joint's own deltas; see
[Data formats](../concepts/data_formats.md).

Example script: `examples/example_custom_joints.py`.

### Keeping joints local

The candidate mask and the starting weights decide which joints a vertex can
use and where it starts. `distance_penalty` adds a preference inside that
choice: weight on a joint farther from the vertex than its nearest joint
costs, in proportion to how much farther (zero on the nearest joint and on the
root, growing with the square of the extra relative distance). Each joint's
skin is pulled toward the joint for the whole solve.

```python
compressor = SkinCompressor(
    model_data=model_data,
    rest_joint_matrices=joint_matrices,
    distance_penalty=0.1,
)
```

The strength is relative to the data's motion, so the same value means the
same trade on any scene unit, but not on any rig: it trades accuracy for
locality, and how much of each depends on the joint layout. Raise it until the
error starts to climb. Off by default.

### Smoothing the error instead of the result

The smoothness term, weighted by `alpha`, penalises a reconstruction whose
motion varies sharply between neighbouring vertices. Real shapes have such
detail (wrinkles, creases), so the term also pulls the fit away from it.
`smoothing="residual"` (`--smoothing residual`) applies the same term to the
error between the result and the target instead: it still discourages jagged
weight maps, which produce a jagged error, but no longer penalises detail the
target has. A higher `alpha` is then usually affordable. The default stays the
paper's term.

### Skin that does not move

On many heads part of the skin never moves: the skull, the back of the neck.
With `still_vertices="static"` (`--still-vertices static`) the vertices no
shape moves (under a tenth of a percent of the largest motion) are pinned to
the root and left out of the solve. A distance instead of `"static"` pins
every vertex whose largest motion stays below it, in model units (`0.05` for
half a millimetre in a centimetre scene); that motion is then lost, so keep it
small.

Pinned vertices still count: the reported error covers every vertex, and they
keep their place in the mesh for the joint distances and the smoothness term,
so nothing changes for their moving neighbours. The run gets faster roughly in
proportion to the share of vertices pinned. It needs the root influence, the
default with joint matrices.

## GPU and timing

| Hardware | 10 000 iterations per phase, Aura sample (5 944 vertices, 267 shapes) |
|----------|------|
| NVIDIA A6000 | a few minutes |
| Consumer RTX GPU | a few minutes |
| CPU (8 cores) | roughly 50 minutes |

Memory grows with $S \times N$. If CUDA runs out of memory, close other GPU
processes, or compress separate meshes separately rather than merging them.

Nothing in the API selects the device explicitly. To force CPU on a machine
with a GPU, set `CUDA_VISIBLE_DEVICES=""` in the environment before starting
Python.

## Reproducibility

The random seed is the `seed` argument, 12345 by default. The same code,
data, seed, torch version and hardware class give identical output; the
regression tests depend on this. Across CPU and GPU, or across torch
releases, results differ in the low decimals and occasionally in which
joints own a border region. Both are equally valid solutions.

The optimisation is non-convex, so different seeds converge to different
local minima of similar quality — not just numerical jitter. Running a few
seeds and keeping the one with the lowest `maxDelta` is a cheap way to
improve a fit at no runtime cost:

```python
best = None
for seed in (1, 2, 3, 4, 5):
    compressor = SkinCompressor(model_data=model_data, seed=seed)
    compressor.run(f"exports/head_seed{seed}.npz")
    if best is None or compressor.reconstruction_error.max_abs < best:
        best = compressor.reconstruction_error.max_abs
```

## Batch processing

`SkinCompressor` is a plain Python object; loop over files in a script or a
farm job. It prints to stdout and keeps `loss_list` and `abserr_list` (one
entry per 200 iterations) for plotting convergence.

```python
for npz in sorted(Path("exports").glob("*_head.npz")):
    model_data = BlendshapeModelData.from_npz(npz)
    SkinCompressor(model_data=model_data).run(
        npz.with_name(npz.stem + "_compressed.npz")
    )
```

### Comparing schedules and seeds

`scripts/compare_schedules.py` automates the batch job that answers "is an
annealed schedule actually better than the plain baseline, or within seed
noise?" (see [Annealing the influence budget](#annealing-the-influence-budget)
and [Reproducibility](#reproducibility) above). It runs the baseline, a
phase-count control, and two annealed variants over several seeds, all at
the same total step count, and prints mean/std/min/max of the final MXE and
MAE per variant:

```bash
python scripts/compare_schedules.py exports/head.npz runs/compare \
    --seeds 1,2,3,4,5 --total-iterations 40000 --anneal-stages 3
```

Pass the same `--number-of-bones`, `--max-influences`, `--total-nnz-b-rt`
and `--alpha` you normally compress that model with, so every variant is
compared at the budgets you actually ship. Read the table by comparing the annealed variants' mean against the
baseline's own std: an improvement smaller than that std is not
distinguishable from seed luck. `results.csv` under the output directory has
one row per run for further analysis; `--dry-run` prints the planned runs
without compressing anything.

## After the run

Read the last two lines, `maxDelta` and `meanDelta`, then go to
[Evaluating results](evaluating_results.md) before shipping anything.
`compressor.reconstruction_error` holds the same two numbers as `max_abs`
and `mean_abs`, so a script can read them without parsing stdout.
