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
`--seed`). Joint matrices stored
in the model file by the exporter are used unless you pass
`--ignore-joint-matrices`. This is the command the Maya pipeline runs in a
subprocess ([Maya rig workflow](maya_rig_workflow.md#42-one-call-from-maya)).

## Settings

Constructor arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_data` | required | The input model. |
| `iterations` | 10 000 | Steps per stage. An int is one stage (the classic two-phase run). A sequence such as `(5000, 5000, 10000)` runs one stage per entry, halving the influence budget each stage from $K \cdot 2^{N-1}$ down to $K$; the unnormalised warm-up takes the first entry's length. Needs $K \cdot 2^{N-1} < P$. Make the last stage the longest. |
| `rest_joint_matrices` | `None` | `(P, 4, 4)` joint rest matrices. When given, $P$ becomes the number of matrices. |
| `number_of_bones` | 100 | $P$. Only needed without `rest_joint_matrices`; if both are given they must agree. |
| `max_influences` | 8 | $K$, non-zero weights per vertex. Must be less than $P$. |
| `total_nnz_B_rt` | 6000 | $L$, non-zero delta coefficients across the whole model. Six coefficients make one $(k, j)$ block. |
| `power` | 2 | Exponent $p$ of the error norm. |
| `init_weight` | 1e-3 | Scale of the random initial deltas. Rarely worth touching. |
| `seed` | 12345 | Torch seed for the random initial deltas and weights. Change it to explore other local minima. |

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

**Result looks blurred or loses wrinkles.** Lower `alpha`. If it looks noisy
or the weight map is speckled, raise it.

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

What custom joints do and do not change:

- They set $P$. At least `max_influences + 1` are required.
- They are echoed into `restXform` in the output, so a rig builder can place
  joints where you expect them.
- They do **not** change the solve. The deltas are computed as if every joint
  sat at the origin with identity orientation; joint placement in the rig is
  handled by the bind pose. See [Maya rig workflow](maya_rig_workflow.md).

Example script: `examples/example_custom_joints.py`.

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
