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

Every constructor setting below has a matching option (`--number-of-bones`,
`--max-influences`, `--total-nnz-b-rt`, `--init-weight`, `--power`, `--alpha`). Joint matrices stored
in the model file by the exporter are used unless you pass
`--ignore-joint-matrices`. This is the command the Maya pipeline runs in a
subprocess ([Maya rig workflow](maya_rig_workflow.md#42-one-call-from-maya)).

## Settings

Constructor arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `model_data` | required | The input model. |
| `iterations` | 10 000 | Steps **per phase**; two phases run, so 20 000 steps in total. |
| `rest_joint_matrices` | `None` | `(P, 4, 4)` joint rest matrices. When given, $P$ becomes the number of matrices. |
| `number_of_bones` | 100 | $P$. Only needed without `rest_joint_matrices`; if both are given they must agree. |
| `max_influences` | 8 | $K$, non-zero weights per vertex. Must be less than $P$. |
| `total_nnz_B_rt` | 6000 | $L$, non-zero delta coefficients across the whole model. Six coefficients make one $(k, j)$ block. |
| `power` | 2 | Exponent $p$ of the error norm. |
| `init_weight` | 1e-3 | Scale of the random initial deltas. Rarely worth touching. |

Attributes you can change after construction and before `run`:

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `alpha` | from `model_data.alpha` | Laplacian smoothness weight. |

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

**Smoke-testing a pipeline.** `iterations=600` runs in about a minute on CPU
and produces a valid file with a few times the final error.

## Custom joints

By default the joints are 100 anonymous handles with identity rest transforms.
If your rig already has facial joints, or you want a specific joint count and
placement for organisational reasons, pass their rest matrices:

```python
import json
import numpy as np

with open("matrices.json", encoding="utf-8") as f:
    joint_matrices = np.array(json.load(f)).reshape(-1, 4, 4)   # (P, 4, 4)

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

The random seed is fixed (12345) in the constructor. The same code, data,
torch version and hardware class give identical output; the regression tests
depend on this. Across CPU and GPU, or across torch releases, results differ in
the low decimals and occasionally in which joints own a border region. Both
are equally valid solutions.

## Batch processing

`SkinCompressor` is a plain Python object; loop over files in a script or a
farm job. It prints to stdout and keeps `loss_list` and `abserr_list` (one
entry per 200 iterations) for plotting convergence.

```python
for npz in sorted(Path("exports").glob("*_head.npz")):
    model_data = BlendshapeModelData.from_npz(npz)
    SkinCompressor(model_data=model_data).run(npz.with_name(npz.stem + "_compressed.npz"))
```

## After the run

Read the last two lines, `maxDelta` and `meanDelta`, then go to
[Evaluating results](evaluating_results.md) before shipping anything.
