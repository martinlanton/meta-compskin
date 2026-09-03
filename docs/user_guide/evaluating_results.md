# Evaluating results

The compressed file is an approximation of the blendshapes. This page explains
the numbers the compressor prints, how to compute your own, how to look at the
reconstruction, and what to do when it is not good enough.

## The two headline numbers

At the end of `run`:

```
maxDelta 0.58
meanDelta 0.0038
```

| Name | Paper | Definition | Read it as |
|------|-------|-----------|------------|
| `meanDelta` | MAE | mean over every vertex and every shape of the per-axis absolute difference between original delta and reconstruction | "typical error anywhere on the face" |
| `maxDelta` | MXE | the largest single per-axis difference across all vertices and shapes | "the worst spot on the worst shape" |

Units are the model's units (centimetres for the sample heads). The paper's
Table 1, in millimetres, gives the reference for human-sized heads at
10 000 iterations per phase:

| Model | Vertices | Shapes | MXE (mm) | MAE (mm) |
|-------|----------|--------|----------|----------|
| Aura | 5 944 | 267 | 5.82 | 0.038 |
| Jupiter | 5 944 | 319 | 8.26 | 0.030 |
| Proteus | 23 735 | 287 | 4.8 | 0.030 |
| Bowen | 23 735 | 253 | 5.99 | 0.034 |

A mean error under 0.05 mm is invisible. A maximum of a few millimetres is
usually a single vertex inside the mouth bag or at an eyelid crease on one
extreme shape; whether it matters depends on where it is, which is why the
next step is to look.

## Computing errors per shape

`AnimationFrameGenerator` is the reference runtime. Reconstruct each shape on
its own and compare it to the original delta:

```python
import numpy as np
from metacompskin import AnimationFrameGenerator, BlendshapeModelData

model_data = BlendshapeModelData.from_npz("exports/head.npz")
generator = AnimationFrameGenerator("exports/head_compressed.npz", model_data)

rest_centred = model_data.rest_verts - model_data.rest_verts.mean(axis=0)
names = np.load("exports/head.npz", allow_pickle=True)["shape_names"]

for k, name in enumerate(names):
    c = np.zeros(model_data.n_blendshapes)
    c[k] = 1.0
    reconstructed_delta = generator.compute_frame_vertices(c) - rest_centred
    err = np.linalg.norm(reconstructed_delta - model_data.deltas[k], axis=1)
    print(f"{name:32s} mean {err.mean():.4f}  max {err.max():.3f}  at vertex {err.argmax()}")
```

Sort by `max` and look at the top few. The vertex index tells you where; the
shape name tells you what to test in the rig.

Note that `compute_frame_vertices` returns positions in the **centred** frame
(the output file's `rest`), hence the subtraction above.

## Looking at it

Numbers do not show whether an error is a visible dent or a hidden vertex.
Three ways to look:

1. **Per-shape OBJs.** Write `generator.compute_frame_vertices(c)` with the
   output's `quads` to OBJ files (the `_write_obj` helper in
   `animation_generator.py` does exactly this) and load them next to the
   original targets in your DCC. Toggle between them.
2. **Error as colour.** Write the per-vertex `err` array above as a vertex
   colour or a weight map on the neutral mesh. The paper's Figure 3 uses red
   for 5 mm and above.
3. **In the rig.** Select the head in Maya and call
   `build_skinned_rig("exports/head_compressed.npz")`: it skins a duplicate and
   keys one blendshape per frame, so scrubbing the timeline steps through every
   shape next to the original blendShape mesh
   ([Maya rig workflow](maya_rig_workflow.md), section 4).

## Full animations

For the sample heads, whose rig logic ships in `metacompskin.rig.riglogic`,
`generate_frames` runs animator controls through that logic and writes one OBJ
per frame:

```python
generator.generate_frames(
    animation_weights_path="tests/test_data/source_models/test_anim.npz",
    output_dir="output/frames",
)
```

For your own rig this method does not apply: `generate_frames` needs the
`inbetween_info` and `combination_info` dictionaries that only the sample
files carry, and your rig logic lives in your rig. Evaluate your controls to
shape coefficients with your own tooling and call `compute_frame_vertices`
per frame instead.

## When it is not good enough

Work through these in order; each is cheaper than the next.

| Symptom | Try |
|---------|-----|
| Error is fine but one shape is bad | Check the input: was that target captured correctly? A stray transform on one target shows up exactly like this. |
| Mean error is fine, max error is too high on visible areas | More iterations first, then more joints. If it is one localised feature such as a wrinkle, `power = 12` with more capacity. |
| Everything is a little off, weights look speckled | Raise `alpha`. |
| Everything is a little blurred, creases lost | Lower `alpha`, then raise $L$. |
| Error rose after reducing $P$ or $L$ for performance | That is the trade; go back one step, or accept it after looking at the shapes that got worse. |
| Different machine, slightly different numbers | Expected; see [Compressing](compressing.md#reproducibility). |

Any change to the input or the settings means a full re-run: nothing in the
output can be edited in place.

## Beyond the compressor

The compressor's numbers describe the skinned mesh evaluated exactly as
`AnimationFrameGenerator` does it. A rig or an engine adds its own sources of
error: a rig that discards shear, an exporter that quantises weights, an engine
that limits influences. Measure again at the end of the pipeline; the
comparison method above works on any set of vertex positions you can get back
out.

Next: [Pipeline integration](pipeline_integration.md) or
[Maya rig workflow](maya_rig_workflow.md).
