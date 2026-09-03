# Quick start

This page runs the whole pipeline once, end to end, on the sample data that
ships with the repository. It takes a few minutes on a GPU, longer on CPU. Each
step links to the page that explains it in depth.

```
   Maya scene / OBJ files                 sample .npz
            │                                  │
            ▼                                  ▼
   1. Export blendshapes  ───────────►  BlendshapeModelData
                                               │
                                               ▼
                                     2. SkinCompressor.run()
                                               │
                                               ▼
                                   compressed .npz (weights + joint motion)
                                               │
                                               ▼
                          3. AnimationFrameGenerator  ──►  deformed meshes / OBJ frames
```

## 1. Load a blendshape model

The compressor reads a `BlendshapeModelData`: the neutral mesh, its faces, and
one offset per vertex per blendshape. The repository ships four sample heads
under `tests/test_data/source_models/`.

```python
from metacompskin import BlendshapeModelData

model_data = BlendshapeModelData.from_npz("tests/test_data/source_models/aura.npz")
print(model_data)
# BlendshapeModelData(model_name='aura', n_blendshapes=267, n_vertices=5944)
```

To use your own head instead, export it from Maya with `MayaBlendshapeExporter`
or build the `.npz` file yourself. Both are covered in
[Preparing data](../user_guide/preparing_data.md).

## 2. Compress

```python
from metacompskin import SkinCompressor

compressor = SkinCompressor(model_data=model_data, iterations=10000)
compressor.run(output_location="output/aura_compressed.npz")
```

The output folder must exist. While it runs, the compressor prints one line
every 200 iterations and finishes with two numbers, which for the Aura sample
at these settings land around:

```
maxDelta 0.58
meanDelta 0.0038
```

These are the worst and the average difference between the original blendshapes
and their skinned reconstruction, in the model's units (centimetres for the
sample heads). What they mean and how to improve them is in
[Evaluating results](../user_guide/evaluating_results.md); every setting you can
change is in [Compressing](../user_guide/compressing.md).

Use `iterations=600` for a first smoke test. Quality is noticeably worse, but
it runs in about a minute on CPU.

## 3. Use the result

The compressed file holds the skin weights and, for every blendshape, the small
motion of every virtual joint. `AnimationFrameGenerator` is the reference
runtime that turns shape coefficients back into vertices:

```python
import numpy as np
from metacompskin import AnimationFrameGenerator

generator = AnimationFrameGenerator("output/aura_compressed.npz", model_data)

coefficients = np.zeros(model_data.n_blendshapes)
coefficients[12] = 1.0                       # shape 12 fully on
vertices = generator.compute_frame_vertices(coefficients)   # (N, 3)
```

For the sample heads, which carry their rig-logic metadata, you can also drive
a whole animation from animator controls and write OBJ files per frame:

```python
generator.generate_frames(
    animation_weights_path="tests/test_data/source_models/test_anim.npz",
    output_dir="output/frames",
)
```

## 4. Where the result goes

In production the compressed file is consumed by a rig or an engine, not by
this package:

- Riggers: [Maya rig workflow](../user_guide/maya_rig_workflow.md) explains what
  the arrays are and how to wire them into a joint-based rig.
- Engineers: [Pipeline integration](../user_guide/pipeline_integration.md)
  gives the per-frame algorithm and the storage layout.

The complete script version of this page is `examples/example_usage.py`.
