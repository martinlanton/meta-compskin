# Overview

This page explains what the tool does and why, in plain language. It is written
for riggers and animators first; the maths comes later, in
[From blendshapes to skinning](blendshapes_to_skinning.md).

## The problem

A facial blendshape model stores, for every expression shape, where every vertex
goes. With a few hundred shapes on a head of several thousand vertices that is
millions of numbers, and every frame the runtime has to walk through the active
shapes and add up their offsets. On a mobile device, or when several characters
are on screen, that cost matters.

Body animation solved the equivalent problem long ago with **skinning**: a
handful of joints move, and each vertex follows a weighted mix of the joints
near it. A skinned mesh costs almost nothing to evaluate because the GPU does it
in the vertex shader, and the per-frame data is tiny (one transform per joint).

## The idea

Take the blendshape model and find a skinned model that behaves the same way.

The tool invents a small set of **virtual joints** (40 by default), decides
which vertices each joint should influence and by how much (the **skin
weights**), and works out, for every blendshape, how each joint must move to
reproduce that shape (the **joint motion**). Nothing is modelled by hand. An
optimiser starts from random values and adjusts weights and motion until the
skinned mesh matches the original shapes as closely as it can.

```
       Blendshape model                          Compressed skinning model
  ┌──────────────────────────┐              ┌──────────────────────────────┐
  │ neutral head             │              │ neutral head                 │
  │ shape 1: offset/vertex   │   solve      │ 40 virtual joints            │
  │ shape 2: offset/vertex   │  ───────►    │ skin weights (8 per vertex)  │
  │ ...                      │              │ shape 1: motion of joints    │
  │ shape S: offset/vertex   │              │ shape 2: motion of joints    │
  └──────────────────────────┘              │ ...   (mostly zeros)         │
                                            └──────────────────────────────┘
```

At runtime, the animator's controls still produce shape values exactly as
before. Those values are combined into one transform per joint, and the GPU
skins the mesh with them. From the animator's point of view nothing changed;
from the engine's point of view the face is now just another skinned mesh.

## Why the joints are "virtual"

The joints do not correspond to anatomy. They are placed by the optimiser
wherever they best explain the shapes, so one joint might handle the left
corner of the mouth and part of the cheek, another the brow, and so on. Their
position in space does not even matter to the result; only their influence
region and their motion do. They are best thought of as forty independent
"handles" that the shapes pull on.

## Why it is called compressed

Existing skinning decompositions (Dem Bones is the best known) also turn
animation into joints and weights, but every shape moves every joint a little.
This method adds a constraint: **most of the joint motion must be exactly
zero**. A brow shape ends up moving two or three joints and leaves the other
thirty-seven untouched. The solver enforces this from the start rather than
zeroing small values afterwards, which is why accuracy stays as good as the
dense version while the data becomes 5 to 7 times smaller and 2 to 3 times
faster to evaluate.

## Why the joints stretch a little

Shape values are added together: two shapes at 50% is the sum of half of each.
For that to keep working after conversion, joint motions must also be added
together, not chained one rotation after another. Adding rotations is only
well-behaved for small ones, so the method uses "linearised" rotations. The
practical consequence is that a joint's transform is not a pure rotation plus
translation; it carries a small amount of scale and shear. Maya joints handle
this natively. Engines have to accept full affine matrices in their skinning
shader, which most do, and this is the one hard requirement the method places
on the target platform.

## What it costs

- **Accuracy.** The conversion is an approximation. On the paper's test heads
  the average error is well under a tenth of a millimetre and the worst single
  vertex on the worst shape is a few millimetres. The compressor prints both
  numbers when it finishes so you always know where you stand.
- **Time.** Solving takes minutes on a GPU, or most of an hour on a CPU, per
  head. It is an offline step, run once per rig revision.
- **Iteration.** Changing a shape or the mesh means running the solve again.
  The weights and the joint motion are found together and cannot be edited
  piecemeal.

## What stays the same

Everything upstream of the shape values: controls, drivers, corrective and
in-between logic, animation curves. The compressor sees every target, including
correctives and in-betweens, as just another shape to reproduce. The rig
function that decides how much of each shape to use is treated as a black box.

## Where to go next

- Riggers: [Preparing data](../user_guide/preparing_data.md), then
  [Maya rig workflow](../user_guide/maya_rig_workflow.md).
- Engineers: [From blendshapes to skinning](blendshapes_to_skinning.md) for the
  equations, then [Pipeline integration](../user_guide/pipeline_integration.md).
