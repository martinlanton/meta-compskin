# Preparing data

The compressor needs one thing: a neutral mesh and, for every blendshape, where
each vertex of that mesh goes. This page covers the three ways to produce that
input, what the geometry must satisfy, and how rig features such as in-betweens
and correctives fit in.

## What the compressor needs

| | Requirement | Why |
|---|---|---|
| One mesh | A single polygon mesh, all triangles or all quads. | The smoothness term needs face connectivity. Mixed topology is rejected. |
| Identical topology across shapes | Every target has the same vertex count **and order** as the neutral mesh. | Deltas are computed per vertex index. |
| Neutral pose | The mesh with every shape at zero. | Deltas are measured from it; the output is centred on it. |
| All shapes you want reproduced | Every target the rig can activate, including in-betweens and correctives. | The solver only reproduces what it sees. |
| Sensible units | Any, but consistent. | Error reports and thresholds are in these units. |

Separate meshes (eyes, teeth, tongue) are separate models. Compress each one
on its own, or leave them as they are if they do not use blendshapes.

## Option 1: export from a Maya scene (recommended)

`MayaBlendshapeExporter` runs inside Maya or `mayapy`, reads the geometry
through the Maya API, and writes the input file. It needs only numpy inside
Maya; see [Installation](../getting_started/installation.md#inside-autodesk-maya).

### From a blendShape node

The simplest mode. The exporter finds the `blendShape` deformer in the mesh's
history, switches each target weight to 1.0 in turn with every other weight at
0, records the resulting vertex positions, and restores the original weights
and envelope afterwards (also on failure).

```python
from metacompskin.maya_exporter import MayaBlendshapeExporter

MayaBlendshapeExporter("head_GEO").export("D:/exports/head.npz")

# If the mesh has several blendShape nodes, say which one:
MayaBlendshapeExporter("head_GEO", blendshape_node="face_BS").export("D:/exports/head.npz")
```

Points are read in object space after the full deformer stack, so anything
else deforming the mesh (a skin cluster, a lattice) is baked into the export.
Make sure the rest of the stack is at rest, or disable it, before exporting.

Target weights must be settable: if a weight is locked or driven by an incoming
connection, the exporter raises rather than exporting a wrong shape. Break the
connections on a copy of the scene, or use the target-mesh mode.

### From separate target meshes

When the shapes exist as individual meshes in the scene, for example a
modelling file where the targets are laid out in a grid:

```python
MayaBlendshapeExporter(
    "head_GEO",
    target_meshes=["jawOpen_GEO", "smile_L_GEO", "smile_R_GEO"],
).export("D:/exports/head.npz")
```

Points are read in object space, so targets translated aside for layout still
produce correct deltas. Every target must have the same vertex count as the
rest mesh; the exporter names the offending mesh otherwise.

### With joint rest matrices

If you want the output's joints to line up with real joints in your rig, pass
their names. Their world matrices are written under `rest_joint_matrices`,
already transposed into the package's column-vector convention.

```python
MayaBlendshapeExporter(
    "head_GEO",
    joints=["face_00_JNT", "face_01_JNT", ...],   # at least 9
).export("D:/exports/head.npz")
```

Two things to know before doing this:

- The number of joints you pass becomes the number of virtual joints $P$. The
  solver needs at least `max_influences + 1` of them, 9 by default, and works
  best with several dozen.
- Joint placement does **not** affect the solve. The rest matrices are echoed
  into the output for rig builders; the deltas are solved as if every joint
  were at the origin. See [Data formats](../concepts/data_formats.md).

### What the exporter writes

`deltas`, `rest_verts`, `rest_faces`, empty `inbetween_info` and
`combination_info`, `shape_names` (the blendShape target aliases or the target
mesh names, in order), and `rest_joint_matrices` when joints were given. The
file loads with `BlendshapeModelData.from_npz` on any machine.

## Option 2: OBJ files through Maya's importer

`MayaBlendshapeModelData.from_obj_files` takes a neutral OBJ and a list of
target OBJs and computes deltas from them. It can run inside Maya, or from a
normal Python session by pointing it at `mayapy`, which it spawns once per
file:

```python
from pathlib import Path
from metacompskin import MayaBlendshapeModelData

shapes_dir = Path("D:/exports/obj")
rest = shapes_dir / "HEAD.obj"
targets = sorted(p for p in shapes_dir.glob("*.obj") if p != rest)

model_data = MayaBlendshapeModelData.from_obj_files(
    rest_obj_path=rest,
    blendshape_paths=targets,
    model_name="head",
    maya_interpreter_path=Path("C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe"),
)
```

Each spawn costs about five seconds of Maya start-up, so a 70-shape head takes
several minutes to load. Shape order is the order of the list you pass; keep
it (for example `[p.stem for p in targets]`) because nothing else records it.
Vertex order must survive the OBJ round trip: export all targets from the same
scene with the same options.

Use this route when the shapes already exist as OBJ files or when you cannot
run scripts in the source scene. Otherwise Option 1 is faster and keeps the
names.

## Option 3: build the file yourself

Any DCC, or any script that can produce vertex arrays, can write the input.
Assemble the arrays and save them as described in
[Data formats](../concepts/data_formats.md#input-blendshape-model):

```python
import numpy as np
from metacompskin import BlendshapeModelData

deltas = np.stack([target - rest_verts for target in targets])   # (S, N, 3)

np.savez(
    "head.npz",
    rest_verts=rest_verts.astype(np.float32),
    rest_faces=rest_faces.astype(np.int32),
    deltas=deltas.astype(np.float32),
    inbetween_info=np.array({}, dtype=object),
    combination_info=np.array({}, dtype=object),
    shape_names=np.array(names),
)
model_data = BlendshapeModelData.from_npz("head.npz")
```

You can also skip the file and construct `BlendshapeModelData` directly with
the same arrays plus `model_name` and `alpha`. Direct construction does not
validate; call it only with arrays you trust.

## In-betweens, correctives and the rig function

The compressor treats every target as an independent shape. That is the right
thing to do: an in-between at 50% jaw open is a mesh, a corrective for "jaw
open and lips together" is a mesh, and each gets its own set of joint deltas.
At runtime your rig function still decides how much of each to use, and those
values feed Equation 7 unchanged
([From blendshapes to skinning](../concepts/blendshapes_to_skinning.md)).

Two consequences:

- Export **all** targets, not just the primary ones. A corrective that is
  missing from the export cannot be reproduced.
- The `inbetween_info` and `combination_info` dictionaries in the file are for
  the sample rigs' built-in rig logic and stay empty for your data. Nothing in
  the solve uses them.

## Checking the input before compressing

```python
print(model_data)                 # name, S, N
model_data.print_details()        # shapes and dtypes of every array

import numpy as np
per_shape = np.linalg.norm(model_data.deltas, axis=2).max(axis=1)
print("largest displacement per shape:", per_shape.round(2))
```

Shapes whose largest displacement is zero are empty targets; remove them, they
waste solver capacity. A shape whose displacement is far larger than the rest
is often a target that was captured with the wrong pose or a stray transform.

Next: [Compressing](compressing.md).
