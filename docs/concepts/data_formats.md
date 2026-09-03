# Data formats

Every file the package reads or writes is a NumPy `.npz` archive: a zip of
named arrays, readable from any language with a zip library and a NumPy
`.npy` parser, and from Python with `numpy.load`. This page is the reference
for each file. Dimension letters are the ones used throughout the
documentation:

| Letter | Meaning |
|--------|---------|
| $N$ | vertices |
| $S$ | blendshapes |
| $P$ | virtual joints |
| $F$ | faces |
| $K$ | maximum influences per vertex |

## Input: blendshape model

Read by `BlendshapeModelData.from_npz`. Written by `MayaBlendshapeExporter`, or
by you (see [Preparing data](../user_guide/preparing_data.md)).

| Key | Shape | dtype | Required | Content |
|-----|-------|-------|----------|---------|
| `rest_verts` | `(N, 3)` | float32 | yes | Neutral pose vertex positions, in scene units. |
| `rest_faces` | `(F, 3)` or `(F, 4)` | int32 | yes | Face vertex indices, 0-based. All faces must have the same vertex count: all triangles or all quads. Used only to build the smoothness Laplacian. |
| `deltas` | `(S, N, 3)` | float32 | yes | Per-shape vertex offsets from the neutral pose, in the same vertex order as `rest_verts`. |
| `inbetween_info` | 0-d object | dict | yes (may be `{}`) | Rig-logic metadata for in-between shapes. Only meaningful for the sample rigs; write an empty dict for your own data. |
| `combination_info` | 0-d object | dict | yes (may be `{}`) | Rig-logic metadata for corrective shapes. Same remark. |
| `shape_names` | `(S,)` | str | no | Target names, aligned with the first axis of `deltas`. Written by the exporter; not read by the compressor but essential downstream. |
| `rest_joint_matrices` | `(P, 4, 4)` | float64 | no | World-space rest matrices of real joints, translation in the last column. Written by the exporter when `joints` is given. Pass to `SkinCompressor(rest_joint_matrices=...)`. |

Loading validates that the shapes agree ($N$ consistent across the three
arrays, faces homogeneous, indices in range) and raises `ValueError` with a
description otherwise. The **model name** is the file stem (`aura.npz` gives
`aura`) and selects the default `alpha`; unknown names get 10.0.

The sample files under `tests/test_data/source_models/` carry additional keys
(eye and mouth geometry, pivots) that the package ignores.

The dictionaries must be stored as 0-d object arrays so that `np.load(...,
allow_pickle=True)` returns them with `.item()`:

```python
np.savez(
    "head.npz",
    rest_verts=rest_verts.astype(np.float32),
    rest_faces=rest_faces.astype(np.int32),
    deltas=deltas.astype(np.float32),
    inbetween_info=np.array({}, dtype=object),
    combination_info=np.array({}, dtype=object),
    shape_names=np.array(names),
)
```

## Output: compressed skinning

Written by `SkinCompressor.run`. Read by `AnimationFrameGenerator`, by rig
builders, and by engine importers.

| Key | Shape | dtype | Content |
|-----|-------|-------|---------|
| `rest` | `(N, 3)` | float32 | Neutral pose **centred at the origin**: `rest_verts` minus its mean. Same vertex order as the input. |
| `quads` | `(F, 3)` or `(F, 4)` | int32 | `rest_faces`, copied through so the file is self-contained. |
| `weights` | `(N, P)` | float32 | Skin weights. Row $i$ is vertex $i$, column $j$ is joint $j$. Non-negative, each row sums to 1, at most $K$ non-zero per row. |
| `restXform` | `(P, 3, 4)` | float32 or float64 | Rest transform of each joint: the first three rows of the `rest_joint_matrices` you passed in, or $3 \times 4$ identities. **Not used by the maths**; carried for rig builders that want to place joints. |
| `shapeXform` | `(3S, 4P)` | float32 | The matrix $\mathbf{B}$. Block $(k, j)$, rows `3k:3k+3`, columns `4j:4j+4`, is the delta transform $\mathbf{N}_{k,j}$ for shape $k$ and joint $j$. About 90% of blocks are exactly zero. |

There is no `shape_names` key in the output. Keep the input file, or write
the names alongside (the private test scripts write a `shape_names.json`
next to the output), because downstream consumers need to know which row block
is which shape.

### Reading blocks

```python
data = np.load("head_compressed.npz")
S = data["shapeXform"].shape[0] // 3
P = data["weights"].shape[1]

def delta_transform(k: int, j: int) -> np.ndarray:      # (3, 4)
    return data["shapeXform"][3 * k : 3 * k + 3, 4 * j : 4 * j + 4]

active = [(k, j) for k in range(S) for j in range(P)
          if np.any(delta_transform(k, j) != 0)]
```

### Centring

The compressor subtracts the mean vertex position before solving, and every
transform in `shapeXform` acts on those centred coordinates with the joints'
rest transform equal to the identity at the origin. A consumer that keeps the
mesh at its original position has to account for the offset; the formula is in
[Pipeline integration](../user_guide/pipeline_integration.md).

### Units

Whatever the input was in. The sample heads are in centimetres. Errors printed
by the compressor are in the same units.

## Input: animation

Read by `AnimationFrameGenerator.generate_frames`.

| Key | Shape | Content |
|-----|-------|---------|
| `weights` | `(frames, controls)` | Animator control values per frame. Only the first `max_control_weights` columns (72 by default) are used; they are passed through the sample rigs' rig logic to obtain the $S$ shape coefficients. |

This format is specific to the sample rigs, whose rig logic is implemented in
`metacompskin.rig.riglogic`. For your own rig, evaluate your own rig logic and
call `compute_frame_vertices` with the resulting `(S,)` coefficients directly.

## Joint matrices (JSON)

Not a package format, but the convention used by the examples and the private
test data for hand-supplied joints: a JSON list of $P$ lists of 16 floats, each
a row-major $4 \times 4$ homogeneous matrix with translation in the last column
and bottom row `[0, 0, 0, 1]`:

```python
with open("matrices.json", encoding="utf-8") as f:
    joint_matrices = np.array(json.load(f)).reshape(-1, 4, 4)
```

`SkinCompressor` uses only the first three rows of each matrix and only echoes
them into `restXform`; they do not influence the solve. What they do set is
$P$, the number of joints, which becomes the number of matrices.

## Matrix conventions

All matrices in this package are **column-vector** convention: a point is a
column, transforms multiply on the left, translation is the last column, and
`shapeXform` blocks are $3 \times 4$ with that layout. Maya's API and its
matrix nodes use **row vectors**: translation in the last row and reversed
multiplication order. `MayaBlendshapeExporter` transposes joint matrices when
writing `rest_joint_matrices`; anything you write to feed Maya from
`shapeXform` must transpose the other way.
