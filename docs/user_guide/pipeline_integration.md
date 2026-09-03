# Pipeline integration

This page is for the engineer who has to consume the compressed file: in an
engine, a custom runtime, or a DCC exporter. It gives the per-frame algorithm,
the storage layout that makes it fast, the coordinate conventions, and the
requirements the method places on the platform. The Maya-specific version for
riggers is [Maya rig workflow](maya_rig_workflow.md).

## What the runtime has to do

Per character, once at load time:

- Read `weights` `(N, P)`, `rest` `(N, 3)`, and `shapeXform` `(3S, 4P)`
  from the compressed file ([Data formats](../concepts/data_formats.md)).
- Build the skinned mesh: the neutral vertices with up to $K$ (joint index,
  weight) pairs each. This is an ordinary skinned mesh asset.
- Store the non-zero $3 \times 4$ blocks of `shapeXform` in a sparse structure
  indexed by shape (below).

Per frame:

1. Evaluate the rig as before to obtain the shape coefficients $c_k$.
2. For each joint $j$, compute $\mathbf{M}_j = \mathbf{I} + \sum_k c_k \mathbf{N}_{k,j}$.
3. Optionally compose each $\mathbf{M}_j$ with the head or root transform.
4. Upload the $P$ matrices to the skinning shader and draw.

Step 2 is the only new code, and it is a few lines.

## The per-frame sum

Reference implementation, dense, straight from
`AnimationFrameGenerator._generate_skinning_transforms`:

```python
def skinning_transforms(shape_xform: np.ndarray, c: np.ndarray) -> np.ndarray:
    """shape_xform: (3S, 4P); c: (S,). Returns (P, 3, 4)."""
    S = c.shape[0]
    P = shape_xform.shape[1] // 4
    # Weighted sum of the 3-row blocks: (3, 4P)
    M = np.zeros((3, 4 * P))
    for k in np.flatnonzero(c):
        M += c[k] * shape_xform[3 * k : 3 * k + 3]
    M = M.reshape(3, P, 4).transpose(1, 0, 2)      # (P, 3, 4)
    M[:, :, :3] += np.eye(3)                       # add identity
    return M
```

Production version, sparse. Precompute, per shape $k$, the list of joints it
touches and their blocks:

```
blocks[k] = [(j, N_kj) for j in range(P) if N_kj is not all zero]
```

Then:

```
M[j] = identity_3x4 for all j
for k in active shapes:                 # c[k] != 0
    for (j, N) in blocks[k]:
        M[j] += c[k] * N
```

With the default budget of 6000 coefficients there are at most 1000 non-empty
blocks in total, and a typical frame activates a few dozen shapes, so the
inner loop runs a few hundred fused multiply-adds of 12 floats. The paper
measures 160 to 250 µs per frame for this on a Snapdragon 652, versus 520 to
650 µs for the dense sum, and 81 to 87 KB of storage versus roughly half a
megabyte (Tables 2 and 3). Compressed row storage with 16-bit indices is what
the paper used.

Shape coefficients can be negative or exceed 1; the sum handles it the same
way blendshapes do. Do not clamp them unless the rig did.

## Coordinate conventions

**Column vectors.** Every matrix in the file is column-vector convention:
$\mathbf{M}_j$ is $3 \times 4$, applied as
$\mathbf{M}_j \, [x\ y\ z\ 1]^\top$. Engines and maths libraries that use row
vectors (DirectX-style code, Maya) need the transpose. A
symptom of getting this wrong is a face that looks right on single shapes at
low intensity and shears as shapes combine.

**Centred rest pose.** `rest` is the neutral mesh with its mean vertex
position subtracted, and the deltas act on those coordinates with the joints'
rest transform at the identity. If the skinned mesh asset uses `rest` as its
vertices, use $\mathbf{M}_j$ as-is. If the asset keeps the original vertex
positions $\hat{\mathbf{v}} = \mathbf{v}_{\text{rest}} + \mathbf{c}$ (where
$\mathbf{c}$ is the mean), conjugate every joint transform by the offset:

$$
\mathbf{M}'_j = T(\mathbf{c})\, \mathbf{M}_j\, T(-\mathbf{c})
$$

where $T$ is a translation. This is a constant pre- and post-multiplication
that can be folded in at load time.

**Bind pose.** Engines that skin as `world[j] × inverseBind[j]` expect an
inverse bind matrix per joint. With virtual joints at the identity, the inverse
bind is the identity (or $T(-\mathbf{c})$ if you keep the offset), and the
joint's world matrix at a frame is $\mathbf{M}_j$ (or $\mathbf{M}'_j$). If you
placed the joints somewhere else for authoring, the general relation is

$$
\text{world}_j = T(\mathbf{c})\, \mathbf{M}_j\, T(-\mathbf{c})\, \text{bind}_j
$$

so that `world × inverseBind` reduces to the intended skinning matrix.

**Head and body.** Compose *after* the sum: $\mathbf{H}\,\mathbf{M}_j$ where
$\mathbf{H}$ is the head's world transform. Equivalently, parent the virtual
joints under the head joint and treat $\mathbf{M}_j$ as their local matrices.

**Units.** Those of the input mesh. Nothing rescales.

## Platform requirements

- **Affine skinning.** $\mathbf{M}_j$ contains scale and shear. The vertex
  shader must accept full $3 \times 4$ (or $4 \times 4$) matrices per joint.
  Shaders that reconstruct joints from a quaternion and a translation, or from
  dual quaternions, cannot represent the transforms. The paper states this as
  the method's one hard requirement. If the platform decomposes and
  re-composes joint transforms on import, verify that non-uniform scale and
  shear survive; test the jaw-open shape first.
- **Normals.** Linear blend skinning transforms positions only. Normals are
  typically recomputed or transformed with the inverse-transpose of the
  $3 \times 3$ part; because the transforms are affine, the inverse-transpose
  is not equal to the matrix itself. Use whatever the engine already does for
  scaled joints.
- **Influences.** $K = 8$ by default. Platforms limited to 4 influences need
  a solve with `max_influences = 4`; do not prune afterwards.
- **Precision.** Weights and blocks are float32. Half precision for the blocks
  is fine for the paper's error levels; for weights it usually is too, but
  measure.

## Baking instead of evaluating

If the target platform cannot run the sum but can play joint animation, bake:
for each frame of a clip, evaluate $\mathbf{M}_j$ offline and store it as the
joint's transform (translate, rotate, scale, shear). Any joint animation format
that keeps scale and shear works. The cost is that the face is no longer driven
by shape coefficients at runtime, so procedural or interactive facial animation
is not possible in that mode. This is the usual path through Maya and FBX; see
[Maya rig workflow](maya_rig_workflow.md).

## Validating an implementation

`AnimationFrameGenerator.compute_frame_vertices(c)` returns the reference
vertex positions for any coefficient vector, in the centred frame. Feed the
same coefficients through your runtime, read the vertices back, subtract the
centring offset if you kept it, and compare. Agreement should be at float32
noise level; anything above is a convention bug, not solver error. A good test
set is every shape alone at 1.0, plus a few random combinations of 10 to 20
shapes at values between 0.2 and 1.2.

## Checklist

- [ ] `weights` applied to the skinned asset, joint order preserved, not
  renormalised or pruned.
- [ ] Non-zero blocks of `shapeXform` stored per shape; shape order recorded.
- [ ] Per-frame sum implemented with identity added once per joint.
- [ ] Column-vector convention respected, or transposed once.
- [ ] Centring offset handled (mesh centred, or transforms conjugated).
- [ ] Skinning shader accepts affine matrices; shear verified on jaw-open.
- [ ] Output compared against `compute_frame_vertices` on single shapes and
      combinations.
