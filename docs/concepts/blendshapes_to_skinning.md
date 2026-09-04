# From blendshapes to skinning

This page walks through the maths of the conversion at the level needed to
implement or debug it. It follows Sections 2 and 3 of the paper but spells out
every symbol and stops to say what each equation means in rigging terms. No
optimisation yet; that is [How the solver works](how_the_solver_works.md).

## Notation

| Symbol | Meaning | In the code |
|--------|---------|-------------|
| $N$ | number of vertices | `n_vertices` |
| $S$ | number of blendshapes | `n_blendshapes` |
| $P$ | number of virtual joints (the paper says *bones*) | `number_of_bones`, default 100 |
| $K$ | maximum joints influencing one vertex | `max_influences`, default 8 |
| $\hat{\mathbf{v}}_{0,i}$ | position of vertex $i$ in the neutral pose | `rest_verts[i]` |
| $\hat{\mathbf{v}}_{k,i}$ | position of vertex $i$ in shape $k$ | `rest_verts[i] + deltas[k, i]` |
| $c_k$ | current value of shape $k$ (0 = off, 1 = fully on) | shape coefficients |
| $w_{i,j}$ | skin weight of joint $j$ on vertex $i$ | `weights[i, j]` |
| $\mathbf{N}_{k,j}$ | motion of joint $j$ for shape $k$, a $3 \times 4$ matrix | block of `shapeXform` |
| $\mathbf{M}_j$ | transform of joint $j$ at the current frame, $3 \times 4$ | computed per frame |

Vertices are used in homogeneous form where needed: $\mathbf{v}_{0,i}$ is
$\hat{\mathbf{v}}_{0,i}$ with a fourth coordinate equal to 1, so that a
$3 \times 4$ matrix can rotate and translate it in one multiplication.

## Blendshapes (Equation 1)

A delta blendshape model computes each vertex as the neutral position plus a
weighted sum of offsets:

$$
\hat{\mathbf{v}}_{0,i} + \sum_{k=1}^{S} c_k \,(\hat{\mathbf{v}}_{k,i} - \hat{\mathbf{v}}_{0,i})
$$

The offsets $\hat{\mathbf{v}}_{k,i} - \hat{\mathbf{v}}_{0,i}$ are exactly what
the input file stores in `deltas`. Animators do not set $c_k$ directly; a rig
function turns a few dozen controls into the two or three hundred $c_k$, adding
in-betweens and correctives on the way. The method never looks inside that
function.

## Linear blend skinning (Equation 2)

A skinned mesh computes each vertex as a weighted mix of what each joint would
do to it:

$$
\sum_{j=1}^{P} w_{i,j}\, \mathbf{M}_j\, \mathbf{v}_{0,i}
$$

with three constraints on the weights that every GPU skinning pipeline assumes:
non-negative ($w_{i,j} \ge 0$), summing to one for each vertex
($\sum_j w_{i,j} = 1$), and at most $K$ non-zero per vertex. The transforms
$\mathbf{M}_j$ are the only thing that changes per frame. They are general
affine $3 \times 4$ matrices: rotation, translation, and, as we will see,
unavoidably some scale and shear.

## Matching the two (Equations 3 to 5)

Put the neutral pose aside for a moment and ask: what would a joint transform
have to be to reproduce one shape on its own? Write the shape's per-joint motion
as $\mathbf{I} + \mathbf{N}_{k,j}$, where $\mathbf{I}$ is the $3 \times 4$
identity (a $3 \times 3$ identity with a zero column). Plugging that into
skinning:

$$
\sum_{j} w_{i,j}\,(\mathbf{I} + \mathbf{N}_{k,j})\,\mathbf{v}_{0,i} = \hat{\mathbf{v}}_{k,i}
$$

Because the weights sum to one, the identity part just returns the neutral
vertex, and the equation simplifies to

$$
\sum_{j} w_{i,j}\, \mathbf{N}_{k,j}\, \mathbf{v}_{0,i} = \hat{\mathbf{v}}_{k,i} - \hat{\mathbf{v}}_{0,i}
$$

This is the key statement. The right-hand side is the delta of shape $k$. The
left-hand side says: **the deltas $\mathbf{N}_{k,1} \dots \mathbf{N}_{k,P}$ are
the joint-space version of blendshape $k$**. Skinning "adds identity" to get
back to the neutral pose in the same way that a delta blendshape "adds zero".
Subtracting identity from a skinning transform is the same act as subtracting
the neutral mesh from a target.

Stacking this for all vertices and shapes gives the matrix form used
throughout the code:

$$
\mathbf{A} \approx \mathbf{B}\,\mathbf{C},
\qquad
\mathbf{A} \in \mathbb{R}^{3S \times N},\;
\mathbf{B} \in \mathbb{R}^{3S \times 4P},\;
\mathbf{C} \in \mathbb{R}^{4P \times N}
$$

- $\mathbf{A}$ holds all the deltas, three rows per shape, one column per vertex.
- $\mathbf{B}$ holds all the $\mathbf{N}_{k,j}$ blocks, shape $k$ in rows
  $3k \dots 3k+2$, joint $j$ in columns $4j \dots 4j+3$. This is the
  `shapeXform` array in the output file.
- $\mathbf{C}$ holds each neutral vertex, in homogeneous form, scaled by its
  weight for each joint. It is built from `weights` and `rest`.

Finding $\mathbf{B}$ and $\mathbf{C}$ from $\mathbf{A}$ is the solver's job.

## The runtime formula (Equations 6 and 7)

Now bring several shapes back. Substitute the joint-space deltas into the
blendshape formula and regroup the sums:

$$
\hat{\mathbf{v}}_{0,i} + \sum_{k} c_k \sum_{j} w_{i,j}\,\mathbf{N}_{k,j}\,\mathbf{v}_{0,i}
\;=\;
\sum_{j} w_{i,j} \Big( \mathbf{I} + \sum_{k} c_k\,\mathbf{N}_{k,j} \Big) \mathbf{v}_{0,i}
$$

The bracket is a per-joint transform that depends only on the current shape
values. Define

$$
\boxed{\;\mathbf{M}_j = \mathbf{I} + \sum_{k=1}^{S} c_k\,\mathbf{N}_{k,j}\;}
$$

and the skinning formula gives **exactly** the blendshape result (exactly, that
is, when Equation 5 is satisfied exactly; in practice to within the solver's
error). This is Equation 7, the one line every runtime has to implement:

1. Get the shape values $c_k$ from the rig, as before.
2. For each joint, start from identity and add each active shape's delta block
   scaled by its value.
3. Hand the $P$ resulting matrices to the skinning shader.

Step 2 is a weighted sum of matrices. It is the same arithmetic as blending
vertex offsets, moved from $N$ vertices to $P$ joints. Because most
$\mathbf{N}_{k,j}$ are zero, the sum only touches a few blocks per active shape.
The reference implementation is
`AnimationFrameGenerator._generate_skinning_transforms`, which does it as one
sparse-friendly matrix product.

## Why the transforms are linearised rotations (Equation 8)

Equation 7 adds transforms. Rotation matrices are not closed under addition:
half of one rotation plus half of another is not a rotation. If the deltas were
constrained to rotations, the blended $\mathbf{M}_j$ would still not be, so the
constraint buys nothing. Instead each delta is restricted to the form

$$
\mathbf{N}_{k,j} =
\begin{bmatrix}
0 & -r_3 & r_2 & t_1 \\
r_3 & 0 & -r_1 & t_2 \\
-r_2 & r_1 & 0 & t_3
\end{bmatrix}
$$

Six numbers: a **linearised rotation** $(r_1, r_2, r_3)$, which is the
first-order approximation of a rotation about the axis $r$ by angle $|r|$, and a
translation $(t_1, t_2, t_3)$. Sums and scalar multiples of matrices of this
form stay in this form, so blending is a plain linear combination with no
normalisation, no quaternion shortest-path logic, no special cases.

The price is that $\mathbf{I} + \mathbf{N}_{k,j}$ is not orthonormal: a
linearised rotation applied to a vertex stretches it slightly, in proportion to
the square of the angle. The solver knows this and compensates in the
translation and weights, so the *result* is accurate; but the transforms
themselves must be used as they are. Orthonormalising them, or converting them
to rotation plus translation, throws away part of the fit.

In the code the six coefficients per (shape, joint) live in `B_rt`, shape
`(6, S, P, 1, 1)`, and the six fixed basis matrices in `TR`, built by
`SkinCompressor.buildTR`. Their product, summed over the six, is $\mathbf{B}$.

## Composition with the head

Facial skinning rarely lives alone. The head moves with the body, and the face
must follow. Because $\mathbf{M}_j$ is an ordinary affine transform, this is
the standard hierarchical case: multiply each $\mathbf{M}_j$ by the head's
world transform on the CPU and skin as usual. The deltas were solved in the
rest pose's own frame, so what matters is that the composition happens
*outside* the sum in Equation 7, not inside it. The practical version of this
for a Maya rig is in [Maya rig workflow](../user_guide/maya_rig_workflow.md),
section "Rest pose and coordinate space".

## Summary

- A blendshape is a set of per-vertex offsets. Its skinned equivalent is a set
  of per-joint delta transforms $\mathbf{N}_{k,j}$ plus shared skin weights.
- Runtime: $\mathbf{M}_j = \mathbf{I} + \sum_k c_k \mathbf{N}_{k,j}$, then
  standard linear blend skinning.
- Deltas are 6-parameter linearised rigid motions so that they add correctly;
  the blended transforms are affine, not rigid.
- Most deltas are zero. That is the compression.

Next: [How the solver works](how_the_solver_works.md).
